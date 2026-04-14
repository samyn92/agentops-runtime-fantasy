/*
Agent Runtime — Fantasy (Go)

Main entrypoint. Two subcommands:
  - daemon: HTTP server on :4096 (Deployment mode) — full FEP SSE protocol
  - task: Read AGENT_PROMPT, run once, JSON to stdout (Job mode)
*/
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"sync"
	"syscall"
	"time"

	"charm.land/fantasy"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

// version is set at build time via -ldflags="-X main.version=..."
// Falls back to "dev" for local builds without ldflags.
var version = "dev"

const (
	configPath = "/etc/operator/config.json"
	port       = 4096
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "Usage: agent-runtime <daemon|task>")
		os.Exit(1)
	}

	slog.SetDefault(slog.New(slog.NewJSONHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelInfo})))

	switch os.Args[1] {
	case "daemon":
		if err := runDaemon(); err != nil {
			slog.Error("daemon failed", "error", err)
			os.Exit(1)
		}
	case "task":
		if err := runTask(); err != nil {
			slog.Error("task failed", "error", err)
			os.Exit(1)
		}
	default:
		fmt.Fprintf(os.Stderr, "Unknown command: %s\n", os.Args[1])
		os.Exit(1)
	}
}

// ====================================================================
// Shared setup
// ====================================================================

func loadConfig() (*Config, error) {
	data, err := os.ReadFile(configPath)
	if err != nil {
		return nil, fmt.Errorf("read config: %w", err)
	}
	var cfg Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}
	return &cfg, nil
}

// agentBundle holds the built agent and resources that need cleanup.
type agentBundle struct {
	agent     fantasy.Agent
	providers map[string]fantasy.Provider
	mcpConns  []mcpConnection
	toolCount int // number of tools registered (for budget estimation)
	// baseTools and baseOpts are preserved for rebuilding with permission gates / question tool.
	// This ensures the rebuild never diverges from the primary path.
	baseTools []fantasy.AgentTool
	baseOpts  []fantasy.AgentOption
	// delegationWatcher tracks parallel fan-out delegation groups (nil if K8s unavailable)
	delegationWatcher *DelegationWatcher
}

func buildAgentBundle(ctx context.Context, cfg *Config, engram *EngramClient, injector *contextInjector, extraTools ...fantasy.AgentTool) (*agentBundle, error) {
	// Resolve providers
	providers := make(map[string]fantasy.Provider)
	for _, p := range cfg.Providers {
		provider, err := resolveProvider(p)
		if err != nil {
			slog.Warn("failed to resolve provider", "name", p.Name, "type", p.Type, "error", err)
			continue
		}
		providers[p.Name] = provider
		slog.Info("registered provider", "name", p.Name, "type", p.Type)
	}

	// Resolve primary model
	model, err := resolveModel(ctx, cfg.PrimaryModel, providers, cfg.PrimaryProvider)
	if err != nil {
		return nil, fmt.Errorf("resolve model %s: %w", cfg.PrimaryModel, err)
	}

	// Build tools
	tools := buildBuiltinTools(cfg.BuiltinTools)

	// Load MCP tools from OCI-packaged servers
	var allMCPConns []mcpConnection
	if len(cfg.Tools) > 0 {
		ociTools, conns, err := loadOCITools(ctx, cfg.Tools)
		if err != nil {
			slog.Warn("failed to load some OCI tools", "error", err)
		}
		tools = append(tools, ociTools...)
		allMCPConns = append(allMCPConns, conns...)
	}

	// Load MCP tools from gateway sidecars
	if len(cfg.MCPServers) > 0 {
		gwTools, conns, err := loadGatewayMCPTools(ctx, cfg.MCPServers)
		if err != nil {
			slog.Warn("failed to load some gateway MCP tools", "error", err)
		}
		tools = append(tools, gwTools...)
		allMCPConns = append(allMCPConns, conns...)
	}

	// Add built-in git tools when a git workspace is configured.
	// These replace the mcp-git sidecar — pure go-git, no CLI or gateway needed.
	if os.Getenv("GIT_REPO_URL") != "" {
		tools = append(tools, gitTools()...)
		slog.Info("built-in git tools enabled", "count", len(gitTools()))
	}

	// Add orchestration tools (run_agent, run_agents, get_agent_run, list_task_agents)
	k8sClient, err := NewK8sClient()
	var delegationWatcher *DelegationWatcher
	if err != nil {
		slog.Warn("K8s client unavailable, orchestration tools disabled", "error", err)
		tools = append(tools, newRunAgentToolStub(), newRunAgentsToolStub(), newGetAgentRunToolStub(), newListTaskAgentsToolStub())
	} else {
		delegationWatcher = NewDelegationWatcher(k8sClient)
		tools = append(tools, newRunAgentTool(k8sClient, cfg.Resources), newRunAgentsTool(k8sClient, cfg.Resources, delegationWatcher, cfg.Delegation), newGetAgentRunTool(k8sClient), newListTaskAgentsTool(k8sClient))
	}

	// Add any extra tools (e.g. memory tools from Engram)
	tools = append(tools, extraTools...)

	// Wrap ALL tools with security hooks + output truncation + tracing spans.
	// This must be done AFTER all tools are added so every tool gets traced.
	tools = wrapToolsWithHooks(tools, cfg.ToolHooks, cfg.MaxToolResultChars, engram)

	// Apply Anthropic prompt caching to tool definitions (if using Anthropic)
	if isAnthropicProvider(cfg) {
		applyToolCaching(tools)
		slog.Info("anthropic prompt caching enabled for tools", "tool_count", len(tools))
	}

	// Build agent options
	opts := []fantasy.AgentOption{
		fantasy.WithTools(tools...),
	}

	// Add context injection + Anthropic caching via PrepareStep.
	// This moves memory context and resource context from the user message into
	// the system message as separate TextParts — correct semantics and cacheable.
	opts = append(opts, fantasy.WithPrepareStep(
		prepareStepWithContextInjection(injector, isAnthropicProvider(cfg)),
	))

	if cfg.SystemPrompt != "" {
		opts = append(opts, fantasy.WithSystemPrompt(cfg.SystemPrompt))
	}
	if cfg.Temperature != nil {
		opts = append(opts, fantasy.WithTemperature(*cfg.Temperature))
	}
	if cfg.MaxOutputTokens != nil {
		opts = append(opts, fantasy.WithMaxOutputTokens(*cfg.MaxOutputTokens))
	}
	if cfg.MaxSteps != nil {
		opts = append(opts, fantasy.WithStopConditions(fantasy.StepCountIs(*cfg.MaxSteps)))
	}

	// Input token budget guard: stop the agent loop before context window overflow.
	// Uses 75% of the model's context window as the budget by default.
	budgetFraction := DefaultBudgetFraction
	if cfg.BudgetFraction != nil && *cfg.BudgetFraction > 0 && *cfg.BudgetFraction < 1 {
		budgetFraction = *cfg.BudgetFraction
	}
	contextWindow := contextWindowForModel(cfg.PrimaryModel)
	budgetTokens := int64(float64(contextWindow) * budgetFraction)
	opts = append(opts, fantasy.WithStopConditions(InputTokenBudget(budgetTokens)))
	slog.Info("input token budget configured",
		"model", cfg.PrimaryModel,
		"context_window", contextWindow,
		"budget_fraction", budgetFraction,
		"budget_tokens", budgetTokens,
	)

	return &agentBundle{
		agent:             fantasy.NewAgent(model, opts...),
		providers:         providers,
		mcpConns:          allMCPConns,
		toolCount:         len(tools),
		baseTools:         tools,
		baseOpts:          opts,
		delegationWatcher: delegationWatcher,
	}, nil
}

// buildFallbackAgent creates a new agent with a fallback model (reusing options from config).
func buildFallbackAgent(ctx context.Context, cfg *Config, providers map[string]fantasy.Provider, modelStr string) (fantasy.Agent, error) {
	model, err := resolveModel(ctx, modelStr, providers, cfg.PrimaryProvider)
	if err != nil {
		return nil, err
	}

	opts := []fantasy.AgentOption{}
	if cfg.SystemPrompt != "" {
		opts = append(opts, fantasy.WithSystemPrompt(cfg.SystemPrompt))
	}
	if cfg.Temperature != nil {
		opts = append(opts, fantasy.WithTemperature(*cfg.Temperature))
	}
	if cfg.MaxOutputTokens != nil {
		opts = append(opts, fantasy.WithMaxOutputTokens(*cfg.MaxOutputTokens))
	}
	if cfg.MaxSteps != nil {
		opts = append(opts, fantasy.WithStopConditions(fantasy.StepCountIs(*cfg.MaxSteps)))
	}

	return fantasy.NewAgent(model, opts...), nil
}

// streamWithFallback tries the primary model, then fallbacks on retryable errors.
// Creates a gen_ai.stream span covering the LLM call (including fallback attempts).
func streamWithFallback(ctx context.Context, cfg *Config, bundle *agentBundle, call fantasy.AgentStreamCall) (*fantasy.AgentResult, string, error) {
	ctx, span := tracer.Start(ctx, "gen_ai.stream", trace.WithAttributes(
		attrGenAIOperationName.String("chat"),
		attrGenAIProviderName.String(detectGenAIProvider(cfg.PrimaryModel, cfg.PrimaryProvider)),
		attrGenAIRequestModel.String(cfg.PrimaryModel),
	))
	defer span.End()

	// Add optional request parameters
	if cfg.Temperature != nil {
		span.SetAttributes(attrGenAITemperature.Float64(*cfg.Temperature))
	}
	if cfg.MaxOutputTokens != nil {
		span.SetAttributes(attrGenAIMaxTokens.Int64(*cfg.MaxOutputTokens))
	}

	result, err := bundle.agent.Stream(ctx, call)
	if err == nil {
		setLLMResultAttributes(span, result, cfg.PrimaryModel)
		return result, cfg.PrimaryModel, nil
	}

	if !isRetryableError(err) || len(cfg.FallbackModels) == 0 {
		recordError(span, err)
		return nil, cfg.PrimaryModel, err
	}

	slog.Warn("primary model failed on stream, trying fallbacks",
		"model", cfg.PrimaryModel, "error", err)

	for _, fbModel := range cfg.FallbackModels {
		// Record fallback event on the span
		span.AddEvent("model.fallback", trace.WithAttributes(
			attribute.String("fallback.from", cfg.PrimaryModel),
			attribute.String("fallback.to", fbModel),
			attribute.String("fallback.error", err.Error()),
		))

		fbAgent, fbErr := buildFallbackAgent(ctx, cfg, bundle.providers, fbModel)
		if fbErr != nil {
			continue
		}

		result, err = fbAgent.Stream(ctx, call)
		if err == nil {
			span.SetAttributes(attrGenAIResponseModel.String(fbModel))
			setLLMResultAttributes(span, result, fbModel)
			return result, fbModel, nil
		}

		if !isRetryableError(err) {
			recordError(span, err)
			return nil, fbModel, err
		}
	}

	recordError(span, err)
	return nil, cfg.PrimaryModel, fmt.Errorf("all models failed, last error: %w", err)
}

// generateWithFallback tries the primary model, then fallbacks on retryable errors.
// Creates a gen_ai.generate span covering the LLM call (including fallback attempts).
func generateWithFallback(ctx context.Context, cfg *Config, bundle *agentBundle, call fantasy.AgentCall) (*fantasy.AgentResult, string, error) {
	ctx, span := tracer.Start(ctx, "gen_ai.generate", trace.WithAttributes(
		attrGenAIOperationName.String("chat"),
		attrGenAIProviderName.String(detectGenAIProvider(cfg.PrimaryModel, cfg.PrimaryProvider)),
		attrGenAIRequestModel.String(cfg.PrimaryModel),
	))
	defer span.End()

	// Add optional request parameters
	if cfg.Temperature != nil {
		span.SetAttributes(attrGenAITemperature.Float64(*cfg.Temperature))
	}
	if cfg.MaxOutputTokens != nil {
		span.SetAttributes(attrGenAIMaxTokens.Int64(*cfg.MaxOutputTokens))
	}

	result, err := bundle.agent.Generate(ctx, call)
	if err == nil {
		setLLMResultAttributes(span, result, cfg.PrimaryModel)
		return result, cfg.PrimaryModel, nil
	}

	if !isRetryableError(err) || len(cfg.FallbackModels) == 0 {
		recordError(span, err)
		return nil, cfg.PrimaryModel, err
	}

	slog.Warn("primary model failed, trying fallbacks",
		"model", cfg.PrimaryModel, "error", err)

	for _, fbModel := range cfg.FallbackModels {
		// Record fallback event on the span
		span.AddEvent("model.fallback", trace.WithAttributes(
			attribute.String("fallback.from", cfg.PrimaryModel),
			attribute.String("fallback.to", fbModel),
			attribute.String("fallback.error", err.Error()),
		))

		fbAgent, fbErr := buildFallbackAgent(ctx, cfg, bundle.providers, fbModel)
		if fbErr != nil {
			slog.Warn("failed to build fallback agent", "model", fbModel, "error", fbErr)
			continue
		}

		result, err = fbAgent.Generate(ctx, call)
		if err == nil {
			slog.Info("fallback model succeeded", "model", fbModel)
			span.SetAttributes(attrGenAIResponseModel.String(fbModel))
			setLLMResultAttributes(span, result, fbModel)
			return result, fbModel, nil
		}

		if !isRetryableError(err) {
			recordError(span, err)
			return nil, fbModel, err
		}
		slog.Warn("fallback model failed", "model", fbModel, "error", err)
	}

	recordError(span, err)
	return nil, cfg.PrimaryModel, fmt.Errorf("all models failed, last error: %w", err)
}

// isRetryableError checks if an error should trigger fallback.
func isRetryableError(err error) bool {
	if err == nil {
		return false
	}
	s := err.Error()
	return strings.Contains(s, "429") ||
		strings.Contains(s, "500") ||
		strings.Contains(s, "502") ||
		strings.Contains(s, "503") ||
		strings.Contains(s, "rate limit") ||
		strings.Contains(s, "overloaded")
}

// ====================================================================
// Daemon mode: HTTP server with full FEP SSE protocol
// ====================================================================

func runDaemon() error {
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	cfg, err := loadConfig()
	if err != nil {
		return err
	}

	slog.Info("starting Fantasy daemon agent",
		"model", cfg.PrimaryModel,
		"providers", len(cfg.Providers),
		"builtinTools", builtinToolCount(cfg.BuiltinTools),
		"ociTools", len(cfg.Tools),
		"mcpServers", len(cfg.MCPServers),
		"fallbackModels", len(cfg.FallbackModels),
	)

	// Determine agent name (used as Engram project scope)
	agentName := os.Getenv("AGENT_NAME")
	if agentName == "" {
		agentName = "default"
	}

	// Initialize OpenTelemetry tracing
	agentNamespace := os.Getenv("AGENT_NAMESPACE")
	tracingFns, err := initTracing(ctx, agentName, agentNamespace, "daemon")
	if err != nil {
		slog.Warn("tracing init failed, continuing without tracing", "error", err)
	}

	// Initialize memory system
	contextLimit := 5
	var engram *EngramClient
	var preBuiltMemory *WorkingMemory

	if cfg.Memory != nil {
		if cfg.Memory.ContextLimit > 0 {
			contextLimit = cfg.Memory.ContextLimit
		}

		project := cfg.Memory.Project
		if project == "" {
			project = agentName
		}

		engram = NewEngramClient(cfg.Memory.ServerURL, project)

		// Pre-flight: check for a working memory checkpoint with a session ID.
		// If found, restore the session ID so the memory service reuses the
		// pre-crash session instead of creating a new one.
		wm := NewWorkingMemory()
		if restored := wm.RestoreCheckpoint(); restored > 0 {
			slog.Info("recovered from checkpoint", "messages", restored)
			if sid := wm.SessionID(); sid != "" {
				engram.SetSessionID(sid)
				slog.Info("restored memory session from checkpoint", "sessionID", sid)
			}
		}

		if err := engram.Init(); err != nil {
			slog.Warn("memory init failed, running without persistent memory", "error", err)
			engram = nil
		}

		// Store the active session ID on working memory for future checkpoints
		if engram != nil {
			wm.SetSessionID(engram.SessionID())
		}

		// Stash wm for daemonServer construction below
		preBuiltMemory = wm
	}

	_ = contextLimit // used in prompt handlers via cfg

	// Create context injector — holds per-turn memory/resource context for PrepareStep
	ctxInjector := &contextInjector{}

	// Set platform protocol — stable across turns, injected as a separate system message part
	if cfg.PlatformProtocol != "" {
		ctxInjector.SetPlatformProtocol(cfg.PlatformProtocol)
	}

	// Build agent bundle (includes memory tools if engram is available)
	bundle, err := buildAgentBundle(ctx, cfg, engram, ctxInjector, buildMemoryTools(engram)...)
	if err != nil {
		return err
	}
	defer shutdownMCPConnections(bundle.mcpConns)

	// Use pre-built working memory if checkpoint restore happened, otherwise create fresh
	memory := preBuiltMemory
	if memory == nil {
		memory = NewWorkingMemory()
	}

	// Create pre-flight context budget tracker
	budgetFrac := DefaultBudgetFraction
	if cfg.BudgetFraction != nil && *cfg.BudgetFraction > 0 && *cfg.BudgetFraction < 1 {
		budgetFrac = *cfg.BudgetFraction
	}
	ctxBudget := NewContextBudget(cfg.PrimaryModel, budgetFrac)
	// Include both platform protocol and user prompt in fixed budget calculation
	fixedPrompt := cfg.PlatformProtocol + "\n" + cfg.SystemPrompt
	ctxBudget.UpdateFixed(fixedPrompt, bundle.toolCount)

	srv := &daemonServer{
		bundle:      bundle,
		cfg:         cfg,
		memory:      memory,
		engram:      engram,
		injector:    ctxInjector,
		budget:      ctxBudget,
		activeModel: cfg.PrimaryModel,
		agentName:   agentName,
		delegation:  bundle.delegationWatcher,
		convCtx:     &sessionContext{},
	}

	// Wire up the DelegationWatcher trigger and emitter functions.
	// The trigger queues delegation results as a steer message when the agent
	// is busy, or starts a new internal prompt when idle.
	if srv.delegation != nil {
		srv.delegation.SetTrigger(func(prompt string) {
			sc := srv.convCtx
			sc.mu.Lock()
			if sc.busy {
				// Agent is busy — queue as steer message for injection at next step boundary
				sc.steerMsg = prompt
				sc.mu.Unlock()
				slog.Info("delegation callback queued as steer (agent busy)")
				return
			}
			sc.mu.Unlock()

			// Agent is idle — trigger a new prompt via the streaming endpoint internally
			go srv.handleInternalPrompt(prompt)
		})

		srv.delegation.SetEmitterFn(func() *fepEmitter {
			srv.convCtx.mu.Lock()
			emit := srv.convCtx.emitter
			srv.convCtx.mu.Unlock()
			return emit
		})
	}

	// Initialize permission gate (emits permission_asked via the active SSE emitter)
	srv.permGate = newPermissionGate(
		func(id, sessionId, toolName, input, description string) {
			srv.convCtx.mu.Lock()
			emit := srv.convCtx.emitter
			srv.convCtx.mu.Unlock()
			if emit != nil {
				emit.emitPermissionAsked(id, srv.agentName, toolName, input, description)
			}
		},
	)

	// Initialize question gate (emits question_asked via the active SSE emitter)
	srv.questionGate = newQuestionGate(
		func(id, sessionId string, questions json.RawMessage) {
			srv.convCtx.mu.Lock()
			emit := srv.convCtx.emitter
			srv.convCtx.mu.Unlock()
			if emit != nil {
				emit.emitQuestionAsked(id, srv.agentName, questions)
			}
		},
	)

	// Apply permission wrapping and/or question tool if configured.
	// Uses baseTools and baseOpts from the primary build path to ensure
	// no tools or options are silently lost (git tools, listTaskAgents,
	// InputTokenBudget, PrepareStep, etc.).
	if len(cfg.PermissionTools) > 0 || cfg.EnableQuestionTool {
		tools := make([]fantasy.AgentTool, len(bundle.baseTools))
		copy(tools, bundle.baseTools)

		// Add question tool if enabled
		if cfg.EnableQuestionTool {
			tools = append(tools, newQuestionTool(srv.questionGate))
		}

		// Wrap with permission gates if configured
		if len(cfg.PermissionTools) > 0 {
			tools = srv.permGate.wrapTools(tools, cfg.PermissionTools)
		}

		// Rebuild agent with the augmented tools but all original options preserved.
		// Filter out the old WithTools option and replace with the new tool set.
		opts := []fantasy.AgentOption{fantasy.WithTools(tools...)}
		for _, opt := range bundle.baseOpts {
			opts = append(opts, opt)
		}

		model, _ := resolveModel(ctx, cfg.PrimaryModel, bundle.providers, cfg.PrimaryProvider)
		if model != nil {
			bundle.agent = fantasy.NewAgent(model, opts...)
			bundle.toolCount = len(tools)
		}

		if len(cfg.PermissionTools) > 0 {
			slog.Info("permission gates applied", "tools", cfg.PermissionTools)
		}
		if cfg.EnableQuestionTool {
			slog.Info("question tool enabled")
		}
	}

	// Restore delegation groups from checkpoint (crash recovery).
	// Must happen after watcher trigger/emitter are wired up.
	if srv.delegation != nil {
		if groups := RestoreDelegationCheckpoint(); len(groups) > 0 {
			srv.delegation.RestoreGroups(groups)
		}
	}

	mux := http.NewServeMux()

	// Prompt endpoints (single conversation per agent — no session ID needed)
	mux.HandleFunc("POST /prompt", srv.handlePrompt)
	mux.HandleFunc("POST /prompt/stream", srv.handlePromptStream)

	// Conversation control
	mux.HandleFunc("POST /steer", srv.handleSteer)
	mux.HandleFunc("DELETE /abort", srv.handleAbort)

	// Permission and question reply endpoints
	mux.HandleFunc("POST /permission/{pid}/reply", srv.handlePermissionReply)
	mux.HandleFunc("POST /question/{qid}/reply", srv.handleQuestionReply)

	// Health and status
	mux.HandleFunc("GET /healthz", srv.handleHealthz)
	mux.HandleFunc("GET /status", srv.handleStatus)

	// Live configuration
	mux.HandleFunc("GET /working-memory", srv.handleGetWorkingMemory)

	// AI-assisted memory extraction
	mux.HandleFunc("POST /memory/extract", srv.handleMemoryExtract)

	httpSrv := &http.Server{Addr: fmt.Sprintf(":%d", port), Handler: mux}

	go func() {
		<-ctx.Done()
		slog.Info("shutting down...")

		// Stop delegation watcher and cancel active watches
		if srv.delegation != nil {
			SaveDelegationCheckpoint(srv.delegation.CheckpointGroups())
			srv.delegation.Stop()
		}

		// Save working memory checkpoint to PVC for crash recovery
		srv.memory.SaveCheckpoint()

		// End memory session with raw messages for summarization
		if srv.engram != nil {
			srv.engram.EndSession(fantasyToEngramMessages(srv.memory.Messages()))
		}

		// Flush pending traces before exit
		if tracingFns != nil {
			if err := tracingFns.Shutdown(context.Background()); err != nil {
				slog.Warn("tracing shutdown error", "error", err)
			}
		}

		httpSrv.Close()
	}()

	slog.Info("listening", "port", port)
	return httpSrv.ListenAndServe()
}

// sessionContext tracks per-conversation runtime state (cancel func, busy flag, steer msgs).
// With the memory rewrite there is one conversation per daemon agent, so only
// one sessionContext exists, but we keep the type for steer/abort/permission/question.
type sessionContext struct {
	mu       sync.Mutex
	busy     bool
	cancel   context.CancelFunc
	steerMsg string      // pending steer message, consumed on next step boundary
	emitter  *fepEmitter // current SSE emitter (set during streaming)
}

func (sc *sessionContext) popSteerMessage() string {
	sc.mu.Lock()
	defer sc.mu.Unlock()
	msg := sc.steerMsg
	sc.steerMsg = ""
	return msg
}

type daemonServer struct {
	bundle       *agentBundle
	cfg          *Config
	memory       *WorkingMemory
	engram       *EngramClient
	injector     *contextInjector // per-turn memory/resource context for PrepareStep
	budget       *ContextBudget   // pre-flight context window budget tracker
	activeModel  string
	totalSteps   int
	agentName    string // from AGENT_NAME env var
	lastTraceID  string // trace ID of the most recent prompt execution
	permGate     *permissionGate
	questionGate *questionGate
	delegation   *DelegationWatcher // tracks parallel fan-out delegation groups

	mu      sync.Mutex
	convCtx *sessionContext // single conversation context
}

// ── Request/Response types ──

type promptRequest struct {
	Prompt  string            `json:"prompt"`
	Context []ResourceContext `json:"context,omitempty"` // per-turn resource context from console
}

type promptResponse struct {
	Output string `json:"output"`
	Model  string `json:"model"`
}

type steerRequest struct {
	Message string `json:"message"`
}

// ── Prompt handler (non-streaming) ──

func (s *daemonServer) handlePrompt(w http.ResponseWriter, r *http.Request) {
	var req promptRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.Prompt == "" {
		http.Error(w, `{"error":"prompt is required"}`, http.StatusBadRequest)
		return
	}

	sc := s.convCtx
	sc.mu.Lock()
	if sc.busy {
		sc.mu.Unlock()
		http.Error(w, `{"error":"agent is busy"}`, http.StatusTooManyRequests)
		return
	}
	sc.busy = true
	sc.mu.Unlock()

	defer func() {
		sc.mu.Lock()
		sc.busy = false
		sc.cancel = nil
		sc.mu.Unlock()
	}()

	ctx, cancel := context.WithCancel(r.Context())
	sc.mu.Lock()
	sc.cancel = cancel
	sc.mu.Unlock()
	defer cancel()

	// Start root tracing span
	// If this prompt was triggered by another agent (operator sends traceparent header),
	// create a span link back to the parent's orchestration span.
	spanOpts := []trace.SpanStartOption{
		trace.WithAttributes(
			attrAgentName.String(s.agentName),
			attrAgentMode.String("daemon"),
			attrGenAIOperationName.String("invoke_agent"),
			attrGenAIProviderName.String(detectGenAIProvider(s.cfg.PrimaryModel, s.cfg.PrimaryProvider)),
			attrGenAIRequestModel.String(s.cfg.PrimaryModel),
		),
	}
	if tp := r.Header.Get("Traceparent"); tp != "" {
		parentAgent := r.Header.Get("X-AgentOps-Parent-Agent")
		runName := r.Header.Get("X-AgentOps-Run-Name")
		spanOpts = append(spanOpts, delegationSpanOptions(tp, parentAgent, runName)...)
		slog.Info("delegation trace link created (prompt)", "parentAgent", parentAgent, "runName", runName)
	}
	ctx, promptSpan := tracer.Start(ctx, "agent.prompt", spanOpts...)
	defer promptSpan.End()

	// Record the user prompt as a content event for trace visibility
	recordPromptEvent(promptSpan, req.Prompt)

	// Get messages from working memory
	messages := s.memory.Messages()

	// Set per-turn context on the injector — PrepareStep will inject it
	// into the system message as separate TextParts.
	var memoryCtx, resourceCtx string
	if s.engram != nil {
		contextLimit := 5
		if s.cfg.Memory != nil && s.cfg.Memory.ContextLimit > 0 {
			contextLimit = s.cfg.Memory.ContextLimit
		}
		memoryCtx = s.engram.FetchContext(ctx, contextLimit, req.Prompt)
	}
	if len(req.Context) > 0 {
		resourceCtx = formatResourceContext(req.Context)
		slog.Info("resource context injected", "items", len(req.Context))
	}
	s.injector.Set(memoryCtx, resourceCtx)
	defer s.injector.Clear()

	// Pre-flight: update budget estimates and trim working memory to fit
	s.budget.UpdatePerTurn(memoryCtx, resourceCtx, req.Prompt, messages)
	convBudget := s.budget.ConversationBudget()
	if trimmed, _ := s.memory.TrimToTokenBudget(convBudget); trimmed > 0 {
		// Re-fetch messages after trim
		messages = s.memory.Messages()
		s.budget.UpdatePerTurn(memoryCtx, resourceCtx, req.Prompt, messages)
	}

	result, usedModel, err := generateWithFallback(ctx, s.cfg, s.bundle, fantasy.AgentCall{
		Prompt:   req.Prompt,
		Messages: messages,
	})
	if err != nil {
		recordError(promptSpan, err)
		http.Error(w, fmt.Sprintf(`{"error":"%s"}`, err.Error()), http.StatusInternalServerError)
		return
	}

	output := result.Response.Content.Text()

	// Record the assistant response as a content event
	recordCompletionEvent(promptSpan, output)

	// Append to working memory
	s.memory.Append(fantasy.NewUserMessage(req.Prompt))
	for _, step := range result.Steps {
		s.memory.Append(step.Messages...)
	}
	s.memory.CompleteTurn()

	traceID := traceIDFromContext(ctx)

	s.mu.Lock()
	s.activeModel = usedModel
	s.totalSteps += len(result.Steps)
	s.lastTraceID = traceID
	s.mu.Unlock()

	// Set final attributes on root span
	promptSpan.SetAttributes(
		attrGenAIResponseModel.String(usedModel),
		attrGenAIInputTokens.Int64(result.TotalUsage.InputTokens),
		attrGenAIOutputTokens.Int64(result.TotalUsage.OutputTokens),
		attribute.Int("agent.steps", len(result.Steps)),
	)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(promptResponse{Output: output, Model: usedModel})
}

// ── Streaming prompt with full FEP ──

func (s *daemonServer) handlePromptStream(w http.ResponseWriter, r *http.Request) {
	var req promptRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.Prompt == "" {
		http.Error(w, `{"error":"prompt is required"}`, http.StatusBadRequest)
		return
	}

	sc := s.convCtx
	sc.mu.Lock()
	if sc.busy {
		sc.mu.Unlock()
		http.Error(w, `{"error":"agent is busy"}`, http.StatusTooManyRequests)
		return
	}
	sc.busy = true
	sc.mu.Unlock()

	defer func() {
		sc.mu.Lock()
		sc.busy = false
		sc.cancel = nil
		sc.mu.Unlock()
	}()

	// Set SSE headers
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")

	ctx, cancel := context.WithCancel(r.Context())
	sc.mu.Lock()
	sc.cancel = cancel
	sc.mu.Unlock()
	defer cancel()

	// Start root tracing span for this prompt execution
	// If this prompt was triggered by another agent (operator sends traceparent header),
	// create a span link back to the parent's orchestration span.
	streamSpanOpts := []trace.SpanStartOption{
		trace.WithAttributes(
			attrAgentName.String(s.agentName),
			attrAgentMode.String("daemon"),
			attrGenAIOperationName.String("invoke_agent"),
			attrGenAIProviderName.String(detectGenAIProvider(s.cfg.PrimaryModel, s.cfg.PrimaryProvider)),
			attrGenAIRequestModel.String(s.cfg.PrimaryModel),
		),
	}
	if tp := r.Header.Get("Traceparent"); tp != "" {
		parentAgent := r.Header.Get("X-AgentOps-Parent-Agent")
		runName := r.Header.Get("X-AgentOps-Run-Name")
		streamSpanOpts = append(streamSpanOpts, delegationSpanOptions(tp, parentAgent, runName)...)
		slog.Info("delegation trace link created (stream)", "parentAgent", parentAgent, "runName", runName)
	}
	ctx, promptSpan := tracer.Start(ctx, "agent.prompt", streamSpanOpts...)
	defer promptSpan.End()

	// Stash root span in context so tool wrappers can record tool.call events
	// that survive even when individual tool.execute spans are lost.
	ctx = withRootSpan(ctx, promptSpan)

	// Record the user prompt as a content event for trace visibility
	recordPromptEvent(promptSpan, req.Prompt)

	emit := newFEPEmitter(w)

	// Store the emitter on the conversation context so permission/question
	// gates can emit FEP events through the active SSE stream.
	sc.mu.Lock()
	sc.emitter = emit
	sc.mu.Unlock()
	defer func() {
		sc.mu.Lock()
		sc.emitter = nil
		sc.mu.Unlock()
	}()

	// Inject agent name into Go context so tools (permission gate, question tool) can read it.
	ctx = context.WithValue(ctx, agentContextKey{}, s.agentName)

	// Get messages from working memory
	messages := s.memory.Messages()

	// Set per-turn context on the injector — PrepareStep will inject it
	// into the system message as separate TextParts.
	var memoryCtx, resourceCtx string
	if s.engram != nil {
		contextLimit := 5
		if s.cfg.Memory != nil && s.cfg.Memory.ContextLimit > 0 {
			contextLimit = s.cfg.Memory.ContextLimit
		}
		memoryCtx = s.engram.FetchContext(ctx, contextLimit, req.Prompt)
	}
	if len(req.Context) > 0 {
		resourceCtx = formatResourceContext(req.Context)
		slog.Info("resource context injected (stream)", "items", len(req.Context))
	}
	s.injector.Set(memoryCtx, resourceCtx)
	defer s.injector.Clear()

	// Pre-flight: update budget estimates and trim working memory to fit
	s.budget.UpdatePerTurn(memoryCtx, resourceCtx, req.Prompt, messages)
	convBudget := s.budget.ConversationBudget()
	if trimmed, _ := s.memory.TrimToTokenBudget(convBudget); trimmed > 0 {
		// Re-fetch messages after trim
		messages = s.memory.Messages()
		s.budget.UpdatePerTurn(memoryCtx, resourceCtx, req.Prompt, messages)
	}

	// Step counter (shared across callbacks)
	var stepCount int
	var stepMu sync.Mutex

	// Track active step span for proper parent-child nesting.
	// Tool calls within a step become children of the step span.
	var activeStepCtx context.Context
	var activeStepSpan trace.Span

	// Emit agent start with trace ID and context budget snapshot
	traceID := traceIDFromContext(ctx)
	budgetSnap := s.budget.Snapshot()
	emit.emitAgentStart(s.agentName, req.Prompt, traceID, &budgetSnap)

	result, usedModel, err := streamWithFallback(ctx, s.cfg, s.bundle, fantasy.AgentStreamCall{
		Prompt:   req.Prompt,
		Messages: messages,

		// ── Agent lifecycle ──

		OnAgentStart: func() {
			// Already emitted above before the call
		},

		OnAgentFinish: func(ar *fantasy.AgentResult) error {
			// Handled after streamWithFallback returns for cleaner flow
			return nil
		},

		// ── Step lifecycle ──

		OnStepStart: func(stepNumber int) error {
			stepMu.Lock()
			stepCount = stepNumber

			// End previous step span if still open
			if activeStepSpan != nil {
				activeStepSpan.End()
			}

			// Start a new step span as child of the root prompt span
			activeStepCtx, activeStepSpan = tracer.Start(ctx, "agent.step", trace.WithAttributes(
				attrStepNumber.Int(stepNumber),
				attrAgentName.String(s.agentName),
			))
			stepMu.Unlock()

			emit.emitStepStart(stepNumber, s.agentName)

			// Inject steer message if pending
			if msg := sc.popSteerMessage(); msg != "" {
				s.memory.Append(fantasy.NewUserMessage("[STEER] " + msg))
				slog.Info("steer message injected", "message", truncate(msg, 100))
			}

			return nil
		},

		OnStepFinish: func(sr fantasy.StepResult) error {
			stepMu.Lock()
			currentStep := stepCount
			stepSpan := activeStepSpan
			stepMu.Unlock()

			toolCallCount := len(sr.Content.ToolCalls())

			// Set step span attributes and end it
			if stepSpan != nil {
				stepSpan.SetAttributes(
					attrStepFinishReason.String(string(sr.FinishReason)),
					attrStepToolCalls.Int(toolCallCount),
					attrGenAIInputTokens.Int64(sr.Usage.InputTokens),
					attrGenAIOutputTokens.Int64(sr.Usage.OutputTokens),
				)
				if sr.Usage.ReasoningTokens > 0 {
					stepSpan.SetAttributes(attrGenAIReasoningTokens.Int64(sr.Usage.ReasoningTokens))
				}
				stepSpan.End()

				stepMu.Lock()
				activeStepSpan = nil
				activeStepCtx = nil
				stepMu.Unlock()
			}

			// Update budget with actual token usage from API response
			s.budget.UpdateActual(sr.Usage)
			stepBudgetSnap := s.budget.Snapshot()
			emit.emitStepFinish(currentStep, s.agentName, sr.Usage, sr.FinishReason, toolCallCount, &stepBudgetSnap)
			return nil
		},

		// ── Text streaming ──

		OnTextStart: func(id string) error {
			emit.emitTextStart(id)
			return nil
		},

		OnTextDelta: func(id, text string) error {
			emit.emitTextDelta(id, text)
			return nil
		},

		OnTextEnd: func(id string) error {
			emit.emitTextEnd(id)
			return nil
		},

		// ── Reasoning streaming ──

		OnReasoningStart: func(id string, _ fantasy.ReasoningContent) error {
			emit.emitReasoningStart(id)
			return nil
		},

		OnReasoningDelta: func(id, text string) error {
			emit.emitReasoningDelta(id, text)
			return nil
		},

		OnReasoningEnd: func(id string, _ fantasy.ReasoningContent) error {
			emit.emitReasoningEnd(id)
			return nil
		},

		// ── Tool input streaming (the big UX win — Pi cannot do this) ──

		OnToolInputStart: func(id, toolName string) error {
			emit.emitToolInputStart(id, toolName)
			return nil
		},

		OnToolInputDelta: func(id, delta string) error {
			emit.emitToolInputDelta(id, delta)
			return nil
		},

		OnToolInputEnd: func(id string) error {
			emit.emitToolInputEnd(id)
			return nil
		},

		// ── Tool execution ──

		OnToolCall: func(tc fantasy.ToolCallContent) error {
			emit.emitToolCall(tc.ToolCallID, tc.ToolName, tc.Input, tc.ProviderExecuted)
			return nil
		},

		OnToolResult: func(tr fantasy.ToolResultContent) error {
			// Extract output text and error status from the Result interface
			var outputText string
			var isError bool
			var mediaType, mediaData string

			switch v := tr.Result.(type) {
			case fantasy.ToolResultOutputContentText:
				outputText = v.Text
			case fantasy.ToolResultOutputContentError:
				outputText = v.Error.Error()
				isError = true
			case fantasy.ToolResultOutputContentMedia:
				outputText = v.Text
				mediaType = v.MediaType
				mediaData = v.Data
			}

			// Persist metadata so it survives working memory serialization
			s.memory.StoreToolMeta(tr.ToolCallID, tr.ClientMetadata)

			emit.emitToolResult(tr.ToolCallID, tr.ToolName, outputText, isError, tr.ClientMetadata, mediaType, mediaData)
			return nil
		},

		// ── Sources, warnings, stream finish ──

		OnSource: func(src fantasy.SourceContent) error {
			emit.emitSource(src.ID, string(src.SourceType), src.URL, src.Title)
			return nil
		},

		OnWarnings: func(warnings []fantasy.CallWarning) error {
			emit.emitWarnings(warnings)
			return nil
		},

		OnStreamFinish: func(u fantasy.Usage, fr fantasy.FinishReason, _ fantasy.ProviderMetadata) error {
			emit.emitStreamFinish(u, fr)
			return nil
		},

		// ── Error ──

		OnError: func(err error) {
			emit.emitAgentError(s.agentName, err, isRetryableError(err))
		},
	})

	// End any step span that wasn't closed (edge case: error during step)
	stepMu.Lock()
	if activeStepSpan != nil {
		activeStepSpan.End()
		activeStepSpan = nil
	}
	stepMu.Unlock()
	_ = activeStepCtx // used by tool spans via context propagation

	if err != nil {
		recordError(promptSpan, err)
		emit.emitAgentError(s.agentName, err, isRetryableError(err))
	} else {
		// Append to working memory
		s.memory.Append(fantasy.NewUserMessage(req.Prompt))
		for _, step := range result.Steps {
			s.memory.Append(step.Messages...)
		}
		s.memory.CompleteTurn()

		stepMu.Lock()
		finalSteps := stepCount
		stepMu.Unlock()

		s.mu.Lock()
		s.activeModel = usedModel
		s.totalSteps += finalSteps
		s.lastTraceID = traceID
		s.mu.Unlock()

		// Set final attributes on root span
		promptSpan.SetAttributes(
			attrGenAIResponseModel.String(usedModel),
			attrGenAIInputTokens.Int64(result.TotalUsage.InputTokens),
			attrGenAIOutputTokens.Int64(result.TotalUsage.OutputTokens),
			attribute.Int("agent.steps", finalSteps),
		)

		// Record the assistant response as a content event
		recordCompletionEvent(promptSpan, result.Response.Content.Text())

		// Emit agent finish with total usage
		emit.emitAgentFinish(s.agentName, result.TotalUsage, finalSteps, usedModel)
	}

	// Emit idle
	emit.emitSessionIdle(s.agentName)
}

// handleInternalPrompt runs the agent loop with an internal prompt (no HTTP request).
// Used by the DelegationWatcher to inject delegation results when the agent is idle.
// This follows the same flow as handlePrompt (non-streaming) to avoid requiring an
// active SSE connection. Results are appended to working memory.
func (s *daemonServer) handleInternalPrompt(prompt string) {
	sc := s.convCtx
	sc.mu.Lock()
	if sc.busy {
		// Race: agent became busy between our check and now — fall back to steer
		sc.steerMsg = prompt
		sc.mu.Unlock()
		slog.Info("delegation callback fell back to steer (agent became busy)")
		return
	}
	sc.busy = true
	sc.mu.Unlock()

	defer func() {
		sc.mu.Lock()
		sc.busy = false
		sc.cancel = nil
		sc.mu.Unlock()
	}()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	sc.mu.Lock()
	sc.cancel = cancel
	sc.mu.Unlock()
	defer cancel()

	// Start tracing span
	ctx, promptSpan := tracer.Start(ctx, "agent.prompt.internal", trace.WithAttributes(
		attrAgentName.String(s.agentName),
		attrAgentMode.String("daemon"),
		attrGenAIOperationName.String("invoke_agent"),
		attribute.String("prompt.source", "delegation_callback"),
	))
	defer promptSpan.End()

	recordPromptEvent(promptSpan, prompt)

	// Get messages from working memory
	messages := s.memory.Messages()

	// Set per-turn context
	var memoryCtx string
	if s.engram != nil {
		contextLimit := 5
		if s.cfg.Memory != nil && s.cfg.Memory.ContextLimit > 0 {
			contextLimit = s.cfg.Memory.ContextLimit
		}
		memoryCtx = s.engram.FetchContext(ctx, contextLimit, prompt)
	}
	s.injector.Set(memoryCtx, "")
	defer s.injector.Clear()

	// Pre-flight budget
	s.budget.UpdatePerTurn(memoryCtx, "", prompt, messages)
	convBudget := s.budget.ConversationBudget()
	if trimmed, _ := s.memory.TrimToTokenBudget(convBudget); trimmed > 0 {
		messages = s.memory.Messages()
		s.budget.UpdatePerTurn(memoryCtx, "", prompt, messages)
	}

	result, usedModel, err := generateWithFallback(ctx, s.cfg, s.bundle, fantasy.AgentCall{
		Prompt:   prompt,
		Messages: messages,
	})
	if err != nil {
		recordError(promptSpan, err)
		slog.Error("delegation callback prompt failed", "error", err)
		return
	}

	// Append to working memory
	s.memory.Append(fantasy.NewUserMessage(prompt))
	for _, step := range result.Steps {
		s.memory.Append(step.Messages...)
	}
	s.memory.CompleteTurn()

	traceID := traceIDFromContext(ctx)

	s.mu.Lock()
	s.activeModel = usedModel
	s.totalSteps += len(result.Steps)
	s.lastTraceID = traceID
	s.mu.Unlock()

	promptSpan.SetAttributes(
		attrGenAIResponseModel.String(usedModel),
		attrGenAIInputTokens.Int64(result.TotalUsage.InputTokens),
		attrGenAIOutputTokens.Int64(result.TotalUsage.OutputTokens),
		attribute.Int("agent.steps", len(result.Steps)),
	)

	recordCompletionEvent(promptSpan, result.Response.Content.Text())
	slog.Info("delegation callback prompt completed",
		"model", usedModel,
		"steps", len(result.Steps),
		"traceId", traceID,
	)
}

// ── Steer handler ──

func (s *daemonServer) handleSteer(w http.ResponseWriter, r *http.Request) {
	var req steerRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.Message == "" {
		http.Error(w, `{"error":"message is required"}`, http.StatusBadRequest)
		return
	}

	sc := s.convCtx
	sc.mu.Lock()
	sc.steerMsg = req.Message
	sc.mu.Unlock()

	slog.Info("steer message queued", "message", truncate(req.Message, 100))

	w.Header().Set("Content-Type", "application/json")
	io.WriteString(w, `{"ok":true}`)
}

// ── Abort handler ──

func (s *daemonServer) handleAbort(w http.ResponseWriter, r *http.Request) {
	sc := s.convCtx
	sc.mu.Lock()
	if sc.cancel != nil {
		sc.cancel()
	}
	sc.busy = false
	sc.mu.Unlock()

	slog.Info("conversation aborted")

	w.Header().Set("Content-Type", "application/json")
	io.WriteString(w, `{"ok":true}`)
}

// ── Permission reply handler ──

func (s *daemonServer) handlePermissionReply(w http.ResponseWriter, r *http.Request) {
	permId := r.PathValue("pid")

	var req PermissionResponse
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.Response == "" {
		http.Error(w, `{"error":"response is required (once|always|deny)"}`, http.StatusBadRequest)
		return
	}

	if !s.permGate.reply(permId, req) {
		http.Error(w, `{"error":"permission request not found or already resolved"}`, http.StatusNotFound)
		return
	}

	slog.Info("permission replied", "id", permId, "response", req.Response)
	w.Header().Set("Content-Type", "application/json")
	io.WriteString(w, `{"ok":true}`)
}

// ── Question reply handler ──

func (s *daemonServer) handleQuestionReply(w http.ResponseWriter, r *http.Request) {
	qId := r.PathValue("qid")

	var req QuestionResponse
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, `{"error":"answers are required"}`, http.StatusBadRequest)
		return
	}

	if !s.questionGate.reply(qId, req) {
		http.Error(w, `{"error":"question not found or already resolved"}`, http.StatusNotFound)
		return
	}

	slog.Info("question replied", "id", qId)
	w.Header().Set("Content-Type", "application/json")
	io.WriteString(w, `{"ok":true}`)
}

// ── Health and status ──

func (s *daemonServer) handleHealthz(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	io.WriteString(w, `{"status":"ok"}`)
}

func (s *daemonServer) handleStatus(w http.ResponseWriter, _ *http.Request) {
	s.mu.Lock()
	model := s.activeModel
	steps := s.totalSteps
	traceID := s.lastTraceID
	s.mu.Unlock()

	sc := s.convCtx
	sc.mu.Lock()
	busy := sc.busy
	sc.mu.Unlock()

	resp := map[string]any{
		"model":          model,
		"total_steps":    steps,
		"busy":           busy,
		"messages":       s.memory.MessageCount(),
		"turns":          s.memory.TurnCount(),
		"memory_enabled": s.engram != nil,
		"context_budget": s.budget.Snapshot(),
	}
	if traceID != "" {
		resp["trace_id"] = traceID
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// ── Get working memory (serialized for the console frontend) ──

// serializableToolOutput mirrors the frontend RuntimeToolOutput type.
type serializableToolOutput struct {
	Type      string `json:"type"`
	Text      string `json:"text,omitempty"`
	Error     string `json:"error,omitempty"`
	Data      string `json:"data,omitempty"`
	MediaType string `json:"media_type,omitempty"`
}

// serializablePart mirrors the frontend RuntimeMessagePart type.
type serializablePart struct {
	Type             string                  `json:"type"`
	Text             string                  `json:"text,omitempty"`
	Filename         string                  `json:"filename,omitempty"`
	Data             string                  `json:"data,omitempty"`
	MediaType        string                  `json:"media_type,omitempty"`
	ToolCallID       string                  `json:"tool_call_id,omitempty"`
	ToolName         string                  `json:"tool_name,omitempty"`
	Input            string                  `json:"input,omitempty"`
	ProviderExecuted bool                    `json:"provider_executed,omitempty"`
	Output           *serializableToolOutput `json:"output,omitempty"`
	Metadata         string                  `json:"metadata,omitempty"` // ClientMetadata JSON for tool-result parts
}

// serializableMessage mirrors the frontend RuntimeMessage type.
type serializableMessage struct {
	Role    string             `json:"role"`
	Content []serializablePart `json:"content"`
}

func (s *daemonServer) handleGetWorkingMemory(w http.ResponseWriter, r *http.Request) {
	messages := s.memory.Messages()
	toolMeta := s.memory.ToolMeta()

	result := make([]serializableMessage, 0, len(messages))
	for _, msg := range messages {
		sm := serializableMessage{
			Role:    string(msg.Role),
			Content: make([]serializablePart, 0, len(msg.Content)),
		}
		for _, part := range msg.Content {
			switch p := part.(type) {
			case fantasy.TextPart:
				sm.Content = append(sm.Content, serializablePart{
					Type: "text",
					Text: p.Text,
				})
			case fantasy.ReasoningPart:
				sm.Content = append(sm.Content, serializablePart{
					Type: "reasoning",
					Text: p.Text,
				})
			case fantasy.FilePart:
				sm.Content = append(sm.Content, serializablePart{
					Type:      "file",
					Filename:  p.Filename,
					Data:      string(p.Data),
					MediaType: p.MediaType,
				})
			case fantasy.ToolCallPart:
				sm.Content = append(sm.Content, serializablePart{
					Type:             "tool-call",
					ToolCallID:       p.ToolCallID,
					ToolName:         p.ToolName,
					Input:            p.Input,
					ProviderExecuted: p.ProviderExecuted,
				})
			case fantasy.ToolResultPart:
				sp := serializablePart{
					Type:       "tool-result",
					ToolCallID: p.ToolCallID,
					Metadata:   toolMeta[p.ToolCallID],
				}
				switch v := p.Output.(type) {
				case fantasy.ToolResultOutputContentText:
					sp.Output = &serializableToolOutput{Type: "text", Text: v.Text}
				case fantasy.ToolResultOutputContentError:
					sp.Output = &serializableToolOutput{Type: "error", Error: v.Error.Error()}
				case fantasy.ToolResultOutputContentMedia:
					sp.Output = &serializableToolOutput{Type: "media", Data: v.Data, MediaType: v.MediaType, Text: v.Text}
				}
				sm.Content = append(sm.Content, sp)
			default:
				sm.Content = append(sm.Content, serializablePart{Type: "unknown"})
			}
		}
		result = append(result, sm)
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(result)
}

// ── AI-assisted memory extraction ──

const memoryExtractionSystemPrompt = `You are a memory extraction assistant. Your job is to analyze a conversation between a user and an AI agent, then produce a structured observation suitable for long-term memory storage.

Output ONLY valid JSON with these fields:
{
  "type": "<one of: decision, bugfix, discovery, pattern, architecture, config, learning>",
  "title": "<concise title, max 80 chars>",
  "content": "<detailed content — what was learned, why it matters, how to apply it. 2-5 sentences.>",
  "tags": ["<relevant>", "<tags>"]
}

Guidelines:
- The "type" should match the nature of the knowledge: use "decision" for choices made, "bugfix" for problems solved, "discovery" for new findings, "pattern" for recurring approaches, "architecture" for structural decisions, "config" for configuration knowledge, "learning" for general lessons.
- The "title" should be specific and searchable — someone should find this by searching keywords.
- The "content" should be self-contained: readable without the original conversation. Include the WHY, not just the WHAT.
- The "tags" should be 2-5 lowercase keywords for categorization.
- If the user provides a focus hint, prioritize that aspect of the conversation.
- If the conversation is too short or trivial to extract meaningful knowledge, return: {"type":"learning","title":"No significant knowledge to extract","content":"The conversation did not contain extractable knowledge worth persisting.","tags":["empty"]}`

type memoryExtractRequest struct {
	Focus string `json:"focus,omitempty"` // optional focus hint from the user
	Type  string `json:"type,omitempty"`  // optional type hint
}

type memoryExtractResponse struct {
	Type    string   `json:"type"`
	Title   string   `json:"title"`
	Content string   `json:"content"`
	Tags    []string `json:"tags"`
}

func (s *daemonServer) handleMemoryExtract(w http.ResponseWriter, r *http.Request) {
	var req memoryExtractRequest
	// Body is optional — extraction works with no hints
	json.NewDecoder(r.Body).Decode(&req)

	// Get all messages from working memory
	messages := s.memory.Messages()
	if len(messages) == 0 {
		http.Error(w, `{"error":"working memory is empty — nothing to extract from"}`, http.StatusBadRequest)
		return
	}

	// Format conversation for the extraction model
	var convBuf strings.Builder
	convBuf.WriteString("=== Conversation ===\n\n")
	for _, msg := range messages {
		role := string(msg.Role)
		for _, part := range msg.Content {
			switch p := part.(type) {
			case fantasy.TextPart:
				if p.Text != "" {
					convBuf.WriteString(fmt.Sprintf("[%s] %s\n\n", role, p.Text))
				}
			case fantasy.ToolCallPart:
				convBuf.WriteString(fmt.Sprintf("[%s] Tool call: %s\n", role, p.ToolName))
			}
		}
	}

	// Build the extraction prompt
	var prompt strings.Builder
	prompt.WriteString(convBuf.String())
	prompt.WriteString("\n=== Task ===\n\n")
	prompt.WriteString("Extract the most important knowledge from the conversation above into a structured observation.\n")
	if req.Focus != "" {
		prompt.WriteString(fmt.Sprintf("\nUser focus: %s\n", req.Focus))
	}
	if req.Type != "" {
		prompt.WriteString(fmt.Sprintf("\nPreferred observation type: %s\n", req.Type))
	}
	prompt.WriteString("\nRespond with ONLY the JSON object, no markdown fences or extra text.")

	// Build a lightweight one-shot agent (no tools, custom system prompt)
	model, err := resolveModel(r.Context(), s.cfg.PrimaryModel, s.bundle.providers, s.cfg.PrimaryProvider)
	if err != nil {
		http.Error(w, fmt.Sprintf(`{"error":"model resolution failed: %s"}`, err.Error()), http.StatusInternalServerError)
		return
	}

	extractAgent := fantasy.NewAgent(model,
		fantasy.WithSystemPrompt(memoryExtractionSystemPrompt),
		fantasy.WithMaxOutputTokens(1024),
	)

	result, err := extractAgent.Generate(r.Context(), fantasy.AgentCall{
		Prompt: prompt.String(),
	})
	if err != nil {
		http.Error(w, fmt.Sprintf(`{"error":"extraction failed: %s"}`, err.Error()), http.StatusInternalServerError)
		return
	}

	output := strings.TrimSpace(result.Response.Content.Text())

	// Strip markdown code fences if the model wrapped it
	output = strings.TrimPrefix(output, "```json")
	output = strings.TrimPrefix(output, "```")
	output = strings.TrimSuffix(output, "```")
	output = strings.TrimSpace(output)

	// Parse the AI output
	var extracted memoryExtractResponse
	if err := json.Unmarshal([]byte(output), &extracted); err != nil {
		// If JSON parse fails, return the raw text so the frontend can still show something
		slog.Warn("extraction JSON parse failed, returning raw", "output", output, "error", err)
		extracted = memoryExtractResponse{
			Type:    "learning",
			Title:   "Extracted knowledge",
			Content: output,
			Tags:    []string{"auto-extracted"},
		}
	}

	// Apply type hint if provided and AI didn't override
	if req.Type != "" && extracted.Type == "" {
		extracted.Type = req.Type
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(extracted)
}

// ====================================================================
// Task mode: one-shot execution
// ====================================================================

type taskResult struct {
	Output         string `json:"output"`
	Steps          int    `json:"steps"`
	Model          string `json:"model"`
	Success        bool   `json:"success"`
	Error          string `json:"error,omitempty"`
	TraceID        string `json:"traceID,omitempty"`
	PullRequestURL string `json:"pullRequestURL,omitempty"`
	Commits        int    `json:"commits,omitempty"`
	Branch         string `json:"branch,omitempty"`
}

func runTask() error {
	ctx := context.Background()

	prompt := os.Getenv("AGENT_PROMPT")
	if prompt == "" {
		result := taskResult{Success: false, Error: "AGENT_PROMPT environment variable is not set"}
		writeTaskResult(result)
		return fmt.Errorf("AGENT_PROMPT not set")
	}

	cfg, err := loadConfig()
	if err != nil {
		result := taskResult{Success: false, Error: err.Error()}
		writeTaskResult(result)
		return err
	}

	// Initialize OpenTelemetry tracing
	agentName := os.Getenv("AGENT_NAME")
	if agentName == "" {
		agentName = "default"
	}
	agentNamespace := os.Getenv("AGENT_NAMESPACE")
	tracingFns, err := initTracing(ctx, agentName, agentNamespace, "task")
	if err != nil {
		slog.Warn("tracing init failed, continuing without tracing", "error", err)
	}
	defer func() {
		if tracingFns != nil {
			// Give the batch exporter up to 30s to flush all spans to Tempo.
			// ForceFlush first to ensure all ended spans (including tool.execute
			// children of gen_ai.generate) are exported before shutdown.
			// Without this, short-lived task pods often exit before the batch
			// exporter delivers tool spans, causing the console to fall back
			// to lower-fidelity virtual rows from tool.call events.
			flushCtx, flushCancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer flushCancel()
			if err := tracingFns.ForceFlush(flushCtx); err != nil {
				slog.Warn("tracing force flush error", "error", err)
			}
			if err := tracingFns.Shutdown(flushCtx); err != nil {
				slog.Warn("tracing shutdown error", "error", err)
			}
		}
	}()

	// Start root tracing span for this task execution
	// If this task was spawned by another agent (via run_agent), create a span link
	// back to the parent's orchestration span for cross-agent trace correlation.
	spanOpts := []trace.SpanStartOption{
		trace.WithAttributes(
			attrAgentName.String(agentName),
			attrAgentMode.String("task"),
			attrGenAIOperationName.String("invoke_agent"),
			attrGenAIProviderName.String(detectGenAIProvider(cfg.PrimaryModel, cfg.PrimaryProvider)),
			attrGenAIRequestModel.String(cfg.PrimaryModel),
		),
	}
	if tp := os.Getenv("TRACEPARENT"); tp != "" {
		parentAgent := os.Getenv("AGENT_RUN_SOURCE_AGENT")
		runName := os.Getenv("AGENT_RUN_NAME")
		spanOpts = append(spanOpts, delegationSpanOptions(tp, parentAgent, runName)...)
		slog.Info("delegation trace link created", "parentAgent", parentAgent, "runName", runName)
	}
	ctx, promptSpan := tracer.Start(ctx, "agent.prompt", spanOpts...)

	// Stash root span in context so tool wrappers can record tool.call events
	// that survive even when individual tool.execute spans are lost.
	ctx = withRootSpan(ctx, promptSpan)

	// Record the user prompt as a content event for trace visibility
	recordPromptEvent(promptSpan, prompt)

	// Set up git workspace if GIT_REPO_URL is set (injected by operator for spec.git runs)
	gitBranch := os.Getenv("GIT_BRANCH")
	if repoURL := os.Getenv("GIT_REPO_URL"); repoURL != "" {
		if err := setupGitWorkspace(repoURL, gitBranch, os.Getenv("GIT_BASE_BRANCH")); err != nil {
			recordError(promptSpan, err)
			promptSpan.End()
			result := taskResult{Success: false, Error: fmt.Sprintf("git workspace setup failed: %v", err), Branch: gitBranch, TraceID: traceIDFromContext(ctx)}
			writeTaskResult(result)
			return err
		}
	}

	slog.Info("running Fantasy task agent",
		"model", cfg.PrimaryModel,
		"prompt", truncate(prompt, 100),
	)

	// Task agents: memory tools optional (short-lived, but can still save observations)
	var engram *EngramClient
	if cfg.Memory != nil {
		project := cfg.Memory.Project
		if project == "" {
			project = agentName
		}
		engram = NewEngramClient(cfg.Memory.ServerURL, project)
		if err := engram.Init(); err != nil {
			slog.Warn("memory init failed for task, running without memory", "error", err)
			engram = nil
		}
	}

	// Create context injector for task agent
	taskInjector := &contextInjector{}

	// Set platform protocol — stable, injected as separate system message part
	if cfg.PlatformProtocol != "" {
		taskInjector.SetPlatformProtocol(cfg.PlatformProtocol)
	}

	// Fetch memory context (relevance-ranked by task prompt) and set on injector
	if engram != nil {
		contextLimit := 5
		if cfg.Memory != nil && cfg.Memory.ContextLimit > 0 {
			contextLimit = cfg.Memory.ContextLimit
		}
		if memCtx := engram.FetchContext(ctx, contextLimit, prompt); memCtx != "" {
			taskInjector.Set(memCtx, "")
		}
	}

	bundle, err := buildAgentBundle(ctx, cfg, engram, taskInjector, buildMemoryTools(engram)...)
	if err != nil {
		recordError(promptSpan, err)
		promptSpan.End()
		result := taskResult{Success: false, Error: err.Error(), Model: cfg.PrimaryModel, TraceID: traceIDFromContext(ctx)}
		writeTaskResult(result)
		return err
	}
	defer shutdownMCPConnections(bundle.mcpConns)

	agentResult, usedModel, err := generateWithFallback(ctx, cfg, bundle, fantasy.AgentCall{Prompt: prompt})
	if err != nil {
		recordError(promptSpan, err)
		promptSpan.End()
		result := taskResult{Success: false, Error: err.Error(), Model: cfg.PrimaryModel, TraceID: traceIDFromContext(ctx)}
		writeTaskResult(result)
		return err
	}

	output := agentResult.Response.Content.Text()
	traceID := traceIDFromContext(ctx)

	// Record the assistant response as a content event
	recordCompletionEvent(promptSpan, output)

	result := taskResult{
		Output:  output,
		Steps:   len(agentResult.Steps),
		Model:   usedModel,
		Success: true,
		TraceID: traceID,
		Branch:  gitBranch,
	}

	// Set final attributes on root span
	promptSpan.SetAttributes(
		attrGenAIResponseModel.String(usedModel),
		attrGenAIInputTokens.Int64(agentResult.TotalUsage.InputTokens),
		attrGenAIOutputTokens.Int64(agentResult.TotalUsage.OutputTokens),
		attribute.Int("agent.steps", len(agentResult.Steps)),
	)

	// Extract git info from the agent's output if this was a git workspace run
	if os.Getenv("GIT_REPO_URL") != "" {
		result.Commits, _ = extractGitInfo()
		result.PullRequestURL = extractPullRequestURL(agentResult.Steps)
	}

	// End memory session with raw messages (fixes session leak in task mode).
	if engram != nil {
		taskMessages := []fantasy.Message{
			fantasy.NewUserMessage(prompt),
		}
		for _, step := range agentResult.Steps {
			taskMessages = append(taskMessages, step.Messages...)
		}
		engram.EndSession(fantasyToEngramMessages(taskMessages))
	}

	promptSpan.End()
	writeTaskResult(result)

	return nil
}

// writeTaskResult writes the task result to both stdout (for logs) and
// /dev/termination-log (so the operator can read it from pod status).
func writeTaskResult(result taskResult) {
	json.NewEncoder(os.Stdout).Encode(result)
	if data, err := json.Marshal(result); err == nil {
		os.WriteFile("/dev/termination-log", data, 0644)
	}
}

// ====================================================================
// Orchestration tools (run_agent, get_agent_run)
// ====================================================================

type runAgentInput struct {
	Agent         string `json:"agent" description:"Agent name to run"`
	Prompt        string `json:"prompt" description:"Prompt to send to the agent"`
	GitResource   string `json:"git_resource,omitempty" description:"AgentResource name for git workspace (github-repo, gitlab-project, or git-repo)"`
	GitBranch     string `json:"git_branch,omitempty" description:"Feature branch to work on. Created from base branch if it doesn't exist."`
	GitBaseBranch string `json:"git_base_branch,omitempty" description:"Base branch for PR/MR target (e.g. main). Defaults to repo default."`
}

// buildRunAgentDescription constructs a dynamic tool description that includes
// the available git resources bound to this agent. This gives the LLM full
// context about what repos it can delegate work to without needing to query
// the cluster.
func buildRunAgentDescription(resources []ResourceEntry) string {
	base := "Delegate a task to another agent. Creates an AgentRun tracked by the operator. " +
		"IMPORTANT: After calling run_agent, report back to the user that the task has been delegated " +
		"and offer to check on it later. Do NOT automatically call get_agent_run — let the user decide when to check."

	// Build git resource list
	var gitResources []string
	for _, r := range resources {
		switch r.Kind {
		case "github-repo":
			if r.GitHub != nil {
				detail := fmt.Sprintf("  - %q (GitHub: %s/%s", r.Name, r.GitHub.Owner, r.GitHub.Repo)
				if r.GitHub.DefaultBranch != "" {
					detail += fmt.Sprintf(", default branch: %s", r.GitHub.DefaultBranch)
				}
				detail += ")"
				if r.Description != "" {
					detail += " — " + r.Description
				}
				gitResources = append(gitResources, detail)
			}
		case "gitlab-project":
			if r.GitLab != nil {
				detail := fmt.Sprintf("  - %q (GitLab: %s", r.Name, r.GitLab.Project)
				if r.GitLab.DefaultBranch != "" {
					detail += fmt.Sprintf(", default branch: %s", r.GitLab.DefaultBranch)
				}
				detail += ")"
				if r.Description != "" {
					detail += " — " + r.Description
				}
				gitResources = append(gitResources, detail)
			}
		case "git-repo":
			if r.Git != nil {
				detail := fmt.Sprintf("  - %q (git: %s)", r.Name, r.Git.URL)
				if r.Description != "" {
					detail += " — " + r.Description
				}
				gitResources = append(gitResources, detail)
			}
		}
	}

	if len(gitResources) > 0 {
		base += "\n\nFor coding/git tasks, set git_resource + git_branch to give the task agent a cloned repo workspace. " +
			"The agent will clone the repo, create/checkout the branch, work, commit, push, and create a PR/MR.\n\n" +
			"Available git resources (use the name as git_resource value):\n" +
			strings.Join(gitResources, "\n")
	}

	return base
}

func newRunAgentTool(k8s *K8sClient, resources []ResourceEntry) fantasy.AgentTool {
	desc := buildRunAgentDescription(resources)
	return fantasy.NewAgentTool("run_agent", desc,
		func(ctx context.Context, input runAgentInput, _ fantasy.ToolCall) (fantasy.ToolResponse, error) {
			if input.Agent == "" || input.Prompt == "" {
				return fantasy.NewTextErrorResponse("agent and prompt are required"), nil
			}

			// Validate git params: if any git field is set, resource + branch are required
			if input.GitResource != "" || input.GitBranch != "" {
				if input.GitResource == "" || input.GitBranch == "" {
					return fantasy.NewTextErrorResponse("git_resource and git_branch are both required when using a git workspace"), nil
				}
			}

			// Prevent self-invocation (daemon agents can't handle concurrent prompts)
			agentName := os.Getenv("AGENT_NAME")
			if agentName == "" {
				agentName = "unknown"
			}
			if input.Agent == agentName {
				return fantasy.NewTextErrorResponse(fmt.Sprintf(
					"Cannot run_agent on yourself (%q). Use run_agent to trigger a different agent (typically a task-mode agent).", agentName)), nil
			}

			// Validate target agent exists before creating the AgentRun CR
			agentInfo, err := k8s.GetAgent(ctx, input.Agent)
			if err != nil {
				// Agent not found — list available agents to help the LLM
				available, listErr := k8s.ListAgents(ctx)
				if listErr != nil || len(available) == 0 {
					return fantasy.NewTextErrorResponse(fmt.Sprintf(
						"Agent %q not found. Could not list available agents.", input.Agent)), nil
				}
				var agentList string
				for _, a := range available {
					if a.Name != agentName { // exclude self
						agentList += fmt.Sprintf("\n  - %s (mode: %s, phase: %s)", a.Name, a.Mode, a.Phase)
					}
				}
				if agentList == "" {
					return fantasy.NewTextErrorResponse(fmt.Sprintf(
						"Agent %q not found. No other agents are available to run.", input.Agent)), nil
				}
				return fantasy.NewTextErrorResponse(fmt.Sprintf(
					"Agent %q not found. Available agents:%s", input.Agent, agentList)), nil
			}

			// Check delegation constraints: is this agent allowed to delegate to the target?
			targetScope, targetCallers, discErr := k8s.GetAgentDiscovery(ctx, input.Agent)
			if discErr == nil && !isAgentVisible(targetScope, targetCallers, agentName) {
				return fantasy.NewTextErrorResponse(fmt.Sprintf(
					"Agent %q is not available for delegation (scope: %s). You are not in its allowedCallers list.",
					input.Agent, targetScope)), nil
			}

			// Warn if targeting a daemon (will use HTTP prompt, not spawn a Job)
			if agentInfo.Mode == "daemon" && agentInfo.Phase != "Running" {
				return fantasy.NewTextErrorResponse(fmt.Sprintf(
					"Agent %q is a daemon in phase %q (not Running). Wait for it to be Running first.", input.Agent, agentInfo.Phase)), nil
			}

			// Build optional git params
			var gitParams *AgentRunGitParams
			if input.GitResource != "" {
				gitParams = &AgentRunGitParams{
					ResourceRef: input.GitResource,
					Branch:      input.GitBranch,
					BaseBranch:  input.GitBaseBranch,
				}
			}

			run, err := k8s.CreateAgentRun(ctx, input.Agent, input.Prompt, "agent", agentName, traceparentFromContext(ctx), gitParams, nil)
			if err != nil {
				return fantasy.NewTextErrorResponse(fmt.Sprintf("Failed to create AgentRun: %s", err)), nil
			}

			// Set delegation attributes on the current tool.execute span so the
			// parent trace waterfall can link to the child agent's trace.
			// The span was created by hookWrappedTool.Run() in hooks.go.
			currentSpan := trace.SpanFromContext(ctx)
			currentSpan.SetAttributes(
				attribute.String("delegation.child_agent", input.Agent),
				attribute.String("delegation.child_run", run.Name),
				attribute.String("delegation.child_namespace", os.Getenv("AGENT_NAMESPACE")),
			)

			modeHint := ""
			if agentInfo.Mode == "task" {
				modeHint = " A task pod will be created to execute this."
			}
			if gitParams != nil {
				modeHint += fmt.Sprintf(" Git workspace: resource=%s branch=%s.", gitParams.ResourceRef, gitParams.Branch)
			}

			resp := fantasy.NewTextResponse(fmt.Sprintf("AgentRun %s created for agent %q.%s\n\nTell the user the task has been delegated. They can ask you to check on it anytime with get_agent_run name=%q.", run.Name, input.Agent, modeHint, run.Name))
			resp = fantasy.WithResponseMetadata(resp, map[string]any{
				"ui":        "agent-run",
				"agent":     input.Agent,
				"runName":   run.Name,
				"namespace": os.Getenv("AGENT_NAMESPACE"),
			})
			return resp, nil
		})
}

func newRunAgentToolStub() fantasy.AgentTool {
	return fantasy.NewAgentTool("run_agent",
		"Trigger another agent with a prompt. (Unavailable: K8s client not configured)",
		func(_ context.Context, _ runAgentInput, _ fantasy.ToolCall) (fantasy.ToolResponse, error) {
			return fantasy.NewTextErrorResponse("run_agent unavailable: not running in Kubernetes"), nil
		})
}

type getAgentRunInput struct {
	Name string `json:"name" description:"AgentRun name to check"`
}

func newGetAgentRunTool(k8s *K8sClient) fantasy.AgentTool {
	return fantasy.NewAgentTool("get_agent_run",
		"Check the status and output of an AgentRun. Returns phase, output, model, tool call count, and git info (PR URL, commits, branch) if applicable.",
		func(ctx context.Context, input getAgentRunInput, _ fantasy.ToolCall) (fantasy.ToolResponse, error) {
			if input.Name == "" {
				return fantasy.NewTextErrorResponse("name is required"), nil
			}
			status, err := k8s.GetAgentRun(ctx, input.Name)
			if err != nil {
				return fantasy.NewTextErrorResponse(fmt.Sprintf("Failed to get AgentRun: %s", err)), nil
			}

			// Build a rich text response
			text := fmt.Sprintf("Phase: %s", status.Phase)
			if status.Model != "" {
				text += fmt.Sprintf("\nModel: %s", status.Model)
			}
			if status.ToolCalls > 0 {
				text += fmt.Sprintf("\nTool calls: %d", status.ToolCalls)
			}
			if status.Output != "" {
				text += fmt.Sprintf("\nOutput: %s", status.Output)
			}

			// Git-specific status
			if status.PullRequestURL != "" {
				text += fmt.Sprintf("\nPull Request: %s", status.PullRequestURL)
			}
			if status.Commits > 0 {
				text += fmt.Sprintf("\nCommits: %d", status.Commits)
			}
			if status.Branch != "" {
				text += fmt.Sprintf("\nBranch: %s", status.Branch)
			}

			// Hint if still running
			if status.Phase == "Running" || status.Phase == "Pending" || status.Phase == "Queued" || status.Phase == "Unknown" {
				text += "\n\n(Run is still in progress. Report this to the user — they can ask you to check again later.)"
			}

			metadata := map[string]any{
				"ui":     "agent-run-status",
				"name":   input.Name,
				"phase":  status.Phase,
				"output": status.Output,
			}
			if status.PullRequestURL != "" {
				metadata["pullRequestURL"] = status.PullRequestURL
			}
			if status.Branch != "" {
				metadata["branch"] = status.Branch
			}

			resp := fantasy.NewTextResponse(text)
			resp = fantasy.WithResponseMetadata(resp, metadata)
			return resp, nil
		})
}

func newGetAgentRunToolStub() fantasy.AgentTool {
	return fantasy.NewAgentTool("get_agent_run",
		"Check the status and output of an AgentRun. (Unavailable: K8s client not configured)",
		func(_ context.Context, _ getAgentRunInput, _ fantasy.ToolCall) (fantasy.ToolResponse, error) {
			return fantasy.NewTextErrorResponse("get_agent_run unavailable: not running in Kubernetes"), nil
		})
}

// ====================================================================
// list_task_agents — discover available agents and their resources
// ====================================================================

type listTaskAgentsInput struct {
	IncludeDaemons bool `json:"include_daemons,omitempty" description:"Include daemon agents in the list (default: only task agents)"`
}

func newListTaskAgentsTool(k8s *K8sClient) fantasy.AgentTool {
	return fantasy.NewAgentTool("list_task_agents",
		"List available agents with their capabilities and bound resources. "+
			"Returns each agent's name, mode, phase, model, description (or system prompt summary), tags, and bound resources "+
			"(git repos with owner/project info and default branches). "+
			"Agents with scope=hidden are excluded. Agents with scope=explicit are only shown if you are in their allowedCallers. "+
			"Use this to decide which agent to delegate to and which git resources are available for run_agent.",
		func(ctx context.Context, input listTaskAgentsInput, _ fantasy.ToolCall) (fantasy.ToolResponse, error) {
			selfName := os.Getenv("AGENT_NAME")
			if selfName == "" {
				selfName = "unknown"
			}

			agents, err := k8s.ListAgentDetails(ctx, selfName)
			if err != nil {
				return fantasy.NewTextErrorResponse(fmt.Sprintf("Failed to list agents: %s", err)), nil
			}

			var text strings.Builder
			text.WriteString("Available agents:\n")
			count := 0

			for _, a := range agents {
				// Skip self
				if a.Name == selfName {
					continue
				}
				// Filter by mode
				if !input.IncludeDaemons && a.Mode == "daemon" {
					continue
				}

				count++
				text.WriteString(fmt.Sprintf("\n## %s\n", a.Name))
				text.WriteString(fmt.Sprintf("  Mode: %s | Phase: %s", a.Mode, a.Phase))
				if a.Model != "" {
					text.WriteString(fmt.Sprintf(" | Model: %s", a.Model))
				}
				text.WriteString("\n")

				// Prefer discovery.description over truncated system prompt
				if a.Description != "" {
					text.WriteString(fmt.Sprintf("  Description: %s\n", a.Description))
				} else if a.SystemPrompt != "" {
					text.WriteString(fmt.Sprintf("  Purpose: %s\n", a.SystemPrompt))
				}

				// Show tags if present
				if len(a.Tags) > 0 {
					text.WriteString(fmt.Sprintf("  Tags: %s\n", strings.Join(a.Tags, ", ")))
				}

				if len(a.ResourceBindings) > 0 {
					text.WriteString("  Resources:\n")
					for _, r := range a.ResourceBindings {
						switch r.Kind {
						case "github-repo":
							text.WriteString(fmt.Sprintf("    - %q (GitHub: %s/%s", r.Name, r.GitHubOwner, r.GitHubRepo))
							if r.DefaultBranch != "" {
								text.WriteString(fmt.Sprintf(", default: %s", r.DefaultBranch))
							}
							text.WriteString(")")
						case "gitlab-project":
							text.WriteString(fmt.Sprintf("    - %q (GitLab: %s", r.Name, r.GitLabProject))
							if r.DefaultBranch != "" {
								text.WriteString(fmt.Sprintf(", default: %s", r.DefaultBranch))
							}
							text.WriteString(")")
						case "git-repo":
							text.WriteString(fmt.Sprintf("    - %q (git: %s)", r.Name, r.GitURL))
						default:
							text.WriteString(fmt.Sprintf("    - %q (%s)", r.Name, r.Kind))
						}
						if r.Description != "" {
							text.WriteString(fmt.Sprintf(" — %s", r.Description))
						}
						text.WriteString("\n")
					}
				}
			}

			if count == 0 {
				return fantasy.NewTextResponse("No other agents found in this namespace."), nil
			}

			text.WriteString(fmt.Sprintf("\n(%d agents listed. Use run_agent to delegate tasks.)", count))

			resp := fantasy.NewTextResponse(text.String())
			resp = fantasy.WithResponseMetadata(resp, map[string]any{
				"ui":    "agent-list",
				"count": count,
			})
			return resp, nil
		})
}

func newListTaskAgentsToolStub() fantasy.AgentTool {
	return fantasy.NewAgentTool("list_task_agents",
		"List available agents with their capabilities and bound resources. (Unavailable: K8s client not configured)",
		func(_ context.Context, _ listTaskAgentsInput, _ fantasy.ToolCall) (fantasy.ToolResponse, error) {
			return fantasy.NewTextErrorResponse("list_task_agents unavailable: not running in Kubernetes"), nil
		})
}
