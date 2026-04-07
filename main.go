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
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"time"

	"charm.land/fantasy"
)

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
}

func buildAgentBundle(ctx context.Context, cfg *Config) (*agentBundle, error) {
	// Resolve providers
	providers := make(map[string]fantasy.Provider)
	for _, p := range cfg.Providers {
		provider, err := resolveProvider(p.Name)
		if err != nil {
			slog.Warn("failed to resolve provider", "name", p.Name, "error", err)
			continue
		}
		providers[p.Name] = provider
		slog.Info("registered provider", "name", p.Name)
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

	// Wrap with security hooks
	tools = wrapToolsWithHooks(tools, cfg.ToolHooks)

	// Add orchestration tools (run_agent, get_agent_run)
	k8sClient, err := NewK8sClient()
	if err != nil {
		slog.Warn("K8s client unavailable, orchestration tools disabled", "error", err)
		tools = append(tools, newRunAgentToolStub(), newGetAgentRunToolStub())
	} else {
		tools = append(tools, newRunAgentTool(k8sClient), newGetAgentRunTool(k8sClient))
	}

	// Build agent options
	opts := []fantasy.AgentOption{
		fantasy.WithTools(tools...),
	}

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

	return &agentBundle{
		agent:     fantasy.NewAgent(model, opts...),
		providers: providers,
		mcpConns:  allMCPConns,
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
func streamWithFallback(ctx context.Context, cfg *Config, bundle *agentBundle, call fantasy.AgentStreamCall) (*fantasy.AgentResult, string, error) {
	result, err := bundle.agent.Stream(ctx, call)
	if err == nil {
		return result, cfg.PrimaryModel, nil
	}

	if !isRetryableError(err) || len(cfg.FallbackModels) == 0 {
		return nil, cfg.PrimaryModel, err
	}

	slog.Warn("primary model failed on stream, trying fallbacks",
		"model", cfg.PrimaryModel, "error", err)

	for _, fbModel := range cfg.FallbackModels {
		fbAgent, fbErr := buildFallbackAgent(ctx, cfg, bundle.providers, fbModel)
		if fbErr != nil {
			continue
		}

		result, err = fbAgent.Stream(ctx, call)
		if err == nil {
			return result, fbModel, nil
		}

		if !isRetryableError(err) {
			return nil, fbModel, err
		}
	}

	return nil, cfg.PrimaryModel, fmt.Errorf("all models failed, last error: %w", err)
}

// generateWithFallback tries the primary model, then fallbacks on retryable errors.
func generateWithFallback(ctx context.Context, cfg *Config, bundle *agentBundle, call fantasy.AgentCall) (*fantasy.AgentResult, string, error) {
	result, err := bundle.agent.Generate(ctx, call)
	if err == nil {
		return result, cfg.PrimaryModel, nil
	}

	if !isRetryableError(err) || len(cfg.FallbackModels) == 0 {
		return nil, cfg.PrimaryModel, err
	}

	slog.Warn("primary model failed, trying fallbacks",
		"model", cfg.PrimaryModel, "error", err)

	for _, fbModel := range cfg.FallbackModels {
		fbAgent, fbErr := buildFallbackAgent(ctx, cfg, bundle.providers, fbModel)
		if fbErr != nil {
			slog.Warn("failed to build fallback agent", "model", fbModel, "error", fbErr)
			continue
		}

		result, err = fbAgent.Generate(ctx, call)
		if err == nil {
			slog.Info("fallback model succeeded", "model", fbModel)
			return result, fbModel, nil
		}

		if !isRetryableError(err) {
			return nil, fbModel, err
		}
		slog.Warn("fallback model failed", "model", fbModel, "error", err)
	}

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
		"builtinTools", len(cfg.BuiltinTools),
		"ociTools", len(cfg.Tools),
		"mcpServers", len(cfg.MCPServers),
		"fallbackModels", len(cfg.FallbackModels),
	)

	bundle, err := buildAgentBundle(ctx, cfg)
	if err != nil {
		return err
	}
	defer shutdownMCPConnections(bundle.mcpConns)

	// Session data directory: /data/sessions by default, override with DATA_DIR env var.
	sessionDir := "/data/sessions"
	if d := os.Getenv("DATA_DIR"); d != "" {
		sessionDir = filepath.Join(d, "sessions")
	}

	srv := &daemonServer{
		bundle:      bundle,
		cfg:         cfg,
		sessions:    NewSessionStore(sessionDir),
		activeModel: cfg.PrimaryModel,
		sessionCtx:  make(map[string]*sessionContext),
	}

	// Initialize permission gate (emits permission_asked via the session's active SSE emitter)
	srv.permGate = newPermissionGate(
		func(id, sessionId, toolName, input, description string) {
			sc := srv.getOrCreateSessionCtx(sessionId)
			sc.mu.Lock()
			emit := sc.emitter
			sc.mu.Unlock()
			if emit != nil {
				emit.emitPermissionAsked(id, sessionId, toolName, input, description)
			}
		},
	)

	// Initialize question gate (emits question_asked via the session's active SSE emitter)
	srv.questionGate = newQuestionGate(
		func(id, sessionId string, questions json.RawMessage) {
			sc := srv.getOrCreateSessionCtx(sessionId)
			sc.mu.Lock()
			emit := sc.emitter
			sc.mu.Unlock()
			if emit != nil {
				emit.emitQuestionAsked(id, sessionId, questions)
			}
		},
	)

	// Apply permission wrapping if configured
	if len(cfg.PermissionTools) > 0 {
		agent := bundle.agent
		opts := []fantasy.AgentOption{}

		// Rebuild tools with permission gates applied
		tools := buildBuiltinTools(cfg.BuiltinTools)
		if len(cfg.Tools) > 0 {
			ociTools, _, _ := loadOCITools(ctx, cfg.Tools)
			tools = append(tools, ociTools...)
		}
		if len(cfg.MCPServers) > 0 {
			gwTools, _, _ := loadGatewayMCPTools(ctx, cfg.MCPServers)
			tools = append(tools, gwTools...)
		}
		tools = wrapToolsWithHooks(tools, cfg.ToolHooks)

		// Wrap with permission gates
		tools = srv.permGate.wrapTools(tools, cfg.PermissionTools)

		// Add orchestration tools
		k8sClient, _ := NewK8sClient()
		if k8sClient != nil {
			tools = append(tools, newRunAgentTool(k8sClient), newGetAgentRunTool(k8sClient))
		}

		// Add question tool if enabled
		if cfg.EnableQuestionTool {
			tools = append(tools, newQuestionTool(srv.questionGate))
		}

		opts = append(opts, fantasy.WithTools(tools...))
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

		// Resolve model
		model, _ := resolveModel(ctx, cfg.PrimaryModel, bundle.providers, cfg.PrimaryProvider)
		if model != nil {
			bundle.agent = fantasy.NewAgent(model, opts...)
		} else {
			_ = agent // keep original if model resolution fails
		}

		slog.Info("permission gates applied", "tools", cfg.PermissionTools)
	} else if cfg.EnableQuestionTool {
		// Only need to add question tool (no permission wrapping needed)
		// Rebuild agent with question tool added
		tools := buildBuiltinTools(cfg.BuiltinTools)
		if len(cfg.Tools) > 0 {
			ociTools, _, _ := loadOCITools(ctx, cfg.Tools)
			tools = append(tools, ociTools...)
		}
		if len(cfg.MCPServers) > 0 {
			gwTools, _, _ := loadGatewayMCPTools(ctx, cfg.MCPServers)
			tools = append(tools, gwTools...)
		}
		tools = wrapToolsWithHooks(tools, cfg.ToolHooks)

		k8sClient, _ := NewK8sClient()
		if k8sClient != nil {
			tools = append(tools, newRunAgentTool(k8sClient), newGetAgentRunTool(k8sClient))
		}

		tools = append(tools, newQuestionTool(srv.questionGate))

		opts := []fantasy.AgentOption{fantasy.WithTools(tools...)}
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

		model, _ := resolveModel(ctx, cfg.PrimaryModel, bundle.providers, cfg.PrimaryProvider)
		if model != nil {
			bundle.agent = fantasy.NewAgent(model, opts...)
		}

		slog.Info("question tool enabled")
	}

	mux := http.NewServeMux()

	// Legacy endpoints (kept for backward compat, use default session)
	mux.HandleFunc("POST /prompt", srv.handlePromptLegacy)
	mux.HandleFunc("POST /prompt/stream", srv.handlePromptStreamLegacy)

	// Session CRUD
	mux.HandleFunc("POST /sessions", srv.handleSessionCreate)
	mux.HandleFunc("GET /sessions", srv.handleSessionList)
	mux.HandleFunc("GET /sessions/{id}", srv.handleSessionGet)
	mux.HandleFunc("GET /sessions/{id}/messages", srv.handleSessionMessages)
	mux.HandleFunc("DELETE /sessions/{id}", srv.handleSessionDelete)

	// Session-scoped prompt/stream
	mux.HandleFunc("POST /sessions/{id}/prompt", srv.handleSessionPrompt)
	mux.HandleFunc("POST /sessions/{id}/prompt/stream", srv.handleSessionPromptStream)

	// Session-scoped control
	mux.HandleFunc("POST /sessions/{id}/steer", srv.handleSessionSteer)
	mux.HandleFunc("DELETE /sessions/{id}/abort", srv.handleSessionAbort)

	// Permission and question reply endpoints
	mux.HandleFunc("POST /sessions/{id}/permission/{pid}/reply", srv.handlePermissionReply)
	mux.HandleFunc("POST /sessions/{id}/question/{qid}/reply", srv.handleQuestionReply)

	// Health and status
	mux.HandleFunc("GET /healthz", srv.handleHealthz)
	mux.HandleFunc("GET /status", srv.handleStatus)

	httpSrv := &http.Server{Addr: fmt.Sprintf(":%d", port), Handler: mux}

	go func() {
		<-ctx.Done()
		slog.Info("shutting down...")
		httpSrv.Close()
	}()

	slog.Info("listening", "port", port)
	return httpSrv.ListenAndServe()
}

// sessionContext tracks per-session runtime state (cancel func, busy flag, steer msgs).
type sessionContext struct {
	mu        sync.Mutex
	busy      bool
	cancel    context.CancelFunc
	steerMsg  string      // pending steer message, consumed on next step boundary
	emitter   *fepEmitter // current SSE emitter (set during streaming)
	sessionId string      // this session's ID
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
	sessions     *SessionStore
	activeModel  string
	totalSteps   int
	permGate     *permissionGate
	questionGate *questionGate

	mu         sync.Mutex
	sessionCtx map[string]*sessionContext // sessionId -> runtime context
}

// getOrCreateSessionCtx returns (or creates) the runtime context for a session.
func (s *daemonServer) getOrCreateSessionCtx(sessionId string) *sessionContext {
	s.mu.Lock()
	defer s.mu.Unlock()
	sc, ok := s.sessionCtx[sessionId]
	if !ok {
		sc = &sessionContext{}
		s.sessionCtx[sessionId] = sc
	}
	return sc
}

func (s *daemonServer) deleteSessionCtx(sessionId string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if sc, ok := s.sessionCtx[sessionId]; ok {
		sc.mu.Lock()
		if sc.cancel != nil {
			sc.cancel()
		}
		sc.mu.Unlock()
		delete(s.sessionCtx, sessionId)
	}
}

// generateTitle fires a background goroutine that calls the LLM to generate
// a short, descriptive title for a session based on the user's first prompt.
// The title is persisted to the session store; the frontend picks it up on
// its next session list refetch (triggered by agent_finish).
func (s *daemonServer) generateTitle(sessionId, userPrompt string) {
	// Set an immediate fallback title so the session isn't untitled while we wait
	s.sessions.UpdateTitle(sessionId, truncate(userPrompt, 80))

	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()

		model, err := resolveModel(ctx, s.cfg.PrimaryModel, s.bundle.providers, s.cfg.PrimaryProvider)
		if err != nil {
			slog.Warn("title generation: failed to resolve model", "error", err)
			return // fallback title already set
		}

		titleAgent := fantasy.NewAgent(model,
			fantasy.WithSystemPrompt("Generate a short title (max 60 chars) for a conversation that starts with the following user message. Reply with ONLY the title, no quotes, no punctuation at the end, no explanation."),
			fantasy.WithMaxOutputTokens(100),
		)

		result, err := titleAgent.Generate(ctx, fantasy.AgentCall{
			Prompt: userPrompt,
		})
		if err != nil {
			slog.Warn("title generation: LLM call failed", "error", err)
			return // fallback title already set
		}

		title := strings.TrimSpace(result.Response.Content.Text())
		if title == "" {
			return // keep fallback
		}
		if len(title) > 80 {
			title = truncate(title, 80)
		}

		s.sessions.UpdateTitle(sessionId, title)
		slog.Info("session title generated", "session", sessionId, "title", title)
	}()
}

// ── Request/Response types ──

type promptRequest struct {
	Prompt    string `json:"prompt"`
	SessionID string `json:"session_id,omitempty"` // optional, for legacy endpoints
}

type promptResponse struct {
	Output string `json:"output"`
	Model  string `json:"model"`
}

type sessionCreateRequest struct {
	Title string `json:"title,omitempty"`
}

type sessionCreateResponse struct {
	ID    string `json:"id"`
	Title string `json:"title"`
}

type steerRequest struct {
	Message string `json:"message"`
}

// ── Session CRUD handlers ──

func (s *daemonServer) handleSessionCreate(w http.ResponseWriter, r *http.Request) {
	var req sessionCreateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		// Allow empty body — title defaults to "New Session"
		req = sessionCreateRequest{}
	}

	session := s.sessions.Create(req.Title)
	slog.Info("session created", "id", session.ID, "title", session.Title)

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(sessionCreateResponse{
		ID:    session.ID,
		Title: session.Title,
	})
}

func (s *daemonServer) handleSessionList(w http.ResponseWriter, _ *http.Request) {
	sessions := s.sessions.List()
	infos := make([]SessionInfo, len(sessions))
	for i, sess := range sessions {
		infos[i] = sess.Info()
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(infos)
}

func (s *daemonServer) handleSessionGet(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	session, ok := s.sessions.Get(id)
	if !ok {
		http.Error(w, `{"error":"session not found"}`, http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(session.Info())
}

func (s *daemonServer) handleSessionMessages(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	msgs, ok := s.sessions.GetSerializedMessages(id)
	if !ok {
		http.Error(w, `{"error":"session not found"}`, http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(msgs)
}

func (s *daemonServer) handleSessionDelete(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")

	// Cancel any active work
	s.deleteSessionCtx(id)

	if !s.sessions.Delete(id) {
		http.Error(w, `{"error":"session not found"}`, http.StatusNotFound)
		return
	}

	slog.Info("session deleted", "id", id)
	w.Header().Set("Content-Type", "application/json")
	io.WriteString(w, `{"ok":true}`)
}

// ── Session-scoped prompt (non-streaming) ──

func (s *daemonServer) handleSessionPrompt(w http.ResponseWriter, r *http.Request) {
	sessionId := r.PathValue("id")
	sess, ok := s.sessions.Get(sessionId)
	if !ok {
		http.Error(w, `{"error":"session not found"}`, http.StatusNotFound)
		return
	}

	var req promptRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.Prompt == "" {
		http.Error(w, `{"error":"prompt is required"}`, http.StatusBadRequest)
		return
	}

	sc := s.getOrCreateSessionCtx(sessionId)
	sc.mu.Lock()
	if sc.busy {
		sc.mu.Unlock()
		http.Error(w, `{"error":"session is busy"}`, http.StatusTooManyRequests)
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

	messages := s.sessions.GetMessages(sessionId)

	result, usedModel, err := generateWithFallback(ctx, s.cfg, s.bundle, fantasy.AgentCall{
		Prompt:   req.Prompt,
		Messages: messages,
	})
	if err != nil {
		http.Error(w, fmt.Sprintf(`{"error":"%s"}`, err.Error()), http.StatusInternalServerError)
		return
	}

	output := result.Response.Content.Text()

	// Persist conversation history
	s.sessions.AppendMessages(sessionId, fantasy.NewUserMessage(req.Prompt))
	for _, step := range result.Steps {
		s.sessions.AppendMessages(sessionId, step.Messages...)
	}

	// Auto-title if this is the first message (AI-generated, fire-and-forget)
	if sess.MessageCount == 0 {
		s.generateTitle(sessionId, req.Prompt)
	}

	s.mu.Lock()
	s.activeModel = usedModel
	s.totalSteps += len(result.Steps)
	s.mu.Unlock()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(promptResponse{Output: output, Model: usedModel})
}

// ── Session-scoped streaming prompt with full FEP ──

func (s *daemonServer) handleSessionPromptStream(w http.ResponseWriter, r *http.Request) {
	sessionId := r.PathValue("id")
	_, ok := s.sessions.Get(sessionId)
	if !ok {
		http.Error(w, `{"error":"session not found"}`, http.StatusNotFound)
		return
	}

	var req promptRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.Prompt == "" {
		http.Error(w, `{"error":"prompt is required"}`, http.StatusBadRequest)
		return
	}

	sc := s.getOrCreateSessionCtx(sessionId)
	sc.mu.Lock()
	if sc.busy {
		sc.mu.Unlock()
		http.Error(w, `{"error":"session is busy"}`, http.StatusTooManyRequests)
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

	emit := newFEPEmitter(w)

	// Store the emitter and sessionId on the session context so permission/question
	// gates can emit FEP events through the active SSE stream.
	sc.mu.Lock()
	sc.emitter = emit
	sc.sessionId = sessionId
	sc.mu.Unlock()
	defer func() {
		sc.mu.Lock()
		sc.emitter = nil
		sc.mu.Unlock()
	}()

	// Inject sessionId into Go context so tools (permission gate, question tool) can read it.
	ctx = context.WithValue(ctx, sessionIdContextKey{}, sessionId)

	// Get conversation history for this session
	messages := s.sessions.GetMessages(sessionId)

	// Step counter (shared across callbacks)
	var stepCount int
	var stepMu sync.Mutex

	// Emit agent start
	emit.emitAgentStart(sessionId, req.Prompt)

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
			stepMu.Unlock()

			emit.emitStepStart(stepNumber, sessionId)

			// Inject steer message if pending
			if msg := sc.popSteerMessage(); msg != "" {
				s.sessions.AppendMessages(sessionId, fantasy.NewUserMessage("[STEER] "+msg))
				slog.Info("steer message injected", "session", sessionId, "message", truncate(msg, 100))
			}

			return nil
		},

		OnStepFinish: func(sr fantasy.StepResult) error {
			stepMu.Lock()
			currentStep := stepCount
			stepMu.Unlock()

			toolCallCount := len(sr.Content.ToolCalls())

			emit.emitStepFinish(currentStep, sessionId, sr.Usage, sr.FinishReason, toolCallCount)
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
			emit.emitAgentError(sessionId, err, isRetryableError(err))
		},
	})

	if err != nil {
		emit.emitAgentError(sessionId, err, isRetryableError(err))
	} else {
		// Persist conversation history
		s.sessions.AppendMessages(sessionId, fantasy.NewUserMessage(req.Prompt))
		for _, step := range result.Steps {
			s.sessions.AppendMessages(sessionId, step.Messages...)
		}

		// Auto-title from first prompt (AI-generated, fire-and-forget)
		sess, ok := s.sessions.Get(sessionId)
		if ok && sess.MessageCount <= 2 {
			s.generateTitle(sessionId, req.Prompt)
		}

		stepMu.Lock()
		finalSteps := stepCount
		stepMu.Unlock()

		s.mu.Lock()
		s.activeModel = usedModel
		s.totalSteps += finalSteps
		s.mu.Unlock()

		// Emit agent finish with total usage
		emit.emitAgentFinish(sessionId, result.TotalUsage, finalSteps, usedModel)
	}

	// Emit idle
	emit.emitSessionIdle(sessionId)
}

// ── Steer handler ──

func (s *daemonServer) handleSessionSteer(w http.ResponseWriter, r *http.Request) {
	sessionId := r.PathValue("id")
	if _, ok := s.sessions.Get(sessionId); !ok {
		http.Error(w, `{"error":"session not found"}`, http.StatusNotFound)
		return
	}

	var req steerRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.Message == "" {
		http.Error(w, `{"error":"message is required"}`, http.StatusBadRequest)
		return
	}

	sc := s.getOrCreateSessionCtx(sessionId)
	sc.mu.Lock()
	sc.steerMsg = req.Message
	sc.mu.Unlock()

	slog.Info("steer message queued", "session", sessionId, "message", truncate(req.Message, 100))

	w.Header().Set("Content-Type", "application/json")
	io.WriteString(w, `{"ok":true}`)
}

// ── Abort handler (per-session) ──

func (s *daemonServer) handleSessionAbort(w http.ResponseWriter, r *http.Request) {
	sessionId := r.PathValue("id")

	sc := s.getOrCreateSessionCtx(sessionId)
	sc.mu.Lock()
	if sc.cancel != nil {
		sc.cancel()
	}
	sc.busy = false
	sc.mu.Unlock()

	slog.Info("session aborted", "session", sessionId)

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

// ── Legacy endpoints (backward compat — create/use default session) ──

const legacySessionID = "__legacy__"

func (s *daemonServer) ensureLegacySession() {
	if _, ok := s.sessions.Get(legacySessionID); !ok {
		// Manually create since we need a fixed ID
		now := time.Now()
		sess := &Session{
			ID:        legacySessionID,
			Title:     "Legacy Session",
			Messages:  []fantasy.Message{},
			CreatedAt: now,
			UpdatedAt: now,
		}
		s.sessions.mu.Lock()
		s.sessions.sessions[legacySessionID] = sess
		s.sessions.persist(sess)
		s.sessions.mu.Unlock()
	}
}

func (s *daemonServer) handlePromptLegacy(w http.ResponseWriter, r *http.Request) {
	s.ensureLegacySession()
	// Rewrite the path value so handleSessionPrompt can read it
	r.SetPathValue("id", legacySessionID)
	s.handleSessionPrompt(w, r)
}

func (s *daemonServer) handlePromptStreamLegacy(w http.ResponseWriter, r *http.Request) {
	s.ensureLegacySession()
	r.SetPathValue("id", legacySessionID)
	s.handleSessionPromptStream(w, r)
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
	s.mu.Unlock()

	// Count busy sessions
	busySessions := 0
	s.mu.Lock()
	for _, sc := range s.sessionCtx {
		sc.mu.Lock()
		if sc.busy {
			busySessions++
		}
		sc.mu.Unlock()
	}
	s.mu.Unlock()

	sessionCount := len(s.sessions.List())

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"model":         model,
		"total_steps":   steps,
		"sessions":      sessionCount,
		"busy_sessions": busySessions,
	})
}

// ====================================================================
// Task mode: one-shot execution
// ====================================================================

type taskResult struct {
	Output  string `json:"output"`
	Steps   int    `json:"steps"`
	Model   string `json:"model"`
	Success bool   `json:"success"`
	Error   string `json:"error,omitempty"`
}

func runTask() error {
	ctx := context.Background()

	prompt := os.Getenv("AGENT_PROMPT")
	if prompt == "" {
		result := taskResult{Success: false, Error: "AGENT_PROMPT environment variable is not set"}
		json.NewEncoder(os.Stdout).Encode(result)
		return fmt.Errorf("AGENT_PROMPT not set")
	}

	cfg, err := loadConfig()
	if err != nil {
		result := taskResult{Success: false, Error: err.Error()}
		json.NewEncoder(os.Stdout).Encode(result)
		return err
	}

	slog.Info("running Fantasy task agent",
		"model", cfg.PrimaryModel,
		"prompt", truncate(prompt, 100),
	)

	bundle, err := buildAgentBundle(ctx, cfg)
	if err != nil {
		result := taskResult{Success: false, Error: err.Error(), Model: cfg.PrimaryModel}
		json.NewEncoder(os.Stdout).Encode(result)
		return err
	}
	defer shutdownMCPConnections(bundle.mcpConns)

	agentResult, usedModel, err := generateWithFallback(ctx, cfg, bundle, fantasy.AgentCall{Prompt: prompt})
	if err != nil {
		result := taskResult{Success: false, Error: err.Error(), Model: cfg.PrimaryModel}
		json.NewEncoder(os.Stdout).Encode(result)
		return err
	}

	output := agentResult.Response.Content.Text()
	result := taskResult{
		Output:  output,
		Steps:   len(agentResult.Steps),
		Model:   usedModel,
		Success: true,
	}
	json.NewEncoder(os.Stdout).Encode(result)
	return nil
}

// ====================================================================
// Orchestration tools (run_agent, get_agent_run)
// ====================================================================

type runAgentInput struct {
	Agent  string `json:"agent" description:"Agent name to run"`
	Prompt string `json:"prompt" description:"Prompt to send to the agent"`
}

func newRunAgentTool(k8s *K8sClient) fantasy.AgentTool {
	return fantasy.NewAgentTool("run_agent",
		"Trigger another agent with a prompt. Creates an AgentRun CR tracked by the operator.",
		func(ctx context.Context, input runAgentInput, _ fantasy.ToolCall) (fantasy.ToolResponse, error) {
			if input.Agent == "" || input.Prompt == "" {
				return fantasy.NewTextErrorResponse("agent and prompt are required"), nil
			}
			agentName := os.Getenv("AGENT_NAME")
			if agentName == "" {
				agentName = "unknown"
			}
			run, err := k8s.CreateAgentRun(ctx, input.Agent, input.Prompt, "agent", agentName)
			if err != nil {
				return fantasy.NewTextErrorResponse(fmt.Sprintf("Failed to create AgentRun: %s", err)), nil
			}

			resp := fantasy.NewTextResponse(fmt.Sprintf("AgentRun %s created for agent %q", run.Name, input.Agent))
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
		"Check the status and output of an AgentRun.",
		func(ctx context.Context, input getAgentRunInput, _ fantasy.ToolCall) (fantasy.ToolResponse, error) {
			if input.Name == "" {
				return fantasy.NewTextErrorResponse("name is required"), nil
			}
			status, err := k8s.GetAgentRun(ctx, input.Name)
			if err != nil {
				return fantasy.NewTextErrorResponse(fmt.Sprintf("Failed to get AgentRun: %s", err)), nil
			}

			resp := fantasy.NewTextResponse(fmt.Sprintf("Phase: %s\nOutput: %s", status.Phase, status.Output))
			resp = fantasy.WithResponseMetadata(resp, map[string]any{
				"ui":     "agent-run-status",
				"name":   input.Name,
				"phase":  status.Phase,
				"output": status.Output,
			})
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
