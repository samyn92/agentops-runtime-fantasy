/*
Agent Runtime — Fantasy (Go)

Memory system: three-layer architecture.

 1. Working Memory — unbounded list of fantasy.Message in Go memory.
    Trimmed by token budget (TrimToTokenBudget) before each turn.
    Ephemeral: lost on pod restart. This is what gets passed to the Fantasy SDK.

 2. Short-term Memory — session summaries stored in agentops-memory (auto-managed).
    Fetched via HTTP before each turn and prepended as context.

 3. Long-term Memory — explicit observations in agentops-memory (user/agent-managed).
    Same memory service instance, searched on demand.

agentops-memory is accessed via its REST API (not MCP). The runtime is a thin HTTP client.
*/
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"strconv"
	"sync"
	"time"

	"charm.land/fantasy"
	"github.com/google/uuid"
	"go.opentelemetry.io/otel/attribute"
)

// ====================================================================
// Working Memory — token-budget-managed message list
// ====================================================================

// WorkingMemory holds conversation messages, trimmed by token budget before each turn.
// Thread-safe. Ephemeral — lost on pod restart by design.
type WorkingMemory struct {
	mu       sync.RWMutex
	messages []fantasy.Message
	turnNum  int // number of completed turns (user prompt + assistant response)

	// sessionID is the memory service session ID. Set externally by the daemon
	// server so SaveCheckpoint can persist it for crash recovery.
	sessionID string

	// toolMeta maps toolCallID → ClientMetadata JSON string.
	// Populated from OnToolResult callbacks; persisted in checkpoints.
	// Used to enrich tool-result parts during serialization since
	// Fantasy's ToolResultPart doesn't carry ClientMetadata.
	toolMeta map[string]string
}

// NewWorkingMemory creates an empty working memory.
func NewWorkingMemory() *WorkingMemory {
	return &WorkingMemory{
		messages: make([]fantasy.Message, 0, 32),
		toolMeta: make(map[string]string),
	}
}

// Messages returns a copy of the current message window.
func (wm *WorkingMemory) Messages() []fantasy.Message {
	wm.mu.RLock()
	defer wm.mu.RUnlock()
	out := make([]fantasy.Message, len(wm.messages))
	copy(out, wm.messages)
	return out
}

// StoreToolMeta records the ClientMetadata JSON for a tool call ID.
// Called from OnToolResult so it can be re-attached during serialization.
func (wm *WorkingMemory) StoreToolMeta(toolCallID, metadata string) {
	if toolCallID == "" || metadata == "" {
		return
	}
	wm.mu.Lock()
	defer wm.mu.Unlock()
	wm.toolMeta[toolCallID] = metadata
}

// ToolMeta returns a snapshot of the tool metadata map.
func (wm *WorkingMemory) ToolMeta() map[string]string {
	wm.mu.RLock()
	defer wm.mu.RUnlock()
	out := make(map[string]string, len(wm.toolMeta))
	for k, v := range wm.toolMeta {
		out[k] = v
	}
	return out
}

// Append adds messages to the working memory.
// No size limit — trimming is handled by TrimToTokenBudget before each turn.
func (wm *WorkingMemory) Append(msgs ...fantasy.Message) {
	wm.mu.Lock()
	defer wm.mu.Unlock()
	wm.messages = append(wm.messages, msgs...)
}

// CompleteTurn increments the turn counter. Called after each prompt/response cycle.
func (wm *WorkingMemory) CompleteTurn() {
	wm.mu.Lock()
	defer wm.mu.Unlock()
	wm.turnNum++
}

// TurnCount returns the number of completed turns.
func (wm *WorkingMemory) TurnCount() int {
	wm.mu.RLock()
	defer wm.mu.RUnlock()
	return wm.turnNum
}

// MessageCount returns the current number of messages in the window.
func (wm *WorkingMemory) MessageCount() int {
	wm.mu.RLock()
	defer wm.mu.RUnlock()
	return len(wm.messages)
}

// Clear drops all messages and resets the turn counter.

// TrimToTokenBudget removes messages from the front of the working memory
// until the estimated token count fits within the given budget.
// Uses safe-boundary logic (never orphans tool results).
//
// This is the primary mechanism for keeping conversation history within
// the model's context window. Called before each turn with the available
// conversation budget from the pre-flight ContextBudget.
//
// Returns the number of messages trimmed and the estimated token count after trimming.
func (wm *WorkingMemory) TrimToTokenBudget(budgetTokens int64) (trimmed int, estimatedTokens int64) {
	wm.mu.Lock()
	defer wm.mu.Unlock()

	if len(wm.messages) == 0 {
		return 0, 0
	}

	estimatedTokens = EstimateMessageTokens(wm.messages)
	if estimatedTokens <= budgetTokens {
		return 0, estimatedTokens
	}

	originalLen := len(wm.messages)

	// Trim from front until we fit, always at safe boundaries.
	for estimatedTokens > budgetTokens && len(wm.messages) > 1 {
		// Find the next safe trim point (user or assistant message start)
		trimAt := 1 // default: remove first message
		for i := 1; i < len(wm.messages); i++ {
			role := wm.messages[i].Role
			if role == fantasy.MessageRoleUser || role == fantasy.MessageRoleAssistant {
				trimAt = i
				break
			}
		}

		wm.messages = wm.messages[trimAt:]
		estimatedTokens = EstimateMessageTokens(wm.messages)
	}

	trimmed = originalLen - len(wm.messages)
	if trimmed > 0 {
		slog.Info("working memory trimmed by token budget",
			"trimmed_messages", trimmed,
			"remaining_messages", len(wm.messages),
			"estimated_tokens", estimatedTokens,
			"budget_tokens", budgetTokens,
		)
	}

	return trimmed, estimatedTokens
}

func (wm *WorkingMemory) Clear() {
	wm.mu.Lock()
	defer wm.mu.Unlock()
	wm.messages = wm.messages[:0]
	wm.turnNum = 0
}

// SetSessionID stores the memory service session ID for checkpoint persistence.
func (wm *WorkingMemory) SetSessionID(id string) {
	wm.mu.Lock()
	defer wm.mu.Unlock()
	wm.sessionID = id
}

// SessionID returns the memory service session ID (may be empty).
func (wm *WorkingMemory) SessionID() string {
	wm.mu.RLock()
	defer wm.mu.RUnlock()
	return wm.sessionID
}

// IsBusy and related state are tracked on daemonServer, not here.

// ====================================================================
// Engram Client — HTTP REST client for the shared memory server
// ====================================================================

// EngramClient talks to the Engram REST API for persistent memory operations.
type EngramClient struct {
	baseURL   string // e.g. "http://engram.agents.svc.cluster.local:7437"
	project   string // scoped to this agent (defaults to agent name)
	sessionID string // Engram session ID for this runtime lifecycle
	client    *http.Client
}

// NewEngramClient creates a client. Returns nil if serverURL is empty (memory disabled).
func NewEngramClient(serverURL, project string) *EngramClient {
	if serverURL == "" {
		return nil
	}
	return &EngramClient{
		baseURL:   serverURL,
		project:   project,
		sessionID: uuid.NewString(),
		client: &http.Client{
			Timeout: 10 * time.Second,
		},
	}
}

// SetSessionID overrides the session ID. Used to restore a session from a
// checkpoint so that pre-crash and post-crash memory share the same session.
func (ec *EngramClient) SetSessionID(id string) {
	if ec != nil && id != "" {
		ec.sessionID = id
	}
}

// SessionID returns the current session ID.
func (ec *EngramClient) SessionID() string {
	if ec == nil {
		return ""
	}
	return ec.sessionID
}

// Init creates a session in Engram for this runtime lifecycle.
// Called once at daemon startup.
func (ec *EngramClient) Init() error {
	if ec == nil {
		return nil
	}

	body := map[string]string{
		"id":      ec.sessionID,
		"project": ec.project,
	}

	_, err := ec.post("/sessions", body)
	if err != nil {
		return fmt.Errorf("memory init session: %w", err)
	}

	slog.Info("memory session created", "session_id", ec.sessionID, "project", ec.project)
	return nil
}

// FetchContext retrieves recent context from the memory service (short-term + long-term memories).
// When a query is provided, the memory service uses FTS5 BM25 for relevance-ranked retrieval.
// Returns a formatted string suitable for prepending to the system context.
// Returns empty string on error or if no context is available.
func (ec *EngramClient) FetchContext(ctx context.Context, limit int, query string) string {
	if ec == nil {
		return ""
	}

	ctx, span := tracer.Start(ctx, "memory.fetch_context")
	defer span.End()
	span.SetAttributes(
		attrMemoryOp.String("fetch_context"),
		attrMemoryProject.String(ec.project),
	)

	params := url.Values{
		"project": {ec.project},
		"limit":   {strconv.Itoa(limit)},
	}
	// Pass user prompt for relevance-ranked retrieval (FTS5 BM25).
	if query != "" {
		// Truncate to 500 chars — captures intent without sending full payloads.
		if len(query) > 500 {
			query = query[:500]
		}
		params.Set("query", query)
		span.SetAttributes(attribute.String("memory.context_query", query))
	}

	resp, err := ec.get("/context", params)
	if err != nil {
		slog.Warn("memory fetch context failed", "error", err)
		recordError(span, err)
		return ""
	}

	// The /context endpoint returns a structured response with recent observations
	// and session summaries. Parse and format for injection.
	var contextResp struct {
		RecentObservations []struct {
			Type    string `json:"type"`
			Title   string `json:"title"`
			Content string `json:"content"`
		} `json:"recent_observations"`
		RecentSessions []struct {
			Summary string `json:"summary"`
		} `json:"recent_sessions"`
	}

	if err := json.Unmarshal(resp, &contextResp); err != nil {
		slog.Warn("memory context parse failed", "error", err)
		return ""
	}

	var buf bytes.Buffer
	hasContent := false

	if len(contextResp.RecentSessions) > 0 {
		buf.WriteString("<memory:sessions>\n")
		for _, s := range contextResp.RecentSessions {
			if s.Summary != "" {
				buf.WriteString("- ")
				buf.WriteString(s.Summary)
				buf.WriteString("\n")
				hasContent = true
			}
		}
		buf.WriteString("</memory:sessions>\n")
	}

	if len(contextResp.RecentObservations) > 0 {
		buf.WriteString("<memory:context>\n")
		for _, o := range contextResp.RecentObservations {
			if o.Content != "" {
				buf.WriteString("- [")
				buf.WriteString(o.Type)
				buf.WriteString("] ")
				buf.WriteString(o.Title)
				buf.WriteString(": ")
				buf.WriteString(o.Content)
				buf.WriteString("\n")
				hasContent = true
			}
		}
		buf.WriteString("</memory:context>\n")
	}

	if !hasContent {
		return ""
	}
	return buf.String()
}

// SaveObservation explicitly saves an observation (long-term memory).
// Types: "decision", "discovery", "bugfix", "lesson", "procedure"
func (ec *EngramClient) SaveObservation(ctx context.Context, obsType, title, content string, tags []string) error {
	if ec == nil {
		return nil
	}

	ctx, span := tracer.Start(ctx, "memory.save_observation")
	defer span.End()
	span.SetAttributes(
		attrMemoryOp.String("save_observation"),
		attrMemoryProject.String(ec.project),
	)

	body := map[string]any{
		"session_id": ec.sessionID,
		"type":       obsType,
		"title":      title,
		"content":    content,
		"project":    ec.project,
	}
	if len(tags) > 0 {
		body["tags"] = tags
	}

	_, err := ec.post("/observations", body)
	if err != nil {
		recordError(span, err)
		return fmt.Errorf("memory save observation: %w", err)
	}

	slog.Info("memory observation saved", "type", obsType, "title", title)
	return nil
}

// EndSession ends the memory session with an optional conversation transcript.
// Engram is responsible for summarization — the runtime sends raw messages.
// Called on daemon shutdown and task completion.
func (ec *EngramClient) EndSession(messages []EngramSessionMessage) {
	if ec == nil {
		return
	}

	body := map[string]any{}
	if len(messages) > 0 {
		body["messages"] = messages
	}

	_, err := ec.post("/sessions/"+ec.sessionID+"/end", body)
	if err != nil {
		slog.Warn("memory end session failed", "error", err)
	} else {
		slog.Info("memory session ended", "session_id", ec.sessionID)
	}
}

// EngramSessionMessage is a minimal message representation sent to Engram
// for session persistence and server-side summarization.
type EngramSessionMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// fantasyToEngramMessages converts Fantasy SDK messages to the minimal
// Engram session message format for session persistence.
func fantasyToEngramMessages(messages []fantasy.Message) []EngramSessionMessage {
	var result []EngramSessionMessage
	for _, msg := range messages {
		var text string
		for _, part := range msg.Content {
			switch p := part.(type) {
			case fantasy.TextPart:
				if p.Text != "" {
					text += p.Text
				}
			case fantasy.ToolCallPart:
				text += fmt.Sprintf("[tool_call: %s]", p.ToolName)
			}
		}
		if text != "" {
			// Cap individual messages to avoid sending enormous payloads
			if len(text) > 5000 {
				text = text[:5000] + "..."
			}
			result = append(result, EngramSessionMessage{
				Role:    string(msg.Role),
				Content: text,
			})
		}
	}
	return result
}

// Search performs a full-text search across memories.
func (ec *EngramClient) Search(ctx context.Context, query string, limit int) ([]EngramSearchResult, error) {
	if ec == nil {
		return nil, nil
	}

	ctx, span := tracer.Start(ctx, "memory.search")
	defer span.End()
	span.SetAttributes(
		attrMemoryOp.String("search"),
		attrMemoryProject.String(ec.project),
	)

	params := url.Values{
		"q":       {query},
		"project": {ec.project},
	}
	if limit > 0 {
		params.Set("limit", strconv.Itoa(limit))
	}

	resp, err := ec.get("/search", params)
	if err != nil {
		recordError(span, err)
		return nil, fmt.Errorf("memory search: %w", err)
	}

	var results []EngramSearchResult
	if err := json.Unmarshal(resp, &results); err != nil {
		return nil, fmt.Errorf("memory search parse: %w", err)
	}
	return results, nil
}

// EngramSearchResult represents a search hit from Engram.
type EngramSearchResult struct {
	ID      int     `json:"id"`
	Type    string  `json:"type"`
	Title   string  `json:"title"`
	Content string  `json:"content"`
	Rank    float64 `json:"rank"`
}

// ── HTTP helpers ──

func (ec *EngramClient) get(path string, params url.Values) ([]byte, error) {
	u := ec.baseURL + path
	if len(params) > 0 {
		u += "?" + params.Encode()
	}

	resp, err := ec.client.Get(u)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(body))
	}
	return body, nil
}

func (ec *EngramClient) post(path string, payload any) ([]byte, error) {
	data, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}

	resp, err := ec.client.Post(ec.baseURL+path, "application/json", bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(body))
	}
	return body, nil
}

// ====================================================================
// Memory Tools — Fantasy AgentTools that let the agent interact with Engram
// ====================================================================

// buildMemoryTools returns Fantasy AgentTools for memory operations.
// Returns nil if engram is nil (memory disabled).
func buildMemoryTools(engram *EngramClient) []fantasy.AgentTool {
	if engram == nil {
		return nil
	}
	return []fantasy.AgentTool{
		newMemSaveTool(engram),
		newMemSearchTool(engram),
		newMemContextTool(engram),
	}
}

// ── mem_save ──

type memSaveInput struct {
	Type    string   `json:"type" description:"Observation type: decision, discovery, bugfix, pattern, architecture, config, learning, preference"`
	Title   string   `json:"title" description:"Brief title summarizing what was learned or decided"`
	Content string   `json:"content" description:"Detailed content — what was learned, why it matters, how to apply it"`
	Tags    []string `json:"tags,omitempty" description:"Optional tags for categorization"`
}

func newMemSaveTool(engram *EngramClient) fantasy.AgentTool {
	return fantasy.NewAgentTool("mem_save",
		"Save an observation to long-term memory (Engram). Use this proactively to remember important decisions, discoveries, bugfixes, patterns, and lessons learned. These memories persist across conversations and restarts.",
		func(ctx context.Context, input memSaveInput, _ fantasy.ToolCall) (fantasy.ToolResponse, error) {
			if input.Title == "" || input.Content == "" {
				return fantasy.NewTextErrorResponse("title and content are required"), nil
			}
			if input.Type == "" {
				input.Type = "discovery"
			}

			err := engram.SaveObservation(ctx, input.Type, input.Title, input.Content, input.Tags)
			if err != nil {
				return fantasy.NewTextErrorResponse(fmt.Sprintf("failed to save memory: %s", err)), nil
			}

			return fantasy.NewTextResponse(fmt.Sprintf("Saved [%s] %s", input.Type, input.Title)), nil
		})
}

// ── mem_search ──

type memSearchInput struct {
	Query string `json:"query" description:"Full-text search query across all memories"`
	Limit int    `json:"limit,omitempty" description:"Max results to return (default: 10)"`
}

func newMemSearchTool(engram *EngramClient) fantasy.AgentTool {
	return fantasy.NewAgentTool("mem_search",
		"Search long-term memory (Engram) using full-text search. Returns matching observations ranked by relevance. Use this to recall past decisions, discoveries, bugfixes, and lessons.",
		func(ctx context.Context, input memSearchInput, _ fantasy.ToolCall) (fantasy.ToolResponse, error) {
			if input.Query == "" {
				return fantasy.NewTextErrorResponse("query is required"), nil
			}
			limit := input.Limit
			if limit <= 0 {
				limit = 10
			}

			results, err := engram.Search(ctx, input.Query, limit)
			if err != nil {
				return fantasy.NewTextErrorResponse(fmt.Sprintf("search failed: %s", err)), nil
			}

			if len(results) == 0 {
				return fantasy.NewTextResponse("No memories found matching the query."), nil
			}

			var buf bytes.Buffer
			fmt.Fprintf(&buf, "Found %d memories:\n\n", len(results))
			for i, r := range results {
				fmt.Fprintf(&buf, "%d. [%s] %s (rank: %.1f)\n   %s\n\n",
					i+1, r.Type, r.Title, r.Rank, r.Content)
			}

			return fantasy.NewTextResponse(buf.String()), nil
		})
}

// ── mem_context ──

type memContextInput struct {
	Limit int `json:"limit,omitempty" description:"Number of recent memory items to retrieve (default: 5)"`
}

func newMemContextTool(engram *EngramClient) fantasy.AgentTool {
	return fantasy.NewAgentTool("mem_context",
		"Retrieve recent memory context — recent observations and session summaries. Use this to refresh your knowledge about what happened in previous sessions.",
		func(ctx context.Context, input memContextInput, _ fantasy.ToolCall) (fantasy.ToolResponse, error) {
			limit := input.Limit
			if limit <= 0 {
				limit = 5
			}

			memCtx := engram.FetchContext(ctx, limit, "")
			if memCtx == "" {
				return fantasy.NewTextResponse("No recent memory context available."), nil
			}

			return fantasy.NewTextResponse(memCtx), nil
		})
}
