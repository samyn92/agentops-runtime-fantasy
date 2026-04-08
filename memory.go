/*
Agent Runtime — Fantasy (Go)

Memory system: replaces session.go with a three-layer architecture.

 1. Working Memory — fixed-size sliding window of fantasy.Message in Go memory.
    Ephemeral: lost on pod restart. This is what gets passed to the Fantasy SDK.

 2. Short-term Memory — session summaries stored in Engram (auto-managed).
    Fetched via HTTP before each turn and prepended as context.

 3. Long-term Memory — explicit observations in Engram (user/agent-managed).
    Same Engram instance, searched on demand.

Engram is accessed via its REST API (not MCP). The runtime is a thin HTTP client.
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
)

// ====================================================================
// Working Memory — bounded sliding window of messages
// ====================================================================

// WorkingMemory holds the last N messages in a sliding window.
// Thread-safe. Ephemeral — lost on pod restart by design.
type WorkingMemory struct {
	mu       sync.RWMutex
	messages []fantasy.Message
	maxSize  int
	turnNum  int // number of completed turns (user prompt + assistant response)
}

// NewWorkingMemory creates a working memory with the given window size.
func NewWorkingMemory(windowSize int) *WorkingMemory {
	if windowSize < 2 {
		windowSize = 20
	}
	return &WorkingMemory{
		messages: make([]fantasy.Message, 0, windowSize),
		maxSize:  windowSize,
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

// Append adds messages to the window, dropping the oldest if the window is full.
// Messages are trimmed from the front to stay within maxSize. We trim at clean
// boundaries to avoid orphaning tool messages (assistant tool_use must stay
// paired with the following tool result).
//
// A clean boundary is a user message that is NOT a tool-result message
// (role == "user", not role == "tool"). We scan forward from the excess point
// to find such a boundary.
func (wm *WorkingMemory) Append(msgs ...fantasy.Message) {
	wm.mu.Lock()
	defer wm.mu.Unlock()

	wm.messages = append(wm.messages, msgs...)

	// Trim from front if over capacity
	if len(wm.messages) > wm.maxSize {
		excess := len(wm.messages) - wm.maxSize

		// Find a safe trim point: must be a user message (not tool/assistant).
		// Scan forward from excess to find one. This ensures we never leave an
		// orphaned tool-result at the start (which requires its preceding
		// assistant tool_use message).
		trimAt := excess
		for i := excess; i < len(wm.messages); i++ {
			if wm.messages[i].Role == fantasy.MessageRoleUser {
				trimAt = i
				break
			}
			// If we hit another assistant message, that's also safe — it starts
			// a new exchange. But prefer user messages.
			if wm.messages[i].Role == fantasy.MessageRoleAssistant && i > excess {
				trimAt = i
				break
			}
		}

		// Safety: if trimAt would leave us with 0 messages, just keep the last maxSize
		if trimAt >= len(wm.messages) {
			trimAt = len(wm.messages) - wm.maxSize
			if trimAt < 0 {
				trimAt = 0
			}
		}

		wm.messages = wm.messages[trimAt:]
	}
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

// WindowSize returns the maximum sliding window capacity.
func (wm *WorkingMemory) WindowSize() int {
	wm.mu.RLock()
	defer wm.mu.RUnlock()
	return wm.maxSize
}

// SetWindowSize changes the sliding window capacity at runtime.
// If the new size is smaller than the current message count, excess messages
// are trimmed from the front (oldest first) at the next Append call.
// Minimum size is 2.
func (wm *WorkingMemory) SetWindowSize(size int) {
	if size < 2 {
		size = 2
	}
	wm.mu.Lock()
	defer wm.mu.Unlock()
	wm.maxSize = size
	// Eagerly trim if currently over the new limit
	if len(wm.messages) > wm.maxSize {
		excess := len(wm.messages) - wm.maxSize
		trimAt := excess
		for i := excess; i < len(wm.messages); i++ {
			if wm.messages[i].Role == fantasy.MessageRoleUser {
				trimAt = i
				break
			}
			if wm.messages[i].Role == fantasy.MessageRoleAssistant && i > excess {
				trimAt = i
				break
			}
		}
		if trimAt >= len(wm.messages) {
			trimAt = len(wm.messages) - wm.maxSize
			if trimAt < 0 {
				trimAt = 0
			}
		}
		wm.messages = wm.messages[trimAt:]
	}
}

// Clear drops all messages and resets the turn counter.
func (wm *WorkingMemory) Clear() {
	wm.mu.Lock()
	defer wm.mu.Unlock()
	wm.messages = wm.messages[:0]
	wm.turnNum = 0
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
		return fmt.Errorf("engram init session: %w", err)
	}

	slog.Info("engram session created", "session_id", ec.sessionID, "project", ec.project)
	return nil
}

// FetchContext retrieves recent context from Engram (short-term + long-term memories).
// Returns a formatted string suitable for prepending to the system context.
// Returns empty string on error or if no context is available.
func (ec *EngramClient) FetchContext(limit int) string {
	if ec == nil {
		return ""
	}

	params := url.Values{
		"project": {ec.project},
		"limit":   {strconv.Itoa(limit)},
	}

	resp, err := ec.get("/context", params)
	if err != nil {
		slog.Warn("engram fetch context failed", "error", err)
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
		slog.Warn("engram context parse failed", "error", err)
		return ""
	}

	var buf bytes.Buffer
	hasContent := false

	if len(contextResp.RecentSessions) > 0 {
		buf.WriteString("[Previous Session Context]\n")
		for _, s := range contextResp.RecentSessions {
			if s.Summary != "" {
				buf.WriteString("- ")
				buf.WriteString(s.Summary)
				buf.WriteString("\n")
				hasContent = true
			}
		}
		buf.WriteString("\n")
	}

	if len(contextResp.RecentObservations) > 0 {
		buf.WriteString("[Memory — Relevant Knowledge]\n")
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
		buf.WriteString("\n")
	}

	if !hasContent {
		return ""
	}
	return buf.String()
}

// SaveObservation explicitly saves an observation (long-term memory).
// Types: "decision", "discovery", "bugfix", "lesson", "procedure"
func (ec *EngramClient) SaveObservation(obsType, title, content string, tags []string) error {
	if ec == nil {
		return nil
	}

	body := map[string]any{
		"session_id": ec.sessionID,
		"type":       obsType,
		"title":      title,
		"content":    content,
	}
	if len(tags) > 0 {
		body["tags"] = tags
	}

	_, err := ec.post("/observations", body)
	if err != nil {
		return fmt.Errorf("engram save observation: %w", err)
	}

	slog.Info("engram observation saved", "type", obsType, "title", title)
	return nil
}

// PassiveCapture sends assistant output for Engram's passive extraction.
// Engram auto-detects noteworthy content (decisions, discoveries, etc.)
// and stores it. Fire-and-forget — errors are logged but not propagated.
func (ec *EngramClient) PassiveCapture(assistantOutput string) {
	if ec == nil || assistantOutput == "" {
		return
	}

	body := map[string]string{
		"session_id": ec.sessionID,
		"content":    assistantOutput,
	}

	go func() {
		_, err := ec.post("/observations/passive", body)
		if err != nil {
			slog.Debug("engram passive capture failed", "error", err)
		}
	}()
}

// EndSession ends the Engram session with an optional summary.
// Called on daemon shutdown.
func (ec *EngramClient) EndSession(summary string) {
	if ec == nil {
		return
	}

	body := map[string]string{}
	if summary != "" {
		body["summary"] = summary
	}

	_, err := ec.post("/sessions/"+ec.sessionID+"/end", body)
	if err != nil {
		slog.Warn("engram end session failed", "error", err)
	} else {
		slog.Info("engram session ended", "session_id", ec.sessionID)
	}
}

// Search performs a full-text search across memories.
func (ec *EngramClient) Search(query string, limit int) ([]EngramSearchResult, error) {
	if ec == nil {
		return nil, nil
	}

	params := url.Values{
		"q":       {query},
		"project": {ec.project},
	}
	if limit > 0 {
		params.Set("limit", strconv.Itoa(limit))
	}

	resp, err := ec.get("/search", params)
	if err != nil {
		return nil, fmt.Errorf("engram search: %w", err)
	}

	var results []EngramSearchResult
	if err := json.Unmarshal(resp, &results); err != nil {
		return nil, fmt.Errorf("engram search parse: %w", err)
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
		func(_ context.Context, input memSaveInput, _ fantasy.ToolCall) (fantasy.ToolResponse, error) {
			if input.Title == "" || input.Content == "" {
				return fantasy.NewTextErrorResponse("title and content are required"), nil
			}
			if input.Type == "" {
				input.Type = "discovery"
			}

			err := engram.SaveObservation(input.Type, input.Title, input.Content, input.Tags)
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
		func(_ context.Context, input memSearchInput, _ fantasy.ToolCall) (fantasy.ToolResponse, error) {
			if input.Query == "" {
				return fantasy.NewTextErrorResponse("query is required"), nil
			}
			limit := input.Limit
			if limit <= 0 {
				limit = 10
			}

			results, err := engram.Search(input.Query, limit)
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
		"Retrieve recent memory context from Engram — recent observations and session summaries. Use this to refresh your knowledge about what happened in previous sessions.",
		func(_ context.Context, input memContextInput, _ fantasy.ToolCall) (fantasy.ToolResponse, error) {
			limit := input.Limit
			if limit <= 0 {
				limit = 5
			}

			ctx := engram.FetchContext(limit)
			if ctx == "" {
				return fantasy.NewTextResponse("No recent memory context available."), nil
			}

			return fantasy.NewTextResponse(ctx), nil
		})
}
