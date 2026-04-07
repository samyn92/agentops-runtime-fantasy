/*
Agent Runtime — Fantasy (Go)

Session management for the daemon server.
Tracks conversation sessions with message history so the console
can maintain multiple concurrent chat threads against a single agent.

Sessions are persisted as JSON files in the data directory (default /data/sessions/).
Each session is one file: {id}.json. The store loads all sessions from disk on startup
and writes back on every mutation. No PVC = ephemeral (acceptable for dev). PVC = persistent.
*/
package main

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"charm.land/fantasy"
	"github.com/google/uuid"
)

// Session represents a single conversation with the agent.
type Session struct {
	ID           string            `json:"id"`
	Title        string            `json:"title"`
	Messages     []fantasy.Message `json:"-"` // handled by custom serialization
	CreatedAt    time.Time         `json:"created_at"`
	UpdatedAt    time.Time         `json:"updated_at"`
	MessageCount int               `json:"message_count"`
	// Usage from the last agent call (persisted for the console to display on reload)
	TotalUsage *sessionUsage `json:"total_usage,omitempty"`
	Model      string        `json:"model,omitempty"`
	// Per-turn usage: one entry per agent call (user prompt -> agent response).
	// This allows the console to display token counts on each assistant message after reload.
	TurnUsages []turnUsage `json:"turn_usages,omitempty"`
}

// sessionUsage stores token counts from the last agent call.
type sessionUsage struct {
	InputTokens         int64 `json:"input_tokens"`
	OutputTokens        int64 `json:"output_tokens"`
	TotalTokens         int64 `json:"total_tokens"`
	ReasoningTokens     int64 `json:"reasoning_tokens"`
	CacheCreationTokens int64 `json:"cache_creation_tokens"`
	CacheReadTokens     int64 `json:"cache_read_tokens"`
}

// turnUsage stores token counts for a single agent call (one user prompt -> one agent response).
// StepUsages contains per-step breakdown so each assistant message can show its own token count.
type turnUsage struct {
	Usage      sessionUsage   `json:"usage"`
	Model      string         `json:"model"`
	Steps      int            `json:"steps"`
	StepUsages []sessionUsage `json:"step_usages,omitempty"`
}

// SessionInfo is the API-facing representation of a session (times as RFC3339).
type SessionInfo struct {
	ID           string        `json:"id"`
	Title        string        `json:"title"`
	CreatedAt    string        `json:"created_at"`
	UpdatedAt    string        `json:"updated_at"`
	MessageCount int           `json:"message_count"`
	TotalUsage   *sessionUsage `json:"total_usage,omitempty"`
	Model        string        `json:"model,omitempty"`
	TurnUsages   []turnUsage   `json:"turn_usages,omitempty"`
}

// Info returns an API-safe representation with RFC3339 timestamps.
func (s *Session) Info() SessionInfo {
	return SessionInfo{
		ID:           s.ID,
		Title:        s.Title,
		CreatedAt:    s.CreatedAt.Format(time.RFC3339),
		UpdatedAt:    s.UpdatedAt.Format(time.RFC3339),
		MessageCount: s.MessageCount,
		TotalUsage:   s.TotalUsage,
		Model:        s.Model,
		TurnUsages:   s.TurnUsages,
	}
}

// ── Serializable types for JSON persistence ──
// fantasy.MessagePart and fantasy.ToolResultOutputContent are interfaces,
// so we need concrete wrapper types for marshaling/unmarshaling.

type serializableSession struct {
	ID           string                `json:"id"`
	Title        string                `json:"title"`
	Messages     []serializableMessage `json:"messages"`
	CreatedAt    time.Time             `json:"created_at"`
	UpdatedAt    time.Time             `json:"updated_at"`
	MessageCount int                   `json:"message_count"`
	TotalUsage   *sessionUsage         `json:"total_usage,omitempty"`
	Model        string                `json:"model,omitempty"`
	TurnUsages   []turnUsage           `json:"turn_usages,omitempty"`
}

type serializableMessage struct {
	Role    string               `json:"role"`
	Content []serializablePartOK `json:"content"`
}

type serializablePartOK struct {
	Type string `json:"type"`

	// TextPart / ReasoningPart
	Text string `json:"text,omitempty"`

	// FilePart
	Filename  string `json:"filename,omitempty"`
	Data      []byte `json:"data,omitempty"`
	MediaType string `json:"media_type,omitempty"`

	// ToolCallPart
	ToolCallID       string `json:"tool_call_id,omitempty"`
	ToolName         string `json:"tool_name,omitempty"`
	Input            string `json:"input,omitempty"`
	ProviderExecuted bool   `json:"provider_executed,omitempty"`

	// ToolResultPart
	Output *serializableToolOutput `json:"output,omitempty"`
}

type serializableToolOutput struct {
	Type      string `json:"type"` // text, error, media
	Text      string `json:"text,omitempty"`
	Error     string `json:"error,omitempty"`
	Data      string `json:"data,omitempty"`
	MediaType string `json:"media_type,omitempty"`
}

func serializeMessage(msg fantasy.Message) serializableMessage {
	sm := serializableMessage{Role: string(msg.Role)}
	for _, part := range msg.Content {
		sp := serializablePartOK{}
		switch p := part.(type) {
		case fantasy.TextPart:
			sp.Type = "text"
			sp.Text = p.Text
		case fantasy.ReasoningPart:
			sp.Type = "reasoning"
			sp.Text = p.Text
		case fantasy.FilePart:
			sp.Type = "file"
			sp.Filename = p.Filename
			sp.Data = p.Data
			sp.MediaType = p.MediaType
		case fantasy.ToolCallPart:
			sp.Type = "tool-call"
			sp.ToolCallID = p.ToolCallID
			sp.ToolName = p.ToolName
			sp.Input = p.Input
			sp.ProviderExecuted = p.ProviderExecuted
		case fantasy.ToolResultPart:
			sp.Type = "tool-result"
			sp.ToolCallID = p.ToolCallID
			sp.ProviderExecuted = p.ProviderExecuted
			sp.Output = serializeToolOutput(p.Output)
		default:
			sp.Type = "unknown"
			sp.Text = fmt.Sprintf("%v", part)
		}
		sm.Content = append(sm.Content, sp)
	}
	return sm
}

func serializeToolOutput(out fantasy.ToolResultOutputContent) *serializableToolOutput {
	if out == nil {
		return nil
	}
	switch o := out.(type) {
	case fantasy.ToolResultOutputContentText:
		return &serializableToolOutput{Type: "text", Text: o.Text}
	case fantasy.ToolResultOutputContentError:
		errStr := ""
		if o.Error != nil {
			errStr = o.Error.Error()
		}
		return &serializableToolOutput{Type: "error", Error: errStr}
	case fantasy.ToolResultOutputContentMedia:
		return &serializableToolOutput{Type: "media", Data: o.Data, MediaType: o.MediaType, Text: o.Text}
	default:
		return &serializableToolOutput{Type: "text", Text: fmt.Sprintf("%v", out)}
	}
}

func deserializeMessage(sm serializableMessage) fantasy.Message {
	msg := fantasy.Message{Role: fantasy.MessageRole(sm.Role)}
	for _, sp := range sm.Content {
		var part fantasy.MessagePart
		switch sp.Type {
		case "text":
			part = fantasy.TextPart{Text: sp.Text}
		case "reasoning":
			part = fantasy.ReasoningPart{Text: sp.Text}
		case "file":
			part = fantasy.FilePart{Filename: sp.Filename, Data: sp.Data, MediaType: sp.MediaType}
		case "tool-call":
			part = fantasy.ToolCallPart{
				ToolCallID:       sp.ToolCallID,
				ToolName:         sp.ToolName,
				Input:            sp.Input,
				ProviderExecuted: sp.ProviderExecuted,
			}
		case "tool-result":
			part = fantasy.ToolResultPart{
				ToolCallID:       sp.ToolCallID,
				ProviderExecuted: sp.ProviderExecuted,
				Output:           deserializeToolOutput(sp.Output),
			}
		default:
			// Unknown type, store as text to avoid data loss
			part = fantasy.TextPart{Text: sp.Text}
		}
		msg.Content = append(msg.Content, part)
	}
	return msg
}

func deserializeToolOutput(so *serializableToolOutput) fantasy.ToolResultOutputContent {
	if so == nil {
		return fantasy.ToolResultOutputContentText{Text: ""}
	}
	switch so.Type {
	case "text":
		return fantasy.ToolResultOutputContentText{Text: so.Text}
	case "error":
		return fantasy.ToolResultOutputContentError{Error: fmt.Errorf("%s", so.Error)}
	case "media":
		return fantasy.ToolResultOutputContentMedia{Data: so.Data, MediaType: so.MediaType, Text: so.Text}
	default:
		return fantasy.ToolResultOutputContentText{Text: so.Text}
	}
}

// ── SessionStore — file-backed persistent store ──

// SessionStore manages agent sessions with file-backed persistence.
type SessionStore struct {
	sessions map[string]*Session
	dataDir  string // directory for JSON files (e.g. /data/sessions/)
	mu       sync.RWMutex
}

// NewSessionStore creates a session store backed by the given directory.
// If dir is empty, sessions are in-memory only (no persistence).
// On creation, it loads any existing session files from disk.
func NewSessionStore(dir string) *SessionStore {
	ss := &SessionStore{
		sessions: make(map[string]*Session),
		dataDir:  dir,
	}

	if dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			slog.Warn("failed to create session directory, running in-memory only", "dir", dir, "error", err)
			ss.dataDir = ""
		} else {
			ss.loadFromDisk()
		}
	}

	return ss
}

// loadFromDisk reads all .json files from the data directory.
func (ss *SessionStore) loadFromDisk() {
	entries, err := os.ReadDir(ss.dataDir)
	if err != nil {
		slog.Warn("failed to read session directory", "dir", ss.dataDir, "error", err)
		return
	}

	loaded := 0
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".json") {
			continue
		}

		data, err := os.ReadFile(filepath.Join(ss.dataDir, e.Name()))
		if err != nil {
			slog.Warn("failed to read session file", "file", e.Name(), "error", err)
			continue
		}

		var stored serializableSession
		if err := json.Unmarshal(data, &stored); err != nil {
			slog.Warn("failed to parse session file", "file", e.Name(), "error", err)
			continue
		}

		s := &Session{
			ID:           stored.ID,
			Title:        stored.Title,
			CreatedAt:    stored.CreatedAt,
			UpdatedAt:    stored.UpdatedAt,
			MessageCount: stored.MessageCount,
			TotalUsage:   stored.TotalUsage,
			Model:        stored.Model,
			TurnUsages:   stored.TurnUsages,
		}
		for _, sm := range stored.Messages {
			s.Messages = append(s.Messages, deserializeMessage(sm))
		}

		ss.sessions[s.ID] = s
		loaded++
	}

	if loaded > 0 {
		slog.Info("loaded sessions from disk", "count", loaded, "dir", ss.dataDir)
	}
}

// persist writes a single session to disk as JSON.
func (ss *SessionStore) persist(s *Session) {
	if ss.dataDir == "" {
		return
	}

	stored := serializableSession{
		ID:           s.ID,
		Title:        s.Title,
		CreatedAt:    s.CreatedAt,
		UpdatedAt:    s.UpdatedAt,
		MessageCount: s.MessageCount,
		TotalUsage:   s.TotalUsage,
		Model:        s.Model,
		TurnUsages:   s.TurnUsages,
	}
	for _, msg := range s.Messages {
		stored.Messages = append(stored.Messages, serializeMessage(msg))
	}

	data, err := json.Marshal(stored)
	if err != nil {
		slog.Warn("failed to marshal session", "id", s.ID, "error", err)
		return
	}

	path := filepath.Join(ss.dataDir, s.ID+".json")
	if err := os.WriteFile(path, data, 0o644); err != nil {
		slog.Warn("failed to write session file", "path", path, "error", err)
	}
}

// remove deletes a session file from disk.
func (ss *SessionStore) remove(id string) {
	if ss.dataDir == "" {
		return
	}
	path := filepath.Join(ss.dataDir, id+".json")
	_ = os.Remove(path)
}

// Create makes a new session. If title is empty, "New Session" is used.
func (ss *SessionStore) Create(title string) *Session {
	if title == "" {
		title = "New Session"
	}
	now := time.Now()
	s := &Session{
		ID:        uuid.NewString(),
		Title:     title,
		Messages:  []fantasy.Message{},
		CreatedAt: now,
		UpdatedAt: now,
	}

	ss.mu.Lock()
	ss.sessions[s.ID] = s
	ss.persist(s)
	ss.mu.Unlock()

	return s
}

// Get retrieves a session by ID.
func (ss *SessionStore) Get(id string) (*Session, bool) {
	ss.mu.RLock()
	defer ss.mu.RUnlock()
	s, ok := ss.sessions[id]
	return s, ok
}

// List returns all sessions sorted by UpdatedAt descending (most recent first).
func (ss *SessionStore) List() []*Session {
	ss.mu.RLock()
	defer ss.mu.RUnlock()

	list := make([]*Session, 0, len(ss.sessions))
	for _, s := range ss.sessions {
		list = append(list, s)
	}
	sort.Slice(list, func(i, j int) bool {
		return list[i].UpdatedAt.After(list[j].UpdatedAt)
	})
	return list
}

// Delete removes a session by ID. Returns true if the session existed.
func (ss *SessionStore) Delete(id string) bool {
	ss.mu.Lock()
	defer ss.mu.Unlock()
	_, ok := ss.sessions[id]
	if ok {
		delete(ss.sessions, id)
		ss.remove(id)
	}
	return ok
}

// AppendMessages adds messages to a session and updates the timestamp and count.
func (ss *SessionStore) AppendMessages(id string, msgs ...fantasy.Message) {
	ss.mu.Lock()
	defer ss.mu.Unlock()
	s, ok := ss.sessions[id]
	if !ok {
		return
	}
	s.Messages = append(s.Messages, msgs...)
	s.MessageCount += len(msgs)
	s.UpdatedAt = time.Now()
	ss.persist(s)
}

// GetMessages returns the message history for a session.
func (ss *SessionStore) GetMessages(id string) []fantasy.Message {
	ss.mu.RLock()
	defer ss.mu.RUnlock()
	s, ok := ss.sessions[id]
	if !ok {
		return nil
	}
	// Return a copy to avoid races.
	out := make([]fantasy.Message, len(s.Messages))
	copy(out, s.Messages)
	return out
}

// GetSerializedMessages returns the message history as JSON-safe serializable structs.
// This is used by the HTTP API to return messages without losing interface type info.
func (ss *SessionStore) GetSerializedMessages(id string) ([]serializableMessage, bool) {
	ss.mu.RLock()
	defer ss.mu.RUnlock()
	s, ok := ss.sessions[id]
	if !ok {
		return nil, false
	}
	out := make([]serializableMessage, len(s.Messages))
	for i, msg := range s.Messages {
		out[i] = serializeMessage(msg)
	}
	return out, true
}

// UpdateTitle changes the title of a session.
func (ss *SessionStore) UpdateTitle(id string, title string) {
	ss.mu.Lock()
	defer ss.mu.Unlock()
	s, ok := ss.sessions[id]
	if !ok {
		return
	}
	s.Title = title
	s.UpdatedAt = time.Now()
	ss.persist(s)
}

// UpdateUsage stores the total token usage and model for a session,
// and appends a per-turn usage entry (with per-step breakdown) so each
// assistant message can show its own token count after a browser refresh.
// Called after agent_finish so the data survives a browser refresh.
func (ss *SessionStore) UpdateUsage(id string, totalUsage fantasy.Usage, model string, stepResults []fantasy.StepResult) {
	ss.mu.Lock()
	defer ss.mu.Unlock()
	s, ok := ss.sessions[id]
	if !ok {
		return
	}
	u := usageToSession(totalUsage)
	s.TotalUsage = &u
	s.Model = model

	// Build per-step usage breakdown
	stepUsages := make([]sessionUsage, len(stepResults))
	for i, sr := range stepResults {
		stepUsages[i] = usageToSession(sr.Usage)
	}

	s.TurnUsages = append(s.TurnUsages, turnUsage{
		Usage:      u,
		Model:      model,
		Steps:      len(stepResults),
		StepUsages: stepUsages,
	})
	ss.persist(s)
}

func usageToSession(u fantasy.Usage) sessionUsage {
	return sessionUsage{
		InputTokens:         u.InputTokens,
		OutputTokens:        u.OutputTokens,
		TotalTokens:         u.TotalTokens,
		ReasoningTokens:     u.ReasoningTokens,
		CacheCreationTokens: u.CacheCreationTokens,
		CacheReadTokens:     u.CacheReadTokens,
	}
}
