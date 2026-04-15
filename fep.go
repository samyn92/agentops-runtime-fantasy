/*
Agent Runtime — Fantasy (Go)

FEP (Fantasy Event Protocol) SSE emitter.
Maps Fantasy SDK streaming callbacks to SSE events for the console.
Supports multiple backends: HTTP SSE (per-request) and NATS (persistent).
*/
package main

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"sync"
	"time"

	"charm.land/fantasy"
	"github.com/nats-io/nats.go"
)

// ── Emitter abstraction ──
// All emit* methods call emit(). Backends implement this single method.

// fepBackend is the low-level write target for FEP events.
type fepBackend interface {
	emit(eventType string, fields map[string]any)
}

// fepEmitter is the main FEP emitter used throughout the runtime.
// It holds one or more backends (SSE writer, NATS publisher, etc.)
// and fans out every event to all of them.
type fepEmitter struct {
	backends []fepBackend
	mu       sync.Mutex
}

// newFEPEmitter creates an emitter with an SSE backend writing to w.
func newFEPEmitter(w http.ResponseWriter) *fepEmitter {
	return &fepEmitter{
		backends: []fepBackend{&sseBackend{w: w, f: w.(http.Flusher)}},
	}
}

// newNATSOnlyEmitter creates an emitter backed only by NATS (no HTTP writer).
// Used by handleInternalPrompt where no browser SSE stream exists.
func newNATSOnlyEmitter(np *natsPublisher) *fepEmitter {
	if np == nil {
		return nil
	}
	return &fepEmitter{
		backends: []fepBackend{np},
	}
}

// withNATS returns a new emitter that also publishes to NATS.
func (e *fepEmitter) withNATS(np *natsPublisher) *fepEmitter {
	if np == nil {
		return e
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	e.backends = append(e.backends, np)
	return e
}

// emit fans out to all backends.
func (e *fepEmitter) emit(eventType string, fields map[string]any) {
	if fields == nil {
		fields = make(map[string]any)
	}
	fields["type"] = eventType
	fields["timestamp"] = time.Now().UTC().Format(time.RFC3339)

	e.mu.Lock()
	defer e.mu.Unlock()
	for _, b := range e.backends {
		b.emit(eventType, fields)
	}
}

// ── SSE backend — writes to HTTP response ──

type sseBackend struct {
	w  http.ResponseWriter
	f  http.Flusher
	mu sync.Mutex
}

func (s *sseBackend) emit(_ string, fields map[string]any) {
	payload, err := json.Marshal(fields)
	if err != nil {
		payload = []byte(fmt.Sprintf(`{"type":"%s","error":"marshal failed"}`, fields["type"]))
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	fmt.Fprintf(s.w, "data: %s\n\n", payload)
	if s.f != nil {
		s.f.Flush()
	}
}

// ── NATS backend — publishes to NATS subjects ──
// Subject format: agents.{namespace}.{agentName}.fep.{eventType}
// The BFF subscribes to agents.> and injects events into its SSE multiplexer.

type natsPublisher struct {
	nc        *nats.Conn
	namespace string
	agentName string
}

// newNATSPublisher connects to NATS and returns a publisher.
// Returns nil if NATS_URL is not set (graceful degradation).
func newNATSPublisher(namespace, agentName string) *natsPublisher {
	url := os.Getenv("NATS_URL")
	if url == "" {
		slog.Info("NATS_URL not set, FEP NATS publishing disabled")
		return nil
	}

	nc, err := nats.Connect(url,
		nats.Name(fmt.Sprintf("agentops-runtime/%s/%s", namespace, agentName)),
		nats.RetryOnFailedConnect(true),
		nats.MaxReconnects(-1), // reconnect forever
		nats.ReconnectWait(2*time.Second),
		nats.DisconnectErrHandler(func(_ *nats.Conn, err error) {
			if err != nil {
				slog.Warn("NATS disconnected", "error", err)
			}
		}),
		nats.ReconnectHandler(func(_ *nats.Conn) {
			slog.Info("NATS reconnected")
		}),
	)
	if err != nil {
		slog.Error("failed to connect to NATS", "url", url, "error", err)
		return nil
	}

	slog.Info("NATS publisher connected", "url", url, "namespace", namespace, "agent", agentName)
	return &natsPublisher{
		nc:        nc,
		namespace: namespace,
		agentName: agentName,
	}
}

func (n *natsPublisher) emit(eventType string, fields map[string]any) {
	if n.nc == nil {
		return
	}

	// Subject: agents.{namespace}.{agentName}.fep.{eventType}
	subject := fmt.Sprintf("agents.%s.%s.fep.%s", n.namespace, n.agentName, eventType)

	payload, err := json.Marshal(fields)
	if err != nil {
		slog.Warn("NATS: failed to marshal FEP event", "type", eventType, "error", err)
		return
	}

	if err := n.nc.Publish(subject, payload); err != nil {
		slog.Warn("NATS: failed to publish FEP event", "subject", subject, "error", err)
	}
}

func (n *natsPublisher) close() {
	if n.nc != nil {
		n.nc.Drain()
	}
}

// --------------------------------------------------------------------
// Agent lifecycle events
// --------------------------------------------------------------------

func (e *fepEmitter) emitAgentStart(sessionId, prompt string, traceID string, budget *ContextBudgetSnapshot) {
	fields := map[string]any{
		"session_id": sessionId,
		"prompt":     prompt,
	}
	if traceID != "" {
		fields["trace_id"] = traceID
	}
	if budget != nil {
		fields["context_budget"] = budget
	}
	e.emit("agent_start", fields)
}

func (e *fepEmitter) emitAgentFinish(sessionId string, totalUsage fantasy.Usage, stepCount int, model string) {
	e.emit("agent_finish", map[string]any{
		"session_id":  sessionId,
		"total_usage": usageMap(totalUsage),
		"step_count":  stepCount,
		"model":       model,
	})
}

func (e *fepEmitter) emitAgentError(sessionId string, err error, retryable bool) {
	e.emit("agent_error", map[string]any{
		"session_id": sessionId,
		"error":      err.Error(),
		"retryable":  retryable,
	})
}

// --------------------------------------------------------------------
// Step events
// --------------------------------------------------------------------

func (e *fepEmitter) emitStepStart(stepNumber int, sessionId string) {
	e.emit("step_start", map[string]any{
		"step_number": stepNumber,
		"session_id":  sessionId,
	})
}

func (e *fepEmitter) emitStepFinish(stepNumber int, sessionId string, usage fantasy.Usage, finishReason fantasy.FinishReason, toolCallCount int, budget *ContextBudgetSnapshot) {
	fields := map[string]any{
		"step_number":     stepNumber,
		"session_id":      sessionId,
		"usage":           usageMap(usage),
		"finish_reason":   string(finishReason),
		"tool_call_count": toolCallCount,
	}
	if budget != nil {
		fields["context_budget"] = budget
	}
	e.emit("step_finish", fields)
}

// --------------------------------------------------------------------
// Text streaming events
// --------------------------------------------------------------------

func (e *fepEmitter) emitTextStart(id string) {
	e.emit("text_start", map[string]any{"id": id})
}

func (e *fepEmitter) emitTextDelta(id, delta string) {
	e.emit("text_delta", map[string]any{"id": id, "delta": delta})
}

func (e *fepEmitter) emitTextEnd(id string) {
	e.emit("text_end", map[string]any{"id": id})
}

// --------------------------------------------------------------------
// Reasoning streaming events
// --------------------------------------------------------------------

func (e *fepEmitter) emitReasoningStart(id string) {
	e.emit("reasoning_start", map[string]any{"id": id})
}

func (e *fepEmitter) emitReasoningDelta(id, delta string) {
	e.emit("reasoning_delta", map[string]any{"id": id, "delta": delta})
}

func (e *fepEmitter) emitReasoningEnd(id string) {
	e.emit("reasoning_end", map[string]any{"id": id})
}

// --------------------------------------------------------------------
// Tool input streaming events
// --------------------------------------------------------------------

func (e *fepEmitter) emitToolInputStart(id, toolName string) {
	e.emit("tool_input_start", map[string]any{"id": id, "tool_name": toolName})
}

func (e *fepEmitter) emitToolInputDelta(id, delta string) {
	e.emit("tool_input_delta", map[string]any{"id": id, "delta": delta})
}

func (e *fepEmitter) emitToolInputEnd(id string) {
	e.emit("tool_input_end", map[string]any{"id": id})
}

// --------------------------------------------------------------------
// Tool execution events
// --------------------------------------------------------------------

func (e *fepEmitter) emitToolCall(id, toolName, input string, providerExecuted bool) {
	e.emit("tool_call", map[string]any{
		"id":                id,
		"tool_name":         toolName,
		"input":             input,
		"provider_executed": providerExecuted,
	})
}

func (e *fepEmitter) emitToolResult(id, toolName, output string, isError bool, metadata string, mediaType, data string) {
	fields := map[string]any{
		"id":        id,
		"tool_name": toolName,
		"output":    output,
		"is_error":  isError,
	}
	if metadata != "" {
		fields["metadata"] = metadata
	}
	if mediaType != "" {
		fields["media_type"] = mediaType
	}
	if data != "" {
		fields["data"] = data
	}
	e.emit("tool_result", fields)
}

// --------------------------------------------------------------------
// Source / citation events
// --------------------------------------------------------------------

func (e *fepEmitter) emitSource(id, sourceType, url, title string) {
	e.emit("source", map[string]any{
		"id":          id,
		"source_type": sourceType,
		"url":         url,
		"title":       title,
	})
}

// --------------------------------------------------------------------
// Warning events
// --------------------------------------------------------------------

func (e *fepEmitter) emitWarnings(warnings []fantasy.CallWarning) {
	msgs := make([]string, len(warnings))
	for i, w := range warnings {
		msgs[i] = w.Message
	}
	e.emit("warnings", map[string]any{
		"warnings": msgs,
	})
}

// --------------------------------------------------------------------
// Stream finish
// --------------------------------------------------------------------

func (e *fepEmitter) emitStreamFinish(usage fantasy.Usage, finishReason fantasy.FinishReason) {
	e.emit("stream_finish", map[string]any{
		"usage":         usageMap(usage),
		"finish_reason": string(finishReason),
	})
}

// --------------------------------------------------------------------
// Interactive events (permission / question / idle)
// --------------------------------------------------------------------

func (e *fepEmitter) emitPermissionAsked(id, sessionId, toolName, input, description string) {
	e.emit("permission_asked", map[string]any{
		"id":          id,
		"session_id":  sessionId,
		"tool_name":   toolName,
		"input":       input,
		"description": description,
	})
}

func (e *fepEmitter) emitQuestionAsked(id, sessionId string, questions json.RawMessage) {
	e.emit("question_asked", map[string]any{
		"id":         id,
		"session_id": sessionId,
		"questions":  questions,
	})
}

func (e *fepEmitter) emitSessionIdle(sessionId string) {
	e.emit("session_idle", map[string]any{
		"session_id": sessionId,
	})
}

// --------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------

// usageMap converts a fantasy.Usage into a map for JSON serialisation.
func usageMap(u fantasy.Usage) map[string]any {
	return map[string]any{
		"input_tokens":          u.InputTokens,
		"output_tokens":         u.OutputTokens,
		"total_tokens":          u.TotalTokens,
		"reasoning_tokens":      u.ReasoningTokens,
		"cache_creation_tokens": u.CacheCreationTokens,
		"cache_read_tokens":     u.CacheReadTokens,
	}
}
