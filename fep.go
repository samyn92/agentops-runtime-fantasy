/*
Agent Runtime — Fantasy (Go)

FEP (Fantasy Event Protocol) SSE emitter.
Maps Fantasy SDK streaming callbacks to SSE events for the console.
*/
package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"sync"

	"charm.land/fantasy"
)

// fepEmitter writes FEP-formatted SSE events to an HTTP response.
// All writes are mutex-protected because Fantasy callbacks may fire
// from different goroutines.
type fepEmitter struct {
	w  http.ResponseWriter
	f  http.Flusher
	mu sync.Mutex
}

// newFEPEmitter creates a new FEP SSE emitter. The caller must have
// already set Content-Type: text/event-stream on w.
func newFEPEmitter(w http.ResponseWriter) *fepEmitter {
	f, _ := w.(http.Flusher)
	return &fepEmitter{w: w, f: f}
}

// emit writes a single SSE data frame: data: {"type":"<eventType>", ...fields}\n\n
func (e *fepEmitter) emit(eventType string, fields map[string]any) {
	if fields == nil {
		fields = make(map[string]any)
	}
	fields["type"] = eventType

	payload, err := json.Marshal(fields)
	if err != nil {
		payload = []byte(fmt.Sprintf(`{"type":"%s","error":"marshal failed"}`, eventType))
	}

	e.mu.Lock()
	defer e.mu.Unlock()
	fmt.Fprintf(e.w, "data: %s\n\n", payload)
	if e.f != nil {
		e.f.Flush()
	}
}

// --------------------------------------------------------------------
// Agent lifecycle events
// --------------------------------------------------------------------

func (e *fepEmitter) emitAgentStart(sessionId, prompt string) {
	e.emit("agent_start", map[string]any{
		"session_id": sessionId,
		"prompt":     prompt,
	})
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

func (e *fepEmitter) emitStepFinish(stepNumber int, sessionId string, usage fantasy.Usage, finishReason fantasy.FinishReason, toolCallCount int) {
	e.emit("step_finish", map[string]any{
		"step_number":     stepNumber,
		"session_id":      sessionId,
		"usage":           usageMap(usage),
		"finish_reason":   string(finishReason),
		"tool_call_count": toolCallCount,
	})
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

func (e *fepEmitter) emitSessionStatus(sessionId, status string) {
	e.emit("session_status", map[string]any{
		"session_id": sessionId,
		"status":     status,
	})
}

func (e *fepEmitter) emitSessionTitleUpdated(sessionId, title string) {
	e.emit("session_title_updated", map[string]any{
		"session_id": sessionId,
		"title":      title,
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
