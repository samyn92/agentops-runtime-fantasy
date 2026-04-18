/*
Agent Runtime — Fantasy (Go)

Tool hooks: security enforcement, memory-aware auditing, declarative
memory capture (memorySaveRules), and pre-execution context injection
(contextInjectTools). Applied as wrappers around tool execution.
*/
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"regexp"
	"strings"
	"time"

	"charm.land/fantasy"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
)

// DefaultMaxToolResultChars is the default limit for tool result content.
// 50 000 chars ≈ 12 500 tokens — large enough for useful output,
// small enough to prevent a single tool result from consuming >6% of
// Claude's 200k context window.
const DefaultMaxToolResultChars = 50_000

// hookWrappedTool wraps a tool with security hooks, memory integration, and output truncation.
type hookWrappedTool struct {
	inner              fantasy.AgentTool
	hooks              *ToolHooksEntry
	engram             *EngramClient
	maxToolResultChars int
}

func wrapToolsWithHooks(tools []fantasy.AgentTool, hooks *ToolHooksEntry, maxResultChars int, engram *EngramClient) []fantasy.AgentTool {
	if hooks == nil && maxResultChars <= 0 && engram == nil {
		return tools
	}
	if maxResultChars <= 0 {
		maxResultChars = DefaultMaxToolResultChars
	}
	wrapped := make([]fantasy.AgentTool, len(tools))
	for i, t := range tools {
		wrapped[i] = &hookWrappedTool{inner: t, hooks: hooks, engram: engram, maxToolResultChars: maxResultChars}
	}
	return wrapped
}

func (h *hookWrappedTool) Info() fantasy.ToolInfo {
	return h.inner.Info()
}

func (h *hookWrappedTool) ProviderOptions() fantasy.ProviderOptions {
	return h.inner.ProviderOptions()
}

func (h *hookWrappedTool) SetProviderOptions(opts fantasy.ProviderOptions) {
	h.inner.SetProviderOptions(opts)
}

func (h *hookWrappedTool) Run(ctx context.Context, call fantasy.ToolCall) (fantasy.ToolResponse, error) {
	toolName := h.inner.Info().Name

	// Start a tracing span for this tool execution (name includes tool for readability)
	ctx, span := tracer.Start(ctx, "tool.execute: "+toolName)
	defer span.End()

	// Set tool identity attributes at creation (important for sampling decisions)
	span.SetAttributes(
		attrGenAIOperationName.String("execute_tool"),
		attrToolName.String(toolName),
		attrToolType.String(classifyToolType(toolName)),
		attrGenAIToolName.String(toolName),
	)
	if call.ID != "" {
		span.SetAttributes(attrGenAIToolCallID.String(call.ID))
	}

	// Parse input for inspection
	var args map[string]any
	_ = json.Unmarshal([]byte(call.Input), &args)

	// Record tool input as a span event for trace visibility
	recordToolInputEvent(span, toolName, call.Input)

	// Surface a compact preview as a span attribute so the trace waterfall
	// can label rows meaningfully (e.g. "bash: git push origin feat/...").
	if preview := buildToolCallPreview(toolName, args); preview != "" {
		span.SetAttributes(attribute.String("tool.preview", preview))
	}

	// Before: blocked commands — save violation to memory
	if h.hooks != nil && toolName == "bash" && len(h.hooks.BlockedCommands) > 0 {
		cmd, _ := args["command"].(string)
		for _, blocked := range h.hooks.BlockedCommands {
			if strings.Contains(cmd, blocked) {
				span.SetAttributes(attrToolError.Bool(true))
				span.SetStatus(codes.Error, "blocked command")
				// Persist security violation to memory (non-blocking)
				h.saveSecurityObservation(ctx, span, "blocked_command",
					fmt.Sprintf("Blocked command: %s", blocked),
					fmt.Sprintf("Agent attempted bash command matching blocked pattern.\nPattern: %s\nCommand: %s", blocked, truncateForHook(cmd, 500)),
				)
				return fantasy.NewTextErrorResponse(
					fmt.Sprintf("Blocked command pattern: %s", blocked)), nil
			}
		}
	}

	// Before: path access control — save violation to memory
	if h.hooks != nil && len(h.hooks.AllowedPaths) > 0 {
		path, _ := args["path"].(string)
		if path != "" {
			allowed := false
			for _, prefix := range h.hooks.AllowedPaths {
				if strings.HasPrefix(path, prefix) {
					allowed = true
					break
				}
			}
			if !allowed {
				span.SetAttributes(attrToolError.Bool(true))
				span.SetStatus(codes.Error, "path not allowed")
				// Persist security violation to memory (non-blocking)
				h.saveSecurityObservation(ctx, span, "path_denied",
					fmt.Sprintf("Path denied: %s via %s", path, toolName),
					fmt.Sprintf("Agent attempted to access path outside allowlist.\nTool: %s\nPath: %s\nAllowed prefixes: %s", toolName, path, strings.Join(h.hooks.AllowedPaths, ", ")),
				)
				return fantasy.NewTextErrorResponse(
					fmt.Sprintf("Path not in allowed list: %s", path)), nil
			}
		}
	}

	// Before: context injection — query memory and prepend relevant context
	if h.hooks != nil && h.engram != nil {
		for _, rule := range h.hooks.ContextInjectTools {
			if rule.Tool == toolName {
				h.injectContextFromMemory(ctx, span, toolName, args, &call, rule)
				break
			}
		}
	}

	// Execute with timing
	start := time.Now()
	result, err := h.inner.Run(ctx, call)
	elapsed := time.Since(start)

	span.SetAttributes(attrToolDuration.Int64(elapsed.Milliseconds()))

	if err != nil {
		span.SetAttributes(attrToolError.Bool(true))
		recordError(span, err)
	} else if result.IsError {
		span.SetAttributes(attrToolError.Bool(true))
		span.SetStatus(codes.Error, "tool returned error")
	}

	// After: truncate large tool results to prevent context window blowup.
	// Applied before audit/memory hooks so they reflect the truncated state.
	if err == nil && h.maxToolResultChars > 0 && len(result.Content) > h.maxToolResultChars {
		originalLen := len(result.Content)
		result.Content = result.Content[:h.maxToolResultChars] + fmt.Sprintf(
			"\n\n[truncated — showing %d of %d chars. Use offset/limit params or narrower queries for full content]",
			h.maxToolResultChars, originalLen,
		)
		slog.Warn("tool result truncated",
			"tool", toolName,
			"original_chars", originalLen,
			"truncated_to", h.maxToolResultChars,
		)
	}

	// After: audit logging — enhanced with memory persistence
	if h.hooks != nil {
		for _, auditTool := range h.hooks.AuditTools {
			if auditTool == toolName {
				slog.Info("audit",
					"type", "tool_call",
					"tool", toolName,
					"timestamp", time.Now().Unix(),
					"is_error", result.IsError,
				)
				// Persist audit trail as a memory observation (searchable via FTS5).
				// topic_key scoped per-tool so repeated calls upsert (Tier 1 dedup).
				h.saveAuditObservation(ctx, span, toolName, elapsed, result.IsError)
				break
			}
		}
	}

	// After: memory save rules — declarative capture of tool results
	if h.hooks != nil && h.engram != nil && err == nil {
		for _, rule := range h.hooks.MemorySaveRules {
			if rule.Tool == toolName {
				h.applyMemorySaveRule(ctx, span, toolName, args, result, elapsed, rule)
				break // one rule per tool per call
			}
		}
	}

	// Record tool output as a span event for trace visibility
	recordToolOutputEvent(span, toolName, result.Content, result.IsError)

	// Record a compact tool.call event on the root prompt span.
	// This survives even when individual tool.execute spans are lost
	// by the batch exporter (common for short-lived task pods).
	// The console UI renders these as tool rows in the trace waterfall.
	if rootSpan := rootSpanFromContext(ctx); rootSpan != nil {
		evAttrs := []attribute.KeyValue{
			attribute.String("tool.name", toolName),
			attribute.String("tool.type", classifyToolType(toolName)),
			attribute.Int64("tool.duration_ms", elapsed.Milliseconds()),
		}
		// Compact preview of the call: for bash, the command; for other tools,
		// the first scalar arg. Lets the trace UI render meaningful row labels
		// without parsing tool.input from a separate event.
		if preview := buildToolCallPreview(toolName, args); preview != "" {
			evAttrs = append(evAttrs, attribute.String("tool.preview", preview))
		}
		if err != nil || result.IsError {
			evAttrs = append(evAttrs, attribute.Bool("tool.error", true))
		}
		rootSpan.AddEvent("tool.call", trace.WithAttributes(evAttrs...))
	} else {
		slog.Debug("root span not in context, skipping tool.call event", "tool", toolName)
	}

	return result, err
}

// ====================================================================
// Memory-integrated hook helpers
// ====================================================================

// saveSecurityObservation persists a security violation as a memory observation.
// Uses topic_key so repeated violations of the same type upsert (Tier 1 dedup).
// Non-blocking: logs on failure, never fails the tool call.
func (h *hookWrappedTool) saveSecurityObservation(ctx context.Context, span trace.Span, violation, title, content string) {
	if h.engram == nil {
		return
	}
	// Record event synchronously before goroutine to avoid race with span.End()
	span.AddEvent("hook.security_saved", trace.WithAttributes(
		attribute.String("violation", violation),
		attribute.String("title", title),
	))
	go func() {
		body := map[string]any{
			"session_id": h.engram.SessionID(),
			"type":       "security",
			"title":      title,
			"content":    content,
			"project":    h.engram.project,
			"scope":      "project",
			"topic_key":  fmt.Sprintf("security/%s", violation),
			"tags":       []string{"security", "hook", violation},
		}
		if _, err := h.engram.post("/observations", body); err != nil {
			slog.Warn("hook: failed to save security observation", "error", err, "violation", violation)
		}
	}()
}

// saveAuditObservation persists an audited tool call as a memory observation.
// Uses topic_key per-tool so the latest call upserts (Tier 1 dedup).
func (h *hookWrappedTool) saveAuditObservation(ctx context.Context, span trace.Span, toolName string, elapsed time.Duration, isError bool) {
	if h.engram == nil {
		return
	}
	status := "success"
	if isError {
		status = "error"
	}
	// Record event synchronously before goroutine to avoid race with span.End()
	span.AddEvent("hook.audit_saved", trace.WithAttributes(
		attribute.String("tool", toolName),
		attribute.String("status", status),
		attribute.Int64("duration_ms", elapsed.Milliseconds()),
	))
	go func() {
		body := map[string]any{
			"session_id": h.engram.SessionID(),
			"type":       "audit",
			"title":      fmt.Sprintf("Audit: %s (%s, %dms)", toolName, status, elapsed.Milliseconds()),
			"content":    fmt.Sprintf("Tool: %s\nStatus: %s\nDuration: %s", toolName, status, elapsed.Round(time.Millisecond)),
			"project":    h.engram.project,
			"scope":      "project",
			"topic_key":  fmt.Sprintf("audit/%s", toolName),
			"tags":       []string{"audit", "hook", toolName},
		}
		if _, err := h.engram.post("/observations", body); err != nil {
			slog.Warn("hook: failed to save audit observation", "error", err, "tool", toolName)
		}
	}()
}

// applyMemorySaveRule evaluates a declarative memory save rule against a tool result.
// If the rule matches (no matchOutput, or regex matches output), saves an observation.
func (h *hookWrappedTool) applyMemorySaveRule(ctx context.Context, span trace.Span, toolName string, args map[string]any, result fantasy.ToolResponse, elapsed time.Duration, rule MemorySaveRule) {
	// Check matchOutput pattern (if configured)
	if rule.MatchOutput != "" {
		matched, err := regexp.MatchString(rule.MatchOutput, result.Content)
		if err != nil {
			slog.Warn("hook: invalid matchOutput regex", "pattern", rule.MatchOutput, "error", err)
			span.AddEvent("hook.memory_save_skipped", trace.WithAttributes(
				attribute.String("tool", toolName),
				attribute.String("reason", "invalid_regex"),
				attribute.String("pattern", rule.MatchOutput),
			))
			return
		}
		if !matched {
			span.AddEvent("hook.memory_save_skipped", trace.WithAttributes(
				attribute.String("tool", toolName),
				attribute.String("reason", "output_not_matched"),
			))
			return
		}
	}

	// Check matchArgs patterns (if configured)
	if len(rule.MatchArgs) > 0 {
		for argKey, argPattern := range rule.MatchArgs {
			argVal, _ := args[argKey].(string)
			if argVal == "" {
				span.AddEvent("hook.memory_save_skipped", trace.WithAttributes(
					attribute.String("tool", toolName),
					attribute.String("reason", "arg_missing"),
					attribute.String("arg", argKey),
				))
				return // arg not present — skip
			}
			matched, err := regexp.MatchString(argPattern, argVal)
			if err != nil || !matched {
				span.AddEvent("hook.memory_save_skipped", trace.WithAttributes(
					attribute.String("tool", toolName),
					attribute.String("reason", "arg_not_matched"),
					attribute.String("arg", argKey),
				))
				return
			}
		}
	}

	obsType := rule.Type
	if obsType == "" {
		obsType = "discovery"
	}
	scope := rule.Scope
	if scope == "" {
		scope = "project"
	}

	// Build a concise title from the tool name and first significant arg
	title := fmt.Sprintf("Auto-captured: %s", toolName)
	if cmd, ok := args["command"].(string); ok && cmd != "" {
		title = fmt.Sprintf("Auto-captured: %s — %s", toolName, truncateForHook(cmd, 80))
	} else if path, ok := args["path"].(string); ok && path != "" {
		title = fmt.Sprintf("Auto-captured: %s — %s", toolName, truncateForHook(path, 80))
	}

	// Content: truncated output (enough to be useful, not overwhelming)
	content := truncateForHook(result.Content, 1000)

	// Record the event synchronously BEFORE the goroutine to avoid race with span.End()
	span.AddEvent("hook.memory_saved", trace.WithAttributes(
		attribute.String("tool", toolName),
		attribute.String("type", obsType),
		attribute.String("title", title),
		attribute.Int("content_length", len(content)),
	))

	go func() {
		body := map[string]any{
			"session_id": h.engram.SessionID(),
			"type":       obsType,
			"title":      title,
			"content":    content,
			"project":    h.engram.project,
			"scope":      scope,
			"tags":       []string{"auto-capture", "hook", toolName},
		}
		if _, err := h.engram.post("/observations", body); err != nil {
			slog.Warn("hook: failed to save memory save rule observation", "error", err, "tool", toolName)
		} else {
			slog.Info("hook: memory save rule triggered", "tool", toolName, "type", obsType)
		}
	}()
}

// injectContextFromMemory queries agentops-memory before a tool call and logs
// the relevant context as a span event. For tools like bash, this provides
// the agent with past experience before execution.
func (h *hookWrappedTool) injectContextFromMemory(ctx context.Context, span trace.Span, toolName string, args map[string]any, call *fantasy.ToolCall, rule ContextInjectRule) {
	// Determine query: use tool args as the search query
	var query string
	switch rule.Query {
	case "from_tool_args", "":
		// Use the most significant arg as the query
		if cmd, ok := args["command"].(string); ok && cmd != "" {
			query = cmd
		} else if q, ok := args["query"].(string); ok && q != "" {
			query = q
		} else if path, ok := args["path"].(string); ok && path != "" {
			query = path
		}
	default:
		query = rule.Query // static query string
	}
	if query == "" {
		span.AddEvent("hook.context_inject_skipped", trace.WithAttributes(
			attribute.String("tool", toolName),
			attribute.String("reason", "empty_query"),
		))
		return
	}

	limit := rule.Limit
	if limit <= 0 {
		limit = 3
	}

	memCtx := h.engram.FetchContext(ctx, limit, query)
	if memCtx == "" {
		span.AddEvent("hook.context_inject_empty", trace.WithAttributes(
			attribute.String("tool", toolName),
			attribute.String("query", truncateForHook(query, 200)),
		))
		return
	}

	// Record what was injected for trace visibility
	span.AddEvent("hook.context_injected", trace.WithAttributes(
		attribute.String("tool", toolName),
		attribute.String("query", truncateForHook(query, 200)),
		attribute.Int("results", strings.Count(memCtx, "\n- ")),
	))
	slog.Debug("hook: context injected before tool call", "tool", toolName, "query_len", len(query))
}

// truncateForHook truncates a string with an ellipsis suffix if it exceeds maxLen.
func truncateForHook(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

// buildToolCallPreview returns a compact, human-readable preview of a tool
// invocation, suitable for trace waterfall labels. Picks the most informative
// scalar argument per tool family. Capped at ~80 chars on a single line.
func buildToolCallPreview(toolName string, args map[string]any) string {
	if args == nil {
		return ""
	}
	// Pick the best argument by tool family
	candidates := []string{"command", "path", "pattern", "url", "query", "branch", "message", "name"}
	var picked string
	for _, k := range candidates {
		if v, ok := args[k].(string); ok && v != "" {
			picked = v
			break
		}
	}
	if picked == "" {
		return ""
	}
	// Single-line, truncated
	picked = strings.ReplaceAll(picked, "\n", " ")
	picked = strings.ReplaceAll(picked, "\t", " ")
	return truncateForHook(picked, 80)
}
