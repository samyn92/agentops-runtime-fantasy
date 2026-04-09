/*
Agent Runtime — Fantasy (Go)

Tool security hooks: blocked commands, allowed paths, audit logging.
Applied as wrappers around tool execution.
*/
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"
	"time"

	"charm.land/fantasy"
)

// DefaultMaxToolResultChars is the default limit for tool result content.
// 50 000 chars ≈ 12 500 tokens — large enough for useful output,
// small enough to prevent a single tool result from consuming >6% of
// Claude's 200k context window.
const DefaultMaxToolResultChars = 50_000

// hookWrappedTool wraps a tool with security hooks and output truncation.
type hookWrappedTool struct {
	inner              fantasy.AgentTool
	hooks              *ToolHooksEntry
	maxToolResultChars int
}

func wrapToolsWithHooks(tools []fantasy.AgentTool, hooks *ToolHooksEntry, maxResultChars int) []fantasy.AgentTool {
	if hooks == nil && maxResultChars <= 0 {
		return tools
	}
	if maxResultChars <= 0 {
		maxResultChars = DefaultMaxToolResultChars
	}
	wrapped := make([]fantasy.AgentTool, len(tools))
	for i, t := range tools {
		wrapped[i] = &hookWrappedTool{inner: t, hooks: hooks, maxToolResultChars: maxResultChars}
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

	// Parse input for inspection
	var args map[string]any
	_ = json.Unmarshal([]byte(call.Input), &args)

	// Before: blocked commands
	if h.hooks != nil && toolName == "bash" && len(h.hooks.BlockedCommands) > 0 {
		cmd, _ := args["command"].(string)
		for _, blocked := range h.hooks.BlockedCommands {
			if strings.Contains(cmd, blocked) {
				return fantasy.NewTextErrorResponse(
					fmt.Sprintf("Blocked command pattern: %s", blocked)), nil
			}
		}
	}

	// Before: path access control
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
				return fantasy.NewTextErrorResponse(
					fmt.Sprintf("Path not in allowed list: %s", path)), nil
			}
		}
	}

	// Execute
	result, err := h.inner.Run(ctx, call)

	// After: truncate large tool results to prevent context window blowup.
	// Applied before audit logging so the log reflects the truncated state.
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

	// After: audit logging
	if h.hooks != nil {
		for _, auditTool := range h.hooks.AuditTools {
			if auditTool == toolName {
				slog.Info("audit",
					"type", "tool_call",
					"tool", toolName,
					"timestamp", time.Now().Unix(),
					"is_error", result.IsError,
				)
				break
			}
		}
	}

	return result, err
}
