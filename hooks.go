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

// hookWrappedTool wraps a tool with security hooks.
type hookWrappedTool struct {
	inner fantasy.AgentTool
	hooks *ToolHooksEntry
}

func wrapToolsWithHooks(tools []fantasy.AgentTool, hooks *ToolHooksEntry) []fantasy.AgentTool {
	if hooks == nil {
		return tools
	}
	wrapped := make([]fantasy.AgentTool, len(tools))
	for i, t := range tools {
		wrapped[i] = &hookWrappedTool{inner: t, hooks: hooks}
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
	if toolName == "bash" && len(h.hooks.BlockedCommands) > 0 {
		cmd, _ := args["command"].(string)
		for _, blocked := range h.hooks.BlockedCommands {
			if strings.Contains(cmd, blocked) {
				return fantasy.NewTextErrorResponse(
					fmt.Sprintf("Blocked command pattern: %s", blocked)), nil
			}
		}
	}

	// Before: path access control
	if len(h.hooks.AllowedPaths) > 0 {
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

	// After: audit logging
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

	return result, err
}
