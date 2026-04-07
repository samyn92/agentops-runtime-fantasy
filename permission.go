/*
Agent Runtime — Fantasy (Go)

Permission gate: wraps tools to require user approval before execution.
The gate emits a permission_asked FEP event and blocks on a Go channel
until the console replies with once/always/deny.
*/
package main

import (
	"context"
	"encoding/json"
	"sync"
	"time"

	"charm.land/fantasy"
	"github.com/google/uuid"
)

// PermissionResponse represents the user's reply to a permission request.
type PermissionResponse struct {
	Response string `json:"response"` // "once" | "always" | "deny"
}

// permissionGate manages pending permission requests and persistent allow-lists.
type permissionGate struct {
	mu            sync.Mutex
	pending       map[string]chan PermissionResponse // permId -> response channel
	alwaysAllowed map[string]bool                    // toolName -> permanently allowed
	emitEvent     func(id, sessionId, toolName, input, description string)
}

func newPermissionGate(
	emitEvent func(id, sessionId, toolName, input, description string),
) *permissionGate {
	return &permissionGate{
		pending:       make(map[string]chan PermissionResponse),
		alwaysAllowed: make(map[string]bool),
		emitEvent:     emitEvent,
	}
}

// resolvePermission blocks until the user replies or context is cancelled.
func (g *permissionGate) resolvePermission(ctx context.Context, id, sessionId, toolName, inputJSON, description string) PermissionResponse {
	ch := make(chan PermissionResponse, 1)

	g.mu.Lock()
	g.pending[id] = ch
	g.mu.Unlock()

	defer func() {
		g.mu.Lock()
		delete(g.pending, id)
		g.mu.Unlock()
	}()

	// Emit the permission_asked FEP event
	g.emitEvent(id, sessionId, toolName, inputJSON, description)

	// Wait for reply with a generous timeout
	timeout := time.After(5 * time.Minute)

	select {
	case resp := <-ch:
		return resp
	case <-timeout:
		return PermissionResponse{Response: "deny"}
	case <-ctx.Done():
		return PermissionResponse{Response: "deny"}
	}
}

// reply delivers a response to a pending permission request.
func (g *permissionGate) reply(permId string, resp PermissionResponse) bool {
	g.mu.Lock()
	ch, ok := g.pending[permId]
	g.mu.Unlock()

	if !ok {
		return false
	}

	// If "always", add to permanent allow-list
	if resp.Response == "always" {
		// We'll check this before even asking next time
		// The tool name is not stored here; the caller tracks it
	}

	select {
	case ch <- resp:
		return true
	default:
		return false
	}
}

// wrapTool wraps a tool with a permission gate that blocks before execution.
func (g *permissionGate) wrapTool(tool fantasy.AgentTool, requireApproval bool) fantasy.AgentTool {
	if !requireApproval {
		return tool
	}

	info := tool.Info()

	return fantasy.NewAgentTool(
		info.Name,
		info.Description,
		func(ctx context.Context, input json.RawMessage, call fantasy.ToolCall) (fantasy.ToolResponse, error) {
			// Check if permanently allowed
			g.mu.Lock()
			allowed := g.alwaysAllowed[info.Name]
			g.mu.Unlock()

			if !allowed {
				permId := uuid.New().String()
				inputStr := string(input)

				description := "Tool " + info.Name + " wants to execute"

				// Get sessionId from context
				sessionId := ""
				if sid, ok := ctx.Value(sessionIdContextKey{}).(string); ok {
					sessionId = sid
				}

				resp := g.resolvePermission(ctx, permId, sessionId, info.Name, inputStr, description)

				switch resp.Response {
				case "deny":
					return fantasy.NewTextErrorResponse("Permission denied by user"), nil
				case "always":
					g.mu.Lock()
					g.alwaysAllowed[info.Name] = true
					g.mu.Unlock()
				case "once":
					// Continue with execution
				default:
					return fantasy.NewTextErrorResponse("Permission denied (unknown response)"), nil
				}
			}

			// Execute the original tool
			return tool.Run(ctx, call)
		},
	)
}

// sessionIdContextKey is the context key for storing session ID.
type sessionIdContextKey struct{}

// wrapTools wraps multiple tools, applying permission gates to the specified tool names.
func (g *permissionGate) wrapTools(tools []fantasy.AgentTool, requireApprovalFor []string) []fantasy.AgentTool {
	requireSet := make(map[string]bool)
	for _, name := range requireApprovalFor {
		requireSet[name] = true
	}

	wrapped := make([]fantasy.AgentTool, len(tools))
	for i, tool := range tools {
		wrapped[i] = g.wrapTool(tool, requireSet[tool.Info().Name])
	}
	return wrapped
}
