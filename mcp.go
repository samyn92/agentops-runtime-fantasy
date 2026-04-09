/*
Agent Runtime — Fantasy (Go)

MCP tool loading: starts OCI-packaged MCP servers as subprocesses
and wraps their tools as fantasy.AgentTool via the MCP protocol.

Also loads tools from MCP gateway sidecars (shared MCPServers).
*/
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"charm.land/fantasy"
	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// ToolManifest describes how to start an MCP server from an OCI artifact.
type ToolManifest struct {
	Name      string `json:"name"`
	Command   string `json:"command"`
	Transport string `json:"transport"` // "stdio" (default) or "sse"
}

// mcpConnection holds a client session and its cleanup function.
type mcpConnection struct {
	session *mcp.ClientSession
	cleanup func()
}

// loadOCITools discovers and starts MCP servers from OCI-pulled tool directories.
func loadOCITools(ctx context.Context, toolRefs []ToolEntry) ([]fantasy.AgentTool, []mcpConnection, error) {
	var tools []fantasy.AgentTool
	var conns []mcpConnection

	for _, ref := range toolRefs {
		manifestPath := filepath.Join(ref.Path, "manifest.json")
		data, err := os.ReadFile(manifestPath)
		if err != nil {
			slog.Warn("no manifest.json in tool directory, skipping",
				"tool", ref.Name, "path", ref.Path, "error", err)
			continue
		}

		var manifest ToolManifest
		if err := json.Unmarshal(data, &manifest); err != nil {
			slog.Error("failed to parse manifest.json", "tool", ref.Name, "error", err)
			continue
		}

		if manifest.Command == "" {
			manifest.Command = ref.Name
		}

		binPath := filepath.Join(ref.Path, "bin", manifest.Command)
		if _, err := os.Stat(binPath); err != nil {
			binPath = filepath.Join(ref.Path, manifest.Command)
			if _, err := os.Stat(binPath); err != nil {
				slog.Error("MCP server binary not found", "tool", ref.Name, "tried", binPath)
				continue
			}
		}

		conn, err := startStdioMCP(ctx, ref.Name, binPath)
		if err != nil {
			slog.Error("failed to start MCP server", "tool", ref.Name, "error", err)
			continue
		}
		conns = append(conns, *conn)

		mcpTools, err := discoverMCPTools(ctx, conn.session, ref.Name, "stdio", ref.UIHint)
		if err != nil {
			slog.Error("failed to discover MCP tools", "tool", ref.Name, "error", err)
			continue
		}

		tools = append(tools, mcpTools...)
		slog.Info("loaded MCP tools from OCI", "tool", ref.Name, "count", len(mcpTools))
	}

	return tools, conns, nil
}

// loadGatewayMCPTools connects to MCP gateway sidecars and discovers tools.
func loadGatewayMCPTools(ctx context.Context, mcpServers []MCPEntry) ([]fantasy.AgentTool, []mcpConnection, error) {
	var tools []fantasy.AgentTool
	var conns []mcpConnection

	for _, srv := range mcpServers {
		sseURL := fmt.Sprintf("http://localhost:%d/sse", srv.Port)

		conn, err := startSSEMCP(ctx, srv.Name, sseURL)
		if err != nil {
			slog.Error("failed to connect to MCP gateway", "server", srv.Name, "error", err)
			continue
		}
		conns = append(conns, *conn)

		mcpTools, err := discoverMCPTools(ctx, conn.session, srv.Name, "sse", srv.UIHint)
		if err != nil {
			slog.Error("failed to discover MCP tools from gateway", "server", srv.Name, "error", err)
			continue
		}

		tools = append(tools, mcpTools...)
		slog.Info("loaded MCP tools from gateway", "server", srv.Name, "count", len(mcpTools))
	}

	return tools, conns, nil
}

// startStdioMCP starts an MCP server process and connects via stdio.
func startStdioMCP(ctx context.Context, name, binPath string) (*mcpConnection, error) {
	impl := &mcp.Implementation{Name: "agentops-fantasy", Version: "0.1.0"}
	client := mcp.NewClient(impl, nil)

	transport := &mcp.CommandTransport{
		Command: exec.CommandContext(ctx, binPath),
	}

	session, err := client.Connect(ctx, transport, nil)
	if err != nil {
		return nil, fmt.Errorf("connect to MCP server: %w", err)
	}

	slog.Info("started MCP stdio server", "name", name, "bin", binPath)
	return &mcpConnection{
		session: session,
		cleanup: func() { session.Close() },
	}, nil
}

// startSSEMCP connects to an MCP server via SSE transport.
func startSSEMCP(ctx context.Context, name, sseURL string) (*mcpConnection, error) {
	impl := &mcp.Implementation{Name: "agentops-fantasy", Version: "0.1.0"}
	client := mcp.NewClient(impl, nil)

	transport := &mcp.SSEClientTransport{Endpoint: sseURL}

	session, err := client.Connect(ctx, transport, nil)
	if err != nil {
		return nil, fmt.Errorf("connect to SSE MCP server: %w", err)
	}

	slog.Info("connected to MCP SSE server", "name", name, "url", sseURL)
	return &mcpConnection{
		session: session,
		cleanup: func() { session.Close() },
	}, nil
}

// discoverMCPTools lists tools from an MCP session and wraps as fantasy.AgentTool.
func discoverMCPTools(ctx context.Context, session *mcp.ClientSession, serverName string, transport string, crdUIHint string) ([]fantasy.AgentTool, error) {
	result, err := session.ListTools(ctx, nil)
	if err != nil {
		return nil, fmt.Errorf("list tools: %w", err)
	}

	var tools []fantasy.AgentTool
	for _, t := range result.Tools {
		tools = append(tools, newMCPToolAdapter(session, serverName, t, transport, crdUIHint))
	}
	return tools, nil
}

// mcpToolAdapter wraps an MCP tool as a fantasy.AgentTool.
type mcpToolAdapter struct {
	session    *mcp.ClientSession
	serverName string
	mcpTool    *mcp.Tool
	transport  string // "stdio" (OCI/inline) or "sse" (gateway/CRD)
	crdUIHint  string // UI hint from the AgentTool CRD (overrides heuristic detection)
	opts       fantasy.ProviderOptions
}

func newMCPToolAdapter(session *mcp.ClientSession, serverName string, tool *mcp.Tool, transport string, crdUIHint string) *mcpToolAdapter {
	return &mcpToolAdapter{
		session:    session,
		serverName: serverName,
		mcpTool:    tool,
		transport:  transport,
		crdUIHint:  crdUIHint,
	}
}

func (m *mcpToolAdapter) Info() fantasy.ToolInfo {
	params := make(map[string]any)
	var required []string

	// Extract properties from the tool's input schema
	if m.mcpTool.InputSchema != nil {
		schemaBytes, _ := json.Marshal(m.mcpTool.InputSchema)
		var schema struct {
			Properties map[string]any `json:"properties"`
			Required   []string       `json:"required"`
		}
		json.Unmarshal(schemaBytes, &schema)
		params = schema.Properties
		required = schema.Required
	}

	return fantasy.ToolInfo{
		Name:        fmt.Sprintf("mcp_%s_%s", m.serverName, m.mcpTool.Name),
		Description: m.mcpTool.Description,
		Parameters:  params,
		Required:    required,
	}
}

func (m *mcpToolAdapter) Run(ctx context.Context, call fantasy.ToolCall) (fantasy.ToolResponse, error) {
	var args map[string]any
	if err := json.Unmarshal([]byte(call.Input), &args); err != nil {
		return fantasy.NewTextErrorResponse(fmt.Sprintf("invalid input: %s", err)), nil
	}

	start := time.Now()

	result, err := m.session.CallTool(ctx, &mcp.CallToolParams{
		Name:      m.mcpTool.Name,
		Arguments: args,
	})
	if err != nil {
		return fantasy.NewTextErrorResponse(fmt.Sprintf("MCP tool call failed: %s", err)), nil
	}

	elapsed := time.Since(start)

	// Collect text content from result
	var text string
	for _, content := range result.Content {
		if tc, ok := content.(*mcp.TextContent); ok {
			text += tc.Text
		}
	}

	// Build metadata for the frontend
	metadata := map[string]any{
		"server":    m.serverName,
		"tool":      m.mcpTool.Name,
		"transport": m.transport,
		"duration":  elapsed.Milliseconds(),
	}
	// Add tool description from the MCP server for the frontend to render
	if m.mcpTool.Description != "" {
		metadata["description"] = m.mcpTool.Description
	}
	// UI hint: prefer CRD-configured hint, fall back to heuristic detection
	if m.crdUIHint != "" {
		metadata["ui"] = m.crdUIHint
	} else if ui := detectMCPUIHint(m.serverName, m.mcpTool.Name); ui != "" {
		metadata["ui"] = ui
	}

	if result.IsError {
		resp := fantasy.NewTextErrorResponse(text)
		return fantasy.WithResponseMetadata(resp, metadata), nil
	}
	resp := fantasy.NewTextResponse(text)
	return fantasy.WithResponseMetadata(resp, metadata), nil
}

func (m *mcpToolAdapter) ProviderOptions() fantasy.ProviderOptions {
	return m.opts
}

func (m *mcpToolAdapter) SetProviderOptions(opts fantasy.ProviderOptions) {
	m.opts = opts
}

// detectMCPUIHint returns a UI renderer hint based on the MCP server/tool name.
// This allows the frontend to render branded cards for known MCP tool categories.
func detectMCPUIHint(serverName, toolName string) string {
	combined := strings.ToLower(serverName + " " + toolName)
	if strings.Contains(combined, "kubectl") || strings.Contains(combined, "kubernetes") || strings.Contains(combined, "k8s") {
		return "kubernetes-resources"
	}
	if strings.Contains(combined, "helm") {
		return "helm-release"
	}
	return ""
}

// shutdownMCPConnections gracefully closes all MCP connections.
func shutdownMCPConnections(conns []mcpConnection) {
	for _, c := range conns {
		if c.cleanup != nil {
			c.cleanup()
		}
	}
}
