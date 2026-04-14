# AgentOps Runtime

[![CI](https://github.com/samyn92/agentops-runtime/actions/workflows/ci.yaml/badge.svg)](https://github.com/samyn92/agentops-runtime/actions/workflows/ci.yaml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Go](https://img.shields.io/badge/Go-1.26-00ADD8.svg)](https://go.dev/)
[![Fantasy SDK](https://img.shields.io/badge/Fantasy_SDK-0.17-purple.svg)](https://github.com/charmbracelet/fantasy)

Standalone Go binary that powers AI agent pods in [AgentOps](https://github.com/samyn92/agentops-core). Built on the [Charm Fantasy SDK](https://github.com/charmbracelet/fantasy) with a three-layer memory system backed by [agentops-memory](https://github.com/samyn92/agentops-memory), Kubernetes-native agent orchestration, MCP tool integration, and a streaming protocol (FEP) for the [AgentOps console](https://github.com/samyn92/agentops-console).

---

## Table of Contents

- [Architecture](#architecture)
- [Features](#features)
- [Modes](#modes)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [HTTP API](#http-api)
- [Fantasy Event Protocol (FEP)](#fantasy-event-protocol-fep)
- [Built-in Tools](#built-in-tools)
- [MCP Integration](#mcp-integration)
- [Memory](#memory)
- [Providers](#providers)
- [Agent Orchestration](#agent-orchestration)
- [Tool Security](#tool-security)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Building](#building)
- [CI/CD](#cicd)
- [Related Projects](#related-projects)
- [Contributing](#contributing)
- [License](#license)

---

## Architecture

```
+-----------------------------------------------------+
|  Agent Pod                                           |
|                                                      |
|  +------------------------------------------------+  |
|  |  agentops-runtime (this binary)                |  |
|  |                                                |  |
|  |  Fantasy SDK Agent                             |  |
|  |    +-- Provider (Anthropic/OpenAI/Google/...)   |  |
|  |    +-- Built-in tools (bash, read, edit, ...)   |  |
|  |    +-- OCI tools (kubectl, helm, ...)           |  |
|  |    +-- Permission gate                         |  |
|  |    +-- Question gate                           |  |
|  |    +-- Security hooks                          |  |
|  |                                                |  |
|  |  Memory                                        |  |
|  |    +-- Working memory (token-budget trimmed)    |  |
|  |    +-- Memory client (short + long term)       |  |
|  |                                                |  |
|  |  HTTP Server (:4096)                           |  |
|  |    +-- FEP SSE streaming                       |  |
|  +------------------------------------------------+  |
|                                                      |
|  +-------------+  +-----------------------------+    |
|  | MCP Gateway  |  | OCI Tool Sidecars           |    |
|  | (optional)   |  | (stdio MCP servers)         |    |
|  +-------------+  +-----------------------------+    |
+-----------------------------------------------------+
         |                          |
         | SSE/stdio                | HTTP REST
         v                          v
   MCPServer CRs              Memory Service
   (e.g. kubernetes)     (agentops-memory)
```

---

## Features

- **Charm Fantasy SDK** --- sole agent framework, with streaming callbacks, tool system, and multi-provider support
- **Two execution modes** --- long-running daemon (HTTP server) or one-shot task (Job)
- **14 tools** --- 8 built-in + 3 memory + 2 orchestration + 1 interactive question
- **Three-layer memory** --- working memory (token-budget trimmed) + agentops-memory short-term (session summaries) + agentops-memory long-term (explicit saves via FTS5 BM25)
- **MCP integration** --- OCI-packaged tools via stdio and gateway MCP servers via SSE
- **Multi-provider** --- Anthropic, OpenAI, Google/Gemini, OpenRouter, and any OpenAI-compatible endpoint
- **Automatic fallback** --- cycles through fallback models on retryable errors (429, 5xx, rate limits)
- **FEP streaming** --- 21 event types over SSE for real-time console integration
- **Permission gates** --- user approval before tool execution with once/always/deny responses
- **Interactive questions** --- agents can ask structured questions with selectable options
- **Tool security hooks** --- blocked commands, allowed path restrictions, audit logging
- **Agent orchestration** --- agents can spawn and monitor sub-agent runs via Kubernetes CRDs
- **Resource context injection** --- per-turn context from git forges, K8s, and documentation
- **AI-assisted memory extraction** --- extracts structured observations from conversations

---

## Modes

### Daemon

Long-running HTTP server for `Deployment`-backed agents. Serves the FEP streaming protocol on port `4096`, maintains conversation state in working memory, persists knowledge to agentops-memory.

```sh
agentops-runtime daemon
```

### Task

One-shot execution for `Job`-backed agents. Reads `AGENT_PROMPT` from environment, runs the agent, writes JSON result to stdout, then exits.

```sh
AGENT_PROMPT="List all files in /workspace" agentops-runtime task
```

Task output format:

```json
{
  "output": "...",
  "model": "anthropic/claude-sonnet-4-20250514",
  "usage": { "input_tokens": 1234, "output_tokens": 567 }
}
```

---

## Prerequisites

- Kubernetes cluster with [agentops-core](https://github.com/samyn92/agentops-core) operator installed
- Agent CRD deployed (the operator generates `/etc/operator/config.json` and manages the pod)
- At least one LLM provider API key configured as a K8s Secret
- **For development:**
  - Go **1.26**

---

## Quick Start

The runtime is not typically run directly --- the AgentOps operator creates and manages agent pods. But for development/testing:

```sh
# Build
CGO_ENABLED=0 go build -o agentops-runtime .

# Run (requires /etc/operator/config.json and API key env vars)
ANTHROPIC_API_KEY=sk-... AGENT_NAME=test-agent ./agentops-runtime daemon
```

Or via Docker:

```sh
docker build -t agentops-runtime .
docker run -p 4096:4096 \
  -e ANTHROPIC_API_KEY=sk-... \
  -e AGENT_NAME=test-agent \
  -v /path/to/config.json:/etc/operator/config.json:ro \
  agentops-runtime
```

---

## HTTP API

All routes served on port **4096**. The streaming endpoint speaks FEP over SSE.

### Conversation

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/prompt` | Non-streaming prompt. Returns `{"output":"...","model":"..."}`. Returns 429 if busy. |
| `POST` | `/prompt/stream` | Streaming prompt via FEP SSE. Returns `text/event-stream`. |
| `POST` | `/steer` | Inject steering message mid-execution (`[STEER]` prefix at next step). |
| `DELETE` | `/abort` | Cancel the running generation. |

### Interactive Control

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/permission/{pid}/reply` | Reply to permission gate: `{"response":"once\|always\|deny"}` |
| `POST` | `/question/{qid}/reply` | Reply to agent question: `{"answers":[["label1"],["label2"]]}` |

### Memory & Working Memory

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/working-memory` | Get serialized working memory messages. |
| `DELETE` | `/working-memory` | Clear all messages and reset turn counter. |
| `POST` | `/memory/extract` | AI-assisted memory extraction from working memory. |

### Health & Status

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/healthz` | Health probe: `{"status":"ok"}` |
| `GET` | `/status` | Runtime status: model, steps, busy, messages, turns, memory enabled. |

### Request/Response Examples

**Streaming prompt:**

```sh
curl -N -X POST http://localhost:4096/prompt/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "List files in the workspace", "context": []}'
```

**Permission reply:**

```sh
curl -X POST http://localhost:4096/permission/abc-123/reply \
  -H "Content-Type: application/json" \
  -d '{"response": "once"}'
```

---

## Fantasy Event Protocol (FEP)

FEP is the streaming protocol used to communicate with the AgentOps console over SSE. Each event is a JSON `data:` line:

```
data: {"type":"text_delta","timestamp":"2026-04-09T10:00:00Z","id":"abc","delta":"Hello"}
```

### Event Types (21 total)

| Event | Key Fields | Description |
|-------|------------|-------------|
| `agent_start` | session_id, prompt | Execution started |
| `agent_finish` | session_id, total_usage, step_count, model | Execution complete |
| `agent_error` | session_id, error, retryable | Execution failed |
| `step_start` | step_number, session_id | Step began |
| `step_finish` | step_number, usage, finish_reason, tool_call_count | Step complete |
| `text_start` | id | Text block started |
| `text_delta` | id, delta | Incremental text token |
| `text_end` | id | Text block complete |
| `reasoning_start` | id | Reasoning/thinking started |
| `reasoning_delta` | id, delta | Incremental reasoning token |
| `reasoning_end` | id | Reasoning complete |
| `tool_input_start` | id, tool_name | Tool input streaming started |
| `tool_input_delta` | id, delta | Incremental tool input JSON |
| `tool_input_end` | id | Tool input complete |
| `tool_call` | id, tool_name, input, provider_executed | Tool invoked |
| `tool_result` | id, tool_name, output, is_error, metadata | Tool result (with UI hints) |
| `source` | source_type, url, title | Citation/source reference |
| `warnings` | warnings | Provider warnings |
| `stream_finish` | usage, finish_reason | Per-step stream complete |
| `permission_asked` | id, tool_name, input, description | Permission gate triggered |
| `question_asked` | id, questions | Agent asking user a question |
| `session_idle` | session_id | Agent idle, ready for next prompt |

### Usage Tracking

Every `step_finish` and `agent_finish` event includes token usage:

```json
{
  "input_tokens": 1234,
  "output_tokens": 567,
  "total_tokens": 1801,
  "reasoning_tokens": 200,
  "cache_creation_tokens": 0,
  "cache_read_tokens": 800
}
```

---

## Built-in Tools

Eight tools available out of the box, selectable via `spec.builtinTools` on the Agent CRD:

| Tool | Description | UI Hint | Details |
|------|-------------|---------|---------|
| `bash` | Execute shell commands | `terminal` | 120s default timeout, reports exit code, CWD, duration |
| `read` | Read file contents | `code` | Supports `offset` (1-indexed) and `limit` for partial reads |
| `edit` | Exact find-and-replace | `diff` | Requires unique match, rejects ambiguous (>1 occurrence) |
| `write` | Write/create files | `file-created` | Auto-creates parent directories |
| `grep` | Regex search | `search-results` | Uses ripgrep (`rg --line-number --no-heading`) |
| `ls` | List directory contents | `file-tree` | Shows mode, size, name |
| `glob` | Find files by pattern | `file-tree` | Recursive walk with `filepath.Match` |
| `fetch` | Fetch URL content | `web-fetch` | Uses `curl -sSL --max-time 30` |

All tools emit `ui` metadata hints consumed by the console to render specialized tool cards (terminal output, syntax-highlighted code, unified diffs, file trees, search results).

---

## MCP Integration

Two transport modes for external tools:

### OCI Tools (stdio)

Tool packages pulled as OCI artifacts by the operator's crane init container. Each tool directory contains:
- `manifest.json` --- name, command, transport type
- `bin/` --- the MCP server binary

Started as stdio subprocesses via `mcp.CommandTransport`. Tools discovered via `session.ListTools()` and prefixed as `mcp_{server}_{tool}`.

### Gateway MCP (SSE)

Shared MCP servers (AgentTool CRs with `mcpServer` source) accessed through the MCP gateway sidecar. Connected via SSE to `http://localhost:{port}/sse` on the per-server port.

### UI Hints

MCP tools automatically receive UI hints for the console:
- `kubernetes-resources` for kubectl/kubernetes/k8s tools
- `helm-release` for helm tools

---

## Memory

Three-layer memory system replacing unbounded session replay:

| Layer | Storage | Survives Restart | Managed By |
|-------|---------|:----------------:|------------|
| **Working** | Go memory (token-budget trimmed) | No | Automatic |
| **Short-term** | agentops-memory SQLite (PVC) | Yes | Automatic (session summaries) |
| **Long-term** | agentops-memory SQLite (PVC) | Yes | User + agent (explicit saves via `mem_save`) |

### Working Memory

Token-budget trimmed message history. Before each LLM call, the runtime trims the oldest messages to fit within the conversation token budget (derived from the model's context window). Trims at user/assistant message boundaries to avoid orphaning tool results.

### Memory Service Integration

Persistent memory via agentops-memory's HTTP REST API:

| Operation | Endpoint | When |
|-----------|---------|------|
| Create session | `POST /sessions` | On runtime startup |
| Fetch context | `GET /context?project=X&limit=N&query=PROMPT` | Before each prompt (relevance-ranked injection) |
| Save observation | `POST /observations` | Via `mem_save` tool |
| Search memories | `GET /search?q=X&project=Y` | Via `mem_search` tool |
| End session | `POST /sessions/{id}/end` | On graceful shutdown |

### Memory Tools

Three tools added when agentops-memory is configured:

| Tool | Description |
|------|-------------|
| `mem_save` | Save an observation (type, title, content, tags) to long-term memory |
| `mem_search` | Full-text search across all memories (FTS5) |
| `mem_context` | Retrieve recent memory context (observations + session summaries) |

### Configuration

Via the Agent CRD `spec.memory`:

```yaml
spec:
  memory:
    serverRef: agentops-memory  # Service name or AgentTool CR
    project: my-agent        # Memory scope (defaults to agent name)
    contextLimit: 5          # Max context items per turn
    autoSummarize: true      # Auto-capture session summaries
```

---

## Providers

Supports any LLM provider through the Fantasy SDK provider system:

| Provider | Env Var | SDK Package |
|----------|---------|-------------|
| `anthropic` | `ANTHROPIC_API_KEY` | `fantasy/providers/anthropic` |
| `openai` | `OPENAI_API_KEY` | `fantasy/providers/openai` |
| `google` / `gemini` | `GOOGLE_API_KEY` | `fantasy/providers/google` |
| `openrouter` | `OPENROUTER_API_KEY` | `fantasy/providers/openrouter` |
| Custom | `{NAME}_API_KEY` + `{NAME}_BASE_URL` | `fantasy/providers/openaicompat` |

### Model Format

Models are specified as `provider/model-id` (e.g., `anthropic/claude-sonnet-4-20250514`). If no provider prefix is given, `primaryProvider` from config is used.

### Fallback

When the primary model fails with a retryable error (HTTP 429, 500, 502, 503, rate limit, overloaded), the runtime automatically tries each model in `fallbackModels` in order.

---

## Agent Orchestration

Two built-in tools allow agents to delegate work to other agents:

| Tool | Description |
|------|-------------|
| `run_agent` | Creates an `AgentRun` CR to trigger another agent with a prompt |
| `get_agent_run` | Checks the status and output of a running/completed agent run |

Uses the Kubernetes dynamic client with in-cluster credentials. When running outside a cluster, stubs are provided that return informational messages.

AgentRun names follow the pattern `{agent}-run-{unixMilli}`.

---

## Tool Security

### Permission Gates

Tools listed in `spec.permissionTools` require user approval before execution:

1. Runtime emits `permission_asked` FEP event with tool name, input, and description
2. Blocks on a Go channel (5-minute timeout)
3. Console delivers reply via `POST /permission/{pid}/reply`
4. Responses: `once` (allow this call), `always` (permanently allow), `deny` (block)
5. Timeout defaults to `deny`

### Security Hooks

`spec.toolHooks` provides three layers of protection:

| Hook | Description |
|------|-------------|
| **blockedCommands** | Reject bash commands containing these string patterns |
| **allowedPaths** | Restrict file tools to these path prefixes |
| **auditTools** | Log execution of these tools with structured audit entries |

### Tool Wrapping Chain

Tools pass through multiple wrapping layers in order:

1. **Build** --- built-in, OCI, gateway, memory, orchestration, question tools
2. **Security hooks** --- blocked commands, allowed paths, audit logging
3. **Permission gates** --- user approval for specified tools

---

## Configuration

### Primary Config

The runtime reads `/etc/operator/config.json`, generated by the AgentOps operator from the Agent CRD spec. Example:

```json
{
  "runtime": "fantasy",
  "primaryModel": "anthropic/claude-sonnet-4-20250514",
  "primaryProvider": "anthropic",
  "providers": [{ "name": "anthropic" }],
  "fallbackModels": ["openai/gpt-4o"],
  "systemPrompt": "You are a helpful assistant.",
  "builtinTools": ["bash", "read", "edit", "write", "grep", "ls", "glob"],
  "tools": [{ "name": "k8s-helper", "path": "/tools/k8s-helper" }],
  "mcpServers": [{ "name": "gitlab-mcp", "port": 8080 }],
  "toolHooks": {
    "blockedCommands": ["rm -rf /"],
    "allowedPaths": ["/workspace", "/data"],
    "auditTools": ["bash"]
  },
  "permissionTools": ["bash"],
  "enableQuestionTool": true,
  "temperature": 0.3,
  "maxOutputTokens": 8192,
  "maxSteps": 50,
  "memory": {
    "serverURL": "http://agentops-memory.agents.svc:7437",
    "project": "my-agent",
    "contextLimit": 5,
    "autoSummarize": true
  }
}
```

### Environment Variables

| Variable | Required | Description |
|----------|:--------:|-------------|
| `AGENT_NAME` | Yes | Agent identity, used for memory project scoping |
| `AGENT_NAMESPACE` | No | Kubernetes namespace for AgentRun CRs (default: `default`) |
| `AGENT_PROMPT` | Task only | The prompt for one-shot task mode |
| `ANTHROPIC_API_KEY` | Per provider | Anthropic API key |
| `OPENAI_API_KEY` | Per provider | OpenAI API key |
| `GOOGLE_API_KEY` | Per provider | Google/Gemini API key |
| `OPENROUTER_API_KEY` | Per provider | OpenRouter API key |
| `{NAME}_API_KEY` | Per provider | Custom provider API key |
| `{NAME}_BASE_URL` | Custom only | OpenAI-compatible base URL |

### Constants

| Constant | Value | Description |
|----------|-------|-------------|
| Config path | `/etc/operator/config.json` | Operator-generated config |
| HTTP port | `4096` | Server listen port |
| Default context limit | `5` | Memory context items per turn |
| Permission timeout | `5 min` | Time to wait for user approval |
| Question timeout | `10 min` | Time to wait for user answer |
| Bash timeout | `120s` | Default command timeout |
| Fetch timeout | `30s` | curl max-time |

---

## Project Structure

```
agentops-runtime/
  main.go             # Entry point, HTTP server, daemon/task modes, agent bundle,
                      #   fallback logic, orchestration tools, streaming handler
  config.go           # Config types parsed from /etc/operator/config.json
  fep.go              # FEP SSE emitter — 21 event type methods
  tools.go            # 8 built-in tools (bash, read, edit, write, grep, ls, glob, fetch)
  memory.go           # Working memory (token-budget trimmed) + memory service client + 3 memory tools
  mcp.go              # MCP tool loading — OCI (stdio) + gateway (SSE) + tool adapter
  permission.go       # Permission gate — user approval before tool execution
  question.go         # Interactive question tool
  hooks.go            # Tool security hooks — blocked commands, allowed paths, audit
  provider.go         # LLM provider resolution (Anthropic, OpenAI, Google, etc.)
  k8s.go              # Kubernetes dynamic client for AgentRun CRs
  resources.go        # Per-turn resource context formatting
  Dockerfile          # Multi-stage build: golang:1.26 -> alpine:3.21
  go.mod              # 5 direct dependencies (Fantasy SDK, MCP SDK, K8s, UUID)
```

Single `main` package --- 12 source files, ~3,900 lines of Go.

---

## Building

### Binary

```sh
CGO_ENABLED=0 go build -o agentops-runtime .
```

### Container Image

```sh
docker build -t agentops-runtime .
```

The image is based on `alpine:3.21` with `bash`, `curl`, and `ripgrep` installed for the built-in tools. Runs as non-root user `1000:1000`. Creates `/data/sessions`, `/data/repos`, `/data/scratch` directories.

### Published Images

```
ghcr.io/samyn92/agentops-runtime-fantasy:<version>
ghcr.io/samyn92/agentops-runtime-fantasy:latest
```

---

## CI/CD

### CI (`ci.yaml`)

Runs on push to `main` and pull requests:

- `CGO_ENABLED=0 go build` --- compile check
- `go vet ./...` --- static analysis

### Release (`release.yaml`)

Triggered by `v*` tags:

1. Build and vet (same as CI)
2. Build and push Docker image to GHCR with version tag + `latest`
3. Create GitHub Release with auto-generated notes

---

## Related Projects

| Repository | Description |
|------------|-------------|
| [agentops-core](https://github.com/samyn92/agentops-core) | Kubernetes operator (CRDs, controllers, webhooks) |
| [agentops-console](https://github.com/samyn92/agentops-console) | Web console (Go BFF + SolidJS PWA) |
| [agent-channels](https://github.com/samyn92/agent-channels) | Channel bridge images (Telegram, Slack, GitLab, etc.) |
| [agent-tools](https://github.com/samyn92/agent-tools) | OCI tool/agent packaging CLI + tool packages |
| [agentops-memory](https://github.com/samyn92/agentops-memory) | Purpose-built memory service (SQLite + FTS5 BM25) |
| [Charm Fantasy SDK](https://github.com/charmbracelet/fantasy) | AI agent framework |

---

## Contributing

Contributions are welcome. To get started:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Install Go 1.26 and run `go mod download`
4. Make your changes
5. Ensure the build passes (`CGO_ENABLED=0 go build -o agentops-runtime .`)
6. Run static analysis (`go vet ./...`)
7. Commit your changes and open a Pull Request

---

## License

Apache 2.0
