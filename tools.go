/*
Agent Runtime — Fantasy (Go)

Built-in tools: bash, read, edit, write, grep, ls, glob, fetch.
Each implements the fantasy.AgentTool interface.
*/
package main

import (
	"context"
	"fmt"
	"io/fs"
	"net"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"charm.land/fantasy"
)

// buildBuiltinTools returns the requested built-in tools.
// If names is empty, all built-in tools are returned (operator may omit the list).
func buildBuiltinTools(names []string) []fantasy.AgentTool {
	registry := map[string]fantasy.AgentTool{
		"bash":  newBashTool(),
		"read":  newReadTool(),
		"edit":  newEditTool(),
		"write": newWriteTool(),
		"grep":  newGrepTool(),
		"ls":    newLsTool(),
		"glob":  newGlobTool(),
		"fetch": newFetchTool(),
	}

	// Default to all tools when operator doesn't pass the list
	if len(names) == 0 {
		tools := make([]fantasy.AgentTool, 0, len(registry))
		for _, t := range registry {
			tools = append(tools, t)
		}
		return tools
	}

	var tools []fantasy.AgentTool
	for _, name := range names {
		if t, ok := registry[name]; ok {
			tools = append(tools, t)
		}
	}
	return tools
}

// ── bash ──

type bashInput struct {
	Command string `json:"command" description:"The bash command to execute"`
	Timeout int    `json:"timeout,omitempty" description:"Timeout in seconds (default: 120)"`
}

func newBashTool() fantasy.AgentTool {
	return fantasy.NewAgentTool("bash",
		"Execute a bash command. Returns stdout and stderr. Use for running programs, installing packages, file operations, and system tasks.",
		func(ctx context.Context, input bashInput, _ fantasy.ToolCall) (fantasy.ToolResponse, error) {
			if input.Command == "" {
				return fantasy.NewTextErrorResponse("command is required"), nil
			}

			cwd, _ := os.Getwd()
			start := time.Now()

			cmd := exec.CommandContext(ctx, "bash", "-c", input.Command)
			out, err := cmd.CombinedOutput()

			elapsed := time.Since(start)

			exitCode := 0
			if cmd.ProcessState != nil {
				exitCode = cmd.ProcessState.ExitCode()
			}

			metadata := map[string]any{
				"ui":       "terminal",
				"command":  input.Command,
				"exitCode": exitCode,
				"cwd":      cwd,
				"duration": elapsed.Milliseconds(),
			}

			if err != nil {
				result := fantasy.NewTextErrorResponse(fmt.Sprintf("%s\n%s", string(out), err.Error()))
				return fantasy.WithResponseMetadata(result, metadata), nil
			}
			result := fantasy.NewTextResponse(string(out))
			return fantasy.WithResponseMetadata(result, metadata), nil
		})
}

// ── read ──

type readInput struct {
	Path   string `json:"path" description:"Path to the file to read"`
	Offset int    `json:"offset,omitempty" description:"Line number to start reading from (1-indexed)"`
	Limit  int    `json:"limit,omitempty" description:"Maximum number of lines to read"`
}

func newReadTool() fantasy.AgentTool {
	return fantasy.NewAgentTool("read",
		"Read the contents of a file. Supports text files. Use offset/limit for large files.",
		func(_ context.Context, input readInput, _ fantasy.ToolCall) (fantasy.ToolResponse, error) {
			if input.Path == "" {
				return fantasy.NewTextErrorResponse("path is required"), nil
			}
			data, err := os.ReadFile(input.Path)
			if err != nil {
				return fantasy.NewTextErrorResponse(err.Error()), nil
			}
			content := string(data)
			lines := strings.Split(content, "\n")
			lineCount := len(lines)

			if input.Offset > 0 || input.Limit > 0 {
				start := 0
				if input.Offset > 0 {
					start = input.Offset - 1
				}
				if start >= len(lines) {
					metadata := map[string]any{
						"ui":        "code",
						"filePath":  input.Path,
						"offset":    input.Offset,
						"limit":     input.Limit,
						"language":  detectLanguage(input.Path),
						"lineCount": 0,
					}
					result := fantasy.NewTextResponse("")
					return fantasy.WithResponseMetadata(result, metadata), nil
				}
				end := len(lines)
				if input.Limit > 0 && start+input.Limit < end {
					end = start + input.Limit
				}
				lines = lines[start:end]
				lineCount = len(lines)
				content = strings.Join(lines, "\n")
			}

			metadata := map[string]any{
				"ui":        "code",
				"filePath":  input.Path,
				"offset":    input.Offset,
				"limit":     input.Limit,
				"language":  detectLanguage(input.Path),
				"lineCount": lineCount,
			}

			result := fantasy.NewTextResponse(content)
			return fantasy.WithResponseMetadata(result, metadata), nil
		})
}

// ── edit ──

type editEntry struct {
	OldText string `json:"oldText" description:"Exact text to find and replace"`
	NewText string `json:"newText" description:"Replacement text"`
}

type editInput struct {
	Path  string      `json:"path" description:"Path to the file to edit"`
	Edits []editEntry `json:"edits" description:"List of exact text replacements"`
}

func newEditTool() fantasy.AgentTool {
	return fantasy.NewAgentTool("edit",
		"Edit a file using exact text replacement. Each edit's oldText must match a unique region of the file.",
		func(_ context.Context, input editInput, _ fantasy.ToolCall) (fantasy.ToolResponse, error) {
			if input.Path == "" {
				return fantasy.NewTextErrorResponse("path is required"), nil
			}
			data, err := os.ReadFile(input.Path)
			if err != nil {
				return fantasy.NewTextErrorResponse(err.Error()), nil
			}
			content := string(data)
			for _, e := range input.Edits {
				if !strings.Contains(content, e.OldText) {
					return fantasy.NewTextErrorResponse(fmt.Sprintf("oldText not found in file: %q", truncate(e.OldText, 80))), nil
				}
				if strings.Count(content, e.OldText) > 1 {
					return fantasy.NewTextErrorResponse(fmt.Sprintf("oldText matches multiple locations: %q", truncate(e.OldText, 80))), nil
				}
				content = strings.Replace(content, e.OldText, e.NewText, 1)
			}
			if err := os.WriteFile(input.Path, []byte(content), 0644); err != nil {
				return fantasy.NewTextErrorResponse(err.Error()), nil
			}

			metadata := map[string]any{
				"ui":        "diff",
				"filePath":  input.Path,
				"editCount": len(input.Edits),
			}

			result := fantasy.NewTextResponse(fmt.Sprintf("Applied %d edit(s) to %s", len(input.Edits), input.Path))
			return fantasy.WithResponseMetadata(result, metadata), nil
		})
}

// ── write ──

type writeInput struct {
	Path    string `json:"path" description:"Path to the file to write"`
	Content string `json:"content" description:"Content to write to the file"`
}

func newWriteTool() fantasy.AgentTool {
	return fantasy.NewAgentTool("write",
		"Write content to a file. Creates the file if it doesn't exist, overwrites if it does. Automatically creates parent directories.",
		func(_ context.Context, input writeInput, _ fantasy.ToolCall) (fantasy.ToolResponse, error) {
			if input.Path == "" {
				return fantasy.NewTextErrorResponse("path is required"), nil
			}
			if err := os.MkdirAll(filepath.Dir(input.Path), 0755); err != nil {
				return fantasy.NewTextErrorResponse(err.Error()), nil
			}
			if err := os.WriteFile(input.Path, []byte(input.Content), 0644); err != nil {
				return fantasy.NewTextErrorResponse(err.Error()), nil
			}

			metadata := map[string]any{
				"ui":       "file-created",
				"filePath": input.Path,
				"size":     len(input.Content),
				"language": detectLanguage(input.Path),
			}

			result := fantasy.NewTextResponse(fmt.Sprintf("Wrote %d bytes to %s", len(input.Content), input.Path))
			return fantasy.WithResponseMetadata(result, metadata), nil
		})
}

// ── grep ──

type grepInput struct {
	Pattern string `json:"pattern" description:"Regex pattern to search for"`
	Path    string `json:"path" description:"File or directory to search"`
}

func newGrepTool() fantasy.AgentTool {
	return fantasy.NewAgentTool("grep",
		"Search for a regex pattern in files using ripgrep (rg). Returns matching lines with file paths and line numbers.",
		func(ctx context.Context, input grepInput, _ fantasy.ToolCall) (fantasy.ToolResponse, error) {
			if input.Pattern == "" {
				return fantasy.NewTextErrorResponse("pattern is required"), nil
			}
			path := input.Path
			if path == "" {
				path = "."
			}
			cmd := exec.CommandContext(ctx, "rg", "--line-number", "--no-heading", input.Pattern, path)
			out, _ := cmd.CombinedOutput()

			output := string(out)
			matchCount := 0
			if len(out) > 0 {
				matchCount = strings.Count(output, "\n")
				if len(output) > 0 && !strings.HasSuffix(output, "\n") {
					matchCount++
				}
			}

			metadata := map[string]any{
				"ui":         "search-results",
				"pattern":    input.Pattern,
				"path":       path,
				"matchCount": matchCount,
			}

			if len(out) == 0 {
				result := fantasy.NewTextResponse("No matches found.")
				return fantasy.WithResponseMetadata(result, metadata), nil
			}
			result := fantasy.NewTextResponse(output)
			return fantasy.WithResponseMetadata(result, metadata), nil
		})
}

// ── ls ──

type lsInput struct {
	Path string `json:"path" description:"Directory path to list (defaults to current directory)"`
}

func newLsTool() fantasy.AgentTool {
	return fantasy.NewAgentTool("ls",
		"List directory contents with file types and sizes.",
		func(_ context.Context, input lsInput, _ fantasy.ToolCall) (fantasy.ToolResponse, error) {
			path := input.Path
			if path == "" {
				path = "."
			}
			entries, err := os.ReadDir(path)
			if err != nil {
				return fantasy.NewTextErrorResponse(err.Error()), nil
			}
			var sb strings.Builder
			for _, e := range entries {
				info, _ := e.Info()
				if info != nil {
					fmt.Fprintf(&sb, "%s %8d %s\n", info.Mode(), info.Size(), e.Name())
				} else {
					fmt.Fprintf(&sb, "%s\n", e.Name())
				}
			}

			metadata := map[string]any{
				"ui":         "file-tree",
				"path":       path,
				"entryCount": len(entries),
			}

			result := fantasy.NewTextResponse(sb.String())
			return fantasy.WithResponseMetadata(result, metadata), nil
		})
}

// ── glob ──

type globInput struct {
	Pattern string `json:"pattern" description:"Glob pattern to match files (e.g. **/*.go)"`
	Path    string `json:"path,omitempty" description:"Root directory to search from (defaults to current directory)"`
}

func newGlobTool() fantasy.AgentTool {
	return fantasy.NewAgentTool("glob",
		"Find files matching a glob pattern. Returns matching file paths.",
		func(_ context.Context, input globInput, _ fantasy.ToolCall) (fantasy.ToolResponse, error) {
			if input.Pattern == "" {
				return fantasy.NewTextErrorResponse("pattern is required"), nil
			}
			root := input.Path
			if root == "" {
				root = "."
			}
			var matches []string
			err := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
				if err != nil {
					return nil
				}
				matched, _ := filepath.Match(input.Pattern, filepath.Base(path))
				if matched {
					matches = append(matches, path)
				}
				return nil
			})
			if err != nil {
				return fantasy.NewTextErrorResponse(err.Error()), nil
			}

			metadata := map[string]any{
				"ui":      "file-tree",
				"pattern": input.Pattern,
				"path":    root,
				"count":   len(matches),
			}

			if len(matches) == 0 {
				result := fantasy.NewTextResponse("No files matched.")
				return fantasy.WithResponseMetadata(result, metadata), nil
			}
			result := fantasy.NewTextResponse(strings.Join(matches, "\n"))
			return fantasy.WithResponseMetadata(result, metadata), nil
		})
}

// ── fetch ──

type fetchInput struct {
	URL string `json:"url" description:"URL to fetch"`
}

func newFetchTool() fantasy.AgentTool {
	return fantasy.NewAgentTool("fetch",
		"Fetch the contents of a URL. Returns the response body as text.",
		func(ctx context.Context, input fetchInput, _ fantasy.ToolCall) (fantasy.ToolResponse, error) {
			if input.URL == "" {
				return fantasy.NewTextErrorResponse("url is required"), nil
			}
			if err := validateFetchURL(input.URL); err != nil {
				return fantasy.NewTextErrorResponse(fmt.Sprintf("blocked: %s", err)), nil
			}
			cmd := exec.CommandContext(ctx, "curl", "-sSL", "--max-time", "30", input.URL)
			out, err := cmd.CombinedOutput()

			metadata := map[string]any{
				"ui":  "web-fetch",
				"url": input.URL,
			}

			if err != nil {
				result := fantasy.NewTextErrorResponse(fmt.Sprintf("%s\n%s", string(out), err.Error()))
				return fantasy.WithResponseMetadata(result, metadata), nil
			}
			result := fantasy.NewTextResponse(string(out))
			return fantasy.WithResponseMetadata(result, metadata), nil
		})
}

// validateFetchURL blocks requests to internal/metadata endpoints to prevent SSRF.
func validateFetchURL(rawURL string) error {
	u, err := url.Parse(rawURL)
	if err != nil {
		return fmt.Errorf("invalid URL: %s", err)
	}

	// Only allow http/https
	if u.Scheme != "http" && u.Scheme != "https" {
		return fmt.Errorf("scheme %q not allowed, only http/https", u.Scheme)
	}

	host := u.Hostname()

	// Block cloud metadata endpoints
	metadataHosts := []string{
		"169.254.169.254", // AWS/GCP/Azure metadata
		"metadata.google.internal",
		"metadata.internal",
	}
	for _, mh := range metadataHosts {
		if strings.EqualFold(host, mh) {
			return fmt.Errorf("access to metadata endpoint %q is blocked", host)
		}
	}

	// Resolve and block private/loopback IPs
	ips, err := net.LookupHost(host)
	if err != nil {
		// If we can't resolve, let curl try (it'll fail on its own)
		return nil
	}
	for _, ipStr := range ips {
		ip := net.ParseIP(ipStr)
		if ip == nil {
			continue
		}
		if ip.IsLoopback() || ip.IsPrivate() || ip.IsLinkLocalUnicast() || ip.IsLinkLocalMulticast() {
			return fmt.Errorf("access to internal address %s (%s) is blocked", host, ipStr)
		}
	}
	return nil
}

// ── helpers ──

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

// detectLanguage maps a file path's extension to a language identifier.
func detectLanguage(path string) string {
	base := strings.ToLower(filepath.Base(path))
	ext := strings.ToLower(filepath.Ext(path))

	// Handle extensionless filenames.
	if base == "dockerfile" {
		return "dockerfile"
	}

	switch ext {
	case ".go":
		return "go"
	case ".ts", ".tsx":
		return "typescript"
	case ".js", ".jsx":
		return "javascript"
	case ".py":
		return "python"
	case ".rs":
		return "rust"
	case ".rb":
		return "ruby"
	case ".java":
		return "java"
	case ".c", ".h":
		return "c"
	case ".cpp", ".hpp":
		return "cpp"
	case ".cs":
		return "csharp"
	case ".swift":
		return "swift"
	case ".kt":
		return "kotlin"
	case ".scala":
		return "scala"
	case ".sh", ".bash":
		return "bash"
	case ".yaml", ".yml":
		return "yaml"
	case ".json":
		return "json"
	case ".toml":
		return "toml"
	case ".xml":
		return "xml"
	case ".html":
		return "html"
	case ".css":
		return "css"
	case ".scss":
		return "scss"
	case ".sql":
		return "sql"
	case ".md":
		return "markdown"
	case ".dockerfile":
		return "dockerfile"
	case ".tf":
		return "hcl"
	case ".proto":
		return "protobuf"
	case ".graphql":
		return "graphql"
	default:
		return ""
	}
}
