/*
Agent Runtime — Fantasy (Go)

Configuration types parsed from /etc/operator/config.json.
Shared contract with the operator's FantasyExtensionConfig.
*/
package main

// Config is the operator-generated configuration.
type Config struct {
	Runtime         string          `json:"runtime"`
	Providers       []ProviderEntry `json:"providers"`
	PrimaryProvider string          `json:"primaryProvider,omitempty"`
	PrimaryModel    string          `json:"primaryModel"`
	FallbackModels  []string        `json:"fallbackModels,omitempty"`
	TitleModel      string          `json:"titleModel,omitempty"` // fast/cheap model for auto-titling; defaults to primaryModel
	SystemPrompt    string          `json:"systemPrompt,omitempty"`
	BuiltinTools    []string        `json:"builtinTools,omitempty"`
	Tools           []ToolEntry     `json:"tools"`
	MCPServers      []MCPEntry      `json:"mcpServers,omitempty"`
	ToolHooks       *ToolHooksEntry `json:"toolHooks,omitempty"`
	ContextFiles    []ContextEntry  `json:"contextFiles,omitempty"`
	Resources       []ResourceEntry `json:"resources,omitempty"`
	Temperature     *float64        `json:"temperature,omitempty"`
	MaxOutputTokens *int64          `json:"maxOutputTokens,omitempty"`
	MaxSteps        *int            `json:"maxSteps,omitempty"`

	// Permission gate: list of tool names that require user approval before execution.
	// If empty, no tools require approval (all run automatically).
	PermissionTools []string `json:"permissionTools,omitempty"`

	// EnableQuestionTool adds a built-in "question" tool that lets the agent
	// ask the user interactive questions during execution.
	EnableQuestionTool bool `json:"enableQuestionTool,omitempty"`
}

// ProviderEntry describes a configured provider.
type ProviderEntry struct {
	Name string `json:"name"`
}

// ToolEntry describes a tool package path (MCP server in OCI).
type ToolEntry struct {
	Name string `json:"name"`
	Path string `json:"path"`
}

// MCPEntry describes an MCP server binding (via gateway sidecar).
type MCPEntry struct {
	Name        string   `json:"name"`
	Port        int      `json:"port"`
	DirectTools []string `json:"directTools,omitempty"`
}

// ToolHooksEntry holds runtime hook config.
type ToolHooksEntry struct {
	BlockedCommands []string `json:"blockedCommands,omitempty"`
	AllowedPaths    []string `json:"allowedPaths,omitempty"`
	AuditTools      []string `json:"auditTools,omitempty"`
}

// ContextEntry describes a context file.
type ContextEntry struct {
	Path string `json:"path"`
}

// ResourceEntry describes an AgentResource bound to this agent.
// Mirrors the operator's AgentResourceEntry from configmap.go.
type ResourceEntry struct {
	Name        string `json:"name"`
	Kind        string `json:"kind"`
	DisplayName string `json:"displayName"`
	Description string `json:"description,omitempty"`
	ReadOnly    bool   `json:"readOnly,omitempty"`
	AutoContext bool   `json:"autoContext,omitempty"`

	// Kind-specific config (one of these will be set)
	GitHub        *ResourceGitHubEntry        `json:"github,omitempty"`
	GitHubOrg     *ResourceGitHubOrgEntry     `json:"githubOrg,omitempty"`
	GitLab        *ResourceGitLabEntry        `json:"gitlab,omitempty"`
	GitLabGroup   *ResourceGitLabGroupEntry   `json:"gitlabGroup,omitempty"`
	Git           *ResourceGitEntry           `json:"git,omitempty"`
	MCP           *ResourceMCPEntry           `json:"mcp,omitempty"`
	S3            *ResourceS3Entry            `json:"s3,omitempty"`
	Documentation *ResourceDocumentationEntry `json:"documentation,omitempty"`
}

type ResourceGitHubEntry struct {
	Owner         string `json:"owner"`
	Repo          string `json:"repo"`
	DefaultBranch string `json:"defaultBranch,omitempty"`
	APIURL        string `json:"apiURL,omitempty"`
}

type ResourceGitHubOrgEntry struct {
	Org        string   `json:"org"`
	RepoFilter []string `json:"repoFilter,omitempty"`
	APIURL     string   `json:"apiURL,omitempty"`
}

type ResourceGitLabEntry struct {
	BaseURL       string `json:"baseURL"`
	Project       string `json:"project"`
	DefaultBranch string `json:"defaultBranch,omitempty"`
}

type ResourceGitLabGroupEntry struct {
	BaseURL  string   `json:"baseURL"`
	Group    string   `json:"group"`
	Projects []string `json:"projects,omitempty"`
}

type ResourceGitEntry struct {
	URL    string `json:"url"`
	Branch string `json:"branch,omitempty"`
}

type ResourceMCPEntry struct {
	URL       string            `json:"url"`
	Transport string            `json:"transport,omitempty"`
	Headers   map[string]string `json:"headers,omitempty"`
}

type ResourceS3Entry struct {
	Bucket   string `json:"bucket"`
	Region   string `json:"region,omitempty"`
	Endpoint string `json:"endpoint,omitempty"`
	Prefix   string `json:"prefix,omitempty"`
}

type ResourceDocumentationEntry struct {
	URLs []string `json:"urls,omitempty"`
}
