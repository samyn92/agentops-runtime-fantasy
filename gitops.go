/*
Agent Runtime — Fantasy (Go)

Built-in git operations using go-git (pure Go, no git CLI needed).
Replaces the mcp-git sidecar chain with direct git API calls.
*/
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"time"

	"charm.land/fantasy"
	git "github.com/go-git/go-git/v5"
	"github.com/go-git/go-git/v5/config"
	"github.com/go-git/go-git/v5/plumbing"
	"github.com/go-git/go-git/v5/plumbing/object"
	githttp "github.com/go-git/go-git/v5/plumbing/transport/http"
)

// gitWorkspace holds the go-git repo and auth for the agent's lifetime.
type gitWorkspace struct {
	repo       *git.Repository
	dir        string
	auth       *githttp.BasicAuth
	branch     string
	baseBranch string
}

// Global workspace — set by setupGitWorkspace, used by git tools.
var workspace *gitWorkspace

// augmentSystemPromptWithGitContext appends a workspace context block to
// the system prompt when GIT_REPO_URL is set (i.e. the operator has provisioned
// a git workspace for this run).
//
// Why this exists:
// Without this hint, agents commonly invent paths like /tmp/<repo> via bash
// `git clone` and then call MCP `git_*` tools with cwd="/tmp/<repo>". Those
// tools sandbox to /data and reject anything else, so the agent ends up
// thrashing between bash git (which often fails on author identity / push
// auth) and rejected MCP calls.
//
// The block tells the LLM, in its system prompt, exactly:
//   - where the repo is pre-cloned (/data/repo)
//   - which branch it's on
//   - to prefer MCP `git_*` and `github_*`/`gitlab_*` tools over bash git
//   - that omitting `cwd` is the safest default
//
// Returns the prompt unchanged when GIT_REPO_URL is not set, so this is
// safe to call unconditionally.
func augmentSystemPromptWithGitContext(prompt string) string {
	repoURL := os.Getenv("GIT_REPO_URL")
	if repoURL == "" {
		return prompt
	}

	branch := os.Getenv("GIT_BRANCH")
	baseBranch := os.Getenv("GIT_BASE_BRANCH")
	provider := os.Getenv("GIT_PROVIDER")

	var b strings.Builder
	if prompt != "" {
		b.WriteString(prompt)
		b.WriteString("\n\n")
	}
	b.WriteString("## Git Workspace\n\n")
	b.WriteString(fmt.Sprintf("A git workspace is pre-cloned for you at `/data/repo`.\n"))
	b.WriteString(fmt.Sprintf("- Repository: `%s`\n", repoURL))
	if branch != "" {
		b.WriteString(fmt.Sprintf("- Working branch: `%s` (already checked out)\n", branch))
	}
	if baseBranch != "" {
		b.WriteString(fmt.Sprintf("- Base branch: `%s`\n", baseBranch))
	}
	b.WriteString("\n### Tool guidance\n\n")
	b.WriteString("- **Use MCP `git_*` tools** (e.g. `git_status`, `git_add`, `git_commit`, `git_push`) — NOT bash `git`. ")
	b.WriteString("Bash git lacks committer identity and credential helpers in this sandbox; the MCP tools handle both.\n")
	b.WriteString("- **Omit the `cwd` parameter** on MCP git tools. It defaults to the workspace root (`/data`), and the repo is at `/data/repo`. ")
	b.WriteString("Passing absolute paths outside `/data` (like `/tmp/...`) will be rejected.\n")
	if provider == "github" {
		b.WriteString("- **Use MCP `github_*` tools** to create pull requests, comment, etc. — NOT `gh` CLI via bash. ")
		b.WriteString("`GH_TOKEN` is wired into the MCP server, not into bash subprocesses.\n")
	} else if provider == "gitlab" {
		b.WriteString("- **Use MCP `gitlab_*` tools** to create merge requests, comment, etc. — NOT `glab` CLI via bash. ")
		b.WriteString("`GITLAB_TOKEN` is wired into the MCP server, not into bash subprocesses.\n")
	}
	b.WriteString("- Do not `git clone` again — the repo is already cloned and on the right branch.\n")
	return b.String()
}

// setupGitWorkspace clones a repo and checks out the feature branch using go-git.
func setupGitWorkspace(repoURL, branch, baseBranch string) error {
	repoDir := "/data/repo"
	token := os.Getenv("GIT_TOKEN")

	var auth *githttp.BasicAuth
	if token != "" {
		auth = &githttp.BasicAuth{
			Username: "oauth2",
			Password: token,
		}
	}

	if baseBranch == "" {
		baseBranch = "main"
	}

	slog.Info("cloning git repo", "url", repoURL, "baseBranch", baseBranch)

	repo, err := git.PlainClone(repoDir, false, &git.CloneOptions{
		URL:           repoURL,
		Auth:          auth,
		ReferenceName: plumbing.NewBranchReferenceName(baseBranch),
		Depth:         50,
		SingleBranch:  true,
	})
	if err != nil {
		return fmt.Errorf("git clone: %w", err)
	}

	// If a feature branch is requested and differs from base, create/checkout it
	if branch != "" && branch != baseBranch {
		w, err := repo.Worktree()
		if err != nil {
			return fmt.Errorf("worktree: %w", err)
		}

		// Try to fetch the remote branch first
		remoteRef := plumbing.NewRemoteReferenceName("origin", branch)
		err = repo.Fetch(&git.FetchOptions{
			Auth:     auth,
			RefSpecs: []config.RefSpec{config.RefSpec(fmt.Sprintf("+refs/heads/%s:refs/remotes/origin/%s", branch, branch))},
			Depth:    50,
		})

		if err == nil {
			// Remote branch exists — checkout tracking it
			ref, _ := repo.Reference(remoteRef, true)
			if ref != nil {
				err = w.Checkout(&git.CheckoutOptions{
					Branch: plumbing.NewBranchReferenceName(branch),
					Hash:   ref.Hash(),
					Create: true,
				})
				if err != nil {
					return fmt.Errorf("checkout remote branch %s: %w", branch, err)
				}
				slog.Info("checked out existing remote branch", "branch", branch)
			}
		} else {
			// Branch doesn't exist on remote — create from current HEAD
			slog.Info("creating new branch", "branch", branch)
			err = w.Checkout(&git.CheckoutOptions{
				Branch: plumbing.NewBranchReferenceName(branch),
				Create: true,
			})
			if err != nil {
				return fmt.Errorf("create branch %s: %w", branch, err)
			}
		}
	}

	workspace = &gitWorkspace{
		repo:       repo,
		dir:        repoDir,
		auth:       auth,
		branch:     branch,
		baseBranch: baseBranch,
	}

	slog.Info("git workspace ready", "dir", repoDir, "branch", branch)
	return nil
}

// extractGitInfo reads the git state after the agent completes.
func extractGitInfo() (commits int, prURL string) {
	if workspace == nil || workspace.repo == nil {
		return 0, ""
	}

	baseBranch := workspace.baseBranch
	if baseBranch == "" {
		baseBranch = "main"
	}

	// Find the base branch hash
	baseRef, err := workspace.repo.Reference(
		plumbing.NewRemoteReferenceName("origin", baseBranch), true,
	)
	if err != nil {
		baseRef, err = workspace.repo.Reference(
			plumbing.NewBranchReferenceName(baseBranch), true,
		)
		if err != nil {
			return 0, ""
		}
	}

	headRef, err := workspace.repo.Head()
	if err != nil {
		return 0, ""
	}

	// Walk HEAD commits until we hit the base
	iter, err := workspace.repo.Log(&git.LogOptions{From: headRef.Hash()})
	if err != nil {
		return 0, ""
	}

	count := 0
	_ = iter.ForEach(func(c *object.Commit) error {
		if c.Hash == baseRef.Hash() {
			return fmt.Errorf("stop")
		}
		count++
		return nil
	})

	return count, ""
}

// prToolSuffixes are the MCP tool name suffixes that create pull requests / merge requests.
// At runtime, tools are prefixed as "mcp_<serverName>_<toolName>".
var prToolSuffixes = []string{
	"gitlab_create_mr",
	"github_create_pr",
	"create_pull_request",
	"create_merge_request",
}

// extractPullRequestURL scans the agent's step results for tool calls that
// created a PR/MR and extracts the URL from the tool output JSON.
// Returns the first PR URL found, or "" if none.
func extractPullRequestURL(steps []fantasy.StepResult) string {
	for _, step := range steps {
		// Build a map of tool-call-id → tool-name from this step's messages
		toolNames := map[string]string{}
		for _, msg := range step.Messages {
			for _, part := range msg.Content {
				if tc, ok := part.(fantasy.ToolCallPart); ok {
					toolNames[tc.ToolCallID] = tc.ToolName
				}
			}
		}

		// Scan tool results for PR/MR creation responses
		for _, msg := range step.Messages {
			for _, part := range msg.Content {
				tr, ok := part.(fantasy.ToolResultPart)
				if !ok {
					continue
				}

				toolName := toolNames[tr.ToolCallID]
				if !isPRTool(toolName) {
					continue
				}

				// Extract the output text
				var text string
				switch v := tr.Output.(type) {
				case fantasy.ToolResultOutputContentText:
					text = v.Text
				case fantasy.ToolResultOutputContentMedia:
					text = v.Text
				default:
					continue
				}

				if url := parsePRURL(text); url != "" {
					slog.Info("extracted pull request URL from tool result",
						"tool", toolName, "url", url)
					return url
				}
			}
		}
	}

	return ""
}

// isPRTool checks if a tool name matches a known PR/MR creation tool.
func isPRTool(name string) bool {
	lower := strings.ToLower(name)
	for _, suffix := range prToolSuffixes {
		if strings.HasSuffix(lower, suffix) {
			return true
		}
	}
	return false
}

// parsePRURL extracts the PR/MR URL from the raw JSON API response text.
// Looks for "web_url" (GitLab) or "html_url" (GitHub).
func parsePRURL(text string) string {
	// Try to parse as JSON object
	var obj map[string]any
	if err := json.Unmarshal([]byte(text), &obj); err != nil {
		// Not valid JSON — try to find a URL pattern in plain text
		return extractURLFromText(text)
	}

	// GitLab: "web_url"
	if webURL, ok := obj["web_url"].(string); ok && webURL != "" {
		return webURL
	}

	// GitHub: "html_url"
	if htmlURL, ok := obj["html_url"].(string); ok && htmlURL != "" {
		return htmlURL
	}

	return ""
}

// extractURLFromText tries to find a PR/MR URL in plain text output.
// Handles cases where the tool output isn't raw JSON (e.g. formatted text).
func extractURLFromText(text string) string {
	// Look for common PR/MR URL patterns
	patterns := []string{
		"merge_requests/",
		"/pull/",
		"/pulls/",
	}
	for _, line := range strings.Split(text, "\n") {
		for _, pattern := range patterns {
			if idx := strings.Index(line, pattern); idx >= 0 {
				// Walk backwards to find the start of the URL
				start := idx
				for start > 0 && line[start-1] != ' ' && line[start-1] != '"' && line[start-1] != '\'' && line[start-1] != '(' {
					start--
				}
				// Walk forwards to find the end
				end := idx + len(pattern)
				for end < len(line) && line[end] != ' ' && line[end] != '"' && line[end] != '\'' && line[end] != ')' && line[end] != ',' {
					end++
				}
				url := strings.TrimSpace(line[start:end])
				if strings.HasPrefix(url, "http") {
					return url
				}
			}
		}
	}
	return ""
}

// ── Built-in Git Tools ──────────────────────────────────────────────────

type gitStatusInput struct{}

func newGitStatusTool() fantasy.AgentTool {
	return fantasy.NewAgentTool("git_status",
		"Show the working tree status (modified, staged, untracked files).",
		func(_ context.Context, _ gitStatusInput, _ fantasy.ToolCall) (fantasy.ToolResponse, error) {
			if workspace == nil {
				return fantasy.NewTextErrorResponse("no git workspace configured"), nil
			}
			w, err := workspace.repo.Worktree()
			if err != nil {
				return fantasy.NewTextErrorResponse(fmt.Sprintf("worktree error: %v", err)), nil
			}
			status, err := w.Status()
			if err != nil {
				return fantasy.NewTextErrorResponse(fmt.Sprintf("git status: %v", err)), nil
			}
			head, _ := workspace.repo.Head()
			var buf bytes.Buffer
			if head != nil {
				fmt.Fprintf(&buf, "## %s\n", head.Name().Short())
			}
			if status.IsClean() {
				buf.WriteString("working tree clean\n")
			} else {
				buf.WriteString(status.String())
			}
			return fantasy.NewTextResponse(buf.String()), nil
		})
}

type gitDiffInput struct {
	Staged bool   `json:"staged,omitempty" description:"Show staged changes (--cached)"`
	Ref    string `json:"ref,omitempty" description:"Compare against a specific ref (commit/branch)"`
}

func newGitDiffTool() fantasy.AgentTool {
	return fantasy.NewAgentTool("git_diff",
		"Show changes in the working tree or between commits. Returns file status and content diffs.",
		func(_ context.Context, input gitDiffInput, _ fantasy.ToolCall) (fantasy.ToolResponse, error) {
			if workspace == nil {
				return fantasy.NewTextErrorResponse("no git workspace configured"), nil
			}
			w, err := workspace.repo.Worktree()
			if err != nil {
				return fantasy.NewTextErrorResponse(fmt.Sprintf("worktree error: %v", err)), nil
			}
			status, err := w.Status()
			if err != nil {
				return fantasy.NewTextErrorResponse(fmt.Sprintf("status error: %v", err)), nil
			}
			if status.IsClean() {
				return fantasy.NewTextResponse("no changes"), nil
			}

			var buf bytes.Buffer
			// Show file-level status
			for file, s := range status {
				staging := string(s.Staging)
				worktree := string(s.Worktree)
				if input.Staged {
					if staging != " " && staging != "?" {
						fmt.Fprintf(&buf, "%s  %s\n", staging, file)
					}
				} else {
					if worktree != " " || staging != " " {
						fmt.Fprintf(&buf, "%s%s %s\n", staging, worktree, file)
					}
				}
			}

			// Attempt content-level diff for changed files
			headRef, err := workspace.repo.Head()
			if err == nil {
				headCommit, err := workspace.repo.CommitObject(headRef.Hash())
				if err == nil {
					headTree, err := headCommit.Tree()
					if err == nil {
						for file, s := range status {
							if input.Staged && s.Staging == ' ' {
								continue
							}
							if !input.Staged && s.Worktree == ' ' && s.Staging == ' ' {
								continue
							}
							currentPath := filepath.Join(workspace.dir, file)
							currentData, err := os.ReadFile(currentPath)
							if err != nil {
								continue
							}
							entry, err := headTree.File(file)
							if err != nil {
								// New file
								fmt.Fprintf(&buf, "\n--- /dev/null\n+++ b/%s\n", file)
								for _, line := range strings.Split(string(currentData), "\n") {
									fmt.Fprintf(&buf, "+%s\n", line)
								}
								continue
							}
							oldContent, err := entry.Contents()
							if err != nil {
								continue
							}
							if oldContent != string(currentData) {
								fmt.Fprintf(&buf, "\n--- a/%s\n+++ b/%s\n", file, file)
								oldLines := strings.Split(oldContent, "\n")
								newLines := strings.Split(string(currentData), "\n")
								maxLen := len(oldLines)
								if len(newLines) > maxLen {
									maxLen = len(newLines)
								}
								for i := 0; i < maxLen; i++ {
									var ol, nl string
									if i < len(oldLines) {
										ol = oldLines[i]
									}
									if i < len(newLines) {
										nl = newLines[i]
									}
									if ol != nl {
										if i < len(oldLines) {
											fmt.Fprintf(&buf, "-%s\n", ol)
										}
										if i < len(newLines) {
											fmt.Fprintf(&buf, "+%s\n", nl)
										}
									}
								}
							}
						}
					}
				}
			}

			if buf.Len() == 0 {
				return fantasy.NewTextResponse("no changes"), nil
			}
			return fantasy.NewTextResponse(buf.String()), nil
		})
}

type gitLogInput struct {
	Count   int  `json:"count,omitempty" description:"Number of commits to show (default: 20)"`
	Oneline bool `json:"oneline,omitempty" description:"One-line format"`
}

func newGitLogTool() fantasy.AgentTool {
	return fantasy.NewAgentTool("git_log",
		"Show commit logs with hash, author, date, and message.",
		func(_ context.Context, input gitLogInput, _ fantasy.ToolCall) (fantasy.ToolResponse, error) {
			if workspace == nil {
				return fantasy.NewTextErrorResponse("no git workspace configured"), nil
			}
			count := input.Count
			if count <= 0 {
				count = 20
			}
			iter, err := workspace.repo.Log(&git.LogOptions{})
			if err != nil {
				return fantasy.NewTextErrorResponse(fmt.Sprintf("git log: %v", err)), nil
			}
			var buf bytes.Buffer
			n := 0
			_ = iter.ForEach(func(c *object.Commit) error {
				if n >= count {
					return fmt.Errorf("stop")
				}
				if input.Oneline {
					fmt.Fprintf(&buf, "%s %s\n", c.Hash.String()[:7], firstLine(c.Message))
				} else {
					fmt.Fprintf(&buf, "%s %s %s: %s\n",
						c.Hash.String()[:7],
						c.Author.When.Format("2006-01-02"),
						c.Author.Name,
						firstLine(c.Message),
					)
				}
				n++
				return nil
			})
			if buf.Len() == 0 {
				return fantasy.NewTextResponse("no commits"), nil
			}
			return fantasy.NewTextResponse(buf.String()), nil
		})
}

type gitAddInput struct {
	Files string `json:"files" description:"Files to stage (space-separated or '.' for all)"`
}

func newGitAddTool() fantasy.AgentTool {
	return fantasy.NewAgentTool("git_add",
		"Stage files for commit. Use '.' to stage all changes.",
		func(_ context.Context, input gitAddInput, _ fantasy.ToolCall) (fantasy.ToolResponse, error) {
			if workspace == nil {
				return fantasy.NewTextErrorResponse("no git workspace configured"), nil
			}
			if input.Files == "" {
				return fantasy.NewTextErrorResponse("files parameter is required"), nil
			}
			w, err := workspace.repo.Worktree()
			if err != nil {
				return fantasy.NewTextErrorResponse(fmt.Sprintf("worktree error: %v", err)), nil
			}
			files := strings.Fields(input.Files)
			var added []string
			for _, f := range files {
				if f == "." {
					if _, err := w.Add("."); err != nil {
						return fantasy.NewTextErrorResponse(fmt.Sprintf("git add .: %v", err)), nil
					}
					added = append(added, ".")
					break
				}
				if _, err := w.Add(f); err != nil {
					return fantasy.NewTextErrorResponse(fmt.Sprintf("git add %s: %v", f, err)), nil
				}
				added = append(added, f)
			}
			return fantasy.NewTextResponse(fmt.Sprintf("staged: %s", strings.Join(added, ", "))), nil
		})
}

type gitCommitInput struct {
	Message string `json:"message" description:"Commit message"`
}

func newGitCommitTool() fantasy.AgentTool {
	return fantasy.NewAgentTool("git_commit",
		"Create a new commit with staged changes.",
		func(_ context.Context, input gitCommitInput, _ fantasy.ToolCall) (fantasy.ToolResponse, error) {
			if workspace == nil {
				return fantasy.NewTextErrorResponse("no git workspace configured"), nil
			}
			if input.Message == "" {
				return fantasy.NewTextErrorResponse("message parameter is required"), nil
			}
			w, err := workspace.repo.Worktree()
			if err != nil {
				return fantasy.NewTextErrorResponse(fmt.Sprintf("worktree error: %v", err)), nil
			}
			hash, err := w.Commit(input.Message, &git.CommitOptions{
				Author: &object.Signature{
					Name:  "AgentOps Agent",
					Email: "agent@agentops.io",
					When:  time.Now(),
				},
			})
			if err != nil {
				return fantasy.NewTextErrorResponse(fmt.Sprintf("git commit: %v", err)), nil
			}
			return fantasy.NewTextResponse(fmt.Sprintf("committed %s: %s", hash.String()[:7], firstLine(input.Message))), nil
		})
}

type gitPushInput struct {
	Remote string `json:"remote,omitempty" description:"Remote name (default: origin)"`
	Branch string `json:"branch,omitempty" description:"Branch to push (default: current branch)"`
	Force  bool   `json:"force,omitempty" description:"Force push with lease"`
}

func newGitPushTool() fantasy.AgentTool {
	return fantasy.NewAgentTool("git_push",
		"Push commits to the remote repository.",
		func(_ context.Context, input gitPushInput, _ fantasy.ToolCall) (fantasy.ToolResponse, error) {
			if workspace == nil {
				return fantasy.NewTextErrorResponse("no git workspace configured"), nil
			}
			remote := input.Remote
			if remote == "" {
				remote = "origin"
			}
			pushOpts := &git.PushOptions{
				RemoteName: remote,
				Auth:       workspace.auth,
				Force:      input.Force,
			}
			if input.Branch != "" {
				refSpec := config.RefSpec(fmt.Sprintf("refs/heads/%s:refs/heads/%s", input.Branch, input.Branch))
				pushOpts.RefSpecs = []config.RefSpec{refSpec}
			}
			err := workspace.repo.Push(pushOpts)
			if err != nil {
				if err == git.NoErrAlreadyUpToDate {
					return fantasy.NewTextResponse("already up to date"), nil
				}
				return fantasy.NewTextErrorResponse(fmt.Sprintf("git push: %v", err)), nil
			}
			branch := input.Branch
			if branch == "" {
				branch = workspace.branch
			}
			return fantasy.NewTextResponse(fmt.Sprintf("pushed %s to %s", branch, remote)), nil
		})
}

type gitPullInput struct {
	Remote string `json:"remote,omitempty" description:"Remote name (default: origin)"`
	Branch string `json:"branch,omitempty" description:"Branch to pull"`
}

func newGitPullTool() fantasy.AgentTool {
	return fantasy.NewAgentTool("git_pull",
		"Pull changes from the remote repository.",
		func(_ context.Context, input gitPullInput, _ fantasy.ToolCall) (fantasy.ToolResponse, error) {
			if workspace == nil {
				return fantasy.NewTextErrorResponse("no git workspace configured"), nil
			}
			w, err := workspace.repo.Worktree()
			if err != nil {
				return fantasy.NewTextErrorResponse(fmt.Sprintf("worktree error: %v", err)), nil
			}
			remote := input.Remote
			if remote == "" {
				remote = "origin"
			}
			pullOpts := &git.PullOptions{
				RemoteName: remote,
				Auth:       workspace.auth,
			}
			if input.Branch != "" {
				pullOpts.ReferenceName = plumbing.NewBranchReferenceName(input.Branch)
			}
			err = w.Pull(pullOpts)
			if err != nil {
				if err == git.NoErrAlreadyUpToDate {
					return fantasy.NewTextResponse("already up to date"), nil
				}
				return fantasy.NewTextErrorResponse(fmt.Sprintf("git pull: %v", err)), nil
			}
			return fantasy.NewTextResponse(fmt.Sprintf("pulled from %s", remote)), nil
		})
}

type gitBranchInput struct {
	Name   string `json:"name" description:"Branch name to create or switch to"`
	Create bool   `json:"create,omitempty" description:"Create the branch if it doesn't exist"`
}

func newGitBranchTool() fantasy.AgentTool {
	return fantasy.NewAgentTool("git_branch",
		"Create or switch branches.",
		func(_ context.Context, input gitBranchInput, _ fantasy.ToolCall) (fantasy.ToolResponse, error) {
			if workspace == nil {
				return fantasy.NewTextErrorResponse("no git workspace configured"), nil
			}
			if input.Name == "" {
				return fantasy.NewTextErrorResponse("name parameter is required"), nil
			}
			w, err := workspace.repo.Worktree()
			if err != nil {
				return fantasy.NewTextErrorResponse(fmt.Sprintf("worktree error: %v", err)), nil
			}
			err = w.Checkout(&git.CheckoutOptions{
				Branch: plumbing.NewBranchReferenceName(input.Name),
				Create: input.Create,
			})
			if err != nil {
				return fantasy.NewTextErrorResponse(fmt.Sprintf("git checkout: %v", err)), nil
			}
			action := "switched to"
			if input.Create {
				action = "created and switched to"
			}
			return fantasy.NewTextResponse(fmt.Sprintf("%s branch %s", action, input.Name)), nil
		})
}

type gitBranchListInput struct {
	All bool `json:"all,omitempty" description:"Show remote branches too"`
}

func newGitBranchListTool() fantasy.AgentTool {
	return fantasy.NewAgentTool("git_branch_list",
		"List all local and optionally remote branches.",
		func(_ context.Context, input gitBranchListInput, _ fantasy.ToolCall) (fantasy.ToolResponse, error) {
			if workspace == nil {
				return fantasy.NewTextErrorResponse("no git workspace configured"), nil
			}
			refs, err := workspace.repo.References()
			if err != nil {
				return fantasy.NewTextErrorResponse(fmt.Sprintf("list refs: %v", err)), nil
			}
			headRef, _ := workspace.repo.Head()
			var buf bytes.Buffer
			_ = refs.ForEach(func(ref *plumbing.Reference) error {
				name := ref.Name().String()
				if strings.HasPrefix(name, "refs/heads/") {
					branchName := strings.TrimPrefix(name, "refs/heads/")
					marker := "  "
					if headRef != nil && ref.Name() == headRef.Name() {
						marker = "* "
					}
					fmt.Fprintf(&buf, "%s%s %s\n", marker, branchName, ref.Hash().String()[:7])
				} else if input.All && strings.HasPrefix(name, "refs/remotes/") {
					remoteBranch := strings.TrimPrefix(name, "refs/remotes/")
					fmt.Fprintf(&buf, "  remotes/%s %s\n", remoteBranch, ref.Hash().String()[:7])
				}
				return nil
			})
			if buf.Len() == 0 {
				return fantasy.NewTextResponse("no branches"), nil
			}
			return fantasy.NewTextResponse(buf.String()), nil
		})
}

type gitShowInput struct {
	Ref string `json:"ref,omitempty" description:"Commit ref to show (default: HEAD)"`
}

func newGitShowTool() fantasy.AgentTool {
	return fantasy.NewAgentTool("git_show",
		"Show the contents of a commit (message and changed files).",
		func(_ context.Context, input gitShowInput, _ fantasy.ToolCall) (fantasy.ToolResponse, error) {
			if workspace == nil {
				return fantasy.NewTextErrorResponse("no git workspace configured"), nil
			}
			var hash plumbing.Hash
			if input.Ref == "" || input.Ref == "HEAD" {
				ref, err := workspace.repo.Head()
				if err != nil {
					return fantasy.NewTextErrorResponse(fmt.Sprintf("HEAD: %v", err)), nil
				}
				hash = ref.Hash()
			} else {
				h, err := workspace.repo.ResolveRevision(plumbing.Revision(input.Ref))
				if err != nil {
					return fantasy.NewTextErrorResponse(fmt.Sprintf("resolve %s: %v", input.Ref, err)), nil
				}
				hash = *h
			}
			commit, err := workspace.repo.CommitObject(hash)
			if err != nil {
				return fantasy.NewTextErrorResponse(fmt.Sprintf("commit object: %v", err)), nil
			}
			var buf bytes.Buffer
			fmt.Fprintf(&buf, "commit %s\n", commit.Hash.String())
			fmt.Fprintf(&buf, "Author: %s <%s>\n", commit.Author.Name, commit.Author.Email)
			fmt.Fprintf(&buf, "Date:   %s\n\n", commit.Author.When.Format("Mon Jan 2 15:04:05 2006 -0700"))
			fmt.Fprintf(&buf, "    %s\n", commit.Message)
			stats, err := commit.Stats()
			if err == nil && len(stats) > 0 {
				fmt.Fprintf(&buf, "\n")
				for _, s := range stats {
					fmt.Fprintf(&buf, " %s | %d\n", s.Name, s.Addition+s.Deletion)
				}
			}
			return fantasy.NewTextResponse(buf.String()), nil
		})
}

// ── Helpers ─────────────────────────────────────────────────────────────

func firstLine(s string) string {
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		return s[:i]
	}
	return s
}

// gitTools returns all built-in git tools for the agent.
// Only returned when a git workspace is configured (GIT_REPO_URL is set).
func gitTools() []fantasy.AgentTool {
	return []fantasy.AgentTool{
		newGitStatusTool(),
		newGitDiffTool(),
		newGitLogTool(),
		newGitAddTool(),
		newGitCommitTool(),
		newGitPushTool(),
		newGitPullTool(),
		newGitBranchTool(),
		newGitBranchListTool(),
		newGitShowTool(),
	}
}
