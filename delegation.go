/*
Agent Runtime — Fantasy (Go)

DelegationWatcher and run_agents tool for parallel fan-out delegation.

The DelegationWatcher is a background goroutine that tracks active delegation
groups using Kubernetes Watch on AgentRun CRs. When all children in a group
complete (or timeout), it triggers a callback prompt into the agent loop.

The agent remains free (not blocked) during the entire wait — users can
interact with it normally while delegations run in the background.
*/
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"strings"
	"sync"
	"time"

	"charm.land/fantasy"
	"github.com/google/uuid"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
	"k8s.io/apimachinery/pkg/watch"
)

// ════════════════════════════════════════════════════════════════════
// run_agents tool — parallel fan-out delegation
// ════════════════════════════════════════════════════════════════════

type runAgentsInput struct {
	Delegations []DelegationInput `json:"delegations" description:"List of agent delegations to create in parallel"`
	Timeout     string            `json:"timeout,omitempty" description:"Max time to wait for results (default: 30m, max: 4h). Go duration string (e.g. 30m, 1h)."`
}

type DelegationInput struct {
	Agent         string `json:"agent" description:"Agent name to run"`
	Prompt        string `json:"prompt" description:"Prompt to send to the agent"`
	GitResource   string `json:"git_resource,omitempty" description:"AgentResource name for git workspace"`
	GitBranch     string `json:"git_branch,omitempty" description:"Feature branch to work on"`
	GitBaseBranch string `json:"git_base_branch,omitempty" description:"Base branch for PR/MR target (defaults to repo default)"`
}

const (
	defaultDelegationTimeout = 30 * time.Minute
	maxDelegationTimeout     = 4 * time.Hour
)

func newRunAgentsTool(k8s *K8sClient, resources []ResourceEntry, watcher *DelegationWatcher) fantasy.AgentTool {
	desc := buildRunAgentsDescription(resources)
	return fantasy.NewAgentTool("run_agents", desc,
		func(ctx context.Context, input runAgentsInput, _ fantasy.ToolCall) (fantasy.ToolResponse, error) {
			if len(input.Delegations) == 0 {
				return fantasy.NewTextErrorResponse("delegations array is required and must not be empty"), nil
			}
			if len(input.Delegations) > 10 {
				return fantasy.NewTextErrorResponse("max 10 delegations per fan-out (prevent resource exhaustion)"), nil
			}

			// Parse timeout
			timeout := defaultDelegationTimeout
			if input.Timeout != "" {
				parsed, err := time.ParseDuration(input.Timeout)
				if err != nil {
					return fantasy.NewTextErrorResponse(fmt.Sprintf("invalid timeout %q: %s (use Go duration like 30m, 1h)", input.Timeout, err)), nil
				}
				if parsed > maxDelegationTimeout {
					return fantasy.NewTextErrorResponse(fmt.Sprintf("timeout %s exceeds max %s", parsed, maxDelegationTimeout)), nil
				}
				if parsed < time.Minute {
					return fantasy.NewTextErrorResponse("timeout must be at least 1m"), nil
				}
				timeout = parsed
			}

			agentName := os.Getenv("AGENT_NAME")
			if agentName == "" {
				agentName = "unknown"
			}

			// ── Phase 1: Atomic validation ──
			// Validate ALL delegations before creating any AgentRun CRs.
			// If any delegation is invalid, fail fast with no side effects.
			for i, d := range input.Delegations {
				if d.Agent == "" || d.Prompt == "" {
					return fantasy.NewTextErrorResponse(fmt.Sprintf("delegation[%d]: agent and prompt are required", i)), nil
				}
				if d.Agent == agentName {
					return fantasy.NewTextErrorResponse(fmt.Sprintf("delegation[%d]: cannot delegate to yourself (%q)", i, agentName)), nil
				}
				if d.GitResource != "" || d.GitBranch != "" {
					if d.GitResource == "" || d.GitBranch == "" {
						return fantasy.NewTextErrorResponse(fmt.Sprintf("delegation[%d]: git_resource and git_branch are both required when using a git workspace", i)), nil
					}
				}

				// Check agent exists and is visible
				agentInfo, err := k8s.GetAgent(ctx, d.Agent)
				if err != nil {
					available, listErr := k8s.ListAgents(ctx)
					if listErr != nil || len(available) == 0 {
						return fantasy.NewTextErrorResponse(fmt.Sprintf("delegation[%d]: agent %q not found", i, d.Agent)), nil
					}
					var agentList string
					for _, a := range available {
						if a.Name != agentName {
							agentList += fmt.Sprintf("\n  - %s (mode: %s, phase: %s)", a.Name, a.Mode, a.Phase)
						}
					}
					return fantasy.NewTextErrorResponse(fmt.Sprintf("delegation[%d]: agent %q not found. Available agents:%s", i, d.Agent, agentList)), nil
				}

				// Check delegation constraints
				targetScope, targetCallers, discErr := k8s.GetAgentDiscovery(ctx, d.Agent)
				if discErr == nil && !isAgentVisible(targetScope, targetCallers, agentName) {
					return fantasy.NewTextErrorResponse(fmt.Sprintf("delegation[%d]: agent %q is not available for delegation (scope: %s)", i, d.Agent, targetScope)), nil
				}

				// Check daemon readiness
				if agentInfo.Mode == "daemon" && agentInfo.Phase != "Running" {
					return fantasy.NewTextErrorResponse(fmt.Sprintf("delegation[%d]: agent %q is a daemon in phase %q (not Running)", i, d.Agent, agentInfo.Phase)), nil
				}
			}

			// ── Phase 2: Create all AgentRun CRs in parallel ──
			groupID := uuid.New().String()[:8]
			traceparent := traceparentFromContext(ctx)

			type createResult struct {
				index int
				name  string
				agent string
				err   error
			}
			results := make(chan createResult, len(input.Delegations))

			for i, d := range input.Delegations {
				go func(idx int, del DelegationInput) {
					var gitParams *AgentRunGitParams
					if del.GitResource != "" {
						gitParams = &AgentRunGitParams{
							ResourceRef: del.GitResource,
							Branch:      del.GitBranch,
							BaseBranch:  del.GitBaseBranch,
						}
					}

					run, err := k8s.CreateAgentRun(ctx, del.Agent, del.Prompt, "agent", agentName, traceparent, gitParams)
					if err != nil {
						results <- createResult{index: idx, agent: del.Agent, err: err}
						return
					}
					results <- createResult{index: idx, name: run.Name, agent: del.Agent}
				}(i, d)
			}

			// Collect results
			runs := make([]DelegationRun, len(input.Delegations))
			var createErrors []string
			for range input.Delegations {
				r := <-results
				if r.err != nil {
					createErrors = append(createErrors, fmt.Sprintf("  %s: %s", r.agent, r.err))
				} else {
					runs[r.index] = DelegationRun{
						AgentName: r.agent,
						RunName:   r.name,
					}
				}
			}

			if len(createErrors) > 0 {
				// Some creates failed. The successful ones are already created in K8s,
				// so we report them but note the failures.
				var msg strings.Builder
				msg.WriteString("Some delegations failed to create:\n")
				msg.WriteString(strings.Join(createErrors, "\n"))
				msg.WriteString("\n\nSuccessfully created:\n")
				for _, r := range runs {
					if r.RunName != "" {
						msg.WriteString(fmt.Sprintf("  %s → %s\n", r.AgentName, r.RunName))
					}
				}
				return fantasy.NewTextErrorResponse(msg.String()), nil
			}

			// ── Phase 3: Register with DelegationWatcher ──
			group := &DelegationGroup{
				ID:        groupID,
				Runs:      make(map[string]*RunResult),
				Remaining: len(runs),
				Timeout:   time.Now().Add(timeout),
				CreatedAt: time.Now(),
			}
			for _, r := range runs {
				group.Runs[r.RunName] = nil // nil = still running
			}

			if watcher != nil {
				watcher.Register(group)
			}

			// Set delegation attributes on the current span
			currentSpan := trace.SpanFromContext(ctx)
			runNames := make([]string, len(runs))
			for i, r := range runs {
				runNames[i] = r.RunName
			}
			currentSpan.SetAttributes(
				attribute.String("delegation.group_id", groupID),
				attribute.Int("delegation.count", len(runs)),
				attribute.StringSlice("delegation.run_names", runNames),
			)

			// ── Phase 4: Build response ──
			var resp strings.Builder
			resp.WriteString(fmt.Sprintf("Delegated %d tasks in parallel (group: %s):\n", len(runs), groupID))
			for i, r := range runs {
				resp.WriteString(fmt.Sprintf("  %d. %-20s → %s\n", i+1, r.AgentName, r.RunName))
			}
			resp.WriteString(fmt.Sprintf("\nI'll automatically receive results when all agents complete (timeout: %s).\n", timeout))
			resp.WriteString("You can ask me about the status anytime, or give me other tasks while we wait.")

			toolResp := fantasy.NewTextResponse(resp.String())
			toolResp = fantasy.WithResponseMetadata(toolResp, map[string]any{
				"ui":      "delegation-fan-out",
				"groupId": groupID,
				"runs":    runs,
				"timeout": timeout.String(),
			})
			return toolResp, nil
		})
}

func newRunAgentsToolStub() fantasy.AgentTool {
	return fantasy.NewAgentTool("run_agents",
		"Delegate tasks to multiple agents in parallel. (Unavailable: K8s client not configured)",
		func(_ context.Context, _ runAgentsInput, _ fantasy.ToolCall) (fantasy.ToolResponse, error) {
			return fantasy.NewTextErrorResponse("run_agents unavailable: not running in Kubernetes"), nil
		})
}

func buildRunAgentsDescription(resources []ResourceEntry) string {
	base := "Delegate tasks to multiple agents in parallel (fan-out). Creates AgentRun CRs for all delegations atomically — " +
		"all are validated before any are created. Results are delivered automatically via callback when all agents complete.\n\n" +
		"IMPORTANT: After calling run_agents, tell the user the tasks have been delegated. " +
		"You will automatically receive results when all agents finish — do NOT poll with get_agent_run. " +
		"The user can give you other tasks while waiting."

	// Add git resource info (same as run_agent)
	var gitResources []string
	for _, r := range resources {
		switch r.Kind {
		case "github-repo":
			if r.GitHub != nil {
				detail := fmt.Sprintf("  - %q (GitHub: %s/%s", r.Name, r.GitHub.Owner, r.GitHub.Repo)
				if r.GitHub.DefaultBranch != "" {
					detail += fmt.Sprintf(", default branch: %s", r.GitHub.DefaultBranch)
				}
				detail += ")"
				if r.Description != "" {
					detail += " — " + r.Description
				}
				gitResources = append(gitResources, detail)
			}
		case "gitlab-project":
			if r.GitLab != nil {
				detail := fmt.Sprintf("  - %q (GitLab: %s", r.Name, r.GitLab.Project)
				if r.GitLab.DefaultBranch != "" {
					detail += fmt.Sprintf(", default branch: %s", r.GitLab.DefaultBranch)
				}
				detail += ")"
				if r.Description != "" {
					detail += " — " + r.Description
				}
				gitResources = append(gitResources, detail)
			}
		case "git-repo":
			if r.Git != nil {
				detail := fmt.Sprintf("  - %q (git: %s)", r.Name, r.Git.URL)
				if r.Description != "" {
					detail += " — " + r.Description
				}
				gitResources = append(gitResources, detail)
			}
		}
	}

	if len(gitResources) > 0 {
		base += "\n\nFor coding/git tasks, set git_resource + git_branch per delegation. " +
			"Available git resources:\n" + strings.Join(gitResources, "\n")
	}

	return base
}

// ════════════════════════════════════════════════════════════════════
// DelegationWatcher — background K8s Watch for delegation groups
// ════════════════════════════════════════════════════════════════════

// DelegationRun is a single agent run within a delegation group.
type DelegationRun struct {
	AgentName string `json:"agentName"`
	RunName   string `json:"runName"`
}

// RunResult holds the outcome of a completed child run.
type RunResult struct {
	AgentName      string        `json:"agentName"`
	Phase          string        `json:"phase"`
	Output         string        `json:"output"`
	ToolCalls      int64         `json:"toolCalls"`
	Model          string        `json:"model"`
	Duration       time.Duration `json:"duration"`
	PullRequestURL string        `json:"pullRequestURL,omitempty"`
	Commits        int64         `json:"commits,omitempty"`
	Branch         string        `json:"branch,omitempty"`
	FailureReason  string        `json:"failureReason,omitempty"`
}

// DelegationGroup tracks a set of parallel AgentRuns.
type DelegationGroup struct {
	ID        string                `json:"id"`
	Runs      map[string]*RunResult `json:"runs"`      // runName → result (nil = still running)
	Remaining int                   `json:"remaining"` // count of runs not yet terminal
	Timeout   time.Time             `json:"timeout"`
	CreatedAt time.Time             `json:"createdAt"`
	cancelFn  context.CancelFunc    // stops all watch goroutines + timeout timer
}

// DelegationWatcher manages active delegation groups using K8s Watch.
type DelegationWatcher struct {
	k8s       *K8sClient
	groups    map[string]*DelegationGroup // keyed by group ID
	mu        sync.Mutex
	triggerFn func(prompt string) // internal prompt trigger (set by daemon)
	emitterFn func() *fepEmitter  // get current FEP emitter (may be nil)
}

// NewDelegationWatcher creates a new watcher. triggerFn and emitterFn are
// set later by the daemon server after construction.
func NewDelegationWatcher(k8s *K8sClient) *DelegationWatcher {
	return &DelegationWatcher{
		k8s:    k8s,
		groups: make(map[string]*DelegationGroup),
	}
}

// SetTrigger sets the internal prompt trigger function.
func (dw *DelegationWatcher) SetTrigger(fn func(prompt string)) {
	dw.mu.Lock()
	defer dw.mu.Unlock()
	dw.triggerFn = fn
}

// SetEmitterFn sets the FEP emitter accessor.
func (dw *DelegationWatcher) SetEmitterFn(fn func() *fepEmitter) {
	dw.mu.Lock()
	defer dw.mu.Unlock()
	dw.emitterFn = fn
}

// Register adds a delegation group and starts watching its runs.
func (dw *DelegationWatcher) Register(group *DelegationGroup) {
	ctx, cancel := context.WithCancel(context.Background())
	group.cancelFn = cancel

	dw.mu.Lock()
	dw.groups[group.ID] = group
	dw.mu.Unlock()

	slog.Info("delegation group registered",
		"groupId", group.ID,
		"runs", len(group.Runs),
		"timeout", group.Timeout.Format(time.RFC3339),
	)

	// Emit fan-out FEP event
	dw.emitFEP("delegation.fan_out", map[string]any{
		"groupId":  group.ID,
		"count":    len(group.Runs),
		"timeout":  group.Timeout.Format(time.RFC3339),
		"runNames": runNamesFromGroup(group),
	})

	// Start a watch goroutine for each run
	for runName := range group.Runs {
		go dw.watchRun(ctx, group.ID, runName)
	}

	// Start timeout timer
	go dw.timeoutTimer(ctx, group)
}

// ActiveGroups returns a snapshot of active delegation groups (for status queries).
func (dw *DelegationWatcher) ActiveGroups() []DelegationGroupStatus {
	dw.mu.Lock()
	defer dw.mu.Unlock()

	statuses := make([]DelegationGroupStatus, 0, len(dw.groups))
	for _, g := range dw.groups {
		status := DelegationGroupStatus{
			ID:        g.ID,
			Total:     len(g.Runs),
			Remaining: g.Remaining,
			Timeout:   g.Timeout,
			CreatedAt: g.CreatedAt,
		}
		for name, result := range g.Runs {
			if result != nil {
				status.Completed = append(status.Completed, name)
			} else {
				status.Pending = append(status.Pending, name)
			}
		}
		statuses = append(statuses, status)
	}
	return statuses
}

// DelegationGroupStatus is a read-only snapshot for status queries.
type DelegationGroupStatus struct {
	ID        string    `json:"id"`
	Total     int       `json:"total"`
	Remaining int       `json:"remaining"`
	Completed []string  `json:"completed"`
	Pending   []string  `json:"pending"`
	Timeout   time.Time `json:"timeout"`
	CreatedAt time.Time `json:"createdAt"`
}

// Stop cancels all active watches and cleans up.
func (dw *DelegationWatcher) Stop() {
	dw.mu.Lock()
	defer dw.mu.Unlock()

	for id, g := range dw.groups {
		if g.cancelFn != nil {
			g.cancelFn()
		}
		slog.Info("delegation group cancelled on shutdown", "groupId", id)
	}
}

// CheckpointGroups returns the active groups for persistence.
func (dw *DelegationWatcher) CheckpointGroups() []DelegationGroup {
	dw.mu.Lock()
	defer dw.mu.Unlock()

	groups := make([]DelegationGroup, 0, len(dw.groups))
	for _, g := range dw.groups {
		// Copy without cancelFn (not serializable)
		cp := DelegationGroup{
			ID:        g.ID,
			Runs:      g.Runs,
			Remaining: g.Remaining,
			Timeout:   g.Timeout,
			CreatedAt: g.CreatedAt,
		}
		groups = append(groups, cp)
	}
	return groups
}

// RestoreGroups re-establishes watches for groups recovered from checkpoint.
func (dw *DelegationWatcher) RestoreGroups(groups []DelegationGroup) {
	for i := range groups {
		g := &groups[i]

		// Skip expired groups
		if time.Now().After(g.Timeout) {
			slog.Info("skipping expired delegation group from checkpoint", "groupId", g.ID)
			continue
		}

		// Re-count remaining (in case state was partially captured)
		remaining := 0
		for _, r := range g.Runs {
			if r == nil {
				remaining++
			}
		}
		g.Remaining = remaining

		if remaining == 0 {
			// All already completed — trigger callback immediately
			slog.Info("restored delegation group already complete, triggering callback", "groupId", g.ID)
			go dw.triggerCallback(g)
			continue
		}

		dw.Register(g)
		slog.Info("restored delegation group from checkpoint",
			"groupId", g.ID,
			"remaining", remaining,
			"total", len(g.Runs),
		)
	}
}

// ── Watch logic ──

func (dw *DelegationWatcher) watchRun(ctx context.Context, groupID, runName string) {
	slog.Info("watching delegation run", "groupId", groupID, "run", runName)

	for {
		if ctx.Err() != nil {
			return
		}

		watcher, err := dw.k8s.WatchAgentRun(ctx, runName)
		if err != nil {
			slog.Warn("failed to start watch on AgentRun, retrying in 5s",
				"groupId", groupID, "run", runName, "error", err)
			select {
			case <-ctx.Done():
				return
			case <-time.After(5 * time.Second):
				continue
			}
		}

		dw.processWatchEvents(ctx, watcher, groupID, runName)

		// If context cancelled, we're done
		if ctx.Err() != nil {
			return
		}

		// Check if run is already recorded as complete (another event path)
		dw.mu.Lock()
		group, exists := dw.groups[groupID]
		if exists && group.Runs[runName] != nil {
			dw.mu.Unlock()
			return // Already completed
		}
		dw.mu.Unlock()

		// Watch ended without terminal state — reconnect
		slog.Info("watch ended, reconnecting", "groupId", groupID, "run", runName)
	}
}

func (dw *DelegationWatcher) processWatchEvents(ctx context.Context, w watch.Interface, groupID, runName string) {
	defer w.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case event, ok := <-w.ResultChan():
			if !ok {
				return // Watch channel closed, will reconnect
			}

			if event.Type == watch.Error {
				slog.Warn("watch error event", "groupId", groupID, "run", runName)
				return // Reconnect
			}

			if event.Type != watch.Modified && event.Type != watch.Added {
				continue
			}

			// Extract status from the unstructured object
			obj, ok := event.Object.(interface{ UnstructuredContent() map[string]interface{} })
			if !ok {
				continue
			}

			status := extractRunStatus(obj.UnstructuredContent(), runName)
			if status == nil {
				continue
			}

			if !isTerminalPhase(status.Phase) {
				continue
			}

			// Terminal phase — record result
			slog.Info("delegation run completed",
				"groupId", groupID,
				"run", runName,
				"agent", status.AgentName,
				"phase", status.Phase,
			)

			dw.recordCompletion(groupID, runName, status)
			return
		}
	}
}

func (dw *DelegationWatcher) recordCompletion(groupID, runName string, result *RunResult) {
	dw.mu.Lock()
	group, exists := dw.groups[groupID]
	if !exists {
		dw.mu.Unlock()
		return
	}

	// Guard against double-completion
	if group.Runs[runName] != nil {
		dw.mu.Unlock()
		return
	}

	group.Runs[runName] = result
	group.Remaining--
	remaining := group.Remaining
	dw.mu.Unlock()

	// Emit per-run completion FEP event
	dw.emitFEP("delegation.run_completed", map[string]any{
		"groupId":    groupID,
		"runName":    runName,
		"childAgent": result.AgentName,
		"phase":      result.Phase,
		"duration":   result.Duration.String(),
		"remaining":  remaining,
	})

	if remaining == 0 {
		dw.triggerCallback(group)
	}
}

func (dw *DelegationWatcher) triggerCallback(group *DelegationGroup) {
	// Cancel watch goroutines + timeout timer
	if group.cancelFn != nil {
		group.cancelFn()
	}

	// Remove from active groups
	dw.mu.Lock()
	delete(dw.groups, group.ID)
	triggerFn := dw.triggerFn
	dw.mu.Unlock()

	// Build callback prompt
	prompt := buildCallbackPrompt(group, false)

	slog.Info("delegation group complete, triggering callback",
		"groupId", group.ID,
		"elapsed", time.Since(group.CreatedAt).Round(time.Second),
	)

	// Emit all-completed FEP event
	succeeded, failed := countOutcomes(group)
	dw.emitFEP("delegation.all_completed", map[string]any{
		"groupId":       group.ID,
		"succeeded":     succeeded,
		"failed":        failed,
		"totalDuration": time.Since(group.CreatedAt).Round(time.Second).String(),
	})

	// Trigger the agent loop with results
	if triggerFn != nil {
		triggerFn(prompt)
	} else {
		slog.Warn("delegation callback: no trigger function set", "groupId", group.ID)
	}
}

func (dw *DelegationWatcher) timeoutTimer(ctx context.Context, group *DelegationGroup) {
	remaining := time.Until(group.Timeout)
	if remaining <= 0 {
		dw.handleTimeout(group)
		return
	}

	select {
	case <-ctx.Done():
		return // Group completed or daemon shutdown
	case <-time.After(remaining):
		dw.handleTimeout(group)
	}
}

func (dw *DelegationWatcher) handleTimeout(group *DelegationGroup) {
	dw.mu.Lock()
	_, exists := dw.groups[group.ID]
	if !exists {
		dw.mu.Unlock()
		return // Already completed
	}
	delete(dw.groups, group.ID)
	triggerFn := dw.triggerFn
	dw.mu.Unlock()

	// Cancel watch goroutines
	if group.cancelFn != nil {
		group.cancelFn()
	}

	slog.Warn("delegation group timed out", "groupId", group.ID,
		"remaining", group.Remaining, "total", len(group.Runs))

	// Emit timeout FEP event
	completed := len(group.Runs) - group.Remaining
	dw.emitFEP("delegation.timeout", map[string]any{
		"groupId":   group.ID,
		"completed": completed,
		"timedOut":  group.Remaining,
	})

	// Build partial results prompt
	prompt := buildCallbackPrompt(group, true)

	if triggerFn != nil {
		triggerFn(prompt)
	}
}

// ── Helpers ──

func (dw *DelegationWatcher) emitFEP(eventType string, fields map[string]any) {
	dw.mu.Lock()
	emitterFn := dw.emitterFn
	dw.mu.Unlock()

	if emitterFn == nil {
		return
	}
	if emit := emitterFn(); emit != nil {
		fields["parentAgent"] = os.Getenv("AGENT_NAME")
		emit.emit(eventType, fields)
	}
}

func extractRunStatus(obj map[string]interface{}, runName string) *RunResult {
	statusRaw, found := obj["status"]
	if !found {
		return nil
	}
	statusMap, ok := statusRaw.(map[string]interface{})
	if !ok {
		return nil
	}

	phase, _ := statusMap["phase"].(string)
	if phase == "" {
		return nil
	}

	// Extract the agent name from spec.agentRef
	agentName := ""
	if spec, ok := obj["spec"].(map[string]interface{}); ok {
		agentName, _ = spec["agentRef"].(string)
	}

	result := &RunResult{
		AgentName: agentName,
		Phase:     phase,
	}

	// Marshal status to JSON and back for clean extraction
	data, err := json.Marshal(statusMap)
	if err == nil {
		var rs AgentRunStatus
		if json.Unmarshal(data, &rs) == nil {
			result.Output = rs.Output
			result.ToolCalls = rs.ToolCalls
			result.Model = rs.Model
			result.PullRequestURL = rs.PullRequestURL
			result.Commits = rs.Commits
			result.Branch = rs.Branch
		}
	}

	return result
}

func isTerminalPhase(phase string) bool {
	switch phase {
	case "Succeeded", "Failed", "Cancelled":
		return true
	default:
		return false
	}
}

func buildCallbackPrompt(group *DelegationGroup, timedOut bool) string {
	var b strings.Builder

	if timedOut {
		b.WriteString(fmt.Sprintf("[DELEGATION RESULTS] Fan-out group %s TIMED OUT (%s):\n\n",
			group.ID, time.Since(group.CreatedAt).Round(time.Second)))
	} else {
		b.WriteString(fmt.Sprintf("[DELEGATION RESULTS] Fan-out group %s completed (%s):\n\n",
			group.ID, time.Since(group.CreatedAt).Round(time.Second)))
	}

	i := 0
	for runName, result := range group.Runs {
		i++
		if result == nil {
			b.WriteString(fmt.Sprintf("  %d. %s — Still Running (timed out)\n", i, runName))
			continue
		}

		b.WriteString(fmt.Sprintf("  %d. %s — %s", i, runName, result.Phase))
		if result.Duration > 0 {
			b.WriteString(fmt.Sprintf(" (%s)", result.Duration.Round(time.Second)))
		}
		b.WriteString("\n")

		if result.PullRequestURL != "" {
			b.WriteString(fmt.Sprintf("     PR: %s\n", result.PullRequestURL))
		}
		if result.Branch != "" {
			b.WriteString(fmt.Sprintf("     Branch: %s (commits: %d)\n", result.Branch, result.Commits))
		}
		if result.ToolCalls > 0 {
			b.WriteString(fmt.Sprintf("     Tool calls: %d, Model: %s\n", result.ToolCalls, result.Model))
		}

		// Include output (truncated)
		if result.Output != "" {
			output := result.Output
			if len(output) > 2000 {
				output = output[:2000] + "\n     ... (truncated)"
			}
			b.WriteString(fmt.Sprintf("     Output:\n     %s\n", strings.ReplaceAll(output, "\n", "\n     ")))
		}

		if result.FailureReason != "" {
			b.WriteString(fmt.Sprintf("     Failure: %s\n", result.FailureReason))
		}
		b.WriteString("\n")
	}

	if timedOut {
		b.WriteString("\nSome agents did not complete within the timeout. You may want to check on them with get_agent_run.\n")
	} else {
		b.WriteString("\nAll delegated agents have completed. Please review the results and respond to the user.\n")
	}

	return b.String()
}

func countOutcomes(group *DelegationGroup) (succeeded, failed int) {
	for _, r := range group.Runs {
		if r == nil {
			continue
		}
		switch r.Phase {
		case "Succeeded":
			succeeded++
		case "Failed", "Cancelled":
			failed++
		}
	}
	return
}

func runNamesFromGroup(group *DelegationGroup) []string {
	names := make([]string, 0, len(group.Runs))
	for name := range group.Runs {
		names = append(names, name)
	}
	return names
}
