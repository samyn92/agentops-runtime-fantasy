/*
Agent Runtime — Fantasy (Go)

Outcome finalization for AgentRuns.

Per the AgentRunOutcomeSpec proposal (agentops-core docs/PROPOSAL_agentrun_outcome.md),
each AgentRun ends with a structured outcome written to status.outcome:

	intent:    change | plan | incident | discovery | noop
	artifacts: [{kind, url, ref, title, provider}]
	summary:   short human-readable result

Two write paths:

 1. autoFinalizeOutcome — runs unconditionally at end-of-task. Infers a
    sensible default outcome from observed signals (PR/MR tool calls,
    git workspace presence, commit count). Lets every task have a
    populated outcome even before agents start using run_finish.

 2. run_finish built-in tool — agents call this themselves to override
    or enrich the auto-inferred outcome. Adds explicit intent, summary,
    and artifacts (e.g. a memory note ID, a planning Issue URL).

Both paths PATCH status.outcome via K8sClient.PatchAgentRunOutcome,
which requires the per-agent Role to grant `agentruns/status patch`
(granted by agentops-core BuildAgentRole).
*/
package main

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"strings"
	"time"

	"charm.land/fantasy"
)

// ────────────────────────────────────────────────────────────────────────────
// Intent enum — keep in sync with agentops-core api/v1alpha1.AgentRunIntent
// ────────────────────────────────────────────────────────────────────────────

const (
	IntentChange    = "change"    // code/config changes (PR/MR, commits)
	IntentPlan      = "plan"      // a planning Issue, RFC, design doc
	IntentIncident  = "incident"  // an incident Issue, RCA, postmortem
	IntentDiscovery = "discovery" // investigation/research, no artifacts
	IntentNoop      = "noop"      // nothing produced, ran but no effect
)

func validIntent(s string) bool {
	switch s {
	case IntentChange, IntentPlan, IntentIncident, IntentDiscovery, IntentNoop:
		return true
	}
	return false
}

const (
	ArtifactPR     = "pr"
	ArtifactMR     = "mr"
	ArtifactIssue  = "issue"
	ArtifactMemory = "memory"
	ArtifactCommit = "commit"
)

func validArtifactKind(s string) bool {
	switch s {
	case ArtifactPR, ArtifactMR, ArtifactIssue, ArtifactMemory, ArtifactCommit:
		return true
	}
	return false
}

// agentRunNameForOutcome returns the AgentRun CR name this pod is executing.
// In task mode the operator sets AGENT_RUN_NAME on the Job pod; if missing
// the patch is skipped (e.g. local dev / shell runs).
func agentRunNameForOutcome() string {
	return os.Getenv("AGENT_RUN_NAME")
}

// ────────────────────────────────────────────────────────────────────────────
// autoFinalizeOutcome — inferred end-of-task outcome
// ────────────────────────────────────────────────────────────────────────────

// autoFinalizeOutcome infers a baseline outcome from observed signals and
// patches status.outcome on the AgentRun. Best-effort: failures are logged
// but never block task completion. Skipped silently when AGENT_RUN_NAME or
// the global k8sClient is unset (local/shell runs).
//
// Inference rules:
//
//	git workspace + PR URL  → intent=change, artifacts=[pr/mr, commit]
//	git workspace + commits → intent=change, artifacts=[commit]
//	git workspace, no work  → intent=discovery
//	no git workspace        → intent=noop
//
// The agent may override any of this by calling run_finish before exit.
func autoFinalizeOutcome(ctx context.Context, k8s *K8sClient, steps []fantasy.StepResult, gitBranch, agentOutput string) {
	runName := agentRunNameForOutcome()
	if runName == "" || k8s == nil {
		return
	}

	outcome := AgentRunOutcome{}
	hasGit := workspace != nil

	prURL := extractPullRequestURL(steps)
	commits := 0
	if hasGit {
		commits, _ = extractGitInfo()
	}

	switch {
	case prURL != "":
		outcome.Intent = IntentChange
		outcome.Artifacts = append(outcome.Artifacts, AgentRunArtifact{
			Kind:     prArtifactKind(prURL),
			URL:      prURL,
			Provider: providerFromURL(prURL),
			Ref:      gitBranch,
			Title:    truncate(firstNonEmptyLine(agentOutput), 120),
		})
		if commits > 0 {
			outcome.Artifacts = append(outcome.Artifacts, AgentRunArtifact{
				Kind:  ArtifactCommit,
				Ref:   gitBranch,
				Title: fmt.Sprintf("%d commit(s)", commits),
			})
		}
	case hasGit && commits > 0:
		outcome.Intent = IntentChange
		outcome.Artifacts = append(outcome.Artifacts, AgentRunArtifact{
			Kind:  ArtifactCommit,
			Ref:   gitBranch,
			Title: fmt.Sprintf("%d commit(s)", commits),
		})
	case hasGit:
		outcome.Intent = IntentDiscovery
	default:
		outcome.Intent = IntentNoop
	}

	outcome.Summary = truncate(firstNonEmptyLine(agentOutput), 240)

	patchCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	if err := k8s.PatchAgentRunOutcome(patchCtx, runName, outcome); err != nil {
		slog.Warn("autoFinalizeOutcome: patch failed",
			"run", runName, "intent", outcome.Intent, "err", err)
		return
	}
	slog.Info("autoFinalizeOutcome: status.outcome written",
		"run", runName, "intent", outcome.Intent, "artifacts", len(outcome.Artifacts))
}

// ────────────────────────────────────────────────────────────────────────────
// URL / provider helpers
// ────────────────────────────────────────────────────────────────────────────

// prArtifactKind picks "pr" or "mr" based on URL host conventions.
// GitLab uses /merge_requests/, everything else (GitHub, Gitea, Codeberg) is "pr".
func prArtifactKind(url string) string {
	if strings.Contains(url, "/merge_requests/") || strings.Contains(url, "gitlab") {
		return ArtifactMR
	}
	return ArtifactPR
}

// providerFromURL returns a short provider name guessed from the URL host.
// Returns "" when no known provider matches; UI can then just show the URL.
func providerFromURL(url string) string {
	switch {
	case strings.Contains(url, "github.com"):
		return "github"
	case strings.Contains(url, "gitlab"):
		return "gitlab"
	case strings.Contains(url, "gitea"):
		return "gitea"
	case strings.Contains(url, "codeberg"):
		return "codeberg"
	case strings.Contains(url, "bitbucket"):
		return "bitbucket"
	}
	return ""
}

// firstNonEmptyLine returns the first non-blank line of s, trimmed.
// Used to derive a single-line title/summary from the agent's free-form output.
func firstNonEmptyLine(s string) string {
	for _, line := range strings.Split(s, "\n") {
		t := strings.TrimSpace(line)
		if t != "" {
			return t
		}
	}
	return ""
}

// ────────────────────────────────────────────────────────────────────────────
// run_finish built-in tool
// ────────────────────────────────────────────────────────────────────────────

// RunFinishInput is the JSON schema the agent fills when calling run_finish.
// Field tags drive the schema fantasy advertises to the model.
type RunFinishInput struct {
	Intent    string                  `json:"intent" description:"One of: change | plan | incident | discovery | noop. Run-level outcome category."`
	Summary   string                  `json:"summary,omitempty" description:"Short human-readable summary of what this run accomplished (≤240 chars)."`
	Artifacts []RunFinishArtifactItem `json:"artifacts,omitempty" description:"Concrete things produced: PRs, MRs, Issues, memory notes, commits."`
}

// RunFinishArtifactItem is one entry in RunFinishInput.Artifacts.
type RunFinishArtifactItem struct {
	Kind     string `json:"kind" description:"One of: pr | mr | issue | memory | commit."`
	URL      string `json:"url,omitempty" description:"Full URL to the artifact when applicable (PR/MR/Issue link)."`
	Ref      string `json:"ref,omitempty" description:"Git ref (branch / commit sha) or memory note ID."`
	Title    string `json:"title,omitempty" description:"Short human-readable title."`
	Provider string `json:"provider,omitempty" description:"Optional provider hint: github | gitlab | gitea | codeberg | bitbucket."`
}

// newRunFinishTool constructs the run_finish AgentTool. Agents call this
// to write or override status.outcome on their AgentRun. The tool is a
// no-op outside Kubernetes (returns an explanatory error response).
func newRunFinishTool(k8s *K8sClient) fantasy.AgentTool {
	desc := "Record the structured outcome of this run before exiting. " +
		"Sets status.outcome on the AgentRun: intent (change|plan|incident|discovery|noop), " +
		"a short summary, and concrete artifacts (PR/MR/Issue URLs, memory note IDs, commit refs). " +
		"Call this once near the end of your task. Auto-inference runs at task end too, " +
		"but calling run_finish gives you an explicit override."

	return fantasy.NewAgentTool("run_finish", desc,
		func(ctx context.Context, input RunFinishInput, _ fantasy.ToolCall) (fantasy.ToolResponse, error) {
			if input.Intent == "" {
				return fantasy.NewTextErrorResponse(
					"run_finish: intent is required (change | plan | incident | discovery | noop)"), nil
			}
			if !validIntent(input.Intent) {
				return fantasy.NewTextErrorResponse(fmt.Sprintf(
					"run_finish: invalid intent %q — must be one of change | plan | incident | discovery | noop",
					input.Intent)), nil
			}
			for i, a := range input.Artifacts {
				if a.Kind == "" {
					return fantasy.NewTextErrorResponse(fmt.Sprintf(
						"run_finish: artifacts[%d].kind is required", i)), nil
				}
				if !validArtifactKind(a.Kind) {
					return fantasy.NewTextErrorResponse(fmt.Sprintf(
						"run_finish: artifacts[%d].kind %q invalid — must be one of pr | mr | issue | memory | commit",
						i, a.Kind)), nil
				}
			}

			runName := agentRunNameForOutcome()
			if runName == "" {
				return fantasy.NewTextErrorResponse(
					"run_finish: AGENT_RUN_NAME is not set — outcome can only be recorded when running as a Kubernetes Job"), nil
			}
			if k8s == nil {
				return fantasy.NewTextErrorResponse(
					"run_finish: Kubernetes client unavailable — outcome cannot be recorded"), nil
			}

			outcome := AgentRunOutcome{
				Intent:  input.Intent,
				Summary: truncate(input.Summary, 240),
			}
			for _, a := range input.Artifacts {
				provider := a.Provider
				if provider == "" && a.URL != "" {
					provider = providerFromURL(a.URL)
				}
				outcome.Artifacts = append(outcome.Artifacts, AgentRunArtifact{
					Kind:     a.Kind,
					URL:      a.URL,
					Ref:      a.Ref,
					Title:    truncate(a.Title, 120),
					Provider: provider,
				})
			}

			patchCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
			defer cancel()
			if err := k8s.PatchAgentRunOutcome(patchCtx, runName, outcome); err != nil {
				return fantasy.NewTextErrorResponse(fmt.Sprintf(
					"run_finish: patch failed: %v", err)), nil
			}

			return fantasy.NewTextResponse(fmt.Sprintf(
				"Outcome recorded: intent=%s, %d artifact(s). Console will reflect on next refresh.",
				input.Intent, len(outcome.Artifacts))), nil
		},
	)
}

// ────────────────────────────────────────────────────────────────────────────
// Rendering helpers — used by delegation.go for FEP events and callback prompts
// ────────────────────────────────────────────────────────────────────────────

// outcomeToFEPMap renders an AgentRunOutcome as a JSON-friendly map for
// inclusion in delegation.result FEP events. Returns nil when the outcome
// is nil so the field is omitted from the emitted payload.
func outcomeToFEPMap(o *AgentRunOutcome) map[string]any {
	if o == nil {
		return nil
	}
	m := map[string]any{}
	if o.Intent != "" {
		m["intent"] = o.Intent
	}
	if o.Summary != "" {
		m["summary"] = o.Summary
	}
	if len(o.Artifacts) > 0 {
		arts := make([]map[string]any, 0, len(o.Artifacts))
		for _, a := range o.Artifacts {
			am := map[string]any{"kind": a.Kind}
			if a.Provider != "" {
				am["provider"] = a.Provider
			}
			if a.URL != "" {
				am["url"] = a.URL
			}
			if a.Ref != "" {
				am["ref"] = a.Ref
			}
			if a.Title != "" {
				am["title"] = a.Title
			}
			arts = append(arts, am)
		}
		m["artifacts"] = arts
	}
	return m
}

// writeOutcomeLines appends a human-readable rendering of the outcome to b
// for inclusion in delegation callback prompts. Each line is prefixed with
// indent so it slots cleanly into single- and fan-out callback formats.
// Silent when outcome is nil or empty.
func writeOutcomeLines(b *strings.Builder, o *AgentRunOutcome, indent string) {
	if o == nil {
		return
	}
	if o.Intent != "" {
		b.WriteString(fmt.Sprintf("%sIntent: %s\n", indent, o.Intent))
	}
	if o.Summary != "" {
		b.WriteString(fmt.Sprintf("%sSummary: %s\n", indent, o.Summary))
	}
	for _, a := range o.Artifacts {
		// Format: "  - pr (github): https://... [ref] — title"
		line := fmt.Sprintf("%s- %s", indent, a.Kind)
		if a.Provider != "" {
			line += fmt.Sprintf(" (%s)", a.Provider)
		}
		if a.URL != "" {
			line += ": " + a.URL
		}
		if a.Ref != "" {
			line += " [" + a.Ref + "]"
		}
		if a.Title != "" {
			line += " — " + a.Title
		}
		b.WriteString(line + "\n")
	}
}
