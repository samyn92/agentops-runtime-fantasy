/*
Agent Runtime — Fantasy (Go)

Resource context injection (per-turn, user-driven).
Formats dynamic context from the console's resource browser selections
into a human-readable block injected into the system message via PrepareStep.

Caps applied:
  - MaxResourceContextItems: at most 20 items per turn
  - MaxDescriptionChars: descriptions truncated to 500 chars
*/
package main

import (
	"fmt"
	"log/slog"
	"strings"
)

// MaxResourceContextItems caps the number of resource items per turn.
// Beyond this, the model gets diminishing returns and the context bloats.
const MaxResourceContextItems = 20

// MaxDescriptionChars caps the Description field per item.
const MaxDescriptionChars = 500

// ResourceContext is the per-turn dynamic context sent by the console.
// Each entry describes a resource item the user selected in the resource browser.
type ResourceContext struct {
	ResourceName string `json:"resource_name"`         // AgentResource name
	Kind         string `json:"kind"`                  // e.g. "github-repo"
	ItemType     string `json:"item_type"`             // file, commit, branch, issue, merge_request
	Path         string `json:"path,omitempty"`        // file path or branch name
	Ref          string `json:"ref,omitempty"`         // git ref
	Title        string `json:"title,omitempty"`       // issue/MR title
	Number       int    `json:"number,omitempty"`      // issue/MR number
	SHA          string `json:"sha,omitempty"`         // commit SHA
	URL          string `json:"url,omitempty"`         // web URL for the item
	Description  string `json:"description,omitempty"` // extra description text
}

// formatResourceContext converts a list of ResourceContext items into a
// human-readable string for system message injection.
func formatResourceContext(items []ResourceContext) string {
	if len(items) == 0 {
		return ""
	}

	// Cap the number of items
	if len(items) > MaxResourceContextItems {
		slog.Warn("resource context capped",
			"requested", len(items),
			"max", MaxResourceContextItems,
		)
		items = items[:MaxResourceContextItems]
	}

	var sb strings.Builder
	sb.WriteString("[Resource Context — the user selected the following items for this message]\n\n")

	for _, item := range items {
		// Truncate description if too long
		if len(item.Description) > MaxDescriptionChars {
			item.Description = item.Description[:MaxDescriptionChars] + "..."
		}

		switch item.ItemType {
		case "file":
			sb.WriteString(fmt.Sprintf("- File: `%s`", item.Path))
			if item.Ref != "" {
				sb.WriteString(fmt.Sprintf(" (ref: %s)", item.Ref))
			}
			sb.WriteString(fmt.Sprintf(" from %s\n", item.ResourceName))
		case "commit":
			sha := item.SHA
			if len(sha) > 7 {
				sha = sha[:7]
			}
			sb.WriteString(fmt.Sprintf("- Commit: %s", sha))
			if item.Title != "" {
				sb.WriteString(fmt.Sprintf(" — %s", item.Title))
			}
			sb.WriteString(fmt.Sprintf(" from %s\n", item.ResourceName))
		case "branch":
			sb.WriteString(fmt.Sprintf("- Branch: `%s` from %s\n", item.Path, item.ResourceName))
		case "issue":
			sb.WriteString(fmt.Sprintf("- Issue #%d: %s", item.Number, item.Title))
			if item.URL != "" {
				sb.WriteString(fmt.Sprintf(" (%s)", item.URL))
			}
			sb.WriteString(fmt.Sprintf(" from %s\n", item.ResourceName))
		case "merge_request":
			sb.WriteString(fmt.Sprintf("- MR/PR #%d: %s", item.Number, item.Title))
			if item.URL != "" {
				sb.WriteString(fmt.Sprintf(" (%s)", item.URL))
			}
			sb.WriteString(fmt.Sprintf(" from %s\n", item.ResourceName))
		default:
			sb.WriteString(fmt.Sprintf("- %s: %s from %s\n", item.ItemType, item.Path, item.ResourceName))
		}

		// Append description if present
		if item.Description != "" {
			sb.WriteString(fmt.Sprintf("  %s\n", item.Description))
		}
	}

	sb.WriteString("\n")
	return sb.String()
}
