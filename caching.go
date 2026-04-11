/*
Agent Runtime — Fantasy (Go)

Anthropic prompt caching helpers.
Sets cache_control breakpoints on tool definitions to minimize input token costs.

Cache reads cost 90% less than uncached input on Anthropic.

NOTE: System message and conversation boundary caching is now handled in
context_injection.go via prepareStepWithContextInjection, which composes
caching with memory/resource context injection.
*/
package main

import (
	"strings"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/anthropic"
)

// anthropicCacheOptions returns the ProviderOptions that mark a message
// or tool for Anthropic ephemeral caching (5-minute TTL, refreshed on use).
func anthropicCacheOptions() fantasy.ProviderOptions {
	return fantasy.ProviderOptions{
		anthropic.Name: &anthropic.ProviderCacheControlOptions{
			CacheControl: anthropic.CacheControl{Type: "ephemeral"},
		},
	}
}

// applyToolCaching sets Anthropic cache_control on the last tool definition.
// Anthropic allows at most 4 cache breakpoints per request. Marking just the
// last tool is enough — the API caches everything up to and including that
// breakpoint, so the entire tools array is cached with a single breakpoint.
func applyToolCaching(tools []fantasy.AgentTool) {
	if len(tools) == 0 {
		return
	}
	cacheOpts := anthropicCacheOptions()
	tools[len(tools)-1].SetProviderOptions(cacheOpts)
}

// isAnthropicProvider checks if the primary model uses Anthropic.
func isAnthropicProvider(cfg *Config) bool {
	model := cfg.PrimaryModel
	provider := cfg.PrimaryProvider

	// Check explicit provider
	if strings.EqualFold(provider, "anthropic") {
		return true
	}

	// Check model prefix (e.g. "anthropic/claude-opus-4.6")
	if strings.HasPrefix(strings.ToLower(model), "anthropic/") {
		return true
	}

	// Check model name patterns
	lower := strings.ToLower(model)
	return strings.HasPrefix(lower, "claude-")
}
