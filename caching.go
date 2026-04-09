/*
Agent Runtime — Fantasy (Go)

Anthropic prompt caching integration.
Sets cache_control breakpoints on system messages, tool definitions,
and conversation messages to minimize input token costs.

Cache reads cost 90% less than uncached input on Anthropic.
*/
package main

import (
	"context"
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

// prepareStepWithCaching returns a PrepareStepFunction that marks stable
// context sections for Anthropic prompt caching.
//
// Strategy:
//   - Mark the last system message for caching (system prompt + Engram protocol)
//   - Mark the message just before the current user message (conversation history boundary)
//
// This gives us 2 cache breakpoints (Anthropic allows up to 4).
// Tool caching is handled separately via SetProviderOptions on tool definitions.
func prepareStepWithCaching() fantasy.PrepareStepFunction {
	return func(ctx context.Context, opts fantasy.PrepareStepFunctionOptions) (context.Context, fantasy.PrepareStepResult, error) {
		result := fantasy.PrepareStepResult{
			Messages: opts.Messages,
		}

		if len(result.Messages) == 0 {
			return ctx, result, nil
		}

		cacheOpts := anthropicCacheOptions()

		// Find and cache the last system message (system prompt is always first)
		lastSystemIdx := -1
		for i, msg := range result.Messages {
			if msg.Role == fantasy.MessageRoleSystem {
				lastSystemIdx = i
			}
		}
		if lastSystemIdx >= 0 {
			result.Messages[lastSystemIdx].ProviderOptions = cacheOpts
		}

		// Cache the boundary between old history and current turn.
		// The second-to-last message is typically the end of the previous
		// turn's history — stable content that benefits from caching.
		if len(result.Messages) >= 3 {
			result.Messages[len(result.Messages)-2].ProviderOptions = cacheOpts
		}

		return ctx, result, nil
	}
}

// applyToolCaching sets Anthropic cache_control on all tool definitions.
// Tool schemas are identical across all API calls, making them ideal
// cache targets (written once, read on every subsequent call).
func applyToolCaching(tools []fantasy.AgentTool) {
	cacheOpts := anthropicCacheOptions()
	for _, t := range tools {
		t.SetProviderOptions(cacheOpts)
	}
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
