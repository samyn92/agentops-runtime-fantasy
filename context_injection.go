/*
Agent Runtime — Fantasy (Go)

Context injection via PrepareStep.

Moves memory context (short-term + long-term) and resource context out of the
user message and into the system message as separate TextParts. This gives:

 1. Correct semantics — the LLM sees memory as background knowledge (system),
    not as something the user said (user message).
 2. Better caching — the base system prompt (part 1) stays cached across turns.
    Memory context (part 2) gets its own Anthropic cache breakpoint.
 3. Clean user messages — the actual user prompt is uncluttered.

Architecture:
  - contextInjector holds per-turn context (set before each LLM call, cleared after).
  - prepareStepWithContextInjection returns a PrepareStepFunction that reads from
    the injector and builds a multi-part system message.
  - Composes with Anthropic caching — base prompt part gets ephemeral cache,
    memory context part is uncached (changes per turn), conversation boundary
    gets its own breakpoint.
*/
package main

import (
	"context"
	"log/slog"
	"sync"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/anthropic"
)

// contextInjector holds per-turn dynamic context that gets injected into the
// system message via PrepareStep. Thread-safe — set by prompt handlers,
// read by PrepareStep callbacks which may run on different goroutines.
type contextInjector struct {
	mu              sync.RWMutex
	memoryContext   string // formatted <memory:sessions> + <memory:context> from engram
	resourceContext string // formatted resource context from console selections
}

// Set stores the per-turn context. Called once per turn before the LLM call.
func (ci *contextInjector) Set(memoryCtx, resourceCtx string) {
	ci.mu.Lock()
	defer ci.mu.Unlock()
	ci.memoryContext = memoryCtx
	ci.resourceContext = resourceCtx
}

// Clear resets the per-turn context. Called after each turn completes.
func (ci *contextInjector) Clear() {
	ci.mu.Lock()
	defer ci.mu.Unlock()
	ci.memoryContext = ""
	ci.resourceContext = ""
}

// get returns the current per-turn context. Called by PrepareStep on each step.
func (ci *contextInjector) get() (memoryCtx, resourceCtx string) {
	ci.mu.RLock()
	defer ci.mu.RUnlock()
	return ci.memoryContext, ci.resourceContext
}

// prepareStepWithContextInjection returns a PrepareStepFunction that:
//  1. Injects memory context and resource context as additional TextParts in the
//     system message (Messages[0]).
//  2. Applies Anthropic prompt caching on stable parts.
//
// The base system prompt (TextPart 0) gets cached — it's stable across turns.
// Memory context and resource context change per turn so they're not cached.
// The conversation history boundary gets its own cache breakpoint.
//
// Cache breakpoint budget (Anthropic allows 4):
//
//	#1 — tool definitions (applied in applyToolCaching)
//	#2 — base system prompt TextPart (stable, high reuse)
//	#3 — conversation history boundary (second-to-last message)
//	#4 — available for future use
func prepareStepWithContextInjection(injector *contextInjector, isAnthropic bool) fantasy.PrepareStepFunction {
	return func(ctx context.Context, opts fantasy.PrepareStepFunctionOptions) (context.Context, fantasy.PrepareStepResult, error) {
		result := fantasy.PrepareStepResult{
			Messages: opts.Messages,
		}

		if len(result.Messages) == 0 {
			return ctx, result, nil
		}

		// ── Step 1: Inject memory + resource context into system message ──

		memoryCtx, resourceCtx := injector.get()
		hasInjection := memoryCtx != "" || resourceCtx != ""

		if hasInjection && len(result.Messages) > 0 && result.Messages[0].Role == fantasy.MessageRoleSystem {
			// Extract the base system prompt text from the existing system message.
			// It should be the first (and currently only) TextPart.
			var basePrompt string
			if len(result.Messages[0].Content) > 0 {
				if tp, ok := fantasy.AsMessagePart[fantasy.TextPart](result.Messages[0].Content[0]); ok {
					basePrompt = tp.Text
				}
			}

			// Build multi-part system message:
			//   Part 0: base system prompt (stable — cached on Anthropic)
			//   Part 1: memory context (per-turn — not cached)
			//   Part 2: resource context (per-turn — not cached)
			parts := make([]fantasy.MessagePart, 0, 3)

			// Part 0: base system prompt
			basePart := fantasy.TextPart{Text: basePrompt}
			if isAnthropic {
				// Cache the base prompt — it's identical across turns
				basePart.ProviderOptions = fantasy.ProviderOptions{
					anthropic.Name: &anthropic.ProviderCacheControlOptions{
						CacheControl: anthropic.CacheControl{Type: "ephemeral"},
					},
				}
			}
			parts = append(parts, basePart)

			// Part 1: memory context (sessions + observations)
			if memoryCtx != "" {
				parts = append(parts, fantasy.TextPart{Text: memoryCtx})
				slog.Debug("memory context injected into system message", "length", len(memoryCtx))
			}

			// Part 2: resource context (console selections)
			if resourceCtx != "" {
				parts = append(parts, fantasy.TextPart{Text: resourceCtx})
				slog.Debug("resource context injected into system message", "length", len(resourceCtx))
			}

			// Replace the system message with the multi-part version
			result.Messages[0] = fantasy.Message{
				Role:    fantasy.MessageRoleSystem,
				Content: parts,
			}
		}

		// ── Step 2: Anthropic caching on conversation history boundary ──

		if isAnthropic {
			cacheOpts := anthropicCacheOptions()

			// If we didn't inject context (no memory/resource), still cache the
			// system message at the message level (same as before).
			if !hasInjection {
				lastSystemIdx := -1
				for i, msg := range result.Messages {
					if msg.Role == fantasy.MessageRoleSystem {
						lastSystemIdx = i
					}
				}
				if lastSystemIdx >= 0 {
					result.Messages[lastSystemIdx].ProviderOptions = cacheOpts
				}
			}

			// Cache the conversation history boundary (second-to-last message).
			// This is stable content from previous turns.
			if len(result.Messages) >= 3 {
				result.Messages[len(result.Messages)-2].ProviderOptions = cacheOpts
			}
		}

		return ctx, result, nil
	}
}
