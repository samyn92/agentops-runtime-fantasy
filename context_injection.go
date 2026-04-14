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
	mu               sync.RWMutex
	memoryContext    string // formatted <memory:sessions> + <memory:context> from engram
	resourceContext  string // formatted resource context from console selections
	platformProtocol string // stable platform protocol (identity + delegation + memory instructions)
}

// Set stores the per-turn context. Called once per turn before the LLM call.
func (ci *contextInjector) Set(memoryCtx, resourceCtx string) {
	ci.mu.Lock()
	defer ci.mu.Unlock()
	ci.memoryContext = memoryCtx
	ci.resourceContext = resourceCtx
}

// SetPlatformProtocol stores the platform protocol. Called once at startup.
// This is stable across turns and gets its own cached system message part.
func (ci *contextInjector) SetPlatformProtocol(protocol string) {
	ci.mu.Lock()
	defer ci.mu.Unlock()
	ci.platformProtocol = protocol
}

// Clear resets the per-turn context. Called after each turn completes.
func (ci *contextInjector) Clear() {
	ci.mu.Lock()
	defer ci.mu.Unlock()
	ci.memoryContext = ""
	ci.resourceContext = ""
	// NOTE: platformProtocol is NOT cleared — it's stable across turns.
}

// get returns the current per-turn context. Called by PrepareStep on each step.
func (ci *contextInjector) get() (platformProtocol, memoryCtx, resourceCtx string) {
	ci.mu.RLock()
	defer ci.mu.RUnlock()
	return ci.platformProtocol, ci.memoryContext, ci.resourceContext
}

// prepareStepWithContextInjection returns a PrepareStepFunction that:
//  1. Injects platform protocol, memory context, and resource context as
//     additional TextParts in the system message (Messages[0]).
//  2. Applies Anthropic prompt caching on stable parts.
//
// System message structure:
//
//	Part 0a: platform protocol (identity + delegation + memory instructions) [cached]
//	Part 0b: user's system prompt (from spec.systemPrompt) [cached]
//	Part 1:  memory context (per-turn, from engram) [uncached]
//	Part 2:  resource context (per-turn, from console) [uncached]
//
// Cache breakpoint budget (Anthropic allows 4):
//
//	#1 — tool definitions (applied in applyToolCaching)
//	#2 — last stable system message part (platform protocol or user prompt)
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

		// ── Step 1: Inject platform protocol + memory + resource context into system message ──

		platformProtocol, memoryCtx, resourceCtx := injector.get()
		hasInjection := platformProtocol != "" || memoryCtx != "" || resourceCtx != ""

		if hasInjection && len(result.Messages) > 0 && result.Messages[0].Role == fantasy.MessageRoleSystem {
			// Extract the base system prompt text from the existing system message.
			// It should be the first (and currently only) TextPart — the user's systemPrompt.
			var userPrompt string
			if len(result.Messages[0].Content) > 0 {
				if tp, ok := fantasy.AsMessagePart[fantasy.TextPart](result.Messages[0].Content[0]); ok {
					userPrompt = tp.Text
				}
			}

			// Build multi-part system message:
			//   Part 0a: platform protocol (stable — cached on Anthropic)
			//   Part 0b: user system prompt (stable — cached on Anthropic)
			//   Part 1:  memory context (per-turn — not cached)
			//   Part 2:  resource context (per-turn — not cached)
			parts := make([]fantasy.MessagePart, 0, 4)

			// Part 0a: platform protocol (identity + delegation + memory instructions)
			if platformProtocol != "" {
				platformPart := fantasy.TextPart{Text: platformProtocol}
				parts = append(parts, platformPart)
				slog.Debug("platform protocol injected into system message", "length", len(platformProtocol))
			}

			// Part 0b: user's system prompt (verbatim from CRD spec.systemPrompt)
			// This gets the Anthropic cache breakpoint since it's the last stable part.
			userPart := fantasy.TextPart{Text: userPrompt}
			if isAnthropic {
				userPart.ProviderOptions = fantasy.ProviderOptions{
					anthropic.Name: &anthropic.ProviderCacheControlOptions{
						CacheControl: anthropic.CacheControl{Type: "ephemeral"},
					},
				}
			}
			parts = append(parts, userPart)

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

			// If we didn't inject context (no platform/memory/resource), still cache the
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
