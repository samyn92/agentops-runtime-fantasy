/*
Agent Runtime — Fantasy (Go)

Context window budget management.
Prevents within-turn context blowup by monitoring input token growth
and stopping the agent loop before hitting the model's context limit.
*/
package main

import (
	"log/slog"

	"charm.land/fantasy"
)

// Known context window sizes for common models.
// Used when the model doesn't report its context window.
var modelContextWindows = map[string]int64{
	// Anthropic
	"claude-opus-4":     200_000,
	"claude-opus-4.5":   200_000,
	"claude-opus-4.6":   200_000,
	"claude-sonnet-4":   200_000,
	"claude-sonnet-4.5": 200_000,
	"claude-sonnet-4.6": 200_000,
	"claude-haiku-3.5":  200_000,
	"claude-haiku-4.5":  200_000,

	// OpenAI
	"gpt-4o":       128_000,
	"gpt-4o-mini":  128_000,
	"gpt-4.1":      1_047_576,
	"gpt-4.1-mini": 1_047_576,
	"gpt-4.1-nano": 1_047_576,

	// Google
	"gemini-2.5-pro":   1_048_576,
	"gemini-2.5-flash": 1_048_576,
}

// DefaultContextWindow is used when the model is not in the lookup table.
const DefaultContextWindow int64 = 200_000

// DefaultBudgetFraction is the fraction of the context window used as the
// input token budget. 0.75 means the agent stops when a step uses 75%
// of the context window as input tokens, leaving 25% for output + safety.
const DefaultBudgetFraction = 0.75

// contextWindowForModel returns the known context window size for a model string.
// Falls back to DefaultContextWindow if the model is not recognized.
func contextWindowForModel(model string) int64 {
	if size, ok := modelContextWindows[model]; ok {
		return size
	}
	// Try prefix matching for versioned model names (e.g. "claude-opus-4.6@20250901")
	for prefix, size := range modelContextWindows {
		if len(model) > len(prefix) && model[:len(prefix)] == prefix {
			return size
		}
	}
	return DefaultContextWindow
}

// InputTokenBudget returns a StopCondition that halts the agent loop when
// any single step's InputTokens exceeds the budget. Since each step re-sends
// the full context, the last step's InputTokens represents the current total
// context size.
//
// budgetTokens = contextWindow * fraction (e.g. 200k * 0.75 = 150k)
func InputTokenBudget(budgetTokens int64) fantasy.StopCondition {
	return func(steps []fantasy.StepResult) bool {
		if len(steps) == 0 {
			return false
		}
		lastStep := steps[len(steps)-1]
		if lastStep.Usage.InputTokens >= budgetTokens {
			slog.Warn("input token budget exceeded — stopping agent loop",
				"input_tokens", lastStep.Usage.InputTokens,
				"budget", budgetTokens,
				"step", len(steps),
			)
			return true
		}
		return false
	}
}
