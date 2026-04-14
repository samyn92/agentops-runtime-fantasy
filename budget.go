/*
Agent Runtime — Fantasy (Go)

Context window budget management.

Two layers of protection:
 1. Pre-flight budget allocation — estimates token usage per prompt layer
    (system prompt, tools, memory context, conversation) and trims the working
    memory to fit within a conversation budget BEFORE sending to the LLM.
 2. Reactive stop condition — halts the agent loop if actual InputTokens
    (reported by the API response) exceeds the budget.

Token estimation uses a chars/4 heuristic (~4 chars per token for English).
This is intentionally approximate — exact tiktoken counting is not worth the
dependency for pre-flight budgeting. The reactive guard catches any overshoot.
*/
package main

import (
	"log/slog"
	"strings"
	"sync"

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

	// Moonshot / Kimi
	"kimi-k2.5":              256_000,
	"kimi-k2-0905-preview":   256_000,
	"kimi-k2-0711-preview":   128_000,
	"kimi-k2-turbo-preview":  256_000,
	"kimi-k2-thinking":       256_000,
	"kimi-k2-thinking-turbo": 256_000,
	"moonshot-v1-8k":         8_000,
	"moonshot-v1-32k":        32_000,
	"moonshot-v1-128k":       128_000,
}

// DefaultContextWindow is used when the model is not in the lookup table.
const DefaultContextWindow int64 = 200_000

// DefaultBudgetFraction is the fraction of the context window used as the
// input token budget. 0.75 means the agent stops when a step uses 75%
// of the context window as input tokens, leaving 25% for output + safety.
const DefaultBudgetFraction = 0.75

// charsPerToken is the heuristic for token estimation (~4 chars per token
// for English text). Intentionally conservative (underestimates tokens slightly)
// to avoid hitting the actual limit.
const charsPerToken = 4

// contextWindowForModel returns the known context window size for a model string.
// Handles provider-prefixed model names (e.g. "kimi/kimi-k2.5" → "kimi-k2.5").
// Falls back to DefaultContextWindow if the model is not recognized.
func contextWindowForModel(model string) int64 {
	// Strip provider prefix if present (e.g. "kimi/kimi-k2.5" → "kimi-k2.5",
	// "anthropic/claude-sonnet-4" → "claude-sonnet-4").
	bare := model
	if idx := strings.LastIndex(model, "/"); idx >= 0 {
		bare = model[idx+1:]
	}

	// Exact match on bare model name
	if size, ok := modelContextWindows[bare]; ok {
		return size
	}
	// Try prefix matching for versioned model names (e.g. "claude-opus-4.6@20250901")
	for prefix, size := range modelContextWindows {
		if len(bare) > len(prefix) && bare[:len(prefix)] == prefix {
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

// ====================================================================
// Pre-flight budget allocation
// ====================================================================

// ContextBudget tracks estimated token usage across prompt layers.
// Updated per-turn and exposed to the FEP emitter for UI display.
type ContextBudget struct {
	mu sync.RWMutex

	ContextWindow int64 // total context window for the model
	BudgetTokens  int64 // usable budget (contextWindow * fraction)

	// Estimated tokens per layer (pre-flight, chars/4 heuristic)
	SystemPromptTokens  int64 // base system prompt + memory protocol
	ToolTokens          int64 // tool definitions (measured once)
	MemoryContextTokens int64 // per-turn memory context injection
	ConversationTokens  int64 // working memory messages
	PromptTokens        int64 // current user prompt + resource context

	// Actual tokens from last API response (post-flight)
	ActualInputTokens  int64
	ActualOutputTokens int64
	CacheReadTokens    int64
	CacheWriteTokens   int64
}

// NewContextBudget creates a budget tracker for the given model.
func NewContextBudget(model string, budgetFraction float64) *ContextBudget {
	cw := contextWindowForModel(model)
	return &ContextBudget{
		ContextWindow: cw,
		BudgetTokens:  int64(float64(cw) * budgetFraction),
	}
}

// EstimateTokens returns an approximate token count for a string.
// Uses the chars/4 heuristic — intentionally approximate.
func EstimateTokens(s string) int64 {
	if len(s) == 0 {
		return 0
	}
	return int64((len(s) + charsPerToken - 1) / charsPerToken)
}

// EstimateMessageTokens returns an approximate token count for a slice of messages.
func EstimateMessageTokens(msgs []fantasy.Message) int64 {
	var total int64
	for _, msg := range msgs {
		// Per-message overhead (role token, structure)
		total += 4
		for _, part := range msg.Content {
			if tp, ok := fantasy.AsMessagePart[fantasy.TextPart](part); ok {
				total += EstimateTokens(tp.Text)
			} else if tc, ok := fantasy.AsMessagePart[fantasy.ToolCallPart](part); ok {
				total += EstimateTokens(tc.ToolName)
				total += EstimateTokens(tc.Input)
			} else if tr, ok := fantasy.AsMessagePart[fantasy.ToolResultPart](part); ok {
				if text, ok := fantasy.AsToolResultOutputType[fantasy.ToolResultOutputContentText](tr.Output); ok {
					total += EstimateTokens(text.Text)
				}
			}
		}
	}
	return total
}

// ConversationBudget returns the token budget available for conversation history
// after subtracting fixed costs (system prompt, tools) and per-turn costs
// (memory context, user prompt). This is what working memory should trim to.
func (cb *ContextBudget) ConversationBudget() int64 {
	cb.mu.RLock()
	defer cb.mu.RUnlock()
	budget := cb.BudgetTokens - cb.SystemPromptTokens - cb.ToolTokens - cb.MemoryContextTokens - cb.PromptTokens
	if budget < 0 {
		budget = 0
	}
	return budget
}

// UpdateFixed records the estimated token count for fixed-cost layers
// (system prompt, tools). Called once at startup or when they change.
func (cb *ContextBudget) UpdateFixed(systemPrompt string, toolCount int) {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	cb.SystemPromptTokens = EstimateTokens(systemPrompt)
	// Tool definitions: rough estimate of ~150 tokens per tool (name + description + schema)
	cb.ToolTokens = int64(toolCount) * 150
}

// UpdatePerTurn records the estimated token count for per-turn dynamic layers.
// Called before each LLM call.
func (cb *ContextBudget) UpdatePerTurn(memoryCtx, resourceCtx, prompt string, conversationMsgs []fantasy.Message) {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	cb.MemoryContextTokens = EstimateTokens(memoryCtx)
	cb.PromptTokens = EstimateTokens(prompt) + EstimateTokens(resourceCtx)
	cb.ConversationTokens = EstimateMessageTokens(conversationMsgs)
}

// UpdateActual records the actual token usage from the API response.
// Called after each step or agent finish.
func (cb *ContextBudget) UpdateActual(usage fantasy.Usage) {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	cb.ActualInputTokens = usage.InputTokens
	cb.ActualOutputTokens = usage.OutputTokens
	cb.CacheReadTokens = usage.CacheReadTokens
	cb.CacheWriteTokens = usage.CacheCreationTokens
}

// Snapshot returns a copy of the current budget state for FEP emission.
func (cb *ContextBudget) Snapshot() ContextBudgetSnapshot {
	cb.mu.RLock()
	defer cb.mu.RUnlock()
	return ContextBudgetSnapshot{
		ContextWindow:       cb.ContextWindow,
		BudgetTokens:        cb.BudgetTokens,
		SystemPromptTokens:  cb.SystemPromptTokens,
		ToolTokens:          cb.ToolTokens,
		MemoryContextTokens: cb.MemoryContextTokens,
		ConversationTokens:  cb.ConversationTokens,
		PromptTokens:        cb.PromptTokens,
		ActualInputTokens:   cb.ActualInputTokens,
		ActualOutputTokens:  cb.ActualOutputTokens,
		CacheReadTokens:     cb.CacheReadTokens,
		CacheWriteTokens:    cb.CacheWriteTokens,
	}
}

// ContextBudgetSnapshot is an immutable copy of the budget state for serialization.
type ContextBudgetSnapshot struct {
	ContextWindow       int64 `json:"context_window"`
	BudgetTokens        int64 `json:"budget_tokens"`
	SystemPromptTokens  int64 `json:"system_prompt_tokens"`
	ToolTokens          int64 `json:"tool_tokens"`
	MemoryContextTokens int64 `json:"memory_context_tokens"`
	ConversationTokens  int64 `json:"conversation_tokens"`
	PromptTokens        int64 `json:"prompt_tokens"`
	ActualInputTokens   int64 `json:"actual_input_tokens"`
	ActualOutputTokens  int64 `json:"actual_output_tokens"`
	CacheReadTokens     int64 `json:"cache_read_tokens"`
	CacheWriteTokens    int64 `json:"cache_write_tokens"`
}

// UsedTokens returns the total estimated pre-flight token usage.
func (s ContextBudgetSnapshot) UsedTokens() int64 {
	return s.SystemPromptTokens + s.ToolTokens + s.MemoryContextTokens + s.ConversationTokens + s.PromptTokens
}

// UsagePercent returns the percentage of the context window used (0-100).
func (s ContextBudgetSnapshot) UsagePercent() float64 {
	if s.ContextWindow == 0 {
		return 0
	}
	// Use actual tokens if available (more accurate), fall back to estimate
	used := s.ActualInputTokens
	if used == 0 {
		used = s.UsedTokens()
	}
	return float64(used) / float64(s.ContextWindow) * 100
}
