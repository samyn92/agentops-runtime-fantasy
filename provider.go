/*
Agent Runtime — Fantasy (Go)

Provider resolution: maps provider names + env var API keys
to Fantasy SDK provider instances.
*/
package main

import (
	"context"
	"fmt"
	"os"
	"strings"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/anthropic"
	"charm.land/fantasy/providers/google"
	"charm.land/fantasy/providers/openai"
	"charm.land/fantasy/providers/openaicompat"
	"charm.land/fantasy/providers/openrouter"
)

// resolveProvider creates a Fantasy provider from a name and env-based API key.
func resolveProvider(name string) (fantasy.Provider, error) {
	envKey := fmt.Sprintf("%s_API_KEY", strings.ToUpper(name))
	apiKey := os.Getenv(envKey)
	if apiKey == "" {
		return nil, fmt.Errorf("no API key found for provider %s (env: %s)", name, envKey)
	}

	switch strings.ToLower(name) {
	case "anthropic":
		return anthropic.New(anthropic.WithAPIKey(apiKey))
	case "openai":
		return openai.New(openai.WithAPIKey(apiKey))
	case "google", "gemini":
		return google.New(google.WithGeminiAPIKey(apiKey))
	case "openrouter":
		return openrouter.New(openrouter.WithAPIKey(apiKey))
	default:
		// Treat unknown providers as OpenAI-compatible
		baseURL := os.Getenv(fmt.Sprintf("%s_BASE_URL", strings.ToUpper(name)))
		if baseURL == "" {
			return nil, fmt.Errorf("unknown provider %q: set %s_BASE_URL for OpenAI-compatible providers", name, strings.ToUpper(name))
		}
		return openaicompat.New(
			openaicompat.WithAPIKey(apiKey),
			openaicompat.WithBaseURL(baseURL),
		)
	}
}

// resolveModel parses "provider/model" and returns a Fantasy LanguageModel.
// If the model string has no "/" prefix, primaryProvider is prepended.
func resolveModel(ctx context.Context, modelStr string, providers map[string]fantasy.Provider, primaryProvider string) (fantasy.LanguageModel, error) {
	parts := strings.SplitN(modelStr, "/", 2)
	var providerName, modelID string
	if len(parts) == 2 {
		providerName = parts[0]
		modelID = parts[1]
	} else {
		// No provider prefix — use primaryProvider from config
		if primaryProvider == "" {
			return nil, fmt.Errorf("model %q has no provider prefix and no primaryProvider configured", modelStr)
		}
		providerName = primaryProvider
		modelID = modelStr
	}

	provider, ok := providers[providerName]
	if !ok {
		return nil, fmt.Errorf("provider %q not configured", providerName)
	}

	return provider.LanguageModel(ctx, modelID)
}
