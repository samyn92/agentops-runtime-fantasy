/*
Agent Runtime — Fantasy (Go)

Provider resolution: maps ProviderEntry config to Fantasy SDK provider instances.
Uses type-based dispatch from Provider CRs.
*/
package main

import (
	"context"
	"fmt"
	"os"
	"strings"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/anthropic"
	"charm.land/fantasy/providers/azure"
	"charm.land/fantasy/providers/bedrock"
	"charm.land/fantasy/providers/google"
	"charm.land/fantasy/providers/openai"
	"charm.land/fantasy/providers/openaicompat"
	"charm.land/fantasy/providers/openrouter"
)

// resolveProvider creates a Fantasy provider from a ProviderEntry.
// It uses the entry's Type field for type-based dispatch with full SDK option wiring.
func resolveProvider(entry ProviderEntry) (fantasy.Provider, error) {
	envKey := fmt.Sprintf("%s_API_KEY", strings.ToUpper(entry.Name))
	apiKey := os.Getenv(envKey)
	if apiKey == "" {
		return nil, fmt.Errorf("no API key for provider %s (env: %s)", entry.Name, envKey)
	}

	switch entry.Type {
	case "anthropic":
		return newAnthropicProvider(entry, apiKey)
	case "openai":
		return newOpenAIProvider(entry, apiKey)
	case "google":
		return newGoogleProvider(entry, apiKey)
	case "azure":
		return newAzureProvider(entry, apiKey)
	case "bedrock":
		return newBedrockProvider(entry, apiKey)
	case "openrouter":
		return newOpenRouterProvider(entry, apiKey)
	case "openaicompat":
		return newOpenAICompatProvider(entry, apiKey)
	default:
		return nil, fmt.Errorf("unknown provider type %q for provider %q", entry.Type, entry.Name)
	}
}

// --------------------------------------------------------------------
// Type-based provider constructors
// --------------------------------------------------------------------

func newAnthropicProvider(entry ProviderEntry, apiKey string) (fantasy.Provider, error) {
	opts := []anthropic.Option{anthropic.WithAPIKey(apiKey)}
	if entry.BaseURL != "" {
		opts = append(opts, anthropic.WithBaseURL(entry.BaseURL))
	}
	if entry.Vertex != nil {
		opts = append(opts, anthropic.WithVertex(entry.Vertex.Project, entry.Vertex.Location))
	}
	if entry.Bedrock {
		opts = append(opts, anthropic.WithBedrock())
	}
	if len(entry.Headers) > 0 {
		opts = append(opts, anthropic.WithHeaders(entry.Headers))
	}
	return anthropic.New(opts...)
}

func newOpenAIProvider(entry ProviderEntry, apiKey string) (fantasy.Provider, error) {
	opts := []openai.Option{openai.WithAPIKey(apiKey)}
	if entry.BaseURL != "" {
		opts = append(opts, openai.WithBaseURL(entry.BaseURL))
	}
	if entry.Organization != "" {
		opts = append(opts, openai.WithOrganization(entry.Organization))
	}
	if entry.Project != "" {
		opts = append(opts, openai.WithProject(entry.Project))
	}
	if entry.UseResponsesAPI {
		opts = append(opts, openai.WithUseResponsesAPI())
	}
	if len(entry.Headers) > 0 {
		opts = append(opts, openai.WithHeaders(entry.Headers))
	}
	return openai.New(opts...)
}

func newGoogleProvider(entry ProviderEntry, apiKey string) (fantasy.Provider, error) {
	var opts []google.Option
	if entry.Vertex != nil {
		// Vertex AI mode — uses ADC or service account, not API key.
		opts = append(opts, google.WithVertex(entry.Vertex.Project, entry.Vertex.Location))
	} else {
		// Gemini API key mode.
		opts = append(opts, google.WithGeminiAPIKey(apiKey))
	}
	if entry.BaseURL != "" {
		opts = append(opts, google.WithBaseURL(entry.BaseURL))
	}
	if len(entry.Headers) > 0 {
		opts = append(opts, google.WithHeaders(entry.Headers))
	}
	return google.New(opts...)
}

func newAzureProvider(entry ProviderEntry, apiKey string) (fantasy.Provider, error) {
	opts := []azure.Option{azure.WithAPIKey(apiKey)}
	if entry.BaseURL != "" {
		opts = append(opts, azure.WithBaseURL(entry.BaseURL))
	}
	if entry.AzureAPIVersion != "" {
		opts = append(opts, azure.WithAPIVersion(entry.AzureAPIVersion))
	}
	if entry.UseResponsesAPI {
		opts = append(opts, azure.WithUseResponsesAPI())
	}
	if len(entry.Headers) > 0 {
		opts = append(opts, azure.WithHeaders(entry.Headers))
	}
	return azure.New(opts...)
}

func newBedrockProvider(entry ProviderEntry, apiKey string) (fantasy.Provider, error) {
	opts := []bedrock.Option{bedrock.WithAPIKey(apiKey)}
	if entry.BaseURL != "" {
		opts = append(opts, bedrock.WithBaseURL(entry.BaseURL))
	}
	if len(entry.Headers) > 0 {
		opts = append(opts, bedrock.WithHeaders(entry.Headers))
	}
	return bedrock.New(opts...)
}

func newOpenRouterProvider(entry ProviderEntry, apiKey string) (fantasy.Provider, error) {
	opts := []openrouter.Option{openrouter.WithAPIKey(apiKey)}
	// OpenRouter uses a fixed base URL — no WithBaseURL available.
	if len(entry.Headers) > 0 {
		opts = append(opts, openrouter.WithHeaders(entry.Headers))
	}
	return openrouter.New(opts...)
}

func newOpenAICompatProvider(entry ProviderEntry, apiKey string) (fantasy.Provider, error) {
	baseURL := entry.BaseURL
	if baseURL == "" {
		return nil, fmt.Errorf("openaicompat provider %q requires baseURL", entry.Name)
	}
	opts := []openaicompat.Option{
		openaicompat.WithAPIKey(apiKey),
		openaicompat.WithBaseURL(baseURL),
		openaicompat.WithName(entry.Name),
	}
	if entry.UseResponsesAPI {
		opts = append(opts, openaicompat.WithUseResponsesAPI())
	}
	if len(entry.Headers) > 0 {
		opts = append(opts, openaicompat.WithHeaders(entry.Headers))
	}
	return openaicompat.New(opts...)
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
