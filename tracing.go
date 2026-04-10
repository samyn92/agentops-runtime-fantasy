/*
Agent Runtime — Fantasy (Go)

OpenTelemetry tracing: SDK init, tracer, span helpers.

Reads OTEL_EXPORTER_OTLP_ENDPOINT (injected by the operator) to decide
whether to export traces. When the env var is empty, a no-op tracer is
used — zero overhead, no behavioral change.

Follows the GenAI Semantic Conventions for LLM-specific attributes.
*/
package main

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"strings"

	"charm.land/fantasy"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.26.0"
	"go.opentelemetry.io/otel/trace"
)

// Package-level tracer used by all instrumentation points.
var tracer trace.Tracer

// initTracing sets up the OTel TracerProvider with an OTLP gRPC exporter.
// Returns a shutdown function that must be called before process exit.
//
// When OTEL_EXPORTER_OTLP_ENDPOINT is empty, returns a no-op shutdown
// and uses the global no-op tracer — zero overhead in non-instrumented deployments.
func initTracing(ctx context.Context, agentName, agentNamespace, agentMode string) (shutdown func(context.Context) error, err error) {
	endpoint := os.Getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
	if endpoint == "" {
		// No endpoint configured — use no-op tracer
		tracer = otel.Tracer("agentops-runtime")
		slog.Info("tracing disabled (OTEL_EXPORTER_OTLP_ENDPOINT not set)")
		return func(context.Context) error { return nil }, nil
	}

	// Build resource attributes
	attrs := []attribute.KeyValue{
		semconv.ServiceName("agentops-runtime"),
		semconv.ServiceVersion("0.1.0"),
		attribute.String("agent.name", agentName),
		attribute.String("agent.mode", agentMode),
	}
	if agentNamespace != "" {
		attrs = append(attrs, attribute.String("agent.namespace", agentNamespace))
	}

	res, err := resource.New(ctx,
		resource.WithAttributes(attrs...),
		resource.WithProcessRuntimeName(),
		resource.WithHost(),
	)
	if err != nil {
		return nil, fmt.Errorf("create otel resource: %w", err)
	}

	// OTLP gRPC exporter — connects to Tempo (or any OTLP-compatible collector)
	exporter, err := otlptracegrpc.New(ctx,
		otlptracegrpc.WithEndpoint(endpoint),
		otlptracegrpc.WithInsecure(), // in-cluster communication
	)
	if err != nil {
		return nil, fmt.Errorf("create otlp exporter: %w", err)
	}

	tp := sdktrace.NewTracerProvider(
		sdktrace.WithBatcher(exporter),
		sdktrace.WithResource(res),
		// Always sample — we want every agent execution traced.
		// Revisit if Tempo storage becomes a concern (per PLAN_otel.md open question #3).
		sdktrace.WithSampler(sdktrace.AlwaysSample()),
	)

	otel.SetTracerProvider(tp)
	tracer = tp.Tracer("agentops-runtime")

	slog.Info("tracing enabled",
		"endpoint", endpoint,
		"agent", agentName,
		"mode", agentMode,
	)

	return tp.Shutdown, nil
}

// ────────────────────────────────────────────────────────────────────
// Span attribute keys — GenAI Semantic Conventions + AgentOps custom
// ────────────────────────────────────────────────────────────────────

// Agent-level attributes
var (
	attrAgentName      = attribute.Key("agent.name")
	attrAgentNamespace = attribute.Key("agent.namespace")
	attrAgentMode      = attribute.Key("agent.mode")
	attrAgentRun       = attribute.Key("agent.run")
)

// GenAI semantic convention attributes
var (
	attrGenAISystem          = attribute.Key("gen_ai.system")
	attrGenAIRequestModel    = attribute.Key("gen_ai.request.model")
	attrGenAIResponseModel   = attribute.Key("gen_ai.response.model")
	attrGenAIInputTokens     = attribute.Key("gen_ai.usage.input_tokens")
	attrGenAIOutputTokens    = attribute.Key("gen_ai.usage.output_tokens")
	attrGenAIReasoningTokens = attribute.Key("gen_ai.usage.reasoning_tokens")
	attrGenAIFinishReason    = attribute.Key("gen_ai.response.finish_reasons")
	attrGenAICacheCreate     = attribute.Key("gen_ai.usage.cache_creation_tokens")
	attrGenAICacheRead       = attribute.Key("gen_ai.usage.cache_read_tokens")
)

// Step attributes
var (
	attrStepNumber       = attribute.Key("step.number")
	attrStepFinishReason = attribute.Key("step.finish_reason")
	attrStepToolCalls    = attribute.Key("step.tool_call_count")
)

// Tool attributes
var (
	attrToolName      = attribute.Key("tool.name")
	attrToolType      = attribute.Key("tool.type")
	attrToolMCPServer = attribute.Key("tool.mcp.server")
	attrToolTransport = attribute.Key("tool.mcp.transport")
	attrToolError     = attribute.Key("tool.error")
	attrToolDuration  = attribute.Key("tool.duration_ms")
)

// Engram attributes
var (
	attrEngramOp      = attribute.Key("engram.operation")
	attrEngramProject = attribute.Key("engram.project")
)

// ────────────────────────────────────────────────────────────────────
// Span helpers
// ────────────────────────────────────────────────────────────────────

// traceIDFromContext extracts the trace ID hex string from the current span context.
// Returns empty string if there is no active span or the trace ID is invalid.
func traceIDFromContext(ctx context.Context) string {
	sc := trace.SpanFromContext(ctx).SpanContext()
	if !sc.TraceID().IsValid() {
		return ""
	}
	return sc.TraceID().String()
}

// recordError sets the span status to Error and records the error as an event.
func recordError(span trace.Span, err error) {
	span.SetStatus(codes.Error, err.Error())
	span.RecordError(err)
}

// setLLMResultAttributes sets GenAI semantic convention attributes from a Fantasy AgentResult.
func setLLMResultAttributes(span trace.Span, result *fantasy.AgentResult, model string) {
	if result == nil {
		return
	}
	span.SetAttributes(
		attrGenAIResponseModel.String(model),
		attrGenAIInputTokens.Int64(result.TotalUsage.InputTokens),
		attrGenAIOutputTokens.Int64(result.TotalUsage.OutputTokens),
	)
	if result.TotalUsage.ReasoningTokens > 0 {
		span.SetAttributes(attrGenAIReasoningTokens.Int64(result.TotalUsage.ReasoningTokens))
	}
	if result.TotalUsage.CacheCreationTokens > 0 {
		span.SetAttributes(attrGenAICacheCreate.Int64(result.TotalUsage.CacheCreationTokens))
	}
	if result.TotalUsage.CacheReadTokens > 0 {
		span.SetAttributes(attrGenAICacheRead.Int64(result.TotalUsage.CacheReadTokens))
	}
}

// detectGenAISystem infers the gen_ai.system value from model/provider names.
func detectGenAISystem(model, provider string) string {
	lower := strings.ToLower(provider)
	switch {
	case lower == "anthropic":
		return "anthropic"
	case lower == "openai":
		return "openai"
	case lower == "google" || lower == "gemini":
		return "google"
	case lower == "openrouter":
		return "openrouter"
	}
	// Infer from model name
	m := strings.ToLower(model)
	switch {
	case strings.HasPrefix(m, "claude") || strings.HasPrefix(m, "anthropic/"):
		return "anthropic"
	case strings.HasPrefix(m, "gpt") || strings.HasPrefix(m, "openai/"):
		return "openai"
	case strings.HasPrefix(m, "gemini") || strings.HasPrefix(m, "google/"):
		return "google"
	}
	return provider
}

// classifyToolType returns a tool type string for span attributes.
func classifyToolType(toolName string) string {
	switch {
	case strings.HasPrefix(toolName, "mcp_"):
		return "mcp"
	case strings.HasPrefix(toolName, "mem_"):
		return "memory"
	case strings.HasPrefix(toolName, "git_") || toolName == "git_commit" || toolName == "git_push" || toolName == "git_diff":
		return "git"
	case toolName == "run_agent" || toolName == "get_agent_run" || toolName == "list_task_agents":
		return "orchestration"
	default:
		return "builtin"
	}
}
