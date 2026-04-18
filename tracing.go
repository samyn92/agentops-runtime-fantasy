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
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"strconv"
	"strings"
	"time"

	"charm.land/fantasy"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/exporters/otlp/otlpmetric/otlpmetricgrpc"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
	"go.opentelemetry.io/otel/metric"
	"go.opentelemetry.io/otel/propagation"
	sdkmetric "go.opentelemetry.io/otel/sdk/metric"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.26.0"
	"go.opentelemetry.io/otel/trace"
)

// Package-level tracer used by all instrumentation points.
var tracer trace.Tracer

// GenAI metrics — Required by OTel GenAI semantic conventions.
// See: https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-metrics/
var (
	// gen_ai.client.operation.duration — histogram of LLM call durations (seconds).
	metricOperationDuration metric.Float64Histogram
	// gen_ai.client.token.usage — histogram of token counts per operation.
	metricTokenUsage metric.Int64Histogram
	// gen_ai.server.time_to_first_token — histogram of time to first token (seconds).
	metricTTFT metric.Float64Histogram
)

// tracingFuncs holds the ForceFlush and Shutdown functions returned by initTracing.
// ForceFlush must be called before Shutdown in short-lived task pods to ensure
// all ended spans (especially tool.execute children) are exported to Tempo
// before the process exits.
type tracingFuncs struct {
	ForceFlush func(context.Context) error
	Shutdown   func(context.Context) error
}

// initTracing sets up the OTel TracerProvider with an OTLP gRPC exporter.
// Returns tracingFuncs with ForceFlush and Shutdown that must be called before
// process exit. ForceFlush ensures all completed spans are exported; Shutdown
// tears down the provider.
//
// When OTEL_EXPORTER_OTLP_ENDPOINT is empty, returns no-op functions
// and uses the global no-op tracer — zero overhead in non-instrumented deployments.
// otelErrorHandler logs OTEL internal errors (exporter dial failures, batch
// drops, etc.) via slog so they are not silently swallowed.
type otelErrorHandler struct{}

func (otelErrorHandler) Handle(err error) {
	slog.Error("otel internal error", "error", err)
}

func initTracing(ctx context.Context, agentName, agentNamespace, agentMode string) (*tracingFuncs, error) {
	// Surface OTEL exporter / SDK errors that would otherwise be swallowed.
	otel.SetErrorHandler(otelErrorHandler{})

	endpoint := os.Getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
	if endpoint == "" {
		// No endpoint configured — use no-op tracer and metrics
		tracer = otel.Tracer("agentops-runtime")
		initNoopMetrics()
		slog.Info("tracing disabled (OTEL_EXPORTER_OTLP_ENDPOINT not set)")
		noop := func(context.Context) error { return nil }
		return &tracingFuncs{ForceFlush: noop, Shutdown: noop}, nil
	}

	// Build resource attributes
	attrs := []attribute.KeyValue{
		semconv.ServiceName("agentops-runtime"),
		semconv.ServiceVersion(version),
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
		sdktrace.WithBatcher(exporter,
			// Shorter batch timeout for task pods — ensures spans are flushed
			// before short-lived Jobs exit. Daemon pods benefit too since
			// spans appear in Tempo faster.
			sdktrace.WithBatchTimeout(2*time.Second),
		),
		sdktrace.WithResource(res),
		// Always sample — we want every agent execution traced.
		// Revisit if Tempo storage becomes a concern (per PLAN_otel.md open question #3).
		sdktrace.WithSampler(sdktrace.AlwaysSample()),
	)

	otel.SetTracerProvider(tp)
	otel.SetTextMapPropagator(propagation.TraceContext{})
	tracer = tp.Tracer("agentops-runtime")

	// Initialize OTLP gRPC metric exporter and MeterProvider.
	// GenAI semantic conventions require operation.duration and token.usage histograms.
	// Metrics require a separate endpoint (OTEL_EXPORTER_OTLP_METRICS_ENDPOINT) because
	// Tempo only accepts traces. When no metrics endpoint is configured, use noop metrics.
	metricsEndpoint := os.Getenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT")
	if metricsEndpoint == "" {
		slog.Info("metrics disabled (OTEL_EXPORTER_OTLP_METRICS_ENDPOINT not set, Tempo only supports traces)")
		initNoopMetrics()
	} else {
		metricExporter, err := otlpmetricgrpc.New(ctx,
			otlpmetricgrpc.WithEndpoint(metricsEndpoint),
			otlpmetricgrpc.WithInsecure(),
		)
		if err != nil {
			slog.Warn("metric exporter init failed, using noop metrics", "error", err)
			initNoopMetrics()
		} else {
			mp := sdkmetric.NewMeterProvider(
				sdkmetric.WithResource(res),
				sdkmetric.WithReader(sdkmetric.NewPeriodicReader(metricExporter,
					sdkmetric.WithInterval(10*time.Second),
				)),
			)
			otel.SetMeterProvider(mp)
			initMetrics(mp.Meter("agentops-runtime"))
		}
	}

	slog.Info("tracing enabled",
		"endpoint", endpoint,
		"agent", agentName,
		"mode", agentMode,
	)

	// Heartbeat span: emit a startup span and ForceFlush so export issues
	// surface immediately in logs (via otelErrorHandler) rather than after
	// the first long-running delegation. Visible in Tempo as
	// span.name="runtime.startup" under service.name="agentops-runtime".
	go func() {
		hbCtx, hbSpan := tp.Tracer("agentops-runtime").Start(context.Background(), "runtime.startup",
			trace.WithAttributes(
				attribute.String("agent.name", agentName),
				attribute.String("agent.mode", agentMode),
			),
		)
		hbSpan.End()
		flushCtx, cancel := context.WithTimeout(hbCtx, 5*time.Second)
		defer cancel()
		if err := tp.ForceFlush(flushCtx); err != nil {
			slog.Error("otel startup flush failed", "error", err)
		} else {
			slog.Info("otel startup heartbeat flushed", "endpoint", endpoint)
		}
	}()

	return &tracingFuncs{ForceFlush: tp.ForceFlush, Shutdown: tp.Shutdown}, nil
}

// initMetrics creates the GenAI metric instruments using a real meter.
func initMetrics(meter metric.Meter) {
	var err error

	metricOperationDuration, err = meter.Float64Histogram(
		"gen_ai.client.operation.duration",
		metric.WithDescription("Duration of GenAI operations (seconds)"),
		metric.WithUnit("s"),
	)
	if err != nil {
		slog.Warn("failed to create operation.duration metric", "error", err)
	}

	metricTokenUsage, err = meter.Int64Histogram(
		"gen_ai.client.token.usage",
		metric.WithDescription("Token usage per GenAI operation"),
		metric.WithUnit("{token}"),
	)
	if err != nil {
		slog.Warn("failed to create token.usage metric", "error", err)
	}

	metricTTFT, err = meter.Float64Histogram(
		"gen_ai.server.time_to_first_token",
		metric.WithDescription("Time to first token in seconds"),
		metric.WithUnit("s"),
	)
	if err != nil {
		slog.Warn("failed to create time_to_first_token metric", "error", err)
	}
}

// initNoopMetrics creates noop metric instruments when OTEL is disabled.
func initNoopMetrics() {
	meter := otel.Meter("agentops-runtime")
	initMetrics(meter)
}

// recordGenAIMetrics records the Required GenAI metrics after a successful LLM call.
// Must be called from streamWithFallback / generateWithFallback after the call completes.
func recordGenAIMetrics(ctx context.Context, duration time.Duration, result *fantasy.AgentResult, model, provider, operation string) {
	if result == nil {
		return
	}

	commonAttrs := []attribute.KeyValue{
		attrGenAIOperationName.String(operation),
		attrGenAIRequestModel.String(model),
		attrGenAIProviderName.String(provider),
	}

	// gen_ai.client.operation.duration — Required histogram
	if metricOperationDuration != nil {
		metricOperationDuration.Record(ctx, duration.Seconds(),
			metric.WithAttributes(commonAttrs...),
		)
	}

	// gen_ai.client.token.usage — Required histogram with token.type dimension
	if metricTokenUsage != nil {
		if result.TotalUsage.InputTokens > 0 {
			metricTokenUsage.Record(ctx, result.TotalUsage.InputTokens,
				metric.WithAttributes(append(commonAttrs, attribute.String("gen_ai.token.type", "input"))...),
			)
		}
		if result.TotalUsage.OutputTokens > 0 {
			metricTokenUsage.Record(ctx, result.TotalUsage.OutputTokens,
				metric.WithAttributes(append(commonAttrs, attribute.String("gen_ai.token.type", "output"))...),
			)
		}
		if result.TotalUsage.ReasoningTokens > 0 {
			metricTokenUsage.Record(ctx, result.TotalUsage.ReasoningTokens,
				metric.WithAttributes(append(commonAttrs, attribute.String("gen_ai.token.type", "reasoning"))...),
			)
		}
	}
}

// ────────────────────────────────────────────────────────────────────
// Span attribute keys — GenAI Semantic Conventions + AgentOps custom
// ────────────────────────────────────────────────────────────────────

// Agent-level attributes
var (
	attrAgentName      = attribute.Key("agent.name")
	attrAgentNamespace = attribute.Key("agent.namespace")
	attrAgentMode      = attribute.Key("agent.mode")
)

// GenAI semantic convention attributes
// See: https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/
var (
	attrGenAIOperationName   = attribute.Key("gen_ai.operation.name") // Required: "chat", "invoke_agent", "execute_tool"
	attrGenAIProviderName    = attribute.Key("gen_ai.provider.name")  // Required: "anthropic", "openai", etc.
	attrGenAIRequestModel    = attribute.Key("gen_ai.request.model")
	attrGenAIResponseModel   = attribute.Key("gen_ai.response.model")
	attrGenAIResponseID      = attribute.Key("gen_ai.response.id")
	attrGenAIInputTokens     = attribute.Key("gen_ai.usage.input_tokens")
	attrGenAIOutputTokens    = attribute.Key("gen_ai.usage.output_tokens")
	attrGenAIReasoningTokens = attribute.Key("gen_ai.usage.reasoning_tokens")
	attrGenAIFinishReason    = attribute.Key("gen_ai.response.finish_reasons")
	attrGenAICacheCreate     = attribute.Key("gen_ai.usage.cache_creation_tokens")
	attrGenAICacheRead       = attribute.Key("gen_ai.usage.cache_read_tokens")
	attrGenAITemperature     = attribute.Key("gen_ai.request.temperature")
	attrGenAIMaxTokens       = attribute.Key("gen_ai.request.max_tokens")
	attrGenAIToolName        = attribute.Key("gen_ai.tool.name")
	attrGenAIToolCallID      = attribute.Key("gen_ai.tool.call.id")

	// GenAI Agent semantic conventions (development status)
	attrGenAIAgentID        = attribute.Key("gen_ai.agent.id")        // Unique agent identifier (namespace/name)
	attrGenAIAgentName      = attribute.Key("gen_ai.agent.name")      // Human-readable agent name
	attrGenAIConversationID = attribute.Key("gen_ai.conversation.id") // Memory session ID — correlates turns
	attrGenAIServerAddress = attribute.Key("server.address") // LLM API endpoint host
	attrGenAIServerPort    = attribute.Key("server.port")    // LLM API endpoint port
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

// MCP semantic convention attributes
// See: https://opentelemetry.io/docs/specs/semconv/gen-ai/mcp/
var (
	attrMCPMethodName      = attribute.Key("mcp.method.name")
	attrMCPProtocolVersion = attribute.Key("mcp.protocol.version")
	attrMCPToolCallArgs    = attribute.Key("gen_ai.tool.call.arguments")
	attrMCPToolCallResult  = attribute.Key("gen_ai.tool.call.result")
)

// Memory attributes
var (
	attrMemoryOp                = attribute.Key("memory.operation")
	attrMemoryProject           = attribute.Key("memory.project")
	attrMemorySessionID         = attribute.Key("memory.session_id")
	attrMemoryLimit             = attribute.Key("memory.limit")
	attrMemorySessionsCount     = attribute.Key("memory.sessions_count")
	attrMemoryObservationsCount = attribute.Key("memory.observations_count")
	attrMemoryContextLength     = attribute.Key("memory.context_length")
	attrMemoryContextPreview    = attribute.Key("memory.context_preview")
	attrMemoryHasContent        = attribute.Key("memory.has_content")
	attrMemoryObsType           = attribute.Key("memory.observation_type")
	attrMemoryObsTitle          = attribute.Key("memory.observation_title")
	attrMemoryContentLength     = attribute.Key("memory.content_length")
	attrMemoryTags              = attribute.Key("memory.tags")
	attrMemorySearchQuery       = attribute.Key("memory.search_query")
	attrMemoryResultCount       = attribute.Key("memory.result_count")
	attrMemoryMessageCount      = attribute.Key("memory.message_count")
)

// Opt-in content recording attributes (GenAI semconv).
// Gated by OTEL_GENAI_CAPTURE_CONTENT=true env var since they may contain sensitive data.
var (
	attrGenAISystemInstructions = attribute.Key("gen_ai.system_instructions")
	attrGenAIToolDefinitions    = attribute.Key("gen_ai.tool.definitions")
)

// captureContent returns true when opt-in content recording is enabled.
func captureContent() bool {
	v := os.Getenv("OTEL_GENAI_CAPTURE_CONTENT")
	return v == "true" || v == "1"
}

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

// recordError sets the span status to Error, records the error as an event,
// and sets error.type per OTel semantic conventions.
func recordError(span trace.Span, err error) {
	span.SetStatus(codes.Error, err.Error())
	span.RecordError(err)
	span.SetAttributes(attribute.String("error.type", errorType(err)))
}

// errorType classifies an error into a short error.type string per OTel conventions.
func errorType(err error) string {
	if err == nil {
		return ""
	}
	s := err.Error()
	switch {
	case strings.Contains(s, "429") || strings.Contains(s, "rate limit"):
		return "rate_limited"
	case strings.Contains(s, "401") || strings.Contains(s, "403"):
		return "auth_error"
	case strings.Contains(s, "404"):
		return "not_found"
	case strings.Contains(s, "500") || strings.Contains(s, "502") || strings.Contains(s, "503"):
		return "server_error"
	case strings.Contains(s, "timeout") || strings.Contains(s, "deadline"):
		return "timeout"
	case strings.Contains(s, "connection") || strings.Contains(s, "EOF"):
		return "connection_error"
	case strings.Contains(s, "context canceled"):
		return "cancelled"
	default:
		return "error"
	}
}

// setLLMResultAttributes sets GenAI semantic convention attributes from a Fantasy AgentResult.
func setLLMResultAttributes(span trace.Span, result *fantasy.AgentResult, model string) {
	if result == nil {
		return
	}

	attrs := []attribute.KeyValue{
		attrGenAIResponseModel.String(model),
		attrGenAIInputTokens.Int64(result.TotalUsage.InputTokens),
		attrGenAIOutputTokens.Int64(result.TotalUsage.OutputTokens),
	}

	if result.TotalUsage.ReasoningTokens > 0 {
		attrs = append(attrs, attrGenAIReasoningTokens.Int64(result.TotalUsage.ReasoningTokens))
	}
	if result.TotalUsage.CacheCreationTokens > 0 {
		attrs = append(attrs, attrGenAICacheCreate.Int64(result.TotalUsage.CacheCreationTokens))
	}
	if result.TotalUsage.CacheReadTokens > 0 {
		attrs = append(attrs, attrGenAICacheRead.Int64(result.TotalUsage.CacheReadTokens))
	}

	// Extract gen_ai.response.id from the final step's provider metadata.
	// This is provider-specific but important for correlation.
	if len(result.Steps) > 0 {
		lastStep := result.Steps[len(result.Steps)-1]
		if respID := extractResponseID(lastStep.ProviderMetadata); respID != "" {
			attrs = append(attrs, attrGenAIResponseID.String(respID))
		}
	}

	// Collect finish reasons from steps
	var finishReasons []string
	for _, step := range result.Steps {
		if r := string(step.FinishReason); r != "" {
			finishReasons = append(finishReasons, r)
		}
	}
	if len(finishReasons) > 0 {
		attrs = append(attrs, attribute.StringSlice(string(attrGenAIFinishReason), finishReasons))
	}

	span.SetAttributes(attrs...)

	// Record tool.call events from step results on this span.
	// This is the reliable path — individual tool.execute spans are often
	// lost by the batch exporter for short-lived task pods, but the
	// gen_ai.generate span always survives. The console UI synthesizes
	// waterfall rows from these events.
	recordToolCallEventsFromSteps(span, result.Steps)
}

// extractResponseID attempts to extract a response ID from provider metadata.
// Checks common provider metadata types for an ID field.
func extractResponseID(meta fantasy.ProviderMetadata) string {
	if meta == nil {
		return ""
	}
	// Provider metadata is map[string]any — check for known patterns.
	for _, v := range meta {
		if v == nil {
			continue
		}
		// Use reflection-free approach: try JSON round-trip to extract "id" or "response_id".
		data, err := json.Marshal(v)
		if err != nil {
			continue
		}
		var m map[string]any
		if json.Unmarshal(data, &m) != nil {
			continue
		}
		if id, ok := m["response_id"].(string); ok && id != "" {
			return id
		}
		if id, ok := m["id"].(string); ok && id != "" {
			return id
		}
	}
	return ""
}

// maxToolEventContentLen caps tool input/output stored in tool.call events.
// Smaller than maxEventContentLen since there can be many tool calls per span.
const maxToolEventContentLen = 1000

// recordToolCallEventsFromSteps iterates agent steps and records a tool.call
// event for every tool invocation found in the step response content. Each
// event carries the tool name, type, step number, input args, output text,
// and error status so the console UI can render a deep-inspection detail
// panel when clicking a tool row in the trace waterfall.
func recordToolCallEventsFromSteps(span trace.Span, steps []fantasy.StepResult) {
	toolCount := 0
	for stepIdx, step := range steps {
		// First pass: collect tool results keyed by call ID.
		type toolResult struct {
			output  string
			isError bool
		}
		results := make(map[string]toolResult)
		for _, content := range step.Content {
			if content.GetType() != fantasy.ContentTypeToolResult {
				continue
			}
			tr, ok := fantasy.AsContentType[fantasy.ToolResultContent](content)
			if !ok {
				continue
			}
			res := toolResult{}
			if tr.Result != nil {
				switch tr.Result.GetType() {
				case fantasy.ToolResultContentTypeText:
					if txt, ok := fantasy.AsToolResultOutputType[fantasy.ToolResultOutputContentText](tr.Result); ok {
						res.output = txt.Text
					}
				case fantasy.ToolResultContentTypeError:
					res.isError = true
					if errResult, ok := fantasy.AsToolResultOutputType[fantasy.ToolResultOutputContentError](tr.Result); ok && errResult.Error != nil {
						res.output = errResult.Error.Error()
					}
				}
			}
			results[tr.ToolCallID] = res
		}

		// Record gen_ai.assistant.message event for intermediate steps with tool calls.
		// This captures the assistant's reasoning/text before tool invocations.
		var hasToolCalls bool
		var assistantText string
		for _, content := range step.Content {
			switch content.GetType() {
			case fantasy.ContentTypeToolCall:
				hasToolCalls = true
			case fantasy.ContentTypeText:
				if tc, ok := fantasy.AsContentType[fantasy.TextContent](content); ok {
					assistantText += tc.Text
				}
			}
		}
		if hasToolCalls {
			evAttrs := []attribute.KeyValue{
				attribute.String("gen_ai.message.role", "assistant"),
				attribute.Int("gen_ai.message.step", stepIdx+1),
			}
			if assistantText != "" {
				evAttrs = append(evAttrs, attribute.String("gen_ai.message.content", truncateContent(assistantText, maxToolEventContentLen)))
			}
			span.AddEvent("gen_ai.assistant.message", trace.WithAttributes(evAttrs...))
		}

		// Second pass: record tool.call events with input + output.
		for _, content := range step.Content {
			if content.GetType() != fantasy.ContentTypeToolCall {
				continue
			}
			tc, ok := fantasy.AsContentType[fantasy.ToolCallContent](content)
			if !ok {
				continue
			}
			evAttrs := []attribute.KeyValue{
				attribute.String("tool.name", tc.ToolName),
				attribute.String("tool.type", classifyToolType(tc.ToolName)),
				attribute.Int("tool.step", stepIdx+1),
			}

			// Include tool input (JSON args) — the deep inspection data.
			// Uses gen_ai.tool.call.arguments semconv attribute alongside legacy tool.input.
			if tc.Input != "" {
				evAttrs = append(evAttrs,
					attribute.String("tool.input", truncateContent(tc.Input, maxToolEventContentLen)),
					attrMCPToolCallArgs.String(truncateContent(tc.Input, maxToolEventContentLen)),
				)
			}

			// Include tool output if we have a matching result.
			// Uses gen_ai.tool.call.result semconv attribute alongside legacy tool.output.
			if res, ok := results[tc.ToolCallID]; ok {
				if res.isError {
					evAttrs = append(evAttrs, attribute.Bool("tool.error", true))
				}
				if res.output != "" {
					evAttrs = append(evAttrs,
						attribute.String("tool.output", truncateContent(res.output, maxToolEventContentLen)),
						attrMCPToolCallResult.String(truncateContent(res.output, maxToolEventContentLen)),
					)
				}
			}

			span.AddEvent("tool.call", trace.WithAttributes(evAttrs...))
			toolCount++
		}
	}
	if toolCount > 0 {
		span.SetAttributes(attribute.Int("agent.tool_calls", toolCount))
		slog.Info("recorded tool.call events on gen_ai.generate span", "count", toolCount)
	}
}

// detectGenAIProvider infers the gen_ai.provider.name from model/provider names.
// Returns the OTel semconv well-known value (e.g. "anthropic", "openai").
func detectGenAIProvider(model, provider string) string {
	lower := strings.ToLower(provider)
	switch {
	case lower == "anthropic":
		return "anthropic"
	case lower == "openai":
		return "openai"
	case lower == "google" || lower == "gemini":
		return "gcp.gemini"
	case lower == "deepseek":
		return "deepseek"
	case lower == "openrouter":
		return "openrouter"
	case lower == "kimi" || lower == "moonshot":
		return "kimi"
	}
	// Infer from model name
	m := strings.ToLower(model)
	switch {
	case strings.HasPrefix(m, "claude") || strings.HasPrefix(m, "anthropic/"):
		return "anthropic"
	case strings.HasPrefix(m, "gpt") || strings.HasPrefix(m, "openai/"):
		return "openai"
	case strings.HasPrefix(m, "gemini") || strings.HasPrefix(m, "google/"):
		return "gcp.gemini"
	case strings.HasPrefix(m, "deepseek"):
		return "deepseek"
	case strings.HasPrefix(m, "moonshot") || strings.HasPrefix(m, "kimi"):
		return "kimi"
	}
	if provider != "" {
		return provider
	}
	return "unknown"
}

// ────────────────────────────────────────────────────────────────────
// Content events — OTel GenAI semantic convention events
// ────────────────────────────────────────────────────────────────────

// maxEventContentLen caps the content stored in span events to prevent
// massive traces. 2000 chars ≈ 500 tokens — enough for prompt/response
// overview without blowing up Tempo storage.
const maxEventContentLen = 2000

// truncateContent truncates s to maxLen, appending a marker if truncated.
func truncateContent(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "... [truncated]"
}

// recordSystemMessageEvent records the system prompt as a gen_ai.system.message event
// on the given span, following the OTel GenAI semantic conventions.
// When OTEL_GENAI_CAPTURE_CONTENT=true, also sets the gen_ai.system_instructions attribute.
func recordSystemMessageEvent(span trace.Span, systemPrompt string) {
	if systemPrompt == "" {
		return
	}
	span.AddEvent("gen_ai.system.message", trace.WithAttributes(
		attribute.String("gen_ai.message.role", "system"),
		attribute.String("gen_ai.message.content", truncateContent(systemPrompt, maxEventContentLen)),
	))
	if captureContent() {
		span.SetAttributes(attrGenAISystemInstructions.String(systemPrompt))
	}
}

// recordToolDefinitionsAttribute sets the gen_ai.tool.definitions attribute
// on the span when opt-in content recording is enabled. This captures the
// tool schemas available to the model for this invocation.
func recordToolDefinitionsAttribute(span trace.Span, tools []fantasy.AgentTool) {
	if !captureContent() || len(tools) == 0 {
		return
	}
	type toolDef struct {
		Name string `json:"name"`
		Desc string `json:"description,omitempty"`
	}
	defs := make([]toolDef, 0, len(tools))
	for _, t := range tools {
		info := t.Info()
		defs = append(defs, toolDef{Name: info.Name, Desc: info.Description})
	}
	if data, err := json.Marshal(defs); err == nil {
		span.SetAttributes(attrGenAIToolDefinitions.String(string(data)))
	}
}

// recordPromptEvent records the user prompt as a gen_ai.user.message event
// on the given span, following the OTel GenAI semantic conventions for
// structured input messages.
func recordPromptEvent(span trace.Span, prompt string) {
	if prompt == "" {
		return
	}
	span.AddEvent("gen_ai.user.message", trace.WithAttributes(
		attribute.String("gen_ai.message.role", "user"),
		attribute.String("gen_ai.message.content", truncateContent(prompt, maxEventContentLen)),
	))
}

// recordCompletionEvent records the assistant response as a gen_ai.choice event
// following the OTel GenAI semantic conventions for structured output messages.
func recordCompletionEvent(span trace.Span, completion string) {
	if completion == "" {
		return
	}
	span.AddEvent("gen_ai.choice", trace.WithAttributes(
		attribute.Int("gen_ai.choice.index", 0),
		attribute.String("gen_ai.choice.finish_reason", "stop"),
		attribute.String("gen_ai.choice.message.role", "assistant"),
		attribute.String("gen_ai.choice.message.content", truncateContent(completion, maxEventContentLen)),
	))
}

// recordToolInputEvent records the tool input as an event on the span.
func recordToolInputEvent(span trace.Span, toolName, input string) {
	if input == "" {
		return
	}
	span.AddEvent("gen_ai.tool.input", trace.WithAttributes(
		attribute.String("tool.name", toolName),
		attribute.String("tool.input", truncateContent(input, maxEventContentLen)),
	))
}

// recordToolOutputEvent records the tool output as an event on the span.
func recordToolOutputEvent(span trace.Span, toolName, output string, isError bool) {
	if output == "" {
		return
	}
	attrs := []attribute.KeyValue{
		attribute.String("tool.name", toolName),
		attribute.String("tool.output", truncateContent(output, maxEventContentLen)),
	}
	if isError {
		attrs = append(attrs, attribute.Bool("tool.error", true))
	}
	span.AddEvent("gen_ai.tool.output", trace.WithAttributes(attrs...))
}

// ────────────────────────────────────────────────────────────────────
// Root span context — lets tool wrappers record events on the prompt span
// ────────────────────────────────────────────────────────────────────

// agentSpanAttributes returns standard attributes for agent.prompt root spans
// following the GenAI Agent semantic conventions. Includes agent identity,
// conversation correlation, and server address.
func agentSpanAttributes(agentName, agentNamespace, conversationID string, cfg *Config) []attribute.KeyValue {
	attrs := []attribute.KeyValue{
		attrGenAIAgentName.String(agentName),
	}

	// gen_ai.agent.id — unique agent identifier (namespace/name)
	agentID := agentName
	if agentNamespace != "" {
		agentID = agentNamespace + "/" + agentName
	}
	attrs = append(attrs, attrGenAIAgentID.String(agentID))

	// gen_ai.conversation.id — memory session ID correlates turns within a conversation
	if conversationID != "" {
		attrs = append(attrs, attrGenAIConversationID.String(conversationID))
	}

	// server.address + server.port — LLM API endpoint
	if cfg != nil {
		if addr, port := serverAddressFromConfig(cfg); addr != "" {
			attrs = append(attrs, attrGenAIServerAddress.String(addr))
			if port > 0 {
				attrs = append(attrs, attrGenAIServerPort.Int(port))
			}
		}
	}

	return attrs
}

// serverAddressFromConfig extracts the LLM server address and port from the primary
// provider's BaseURL config. Returns the host portion and port (0 if not explicit).
func serverAddressFromConfig(cfg *Config) (string, int) {
	if cfg == nil {
		return "", 0
	}
	for _, p := range cfg.Providers {
		if p.Name == cfg.PrimaryProvider || (cfg.PrimaryProvider == "" && len(cfg.Providers) == 1) {
			if p.BaseURL != "" {
				// Extract host from URL
				if idx := strings.Index(p.BaseURL, "://"); idx >= 0 {
					host := p.BaseURL[idx+3:]
					if slashIdx := strings.Index(host, "/"); slashIdx >= 0 {
						host = host[:slashIdx]
					}
					// Split host:port
					if colonIdx := strings.LastIndex(host, ":"); colonIdx >= 0 {
						portStr := host[colonIdx+1:]
						host = host[:colonIdx]
						if port, err := strconv.Atoi(portStr); err == nil {
							return host, port
						}
						return host, 0
					}
					return host, 0
				}
				return p.BaseURL, 0
			}
			// Return well-known endpoint for the provider type
			switch strings.ToLower(p.Type) {
			case "anthropic":
				return "api.anthropic.com", 443
			case "openai":
				return "api.openai.com", 443
			case "google", "gemini":
				return "generativelanguage.googleapis.com", 443
			case "deepseek":
				return "api.deepseek.com", 443
			case "openrouter":
				return "openrouter.ai", 443
			}
		}
	}
	return "", 0
}

type rootSpanKey struct{}

// withRootSpan stores the root prompt span in the context so tool wrappers
// can record tool.call events that survive even when tool.execute spans
// are lost by the batch exporter.
func withRootSpan(ctx context.Context, span trace.Span) context.Context {
	return context.WithValue(ctx, rootSpanKey{}, span)
}

// rootSpanFromContext retrieves the root prompt span stashed by withRootSpan.
// Returns nil if none is set.
func rootSpanFromContext(ctx context.Context) trace.Span {
	if s, ok := ctx.Value(rootSpanKey{}).(trace.Span); ok {
		return s
	}
	return nil
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

// ────────────────────────────────────────────────────────────────────
// Delegation trace context — span links for cross-agent tracing
// ────────────────────────────────────────────────────────────────────

// Delegation attribute keys — set on the child's root span to identify its parent.
var (
	attrDelegationParentTraceID = attribute.Key("delegation.parent_trace_id")
	attrDelegationParentSpanID  = attribute.Key("delegation.parent_span_id")
	attrDelegationParentAgent   = attribute.Key("delegation.parent_agent")
	attrDelegationRunName       = attribute.Key("delegation.run_name")
)

// traceparentFromContext builds a W3C traceparent header value from the current
// span context. Returns empty string if there is no valid span context.
func traceparentFromContext(ctx context.Context) string {
	carrier := propagation.MapCarrier{}
	otel.GetTextMapPropagator().Inject(ctx, carrier)
	return carrier.Get("traceparent")
}

// spanContextFromTraceparent extracts a remote SpanContext from a W3C traceparent
// string. Returns an invalid SpanContext if the string is empty or malformed.
func spanContextFromTraceparent(traceparent string) trace.SpanContext {
	if traceparent == "" {
		return trace.SpanContext{}
	}
	carrier := propagation.MapCarrier{}
	carrier.Set("traceparent", traceparent)
	ctx := otel.GetTextMapPropagator().Extract(context.Background(), carrier)
	return trace.SpanContextFromContext(ctx)
}

// delegationSpanOptions returns trace.SpanStartOption values that create a span
// link to the parent agent's span and set delegation attributes. The child span
// starts a new independent trace but carries a link back to the parent.
func delegationSpanOptions(traceparent, parentAgent, runName string) []trace.SpanStartOption {
	var opts []trace.SpanStartOption

	parentSC := spanContextFromTraceparent(traceparent)
	if parentSC.IsValid() {
		// Create a span link to the parent agent's orchestration span.
		// The child trace stays independent (separate trace ID) but linked.
		opts = append(opts, trace.WithLinks(trace.Link{
			SpanContext: parentSC,
			Attributes: []attribute.KeyValue{
				attribute.String("link.type", "delegation"),
				attribute.String("link.parent_agent", parentAgent),
				attribute.String("link.run_name", runName),
			},
		}))
	}

	// Set delegation attributes on the child's root span for easy querying.
	attrs := []attribute.KeyValue{}
	if parentSC.IsValid() {
		attrs = append(attrs,
			attrDelegationParentTraceID.String(parentSC.TraceID().String()),
			attrDelegationParentSpanID.String(parentSC.SpanID().String()),
		)
	}
	if parentAgent != "" {
		attrs = append(attrs, attrDelegationParentAgent.String(parentAgent))
	}
	if runName != "" {
		attrs = append(attrs, attrDelegationRunName.String(runName))
	}
	if len(attrs) > 0 {
		opts = append(opts, trace.WithAttributes(attrs...))
	}

	return opts
}

// createRetrospectiveStepSpans creates agent.step child spans with proper timing
// for non-streaming code paths (daemon handlePrompt, task runner runTask).
//
// Since the Fantasy SDK provides no per-step timing data, we distribute the
// total wall-clock duration proportionally across steps based on token count
// (input + output). This is a reasonable proxy because LLM inference time
// correlates strongly with tokens processed/generated.
//
// The streaming path (OnStepStart/OnStepFinish) doesn't need this — it captures
// real wall-clock timestamps via the callbacks.
func createRetrospectiveStepSpans(ctx context.Context, steps []fantasy.StepResult, genStart, genEnd time.Time) {
	if len(steps) == 0 {
		return
	}

	totalElapsed := genEnd.Sub(genStart)

	// Sum total tokens across all steps for proportional distribution.
	var totalTokens int64
	for _, step := range steps {
		totalTokens += step.Usage.InputTokens + step.Usage.OutputTokens
	}

	// If no token data (shouldn't happen), distribute evenly.
	if totalTokens == 0 {
		perStep := totalElapsed / time.Duration(len(steps))
		for i, step := range steps {
			stepStart := genStart.Add(perStep * time.Duration(i))
			stepEnd := stepStart.Add(perStep)
			_, stepSpan := tracer.Start(ctx, "agent.step",
				trace.WithTimestamp(stepStart),
				trace.WithAttributes(
					attrStepNumber.Int(i+1),
					attrGenAIOperationName.String("agent_step"),
					attrStepFinishReason.String(string(step.FinishReason)),
					attrStepToolCalls.Int(len(step.Content.ToolCalls())),
					attrGenAIInputTokens.Int64(step.Usage.InputTokens),
					attrGenAIOutputTokens.Int64(step.Usage.OutputTokens),
				),
			)
			if step.Usage.ReasoningTokens > 0 {
				stepSpan.SetAttributes(attrGenAIReasoningTokens.Int64(step.Usage.ReasoningTokens))
			}
			stepSpan.End(trace.WithTimestamp(stepEnd))
		}
		return
	}

	// Distribute elapsed time proportionally by token count.
	cursor := genStart
	for i, step := range steps {
		stepTokens := step.Usage.InputTokens + step.Usage.OutputTokens
		fraction := float64(stepTokens) / float64(totalTokens)
		stepDuration := time.Duration(fraction * float64(totalElapsed))

		// Last step gets the remainder to avoid rounding drift.
		stepEnd := cursor.Add(stepDuration)
		if i == len(steps)-1 {
			stepEnd = genEnd
		}

		_, stepSpan := tracer.Start(ctx, "agent.step",
			trace.WithTimestamp(cursor),
			trace.WithAttributes(
				attrStepNumber.Int(i+1),
				attrGenAIOperationName.String("agent_step"),
				attrStepFinishReason.String(string(step.FinishReason)),
				attrStepToolCalls.Int(len(step.Content.ToolCalls())),
				attrGenAIInputTokens.Int64(step.Usage.InputTokens),
				attrGenAIOutputTokens.Int64(step.Usage.OutputTokens),
			),
		)
		if step.Usage.ReasoningTokens > 0 {
			stepSpan.SetAttributes(attrGenAIReasoningTokens.Int64(step.Usage.ReasoningTokens))
		}
		stepSpan.End(trace.WithTimestamp(stepEnd))

		cursor = stepEnd
	}
}
