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
	"time"

	"charm.land/fantasy"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.26.0"
	"go.opentelemetry.io/otel/trace"
)

// Package-level tracer used by all instrumentation points.
var tracer trace.Tracer

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
func initTracing(ctx context.Context, agentName, agentNamespace, agentMode string) (*tracingFuncs, error) {
	endpoint := os.Getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
	if endpoint == "" {
		// No endpoint configured — use no-op tracer
		tracer = otel.Tracer("agentops-runtime")
		slog.Info("tracing disabled (OTEL_EXPORTER_OTLP_ENDPOINT not set)")
		noop := func(context.Context) error { return nil }
		return &tracingFuncs{ForceFlush: noop, Shutdown: noop}, nil
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

	slog.Info("tracing enabled",
		"endpoint", endpoint,
		"agent", agentName,
		"mode", agentMode,
	)

	return &tracingFuncs{ForceFlush: tp.ForceFlush, Shutdown: tp.Shutdown}, nil
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
// See: https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/
var (
	attrGenAISystem          = attribute.Key("gen_ai.system")         // deprecated alias; use provider.name
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

// Memory attributes
var (
	attrMemoryOp      = attribute.Key("memory.operation")
	attrMemoryProject = attribute.Key("memory.project")
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

	// Collect finish reasons from steps
	var finishReasons []string
	for _, step := range result.Steps {
		if r := string(step.FinishReason); r != "" {
			finishReasons = append(finishReasons, r)
		}
	}
	if len(finishReasons) > 0 {
		attrs = append(attrs, attrGenAIFinishReason.String(strings.Join(finishReasons, ",")))
	}

	span.SetAttributes(attrs...)

	// Record tool.call events from step results on this span.
	// This is the reliable path — individual tool.execute spans are often
	// lost by the batch exporter for short-lived task pods, but the
	// gen_ai.generate span always survives. The console UI synthesizes
	// waterfall rows from these events.
	recordToolCallEventsFromSteps(span, result.Steps)
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
			if tc.Input != "" {
				evAttrs = append(evAttrs, attribute.String("tool.input", truncateContent(tc.Input, maxToolEventContentLen)))
			}

			// Include tool output if we have a matching result.
			if res, ok := results[tc.ToolCallID]; ok {
				if res.isError {
					evAttrs = append(evAttrs, attribute.Bool("tool.error", true))
				}
				if res.output != "" {
					evAttrs = append(evAttrs, attribute.String("tool.output", truncateContent(res.output, maxToolEventContentLen)))
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
	}
	if provider != "" {
		return provider
	}
	return "unknown"
}

// detectGenAISystem is a backward-compatible alias for detectGenAIProvider.
// Deprecated: use detectGenAIProvider instead.
func detectGenAISystem(model, provider string) string {
	return detectGenAIProvider(model, provider)
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

// recordPromptEvent records the user prompt as a gen_ai.content.prompt event
// on the given span, following the OTel GenAI semantic conventions.
func recordPromptEvent(span trace.Span, prompt string) {
	if prompt == "" {
		return
	}
	span.AddEvent("gen_ai.content.prompt", trace.WithAttributes(
		attribute.String("gen_ai.prompt", truncateContent(prompt, maxEventContentLen)),
	))
}

// recordCompletionEvent records the assistant response as a gen_ai.content.completion
// event on the given span.
func recordCompletionEvent(span trace.Span, completion string) {
	if completion == "" {
		return
	}
	span.AddEvent("gen_ai.content.completion", trace.WithAttributes(
		attribute.String("gen_ai.completion", truncateContent(completion, maxEventContentLen)),
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
