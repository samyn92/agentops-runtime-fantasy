/*
Agent Runtime — Fantasy (Go)

Kubernetes client for creating and querying AgentRun CRs.
Used by the run_agent and get_agent_run orchestration tools.
Runs in-cluster using the pod's service account.
*/
package main

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/rest"
)

const (
	apiGroup            = "agents.agentops.io"
	apiVersion          = "v1alpha1"
	agentRunPlural      = "agentruns"
	agentPlural         = "agents"
	agentResourcePlural = "agentresources"
)

var agentRunGVR = schema.GroupVersionResource{
	Group:    apiGroup,
	Version:  apiVersion,
	Resource: agentRunPlural,
}

var agentGVR = schema.GroupVersionResource{
	Group:    apiGroup,
	Version:  apiVersion,
	Resource: agentPlural,
}

var agentResourceGVR = schema.GroupVersionResource{
	Group:    apiGroup,
	Version:  apiVersion,
	Resource: agentResourcePlural,
}

// K8sClient provides operations for AgentRun CRs.
type K8sClient struct {
	client    dynamic.Interface
	namespace string
}

// NewK8sClient creates a new client using in-cluster config.
func NewK8sClient() (*K8sClient, error) {
	config, err := rest.InClusterConfig()
	if err != nil {
		return nil, fmt.Errorf("in-cluster config: %w", err)
	}

	dynClient, err := dynamic.NewForConfig(config)
	if err != nil {
		return nil, fmt.Errorf("dynamic client: %w", err)
	}

	ns := os.Getenv("AGENT_NAMESPACE")
	if ns == "" {
		ns = "default"
	}

	return &K8sClient{
		client:    dynClient,
		namespace: ns,
	}, nil
}

// AgentInfo holds basic info about an Agent CR.
type AgentInfo struct {
	Name  string `json:"name"`
	Mode  string `json:"mode"`
	Phase string `json:"phase"`
}

// AgentDetail holds rich info about an Agent CR including resource bindings.
type AgentDetail struct {
	Name             string               `json:"name"`
	Mode             string               `json:"mode"`
	Phase            string               `json:"phase"`
	Model            string               `json:"model,omitempty"`
	SystemPrompt     string               `json:"systemPrompt,omitempty"`
	ResourceBindings []AgentResourceBrief `json:"resourceBindings,omitempty"`
}

// AgentResourceBrief holds resolved info about a bound resource.
type AgentResourceBrief struct {
	Name          string `json:"name"`
	Kind          string `json:"kind"`
	DisplayName   string `json:"displayName"`
	Description   string `json:"description,omitempty"`
	DefaultBranch string `json:"defaultBranch,omitempty"`
	// Provider-specific identifiers
	GitHubOwner   string `json:"githubOwner,omitempty"`
	GitHubRepo    string `json:"githubRepo,omitempty"`
	GitLabProject string `json:"gitlabProject,omitempty"`
	GitURL        string `json:"gitURL,omitempty"`
}

// GetAgent checks if an Agent CR exists and returns basic info.
func (k *K8sClient) GetAgent(ctx context.Context, name string) (*AgentInfo, error) {
	obj, err := k.client.Resource(agentGVR).Namespace(k.namespace).Get(ctx, name, metav1.GetOptions{})
	if err != nil {
		return nil, err
	}

	mode, _, _ := unstructured.NestedString(obj.Object, "spec", "mode")
	phase, _, _ := unstructured.NestedString(obj.Object, "status", "phase")

	return &AgentInfo{Name: name, Mode: mode, Phase: phase}, nil
}

// ListAgents returns all Agent CRs in the namespace.
func (k *K8sClient) ListAgents(ctx context.Context) ([]AgentInfo, error) {
	list, err := k.client.Resource(agentGVR).Namespace(k.namespace).List(ctx, metav1.ListOptions{})
	if err != nil {
		return nil, err
	}

	agents := make([]AgentInfo, 0, len(list.Items))
	for _, item := range list.Items {
		name := item.GetName()
		mode, _, _ := unstructured.NestedString(item.Object, "spec", "mode")
		phase, _, _ := unstructured.NestedString(item.Object, "status", "phase")
		agents = append(agents, AgentInfo{Name: name, Mode: mode, Phase: phase})
	}
	return agents, nil
}

// ListAgentDetails returns enriched Agent info including resource bindings.
// It fetches all Agent CRs, reads their resourceBindings, and resolves each
// binding name to the actual AgentResource CR for kind/display/config info.
// teamFilter, if non-nil, restricts results to only agents in the team list.
func (k *K8sClient) ListAgentDetails(ctx context.Context, teamFilter []string) ([]AgentDetail, error) {
	list, err := k.client.Resource(agentGVR).Namespace(k.namespace).List(ctx, metav1.ListOptions{})
	if err != nil {
		return nil, fmt.Errorf("list agents: %w", err)
	}

	// Pre-fetch all AgentResource CRs in the namespace for efficient lookup
	resourceMap, err := k.listAgentResourceMap(ctx)
	if err != nil {
		// Non-fatal: continue without resource details
		resourceMap = map[string]*unstructured.Unstructured{}
	}

	// Build team set for O(1) lookup
	var teamSet map[string]struct{}
	if teamFilter != nil {
		teamSet = make(map[string]struct{}, len(teamFilter))
		for _, name := range teamFilter {
			teamSet[name] = struct{}{}
		}
	}

	agents := make([]AgentDetail, 0, len(list.Items))
	for _, item := range list.Items {
		name := item.GetName()

		// Apply team filter if set
		if teamSet != nil {
			if _, ok := teamSet[name]; !ok {
				continue
			}
		}

		mode, _, _ := unstructured.NestedString(item.Object, "spec", "mode")
		phase, _, _ := unstructured.NestedString(item.Object, "status", "phase")
		model, _, _ := unstructured.NestedString(item.Object, "spec", "model")
		systemPrompt, _, _ := unstructured.NestedString(item.Object, "spec", "systemPrompt")

		// Truncate system prompt for display (first 200 chars)
		if len(systemPrompt) > 200 {
			systemPrompt = systemPrompt[:200] + "..."
		}

		detail := AgentDetail{
			Name:         name,
			Mode:         mode,
			Phase:        phase,
			Model:        model,
			SystemPrompt: systemPrompt,
		}

		// Resolve resource bindings
		bindings, _, _ := unstructured.NestedSlice(item.Object, "spec", "resourceBindings")
		for _, b := range bindings {
			bMap, ok := b.(map[string]interface{})
			if !ok {
				continue
			}
			bindingName, _ := bMap["name"].(string)
			if bindingName == "" {
				continue
			}

			brief := k.resolveResourceBrief(bindingName, resourceMap)
			detail.ResourceBindings = append(detail.ResourceBindings, brief)
		}

		agents = append(agents, detail)
	}
	return agents, nil
}

// listAgentResourceMap fetches all AgentResource CRs and indexes by name.
func (k *K8sClient) listAgentResourceMap(ctx context.Context) (map[string]*unstructured.Unstructured, error) {
	list, err := k.client.Resource(agentResourceGVR).Namespace(k.namespace).List(ctx, metav1.ListOptions{})
	if err != nil {
		return nil, fmt.Errorf("list agent resources: %w", err)
	}
	m := make(map[string]*unstructured.Unstructured, len(list.Items))
	for i := range list.Items {
		m[list.Items[i].GetName()] = &list.Items[i]
	}
	return m, nil
}

// resolveResourceBrief converts an AgentResource CR into a brief summary.
func (k *K8sClient) resolveResourceBrief(name string, resourceMap map[string]*unstructured.Unstructured) AgentResourceBrief {
	brief := AgentResourceBrief{Name: name}

	obj, ok := resourceMap[name]
	if !ok {
		brief.Kind = "unknown"
		brief.DisplayName = name
		return brief
	}

	brief.Kind, _, _ = unstructured.NestedString(obj.Object, "spec", "kind")
	brief.DisplayName, _, _ = unstructured.NestedString(obj.Object, "spec", "displayName")
	brief.Description, _, _ = unstructured.NestedString(obj.Object, "spec", "description")

	switch brief.Kind {
	case "github-repo":
		brief.GitHubOwner, _, _ = unstructured.NestedString(obj.Object, "spec", "github", "owner")
		brief.GitHubRepo, _, _ = unstructured.NestedString(obj.Object, "spec", "github", "repo")
		brief.DefaultBranch, _, _ = unstructured.NestedString(obj.Object, "spec", "github", "defaultBranch")
	case "gitlab-project":
		brief.GitLabProject, _, _ = unstructured.NestedString(obj.Object, "spec", "gitlab", "project")
		brief.DefaultBranch, _, _ = unstructured.NestedString(obj.Object, "spec", "gitlab", "defaultBranch")
	case "git-repo":
		brief.GitURL, _, _ = unstructured.NestedString(obj.Object, "spec", "git", "url")
		brief.DefaultBranch, _, _ = unstructured.NestedString(obj.Object, "spec", "git", "branch")
	}

	return brief
}

// AgentRunResult holds the result of creating an AgentRun.
type AgentRunResult struct {
	Name string `json:"name"`
}

// AgentRunGitParams holds optional git workspace params for an AgentRun.
type AgentRunGitParams struct {
	ResourceRef string `json:"resourceRef"`
	Branch      string `json:"branch"`
	BaseBranch  string `json:"baseBranch,omitempty"`
}

// AgentRunStatus holds the status of an AgentRun.
type AgentRunStatus struct {
	Phase          string `json:"phase"`
	Output         string `json:"output"`
	ToolCalls      int64  `json:"toolCalls"`
	Model          string `json:"model"`
	TraceID        string `json:"traceID,omitempty"`
	PullRequestURL string `json:"pullRequestURL,omitempty"`
	Commits        int64  `json:"commits,omitempty"`
	Branch         string `json:"branch,omitempty"`
}

// CreateAgentRun creates an AgentRun CR. If gitParams is non-nil, spec.git is populated.
// traceparent is the W3C trace context string from the calling agent's span (may be empty).
// extraLabels are merged into the CR's metadata.labels (may be nil).
func (k *K8sClient) CreateAgentRun(ctx context.Context, agentRef, prompt, source, sourceRef, traceparent string, gitParams *AgentRunGitParams, extraLabels map[string]string) (*AgentRunResult, error) {
	// Random 4-byte suffix prevents name collisions when multiple runs
	// are created in the same millisecond (e.g. parallel fan-out).
	var suffix [4]byte
	_, _ = rand.Read(suffix[:])
	name := fmt.Sprintf("%s-run-%d-%s", agentRef, time.Now().UnixMilli(), hex.EncodeToString(suffix[:]))

	spec := map[string]interface{}{
		"agentRef":  agentRef,
		"prompt":    prompt,
		"source":    source,
		"sourceRef": sourceRef,
	}

	if gitParams != nil {
		gitMap := map[string]interface{}{
			"resourceRef": gitParams.ResourceRef,
			"branch":      gitParams.Branch,
		}
		if gitParams.BaseBranch != "" {
			gitMap["baseBranch"] = gitParams.BaseBranch
		}
		spec["git"] = gitMap
	}

	// Build annotations for trace context propagation
	annotations := map[string]interface{}{}
	if traceparent != "" {
		annotations["agents.agentops.io/traceparent"] = traceparent
	}
	if sourceRef != "" {
		annotations["agents.agentops.io/parent-agent"] = sourceRef
	}

	labels := map[string]interface{}{
		"agents.agentops.io/agent": agentRef,
	}
	for k, v := range extraLabels {
		labels[k] = v
	}

	metadata := map[string]interface{}{
		"name":      name,
		"namespace": k.namespace,
		"labels":    labels,
	}
	if len(annotations) > 0 {
		metadata["annotations"] = annotations
	}

	obj := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": fmt.Sprintf("%s/%s", apiGroup, apiVersion),
			"kind":       "AgentRun",
			"metadata":   metadata,
			"spec":       spec,
		},
	}

	_, err := k.client.Resource(agentRunGVR).Namespace(k.namespace).Create(ctx, obj, metav1.CreateOptions{})
	if err != nil {
		return nil, fmt.Errorf("create AgentRun: %w", err)
	}

	return &AgentRunResult{Name: name}, nil
}

// GetAgentRun retrieves an AgentRun CR and returns its status.
func (k *K8sClient) GetAgentRun(ctx context.Context, name string) (*AgentRunStatus, error) {
	obj, err := k.client.Resource(agentRunGVR).Namespace(k.namespace).Get(ctx, name, metav1.GetOptions{})
	if err != nil {
		return nil, fmt.Errorf("get AgentRun %s: %w", name, err)
	}

	status, found, err := unstructured.NestedMap(obj.Object, "status")
	if err != nil || !found {
		return &AgentRunStatus{Phase: "Unknown"}, nil
	}

	// Marshal and unmarshal for clean extraction
	data, _ := json.Marshal(status)
	var result AgentRunStatus
	json.Unmarshal(data, &result)

	return &result, nil
}

// WatchAgentRun starts a Kubernetes Watch on a specific AgentRun CR.
// The returned watch.Interface emits events when the CR changes.
// The caller must call Stop() on the returned interface when done.
func (k *K8sClient) WatchAgentRun(ctx context.Context, name string) (watch.Interface, error) {
	return k.client.Resource(agentRunGVR).Namespace(k.namespace).Watch(ctx, metav1.ListOptions{
		FieldSelector: "metadata.name=" + name,
	})
}
