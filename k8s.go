/*
Agent Runtime — Fantasy (Go)

Kubernetes client for creating and querying AgentRun CRs.
Used by the run_agent and get_agent_run orchestration tools.
Runs in-cluster using the pod's service account.
*/
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/rest"
)

const (
	apiGroup       = "agents.agenticops.io"
	apiVersion     = "v1alpha1"
	agentRunPlural = "agentruns"
)

var agentRunGVR = schema.GroupVersionResource{
	Group:    apiGroup,
	Version:  apiVersion,
	Resource: agentRunPlural,
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

// AgentRunResult holds the result of creating an AgentRun.
type AgentRunResult struct {
	Name string `json:"name"`
}

// AgentRunStatus holds the status of an AgentRun.
type AgentRunStatus struct {
	Phase     string `json:"phase"`
	Output    string `json:"output"`
	ToolCalls int64  `json:"toolCalls"`
	Model     string `json:"model"`
}

// CreateAgentRun creates an AgentRun CR.
func (k *K8sClient) CreateAgentRun(ctx context.Context, agentRef, prompt, source, sourceRef string) (*AgentRunResult, error) {
	name := fmt.Sprintf("%s-run-%d", agentRef, time.Now().UnixMilli())

	obj := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": fmt.Sprintf("%s/%s", apiGroup, apiVersion),
			"kind":       "AgentRun",
			"metadata": map[string]interface{}{
				"name":      name,
				"namespace": k.namespace,
				"labels": map[string]interface{}{
					"agents.agenticops.io/agent": agentRef,
				},
			},
			"spec": map[string]interface{}{
				"agentRef":  agentRef,
				"prompt":    prompt,
				"source":    source,
				"sourceRef": sourceRef,
			},
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
