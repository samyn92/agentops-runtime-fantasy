/*
Agent Runtime — Fantasy (Go)

Question tool: a built-in tool that allows the agent to ask the user
questions interactively. Emits a question_asked FEP event and blocks
on a Go channel until the console delivers the user's answers.
*/
package main

import (
	"context"
	"encoding/json"
	"sync"
	"time"

	"charm.land/fantasy"
	"github.com/google/uuid"
)

// QuestionOption represents a selectable option for a question.
type QuestionOption struct {
	Label       string `json:"label"`
	Description string `json:"description"`
}

// QuestionItem represents a single question to ask the user.
type QuestionItem struct {
	Header   string           `json:"header"`
	Question string           `json:"question"`
	Options  []QuestionOption `json:"options,omitempty"`
	Multiple bool             `json:"multiple,omitempty"`
}

// QuestionResponse represents the user's answers to questions.
type QuestionResponse struct {
	Answers [][]string `json:"answers"` // answers[i] = selected labels for question i
}

// questionGate manages pending question requests.
type questionGate struct {
	mu        sync.Mutex
	pending   map[string]chan QuestionResponse // questionId -> response channel
	emitEvent func(id, sessionId string, questions json.RawMessage)
}

func newQuestionGate(
	emitEvent func(id, sessionId string, questions json.RawMessage),
) *questionGate {
	return &questionGate{
		pending:   make(map[string]chan QuestionResponse),
		emitEvent: emitEvent,
	}
}

// ask blocks until the user replies or context is cancelled.
func (g *questionGate) ask(ctx context.Context, sessionId string, questions []QuestionItem) (QuestionResponse, error) {
	id := uuid.New().String()
	ch := make(chan QuestionResponse, 1)

	g.mu.Lock()
	g.pending[id] = ch
	g.mu.Unlock()

	defer func() {
		g.mu.Lock()
		delete(g.pending, id)
		g.mu.Unlock()
	}()

	// Marshal questions for the FEP event
	qJSON, _ := json.Marshal(questions)
	g.emitEvent(id, sessionId, qJSON)

	timeout := time.After(10 * time.Minute)

	select {
	case resp := <-ch:
		return resp, nil
	case <-timeout:
		return QuestionResponse{}, context.DeadlineExceeded
	case <-ctx.Done():
		return QuestionResponse{}, ctx.Err()
	}
}

// reply delivers the user's answers to a pending question.
func (g *questionGate) reply(questionId string, resp QuestionResponse) bool {
	g.mu.Lock()
	ch, ok := g.pending[questionId]
	g.mu.Unlock()

	if !ok {
		return false
	}

	select {
	case ch <- resp:
		return true
	default:
		return false
	}
}

// questionToolInput is the input schema for the question tool.
type questionToolInput struct {
	Questions []QuestionItem `json:"questions" description:"List of questions to ask the user"`
}

// newQuestionTool creates a Fantasy agent tool that asks the user questions.
func newQuestionTool(gate *questionGate) fantasy.AgentTool {
	return fantasy.NewAgentTool("question",
		"Ask the user one or more questions and wait for their responses. Use this when you need clarification, preferences, or decisions from the user. Each question can optionally include selectable options.",
		func(ctx context.Context, input questionToolInput, _ fantasy.ToolCall) (fantasy.ToolResponse, error) {
			if len(input.Questions) == 0 {
				return fantasy.NewTextErrorResponse("at least one question is required"), nil
			}

			// Get sessionId from context
			sessionId := ""
			if sid, ok := ctx.Value(sessionIdContextKey{}).(string); ok {
				sessionId = sid
			}

			resp, err := gate.ask(ctx, sessionId, input.Questions)
			if err != nil {
				return fantasy.NewTextErrorResponse("User did not respond: " + err.Error()), nil
			}

			// Format answers as structured JSON
			answersJSON, _ := json.MarshalIndent(resp.Answers, "", "  ")
			return fantasy.NewTextResponse(string(answersJSON)), nil
		},
	)
}
