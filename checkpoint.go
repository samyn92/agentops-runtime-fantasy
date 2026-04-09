/*
Agent Runtime — Fantasy (Go)

Working memory checkpoint: serialize to disk on shutdown, restore on startup.
Gives daemon agents crash recovery when backed by a PVC at /data.
*/
package main

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"

	"charm.land/fantasy"
)

const checkpointPath = "/data/sessions/checkpoint.json"

// checkpointData wraps the serialized working memory with metadata.
type checkpointData struct {
	TurnNum  int                   `json:"turnNum"`
	Messages []serializableMessage `json:"messages"`
}

// SaveCheckpoint serializes the working memory to disk.
// Called on graceful shutdown (SIGTERM/SIGINT).
func (wm *WorkingMemory) SaveCheckpoint() {
	wm.mu.RLock()
	defer wm.mu.RUnlock()

	if len(wm.messages) == 0 {
		// Nothing to checkpoint — remove stale file if present
		os.Remove(checkpointPath)
		return
	}

	data := checkpointData{
		TurnNum:  wm.turnNum,
		Messages: serializeMessages(wm.messages),
	}

	raw, err := json.Marshal(data)
	if err != nil {
		slog.Warn("checkpoint marshal failed", "error", err)
		return
	}

	// Ensure directory exists
	os.MkdirAll(filepath.Dir(checkpointPath), 0755)

	// Write atomically: write to temp, rename
	tmpPath := checkpointPath + ".tmp"
	if err := os.WriteFile(tmpPath, raw, 0644); err != nil {
		slog.Warn("checkpoint write failed", "error", err)
		return
	}
	if err := os.Rename(tmpPath, checkpointPath); err != nil {
		slog.Warn("checkpoint rename failed", "error", err)
		os.Remove(tmpPath)
		return
	}

	slog.Info("working memory checkpoint saved",
		"messages", len(wm.messages),
		"turns", wm.turnNum,
	)
}

// RestoreCheckpoint loads working memory from a checkpoint file if it exists.
// Called on startup. Returns the number of messages restored, or 0 if no
// checkpoint was found.
func (wm *WorkingMemory) RestoreCheckpoint() int {
	raw, err := os.ReadFile(checkpointPath)
	if err != nil {
		// No checkpoint file — normal for first run
		return 0
	}

	var data checkpointData
	if err := json.Unmarshal(raw, &data); err != nil {
		slog.Warn("checkpoint parse failed, starting fresh", "error", err)
		os.Remove(checkpointPath)
		return 0
	}

	messages := deserializeMessages(data.Messages)
	if len(messages) == 0 {
		return 0
	}

	wm.mu.Lock()
	defer wm.mu.Unlock()
	wm.messages = messages
	wm.turnNum = data.TurnNum

	slog.Info("working memory restored from checkpoint",
		"messages", len(messages),
		"turns", data.TurnNum,
	)

	// Remove checkpoint after successful restore — it's been consumed
	os.Remove(checkpointPath)

	return len(messages)
}

// RemoveCheckpoint deletes the checkpoint file.
// Called when working memory is explicitly cleared by the user.
func RemoveCheckpoint() {
	os.Remove(checkpointPath)
}

// serializeMessages converts Fantasy messages to the JSON-safe format.
// Reuses the serializableMessage types from handleGetWorkingMemory.
func serializeMessages(messages []fantasy.Message) []serializableMessage {
	result := make([]serializableMessage, 0, len(messages))
	for _, msg := range messages {
		sm := serializableMessage{
			Role:    string(msg.Role),
			Content: make([]serializablePart, 0, len(msg.Content)),
		}
		for _, part := range msg.Content {
			switch p := part.(type) {
			case fantasy.TextPart:
				sm.Content = append(sm.Content, serializablePart{
					Type: "text",
					Text: p.Text,
				})
			case fantasy.ReasoningPart:
				sm.Content = append(sm.Content, serializablePart{
					Type: "reasoning",
					Text: p.Text,
				})
			case fantasy.FilePart:
				sm.Content = append(sm.Content, serializablePart{
					Type:      "file",
					Filename:  p.Filename,
					Data:      string(p.Data),
					MediaType: p.MediaType,
				})
			case fantasy.ToolCallPart:
				sm.Content = append(sm.Content, serializablePart{
					Type:             "tool-call",
					ToolCallID:       p.ToolCallID,
					ToolName:         p.ToolName,
					Input:            p.Input,
					ProviderExecuted: p.ProviderExecuted,
				})
			case fantasy.ToolResultPart:
				sp := serializablePart{
					Type:       "tool-result",
					ToolCallID: p.ToolCallID,
				}
				switch v := p.Output.(type) {
				case fantasy.ToolResultOutputContentText:
					sp.Output = &serializableToolOutput{Type: "text", Text: v.Text}
				case fantasy.ToolResultOutputContentError:
					sp.Output = &serializableToolOutput{Type: "error", Error: v.Error.Error()}
				case fantasy.ToolResultOutputContentMedia:
					sp.Output = &serializableToolOutput{Type: "media", Data: v.Data, MediaType: v.MediaType, Text: v.Text}
				}
				sm.Content = append(sm.Content, sp)
			}
		}
		result = append(result, sm)
	}
	return result
}

// deserializeMessages converts the JSON-safe format back to Fantasy messages.
func deserializeMessages(sms []serializableMessage) []fantasy.Message {
	messages := make([]fantasy.Message, 0, len(sms))
	for _, sm := range sms {
		msg := fantasy.Message{
			Role:    fantasy.MessageRole(sm.Role),
			Content: make([]fantasy.MessagePart, 0, len(sm.Content)),
		}
		for _, sp := range sm.Content {
			switch sp.Type {
			case "text":
				msg.Content = append(msg.Content, fantasy.TextPart{Text: sp.Text})
			case "reasoning":
				msg.Content = append(msg.Content, fantasy.ReasoningPart{Text: sp.Text})
			case "file":
				msg.Content = append(msg.Content, fantasy.FilePart{
					Filename:  sp.Filename,
					Data:      []byte(sp.Data),
					MediaType: sp.MediaType,
				})
			case "tool-call":
				msg.Content = append(msg.Content, fantasy.ToolCallPart{
					ToolCallID:       sp.ToolCallID,
					ToolName:         sp.ToolName,
					Input:            sp.Input,
					ProviderExecuted: sp.ProviderExecuted,
				})
			case "tool-result":
				tr := fantasy.ToolResultPart{
					ToolCallID: sp.ToolCallID,
				}
				if sp.Output != nil {
					switch sp.Output.Type {
					case "text":
						tr.Output = fantasy.ToolResultOutputContentText{Text: sp.Output.Text}
					case "error":
						tr.Output = fantasy.ToolResultOutputContentError{Error: fmt.Errorf("%s", sp.Output.Error)}
					case "media":
						tr.Output = fantasy.ToolResultOutputContentMedia{
							Data:      sp.Output.Data,
							MediaType: sp.Output.MediaType,
							Text:      sp.Output.Text,
						}
					}
				}
				msg.Content = append(msg.Content, tr)
			}
		}
		messages = append(messages, msg)
	}
	return messages
}
