// Package chat implements the line-oriented interactive chat (`YOU >` /
// `AI >`, docs/tui-spec.md, sections 1, 8 and 9): the read-send-print
// loop, its commands, and the requests it sends with the resolved
// generation settings. Conversation state and files belong to
// internal/session.
package chat

import (
	"context"

	"github.com/neurobrau/llm-cli-go/internal/llm"
)

// Engine sends the chat's requests to one provider instance and model with
// fixed generation settings. The same provider is used for every turn, so
// its caches (and remembered rejections of a built-in reasoning default)
// last for the whole chat.
type Engine struct {
	Provider llm.Provider
	Model    string
	Settings Settings
}

// Request builds the request for messages (the committed context plus the
// new user message). The answer style applies to this request only.
func (e *Engine) Request(messages []llm.Message) llm.Request {
	return llm.Request{
		Model:           e.Model,
		Messages:        messages,
		Instructions:    llm.StyleInstructions(e.Settings.AnswerStyle),
		MaxOutputTokens: e.Settings.MaxOutputTokens,
		Reasoning:       e.Settings.Reasoning,
	}
}

// Reply sends messages and returns the final answer.
func (e *Engine) Reply(ctx context.Context, messages []llm.Message) (llm.Response, error) {
	return e.Provider.Chat(ctx, e.Request(messages))
}
