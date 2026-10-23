// Package llm defines the provider-neutral contract shared by all LLM
// providers: plain message, request, response and model types, the Provider
// interface, and normalized errors (docs/tui-spec.md, sections 12 and 16).
//
// It must not depend on any provider SDK or wire format.
package llm

import (
	"context"
	"time"
)

// Role is the author of a chat message.
type Role string

// Message roles, as stored in conversations (docs/tui-spec.md, section 11).
const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
)

// Message is one chat message.
type Message struct {
	Role    Role
	Content string
}

// Request is one non-streaming generation request.
type Request struct {
	Model    string
	Messages []Message // oldest first
	// Instructions is a system/developer instruction for this request only
	// (e.g. the answer style); it is not part of the conversation. "": none.
	Instructions    string
	MaxOutputTokens int // 0: the model's default
	Reasoning       Reasoning
}

// Usage holds token counts exactly as the provider reported them. A nil
// field was not reported; counts are never estimated.
//
// CachedInputTokens and CacheWriteTokens are parts of InputTokens;
// ReasoningTokens is part of OutputTokens.
type Usage struct {
	InputTokens       *int
	OutputTokens      *int
	TotalTokens       *int
	CachedInputTokens *int
	CacheWriteTokens  *int
	ReasoningTokens   *int
}

// Timing holds durations a provider reported. Zero means not reported.
type Timing struct {
	Total      time.Duration
	Load       time.Duration
	PromptEval time.Duration
	Eval       time.Duration
}

// Response is the final answer of a request. Reasoning (thinking) text is
// kept separate and is never part of Text.
//
// A non-empty IncompleteReason means generation stopped early (e.g. at the
// token limit): Text is only the part that arrived and must not be
// presented as a complete answer.
type Response struct {
	Text             string
	Reasoning        string
	IncompleteReason string
	Usage            *Usage // nil: no usage reported
	ID               string // provider's response ID, if reported
	Model            string // model that answered, if reported
	ServiceTier      string // processing tier the provider reports it used
	DoneReason       string // provider's stop reason, if reported
	Timing           Timing
}

// Model is a model reported by a provider's discovery API.
type Model struct {
	ID            string
	Remote        bool     // runs on a remote service although listed locally
	Capabilities  []string // nil: not reported
	Details       string   // short human-readable summary, e.g. "4B Q4_K_M"
	ContextLength int      // 0: unknown
}

// Provider is what the CLI needs from an LLM provider. Every error a method
// returns is either an *Error or wraps context.Canceled.
type Provider interface {
	// Name is the CLI value, e.g. "ollama".
	Name() string
	// ListModels returns the models that can be selected for chat.
	ListModels(ctx context.Context) ([]Model, error)
	// DefaultModel returns the model used when none is chosen.
	DefaultModel(ctx context.Context) (string, error)
	// ReasoningSupport returns the reasoning values a model accepts. It may
	// make one cached request.
	ReasoningSupport(ctx context.Context, model string) (ReasoningSupport, error)
	// MaxOutputLimit returns the largest allowed max-output-tokens value,
	// or 0 if unknown. It may use cached discovery.
	MaxOutputLimit(ctx context.Context, model string) (int, error)
	// Chat sends the conversation and returns the final answer.
	Chat(ctx context.Context, req Request) (Response, error)
}
