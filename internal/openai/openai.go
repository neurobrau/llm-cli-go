// Package openai implements llm.Provider for the OpenAI Responses API using
// the official OpenAI Go SDK.
//
// Requests are sent with `store: false`. Only assistant message text is
// returned as the answer; reasoning items are never part of it.
package openai

import (
	"context"
	"errors"
	"net/http"
	"slices"
	"strings"
	"time"

	sdk "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared"

	"github.com/neurobrau/llm-cli-go/internal/llm"
)

const (
	// APIKeyEnvVar names the environment variable that holds the API key.
	APIKeyEnvVar = "OPENAI_API_KEY"
	// DefaultModel is used when no model is chosen.
	DefaultModel = "gpt-6-luna"
	// DiscoveryTimeout bounds the Models API request; it is never retried.
	DiscoveryTimeout = 5 * time.Second
	// ChatTimeout bounds each chat request attempt, like the Python SDK's
	// default timeout. The caller's context still bounds the whole call.
	ChatTimeout = 600 * time.Second
	// ChatMaxRetries is the number of chat retries after a connection
	// error or an HTTP 408, 409, 429 or 5xx status (the SDK's default
	// policy). Timeouts and cancellation are never retried.
	ChatMaxRetries = 2
)

// ErrMissingAPIKey is the cause of the missing-key error, for errors.Is.
var ErrMissingAPIKey = errors.New(APIKeyEnvVar + " is not set")

// nonTextModelMarkers hide models that obviously cannot produce text
// through the Responses API. It is a heuristic: a listed model can still be
// rejected, which is reported as a normal API error.
var nonTextModelMarkers = []string{
	"embedding", "tts", "whisper", "dall-e", "moderation", "transcribe",
	"realtime", "audio", "image",
}

// Config configures a Provider. Zero values select the defaults.
type Config struct {
	// APIKey authenticates requests; empty means missing, and every request
	// then fails with KindMissingAPIKey before contacting the network.
	APIKey string
	// BaseURL overrides the API endpoint (tests); default: the SDK's,
	// which honors OPENAI_BASE_URL.
	BaseURL          string
	HTTPClient       *http.Client // default: the SDK's
	ChatTimeout      time.Duration
	DiscoveryTimeout time.Duration
	// ChatMaxRetries overrides ChatMaxRetries when not nil.
	ChatMaxRetries *int
}

var _ llm.Provider = (*Provider)(nil)

// Provider talks to the OpenAI API. Discovery results are cached for the
// provider's lifetime. A Provider is not safe for concurrent use.
type Provider struct {
	client           *sdk.Client // nil: no API key
	apiKey           string      // only to remove it from error messages
	chatTimeout      time.Duration
	discoveryTimeout time.Duration
	chatMaxRetries   int

	models      []llm.Model     // nil: not loaded
	noReasoning map[string]bool // models that rejected the built-in reasoning default
}

// New returns a Provider. Nothing is requested until a method is called.
func New(cfg Config) *Provider {
	p := &Provider{
		chatTimeout:      cfg.ChatTimeout,
		discoveryTimeout: cfg.DiscoveryTimeout,
		chatMaxRetries:   ChatMaxRetries,
		noReasoning:      map[string]bool{},
	}
	if p.chatTimeout <= 0 {
		p.chatTimeout = ChatTimeout
	}
	if p.discoveryTimeout <= 0 {
		p.discoveryTimeout = DiscoveryTimeout
	}
	if cfg.ChatMaxRetries != nil {
		p.chatMaxRetries = *cfg.ChatMaxRetries
	}
	if key := strings.TrimSpace(cfg.APIKey); key != "" {
		opts := []option.RequestOption{option.WithAPIKey(key)}
		if cfg.BaseURL != "" {
			opts = append(opts, option.WithBaseURL(cfg.BaseURL))
		}
		if cfg.HTTPClient != nil {
			opts = append(opts, option.WithHTTPClient(cfg.HTTPClient))
		}
		client := sdk.NewClient(opts...)
		p.client = &client
		p.apiKey = key
	}
	return p
}

// Name returns the CLI provider name.
func (p *Provider) Name() string { return "openai" }

// DefaultModel returns DefaultModel; it makes no request.
func (p *Provider) DefaultModel(context.Context) (string, error) { return DefaultModel, nil }

func (p *Provider) sdkClient() (*sdk.Client, error) {
	if p.client == nil {
		return nil, MissingAPIKeyError()
	}
	return p.client, nil
}

// MissingAPIKeyError is the error every request returns without an API key.
// Interactive chat reports it before starting, as the Python edition does.
func MissingAPIKeyError() *llm.Error {
	return &llm.Error{
		Kind:    llm.KindMissingAPIKey,
		Message: APIKeyEnvVar + " is not set. Export it in your environment and try again.",
		Err:     ErrMissingAPIKey,
	}
}

// ListModels returns the text-capable model IDs from the Models API,
// sorted. It makes one request with a short timeout and no retries.
func (p *Provider) ListModels(ctx context.Context) ([]llm.Model, error) {
	if p.models != nil {
		return p.models, nil
	}
	client, err := p.sdkClient()
	if err != nil {
		return nil, err
	}
	page, err := client.Models.List(ctx,
		option.WithMaxRetries(0),
		option.WithRequestTimeout(p.discoveryTimeout),
	)
	if err != nil {
		return nil, p.scrub(translateError(ctx, err))
	}
	var ids []string
	for _, m := range page.Data {
		if m.ID != "" && isTextModel(m.ID) {
			ids = append(ids, m.ID)
		}
	}
	slices.Sort(ids)
	ids = slices.Compact(ids)
	p.models = make([]llm.Model, 0, len(ids))
	for _, id := range ids {
		p.models = append(p.models, llm.Model{ID: id})
	}
	return p.models, nil
}

// CheckAvailability reports a missing API key or a Models API failure; it
// reuses the cached model list.
func (p *Provider) CheckAvailability(ctx context.Context) error {
	_, err := p.ListModels(ctx)
	return err
}

func isTextModel(id string) bool {
	for _, marker := range nonTextModelMarkers {
		if strings.Contains(id, marker) {
			return false
		}
	}
	return true
}

// ReasoningSupport returns the confirmed efforts from the built-in table;
// it makes no request. A model that rejected the built-in default during
// this provider's lifetime accepts none.
func (p *Provider) ReasoningSupport(_ context.Context, model string) (llm.ReasoningSupport, error) {
	if p.noReasoning[model] {
		return llm.ReasoningSupport{Known: true}, nil
	}
	return reasoningSupportFor(model), nil
}

// MaxOutputLimit returns the model's output token limit from the built-in
// table, or 0 if unknown. It makes no request.
func (p *Provider) MaxOutputLimit(_ context.Context, model string) (int, error) {
	m, _ := lookupModel(model)
	return m.maxOutputTokens, nil
}

// Chat sends a non-streaming Responses API request with `store: false`.
//
// Without an explicit reasoning choice, the model's built-in default effort
// is sent ("none" where supported). If the API rejects that default, the
// request is retried once without reasoning and the model is remembered.
// An explicit effort is never dropped; its rejection is reported.
func (p *Provider) Chat(ctx context.Context, req llm.Request) (llm.Response, error) {
	client, err := p.sdkClient()
	if err != nil {
		return llm.Response{}, err
	}
	effort := p.effort(ctx, req)
	resp, err := p.send(ctx, client, req, effort)
	if err == nil || effort == "" || !rejectsReasoning(err) {
		return resp, err
	}
	if req.Reasoning.Explicit {
		return llm.Response{}, rejectedEffort(req.Model, effort, err)
	}
	p.noReasoning[req.Model] = true
	return p.send(ctx, client, req, "")
}

func (p *Provider) send(ctx context.Context, client *sdk.Client, req llm.Request, effort string) (llm.Response, error) {
	resp, err := client.Responses.New(ctx, newParams(req, effort),
		option.WithMaxRetries(p.chatMaxRetries),
		option.WithRequestTimeout(p.chatTimeout),
	)
	if err != nil {
		return llm.Response{}, p.scrub(translateError(ctx, err))
	}
	result, err := convertResponse(resp)
	return result, p.scrub(err)
}

// scrub removes the API key from a user-facing error message, in case the
// server echoed it in text that is shown.
func (p *Provider) scrub(err error) error {
	var e *llm.Error
	if p.apiKey != "" && errors.As(err, &e) {
		e.Message = strings.ReplaceAll(e.Message, p.apiKey, "[redacted]")
	}
	return err
}

// newParams maps a request to Responses API parameters. The model ID and
// effort are passed through unchanged, so any value the API accepts can be
// used; effort "" omits the reasoning parameter.
func newParams(req llm.Request, effort string) responses.ResponseNewParams {
	input := make(responses.ResponseInputParam, 0, len(req.Messages))
	for _, m := range req.Messages {
		input = append(input, responses.ResponseInputItemParamOfMessage(m.Content, responses.EasyInputMessageRole(m.Role)))
	}
	params := responses.ResponseNewParams{
		Model: req.Model,
		Input: responses.ResponseNewParamsInputUnion{OfInputItemList: input},
		Store: sdk.Bool(false), // explicit: never rely on the API default
	}
	if req.Instructions != "" {
		params.Instructions = sdk.String(req.Instructions)
	}
	if effort != "" {
		params.Reasoning = shared.ReasoningParam{Effort: shared.ReasoningEffort(effort)}
	}
	if req.MaxOutputTokens > 0 {
		params.MaxOutputTokens = sdk.Int(int64(req.MaxOutputTokens))
	}
	return params
}
