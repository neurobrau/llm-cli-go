// Package ollama implements llm.Provider for a local Ollama server using
// its native HTTP API (`/api/tags`, `/api/show`, `/api/chat`).
//
// Only models that run locally can be used. Ollama also lists cloud models
// (e.g. `glm-4.7:cloud`) that forward requests to ollama.com; those are
// never selected, queried or sent chat requests. Models are never
// downloaded automatically.
package ollama

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"slices"
	"strings"
	"time"

	"github.com/neurobrau/llm-cli-go/internal/llm"
)

const (
	// ChatTimeout bounds a chat request; the first one may load the model.
	ChatTimeout = 300 * time.Second
	// DiscoveryTimeout bounds /api/tags and /api/show requests.
	DiscoveryTimeout = 3 * time.Second
)

// Sentinel causes of model-selection errors, for errors.Is.
var (
	ErrNoLocalModels     = errors.New("no local Ollama models")
	ErrModelNotInstalled = errors.New("Ollama model not installed")
	ErrCloudModel        = errors.New("Ollama cloud model")
)

// Config configures a Provider. Zero values select the defaults.
type Config struct {
	BaseURL          string       // default DefaultBaseURL
	HTTPClient       *http.Client // default http.DefaultClient
	ChatTimeout      time.Duration
	DiscoveryTimeout time.Duration
}

var _ llm.Provider = (*Provider)(nil)

// Provider talks to one Ollama server. Discovery results are cached for the
// provider's lifetime. A Provider is not safe for concurrent use.
type Provider struct {
	baseURL          string
	client           *http.Client
	chatTimeout      time.Duration
	discoveryTimeout time.Duration

	models []llm.Model          // everything /api/tags reported; nil: not loaded
	shows  map[string]showReply // /api/show replies, per model
}

// New returns a Provider. Nothing is requested until a method is called.
func New(cfg Config) *Provider {
	p := &Provider{
		baseURL:          cfg.BaseURL,
		client:           cfg.HTTPClient,
		chatTimeout:      cfg.ChatTimeout,
		discoveryTimeout: cfg.DiscoveryTimeout,
		shows:            map[string]showReply{},
	}
	if p.baseURL == "" {
		p.baseURL = DefaultBaseURL
	}
	if p.client == nil {
		p.client = http.DefaultClient
	}
	if p.chatTimeout <= 0 {
		p.chatTimeout = ChatTimeout
	}
	if p.discoveryTimeout <= 0 {
		p.discoveryTimeout = DiscoveryTimeout
	}
	return p
}

// Name returns the CLI provider name.
func (p *Provider) Name() string { return "ollama" }

// discover returns all models from /api/tags, including remote ones. It is
// cached unless refresh is set.
func (p *Provider) discover(ctx context.Context, refresh bool) ([]llm.Model, error) {
	if p.models != nil && !refresh {
		return p.models, nil
	}
	data, err := p.request(ctx, http.MethodGet, "/api/tags", nil, p.discoveryTimeout)
	if err != nil {
		return nil, err
	}
	models, err := parseTags(data)
	if err != nil {
		return nil, err
	}
	p.models = models
	return models, nil
}

// ListModels returns the installed local chat models in Ollama's order
// (most recently modified first).
func (p *Provider) ListModels(ctx context.Context) ([]llm.Model, error) {
	all, err := p.discover(ctx, false)
	if err != nil {
		return nil, err
	}
	var models []llm.Model
	for _, m := range all {
		if isSelectable(m) {
			models = append(models, m)
		}
	}
	return models, nil
}

// DefaultModel returns the first selectable model in /api/tags order.
func (p *Provider) DefaultModel(ctx context.Context) (string, error) {
	models, err := p.ListModels(ctx)
	if err != nil {
		return "", err
	}
	if len(models) == 0 {
		return "", &llm.Error{
			Kind: llm.KindAPI,
			Message: "No local Ollama models are installed (cloud models are not used). " +
				"Install one with: ollama pull <model>",
			Err: ErrNoLocalModels,
		}
	}
	return models[0].ID, nil
}

// localModel looks up an installed local model, refreshing the list once
// in case it was installed since. Cloud and missing models are refused.
func (p *Provider) localModel(ctx context.Context, id string) (llm.Model, error) {
	model, found, err := p.find(ctx, id, false)
	if err == nil && !found {
		model, found, err = p.find(ctx, id, true)
	}
	if err != nil {
		return llm.Model{}, err
	}
	if !found {
		return llm.Model{}, &llm.Error{
			Kind: llm.KindAPI,
			Message: fmt.Sprintf("Ollama model '%s' is not installed locally. "+
				"Install it yourself with: ollama pull %s", id, id),
			Err: ErrModelNotInstalled,
		}
	}
	if model.Remote {
		return llm.Model{}, &llm.Error{
			Kind: llm.KindAPI,
			Message: fmt.Sprintf("'%s' is an Ollama cloud model that runs on a remote service. "+
				"Choose a local model.", id),
			Err: ErrCloudModel,
		}
	}
	return model, nil
}

func (p *Provider) find(ctx context.Context, id string, refresh bool) (llm.Model, bool, error) {
	models, err := p.discover(ctx, refresh)
	if err != nil {
		return llm.Model{}, false, err
	}
	for _, m := range models {
		if m.ID == id {
			return m, true, nil
		}
	}
	return llm.Model{}, false, nil
}

// show returns the cached /api/show reply for a model, making one request
// the first time. Only ever called for local models.
func (p *Provider) show(ctx context.Context, model llm.Model) (showReply, error) {
	if reply, ok := p.shows[model.ID]; ok {
		return reply, nil
	}
	data, err := p.request(ctx, http.MethodPost, "/api/show", map[string]string{"model": model.ID}, p.discoveryTimeout)
	if err != nil {
		return showReply{}, err
	}
	reply, err := parseShow(data)
	if err != nil {
		return showReply{}, err
	}
	p.shows[model.ID] = reply
	return reply, nil
}

// modelCapabilities returns the capabilities from /api/tags or, if the tags
// entry omits them, from /api/show.
func (p *Provider) modelCapabilities(ctx context.Context, model llm.Model) ([]string, error) {
	if model.Capabilities != nil {
		return model.Capabilities, nil
	}
	reply, err := p.show(ctx, model)
	if err != nil {
		return nil, err
	}
	return reply.capabilities, nil
}

// ReasoningSupport returns the thinking choices of an installed local
// model: `/api/show` thinking values (false/true → off/on, strings → named
// levels), or off/on for the `thinking` capability without metadata.
// Thinking is off by default where possible; models with off/on have no
// "model default" choice.
func (p *Provider) ReasoningSupport(ctx context.Context, id string) (llm.ReasoningSupport, error) {
	model, err := p.localModel(ctx, id)
	if err != nil {
		return llm.ReasoningSupport{}, err
	}
	caps, err := p.modelCapabilities(ctx, model)
	if err != nil {
		return llm.ReasoningSupport{}, err
	}
	reply, err := p.show(ctx, model)
	if err != nil {
		return llm.ReasoningSupport{}, err
	}
	values := reply.thinkingValues
	if len(values) == 0 && hasCapability(caps, "thinking") {
		values = []string{"off", "on"}
	}
	support := llm.ReasoningSupport{Values: values, Known: true}
	if slices.Contains(values, "off") {
		support.Default = "off"
		support.NoModelDefault = true
	}
	return support, nil
}

// MaxOutputLimit returns the model's context length from /api/tags (0 if
// not reported); num_predict cannot usefully exceed it.
func (p *Provider) MaxOutputLimit(ctx context.Context, id string) (int, error) {
	model, err := p.localModel(ctx, id)
	if err != nil {
		return 0, err
	}
	return model.ContextLength, nil
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatOptions struct {
	NumPredict int `json:"num_predict"`
}

type chatRequest struct {
	Model    string        `json:"model"`
	Messages []chatMessage `json:"messages"`
	Stream   bool          `json:"stream"`
	Think    any           `json:"think,omitempty"` // false, true or a level name
	Options  *chatOptions  `json:"options,omitempty"`
}

// Chat sends a non-streaming /api/chat request. Instructions become a
// leading system message for this request only. Thinking models are asked
// not to think unless the request says otherwise; reasoning they return is
// kept out of the answer text.
func (p *Provider) Chat(ctx context.Context, req llm.Request) (llm.Response, error) {
	payload, err := p.chatPayload(ctx, req)
	if err != nil {
		return llm.Response{}, err
	}
	data, err := p.request(ctx, http.MethodPost, "/api/chat", payload, p.chatTimeout)
	if err != nil {
		return llm.Response{}, explainHTTPError(err, payload.Model)
	}
	return parseChatResponse(data)
}

// chatPayload checks the model (installed, local) and builds the
// /api/chat body shared by Chat and Stream.
func (p *Provider) chatPayload(ctx context.Context, req llm.Request) (chatRequest, error) {
	model, err := p.localModel(ctx, req.Model)
	if err != nil {
		return chatRequest{}, err
	}
	caps, err := p.modelCapabilities(ctx, model)
	if err != nil {
		return chatRequest{}, err
	}
	payload := chatRequest{Model: model.ID, Messages: make([]chatMessage, 0, len(req.Messages)+1)}
	if req.Instructions != "" {
		payload.Messages = append(payload.Messages, chatMessage{Role: string(llm.RoleSystem), Content: req.Instructions})
	}
	for _, m := range req.Messages {
		payload.Messages = append(payload.Messages, chatMessage{Role: string(m.Role), Content: m.Content})
	}
	if hasCapability(caps, "thinking") {
		if payload.Think, err = p.thinkValue(ctx, model, req.Reasoning); err != nil {
			return chatRequest{}, err
		}
	} else if r := req.Reasoning; r.Explicit && r.Effort != "" && r.Effort != "none" && r.Effort != "off" {
		return chatRequest{}, &llm.Error{
			Kind:    llm.KindAPI,
			Message: fmt.Sprintf("%s does not support thinking; turn reasoning off or choose another model.", model.ID),
		}
	}
	if req.MaxOutputTokens > 0 {
		payload.Options = &chatOptions{NumPredict: req.MaxOutputTokens}
	}
	return payload, nil
}

// thinkValue maps a reasoning setting to the `think` parameter of a model
// with the thinking capability; nil omits the parameter.
func (p *Provider) thinkValue(ctx context.Context, model llm.Model, r llm.Reasoning) (any, error) {
	switch {
	case r.Effort == "" && r.Explicit:
		return nil, nil // the model's own default
	case r.Effort == "", r.Effort == "none", r.Effort == "off":
		return false, nil
	case r.Effort == "on":
		return true, nil
	}
	support, err := p.ReasoningSupport(ctx, model.ID)
	if err != nil {
		return nil, err
	}
	switch {
	case slices.Contains(support.Values, r.Effort):
		return r.Effort, nil // a named level such as "high"
	case slices.Contains(support.Values, "on"):
		return true, nil // an effort name on an off/on model
	case r.Explicit:
		return nil, &llm.Error{
			Kind: llm.KindAPI,
			Message: fmt.Sprintf("%s does not support thinking level '%s'. Supported: %s.",
				model.ID, r.Effort, strings.Join(support.Values, ", ")),
		}
	}
	return true, nil
}

// explainHTTPError turns a 404 from /api/chat into an actionable message.
func explainHTTPError(err error, modelID string) error {
	var httpErr *HTTPError
	if !errors.As(err, &httpErr) || httpErr.StatusCode != http.StatusNotFound {
		return err
	}
	return &llm.Error{
		Kind: llm.KindAPI,
		Message: fmt.Sprintf("Ollama model '%s' was not found: %s. "+
			"Install it yourself with: ollama pull %s", modelID, httpErr.Detail, modelID),
		Err: httpErr,
	}
}
