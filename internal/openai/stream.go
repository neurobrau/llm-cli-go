package openai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"strings"

	sdk "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/ssestream"

	"github.com/neurobrau/llm-cli-go/internal/llm"
)

var _ llm.StreamingProvider = (*Provider)(nil)

// errStalled is the cancellation cause when no stream data arrived for
// the chat timeout.
var errStalled = errors.New("no data from OpenAI within the timeout")

// Stream streams a Responses API request (`stream: true`) with the same
// parameters as Chat (docs/tui-spec.md, section 14). Only
// `response.output_text.delta` events become answer text.
//
// If the API rejects the built-in reasoning default before any text
// arrived, the request is sent once more without reasoning, as in Chat;
// an explicit effort is reported instead. There is no limit on the whole
// stream: it fails with a timeout only when no data arrives for
// ChatTimeout (600 s by default), including while connecting. SDK retries
// apply to the initial request only, never after data arrived.
func (p *Provider) Stream(ctx context.Context, req llm.Request) iter.Seq[llm.StreamEvent] {
	return func(yield func(llm.StreamEvent) bool) {
		client, err := p.sdkClient()
		if err != nil {
			yield(llm.StreamFailed{Err: err})
			return
		}
		effort := p.effort(ctx, req)
		terminal, ok := p.streamOnce(ctx, client, req, effort, yield)
		if !ok {
			return // the consumer stopped
		}
		if failed, isFailed := terminal.(llm.StreamFailed); isFailed && effort != "" &&
			failed.PartialText == "" && ctx.Err() == nil && rejectsReasoning(failed.Err) {
			if req.Reasoning.Explicit {
				terminal = llm.StreamFailed{Err: rejectedEffort(req.Model, effort, failed.Err)}
			} else {
				p.noReasoning[req.Model] = true
				if terminal, ok = p.streamOnce(ctx, client, req, "", yield); !ok {
					return
				}
			}
		}
		yield(terminal)
	}
}

// streamOnce sends one streaming request and yields its text deltas. It
// returns the terminal event without yielding it, or ok == false when the
// consumer stopped. The SDK stream (and so the HTTP response) is always
// closed before it returns.
func (p *Provider) streamOnce(ctx context.Context, client *sdk.Client, req llm.Request, effort string,
	yield func(llm.StreamEvent) bool) (terminal llm.StreamEvent, ok bool) {
	streamCtx, cancel := context.WithCancelCause(ctx)
	defer cancel(nil)
	idle := llm.NewIdleTimer(p.chatTimeout, func() { cancel(errStalled) })
	defer idle.Stop()

	stream := client.Responses.NewStreaming(streamCtx, newParams(req, effort), option.WithMaxRetries(p.chatMaxRetries))
	defer stream.Close()

	var text strings.Builder
	failed := func(err error) llm.StreamEvent {
		return llm.StreamFailed{Err: p.scrub(err), PartialText: text.String()}
	}
	for {
		idle.Reset() // runs only while waiting for data
		more := stream.Next()
		idle.Stop()
		if !more {
			break
		}
		event := stream.Current()
		switch event.Type {
		case "response.output_text.delta":
			// The SDK turns a number into a string; check the JSON itself.
			if raw := event.JSON.Delta.Raw(); !event.JSON.Delta.Valid() || !strings.HasPrefix(raw, `"`) {
				return failed(apiError("Invalid stream event from OpenAI: text delta is not a string.")), true
			}
			if event.Delta == "" {
				continue
			}
			text.WriteString(event.Delta)
			if !yield(llm.TextDelta{Text: event.Delta}) {
				return nil, false
			}
		case "response.completed":
			r := event.Response
			resp := llm.Response{
				Text: text.String(), ID: r.ID, Model: r.Model,
				ServiceTier: string(r.ServiceTier), Usage: convertUsage(&r),
			}
			if resp.Text == "" {
				resp.Text, _ = outputText(r.Output)
			}
			return llm.StreamCompleted{Response: resp}, true
		case "response.incomplete":
			r := event.Response
			return llm.StreamIncomplete{
				Reason: incompleteReason(r.IncompleteDetails.Reason), PartialText: text.String(),
				Usage: convertUsage(&r), ServiceTier: string(r.ServiceTier),
			}, true
		case "response.failed":
			return failed(apiError("OpenAI response failed: " + detail(event.Response.Error.Message))), true
		case "error":
			return failed(apiError("OpenAI stream error: " + detail(event.Message))), true
		}
	}
	if err := stream.Err(); err != nil {
		return failed(streamError(ctx, streamCtx, err)), true
	}
	return failed(apiError("OpenAI stream ended before the response was complete.")), true
}

// effort returns the reasoning effort to send: the explicit one, or the
// model's built-in default unless the API rejected it before.
func (p *Provider) effort(ctx context.Context, req llm.Request) string {
	if req.Reasoning.Explicit {
		return req.Reasoning.Effort
	}
	support, _ := p.ReasoningSupport(ctx, req.Model)
	return support.Default
}

func rejectedEffort(model, effort string, err error) *llm.Error {
	return &llm.Error{
		Kind: llm.KindAPI,
		Message: fmt.Sprintf("%s does not accept reasoning effort '%s' (%s). "+
			"Choose another value with --reasoning-effort.", model, effort, err),
		Err: err,
	}
}

// streamError classifies an error from the SDK stream: the caller's
// cancellation, the idle timeout, an `{"error": …}` payload, or the usual
// SDK and network errors.
func streamError(ctx, streamCtx context.Context, err error) error {
	if ctx.Err() == nil && errors.Is(context.Cause(streamCtx), errStalled) {
		return &llm.Error{Kind: llm.KindTimeout, Message: "Request to OpenAI timed out.", Err: err}
	}
	var streamErr *ssestream.StreamError
	if errors.As(err, &streamErr) {
		var payload struct {
			Error struct {
				Message string `json:"message"`
			} `json:"error"`
		}
		json.Unmarshal(streamErr.Event.Data, &payload) // an unreadable payload has no details
		return apiError("OpenAI stream error: " + detail(payload.Error.Message))
	}
	return translateError(ctx, err)
}

func detail(message string) string {
	if message = oneLine(message); message == "" {
		return "no details"
	}
	return message
}
