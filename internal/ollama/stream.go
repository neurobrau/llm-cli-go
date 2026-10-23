package ollama

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"net/http"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/neurobrau/llm-cli-go/internal/llm"
)

var _ llm.StreamingProvider = (*Provider)(nil)

const (
	streamReadSize = 32 << 10
	// maxStreamLine bounds one NDJSON line; Ollama's lines are small.
	maxStreamLine = 16 << 20
)

// errStalled is the cancellation cause when nothing arrived for the chat
// timeout.
var errStalled = errors.New("no data from Ollama within the timeout")

// Stream streams /api/chat (`"stream": true`) with the same body and model
// checks as Chat (docs/tui-spec.md, section 14). The NDJSON body is read
// in raw chunks as they arrive; lines and UTF-8 characters split across
// chunks are decoded intact. `message.thinking` and a leading
// `<think>…</think>` block in the content become ReasoningDelta events,
// never answer text.
//
// There is no limit on the whole answer: the stream fails with a timeout
// only when nothing arrives for ChatTimeout (300 s by default), counted
// from sending the request (which includes loading the model) and again
// after every read.
func (p *Provider) Stream(ctx context.Context, req llm.Request) iter.Seq[llm.StreamEvent] {
	return func(yield func(llm.StreamEvent) bool) {
		payload, err := p.chatPayload(ctx, req)
		if err != nil {
			yield(llm.StreamFailed{Err: err})
			return
		}
		payload.Stream = true

		streamCtx, cancel := context.WithCancelCause(ctx)
		defer cancel(nil)
		idle := llm.NewIdleTimer(p.chatTimeout, func() { cancel(errStalled) })
		defer idle.Stop()

		body, err := p.openStream(ctx, streamCtx, payload)
		if err != nil {
			yield(llm.StreamFailed{Err: explainHTTPError(err, payload.Model)})
			return
		}
		defer body.Close()

		s := chatStream{model: payload.Model, yield: yield}
		terminal, ok := s.read(body, idle)
		if !ok {
			return // the consumer stopped
		}
		if failed, isFailed := terminal.(llm.StreamFailed); isFailed && failed.Err == nil {
			// A read error: the caller's cancellation, the idle timeout or a
			// broken connection.
			terminal = llm.StreamFailed{Err: p.streamReadError(ctx, streamCtx, s.readErr), PartialText: failed.PartialText}
		}
		yield(terminal)
	}
}

// openStream sends the request and returns the body of a 2xx reply. An
// HTTP error status is read (a little) and closed.
func (p *Provider) openStream(ctx, streamCtx context.Context, payload chatRequest) (io.ReadCloser, error) {
	data, err := json.Marshal(payload)
	if err != nil {
		return nil, &llm.Error{Kind: llm.KindAPI, Message: "Could not encode the Ollama request.", Err: err}
	}
	req, err := http.NewRequestWithContext(streamCtx, http.MethodPost, p.baseURL+"/api/chat", bytes.NewReader(data))
	if err != nil {
		return nil, &llm.Error{Kind: llm.KindConnection, Message: fmt.Sprintf("Invalid Ollama URL %q.", p.baseURL), Err: err}
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := p.client.Do(req)
	if err != nil {
		return nil, p.streamTransportError(ctx, streamCtx, err,
			fmt.Sprintf("Could not connect to Ollama at %s. Is the Ollama service running?", p.baseURL))
	}
	if resp.StatusCode < 200 || resp.StatusCode > 299 {
		defer resp.Body.Close()
		detail, _ := io.ReadAll(io.LimitReader(resp.Body, 64<<10))
		httpErr := &HTTPError{StatusCode: resp.StatusCode, Detail: errorDetail(detail)}
		return nil, &llm.Error{Kind: llm.KindAPI, Message: httpErr.Error(), Err: httpErr}
	}
	return resp.Body, nil
}

func (p *Provider) streamTransportError(ctx, streamCtx context.Context, err error, connectionMessage string) error {
	if ctx.Err() == nil && errors.Is(context.Cause(streamCtx), errStalled) {
		return &llm.Error{Kind: llm.KindTimeout, Message: "Request to Ollama timed out.", Err: err}
	}
	return p.transportError(ctx, err, connectionMessage)
}

func (p *Provider) streamReadError(ctx, streamCtx context.Context, err error) error {
	return p.streamTransportError(ctx, streamCtx, err,
		fmt.Sprintf("Connection to Ollama at %s failed: %v", p.baseURL, err))
}

// chatStream turns the NDJSON lines of one response into events.
type chatStream struct {
	model     string
	yield     func(llm.StreamEvent) bool
	answer    strings.Builder
	reasoning strings.Builder
	think     thinkFilter
	readErr   error
}

// read consumes body until the final line. It returns the terminal event
// (not yet yielded), or ok == false if the consumer stopped. A read error
// gives a StreamFailed with a nil Err and s.readErr set.
func (s *chatStream) read(body io.Reader, idle *llm.IdleTimer) (terminal llm.StreamEvent, ok bool) {
	buf := make([]byte, streamReadSize)
	var pending []byte
	for {
		idle.Reset() // runs only while waiting for data
		n, err := body.Read(buf)
		idle.Stop()
		pending = append(pending, buf[:n]...)
		for {
			i := bytes.IndexByte(pending, '\n')
			if i < 0 {
				break
			}
			line := pending[:i]
			pending = pending[i+1:]
			if terminal, ok := s.line(line); terminal != nil || !ok {
				return terminal, ok
			}
		}
		if len(pending) > maxStreamLine {
			return s.failed(invalidResponse("stream line is too long")), true
		}
		if errors.Is(err, io.EOF) {
			// A last line without a newline still counts.
			if terminal, ok := s.line(pending); terminal != nil || !ok {
				return terminal, ok
			}
			return s.failed(&llm.Error{
				Kind:    llm.KindConnection,
				Message: "Ollama stream ended before the response was complete.",
			}), true
		}
		if err != nil {
			s.readErr = err
			return llm.StreamFailed{PartialText: s.answer.String()}, true
		}
	}
}

func (s *chatStream) failed(err error) llm.StreamEvent {
	return llm.StreamFailed{Err: err, PartialText: s.answer.String()}
}

// line handles one NDJSON line. It returns a terminal event when the
// stream ends with it, or ok == false if the consumer stopped.
func (s *chatStream) line(line []byte) (terminal llm.StreamEvent, ok bool) {
	line = bytes.TrimSpace(line) // also a trailing "\r"
	if len(line) == 0 {
		return nil, true
	}
	if !utf8.Valid(line) || !json.Valid(line) {
		return s.failed(invalidResponse("stream line is not valid JSON")), true
	}
	o, isObject := decodeObject(line)
	if !isObject {
		return s.failed(invalidResponse("stream line is not a JSON object")), true
	}
	if msg, isString := o.str("error"); isString {
		return s.failed(&llm.Error{Kind: llm.KindAPI, Message: "Ollama error: " + msg}), true
	}
	var content, thinking string
	if raw, present := o["message"]; present {
		message, isObj := decodeObject(raw)
		rawContent, hasContent := message["content"]
		if !isObj || hasContent && (len(rawContent) == 0 || rawContent[0] != '"') {
			return s.failed(invalidResponse("stream line has no valid 'message.content'")), true
		}
		content, _ = message.str("content")
		thinking, _ = message.str("thinking") // another type: no thinking
	}

	if thinking != "" && !s.emit(llm.ReasoningDelta{Text: thinking}) {
		return nil, false
	}
	text, inline := s.think.feed(content)
	if inline != "" && !s.emit(llm.ReasoningDelta{Text: inline}) {
		return nil, false
	}
	if text != "" && !s.emit(llm.TextDelta{Text: text}) {
		return nil, false
	}
	if string(o["done"]) != "true" {
		return nil, true
	}
	if rest := s.think.flush(); rest != "" && !s.emit(llm.TextDelta{Text: rest}) {
		return nil, false
	}
	return s.finish(o), true
}

// emit yields a delta and keeps the text for the terminal event.
func (s *chatStream) emit(event llm.StreamEvent) bool {
	switch e := event.(type) {
	case llm.TextDelta:
		s.answer.WriteString(e.Text)
	case llm.ReasoningDelta:
		s.reasoning.WriteString(e.Text)
	}
	return s.yield(event)
}

// finish builds the terminal event from the final (`"done": true`) line.
func (s *chatStream) finish(o object) llm.StreamEvent {
	usage := parseUsage(o)
	doneReason, _ := o.str("done_reason")
	if doneReason == "length" {
		return llm.StreamIncomplete{
			Reason: "the max output token limit was reached", PartialText: s.answer.String(), Usage: usage,
		}
	}
	model, _ := o.str("model")
	if model == "" {
		model = s.model
	}
	return llm.StreamCompleted{Response: llm.Response{
		Text: s.answer.String(), Reasoning: strings.TrimSpace(s.reasoning.String()), Usage: usage,
		Model: model, DoneReason: doneReason,
		Timing: llm.Timing{
			Total:      duration(o, "total_duration"),
			Load:       duration(o, "load_duration"),
			PromptEval: duration(o, "prompt_eval_duration"),
			Eval:       duration(o, "eval_duration"),
		},
	}}
}

// thinkFilter separates a leading `<think>…</think>` block, which some
// models put in the answer content, from the answer while it streams.
// Text that could still turn into the opening tag is held back; inside
// the block, reasoning is passed on as it arrives, keeping back only what
// could be the start of the closing tag, so memory stays bounded.
type thinkFilter struct {
	state   int // filterUndecided, filterThinking, filterAfterBlock, filterAnswer
	pending string
}

const (
	filterUndecided = iota
	filterThinking
	filterAfterBlock // skipping whitespace after the closing tag
	filterAnswer
)

// feed returns the answer and reasoning text that can be passed on now.
func (f *thinkFilter) feed(text string) (answer, reasoning string) {
	switch f.state {
	case filterAnswer:
		return text, ""
	case filterAfterBlock:
		answer = strings.TrimLeftFunc(text, unicode.IsSpace)
		if answer != "" {
			f.state = filterAnswer
		}
		return answer, ""
	case filterUndecided:
		f.pending += text
		trimmed := strings.TrimLeftFunc(f.pending, unicode.IsSpace)
		if strings.HasPrefix(thinkOpen, trimmed) {
			return "", "" // could still become "<think>"
		}
		if !strings.HasPrefix(trimmed, thinkOpen) {
			f.state = filterAnswer
			answer, f.pending = f.pending, ""
			return answer, ""
		}
		f.state = filterThinking
		f.pending = trimmed[len(thinkOpen):]
		return f.thinking("")
	default:
		return f.thinking(text)
	}
}

// thinking handles text inside the block.
func (f *thinkFilter) thinking(text string) (answer, reasoning string) {
	f.pending += text
	if inner, rest, closed := strings.Cut(f.pending, thinkClose); closed {
		f.pending = ""
		f.state = filterAfterBlock
		answer, _ = f.feed(rest)
		return answer, inner
	}
	// Keep back what could be the start of "</think>", on a rune boundary.
	keep := len(f.pending) - (len(thinkClose) - 1)
	for keep > 0 && !utf8.RuneStart(f.pending[keep]) {
		keep--
	}
	if keep <= 0 {
		return "", ""
	}
	reasoning, f.pending = f.pending[:keep], f.pending[keep:]
	return "", reasoning
}

// flush returns answer text still held back when the stream ends. A
// partial tag such as "<th" was real text; an unclosed block is thinking
// and never becomes the answer.
func (f *thinkFilter) flush() string {
	text := f.pending
	f.pending = ""
	if f.state != filterUndecided || strings.HasPrefix(strings.TrimLeftFunc(text, unicode.IsSpace), thinkOpen) {
		return ""
	}
	return text
}
