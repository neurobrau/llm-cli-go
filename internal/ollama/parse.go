package ollama

import (
	"encoding/json"
	"strings"
	"time"
	"unicode"

	"github.com/neurobrau/llm-cli-go/internal/llm"
)

// object is a JSON object whose fields are decoded on demand, so that an
// optional field of an unexpected type is ignored instead of failing the
// whole reply.
type object map[string]json.RawMessage

func decodeObject(data []byte) (object, bool) {
	var o object
	if json.Unmarshal(data, &o) != nil || o == nil {
		return nil, false
	}
	return o, true
}

func (o object) str(key string) (string, bool) {
	var s string
	if raw, ok := o[key]; ok && json.Unmarshal(raw, &s) == nil {
		return s, true
	}
	return "", false
}

// integer returns an integral JSON number; floats, strings and booleans are
// not integers.
func (o object) integer(key string) (int64, bool) {
	var n int64
	if raw, ok := o[key]; ok && json.Unmarshal(raw, &n) == nil {
		return n, true
	}
	return 0, false
}

func (o object) obj(key string) (object, bool) {
	raw, ok := o[key]
	if !ok {
		return nil, false
	}
	return decodeObject(raw)
}

// truthy reports whether the field is present and not null, false, 0, ""
// or an empty array/object.
func (o object) truthy(key string) bool {
	var v any
	if raw, ok := o[key]; !ok || json.Unmarshal(raw, &v) != nil {
		return false
	}
	switch v := v.(type) {
	case nil:
		return false
	case bool:
		return v
	case float64:
		return v != 0
	case string:
		return v != ""
	case []any:
		return len(v) > 0
	case map[string]any:
		return len(v) > 0
	}
	return true
}

// stringList returns the string items of an array field; ok is false if the
// field is missing or not an array.
func (o object) stringList(key string) (list []string, ok bool) {
	var items []any
	if raw, found := o[key]; !found || json.Unmarshal(raw, &items) != nil || items == nil {
		return nil, false
	}
	list = []string{}
	for _, item := range items {
		if s, isString := item.(string); isString {
			list = append(list, s)
		}
	}
	return list, true
}

// isCloudName reports whether the model tag marks an Ollama cloud model:
// `glm-4.7:cloud`, `gemma4:31b-cloud`.
func isCloudName(id string) bool {
	tag := id[strings.LastIndex(id, ":")+1:]
	return tag == "cloud" || strings.HasSuffix(tag, "-cloud")
}

func parseTags(data []byte) ([]llm.Model, error) {
	o, ok := decodeObject(data)
	var entries []json.RawMessage
	if !ok || json.Unmarshal(o["models"], &entries) != nil || entries == nil {
		return nil, invalidResponse("expected an object with a 'models' list")
	}
	models := make([]llm.Model, 0, len(entries))
	for _, raw := range entries {
		model, err := parseTagEntry(raw)
		if err != nil {
			return nil, err
		}
		models = append(models, model)
	}
	return models, nil
}

func parseTagEntry(raw json.RawMessage) (llm.Model, error) {
	entry, ok := decodeObject(raw)
	if !ok {
		return llm.Model{}, invalidResponse("model entry is not an object")
	}
	id, _ := entry.str("model")
	if id == "" {
		id, _ = entry.str("name")
	}
	if id == "" {
		return llm.Model{}, invalidResponse("model entry has no name")
	}
	model := llm.Model{
		ID:     id,
		Remote: entry.truthy("remote_host") || entry.truthy("remote_model") || isCloudName(id),
	}
	if caps, ok := entry.stringList("capabilities"); ok {
		model.Capabilities = caps
	}
	if details, ok := entry.obj("details"); ok {
		var summary []string
		for _, key := range []string{"parameter_size", "quantization_level"} {
			if s, _ := details.str(key); s != "" {
				summary = append(summary, s)
			}
		}
		model.Details = strings.Join(summary, " ")
		if n, ok := details.integer("context_length"); ok && n > 0 {
			model.ContextLength = int(n)
		}
	}
	return model, nil
}

// isSelectable reports whether a model is local and can chat. Models
// without reported capabilities (older Ollama) are given the benefit of the
// doubt.
func isSelectable(m llm.Model) bool {
	return !m.Remote && (m.Capabilities == nil || hasCapability(m.Capabilities, "completion"))
}

func hasCapability(caps []string, name string) bool {
	for _, c := range caps {
		if c == name {
			return true
		}
	}
	return false
}

// showReply is what NeuroBräu uses from /api/show.
type showReply struct {
	capabilities   []string // nil if not reported
	thinkingValues []string // thinking.values: false/true → off/on, strings as is
}

func parseShow(data []byte) (showReply, error) {
	o, ok := decodeObject(data)
	if !ok {
		return showReply{}, invalidResponse("expected a JSON object from /api/show")
	}
	var reply showReply
	reply.capabilities, _ = o.stringList("capabilities")
	var values []any
	if thinking, ok := o.obj("thinking"); ok && json.Unmarshal(thinking["values"], &values) == nil {
		for _, v := range values {
			switch v := v.(type) {
			case bool:
				reply.thinkingValues = append(reply.thinkingValues, map[bool]string{false: "off", true: "on"}[v])
			case string:
				if v != "" {
					reply.thinkingValues = append(reply.thinkingValues, v)
				}
			}
		}
	}
	return reply, nil
}

func parseChatResponse(data []byte) (llm.Response, error) {
	o, ok := decodeObject(data)
	if !ok {
		return llm.Response{}, invalidResponse("expected a JSON object")
	}
	if msg, ok := o.str("error"); ok {
		return llm.Response{}, &llm.Error{Kind: llm.KindAPI, Message: "Ollama error: " + msg}
	}
	message, ok := o.obj("message")
	if !ok {
		return llm.Response{}, invalidResponse("missing 'message.content'")
	}
	content, ok := message.str("content")
	if !ok {
		return llm.Response{}, invalidResponse("missing 'message.content'")
	}
	thinking, _ := message.str("thinking")

	var resp llm.Response
	resp.Text, resp.Reasoning = splitReasoning(content, thinking)
	resp.Model, _ = o.str("model")
	resp.DoneReason, _ = o.str("done_reason")
	if resp.DoneReason == "length" {
		resp.IncompleteReason = "the max output token limit was reached"
	}
	resp.Usage = parseUsage(o)
	resp.Timing = llm.Timing{
		Total:      duration(o, "total_duration"),
		Load:       duration(o, "load_duration"),
		PromptEval: duration(o, "prompt_eval_duration"),
		Eval:       duration(o, "eval_duration"),
	}
	return resp, nil
}

// parseUsage maps prompt_eval_count / eval_count to input / output tokens;
// the total is their sum when both are reported.
func parseUsage(o object) *llm.Usage {
	input, hasInput := o.integer("prompt_eval_count")
	output, hasOutput := o.integer("eval_count")
	if !hasInput && !hasOutput {
		return nil
	}
	var usage llm.Usage
	if hasInput {
		usage.InputTokens = intPtr(int(input))
	}
	if hasOutput {
		usage.OutputTokens = intPtr(int(output))
	}
	if hasInput && hasOutput {
		usage.TotalTokens = intPtr(int(input + output))
	}
	return &usage
}

func intPtr(n int) *int { return &n }

// duration reads a nanosecond count; missing or negative values are zero.
func duration(o object, key string) time.Duration {
	if n, ok := o.integer(key); ok && n > 0 {
		return time.Duration(n)
	}
	return 0
}

const (
	thinkOpen  = "<think>"
	thinkClose = "</think>"
)

// splitReasoning separates reasoning from the answer: the `thinking` field,
// or a leading <think>…</think> block in the content. An unclosed leading
// <think> block is all reasoning, never answer text.
func splitReasoning(content, thinking string) (text, reasoning string) {
	if strings.TrimSpace(thinking) != "" {
		reasoning = thinking
	}
	trimmed := strings.TrimLeftFunc(content, unicode.IsSpace)
	if !strings.HasPrefix(trimmed, thinkOpen) {
		return content, reasoning
	}
	inner, rest, closed := strings.Cut(trimmed[len(thinkOpen):], thinkClose)
	if reasoning == "" {
		reasoning = strings.TrimSpace(inner)
	}
	if !closed {
		return "", reasoning
	}
	return strings.TrimLeftFunc(rest, unicode.IsSpace), reasoning
}
