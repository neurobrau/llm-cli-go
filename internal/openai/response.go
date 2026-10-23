package openai

import (
	"fmt"
	"regexp"
	"strings"

	"github.com/openai/openai-go/v3/packages/respjson"
	"github.com/openai/openai-go/v3/responses"

	"github.com/neurobrau/llm-cli-go/internal/llm"
)

// convertResponse maps an SDK response to llm.Response. Failed, refused and
// unfinished responses are errors; a completed response may have empty
// text. An incomplete response keeps its partial text and reports why it
// stopped.
func convertResponse(r *responses.Response) (llm.Response, error) {
	if r == nil {
		return llm.Response{}, apiError("Invalid response from OpenAI: empty reply.")
	}
	text, refusal := outputText(r.Output)
	resp := llm.Response{
		Text:        text,
		ID:          r.ID,
		Model:       r.Model,
		ServiceTier: string(r.ServiceTier),
		Usage:       convertUsage(r),
	}

	switch r.Status {
	case responses.ResponseStatusCompleted, "":
	case responses.ResponseStatusIncomplete:
		resp.IncompleteReason = incompleteReason(r.IncompleteDetails.Reason)
		return resp, nil
	case responses.ResponseStatusFailed:
		message := oneLine(r.Error.Message)
		if message == "" {
			message = "no details"
		}
		return llm.Response{}, apiError("OpenAI response failed: " + message)
	default:
		return llm.Response{}, apiError(fmt.Sprintf("OpenAI response did not complete (status '%s').", oneLine(string(r.Status))))
	}

	if text == "" && refusal != "" {
		return llm.Response{}, apiError("OpenAI refused the request: " + oneLine(refusal))
	}
	return resp, nil
}

// outputText joins the `output_text` parts of the assistant message items,
// and separately any refusals. Reasoning, tool calls and other item types
// are ignored.
func outputText(items []responses.ResponseOutputItemUnion) (text, refusal string) {
	var texts, refusals strings.Builder
	for _, item := range items {
		if item.Type != "message" {
			continue
		}
		for _, part := range item.Content {
			switch part.Type {
			case "output_text":
				texts.WriteString(part.Text)
			case "refusal":
				refusals.WriteString(part.Refusal)
			}
		}
	}
	return texts.String(), refusals.String()
}

func incompleteReason(reason string) string {
	switch reason {
	case "max_output_tokens":
		return "the max output token limit was reached"
	case "content_filter":
		return "the content filter stopped the response"
	case "":
		return "unknown reason"
	default:
		return oneLine(reason)
	}
}

// convertUsage returns the reported token counts, or nil if the response
// has no usage. Missing or negative counts stay nil.
func convertUsage(r *responses.Response) *llm.Usage {
	if !r.JSON.Usage.Valid() {
		return nil
	}
	u := r.Usage
	return &llm.Usage{
		InputTokens:       count(u.InputTokens, u.JSON.InputTokens),
		OutputTokens:      count(u.OutputTokens, u.JSON.OutputTokens),
		TotalTokens:       count(u.TotalTokens, u.JSON.TotalTokens),
		CachedInputTokens: count(u.InputTokensDetails.CachedTokens, u.InputTokensDetails.JSON.CachedTokens),
		CacheWriteTokens:  count(u.InputTokensDetails.CacheWriteTokens, u.InputTokensDetails.JSON.CacheWriteTokens),
		ReasoningTokens:   count(u.OutputTokensDetails.ReasoningTokens, u.OutputTokensDetails.JSON.ReasoningTokens),
	}
}

func count(value int64, field respjson.Field) *int {
	if !field.Valid() || value < 0 {
		return nil
	}
	n := int(value)
	return &n
}

// secretPattern matches OpenAI-style API keys.
var secretPattern = regexp.MustCompile(`sk-[A-Za-z0-9_-]{6,}`)

// oneLine collapses whitespace, hides anything that looks like an API key
// and caps the length of text from the API before it is shown to the user.
func oneLine(s string) string {
	s = secretPattern.ReplaceAllString(strings.Join(strings.Fields(s), " "), "[redacted]")
	if runes := []rune(s); len(runes) > 300 {
		s = string(runes[:300]) + "…"
	}
	return s
}

func apiError(message string) *llm.Error {
	return &llm.Error{Kind: llm.KindAPI, Message: message}
}
