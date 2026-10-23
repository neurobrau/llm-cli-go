package openai

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"strings"

	sdk "github.com/openai/openai-go/v3"

	"github.com/neurobrau/llm-cli-go/internal/llm"
)

// translateError maps an SDK error to a categorized, credential-free
// error. Messages never include the request, its URL, headers or the raw
// error body. Cancellation by the caller is returned as is (wrapping
// context.Canceled), not as a provider error.
func translateError(ctx context.Context, err error) error {
	if ctx.Err() != nil && errors.Is(ctx.Err(), context.Canceled) {
		return fmt.Errorf("OpenAI request canceled: %w", ctx.Err())
	}

	var apiErr *sdk.Error
	if errors.As(err, &apiErr) {
		return statusError(apiErr)
	}

	var netErr net.Error
	if errors.Is(err, context.DeadlineExceeded) || (errors.As(err, &netErr) && netErr.Timeout()) {
		return &llm.Error{Kind: llm.KindTimeout, Message: "Request to OpenAI timed out.", Err: err}
	}

	var urlErr *url.Error
	var opErr *net.OpError
	if errors.As(err, &urlErr) || errors.As(err, &opErr) {
		return &llm.Error{
			Kind:    llm.KindConnection,
			Message: "Could not connect to OpenAI. Check your network connection.",
			Err:     err,
		}
	}

	// What remains are replies the SDK could not read or decode.
	return &llm.Error{Kind: llm.KindAPI, Message: "Invalid response from OpenAI.", Err: err}
}

// rejectsReasoning reports whether the API refused a request because of
// its reasoning parameter: HTTP 400 naming a `reasoning…` parameter. Other
// 400s, authentication, quota and server errors are not rejections.
func rejectsReasoning(err error) bool {
	var apiErr *sdk.Error
	return errors.As(err, &apiErr) && apiErr.StatusCode == http.StatusBadRequest &&
		strings.HasPrefix(apiErr.Param, "reasoning")
}

// statusError classifies an HTTP error status by its code.
func statusError(apiErr *sdk.Error) *llm.Error {
	switch status := apiErr.StatusCode; {
	case status == http.StatusUnauthorized:
		return &llm.Error{
			Kind:    llm.KindAuthentication,
			Message: "OpenAI authentication failed. Check that " + APIKeyEnvVar + " is valid.",
			Err:     apiErr,
		}
	case status == http.StatusForbidden:
		return &llm.Error{
			Kind: llm.KindAuthentication,
			Message: "OpenAI denied access (HTTP 403). Check that " + APIKeyEnvVar +
				" is valid and allowed to use this model.",
			Err: apiErr,
		}
	case status == http.StatusTooManyRequests:
		return &llm.Error{
			Kind:    llm.KindRateLimit,
			Message: "OpenAI rate limit or quota exceeded. Wait and try again, or check your plan and billing.",
			Err:     apiErr,
		}
	default:
		detail := oneLine(apiErr.Message)
		if detail == "" {
			detail = http.StatusText(status)
		}
		if detail == "" {
			detail = "no details"
		}
		return &llm.Error{
			Kind:    llm.KindAPI,
			Message: fmt.Sprintf("OpenAI API error (HTTP %d): %s", status, detail),
			Err:     apiErr,
		}
	}
}
