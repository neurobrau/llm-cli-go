package llm

import (
	"context"
	"errors"
	"fmt"
	"testing"
)

func TestExitCode(t *testing.T) {
	tests := []struct {
		name string
		err  error
		want int
	}{
		{"api", &Error{Kind: KindAPI, Message: "bad"}, 1},
		{"timeout", &Error{Kind: KindTimeout, Message: "slow"}, 6},
		{"connection", &Error{Kind: KindConnection, Message: "down"}, 7},
		{"missing api key", &Error{Kind: KindMissingAPIKey, Message: "no key"}, 3},
		{"authentication", &Error{Kind: KindAuthentication, Message: "denied"}, 4},
		{"rate limit", &Error{Kind: KindRateLimit, Message: "slow down"}, 5},
		{"unknown kind", &Error{Kind: ErrorKind(99), Message: "?"}, 1},
		{"wrapped", fmt.Errorf("context: %w", &Error{Kind: KindConnection}), 7},
		{"plain error", errors.New("other"), 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := ExitCode(tt.err); got != tt.want {
				t.Errorf("ExitCode() = %d, want %d", got, tt.want)
			}
		})
	}
}

func TestErrorUnwrap(t *testing.T) {
	err := error(&Error{Kind: KindTimeout, Message: "Request timed out.", Err: context.DeadlineExceeded})
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Error("errors.Is does not find the cause")
	}
	if err.Error() != "Request timed out." {
		t.Errorf("Error() = %q", err.Error())
	}
	var e *Error
	if !errors.As(fmt.Errorf("wrapped: %w", err), &e) || e.Kind != KindTimeout {
		t.Error("errors.As does not find *Error")
	}
}

func TestErrorKindString(t *testing.T) {
	for kind, want := range map[ErrorKind]string{
		KindAPI: "api", KindTimeout: "timeout", KindConnection: "connection",
		KindMissingAPIKey: "missing_api_key", KindAuthentication: "authentication", KindRateLimit: "rate_limit",
		ErrorKind(99): "unknown",
	} {
		if got := kind.String(); got != want {
			t.Errorf("%d.String() = %q, want %q", int(kind), got, want)
		}
	}
}
