package llm

import "errors"

// ErrorKind classifies provider failures so callers can react to each one.
type ErrorKind int

const (
	KindAPI            ErrorKind = iota // HTTP error, invalid response, unknown or cloud model
	KindTimeout                         // the request timed out
	KindConnection                      // the server could not be reached or the connection failed
	KindMissingAPIKey                   // a required API key is not configured
	KindAuthentication                  // the server rejected the credentials
	KindRateLimit                       // rate limit or quota exceeded
)

func (k ErrorKind) String() string {
	switch k {
	case KindAPI:
		return "api"
	case KindTimeout:
		return "timeout"
	case KindConnection:
		return "connection"
	case KindMissingAPIKey:
		return "missing_api_key"
	case KindAuthentication:
		return "authentication"
	case KindRateLimit:
		return "rate_limit"
	default:
		return "unknown"
	}
}

// ExitCode is the process exit code for a one-shot or startup failure of
// this kind (docs/tui-spec.md, section 16).
func (k ErrorKind) ExitCode() int {
	switch k {
	case KindMissingAPIKey:
		return 3
	case KindAuthentication:
		return 4
	case KindRateLimit:
		return 5
	case KindTimeout:
		return 6
	case KindConnection:
		return 7
	default:
		return 1
	}
}

// Error is a user-facing, credential-free provider error.
type Error struct {
	Kind    ErrorKind
	Message string // shown to the user as is
	Err     error  // underlying cause, if any; for errors.Is / errors.As
}

func (e *Error) Error() string { return e.Message }

func (e *Error) Unwrap() error { return e.Err }

// ExitCode maps an error to a process exit code: the kind's code for an
// *Error anywhere in the chain, otherwise 1 (API error).
func ExitCode(err error) int {
	var e *Error
	if errors.As(err, &e) {
		return e.Kind.ExitCode()
	}
	return KindAPI.ExitCode()
}
