package ollama

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/neurobrau/llm-cli-go/internal/llm"
)

const (
	// HostEnvVar names the environment variable that selects the server.
	HostEnvVar = "OLLAMA_HOST"
	// DefaultBaseURL is used when OLLAMA_HOST is unset or empty.
	DefaultBaseURL = "http://127.0.0.1:11434"
	defaultPort    = "11434"
)

// HTTPError is the cause of an *llm.Error for an HTTP error status.
type HTTPError struct {
	StatusCode int
	Detail     string // Ollama's {"error": "..."} text, or the start of the body
}

func (e *HTTPError) Error() string {
	return fmt.Sprintf("Ollama API error (HTTP %d): %s", e.StatusCode, e.Detail)
}

// ResolveBaseURL turns an OLLAMA_HOST value (`host`, `host:port` or a URL)
// into a base URL. An empty value gives DefaultBaseURL; the bind-all
// addresses 0.0.0.0 and :: mean the local machine.
func ResolveBaseURL(host string) (string, error) {
	value := strings.TrimSpace(host)
	if value == "" {
		return DefaultBaseURL, nil
	}
	if !strings.Contains(value, "://") {
		value = "http://" + value
	}
	u, err := url.Parse(value)
	if err != nil {
		return "", fmt.Errorf("invalid %s value %q: %w", HostEnvVar, host, err)
	}
	hostname := u.Hostname()
	if hostname == "" || hostname == "0.0.0.0" || hostname == "::" {
		hostname = "127.0.0.1"
	}
	port := u.Port()
	if port == "" {
		port = defaultPort
	}
	return u.Scheme + "://" + net.JoinHostPort(hostname, port) + strings.TrimRight(u.Path, "/"), nil
}

// request sends one JSON request (payload nil: no body) and returns the
// response body of a 2xx reply. timeout bounds the whole exchange.
func (p *Provider) request(ctx context.Context, method, path string, payload any, timeout time.Duration) ([]byte, error) {
	var body io.Reader
	if payload != nil {
		data, err := json.Marshal(payload)
		if err != nil {
			return nil, &llm.Error{Kind: llm.KindAPI, Message: "Could not encode the Ollama request.", Err: err}
		}
		body = bytes.NewReader(data)
	}
	reqCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	req, err := http.NewRequestWithContext(reqCtx, method, p.baseURL+path, body)
	if err != nil {
		return nil, &llm.Error{Kind: llm.KindConnection, Message: fmt.Sprintf("Invalid Ollama URL %q.", p.baseURL), Err: err}
	}
	if payload != nil {
		req.Header.Set("Content-Type", "application/json")
	}

	// http.ProxyFromEnvironment, used by the default transport, never
	// proxies loopback hosts, so a local server is always reached directly.
	resp, err := p.client.Do(req)
	if err != nil {
		return nil, p.transportError(ctx, err,
			fmt.Sprintf("Could not connect to Ollama at %s. Is the Ollama service running?", p.baseURL))
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, p.transportError(ctx, err,
			fmt.Sprintf("Connection to Ollama at %s failed: %v", p.baseURL, err))
	}
	if resp.StatusCode < 200 || resp.StatusCode > 299 {
		httpErr := &HTTPError{StatusCode: resp.StatusCode, Detail: errorDetail(data)}
		return nil, &llm.Error{Kind: llm.KindAPI, Message: httpErr.Error(), Err: httpErr}
	}
	return data, nil
}

// transportError classifies a failed exchange. Cancellation by the caller
// is returned as is (wrapping context.Canceled), not as a provider error.
func (p *Provider) transportError(ctx context.Context, err error, connectionMessage string) error {
	if ctx.Err() != nil && errors.Is(ctx.Err(), context.Canceled) {
		return fmt.Errorf("Ollama request canceled: %w", ctx.Err())
	}
	var netErr net.Error
	if errors.Is(err, context.DeadlineExceeded) || (errors.As(err, &netErr) && netErr.Timeout()) {
		return &llm.Error{Kind: llm.KindTimeout, Message: "Request to Ollama timed out.", Err: err}
	}
	return &llm.Error{Kind: llm.KindConnection, Message: connectionMessage, Err: err}
}

// errorDetail extracts Ollama's {"error": "..."} message, falling back to
// the start of the raw body.
func errorDetail(body []byte) string {
	text := strings.TrimSpace(string(body))
	var payload struct {
		Error *string `json:"error"`
	}
	if json.Unmarshal([]byte(text), &payload) == nil && payload.Error != nil {
		return *payload.Error
	}
	if text == "" {
		return "no details"
	}
	return truncate(text, 200)
}

func truncate(s string, n int) string {
	runes := []rune(s)
	if len(runes) <= n {
		return s
	}
	return string(runes[:n])
}

func invalidResponse(reason string) *llm.Error {
	return &llm.Error{Kind: llm.KindAPI, Message: "Invalid response from Ollama: " + reason + "."}
}
