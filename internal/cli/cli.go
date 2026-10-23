// Package cli implements the neurobrau command-line interface.
//
// It is kept independent of os.Args, os.Stdout and os.Exit so that it can be
// tested without spawning processes.
package cli

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"slices"
	"strconv"
	"strings"

	"github.com/neurobrau/llm-cli-go/internal/llm"
	"github.com/neurobrau/llm-cli-go/internal/ollama"
	"github.com/neurobrau/llm-cli-go/internal/openai"
)

const (
	// ProgName is the executable name used in usage and error messages.
	ProgName = "neurobrau"
	// Version is the Go edition's own version. It does not imply feature
	// parity with the Python reference implementation (0.3.0).
	Version = "0.1.0-dev"
)

// Exit codes (docs/tui-spec.md, section 16). Provider failures use the
// codes of llm.ErrorKind.
const (
	ExitOK          = 0
	ExitUsage       = 2
	ExitInterrupted = 130
)

// Providers, in help order; the first is the default.
var providers = []string{"openai", "ollama"}

const helpText = `usage: neurobrau [-h] [--version] [--provider PROVIDER] [--model MODEL_ID]
                 [--reasoning-effort EFFORT] [--max-output-tokens N]
                 [--answer-style STYLE] [--usage]
                 [prompt]

NeuroBräu Go CLI — educational multi-provider LLM command-line client.

positional arguments:
  prompt                     prompt text for a one-shot request

options:
  -h, --help                 show this help message and exit
  --version                  show program's version number and exit
  --provider PROVIDER        LLM provider: openai or ollama (default: openai)
  --model MODEL_ID           model identifier (default: gpt-6-luna for
                             openai, the most recently installed local model
                             for ollama)
  --reasoning-effort EFFORT  reasoning effort: none, minimal, low, medium,
                             high, xhigh, max, model-default. model-default
                             sends no effort, so the model uses its own
                             default (gpt-6-luna: medium). Without this flag:
                             none where supported (Ollama: thinking off)
  --max-output-tokens N      maximum number of tokens to generate (default:
                             the model's)
  --answer-style STYLE       brief, normal or detailed answers (default:
                             normal)
  --usage                    print token usage to stderr after the response

Development version ` + Version + `: non-streaming one-shot prompts only.
OpenAI reads its API key from the OPENAI_API_KEY environment variable;
Ollama uses OLLAMA_HOST (default http://127.0.0.1:11434) and needs no key.
Not implemented yet: interactive chat, streaming, history, saved settings
and the fullscreen UI.

examples:
  neurobrau "What is a neural network?"
  neurobrau --provider ollama --model qwen3.5:4b "Hello"
  neurobrau --reasoning-effort high --answer-style brief --usage "Hello"
`

// Run parses args (without the program name), writes output to stdout and
// stderr, and returns the process exit code. Canceling ctx (e.g. on Ctrl+C)
// aborts a running request.
func Run(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	opts, err := parseArgs(args)
	if errors.Is(err, flag.ErrHelp) {
		fmt.Fprint(stdout, helpText)
		return ExitOK
	}
	if err != nil {
		return usageError(stderr, err.Error())
	}
	if opts.version {
		if len(opts.positional) > 0 {
			return usageError(stderr, fmt.Sprintf("unexpected argument: %q", opts.positional[0]))
		}
		fmt.Fprintf(stdout, "NeuroBräu Go CLI %s\n", Version)
		return ExitOK
	}
	if len(args) == 0 {
		// Temporary until interactive chat is implemented.
		fmt.Fprint(stdout, helpText)
		return ExitOK
	}
	if err := opts.validate(); err != nil {
		return usageError(stderr, err.Error())
	}
	return runPrompt(ctx, opts, stdout, stderr)
}

type options struct {
	version    bool
	usage      bool
	provider   string
	model      string
	reasoning  optionalString
	maxTokens  optionalString
	style      optionalString
	positional []string

	maxOutputTokens int // parsed from maxTokens by validate
}

// optionalString is a string flag that remembers whether it was given.
type optionalString struct {
	value string
	set   bool
}

func (o *optionalString) String() string { return o.value }

func (o *optionalString) Set(v string) error {
	o.value, o.set = v, true
	return nil
}

// parseArgs parses flags and positional arguments in any order, like the
// Python CLI; everything after "--" is positional.
func parseArgs(args []string) (options, error) {
	var opts options
	fs := flag.NewFlagSet(ProgName, flag.ContinueOnError)
	// Errors and usage are reported by Run itself, not by the flag package.
	fs.SetOutput(io.Discard)
	fs.Usage = func() {}
	fs.BoolVar(&opts.version, "version", false, "")
	fs.BoolVar(&opts.usage, "usage", false, "")
	fs.StringVar(&opts.provider, "provider", "", "")
	fs.StringVar(&opts.model, "model", "", "")
	fs.Var(&opts.reasoning, "reasoning-effort", "")
	fs.Var(&opts.maxTokens, "max-output-tokens", "")
	fs.Var(&opts.style, "answer-style", "")

	rest := args
	for len(rest) > 0 {
		if err := fs.Parse(rest); err != nil {
			return opts, err
		}
		remaining := fs.Args()
		consumed := len(rest) - len(remaining)
		if consumed > 0 && rest[consumed-1] == "--" {
			opts.positional = append(opts.positional, remaining...)
			break
		}
		if len(remaining) == 0 {
			break
		}
		opts.positional = append(opts.positional, remaining[0])
		rest = remaining[1:]
	}
	return opts, nil
}

// validate checks a prompt invocation and applies the default provider.
// Flag values are checked before the prompt, as argparse does.
func (o *options) validate() error {
	if o.provider == "" {
		o.provider = providers[0]
	}
	if err := checkChoice("--provider", o.provider, providers); err != nil {
		return err
	}
	if o.reasoning.set {
		if err := checkChoice("--reasoning-effort", o.reasoning.value, append(slices.Clone(llm.ReasoningEfforts), llm.ModelDefault)); err != nil {
			return err
		}
	}
	if o.maxTokens.set {
		n, err := strconv.Atoi(o.maxTokens.value)
		if err != nil {
			return fmt.Errorf("argument --max-output-tokens: invalid integer: %q", o.maxTokens.value)
		}
		if n <= 0 {
			return fmt.Errorf("argument --max-output-tokens: must be greater than 0, got %d", n)
		}
		o.maxOutputTokens = n
	}
	if o.style.set {
		if err := checkChoice("--answer-style", o.style.value, llm.AnswerStyles); err != nil {
			return err
		}
	}
	if len(o.positional) > 1 {
		return fmt.Errorf("unexpected argument: %q", o.positional[1])
	}
	if len(o.positional) == 0 {
		return errors.New("a prompt is required; interactive chat is not implemented yet")
	}
	if strings.TrimSpace(o.positional[0]) == "" {
		return errors.New("prompt must not be empty")
	}
	return nil
}

func checkChoice(flagName, value string, choices []string) error {
	if slices.Contains(choices, value) {
		return nil
	}
	return fmt.Errorf("argument %s: invalid choice: %q (choose from %s)", flagName, value, strings.Join(choices, ", "))
}

// newProvider creates the named provider from the environment. Nothing is
// requested yet; a missing OpenAI key is reported by the first request.
func newProvider(name string) (llm.Provider, error) {
	if name == "openai" {
		return openai.New(openai.Config{APIKey: os.Getenv(openai.APIKeyEnvVar)}), nil
	}
	baseURL, err := ollama.ResolveBaseURL(os.Getenv(ollama.HostEnvVar))
	if err != nil {
		return nil, err
	}
	return ollama.New(ollama.Config{BaseURL: baseURL}), nil
}

// settingsError is a command-line value the model does not support
// (exit code 2).
type settingsError struct{ err error }

func (e settingsError) Error() string { return e.err.Error() }

// buildRequest resolves the generation settings for the model: CLI flags,
// else built-in defaults. Values the model is known not to support are a
// settingsError; nothing is sent to the chat endpoint before this passes.
func buildRequest(ctx context.Context, provider llm.Provider, model string, opts options) (llm.Request, error) {
	req := llm.Request{
		Model:           model,
		Messages:        []llm.Message{{Role: llm.RoleUser, Content: opts.positional[0]}},
		Instructions:    llm.StyleInstructions(opts.style.value),
		MaxOutputTokens: opts.maxOutputTokens,
	}
	if opts.reasoning.set {
		support, err := provider.ReasoningSupport(ctx, model)
		if err != nil {
			return llm.Request{}, err
		}
		if req.Reasoning, err = llm.ResolveReasoning(model, support, opts.reasoning.value); err != nil {
			return llm.Request{}, settingsError{err}
		}
	}
	if req.MaxOutputTokens > 0 {
		// An unknown limit (or a lookup error, e.g. a missing model, which the
		// chat request reports) does not restrict the value.
		limit, _ := provider.MaxOutputLimit(ctx, model)
		if err := llm.CheckMaxOutputTokens(model, req.MaxOutputTokens, limit); err != nil {
			return llm.Request{}, settingsError{err}
		}
	}
	return req, nil
}

func runPrompt(ctx context.Context, opts options, stdout, stderr io.Writer) int {
	provider, err := newProvider(opts.provider)
	if err != nil {
		return usageError(stderr, err.Error())
	}

	model := opts.model
	if model == "" {
		if model, err = provider.DefaultModel(ctx); err != nil {
			return reportError(ctx, stderr, err)
		}
	}
	req, err := buildRequest(ctx, provider, model, opts)
	if err != nil {
		return reportError(ctx, stderr, err)
	}
	resp, err := provider.Chat(ctx, req)
	if err != nil {
		return reportError(ctx, stderr, err)
	}
	// Like the Python reference, a non-streaming answer is printed as is,
	// even if it stopped early (resp.IncompleteReason).
	fmt.Fprintln(stdout, resp.Text)
	if opts.usage {
		fmt.Fprintln(stderr, formatUsage(resp.Usage))
	}
	return ExitOK
}

// formatUsage shows the reported input/output/total token counts only;
// nothing is estimated or derived.
func formatUsage(u *llm.Usage) string {
	if u == nil {
		return "Token usage: not reported"
	}
	var parts []string
	for _, f := range []struct {
		name  string
		count *int
	}{{"input", u.InputTokens}, {"output", u.OutputTokens}, {"total", u.TotalTokens}} {
		if f.count != nil {
			parts = append(parts, fmt.Sprintf("%s=%d", f.name, *f.count))
		}
	}
	if len(parts) == 0 {
		return "Token usage: not reported"
	}
	return "Token usage: " + strings.Join(parts, ", ")
}

func reportError(ctx context.Context, stderr io.Writer, err error) int {
	if errors.Is(err, context.Canceled) && ctx.Err() != nil {
		fmt.Fprintf(stderr, "\n%s: interrupted\n", ProgName)
		return ExitInterrupted
	}
	fmt.Fprintf(stderr, "%s: error: %s\n", ProgName, err)
	var settingsErr settingsError
	if errors.As(err, &settingsErr) {
		return ExitUsage
	}
	return llm.ExitCode(err)
}

func usageError(stderr io.Writer, msg string) int {
	fmt.Fprintf(stderr, "%s: error: %s\n", ProgName, msg)
	return ExitUsage
}
