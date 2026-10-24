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
	"sync"

	"github.com/neurobrau/llm-cli-go/internal/chat"
	"github.com/neurobrau/llm-cli-go/internal/llm"
	"github.com/neurobrau/llm-cli-go/internal/ollama"
	"github.com/neurobrau/llm-cli-go/internal/openai"
)

const (
	// ProgName is the executable name used in usage and error messages.
	ProgName = "neurobrau"
	// Version is the Go edition's own version. It does not imply feature
	// parity with the Python reference implementation (0.3.0).
	Version = "0.4.0-dev"
)

// Exit codes (docs/tui-spec.md, section 16). Provider failures use the
// codes of llm.ErrorKind.
const (
	ExitOK          = 0
	ExitUsage       = 2
	ExitIncomplete  = 8 // a streamed one-shot answer stopped early
	ExitInterrupted = 130
	ExitBrokenPipe  = 141 // stdout closed by the reader (128 + SIGPIPE)
)

// Providers, in help order; the first is the default.
var providers = []string{"openai", "ollama"}

const helpText = `usage: neurobrau [-h] [--version] [--provider PROVIDER] [--model MODEL_ID]
                 [--reasoning-effort EFFORT] [--max-output-tokens N]
                 [--answer-style STYLE] [--usage] [--stream] [--plain]
                 [--no-history] [--transcript-on-exit]
                 [prompt]

NeuroBräu Go CLI — educational multi-provider LLM command-line client.

positional arguments:
  prompt                     prompt text for a one-shot request; omit to
                             start an interactive chat

options:
  -h, --help                 show this help message and exit
  --version                  show program's version number and exit
  --provider PROVIDER        LLM provider: openai or ollama (default: openai;
                             an interactive chat without --provider/--model
                             continues with the last conversation's)
  --model MODEL_ID           model identifier (default: gpt-6-luna for
                             openai, the most recently installed local model
                             for ollama)
  --reasoning-effort EFFORT  reasoning effort: none, minimal, low, medium,
                             high, xhigh, max, model-default. model-default
                             sends no effort, so the model uses its own
                             default (gpt-6-luna: medium). Without this flag:
                             in a chat the saved choice for the model, else
                             none where supported (Ollama: thinking off)
  --max-output-tokens N      maximum number of tokens to generate (default:
                             in a chat the saved choice, else the model's)
  --answer-style STYLE       brief, normal or detailed answers (default: in a
                             chat the saved choice, else normal)
  --usage                    print token usage to stderr after each response
  --stream                   show answers while they are generated (one-shot,
                             plain and fullscreen chat)
  --plain                    use the line-oriented chat instead of the
                             fullscreen UI
  --no-history               do not load or save conversation history for
                             this run
  --transcript-on-exit       fullscreen UI: print the conversation after the
                             terminal is restored

Development version ` + Version + `: one-shot prompts and an interactive
chat that saves conversations, both optionally streamed. Without a prompt,
the fullscreen UI starts when stdin and stdout are terminals and TERM is
set (not dumb); otherwise, or with --plain, the line-oriented chat starts.
OpenAI reads its API key from the OPENAI_API_KEY environment variable;
Ollama uses OLLAMA_HOST (default http://127.0.0.1:11434) and needs no key.
In the fullscreen UI, F2/F3 switch model and provider and F4-F6 change
reasoning, max output tokens and answer style; F4-F6 choices are saved in
settings.json (also with --no-history). /usage shows the conversation's
token usage and its ESTIMATED API cost (not a billed amount).

examples:
  neurobrau
  neurobrau --stream --transcript-on-exit
  neurobrau --plain --stream --usage
  neurobrau "What is a neural network?"
  neurobrau --provider ollama --model qwen3.5:4b "Hello"
  neurobrau --reasoning-effort high --answer-style brief --usage "Hello"
  neurobrau --stream --usage "Explain backpropagation."
`

// Streams are the process's standard streams and its Ctrl+C signals.
type Streams struct {
	Stdin          io.Reader // read by the interactive chat only
	Stdout, Stderr io.Writer
	// Interrupts delivers Ctrl+C (os.Interrupt); nil: never. A one-shot
	// request is canceled by the first one; the chat cancels its running
	// request or exits when idle.
	Interrupts <-chan os.Signal
	// WatchTerminations, if set, starts delivering SIGTERM and SIGHUP to c
	// and returns a function that stops it. Only the fullscreen UI watches
	// them, to restore the terminal before it exits; elsewhere they keep
	// their default effect.
	WatchTerminations func(c chan<- os.Signal) (stop func())
}

// Run parses args (without the program name), runs a one-shot prompt or
// the interactive chat, and returns the process exit code. Canceling ctx
// aborts a running request (exit code 130).
func Run(ctx context.Context, args []string, s Streams) int {
	stdout, stderr := s.Stdout, s.Stderr
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
	if err := opts.validate(); err != nil {
		return usageError(stderr, err.Error())
	}
	if len(opts.positional) == 0 {
		return runInteractive(ctx, opts, s)
	}
	ctx, stop := cancelOnInterrupt(ctx, s.Interrupts)
	defer stop()
	return runPrompt(ctx, opts, stdout, stderr)
}

// cancelOnInterrupt returns a context that the first Ctrl+C cancels, and
// an idempotent function that stops watching, cancels the context and
// reports whether a Ctrl+C was received.
func cancelOnInterrupt(parent context.Context, interrupts <-chan os.Signal) (context.Context, func() bool) {
	ctx, cancel := context.WithCancel(parent)
	if interrupts == nil {
		return ctx, func() bool { cancel(); return false }
	}
	stopped := make(chan struct{})
	finished := make(chan struct{})
	interrupted := false
	go func() {
		defer close(finished)
		select {
		case <-interrupts:
			interrupted = true
			cancel()
		case <-stopped:
		case <-ctx.Done():
		}
	}()
	return ctx, sync.OnceValue(func() bool {
		close(stopped)
		<-finished
		cancel()
		return interrupted
	})
}

type options struct {
	version    bool
	usage      bool
	plain      bool
	stream     bool
	noHistory  bool
	transcript bool
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
	fs.BoolVar(&opts.plain, "plain", false, "")
	fs.BoolVar(&opts.stream, "stream", false, "")
	fs.BoolVar(&opts.noHistory, "no-history", false, "")
	fs.BoolVar(&opts.transcript, "transcript-on-exit", false, "")
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

// validate checks the flag values, then the prompt if one was given, as
// argparse does. An empty provider stays empty (not given).
func (o *options) validate() error {
	if o.provider != "" {
		if err := checkChoice("--provider", o.provider, providers); err != nil {
			return err
		}
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
	if len(o.positional) == 1 && strings.TrimSpace(o.positional[0]) == "" {
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
func newProvider(name string) (llm.StreamingProvider, error) {
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
	if opts.provider == "" {
		opts.provider = providers[0]
	}
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
	if opts.stream {
		return streamPrompt(ctx, provider, req, opts.usage, stdout, stderr)
	}
	resp, err := provider.Chat(ctx, req)
	if err != nil {
		return reportError(ctx, stderr, err)
	}
	// Like the Python reference, a non-streaming answer is printed as is,
	// even if it stopped early (resp.IncompleteReason).
	fmt.Fprintln(stdout, resp.Text)
	if opts.usage {
		fmt.Fprintln(stderr, chat.FormatUsage(resp.Usage))
	}
	return ExitOK
}

func reportError(ctx context.Context, stderr io.Writer, err error) int {
	if errors.Is(err, context.Canceled) && ctx.Err() != nil {
		fmt.Fprintf(stderr, "\n%s: interrupted\n", ProgName)
		return ExitInterrupted
	}
	fmt.Fprintf(stderr, "%s: error: %s\n", ProgName, err)
	var settingsErr settingsError
	var chatSettingsErr *chat.SettingsError
	if errors.As(err, &settingsErr) || errors.As(err, &chatSettingsErr) {
		return ExitUsage
	}
	return llm.ExitCode(err)
}

func usageError(stderr io.Writer, msg string) int {
	fmt.Fprintf(stderr, "%s: error: %s\n", ProgName, msg)
	return ExitUsage
}
