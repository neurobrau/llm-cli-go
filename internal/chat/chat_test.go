package chat

import (
	"bytes"
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/neurobrau/llm-cli-go/internal/llm"
	"github.com/neurobrau/llm-cli-go/internal/session"
	"github.com/neurobrau/llm-cli-go/internal/storage"
)

// fakeProvider answers Chat from a script and records each request.
type fakeProvider struct {
	mu       sync.Mutex
	requests []llm.Request
	replies  []func(ctx context.Context) (llm.Response, error)
	support  llm.ReasoningSupport
	limit    int
	supErr   error
}

func (f *fakeProvider) Name() string                                        { return "openai" }
func (f *fakeProvider) ListModels(context.Context) ([]llm.Model, error)     { return nil, nil }
func (f *fakeProvider) DefaultModel(context.Context) (string, error)        { return "gpt-6-luna", nil }
func (f *fakeProvider) MaxOutputLimit(context.Context, string) (int, error) { return f.limit, nil }
func (f *fakeProvider) ReasoningSupport(context.Context, string) (llm.ReasoningSupport, error) {
	return f.support, f.supErr
}

func (f *fakeProvider) Chat(ctx context.Context, req llm.Request) (llm.Response, error) {
	f.mu.Lock()
	f.requests = append(f.requests, req)
	if len(f.replies) == 0 {
		f.mu.Unlock()
		return llm.Response{}, errors.New("unexpected request")
	}
	next := f.replies[0]
	f.replies = f.replies[1:]
	f.mu.Unlock()
	return next(ctx)
}

func (f *fakeProvider) sent() [][]llm.Message {
	f.mu.Lock()
	defer f.mu.Unlock()
	var all [][]llm.Message
	for _, r := range f.requests {
		all = append(all, r.Messages)
	}
	return all
}

func intp(n int) *int { return &n }

func text(s string, input, output int) func(context.Context) (llm.Response, error) {
	return func(context.Context) (llm.Response, error) {
		return llm.Response{Text: s, Usage: &llm.Usage{InputTokens: intp(input), OutputTokens: intp(output), TotalTokens: intp(input + output)}}, nil
	}
}

func failWith(err error) func(context.Context) (llm.Response, error) {
	return func(context.Context) (llm.Response, error) { return llm.Response{}, err }
}

func msgs(contents ...string) []llm.Message {
	var m []llm.Message
	for i, c := range contents {
		role := llm.RoleUser
		if i%2 == 1 {
			role = llm.RoleAssistant
		}
		m = append(m, llm.Message{Role: role, Content: c})
	}
	return m
}

type harness struct {
	provider   *fakeProvider
	session    *session.Session
	store      *storage.HistoryStore // nil: --no-history
	interrupts chan os.Signal
	stdin      io.Reader
	stdout     *syncBuffer
	stderr     *syncBuffer
	showUsage  bool
	warnings   []string
	stream     *fakeStream // nil: non-streaming
}

func newHarness(t *testing.T, history bool, replies ...func(context.Context) (llm.Response, error)) *harness {
	t.Helper()
	h := &harness{provider: &fakeProvider{replies: replies}, interrupts: make(chan os.Signal, 1),
		stdout: newSyncBuffer(), stderr: newSyncBuffer()}
	if history {
		h.store = storage.NewHistoryStore(filepath.Join(t.TempDir(), "state", "neurobrau", "conversations"))
	}
	h.session, _ = session.Open(session.Options{Store: h.store, Provider: "openai", Model: "gpt-6-luna"})
	return h
}

func (h *harness) run(ctx context.Context, input string) int {
	if h.stdin == nil {
		h.stdin = strings.NewReader(input)
	}
	engine := &Engine{Provider: h.provider, Model: "gpt-6-luna", Settings: Settings{AnswerStyle: "brief", MaxOutputTokens: 50}}
	cfg := Config{
		Title: "NeuroBräu Go CLI test", Session: h.session, Reply: engine.Reply, Warnings: h.warnings,
		ShowUsage: h.showUsage, Stdin: h.stdin, Stdout: h.stdout, Stderr: h.stderr, Interrupts: h.interrupts,
	}
	if h.stream != nil {
		cfg.Stream = h.stream.Stream
	}
	return Run(ctx, cfg)
}

const banner = "NeuroBräu Go CLI test\nProvider: openai | Model: gpt-6-luna\nType /help for commands, /exit to quit.\n\n"

func TestBannerEmptyInputAndExit(t *testing.T) {
	h := newHarness(t, false)
	h.warnings = []string{"Ignoring invalid /x/active.json."}
	code := h.run(context.Background(), "\n   \n/exit\nnever read\n")
	if code != ExitOK {
		t.Errorf("exit code %d", code)
	}
	if want := banner + "YOU > YOU > YOU > Goodbye.\n"; h.stdout.String() != want {
		t.Errorf("stdout %q, want %q", h.stdout.String(), want)
	}
	if h.stderr.String() != "warning: Ignoring invalid /x/active.json.\n" || len(h.provider.sent()) != 0 {
		t.Errorf("stderr %q, %d requests", h.stderr.String(), len(h.provider.sent()))
	}
}

func TestEOFExits(t *testing.T) {
	for _, input := range []string{"", "Hi"} { // EOF, and a last line without a newline
		h := newHarness(t, false, text("Hello", 1, 1))
		if code := h.run(context.Background(), input); code != ExitOK || !strings.HasSuffix(h.stdout.String(), "YOU > \nGoodbye.\n") {
			t.Errorf("%q: code %d stdout %q", input, code, h.stdout.String())
		}
	}
}

func TestTwoTurns(t *testing.T) {
	h := newHarness(t, true, text("Hello!\nSecond line", 10, 3), text("Fine.", 25, 2))
	h.showUsage = true
	code := h.run(context.Background(), "Hi\n  How are you?  \n")
	if code != ExitOK {
		t.Fatalf("exit %d", code)
	}
	want := banner + "YOU > AI > Hello!\nSecond line\nYOU > AI > Fine.\nYOU > \nGoodbye.\n"
	if h.stdout.String() != want {
		t.Errorf("stdout %q, want %q", h.stdout.String(), want)
	}
	if h.stderr.String() != "Token usage: input=10, output=3, total=13\nToken usage: input=25, output=2, total=27\n" {
		t.Errorf("stderr %q", h.stderr.String())
	}
	sent := h.provider.sent()
	if !reflect.DeepEqual(sent, [][]llm.Message{msgs("Hi"), msgs("Hi", "Hello!\nSecond line", "How are you?")}) {
		t.Errorf("sent %q", sent)
	}
	if r := h.provider.requests[1]; r.Model != "gpt-6-luna" || r.MaxOutputTokens != 50 || r.Instructions != llm.StyleInstructions("brief") {
		t.Errorf("request settings %+v", r)
	}
	c, err := h.store.Load(h.store.PathFor(h.session.Conversation().ID))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(c.Messages, msgs("Hi", "Hello!\nSecond line", "How are you?", "Fine.")) || len(c.Requests) != 2 {
		t.Errorf("saved %+v", c)
	}
}

func TestProviderErrorKeepsHistory(t *testing.T) {
	h := newHarness(t, true, text("A1", 1, 1),
		failWith(&llm.Error{Kind: llm.KindRateLimit, Message: "OpenAI rate limit or quota exceeded."}),
		text("A3", 1, 1))
	code := h.run(context.Background(), "Q1\nQ2\nQ3\n")
	if code != ExitOK || h.stderr.String() != "error: OpenAI rate limit or quota exceeded.\n" {
		t.Errorf("code %d stderr %q", code, h.stderr.String())
	}
	if sent := h.provider.sent(); !reflect.DeepEqual(sent[2], msgs("Q1", "A1", "Q3")) {
		t.Errorf("third request %q", sent[2])
	}
	c, _ := h.store.Load(h.store.PathFor(h.session.Conversation().ID))
	if !reflect.DeepEqual(c.Messages, msgs("Q1", "A1", "Q3", "A3")) || len(c.Requests) != 2 {
		t.Errorf("saved %+v", c)
	}
}

func TestIncompleteAnswer(t *testing.T) {
	cut := func(context.Context) (llm.Response, error) {
		return llm.Response{Text: "Partial", IncompleteReason: "the max output token limit was reached",
			Usage: &llm.Usage{InputTokens: intp(7), OutputTokens: intp(50)}}, nil
	}
	h := newHarness(t, true, text("A1", 5, 1), cut, text("A3", 9, 1))
	h.run(context.Background(), "Q1\nQ2\nQ3\n/usage\n")
	if !strings.Contains(h.stdout.String(), "YOU > AI > Partial\nYOU > ") {
		t.Errorf("stdout %q", h.stdout.String())
	}
	if h.stderr.String() != "warning: Response incomplete: the max output token limit was reached. The partial response was not saved.\n" {
		t.Errorf("stderr %q", h.stderr.String())
	}
	if sent := h.provider.sent(); !reflect.DeepEqual(sent[2], msgs("Q1", "A1", "Q3")) {
		t.Errorf("partial answer sent as context: %q", sent[2])
	}
	c, _ := h.store.Load(h.store.PathFor(h.session.Conversation().ID))
	if len(c.Messages) != 4 || len(c.Requests) != 3 || c.Requests[1].Status != storage.StatusIncomplete {
		t.Errorf("saved %+v", c)
	}
	for _, want := range []string{"Conversation usage (3 request(s), as reported by the API):\nInput tokens: 21\nOutput tokens: 52\nTotal tokens: 16\n",
		"Latest request input: 9 tokens", "Includes 1 incomplete answer(s): billed, but not saved as messages."} {
		if !strings.Contains(h.stdout.String(), want) {
			t.Errorf("usage report lacks %q:\n%s", want, h.stdout.String())
		}
	}
}

func TestCommands(t *testing.T) {
	h := newHarness(t, false, text("About /usr/bin", 1, 1))
	h.run(context.Background(), "/HELP\n\\help\n/nope\n\\Nope\n/usr/bin is what?\n/usage\n/reset\n/usage\n\\EXIT\n")
	out := h.stdout.String()
	if strings.Count(out, HelpText) != 2 {
		t.Errorf("help shown %d times", strings.Count(out, HelpText))
	}
	for _, want := range []string{
		"Unknown command: /nope. Type /help for commands.\nYOU > Unknown command: /nope. Type /help for commands.\n",
		"YOU > AI > About /usr/bin\n",
		"Conversation usage (1 request(s), as reported by the API):\nInput tokens: 1\nOutput tokens: 1\nTotal tokens: 2\n" +
			"Latest request input: 1 tokens (the whole history is sent, and billed, with every request)\n" +
			"Estimated cost: approximately $0.000001 (ESTIMATED, not a billed amount; pricing 2026-10-22; Standard tier assumed " +
			"(tier not reported); cached tokens not reported (all input priced as uncached); cache writes not reported " +
			"(priced as ordinary input))\n",
		"YOU > Conversation history cleared.\nYOU > Token usage: no API requests in this conversation yet.\nYOU > Goodbye.\n",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("stdout lacks %q:\n%s", want, out)
		}
	}
	if sent := h.provider.sent(); len(sent) != 1 || sent[0][0].Content != "/usr/bin is what?" {
		t.Errorf("sent %q", sent)
	}
}

func TestResetWithHistory(t *testing.T) {
	h := newHarness(t, true, text("A", 1, 1))
	h.run(context.Background(), "/reset\nQ\n/reset\n")
	out := h.stdout.String()
	if strings.Count(out, "Started a new conversation. The previous one stays saved.\n") != 2 {
		t.Errorf("stdout %q", out)
	}
	if n, _ := h.store.CountConversations(); n != 2 || len(h.session.Messages()) != 0 {
		t.Errorf("%d files, %d messages", n, len(h.session.Messages()))
	}
}

func TestCleanup(t *testing.T) {
	prompt := func(dir string, n int) string {
		return "Delete " + string(rune('0'+n)) + " saved conversation(s) in " + dir + "? This cannot be undone. Type 'yes' to confirm: "
	}
	tests := []struct {
		name, answer string
		deleted      bool
	}{
		{"no", "no\n", false},
		{"exit as answer", "/exit\n", false},
		{"yes with spaces", "  YES \n", true},
		{"EOF", "", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := newHarness(t, true, text("A", 1, 1))
			code := h.run(context.Background(), "Q\n/cleanup\n"+tt.answer)
			if code != ExitOK {
				t.Errorf("exit %d", code)
			}
			out := h.stdout.String()
			if !strings.Contains(out, "YOU > "+prompt(h.store.Dir(), 1)) {
				t.Errorf("no confirmation prompt:\n%s", out)
			}
			n, _ := h.store.CountConversations()
			if tt.deleted != (n == 0) {
				t.Errorf("%d files left", n)
			}
			if tt.deleted {
				if !strings.Contains(out, "Deleted 1 saved conversation(s). Started a new conversation.\n") || len(h.session.Messages()) != 0 {
					t.Errorf("stdout %q", out)
				}
				if _, err := os.Stat(h.store.ActivePath()); err != nil {
					t.Error("active.json deleted")
				}
			} else if !strings.Contains(out, "Cleanup cancelled. Nothing was deleted.\n") || len(h.session.Messages()) != 2 {
				t.Errorf("stdout %q", out)
			}
			if len(h.provider.sent()) != 1 {
				t.Error("the confirmation was sent to the model")
			}
			if !tt.deleted && !strings.HasSuffix(out, "Cleanup cancelled. Nothing was deleted.\nYOU > \nGoodbye.\n") {
				t.Errorf("the answer was used as more than a confirmation:\n%s", out)
			}
		})
	}

	t.Run("no history", func(t *testing.T) {
		h := newHarness(t, false)
		h.run(context.Background(), "/cleanup\nyes\n")
		if !strings.Contains(h.stdout.String(), "Conversation history is disabled for this session (--no-history).\n") || len(h.provider.sent()) != 1 {
			// "yes" is then an ordinary message.
			t.Errorf("stdout %q, %d requests", h.stdout.String(), len(h.provider.sent()))
		}
	})
	t.Run("no files", func(t *testing.T) {
		h := newHarness(t, true)
		h.run(context.Background(), "\\cleanup\n")
		if !strings.Contains(h.stdout.String(), "YOU > No saved conversations.\n") {
			t.Errorf("stdout %q", h.stdout.String())
		}
	})
	t.Run("Ctrl+C at the confirmation", func(t *testing.T) {
		h := newHarness(t, true, text("A", 1, 1))
		in, w := io.Pipe()
		h.stdin = in
		done := make(chan int)
		go func() { done <- h.run(context.Background(), "") }()
		io.WriteString(w, "Q\n/cleanup\n")
		waitFor(t, h.stdout, "Type 'yes' to confirm: ")
		h.interrupts <- os.Interrupt
		waitFor(t, h.stdout, "Cleanup cancelled. Nothing was deleted.\n")
		w.Close()
		if code := <-done; code != ExitOK {
			t.Errorf("exit %d", code)
		}
		if n, _ := h.store.CountConversations(); n != 1 {
			t.Errorf("%d files", n)
		}
	})
}

// syncBuffer is an output stream the test can read while the chat runs;
// changed is signaled after each write.
type syncBuffer struct {
	mu      sync.Mutex
	buf     bytes.Buffer
	changed chan struct{}
}

func newSyncBuffer() *syncBuffer { return &syncBuffer{changed: make(chan struct{}, 1)} }

func (b *syncBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	n, err := b.buf.Write(p)
	b.mu.Unlock()
	select {
	case b.changed <- struct{}{}:
	default:
	}
	return n, err
}

func (b *syncBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.String()
}

// waitFor waits until out contains want.
func waitFor(t *testing.T, out *syncBuffer, want string) {
	t.Helper()
	timeout := time.After(10 * time.Second)
	for !strings.Contains(out.String(), want) {
		select {
		case <-out.changed:
		case <-timeout:
			t.Fatalf("timed out waiting for %q in %q", want, out.String())
		}
	}
}

func TestInterruptWhileIdleExits130(t *testing.T) {
	h := newHarness(t, true)
	in, _ := io.Pipe() // nothing is ever typed
	h.stdin = in
	h.interrupts <- os.Interrupt
	if code := h.run(context.Background(), ""); code != ExitInterrupted || !strings.HasSuffix(h.stdout.String(), "YOU > \nGoodbye.\n") {
		t.Errorf("code %d stdout %q", code, h.stdout.String())
	}
}

func TestInterruptCancelsRequest(t *testing.T) {
	var h *harness
	h = newHarness(t, true,
		func(ctx context.Context) (llm.Response, error) {
			h.interrupts <- os.Interrupt // Ctrl+C while waiting for the answer
			<-ctx.Done()
			return llm.Response{}, ctx.Err()
		},
		text("A2", 1, 1))
	code := h.run(context.Background(), "Q1\nQ2\n")
	if code != ExitOK {
		t.Errorf("exit %d", code)
	}
	if h.stderr.String() != "\nRequest cancelled.\n" {
		t.Errorf("stderr %q", h.stderr.String())
	}
	if sent := h.provider.sent(); !reflect.DeepEqual(sent[1], msgs("Q2")) {
		t.Errorf("second request %q", sent[1])
	}
	c, _ := h.store.Load(h.store.PathFor(h.session.Conversation().ID))
	if !reflect.DeepEqual(c.Messages, msgs("Q2", "A2")) {
		t.Errorf("saved %q", c.Messages)
	}
}

func TestCanceledContextEndsChat(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	h := newHarness(t, true, func(rctx context.Context) (llm.Response, error) {
		cancel()
		<-rctx.Done()
		return llm.Response{}, rctx.Err()
	})
	if code := h.run(ctx, "Q\nnever\n"); code != ExitInterrupted || len(h.session.Messages()) != 0 {
		t.Errorf("code %d, %d messages", code, len(h.session.Messages()))
	}
	if n, _ := h.store.CountConversations(); n != 0 {
		t.Error("canceled request saved")
	}
}

func TestSaveFailureIsAWarning(t *testing.T) {
	h := newHarness(t, true, text("A1", 1, 1), text("A2", 1, 1))
	if err := os.MkdirAll(filepath.Dir(h.store.Dir()), 0o700); err != nil {
		t.Fatal(err)
	}
	os.WriteFile(h.store.Dir(), []byte("a file"), 0o600) // conversations path is a file
	code := h.run(context.Background(), "Q1\nQ2\n")
	if code != ExitOK || strings.Count(h.stderr.String(), "warning: Could not save conversation history at ") != 2 {
		t.Errorf("code %d stderr %q", code, h.stderr.String())
	}
	if !strings.Contains(h.stdout.String(), "AI > A2") || !reflect.DeepEqual(h.provider.sent()[1], msgs("Q1", "A1", "Q2")) {
		t.Errorf("chat did not go on with the exchange in memory: %q", h.provider.sent())
	}
}

func TestRestoredNotice(t *testing.T) {
	store := storage.NewHistoryStore(filepath.Join(t.TempDir(), "conversations"))
	updated := time.Date(2026, 10, 22, 5, 54, 29, 0, time.UTC)
	store.Save(storage.Conversation{ID: "7c1e4b2a-9f3d-4e8a-b6c5-1a2b3c4d5e6f", CreatedAt: updated, UpdatedAt: updated,
		Provider: "openai", Model: "gpt-6-luna", Messages: msgs("Q", "A")})
	h := newHarness(t, false, text("A2", 1, 1))
	h.session, _ = session.Open(session.Options{Store: store})
	h.run(context.Background(), "Q2\n")
	want := banner + "Restored conversation from " + updated.Local().Format("2006-01-02 15:04") +
		" (2 messages). Type /reset to start a new one.\n\nYOU > AI > A2\n"
	if !strings.HasPrefix(h.stdout.String(), want) {
		t.Errorf("stdout %q, want prefix %q", h.stdout.String(), want)
	}
	if sent := h.provider.sent(); !reflect.DeepEqual(sent[0], msgs("Q", "A", "Q2")) {
		t.Errorf("sent %q", sent[0])
	}
}

func TestParseCommand(t *testing.T) {
	for in, want := range map[string]string{"/help": "help", `\RESET`: "reset", "/": "", "/usr/bin": "usr/bin"} {
		if got, ok := ParseCommand(in); !ok || got != want {
			t.Errorf("%q: %q %v", in, got, ok)
		}
	}
	for _, in := range []string{"help", "/usr/bin is what?", "/help\tme", "x/help"} {
		if _, ok := ParseCommand(in); ok {
			t.Errorf("%q parsed as a command", in)
		}
	}
}
