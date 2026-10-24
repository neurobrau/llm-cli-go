package chat

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"iter"
	"os"
	"strings"
	"unicode"

	"github.com/neurobrau/llm-cli-go/internal/llm"
	"github.com/neurobrau/llm-cli-go/internal/session"
)

// Prompts and labels of the plain chat.
const (
	UserPrompt = "YOU > "
	AIPrefix   = "AI > "
)

// Exit codes of the chat (docs/tui-spec.md, sections 9 and 16).
const (
	ExitOK          = 0
	ExitError       = 1
	ExitInterrupted = 130
	ExitBrokenPipe  = 141 // stdout closed by the reader
)

// HelpText is the /help output.
const HelpText = `Commands:
  /help     show this help
  /reset    start a new conversation (the previous one stays saved)
  /cleanup  delete all saved conversations (asks for confirmation)
  /usage    token usage and ESTIMATED cost of this conversation
  /exit     exit the chat (the conversation stays saved)
Backslash aliases also work: \help, \reset, \cleanup, \usage, \exit.
Press Ctrl+D (EOF) or Ctrl+C at the prompt to exit.
Ctrl+C while waiting for an answer cancels the request.`

// Notices shared by the plain chat and the fullscreen UI.
const (
	CleanupWithoutHistory = "Conversation history is disabled for this session (--no-history)."
	NoSavedConversations  = "No saved conversations."
	CleanupCancelled      = "Cleanup cancelled. Nothing was deleted."
)

const cleanupConfirmation = "yes"

// RestoredNotice is the startup notice for a restored conversation; "" if
// none was restored.
func RestoredNotice(s *session.Session) string {
	if !s.Restored() {
		return ""
	}
	conv := s.Conversation()
	return fmt.Sprintf("Restored conversation from %s (%d messages). Type /reset to start a new one.",
		conv.UpdatedAt.Local().Format("2006-01-02 15:04"), len(conv.Messages))
}

// ResetNotice is shown after /reset.
func ResetNotice(persistent bool) string {
	if persistent {
		return "Started a new conversation. The previous one stays saved."
	}
	return "Conversation history cleared."
}

// CleanupPrompt asks to confirm deleting count saved conversations.
func CleanupPrompt(count int, dir string) string {
	return fmt.Sprintf("Delete %d saved conversation(s) in %s? This cannot be undone. Type '%s' to confirm: ",
		count, dir, cleanupConfirmation)
}

// CleanupConfirmed reports whether answer confirms the cleanup: only
// "yes", ignoring case and surrounding spaces.
func CleanupConfirmed(answer string) bool {
	return strings.ToLower(strings.TrimSpace(answer)) == cleanupConfirmation
}

// CleanupDone reports deleted saved conversations.
func CleanupDone(deleted int) string {
	return fmt.Sprintf("Deleted %d saved conversation(s). Started a new conversation.", deleted)
}

// Config is what Run needs. Session and Reply are required.
type Config struct {
	// Title is the first banner line, e.g. "NeuroBräu Go CLI 0.4.0-dev".
	Title   string
	Session *session.Session
	// Reply sends the context and returns the final answer (normally
	// Engine.Reply). Run makes at most one call at a time.
	Reply func(ctx context.Context, messages []llm.Message) (llm.Response, error)
	// Stream, if set, is used instead of Reply: answers are printed as
	// they arrive (--stream; normally Engine.Stream).
	Stream func(ctx context.Context, messages []llm.Message) iter.Seq[llm.StreamEvent]
	// Warnings are shown after the banner (storage, settings, notices).
	Warnings  []string
	ShowUsage bool // --usage: print token usage after each answer

	Stdin          io.Reader
	Stdout, Stderr io.Writer
	// Interrupts delivers Ctrl+C (os.Interrupt); nil: never.
	Interrupts <-chan os.Signal
}

type chat struct {
	Config
	lines <-chan line
}

// line is one input line, or the error (io.EOF) that ended the input.
type line struct {
	text string
	err  error
}

// Run runs the read-send-print loop until /exit or EOF (exit code 0) or
// Ctrl+C at the prompt (130). Canceling ctx also ends it with 130.
func Run(ctx context.Context, cfg Config) int {
	c := &chat{Config: cfg, lines: readLines(cfg.Stdin)}
	c.printBanner()
	for {
		text, status := c.readLine(ctx, UserPrompt)
		switch status {
		case inputEOF:
			fmt.Fprintln(c.Stdout, "\nGoodbye.")
			return ExitOK
		case inputInterrupted:
			fmt.Fprintln(c.Stdout, "\nGoodbye.")
			return ExitInterrupted
		}
		text = strings.TrimSpace(text)
		if text == "" {
			continue
		}
		if name, ok := ParseCommand(text); ok {
			if !c.command(ctx, name) {
				return ExitOK
			}
			continue
		}
		if c.Stream != nil {
			if code, quit := c.sendStream(ctx, text); quit {
				return code
			}
		} else if !c.send(ctx, text) {
			fmt.Fprintln(c.Stdout, "\nGoodbye.")
			return ExitInterrupted
		}
	}
}

// ParseCommand returns the lower-case name of a command such as `/help`
// or `\HELP`. Only a single word starting with `/` or `\` is a command, so
// "/usr/bin is what?" is a message.
func ParseCommand(text string) (string, bool) {
	if !strings.HasPrefix(text, "/") && !strings.HasPrefix(text, `\`) || strings.ContainsFunc(text, unicode.IsSpace) {
		return "", false
	}
	return strings.ToLower(text[1:]), true
}

func (c *chat) printBanner() {
	sel := c.Session.Selection()
	fmt.Fprintln(c.Stdout, c.Title)
	fmt.Fprintf(c.Stdout, "Provider: %s | Model: %s\n", sel.Provider, sel.Model)
	fmt.Fprintln(c.Stdout, "Type /help for commands, /exit to quit.")
	fmt.Fprintln(c.Stdout)
	for _, w := range c.Warnings {
		c.warn(w)
	}
	if notice := RestoredNotice(c.Session); notice != "" {
		fmt.Fprintf(c.Stdout, "%s\n\n", notice)
	}
}

func (c *chat) warn(msg string) { fmt.Fprintf(c.Stderr, "warning: %s\n", msg) }

func (c *chat) printError(msg string) { fmt.Fprintf(c.Stderr, "error: %s\n", msg) }

// readLines reads r line by line in a goroutine, so that waiting for input
// can be interrupted. After the final error the channel is closed.
func readLines(r io.Reader) <-chan line {
	ch := make(chan line)
	go func() {
		defer close(ch)
		br := bufio.NewReader(r)
		for {
			text, err := br.ReadString('\n')
			if text != "" {
				ch <- line{text: strings.TrimSuffix(strings.TrimSuffix(text, "\n"), "\r")}
			}
			if err != nil {
				ch <- line{err: err}
				return
			}
		}
	}()
	return ch
}

type inputStatus int

const (
	inputOK inputStatus = iota
	inputEOF
	inputInterrupted
)

// readLine shows prompt and waits for the next line, the end of the input
// or Ctrl+C. An input error other than EOF also ends the input.
func (c *chat) readLine(ctx context.Context, prompt string) (string, inputStatus) {
	io.WriteString(c.Stdout, prompt)
	select {
	case l, ok := <-c.lines:
		if !ok || l.err != nil {
			if ok && !errors.Is(l.err, io.EOF) {
				c.printError(fmt.Sprintf("Could not read input: %v", l.err))
			}
			return "", inputEOF
		}
		return l.text, inputOK
	case <-c.Interrupts:
		return "", inputInterrupted
	case <-ctx.Done():
		return "", inputInterrupted
	}
}

// command runs a command; false ends the chat.
func (c *chat) command(ctx context.Context, name string) bool {
	switch name {
	case "exit":
		fmt.Fprintln(c.Stdout, "Goodbye.")
		return false
	case "help":
		fmt.Fprintln(c.Stdout, HelpText)
	case "reset":
		err := c.Session.Reset()
		fmt.Fprintln(c.Stdout, ResetNotice(c.Session.Persistent()))
		if err != nil {
			c.warn(err.Error())
		}
	case "cleanup":
		c.cleanup(ctx)
	case "usage":
		fmt.Fprintln(c.Stdout, UsageReport(c.Session.Requests()))
	default:
		fmt.Fprintf(c.Stdout, "Unknown command: /%s. Type /help for commands.\n", name)
	}
	return true
}

// cleanup deletes the saved conversations after the next input line
// confirms it with "yes". That line is never sent or run as a command.
func (c *chat) cleanup(ctx context.Context) {
	if !c.Session.Persistent() {
		fmt.Fprintln(c.Stdout, CleanupWithoutHistory)
		return
	}
	count, err := c.Session.SavedCount()
	if err != nil {
		c.printError(err.Error())
		return
	}
	if count == 0 {
		fmt.Fprintln(c.Stdout, NoSavedConversations)
		return
	}
	answer, status := c.readLine(ctx, CleanupPrompt(count, c.Session.HistoryDir()))
	if status != inputOK {
		fmt.Fprintln(c.Stdout) // EOF or Ctrl+C cancel
	}
	if !CleanupConfirmed(answer) {
		fmt.Fprintln(c.Stdout, CleanupCancelled)
		return
	}
	deleted, err := c.Session.Cleanup()
	if err != nil {
		c.printError(err.Error())
		return
	}
	fmt.Fprintln(c.Stdout, CleanupDone(deleted))
}

type reply struct {
	resp llm.Response
	err  error
}

// send sends text with the committed context and prints the answer. Only a
// completed answer is committed and saved. Ctrl+C cancels the request and
// returns to the prompt; false means ctx was canceled.
func (c *chat) send(ctx context.Context, text string) bool {
	messages := c.Session.Context(text)
	reqCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	done := make(chan reply, 1)
	go func() {
		resp, err := c.Reply(reqCtx, messages)
		done <- reply{resp, err}
	}()

	var r reply
	select {
	case r = <-done:
	case <-c.Interrupts:
		cancel()
		<-done // at most one request in flight
		c.drainInterrupts()
		fmt.Fprintln(c.Stderr, "\nRequest cancelled.")
		return true
	case <-ctx.Done():
		cancel()
		<-done
		return false
	}

	if r.err != nil {
		if ctx.Err() != nil {
			return false
		}
		c.printError(r.err.Error())
		return true
	}
	if r.resp.IncompleteReason != "" {
		c.incomplete(r.resp)
		return true
	}
	saveErr := c.Session.Commit(text, r.resp)
	fmt.Fprintf(c.Stdout, "%s%s\n", AIPrefix, r.resp.Text)
	if c.ShowUsage {
		fmt.Fprintln(c.Stderr, FormatUsage(r.resp.Usage))
	}
	if saveErr != nil {
		c.warn(saveErr.Error())
	}
	return true
}

// incomplete shows an answer that stopped early. It is not added to the
// conversation; its reported usage is kept as an incomplete record.
func (c *chat) incomplete(resp llm.Response) {
	notSaved := ""
	if resp.Text != "" {
		fmt.Fprintf(c.Stdout, "%s%s\n", AIPrefix, resp.Text)
		notSaved = " The partial response was not saved."
	}
	c.warn(fmt.Sprintf("Response incomplete: %s.%s", resp.IncompleteReason, notSaved))
	if err := c.Session.RecordIncomplete(resp); err != nil {
		c.warn(err.Error())
	}
}

// drainInterrupts drops Ctrl+C presses that arrived while a canceled
// request was finishing, so they do not also end the chat.
func (c *chat) drainInterrupts() {
	for {
		select {
		case <-c.Interrupts:
		default:
			return
		}
	}
}
