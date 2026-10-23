package session

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/neurobrau/llm-cli-go/internal/llm"
	"github.com/neurobrau/llm-cli-go/internal/storage"
)

// fakeClock returns times one second apart, starting 2026-10-23 10:00 UTC.
func fakeClock() func() time.Time {
	var mu sync.Mutex
	t := time.Date(2026, 10, 23, 10, 0, 0, 0, time.UTC)
	return func() time.Time {
		mu.Lock()
		defer mu.Unlock()
		t = t.Add(time.Second)
		return t
	}
}

// newStore returns a history store under a temp directory.
func newStore(t *testing.T) *storage.HistoryStore {
	t.Helper()
	return storage.NewHistoryStore(filepath.Join(t.TempDir(), "state", "neurobrau", "conversations"))
}

// copyFixture copies a Python-written fixture into the store directory.
func copyFixture(t *testing.T, store *storage.HistoryStore, name string) string {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("..", "storage", "testdata", "python", name))
	if err != nil {
		t.Fatal(err)
	}
	c, err := storage.DecodeConversation(data)
	if err != nil {
		t.Fatal(err)
	}
	return writeFile(t, store.PathFor(c.ID), string(data))
}

func writeFile(t *testing.T, path, content string) string {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func readFile(t *testing.T, path string) string {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return string(data)
}

func intp(n int) *int { return &n }

func sel(provider, model string) storage.Selection {
	return storage.Selection{Provider: provider, Model: model}
}

func answer(text string, input, output int) llm.Response {
	return llm.Response{Text: text, ServiceTier: "default",
		Usage: &llm.Usage{InputTokens: intp(input), OutputTokens: intp(output), TotalTokens: intp(input + output)}}
}

func msgs(pairs ...string) []llm.Message {
	var m []llm.Message
	for i, content := range pairs {
		role := llm.RoleUser
		if i%2 == 1 {
			role = llm.RoleAssistant
		}
		m = append(m, llm.Message{Role: role, Content: content})
	}
	return m
}

func load(t *testing.T, store *storage.HistoryStore, id string) storage.Conversation {
	t.Helper()
	c, err := store.Load(store.PathFor(id))
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func open(t *testing.T, store *storage.HistoryStore, provider, model string) (*Session, []string) {
	t.Helper()
	return Open(Options{Store: store, Provider: provider, Model: model, Clock: fakeClock()})
}

func TestTwoTurnsAreCommittedAndSaved(t *testing.T) {
	store := newStore(t)
	s, warnings := open(t, store, "ollama", "qwen3.5:4b")
	if warnings != nil || s.Restored() || !s.Persistent() {
		t.Fatalf("warnings %q, restored %v", warnings, s.Restored())
	}
	if _, err := os.Stat(store.Dir()); err == nil {
		t.Error("a new conversation was written before its first exchange")
	}

	if got := s.Context("Hi"); !reflect.DeepEqual(got, msgs("Hi")) {
		t.Errorf("first context %q", got)
	}
	if err := s.Commit("Hi", answer("Hello!", 10, 3)); err != nil {
		t.Fatal(err)
	}
	if got := s.Context("And?"); !reflect.DeepEqual(got, msgs("Hi", "Hello!", "And?")) {
		t.Errorf("second context %q", got)
	}
	if got := s.Messages(); len(got) != 2 {
		t.Errorf("Context committed something: %q", got)
	}
	if err := s.Commit("And?", llm.Response{Text: "More.\nTwo lines."}); err != nil { // no usage reported
		t.Fatal(err)
	}
	if err := s.Commit("Last", answer("Done", 30, 2)); err != nil {
		t.Fatal(err)
	}

	c := load(t, store, s.Conversation().ID)
	if !reflect.DeepEqual(c.Messages, msgs("Hi", "Hello!", "And?", "More.\nTwo lines.", "Last", "Done")) {
		t.Errorf("saved messages %q", c.Messages)
	}
	if len(c.Requests) != 2 || *c.Requests[0].Usage.InputTokens != 10 || *c.Requests[1].Usage.InputTokens != 30 ||
		c.Requests[1].Status != storage.StatusCompleted || c.Requests[1].Provider != "ollama" ||
		c.Requests[1].Model != "qwen3.5:4b" || c.Requests[1].ServiceTier != "default" {
		t.Errorf("saved requests %+v", c.Requests)
	}
	if c.Provider != "ollama" || c.Model != "qwen3.5:4b" || !c.UpdatedAt.After(c.CreatedAt) {
		t.Errorf("saved %+v", c)
	}
	if got := readFile(t, store.ActivePath()); got != `{"schema_version": 1, "provider": "ollama", "model": "qwen3.5:4b"}`+"\n" {
		t.Errorf("active.json %q", got)
	}
}

func TestIncompleteAnswersAreNotCommitted(t *testing.T) {
	store := newStore(t)
	s, _ := open(t, store, "openai", "gpt-6-luna")
	cut := llm.Response{Text: "Partial", IncompleteReason: "max_output_tokens", Usage: &llm.Usage{InputTokens: intp(5), OutputTokens: intp(9)}}

	// A new conversation is not written just for an incomplete answer.
	if err := s.RecordIncomplete(cut); err != nil {
		t.Fatal(err)
	}
	if len(s.Messages()) != 0 || len(s.Requests()) != 1 || s.Requests()[0].Status != storage.StatusIncomplete {
		t.Errorf("messages %q, requests %+v", s.Messages(), s.Requests())
	}
	if _, err := os.Stat(store.PathFor(s.Conversation().ID)); err == nil {
		t.Error("incomplete answer saved a new conversation")
	}
	if err := s.RecordIncomplete(llm.Response{IncompleteReason: "x"}); err != nil || len(s.Requests()) != 1 {
		t.Errorf("no usage reported: %v, %d records", err, len(s.Requests()))
	}

	if err := s.Commit("Q", answer("A", 1, 1)); err != nil {
		t.Fatal(err)
	}
	if err := s.RecordIncomplete(cut); err != nil {
		t.Fatal(err)
	}
	c := load(t, store, s.Conversation().ID)
	if !reflect.DeepEqual(c.Messages, msgs("Q", "A")) || len(c.Requests) != 3 {
		t.Fatalf("saved %+v", c)
	}
	statuses := []string{c.Requests[0].Status, c.Requests[1].Status, c.Requests[2].Status}
	if !reflect.DeepEqual(statuses, []string{"incomplete", "completed", "incomplete"}) {
		t.Errorf("statuses %q", statuses)
	}
}

func TestRestoreSelection(t *testing.T) {
	setup := func(t *testing.T) *storage.HistoryStore {
		store := newStore(t)
		for _, f := range []string{"conversation_ollama_unicode.json", "conversation_openai_usage.json",
			"conversation_legacy_no_provider.json", "conversation_empty_after_reset.json"} {
			copyFixture(t, store, f)
		}
		return store
	}
	const (
		ollamaID = "0b6f8e52-2c1d-4c7e-9a3b-5d2e1f0a9c41"
		lunaID   = "7c1e4b2a-9f3d-4e8a-b6c5-1a2b3c4d5e6f"
		legacyID = "3f2a1b0c-8d7e-4f6a-9b5c-4d3e2f1a0b9c"
		emptyID  = "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d" // openai/gpt-5, newest
	)
	tests := []struct {
		name            string
		active          string // "" = no active.json
		provider, model string
		wantID          string // "" = new conversation
		wantSel         storage.Selection
		wantRestored    bool
		wantWarning     string
	}{
		{"no flags, no active: newest of all", "", "", "", emptyID, sel("openai", "gpt-5"), false, ""},
		{"no flags, active.json", `{"schema_version": 1, "provider": "ollama", "model": "qwen3.5:4b"}`, "", "",
			ollamaID, sel("ollama", "qwen3.5:4b"), true, ""},
		{"active.json with an unknown provider", `{"schema_version": 1, "provider": "anthropic", "model": "c"}`, "", "",
			emptyID, sel("openai", "gpt-5"), false, "Ignoring invalid "},
		{"corrupt active.json", `{"schema_version": 1,`, "", "", emptyID, sel("openai", "gpt-5"), false, "Ignoring unreadable "},
		{"active.json for a model without files", `{"schema_version": 1, "provider": "ollama", "model": "llama3.2:3b"}`, "", "",
			"", sel("ollama", "llama3.2:3b"), false, ""},
		{"provider only adopts its newest model", `{"schema_version": 1, "provider": "ollama", "model": "qwen3.5:4b"}`, "openai", "",
			emptyID, sel("openai", "gpt-5"), false, ""},
		{"provider only, ollama", "", "ollama", "", ollamaID, sel("ollama", "qwen3.5:4b"), true, ""},
		{"exact model", "", "openai", "gpt-6-luna", lunaID, sel("openai", "gpt-6-luna"), true, ""},
		{"legacy file matches any OpenAI model", "", "openai", "o3", legacyID, sel("openai", "o3"), true, ""},
		{"no match: new conversation", "", "ollama", "llama3.2:3b", "", sel("ollama", "llama3.2:3b"), false, ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			store := setup(t)
			if tt.active != "" {
				writeFile(t, store.ActivePath(), tt.active)
			}
			before := snapshot(t, filepath.Dir(store.Dir()))
			s, warnings := open(t, store, tt.provider, tt.model)
			c := s.Conversation()
			if tt.wantID != "" && c.ID != tt.wantID || tt.wantID == "" && (len(c.Messages) != 0 || c.ID == "") {
				t.Errorf("conversation %s (%d messages), want %q", c.ID, len(c.Messages), tt.wantID)
			}
			if s.Selection() != tt.wantSel || s.Restored() != tt.wantRestored {
				t.Errorf("selection %+v restored %v; want %+v %v", s.Selection(), s.Restored(), tt.wantSel, tt.wantRestored)
			}
			if tt.wantWarning == "" && warnings != nil ||
				tt.wantWarning != "" && (len(warnings) != 1 || !strings.HasPrefix(warnings[0], tt.wantWarning+store.ActivePath())) {
				t.Errorf("warnings %q, want %q", warnings, tt.wantWarning)
			}
			if after := snapshot(t, filepath.Dir(store.Dir())); !reflect.DeepEqual(before, after) {
				t.Error("restoring changed files")
			}
		})
	}
}

// snapshot returns the content of every file under dir.
func snapshot(t *testing.T, dir string) map[string]string {
	t.Helper()
	files := map[string]string{}
	filepath.WalkDir(dir, func(path string, d os.DirEntry, err error) error {
		if err == nil && !d.IsDir() {
			data, _ := os.ReadFile(path)
			files[path] = string(data)
		}
		return nil
	})
	return files
}

func TestRestoreSkipsInvalidAndForeignFiles(t *testing.T) {
	store := newStore(t)
	luna := copyFixture(t, store, "conversation_openai_usage.json")
	corrupt := writeFile(t, filepath.Join(store.Dir(), "11111111-2222-4333-8444-555555555555.json"), `{"schema_version": 1,`)
	future := writeFile(t, filepath.Join(store.Dir(), "22222222-2222-4333-8444-555555555555.json"),
		`{"schema_version": 2, "id": "22222222-2222-4333-8444-555555555555", "messages": []}`)
	// A newer conversation of a provider this edition does not know.
	writeFile(t, filepath.Join(store.Dir(), "33333333-2222-4333-8444-555555555555.json"), `{"schema_version": 1,
	  "id": "33333333-2222-4333-8444-555555555555", "created_at": "2027-01-01T00:00:00Z",
	  "updated_at": "2027-01-01T00:00:00Z", "provider": "anthropic", "model": "c", "messages": []}`)
	writeFile(t, filepath.Join(store.Dir(), "notes.txt"), "not a conversation")
	before := snapshot(t, filepath.Dir(store.Dir()))

	s, warnings := open(t, store, "", "")
	if s.Conversation().ID != "7c1e4b2a-9f3d-4e8a-b6c5-1a2b3c4d5e6f" || s.Selection() != (sel("openai", "gpt-6-luna")) {
		t.Errorf("restored %+v", s.Selection())
	}
	if len(warnings) != 2 || !strings.HasPrefix(warnings[0], "Ignoring unreadable conversation file "+corrupt+": ") ||
		!strings.Contains(warnings[1], future) || !strings.HasSuffix(warnings[1], "The file was left unchanged.") {
		t.Errorf("warnings %q", warnings)
	}
	if !reflect.DeepEqual(before, snapshot(t, filepath.Dir(store.Dir()))) {
		t.Error("files changed")
	}

	// Continuing the Python file keeps its data and appends the new exchange.
	if err := s.Commit("Next question", answer("Next answer", 1500, 20)); err != nil {
		t.Fatal(err)
	}
	c := load(t, store, s.Conversation().ID)
	orig, _ := storage.DecodeConversation([]byte(before[luna]))
	if !reflect.DeepEqual(c.Messages[:len(orig.Messages)], orig.Messages) || len(c.Messages) != len(orig.Messages)+2 ||
		!reflect.DeepEqual(c.Requests[:len(orig.Requests)], orig.Requests) || !c.CreatedAt.Equal(orig.CreatedAt) {
		t.Errorf("continued conversation %+v", c)
	}
	if readFile(t, corrupt) != before[corrupt] {
		t.Error("corrupt file changed")
	}
}

func TestLegacyFileTakesTheRequestedModel(t *testing.T) {
	store := newStore(t)
	legacy := copyFixture(t, store, "conversation_legacy_no_provider.json")
	before := readFile(t, legacy)
	s, _ := open(t, store, "", "")
	if s.Selection() != (sel("openai", "")) {
		t.Fatalf("selection %+v", s.Selection())
	}
	s.SetDefaults("openai", "gpt-6-luna") // what the CLI does with the default model
	if s.Selection() != (sel("openai", "gpt-6-luna")) || !s.Restored() {
		t.Errorf("selection %+v", s.Selection())
	}
	if readFile(t, legacy) != before {
		t.Error("restoring rewrote the legacy file")
	}
	s.Commit("Q", answer("A", 1, 1))
	if c := load(t, store, s.Conversation().ID); c.Provider != "openai" || c.Model != "gpt-6-luna" {
		t.Errorf("saved legacy conversation as %s/%s", c.Provider, c.Model)
	}
}

func TestReset(t *testing.T) {
	store := newStore(t)
	s, _ := open(t, store, "openai", "gpt-6-luna")
	firstID := s.Conversation().ID
	if err := s.Reset(); err != nil || s.Conversation().ID != firstID {
		t.Errorf("empty reset: %v, id %s → %s", err, firstID, s.Conversation().ID)
	}
	if _, err := os.Stat(store.Dir()); err == nil {
		t.Error("empty reset wrote a file")
	}

	s.Commit("Q", answer("A", 1, 1))
	old := readFile(t, store.PathFor(firstID))
	if err := s.Reset(); err != nil {
		t.Fatal(err)
	}
	c := s.Conversation()
	if c.ID == firstID || len(c.Messages) != 0 || len(c.Requests) != 0 || c.Provider != "openai" || c.Model != "gpt-6-luna" {
		t.Errorf("after reset %+v", c)
	}
	saved := load(t, store, c.ID)
	if len(saved.Messages) != 0 || saved.Requests != nil {
		t.Errorf("empty conversation not saved: %+v", saved)
	}
	if readFile(t, store.PathFor(firstID)) != old {
		t.Error("reset changed the previous file")
	}
	// The empty conversation is what the next start restores.
	next, _ := open(t, store, "", "")
	if next.Conversation().ID != c.ID || next.Restored() {
		t.Errorf("next start restored %s (restored %v), want %s", next.Conversation().ID, next.Restored(), c.ID)
	}
}

func TestNoHistory(t *testing.T) {
	s, warnings := Open(Options{Provider: "ollama", Model: "m", Clock: fakeClock()})
	if s.Persistent() || warnings != nil || s.HistoryDir() != "" {
		t.Fatal("history enabled")
	}
	first := s.Conversation().ID
	if err := s.Commit("Q", answer("A", 1, 1)); err != nil {
		t.Fatal(err)
	}
	if err := s.Reset(); err != nil || s.Conversation().ID == first || len(s.Messages()) != 0 {
		t.Errorf("reset: %v %+v", err, s.Conversation())
	}
	if n, err := s.SavedCount(); n != 0 || err != nil {
		t.Errorf("SavedCount %d %v", n, err)
	}
	if n, err := s.Cleanup(); n != 0 || err != nil {
		t.Errorf("Cleanup %d %v", n, err)
	}
	s.Commit("Q", answer("A", 1, 1))
	s.Switch("openai", "gpt-6-luna")
	if len(s.Messages()) != 0 {
		t.Error("messages moved to another model")
	}
	s.Switch("ollama", "m")
	if !reflect.DeepEqual(s.Messages(), msgs("Q", "A")) {
		t.Errorf("parked conversation lost: %q", s.Messages())
	}
}

func TestSwitchKeepsConversationsApart(t *testing.T) {
	store := newStore(t)
	lunaFile := copyFixture(t, store, "conversation_openai_usage.json")
	lunaBytes := readFile(t, lunaFile)
	s, _ := open(t, store, "ollama", "qwen3.5:4b")
	s.Commit("Ollama Q", answer("Ollama A", 1, 1))
	ollamaID := s.Conversation().ID

	// A stored conversation comes back without being rewritten.
	if err := s.Switch("openai", "gpt-6-luna"); err != nil {
		t.Fatal(err)
	}
	if s.Conversation().ID != "7c1e4b2a-9f3d-4e8a-b6c5-1a2b3c4d5e6f" || !s.Restored() ||
		s.Context("x")[0].Content != "What is a neural network?" {
		t.Errorf("switched to %+v", s.Conversation())
	}
	if readFile(t, lunaFile) != lunaBytes {
		t.Error("switching rewrote the restored file")
	}
	if !strings.Contains(readFile(t, store.ActivePath()), `"gpt-6-luna"`) {
		t.Error("active.json not updated")
	}
	for _, m := range s.Context("x") {
		if strings.HasPrefix(m.Content, "Ollama") {
			t.Error("Ollama message sent to OpenAI")
		}
	}

	// Without a match, a new conversation is saved right away.
	if err := s.Switch("openai", "gpt-5"); err != nil {
		t.Fatal(err)
	}
	gpt5 := s.Conversation()
	if len(gpt5.Messages) != 0 || s.Restored() || gpt5.Model != "gpt-5" {
		t.Errorf("new selection %+v", gpt5)
	}
	if c := load(t, store, gpt5.ID); c.Model != "gpt-5" || len(c.Messages) != 0 {
		t.Errorf("new empty conversation saved as %+v", c)
	}

	// An unsaved empty conversation is reused for the next new selection.
	if err := s.Switch("ollama", "llama3.2:3b"); err != nil {
		t.Fatal(err)
	}
	if s.Conversation().ID == gpt5.ID {
		t.Error("the saved empty gpt-5 conversation was taken over")
	}

	// Parked conversations come back first.
	if err := s.Switch("ollama", "qwen3.5:4b"); err != nil {
		t.Fatal(err)
	}
	if s.Conversation().ID != ollamaID || !reflect.DeepEqual(s.Messages(), msgs("Ollama Q", "Ollama A")) {
		t.Errorf("parked conversation %+v", s.Conversation())
	}
	if err := s.Switch("openai", "gpt-5"); err != nil || s.Conversation().ID != gpt5.ID {
		t.Errorf("gpt-5 conversation %s, want %s (%v)", s.Conversation().ID, gpt5.ID, err)
	}
}

func TestResetThenSwitchAwayAndBack(t *testing.T) {
	store := newStore(t)
	s, _ := open(t, store, "openai", "gpt-6-luna")
	s.Commit("Q", answer("A", 1, 1))
	archived := s.Conversation().ID
	s.Reset()
	fresh := s.Conversation().ID

	s.Switch("ollama", "qwen3.5:4b")
	if s.Conversation().ID == fresh || s.Conversation().ID == archived {
		t.Error("the OpenAI conversation was taken to Ollama")
	}
	s.Switch("openai", "gpt-6-luna")
	if s.Conversation().ID != fresh || len(s.Messages()) != 0 {
		t.Errorf("back to %s with %d messages; want the empty %s, not the archived %s",
			s.Conversation().ID, len(s.Messages()), fresh, archived)
	}
	if len(load(t, store, archived).Messages) != 2 {
		t.Error("archived conversation changed")
	}
}

func TestCleanup(t *testing.T) {
	store := newStore(t)
	s, _ := open(t, store, "ollama", "m")
	s.Commit("Q", answer("A", 1, 1))
	s.Switch("ollama", "n")
	s.Commit("Q2", answer("A2", 1, 1))
	settings := writeFile(t, filepath.Join(filepath.Dir(filepath.Dir(store.Dir())), "settings.json"), "{}")
	other := writeFile(t, filepath.Join(store.Dir(), "notes.txt"), "keep")
	activeBefore := readFile(t, store.ActivePath())
	before := s.Conversation().ID

	if n, err := s.SavedCount(); n != 2 || err != nil {
		t.Fatalf("SavedCount %d %v", n, err)
	}
	n, err := s.Cleanup()
	if n != 2 || err != nil {
		t.Fatalf("Cleanup %d %v", n, err)
	}
	if c := s.Conversation(); c.ID == before || len(c.Messages) != 0 || len(c.Requests) != 0 || c.Model != "n" {
		t.Errorf("after cleanup %+v", c)
	}
	if n, _ := s.SavedCount(); n != 0 {
		t.Errorf("%d files left", n)
	}
	if readFile(t, store.ActivePath()) != activeBefore || readFile(t, other) != "keep" || readFile(t, settings) != "{}" {
		t.Error("cleanup touched other files")
	}
	// Parked conversations were forgotten, and nothing is restored from disk.
	s.Switch("ollama", "m")
	if len(s.Messages()) != 0 {
		t.Errorf("parked conversation survived cleanup: %q", s.Messages())
	}
}

func TestSaveFailureKeepsTheExchange(t *testing.T) {
	store := newStore(t)
	s, _ := open(t, store, "openai", "gpt-6-luna")
	if err := s.Commit("Q1", answer("A1", 1, 1)); err != nil {
		t.Fatal(err)
	}
	path := store.PathFor(s.Conversation().ID)
	before := readFile(t, path)
	// The conversations directory is replaced by a file: saving must fail.
	moved := store.Dir() + ".moved"
	if err := os.Rename(store.Dir(), moved); err != nil {
		t.Fatal(err)
	}
	writeFile(t, store.Dir(), "not a directory")

	err := s.Commit("Q2", answer("A2", 1, 1))
	if err == nil || !strings.HasPrefix(err.Error(), "Could not save conversation history at "+path+": ") {
		t.Errorf("save error %v", err)
	}
	if !reflect.DeepEqual(s.Messages(), msgs("Q1", "A1", "Q2", "A2")) || len(s.Requests()) != 2 {
		t.Errorf("in memory %q", s.Messages())
	}
	if readFile(t, store.Dir()) != "not a directory" || readFile(t, filepath.Join(moved, filepath.Base(path))) != before {
		t.Error("files changed")
	}

	// Once the directory is back, the next save has everything.
	os.Remove(store.Dir())
	os.Rename(moved, store.Dir())
	if err := s.Commit("Q3", answer("A3", 1, 1)); err != nil {
		t.Fatal(err)
	}
	if c := load(t, store, s.Conversation().ID); len(c.Messages) != 6 {
		t.Errorf("next save has %d messages", len(c.Messages))
	}
	entries, _ := os.ReadDir(store.Dir())
	if len(entries) != 1 {
		t.Errorf("files left behind: %v", entries)
	}
}

func TestUnreadableHistoryDirectory(t *testing.T) {
	if os.Geteuid() == 0 {
		t.Skip("root ignores directory permissions")
	}
	store := newStore(t)
	copyFixture(t, store, "conversation_openai_usage.json")
	os.Chmod(store.Dir(), 0o000)
	t.Cleanup(func() { os.Chmod(store.Dir(), 0o700) })
	s, warnings := open(t, store, "openai", "gpt-6-luna")
	// Opening tightens the app directories to 0700 like the Python edition, so the file is readable again.
	if len(warnings) != 0 || !s.Restored() {
		t.Errorf("warnings %q restored %v", warnings, s.Restored())
	}

	// A conversations path that is a file: warning, chat goes on in memory.
	dir := t.TempDir()
	blocked := storage.NewHistoryStore(writeFile(t, filepath.Join(dir, "neurobrau", "conversations"), "file"))
	s, warnings = open(t, blocked, "openai", "gpt-6-luna")
	if len(warnings) != 1 || !strings.Contains(warnings[0], "Could not read conversation history at ") {
		t.Errorf("warnings %q", warnings)
	}
	if err := s.Commit("Q", answer("A", 1, 1)); err == nil || len(s.Messages()) != 2 {
		t.Errorf("commit: %v, %d messages", err, len(s.Messages()))
	}
}

// TestConcurrentUse checks that state reads (as a future UI does while a
// request runs) are safe while exchanges are committed; run with -race.
func TestConcurrentUse(t *testing.T) {
	s, _ := open(t, newStore(t), "openai", "gpt-6-luna")
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		for i := 0; i < 50; i++ {
			s.Commit("Q", answer("A", 1, 1))
		}
	}()
	go func() {
		defer wg.Done()
		for i := 0; i < 50; i++ {
			_ = s.Requests()
			_ = s.Context("x")
			_ = s.Conversation()
			_ = s.Selection()
		}
	}()
	wg.Wait()
	if len(s.Messages()) != 100 || len(s.Requests()) != 50 {
		t.Errorf("%d messages, %d requests", len(s.Messages()), len(s.Requests()))
	}
}
