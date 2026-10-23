package storage

import (
	"bytes"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/neurobrau/llm-cli-go/internal/llm"
)

// testPaths returns the paths for an isolated HOME in a temp directory.
func testPaths(t *testing.T) (home string, paths Paths) {
	t.Helper()
	home = t.TempDir()
	paths, err := ResolvePaths(func(string) string { return "" }, home)
	if err != nil {
		t.Fatal(err)
	}
	return home, paths
}

func newStore(t *testing.T) *HistoryStore {
	_, paths := testPaths(t)
	return NewHistoryStore(paths.Conversations)
}

func conversationAt(t *testing.T, id, provider, model, updated string) Conversation {
	return Conversation{
		ID: id, Provider: provider, Model: model,
		CreatedAt: mustTime(t, "2026-01-01T00:00:00Z"), UpdatedAt: mustTime(t, updated),
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "q-" + id[:4]}},
	}
}

func writeFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
}

// snapshot records the content and modification time of every file below
// dir, to prove that reading changes nothing.
func snapshot(t *testing.T, dir string) map[string]string {
	t.Helper()
	files := map[string]string{}
	filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
		if err == nil && !d.IsDir() {
			data, _ := os.ReadFile(path)
			info, _ := d.Info()
			files[path] = string(data) + "|" + info.ModTime().String()
		}
		return nil
	})
	return files
}

func perm(t *testing.T, path string) fs.FileMode {
	t.Helper()
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	return info.Mode().Perm()
}

const (
	idA = "aaaaaaaa-0000-4000-8000-000000000001"
	idB = "bbbbbbbb-0000-4000-8000-000000000002"
	idC = "cccccccc-0000-4000-8000-000000000003"
	idD = "dddddddd-0000-4000-8000-000000000004"
)

func TestSaveAndLoad(t *testing.T) {
	s := newStore(t)
	c := sampleConversation(t)
	if err := s.Save(c); err != nil {
		t.Fatal(err)
	}
	got, err := s.Load(s.PathFor(c.ID))
	if err != nil {
		t.Fatal(err)
	}
	if got.ID != c.ID || len(got.Messages) != 3 || got.Messages[0].Content != c.Messages[0].Content {
		t.Errorf("loaded %+v", got)
	}
}

func TestLoadLatestSelection(t *testing.T) {
	s := newStore(t)
	for _, c := range []Conversation{
		conversationAt(t, idA, "openai", "gpt-6-luna", "2026-10-21T10:00:00Z"),
		conversationAt(t, idB, "ollama", "qwen3.5:4b", "2026-10-21T12:00:00Z"),
		conversationAt(t, idC, "", "", "2026-10-21T11:00:00Z"), // legacy: OpenAI, any model
		conversationAt(t, idD, "ollama", "llama3.2", "2026-10-21T09:00:00+00:00"),
	} {
		if err := s.Save(c); err != nil {
			t.Fatal(err)
		}
	}
	// Corrupt and foreign files must not hide valid conversations.
	writeFile(t, filepath.Join(s.Dir(), "ffffffff-0000-4000-8000-00000000000f.json"), `{"schema_version": 1, "id": `)
	writeFile(t, filepath.Join(s.Dir(), "notes.txt"), "not a conversation")
	before := snapshot(t, filepath.Dir(s.Dir()))

	tests := []struct {
		name, provider, model, want string
	}{
		{"any", "", "", idB},
		{"openai", "openai", "", idC},
		{"openai model", "openai", "gpt-6-luna", idC}, // legacy file matches any model
		{"ollama", "ollama", "", idB},
		{"ollama model", "ollama", "llama3.2", idD},
		{"no match", "ollama", "absent", ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, found, warnings := s.LoadLatest(func(c Conversation) bool { return c.Matches(tt.provider, tt.model) })
			if found != (tt.want != "") || got.ID != tt.want {
				t.Errorf("got %q (found %v), want %q", got.ID, found, tt.want)
			}
			if len(warnings) != 1 || !strings.HasPrefix(warnings[0], "Ignoring unreadable conversation file "+s.Dir()) ||
				!strings.HasSuffix(warnings[0], ". The file was left unchanged.") {
				t.Errorf("warnings = %q", warnings)
			}
		})
	}
	if after := snapshot(t, filepath.Dir(s.Dir())); !mapsEqual(before, after) {
		t.Error("restoring changed files on disk")
	}
}

func mapsEqual(a, b map[string]string) bool {
	if len(a) != len(b) {
		return false
	}
	for k, v := range a {
		if b[k] != v {
			return false
		}
	}
	return true
}

func TestLoadLatestTieBrokenByModificationTime(t *testing.T) {
	s := newStore(t)
	same := "2026-10-21T12:00:00.000Z"
	for _, id := range []string{idA, idB, idC} {
		if err := s.Save(conversationAt(t, id, "openai", "m", same)); err != nil {
			t.Fatal(err)
		}
	}
	base := time.Now().Add(-time.Hour)
	for i, id := range []string{idA, idC, idB} { // idB gets the newest mtime
		mtime := base.Add(time.Duration(i) * time.Minute)
		if err := os.Chtimes(s.PathFor(id), mtime, mtime); err != nil {
			t.Fatal(err)
		}
	}
	got, found, warnings := s.LoadLatest(nil)
	if !found || got.ID != idB || warnings != nil {
		t.Errorf("got %q, %v, %q; want %q", got.ID, found, warnings, idB)
	}
}

func TestInvalidFilesAreSkippedAndKept(t *testing.T) {
	s := newStore(t)
	valid := conversationAt(t, idA, "openai", "m", "2026-10-21T08:00:00Z")
	if err := s.Save(valid); err != nil {
		t.Fatal(err)
	}
	bad := map[string]string{
		"schema 2":     `{"schema_version": 2, "id": "` + idB + `"}`,
		"bad uuid":     `{"schema_version": 1, "id": "xyz", "messages": []}`,
		"bad time":     `{"schema_version": 1, "id": "` + idB + `", "messages": [], "created_at": "noon", "updated_at": "noon"}`,
		"bad role":     `{"schema_version": 1, "id": "` + idB + `", "messages": [{"role": "tool", "content": ""}]}`,
		"invalid utf8": "{\"schema_version\": 1, \"id\": \"\xc3\x28\"}",
		"not json":     "hello",
		"empty":        "",
	}
	i := 0
	for _, content := range bad {
		i++
		writeFile(t, filepath.Join(s.Dir(), "bad"+string(rune('a'+i))+".json"), content)
	}
	if err := os.Mkdir(filepath.Join(s.Dir(), "folder.json"), 0o700); err != nil {
		t.Fatal(err)
	}
	before := snapshot(t, s.Dir())
	got, found, warnings := s.LoadLatest(nil)
	if !found || got.ID != idA {
		t.Errorf("valid conversation not restored: %q", got.ID)
	}
	if len(warnings) != len(bad)+1 {
		t.Errorf("%d warnings, want %d: %q", len(warnings), len(bad)+1, warnings)
	}
	if after := snapshot(t, s.Dir()); !mapsEqual(before, after) {
		t.Error("invalid files were changed")
	}
}

func TestMissingDirectoryAndUnreadableDirectory(t *testing.T) {
	s := newStore(t)
	if _, found, warnings := s.LoadLatest(nil); found || warnings != nil {
		t.Errorf("missing dir: found=%v warnings=%q", found, warnings)
	}
	if _, err := os.Stat(s.Dir()); !errors.Is(err, fs.ErrNotExist) {
		t.Error("LoadLatest created the directory")
	}
	writeFile(t, s.Dir(), "a file where the directory should be")
	_, found, warnings := s.LoadLatest(nil)
	if found || len(warnings) != 1 || !strings.HasPrefix(warnings[0], "Could not read conversation history at "+s.Dir()+": ") {
		t.Errorf("found=%v warnings=%q", found, warnings)
	}
	err := s.Save(conversationAt(t, idA, "openai", "m", "2026-10-21T08:00:00Z"))
	if err == nil || !strings.HasPrefix(err.Error(), "Could not save conversation history at ") {
		t.Errorf("Save into a file: %v", err)
	}
}

func TestPermissions(t *testing.T) {
	home, paths := testPaths(t)
	s := NewHistoryStore(paths.Conversations)
	stateHome := filepath.Join(home, ".local", "state")
	if err := os.MkdirAll(stateHome, 0o751); err != nil {
		t.Fatal(err)
	}
	os.Chmod(stateHome, 0o751)
	os.Chmod(home, 0o755)

	if err := s.Save(conversationAt(t, idA, "openai", "m", "2026-10-21T08:00:00Z")); err != nil {
		t.Fatal(err)
	}
	if err := s.SaveActive(Selection{"openai", "m"}); err != nil {
		t.Fatal(err)
	}
	appDir := filepath.Dir(paths.Conversations)
	for path, want := range map[string]fs.FileMode{
		appDir: 0o700, paths.Conversations: 0o700, s.PathFor(idA): 0o600, paths.Active: 0o600,
		stateHome: 0o751, home: 0o755, // ancestors are not ours: unchanged
	} {
		if got := perm(t, path); got != want {
			t.Errorf("%s: mode %o, want %o", path, got, want)
		}
	}

	// Lax permissions left by other tools are tightened, inside the app
	// directory only; an older lax file gets 0600 when it is rewritten.
	os.Chmod(appDir, 0o755)
	os.Chmod(paths.Conversations, 0o777)
	os.Chmod(s.PathFor(idA), 0o644)
	s.LoadLatest(nil)
	if perm(t, appDir) != 0o700 || perm(t, paths.Conversations) != 0o700 || perm(t, stateHome) != 0o751 {
		t.Errorf("after restore: app %o, conversations %o, state %o", perm(t, appDir), perm(t, paths.Conversations), perm(t, stateHome))
	}
	if err := s.Save(conversationAt(t, idA, "openai", "m", "2026-10-21T09:00:00Z")); err != nil {
		t.Fatal(err)
	}
	if got := perm(t, s.PathFor(idA)); got != 0o600 {
		t.Errorf("rewritten file mode %o, want 600", got)
	}
}

func TestAtomicWrite(t *testing.T) {
	s := newStore(t)
	c := conversationAt(t, idA, "openai", "m", "2026-10-21T08:00:00Z")
	if err := s.Save(c); err != nil {
		t.Fatal(err)
	}
	old, _ := os.ReadFile(s.PathFor(idA))

	c.Messages = append(c.Messages, llm.Message{Role: llm.RoleAssistant, Content: "new"})
	c.UpdatedAt = mustTime(t, "2026-10-21T09:00:00Z")
	if err := s.Save(c); err != nil {
		t.Fatal(err)
	}
	if got, err := s.Load(s.PathFor(idA)); err != nil || len(got.Messages) != 2 {
		t.Fatalf("after overwrite: %+v, %v", got, err)
	}
	assertNoTempFiles(t, s.Dir())

	// A failure before the rename keeps the old file and leaves no temp file.
	current, _ := os.ReadFile(s.PathFor(idA))
	syncFile = func(*os.File) error { return errors.New("disk full") }
	t.Cleanup(func() { syncFile = (*os.File).Sync })
	c.Messages = append(c.Messages, llm.Message{Role: llm.RoleUser, Content: "lost"})
	err := s.Save(c)
	if err == nil || err.Error() != "Could not save conversation history at "+s.PathFor(idA)+": Disk full." {
		t.Errorf("err = %v", err)
	}
	if after, _ := os.ReadFile(s.PathFor(idA)); !bytes.Equal(after, current) || bytes.Equal(after, old) {
		t.Error("failed save changed the file")
	}
	assertNoTempFiles(t, s.Dir())
}

func assertNoTempFiles(t *testing.T, dir string) {
	t.Helper()
	entries, _ := os.ReadDir(dir)
	for _, e := range entries {
		if strings.HasSuffix(e.Name(), ".tmp") {
			t.Errorf("temp file left behind: %s", e.Name())
		}
	}
}

func TestDeleteAll(t *testing.T) {
	_, paths := testPaths(t)
	s := NewHistoryStore(paths.Conversations)
	if n, err := s.DeleteAll(); n != 0 || err != nil {
		t.Errorf("no directory: %d, %v", n, err)
	}
	for _, id := range []string{idA, idB} {
		if err := s.Save(conversationAt(t, id, "openai", "m", "2026-10-21T08:00:00Z")); err != nil {
			t.Fatal(err)
		}
	}
	writeFile(t, filepath.Join(s.Dir(), "corrupt.json"), "{")
	writeFile(t, filepath.Join(s.Dir(), "notes.txt"), "keep")
	writeFile(t, filepath.Join(s.Dir(), "backup.json.bak"), "keep")
	os.Mkdir(filepath.Join(s.Dir(), "sub.json"), 0o700)
	if err := s.SaveActive(Selection{"ollama", "qwen3.5:4b"}); err != nil {
		t.Fatal(err)
	}
	settings := NewSettingsStore(paths.Settings)
	if err := settings.Save(DefaultSettings()); err != nil {
		t.Fatal(err)
	}

	if n, err := s.CountConversations(); n != 4 || err != nil { // includes corrupt.json and sub.json
		t.Errorf("CountConversations = %d, %v", n, err)
	}
	n, err := s.DeleteAll()
	if n != 3 || err != nil {
		t.Errorf("DeleteAll = %d, %v; want 3", n, err)
	}
	var left []string
	entries, _ := os.ReadDir(s.Dir())
	for _, e := range entries {
		left = append(left, e.Name())
	}
	if !slices.Equal(left, []string{"backup.json.bak", "notes.txt", "sub.json"}) {
		t.Errorf("left %v", left)
	}
	for _, path := range []string{paths.Active, paths.Settings} {
		if _, err := os.Stat(path); err != nil {
			t.Errorf("%s removed: %v", path, err)
		}
	}
	if n, err := s.DeleteAll(); n != 0 || err != nil {
		t.Errorf("second DeleteAll = %d, %v", n, err)
	}
}

func TestActiveSelection(t *testing.T) {
	_, paths := testPaths(t)
	s := NewHistoryStore(paths.Conversations)
	if s.ActivePath() != paths.Active || strings.HasPrefix(s.ActivePath(), s.Dir()+string(filepath.Separator)) {
		t.Fatalf("active path %s", s.ActivePath())
	}
	if _, found, warning := s.LoadActive(); found || warning != "" {
		t.Errorf("missing file: found=%v warning=%q", found, warning)
	}
	sel := Selection{Provider: "ollama", Model: "hf.co/org/model:Q4_K_M"}
	if err := s.SaveActive(sel); err != nil {
		t.Fatal(err)
	}
	data, _ := os.ReadFile(paths.Active)
	if want := `{"schema_version": 1, "provider": "ollama", "model": "hf.co/org/model:Q4_K_M"}` + "\n"; string(data) != want {
		t.Errorf("active.json = %q, want %q", data, want)
	}
	if got, found, warning := s.LoadActive(); !found || got != sel || warning != "" {
		t.Errorf("LoadActive = %+v, %v, %q", got, found, warning)
	}
	if err := s.SaveActive(Selection{"anthropic", "claude"}); err == nil {
		t.Error("saved an unknown provider")
	}

	invalid := []string{
		`{"schema_version": 1, "provider": "anthropic", "model": "x"}`, // unknown provider: defaults are used
		`{"schema_version": 2, "provider": "openai", "model": "x"}`,
		`{"schema_version": 1, "provider": "openai", "model": ""}`,
		`{"schema_version": 1, "provider": "openai"}`,
		`["openai", "x"]`,
	}
	for _, content := range invalid {
		writeFile(t, paths.Active, content)
		got, found, warning := s.LoadActive()
		if found || got != (Selection{}) || warning != "Ignoring invalid "+paths.Active+"." {
			t.Errorf("%s: %+v, %v, %q", content, got, found, warning)
		}
		if data, _ := os.ReadFile(paths.Active); string(data) != content {
			t.Error("invalid active.json was changed")
		}
	}
	writeFile(t, paths.Active, "{oops")
	if _, found, warning := s.LoadActive(); found || !strings.HasPrefix(warning, "Ignoring unreadable "+paths.Active+": not valid JSON") {
		t.Errorf("unreadable: %v, %q", found, warning)
	}
}
