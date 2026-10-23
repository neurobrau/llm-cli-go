package storage

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/neurobrau/llm-cli-go/internal/llm"
)

// TestPythonReadsGoFiles is an optional development check: it writes files
// with Go into a temp directory and loads them with the Python 0.3.0
// readers. It runs only when NEUROBRAU_PYTHON_SRC points at
// ../llm-cli-python/src, e.g.
//
//	NEUROBRAU_PYTHON_SRC=$PWD/../llm-cli-python/src go test -run PythonReads ./internal/storage
//
// Python never writes outside the temp directory (bytecode is disabled).
func TestPythonReadsGoFiles(t *testing.T) {
	src := os.Getenv("NEUROBRAU_PYTHON_SRC")
	if src == "" {
		t.Skip("set NEUROBRAU_PYTHON_SRC to run the Python parity check")
	}
	dir := t.TempDir()
	store := NewHistoryStore(filepath.Join(dir, "neurobrau", "conversations"))

	written := map[string][]byte{}
	conversations := []Conversation{
		sampleConversation(t),
		{ID: "11111111-2222-4333-8444-555555555555", CreatedAt: mustTime(t, "2026-01-01T00:00:00+02:00"),
			UpdatedAt: mustTime(t, "2026-01-01T00:00:00.5Z")}, // legacy shape, no messages
		{ID: "66666666-7777-4888-9999-000000000000", Provider: "openai", Model: "gpt-6-luna",
			CreatedAt: mustTime(t, "2026-10-22T05:54:00Z"), UpdatedAt: mustTime(t, "2026-10-22T05:54:29.609Z"),
			Messages: []llm.Message{{Role: llm.RoleUser, Content: "Tab\there, emoji 🎉, CJK 世界"}, {Role: llm.RoleAssistant, Content: "ok"}},
			Requests: []UsageRecord{{At: mustTime(t, "2026-10-22T05:54:29.609Z"), Provider: "openai", Model: "gpt-6-luna",
				Status: StatusIncomplete, ServiceTier: "flex", Usage: llm.Usage{InputTokens: intp(5), ReasoningTokens: intp(2)}}}},
	}
	for _, c := range conversations {
		if err := store.Save(c); err != nil {
			t.Fatal(err)
		}
		written[c.ID+".json"], _ = os.ReadFile(store.PathFor(c.ID))
	}
	if err := store.SaveActive(Selection{"openai", "gpt-6-luna"}); err != nil {
		t.Fatal(err)
	}
	settings := DefaultSettings().WithModel("ollama", "qwen3.5:4b", ModelPreferences{Reasoning: "off", MaxOutputTokensSet: true})
	settingsPath := filepath.Join(dir, "config", "neurobrau", "settings.json")
	if err := NewSettingsStore(settingsPath).Save(settings); err != nil {
		t.Fatal(err)
	}
	settingsData, _ := os.ReadFile(settingsPath)

	cmd := exec.Command("python3", "-B", "../../scripts/python_storage_parity.py", "check", src, dir)
	cmd.Env = append(os.Environ(), "PYTHONDONTWRITEBYTECODE=1")
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("python: %v", err)
	}
	var result struct {
		Conversations    map[string]string `json:"conversations"`
		Active           []string          `json:"active"`
		Settings         string            `json:"settings"`
		SettingsWritable bool              `json:"settings_writable"`
		Warnings         []string          `json:"warnings"`
	}
	if err := json.Unmarshal(out, &result); err != nil {
		t.Fatalf("python output: %v\n%s", err, out)
	}
	if len(result.Warnings) != 0 {
		t.Errorf("Python warnings: %q", result.Warnings)
	}
	for name, data := range written {
		if result.Conversations[name] != string(data) {
			t.Errorf("%s: Python writes back\n%s\nGo wrote\n%s", name, result.Conversations[name], data)
		}
	}
	if !reflect.DeepEqual(result.Active, []string{"openai", "gpt-6-luna"}) {
		t.Errorf("Python active selection %q", result.Active)
	}
	if result.Settings != string(settingsData) || !result.SettingsWritable {
		t.Errorf("Python settings\n%s\nGo wrote\n%s", result.Settings, settingsData)
	}
}
