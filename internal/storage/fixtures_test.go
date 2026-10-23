package storage

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/neurobrau/llm-cli-go/internal/llm"
)

// The files in testdata/python were written by the Python 0.3.0 serializers
// from synthetic data with scripts/python_storage_parity.py (generate).
const pythonFixtures = "testdata/python"

func readFixture(t *testing.T, name string) []byte {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(pythonFixtures, name))
	if err != nil {
		t.Fatal(err)
	}
	return data
}

// TestPythonConversationFixtures reads Python-written files and checks
// that Go writes them back byte for byte.
func TestPythonConversationFixtures(t *testing.T) {
	tests := []struct {
		file  string
		check func(t *testing.T, c Conversation)
	}{
		{"conversation_ollama_unicode.json", func(t *testing.T, c Conversation) {
			if c.Provider != "ollama" || c.Model != "qwen3.5:4b" || len(c.Messages) != 4 {
				t.Errorf("%+v", c)
			}
			if c.Messages[0].Content != "Grüße, 世界! Ünïcödé 🎉\nLine two\twith tab" || c.Messages[2] != (llm.Message{Role: llm.RoleUser}) {
				t.Errorf("messages %q", c.Messages)
			}
			if len(c.Requests) != 2 || c.Requests[1].Status != StatusIncomplete || *c.Requests[1].Usage.OutputTokens != 0 ||
				c.Requests[1].Usage.TotalTokens != nil {
				t.Errorf("requests %+v", c.Requests)
			}
		}},
		{"conversation_openai_usage.json", func(t *testing.T, c Conversation) {
			r := c.Requests[0]
			want := llm.Usage{InputTokens: intp(1000), OutputTokens: intp(200), TotalTokens: intp(1200),
				CachedInputTokens: intp(600), CacheWriteTokens: intp(100), ReasoningTokens: intp(50)}
			if r.ServiceTier != "default" || !reflect.DeepEqual(r.Usage, want) || !r.At.Equal(mustTime(t, "2026-10-22T05:54:29.609Z")) {
				t.Errorf("record %+v", r)
			}
		}},
		{"conversation_legacy_no_provider.json", func(t *testing.T, c Conversation) {
			if c.Provider != "" || c.Model != "" || !c.Matches("openai", "gpt-6-luna") || c.Matches("ollama", "") {
				t.Errorf("%+v", c)
			}
		}},
		{"conversation_empty_after_reset.json", func(t *testing.T, c Conversation) {
			if len(c.Messages) != 0 || c.Requests != nil || !c.CreatedAt.Equal(c.UpdatedAt) {
				t.Errorf("%+v", c)
			}
		}},
	}
	for _, tt := range tests {
		t.Run(tt.file, func(t *testing.T) {
			data := readFixture(t, tt.file)
			c, err := DecodeConversation(data)
			if err != nil {
				t.Fatal(err)
			}
			tt.check(t, c)
			again, err := EncodeConversation(c)
			if err != nil || string(again) != string(data) {
				t.Errorf("Go output differs from Python's:\n%s\nPython:\n%s", again, data)
			}
		})
	}
}

func TestPythonActiveAndSettingsFixtures(t *testing.T) {
	_, paths := testPaths(t)
	s := NewHistoryStore(paths.Conversations)
	writeFile(t, paths.Active, string(readFixture(t, "active.json")))
	sel, found, warning := s.LoadActive()
	if !found || sel != (Selection{"ollama", "qwen3.5:4b"}) || warning != "" {
		t.Errorf("active: %+v, %v, %q", sel, found, warning)
	}
	if err := s.SaveActive(sel); err != nil {
		t.Fatal(err)
	}
	if data, _ := os.ReadFile(paths.Active); string(data) != string(readFixture(t, "active.json")) {
		t.Errorf("Go active.json %q differs from Python's", data)
	}

	data := readFixture(t, "settings.json")
	settings, warnings, err := DecodeSettings(data)
	if err != nil || warnings != nil {
		t.Fatalf("%v, %q", err, warnings)
	}
	if settings.AnswerStyle != "brief" || !settings.ForModel("openai", "gpt-6-luna").MaxOutputTokensSet ||
		settings.ForModel("ollama", "hf.co/org/model:Q4_K_M").Reasoning != "high" {
		t.Errorf("settings %+v", settings)
	}
	again, err := EncodeSettings(settings)
	if err != nil || string(again) != string(data) {
		t.Errorf("Go settings differ from Python's:\n%s\nPython:\n%s", again, data)
	}
}
