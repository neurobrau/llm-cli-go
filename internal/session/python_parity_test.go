package session

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/neurobrau/llm-cli-go/internal/llm"
	"github.com/neurobrau/llm-cli-go/internal/storage"
)

// TestPythonReadsContinuedConversation is an optional development check
// like storage's TestPythonReadsGoFiles: Go continues a Python-written
// conversation (a completed and an incomplete request), resets, and the
// Python 0.3.0 readers then load every file in the temp directory without
// warnings and would write them back byte for byte. It runs only with
//
//	NEUROBRAU_PYTHON_SRC=$PWD/../llm-cli-python/src go test -run PythonReads ./internal/session
func TestPythonReadsContinuedConversation(t *testing.T) {
	src := os.Getenv("NEUROBRAU_PYTHON_SRC")
	if src == "" {
		t.Skip("set NEUROBRAU_PYTHON_SRC to run the Python parity check")
	}
	dir := t.TempDir()
	store := storage.NewHistoryStore(filepath.Join(dir, "neurobrau", "conversations"))
	copyFixture(t, store, "conversation_openai_usage.json")
	s, warnings := open(t, store, "openai", "gpt-6-luna")
	if warnings != nil || !s.Restored() {
		t.Fatalf("warnings %q", warnings)
	}
	if err := s.Commit("Grüße 🎉", answer("Antwort\nzwei Zeilen", 2000, 30)); err != nil {
		t.Fatal(err)
	}
	if err := s.RecordIncomplete(llm.Response{IncompleteReason: "max_output_tokens", Usage: &llm.Usage{InputTokens: intp(2100)}}); err != nil {
		t.Fatal(err)
	}
	if err := s.Reset(); err != nil {
		t.Fatal(err)
	}

	cmd := exec.Command("python3", "-B", "../../scripts/python_storage_parity.py", "check", src, dir)
	cmd.Env = append(os.Environ(), "PYTHONDONTWRITEBYTECODE=1")
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("python: %v", err)
	}
	var result struct {
		Conversations map[string]string `json:"conversations"`
		Active        []string          `json:"active"`
		Warnings      []string          `json:"warnings"`
	}
	if err := json.Unmarshal(out, &result); err != nil {
		t.Fatalf("python output: %v\n%s", err, out)
	}
	if len(result.Warnings) != 0 || len(result.Conversations) != 2 || !reflect.DeepEqual(result.Active, []string{"openai", "gpt-6-luna"}) {
		t.Errorf("Python result %+v", result)
	}
	for name, text := range result.Conversations {
		if goText := readFile(t, filepath.Join(store.Dir(), name)); text != goText {
			t.Errorf("%s: Python writes back\n%s\nGo wrote\n%s", name, text, goText)
		}
	}
}
