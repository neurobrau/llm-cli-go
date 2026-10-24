package cli

import (
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/neurobrau/llm-cli-go/internal/storage"
)

// sequence answers the n-th request with handlers[n].
func sequence(handlers ...http.HandlerFunc) http.HandlerFunc {
	var mu sync.Mutex
	n := 0
	return func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		h := handlers[n]
		n++
		mu.Unlock()
		h(w, r)
	}
}

func TestInteractiveStreamRouting(t *testing.T) {
	for _, args := range [][]string{{"--stream"}, {"--plain", "--stream"}} {
		t.Run(strings.Join(args, " "), func(t *testing.T) {
			f := startFakeOllama(t, tags, sequence(
				chunks(ollamaLine("Hel"), ollamaLine("lo!"), ollamaDone),
				chunks(ollamaLine("Part"), ollamaLength),
				chunks(ollamaLine("Fine."), ollamaDone),
			))
			args := append([]string{"--provider", "ollama", "--model", "llama3.2:3b", "--usage"}, args...)
			code, stdout, stderr := chatRun(t, "Hi\nMore\nAnd?\n", args...)
			if code != ExitOK {
				t.Fatalf("code=%d stderr=%q", code, stderr)
			}
			if !strings.Contains(stdout, "YOU > AI > Hello!\nYOU > AI > Part\nYOU > AI > Fine.\nYOU > \nGoodbye.\n") {
				t.Errorf("stdout %q", stdout)
			}
			usage := "Token usage: input=12, output=5, total=17\nDuration: 2.50 s (load 1.00 s, generation 1.20 s)\n"
			if stderr != usage+"warning: Response incomplete: the max output token limit was reached. The partial response was not saved.\n"+usage {
				t.Errorf("stderr %q", stderr)
			}
			for i, c := range f.chats {
				if c["stream"] != true {
					t.Errorf("request %d is not streamed", i)
				}
			}
			if got := messages(f.chats[2]); !reflect.DeepEqual(got, []string{"user: Hi", "assistant: Hello!", "user: And?"}) {
				t.Errorf("third request %q", got)
			}

			paths := statePaths(t)
			files, _ := filepath.Glob(filepath.Join(paths.Conversations, "*.json"))
			if len(files) != 1 {
				t.Fatalf("files %v", files)
			}
			c, _ := storage.NewHistoryStore(paths.Conversations).Load(files[0])
			if len(c.Messages) != 4 || len(c.Requests) != 3 || c.Requests[1].Status != storage.StatusIncomplete {
				t.Errorf("saved %+v", c)
			}

			// A restart continues with the completed exchanges only.
			code, stdout, _ = chatRun(t, "Again\n", args...)
			if code != ExitOK || !strings.Contains(stdout, "(4 messages). Type /reset to start a new one.") {
				t.Errorf("restart stdout %q", stdout)
			}
			if got := messages(f.chats[3]); len(got) != 5 || got[4] != "user: Again" || strings.Contains(strings.Join(got, "|"), "Part") {
				t.Errorf("restored context %q", got)
			}
		})
	}
}

func TestInteractiveStreamUsesSettings(t *testing.T) {
	const settingsJSON = `{"schema_version": 1, "answer_style": "brief", "providers": {"openai": {"models": {
	  "gpt-6-luna": {"reasoning": "low", "max_output_tokens": 300}}}}}`
	tests := []struct {
		name  string
		args  []string
		check map[string]any
	}{
		{"saved", nil, map[string]any{"reasoning": effort("low"), "max_output_tokens": float64(300), "instructions": llmStyle("brief")}},
		{"CLI over saved", []string{"--reasoning-effort", "high", "--max-output-tokens", "50", "--answer-style", "normal"},
			map[string]any{"reasoning": effort("high"), "max_output_tokens": float64(50), "instructions": nil}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			stream := openAIStream(openAIDelta("ok"), sseEvent("response.completed", openAICompleted))
			f := startFakeOpenAI(t, sequence(stream, stream))
			paths := statePaths(t)
			writeTestFile(t, paths.Settings, settingsJSON)
			code, stdout, stderr := chatRun(t, "Hi\nAgain\n", append([]string{"--stream"}, tt.args...)...)
			if code != ExitOK || stderr != "" || strings.Count(stdout, "AI > ok\n") != 2 {
				t.Fatalf("code=%d stdout=%q stderr=%q", code, stdout, stderr)
			}
			for i, body := range f.bodies() {
				if body["stream"] != true || body["store"] != false {
					t.Errorf("request %d: %v", i, body)
				}
				for k, v := range tt.check {
					if !reflect.DeepEqual(body[k], v) {
						t.Errorf("request %d: %s = %v, want %v", i, k, body[k], v)
					}
				}
			}
			if data, _ := os.ReadFile(paths.Settings); string(data) != settingsJSON {
				t.Error("settings.json changed")
			}
		})
	}
}

func TestInteractiveStreamNoHistory(t *testing.T) {
	startFakeOllama(t, tags, sequence(
		chunks(ollamaLine("A1"), ollamaDone),
		chunks(ollamaLine("Part"), ollamaLength),
		chunks(ollamaLine("Broken")), // ends without "done"
		chunks(ollamaLine("A4"), ollamaDone),
	))
	paths := statePaths(t)
	state := filepath.Dir(paths.Conversations)
	writeTestFile(t, filepath.Join(paths.Conversations, "7c1e4b2a-9f3d-4e8a-b6c5-1a2b3c4d5e6f.json"), fixture(t, "conversation_openai_usage.json"))
	os.Chmod(state, 0o755)               // any history access would tighten it to 0700
	os.Chmod(paths.Conversations, 0o311) // unreadable: any read would warn
	t.Cleanup(func() { os.Chmod(paths.Conversations, 0o700) })
	before := tree(t, state)

	code, stdout, stderr := chatRun(t, "Q1\nQ2\nQ3\n/reset\nQ4\n/cleanup\n/exit\n",
		"--stream", "--no-history", "--provider", "ollama", "--model", "llama3.2:3b")
	if code != ExitOK || !strings.Contains(stdout, "Conversation history cleared.") || !strings.Contains(stdout, "AI > A4\n") ||
		!strings.Contains(stderr, "error: Ollama stream ended before the response was complete. The partial response was not saved.") {
		t.Errorf("code=%d stdout=%q stderr=%q", code, stdout, stderr)
	}
	if after := tree(t, state); !reflect.DeepEqual(before, after) {
		t.Errorf("--no-history changed the state directory")
	}
}
