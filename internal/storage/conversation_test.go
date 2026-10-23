package storage

import (
	"encoding/json"
	"errors"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/neurobrau/llm-cli-go/internal/llm"
)

func intp(n int) *int { return &n }

func mustTime(t *testing.T, s string) time.Time {
	t.Helper()
	v, err := time.Parse(time.RFC3339Nano, s)
	if err != nil {
		t.Fatal(err)
	}
	return v.UTC()
}

func TestResolvePaths(t *testing.T) {
	env := func(vars map[string]string) func(string) string {
		return func(k string) string { return vars[k] }
	}
	tests := []struct {
		name string
		vars map[string]string
		want Paths
	}{
		{"defaults", nil, Paths{
			Settings:      "/home/u/.config/neurobrau/settings.json",
			Conversations: "/home/u/.local/state/neurobrau/conversations",
			Active:        "/home/u/.local/state/neurobrau/active.json",
		}},
		{"absolute XDG", map[string]string{"XDG_CONFIG_HOME": "/x/config", "XDG_STATE_HOME": "/x/state/"}, Paths{
			Settings:      "/x/config/neurobrau/settings.json",
			Conversations: "/x/state/neurobrau/conversations",
			Active:        "/x/state/neurobrau/active.json",
		}},
		{"relative XDG ignored", map[string]string{"XDG_CONFIG_HOME": "rel/config", "XDG_STATE_HOME": "./state"}, Paths{
			Settings:      "/home/u/.config/neurobrau/settings.json",
			Conversations: "/home/u/.local/state/neurobrau/conversations",
			Active:        "/home/u/.local/state/neurobrau/active.json",
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ResolvePaths(env(tt.vars), "/home/u")
			if err != nil || got != tt.want {
				t.Errorf("got %+v, %v; want %+v", got, err, tt.want)
			}
		})
	}
	if _, err := ResolvePaths(env(nil), ""); !errors.Is(err, ErrNoHome) {
		t.Errorf("no home: err = %v, want ErrNoHome", err)
	}
	got, err := ResolvePaths(env(map[string]string{"XDG_CONFIG_HOME": "/c", "XDG_STATE_HOME": "/s"}), "")
	if err != nil || got.Settings != "/c/neurobrau/settings.json" {
		t.Errorf("absolute XDG without home: %+v, %v", got, err)
	}
	if filepath.Dir(got.Active) != filepath.Dir(got.Conversations) {
		t.Error("active.json is not next to conversations/")
	}
}

func sampleConversation(t *testing.T) Conversation {
	return Conversation{
		ID:        "4b8ba3bd-5f42-4302-a01e-2d6ed0eb4905",
		CreatedAt: mustTime(t, "2026-10-21T12:00:00Z"),
		UpdatedAt: mustTime(t, "2026-10-21T12:03:41.512987Z"),
		Provider:  ProviderOllama,
		Model:     "qwen3.5:4b",
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: "Grüße 世界 🎉\nsecond line"},
			{Role: llm.RoleAssistant, Content: ""},
			{Role: llm.RoleUser, Content: `<b> & "quoted" \ back`},
		},
		Requests: []UsageRecord{{
			At: mustTime(t, "2026-10-21T12:03:41.512Z"), Provider: ProviderOllama, Model: "qwen3.5:4b",
			Status: StatusCompleted, Usage: llm.Usage{InputTokens: intp(10), OutputTokens: intp(0)},
		}},
	}
}

func TestConversationRoundTrip(t *testing.T) {
	c := sampleConversation(t)
	data, err := EncodeConversation(c)
	if err != nil {
		t.Fatal(err)
	}
	text := string(data)
	for _, want := range []string{
		"{\n  \"schema_version\": 1,\n  \"id\": \"4b8ba3bd-5f42-4302-a01e-2d6ed0eb4905\",\n",
		`"created_at": "2026-10-21T12:00:00.000Z"`,
		`"updated_at": "2026-10-21T12:03:41.512Z"`, // milliseconds, truncated
		`"content": "Grüße 世界 🎉\nsecond line"`,     // non-ASCII as is
		`"content": "<b> & \"quoted\" \\ back"`,    // no HTML escaping
		`"output_tokens": 0`,
	} {
		if !strings.Contains(text, want) {
			t.Errorf("encoded file lacks %q:\n%s", want, text)
		}
	}
	if !strings.HasSuffix(text, "}\n") {
		t.Error("no trailing newline")
	}
	keys := []string{`"schema_version"`, `"id"`, `"created_at"`, `"updated_at"`, `"provider"`, `"model"`, `"messages"`, `"requests"`}
	for i := 1; i < len(keys); i++ {
		if strings.Index(text, keys[i-1]) > strings.Index(text, keys[i]) {
			t.Errorf("%s is written after %s", keys[i-1], keys[i])
		}
	}

	got, err := DecodeConversation(data)
	if err != nil {
		t.Fatal(err)
	}
	want := c
	want.UpdatedAt = mustTime(t, "2026-10-21T12:03:41.512Z")
	if !reflect.DeepEqual(got, want) {
		t.Errorf("round trip:\n got %+v\nwant %+v", got, want)
	}
}

func TestEncodeOmitsUnknownFields(t *testing.T) {
	c := Conversation{ID: "4b8ba3bd-5f42-4302-a01e-2d6ed0eb4905", CreatedAt: time.Unix(0, 0), UpdatedAt: time.Unix(0, 0)}
	data, err := EncodeConversation(c)
	if err != nil {
		t.Fatal(err)
	}
	for _, absent := range []string{"provider", "model", "requests"} {
		if strings.Contains(string(data), absent) {
			t.Errorf("%q written although unknown/empty:\n%s", absent, data)
		}
	}
	if !strings.Contains(string(data), `"messages": []`) {
		t.Errorf("empty messages not written as []:\n%s", data)
	}
	for name, bad := range map[string]Conversation{
		"bad id":       {ID: "not-a-uuid"},
		"bad provider": {ID: c.ID, Provider: "anthropic"},
		"bad role":     {ID: c.ID, Messages: []llm.Message{{Role: llm.RoleSystem, Content: "x"}}},
	} {
		if _, err := EncodeConversation(bad); err == nil {
			t.Errorf("%s: no error", name)
		}
	}
}

func TestDecodeAcceptsOffsetsAndUnknownFields(t *testing.T) {
	data := `{"schema_version": 1, "future_field": {"x": [1]},
	  "id": "4b8ba3bd-5f42-4302-a01e-2d6ed0eb4905",
	  "created_at": "2026-10-21T14:00:00+02:00", "updated_at": "2026-10-21T12:03:41.123456-01:30",
	  "provider": "openai", "model": "gpt-6-luna",
	  "messages": [{"role": "user", "content": "Hi", "id": "m1", "extra": true}],
	  "requests": "not a list"}`
	c, err := DecodeConversation([]byte(data))
	if err != nil {
		t.Fatal(err)
	}
	if !c.CreatedAt.Equal(mustTime(t, "2026-10-21T12:00:00Z")) || c.CreatedAt.Location() != time.UTC {
		t.Errorf("created_at = %v", c.CreatedAt)
	}
	if !c.UpdatedAt.Equal(mustTime(t, "2026-10-21T13:33:41.123456Z")) {
		t.Errorf("updated_at = %v", c.UpdatedAt)
	}
	if len(c.Messages) != 1 || c.Messages[0] != (llm.Message{Role: llm.RoleUser, Content: "Hi"}) || c.Requests != nil {
		t.Errorf("decoded %+v", c)
	}
}

func TestLegacyConversation(t *testing.T) {
	data := `{"schema_version": 1, "id": "4b8ba3bd-5f42-4302-a01e-2d6ed0eb4905",
	  "created_at": "2026-09-01T08:00:00.000Z", "updated_at": "2026-09-01T08:00:00.000Z",
	  "provider": null, "messages": []}`
	c, err := DecodeConversation([]byte(data))
	if err != nil {
		t.Fatal(err)
	}
	if c.Provider != "" || c.Model != "" || c.EffectiveProvider() != ProviderOpenAI {
		t.Errorf("legacy = %+v", c)
	}
	tests := []struct {
		provider, model string
		want            bool
	}{
		{"", "", true},
		{"openai", "", true},
		{"openai", "gpt-5", true}, // no model: any model of the provider
		{"ollama", "", false},
		{"ollama", "qwen3.5:4b", false},
	}
	for _, tt := range tests {
		if got := c.Matches(tt.provider, tt.model); got != tt.want {
			t.Errorf("legacy Matches(%q, %q) = %v, want %v", tt.provider, tt.model, got, tt.want)
		}
	}
	typed := Conversation{Provider: ProviderOllama, Model: "qwen3.5:4b"}
	for _, tt := range []struct {
		provider, model string
		want            bool
	}{{"ollama", "qwen3.5:4b", true}, {"ollama", "", true}, {"", "qwen3.5:4b", true}, {"ollama", "llama3.2", false}, {"openai", "", false}} {
		if got := typed.Matches(tt.provider, tt.model); got != tt.want {
			t.Errorf("Matches(%q, %q) = %v, want %v", tt.provider, tt.model, got, tt.want)
		}
	}
}

func TestDecodeRejectsInvalidConversations(t *testing.T) {
	const id = `"id": "4b8ba3bd-5f42-4302-a01e-2d6ed0eb4905"`
	const times = `"created_at": "2026-10-21T12:00:00Z", "updated_at": "2026-10-21T12:00:00Z"`
	tests := []struct {
		name, data, want string
	}{
		{"malformed JSON", `{"schema_version": 1,`, "not valid JSON"},
		{"invalid UTF-8", "{\"schema_version\": 1, \"x\": \"\xff\"}", "not valid UTF-8"},
		{"UTF-8 BOM", "\ufeff{}", "not valid JSON"},
		{"top level list", `[]`, "top level must be a JSON object"},
		{"schema 2", `{"schema_version": 2}`, "unsupported schema_version 2 (expected 1)"},
		{"schema missing", `{}`, "unsupported schema_version None (expected 1)"},
		{"schema string", `{"schema_version": "1"}`, "unsupported schema_version '1' (expected 1)"},
		{"schema true", `{"schema_version": true}`, "unsupported schema_version True (expected 1)"},
		{"id uppercase", `{"schema_version": 1, "id": "4B8BA3BD-5F42-4302-A01E-2D6ED0EB4905"}`, "'id' must be a UUID string"},
		{"id path", `{"schema_version": 1, "id": "../../etc/passwd"}`, "'id' must be a UUID string"},
		{"messages missing", `{"schema_version": 1, ` + id + `}`, "'messages' must be a list"},
		{"message not object", `{"schema_version": 1, ` + id + `, "messages": ["hi"]}`, "message 0 must be an object"},
		{"system role", `{"schema_version": 1, ` + id + `, "messages": [{"role": "system", "content": "x"}]}`, "message 0 has invalid role 'system'"},
		{"missing role", `{"schema_version": 1, ` + id + `, "messages": [{"role": "user", "content": "a"}, {"content": "x"}]}`, "message 1 has invalid role None"},
		{"content not string", `{"schema_version": 1, ` + id + `, "messages": [{"role": "user", "content": 5}]}`, "message 0 content must be a string"},
		{"content null", `{"schema_version": 1, ` + id + `, "messages": [{"role": "user", "content": null}]}`, "message 0 content must be a string"},
		{"no time zone", `{"schema_version": 1, ` + id + `, "messages": [], "created_at": "2026-10-21T12:00:00", "updated_at": "2026-10-21T12:00:00Z"}`, "'created_at' has no time zone"},
		{"bad timestamp", `{"schema_version": 1, ` + id + `, "messages": [], "created_at": "2026-10-21T12:00:00Z", "updated_at": "yesterday"}`, "'updated_at' is not an RFC 3339 timestamp"},
		{"timestamp number", `{"schema_version": 1, ` + id + `, "messages": [], "created_at": 5}`, "'created_at' must be a string"},
		{"empty provider", `{"schema_version": 1, ` + id + `, "messages": [], ` + times + `, "provider": ""}`, "'provider' must be a non-empty string"},
		{"model number", `{"schema_version": 1, ` + id + `, "messages": [], ` + times + `, "model": 4}`, "'model' must be a non-empty string"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := DecodeConversation([]byte(tt.data))
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Errorf("err = %v, want %q", err, tt.want)
			}
		})
	}
}

func TestUsageRecords(t *testing.T) {
	data := `{"schema_version": 1, "id": "4b8ba3bd-5f42-4302-a01e-2d6ed0eb4905",
	  "created_at": "2026-10-22T05:54:00Z", "updated_at": "2026-10-22T05:54:29.609Z", "messages": [],
	  "requests": [
	    {"at": "2026-10-22T05:54:29.609Z", "provider": "openai", "model": "gpt-6-luna", "status": "completed",
	     "service_tier": "default", "input_tokens": 1000, "output_tokens": 200, "total_tokens": 1200,
	     "cached_input_tokens": 600, "cache_write_tokens": 100, "reasoning_tokens": 50, "unknown": 1},
	    {"at": "2026-10-22T05:55:00+02:00", "provider": "ollama", "model": "m", "output_tokens": 0},
	    {"at": "2026-10-22T05:56:00Z", "provider": "ollama", "model": "m", "status": "incomplete", "input_tokens": 7, "service_tier": null},
	    {"at": "2026-10-22T05:57:00Z", "provider": "ollama", "model": "m", "input_tokens": -1},
	    {"at": "2026-10-22T05:57:00Z", "provider": "ollama", "model": "m", "input_tokens": 1.5},
	    {"at": "2026-10-22T05:57:00Z", "provider": "ollama", "model": "m", "input_tokens": true},
	    {"at": "2026-10-22T05:57:00Z", "provider": "ollama", "model": "m", "input_tokens": "3"},
	    {"at": "2026-10-22T05:57:00Z", "provider": "ollama", "model": "m", "status": "failed"},
	    {"at": "2026-10-22T05:57:00Z", "provider": "ollama", "model": "m", "status": null},
	    {"at": "2026-10-22T05:57:00Z", "provider": "ollama", "model": "m", "service_tier": 5},
	    {"at": "not a time", "provider": "ollama", "model": "m"},
	    {"at": "2026-10-22T05:57:00Z", "model": "m"},
	    "not an object",
	    null
	  ]}`
	c, err := DecodeConversation([]byte(data))
	if err != nil {
		t.Fatalf("a malformed record must not invalidate the conversation: %v", err)
	}
	want := []UsageRecord{
		{At: mustTime(t, "2026-10-22T05:54:29.609Z"), Provider: "openai", Model: "gpt-6-luna", Status: StatusCompleted,
			ServiceTier: "default", Usage: llm.Usage{
				InputTokens: intp(1000), OutputTokens: intp(200), TotalTokens: intp(1200),
				CachedInputTokens: intp(600), CacheWriteTokens: intp(100), ReasoningTokens: intp(50)}},
		{At: mustTime(t, "2026-10-22T03:55:00Z"), Provider: "ollama", Model: "m", Status: StatusCompleted,
			Usage: llm.Usage{OutputTokens: intp(0)}}, // zero is reported, the rest is not
		{At: mustTime(t, "2026-10-22T05:56:00Z"), Provider: "ollama", Model: "m", Status: StatusIncomplete,
			Usage: llm.Usage{InputTokens: intp(7)}},
	}
	if !reflect.DeepEqual(c.Requests, want) {
		t.Errorf("requests:\n got %+v\nwant %+v", c.Requests, want)
	}

	// Written back: counts exactly as stored; no total is derived and
	// reasoning tokens are not added to the output.
	encoded, err := EncodeConversation(c)
	if err != nil {
		t.Fatal(err)
	}
	var back struct {
		Requests []map[string]any `json:"requests"`
	}
	json.Unmarshal(encoded, &back)
	wantJSON := []map[string]any{
		{"at": "2026-10-22T05:54:29.609Z", "provider": "openai", "model": "gpt-6-luna", "status": "completed",
			"service_tier": "default", "input_tokens": 1000.0, "output_tokens": 200.0, "total_tokens": 1200.0,
			"cached_input_tokens": 600.0, "cache_write_tokens": 100.0, "reasoning_tokens": 50.0},
		{"at": "2026-10-22T03:55:00.000Z", "provider": "ollama", "model": "m", "status": "completed", "output_tokens": 0.0},
		{"at": "2026-10-22T05:56:00.000Z", "provider": "ollama", "model": "m", "status": "incomplete", "input_tokens": 7.0},
	}
	if !reflect.DeepEqual(back.Requests, wantJSON) {
		t.Errorf("written requests:\n got %v\nwant %v", back.Requests, wantJSON)
	}
	if strings.Index(string(encoded), `"status"`) > strings.Index(string(encoded), `"service_tier"`) {
		t.Error("record keys out of order")
	}
}

func TestNewID(t *testing.T) {
	seen := map[string]bool{}
	for range 100 {
		id, err := NewID()
		if err != nil || !ValidID(id) || id[14] != '4' || !strings.ContainsRune("89ab", rune(id[19])) || seen[id] {
			t.Fatalf("NewID() = %q, %v", id, err)
		}
		seen[id] = true
	}
}
