package storage

import (
	"crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"time"

	"github.com/neurobrau/llm-cli-go/internal/llm"
)

// Providers that new records may name.
const (
	ProviderOpenAI = "openai"
	ProviderOllama = "ollama"
	// LegacyProvider is the provider of files written before provider
	// selection existed; they have no provider field.
	LegacyProvider = ProviderOpenAI
)

// Conversation is a conversation as stored in `<id>.json`.
type Conversation struct {
	ID        string // canonical lowercase UUID; also the file name
	CreatedAt time.Time
	UpdatedAt time.Time
	Provider  string // "" if not recorded (older files: OpenAI)
	Model     string // "" if not recorded (matches any model of the provider)
	Messages  []llm.Message
	Requests  []UsageRecord
}

// EffectiveProvider returns the provider, counting a missing one as OpenAI.
func (c Conversation) EffectiveProvider() string {
	if c.Provider == "" {
		return LegacyProvider
	}
	return c.Provider
}

// Matches reports whether the conversation was held with provider and
// model ("" matches any). A file without a provider is OpenAI; one without
// a model matches every model of its provider.
func (c Conversation) Matches(provider, model string) bool {
	return (provider == "" || c.EffectiveProvider() == provider) &&
		(model == "" || c.Model == "" || c.Model == model)
}

var uuidPattern = regexp.MustCompile(`^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$`)

// ValidID reports whether id is a canonical lowercase UUID, which also
// keeps it safe as a file name.
func ValidID(id string) bool { return uuidPattern.MatchString(id) }

// NewID returns a random (version 4) UUID.
func NewID() (string, error) {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		return "", err
	}
	b[6] = b[6]&0x0f | 0x40 // version 4
	b[8] = b[8]&0x3f | 0x80 // RFC 4122 variant
	return fmt.Sprintf("%x-%x-%x-%x-%x", b[0:4], b[4:6], b[6:8], b[8:10], b[10:16]), nil
}

const timestampLayout = "2006-01-02T15:04:05.000Z07:00"

// formatTimestamp writes RFC 3339 UTC with millisecond precision, e.g.
// 2026-10-21T12:00:00.000Z.
func formatTimestamp(t time.Time) string { return t.UTC().Format(timestampLayout) }

// parseTimestamp reads an RFC 3339 timestamp with any time zone offset and
// returns it in UTC.
func parseTimestamp(o object, field string) (time.Time, error) {
	value, ok := o.str(field)
	if !ok {
		return time.Time{}, fmt.Errorf("'%s' must be a string", field)
	}
	t, err := time.Parse(time.RFC3339Nano, value)
	if err != nil {
		if _, local := time.Parse("2006-01-02T15:04:05.999999999", value); local == nil {
			return time.Time{}, fmt.Errorf("'%s' has no time zone", field)
		}
		return time.Time{}, fmt.Errorf("'%s' is not an RFC 3339 timestamp", field)
	}
	return t.UTC(), nil
}

type messageJSON struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// conversationJSON fixes the key order of a written file.
type conversationJSON struct {
	SchemaVersion int               `json:"schema_version"`
	ID            string            `json:"id"`
	CreatedAt     string            `json:"created_at"`
	UpdatedAt     string            `json:"updated_at"`
	Provider      string            `json:"provider,omitempty"`
	Model         string            `json:"model,omitempty"`
	Messages      []messageJSON     `json:"messages"`
	Requests      []usageRecordJSON `json:"requests,omitempty"`
}

// EncodeConversation returns the file content: UTF-8 JSON, 2-space indent,
// keys in the specified order, trailing newline. Provider and model are
// omitted when unknown, requests when empty.
func EncodeConversation(c Conversation) ([]byte, error) {
	if !ValidID(c.ID) {
		return nil, fmt.Errorf("invalid conversation id %q", c.ID)
	}
	if c.Provider != "" && c.Provider != ProviderOpenAI && c.Provider != ProviderOllama {
		return nil, fmt.Errorf("unknown provider %q", c.Provider)
	}
	out := conversationJSON{
		SchemaVersion: SchemaVersion,
		ID:            c.ID,
		CreatedAt:     formatTimestamp(c.CreatedAt),
		UpdatedAt:     formatTimestamp(c.UpdatedAt),
		Provider:      c.Provider,
		Model:         c.Model,
		Messages:      make([]messageJSON, 0, len(c.Messages)),
	}
	for i, m := range c.Messages {
		if m.Role != llm.RoleUser && m.Role != llm.RoleAssistant {
			return nil, fmt.Errorf("message %d has invalid role %q", i, m.Role)
		}
		out.Messages = append(out.Messages, messageJSON{Role: string(m.Role), Content: m.Content})
	}
	for _, r := range c.Requests {
		out.Requests = append(out.Requests, r.toJSON())
	}
	return marshalIndent(out)
}

// DecodeConversation validates a conversation file. The error describes the
// first problem found, as in the Python reference. Unknown fields are
// ignored; malformed usage records are skipped.
func DecodeConversation(data []byte) (Conversation, error) {
	o, err := decodeObject(data)
	if err != nil {
		return Conversation{}, err
	}
	if err := checkSchemaVersion(o); err != nil {
		return Conversation{}, err
	}
	var c Conversation
	if id, ok := o.str("id"); ok && ValidID(id) {
		c.ID = id
	} else {
		return Conversation{}, errors.New("'id' must be a UUID string")
	}
	if c.Messages, err = parseMessages(o); err != nil {
		return Conversation{}, err
	}
	if c.CreatedAt, err = parseTimestamp(o, "created_at"); err != nil {
		return Conversation{}, err
	}
	if c.UpdatedAt, err = parseTimestamp(o, "updated_at"); err != nil {
		return Conversation{}, err
	}
	if c.Provider, err = optionalString(o, "provider"); err != nil {
		return Conversation{}, err
	}
	if c.Model, err = optionalString(o, "model"); err != nil {
		return Conversation{}, err
	}
	var requests []json.RawMessage
	if kind(o["requests"]) == '[' && json.Unmarshal(o["requests"], &requests) == nil {
		for _, raw := range requests {
			if r, ok := parseUsageRecord(raw); ok {
				c.Requests = append(c.Requests, r)
			}
		}
	}
	return c, nil
}

// checkSchemaVersion accepts the number 1. Like Python, 1.0 is also 1.
func checkSchemaVersion(o object) error {
	raw := o["schema_version"]
	var v float64
	if k := kind(raw); (k == '-' || k >= '0' && k <= '9') && json.Unmarshal(raw, &v) == nil && v == SchemaVersion {
		return nil
	}
	return fmt.Errorf("unsupported schema_version %s (expected %d)", pyRepr(raw), SchemaVersion)
}

func parseMessages(o object) ([]llm.Message, error) {
	var items []json.RawMessage
	if kind(o["messages"]) != '[' || json.Unmarshal(o["messages"], &items) != nil {
		return nil, errors.New("'messages' must be a list")
	}
	messages := make([]llm.Message, 0, len(items))
	for i, raw := range items {
		m, ok := asObject(raw)
		if !ok {
			return nil, fmt.Errorf("message %d must be an object", i)
		}
		role, _ := m.str("role")
		if role != string(llm.RoleUser) && role != string(llm.RoleAssistant) {
			return nil, fmt.Errorf("message %d has invalid role %s", i, pyRepr(m["role"]))
		}
		content, ok := m.str("content")
		if !ok {
			return nil, fmt.Errorf("message %d content must be a string", i)
		}
		messages = append(messages, llm.Message{Role: llm.Role(role), Content: content})
	}
	return messages, nil
}

// optionalString reads a field that may be missing or null; if present it
// must be a non-empty string.
func optionalString(o object, field string) (string, error) {
	if !o.present(field) {
		return "", nil
	}
	if s, ok := o.str(field); ok && s != "" {
		return s, nil
	}
	return "", fmt.Errorf("'%s' must be a non-empty string", field)
}
