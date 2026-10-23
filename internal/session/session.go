// Package session holds the conversation of an interactive chat: its
// committed messages, usage records and provider/model, and keeps them in
// the history store (docs/tui-spec.md, sections 11 and 15).
//
// A Session prints nothing and makes no network requests; the chat UI sends
// requests and reports the warnings and errors its methods return. Its
// methods may be called from several goroutines (e.g. a future TUI reading
// usage while a request runs); they are serialized by a mutex. The caller
// keeps at most one request in flight.
package session

import (
	"slices"
	"sync"
	"time"

	"github.com/neurobrau/llm-cli-go/internal/llm"
	"github.com/neurobrau/llm-cli-go/internal/storage"
)

// knownProviders are the providers a conversation can be continued with.
var knownProviders = []string{storage.ProviderOpenAI, storage.ProviderOllama}

// Options configure Open.
type Options struct {
	// Store keeps the conversations; nil disables history (--no-history):
	// nothing is read, written, deleted or created.
	Store *storage.HistoryStore
	// Provider and Model are the selection requested on the command line;
	// "" if not given.
	Provider string
	Model    string
	// Clock returns the current time; default time.Now.
	Clock func() time.Time
}

// Session is the active conversation and the conversations set aside
// (parked) for other selections in this run.
type Session struct {
	mu     sync.Mutex
	store  *storage.HistoryStore
	clock  func() time.Time
	conv   storage.Conversation // Provider/Model: the selection ("" until known)
	onDisk bool                 // conv has been saved or was loaded from a file
	// restored is set when Open continued a saved, non-empty conversation.
	restored    bool
	parked      map[storage.Selection]parkedConversation
	activeSaved storage.Selection // last selection written to active.json
}

type parkedConversation struct {
	conv   storage.Conversation
	onDisk bool
}

// Open restores the latest saved conversation for the requested selection,
// or starts a new one (written at its first completed exchange). It
// returns the startup warnings (invalid files, an invalid active.json, I/O
// errors); restoring never rewrites a file.
//
// Without Provider and Model the selection in active.json is used, else
// any conversation. With Provider only, that provider's latest conversation
// and its model are adopted. With Model, the caller passes the provider too
// (the default when not given), and only that exact pair matches. Files
// without a provider count as OpenAI; files without a model match any model
// and take the requested one. Conversations of providers this edition does
// not know are skipped.
func Open(opts Options) (*Session, []string) {
	s := &Session{
		store:  opts.Store,
		clock:  opts.Clock,
		parked: map[storage.Selection]parkedConversation{},
	}
	if s.clock == nil {
		s.clock = time.Now
	}
	provider, model := opts.Provider, opts.Model
	var warnings []string
	if s.store != nil && provider == "" && model == "" {
		sel, found, warning := s.store.LoadActive()
		if warning != "" {
			warnings = append(warnings, warning)
		}
		if found {
			provider, model = sel.Provider, sel.Model
		}
	}
	s.conv.Provider, s.conv.Model = provider, model
	latest, found := storage.Conversation{}, false
	if s.store != nil {
		var loadWarnings []string
		latest, found, loadWarnings = s.store.LoadLatest(matcher(provider, model))
		warnings = append(warnings, loadWarnings...)
	}
	if found {
		s.adopt(latest)
		s.restored = len(s.conv.Messages) > 0
	} else {
		s.startNew()
	}
	return s, warnings
}

// matcher accepts conversations held with provider and model ("" = any) by
// a known provider.
func matcher(provider, model string) func(storage.Conversation) bool {
	return func(c storage.Conversation) bool {
		return slices.Contains(knownProviders, c.EffectiveProvider()) && c.Matches(provider, model)
	}
}

// newID returns a new conversation ID.
func newID() string {
	id, err := storage.NewID()
	if err != nil {
		panic(err) // crypto/rand.Read never fails since Go 1.24
	}
	return id
}

// startNew replaces the conversation with a new, empty, unsaved one for
// the same selection.
func (s *Session) startNew() {
	now := s.clock()
	s.conv = storage.Conversation{
		ID: newID(), CreatedAt: now, UpdatedAt: now,
		Provider: s.conv.Provider, Model: s.conv.Model,
	}
	s.onDisk = false
}

// adopt continues a stored conversation. A missing provider is OpenAI; a
// missing model is the current one.
func (s *Session) adopt(c storage.Conversation) {
	model := s.conv.Model
	s.conv = c
	s.conv.Provider = c.EffectiveProvider()
	if s.conv.Model == "" {
		s.conv.Model = model
	}
	s.onDisk = true
}

// SetDefaults fills in the provider and model of a conversation that has
// none yet (no flag, no saved selection, or a legacy file without a model).
// Fields already known are kept. Nothing is saved.
func (s *Session) SetDefaults(provider, model string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.conv.Provider == "" {
		s.conv.Provider = provider
	}
	if s.conv.Model == "" {
		s.conv.Model = model
	}
}

// Persistent reports whether history is enabled.
func (s *Session) Persistent() bool { return s.store != nil }

// HistoryDir returns the conversations directory ("" without history).
func (s *Session) HistoryDir() string {
	if s.store == nil {
		return ""
	}
	return s.store.Dir()
}

// Selection returns the provider and model of the conversation.
func (s *Session) Selection() storage.Selection {
	s.mu.Lock()
	defer s.mu.Unlock()
	return storage.Selection{Provider: s.conv.Provider, Model: s.conv.Model}
}

// Restored reports whether Open continued a saved, non-empty conversation.
func (s *Session) Restored() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.restored
}

// Conversation returns a copy of the current conversation.
func (s *Session) Conversation() storage.Conversation {
	s.mu.Lock()
	defer s.mu.Unlock()
	c := s.conv
	c.Messages = slices.Clone(c.Messages)
	c.Requests = slices.Clone(c.Requests)
	return c
}

// Messages returns a copy of the committed messages, oldest first.
func (s *Session) Messages() []llm.Message {
	s.mu.Lock()
	defer s.mu.Unlock()
	return slices.Clone(s.conv.Messages)
}

// Requests returns a copy of the usage records.
func (s *Session) Requests() []storage.UsageRecord {
	s.mu.Lock()
	defer s.mu.Unlock()
	return slices.Clone(s.conv.Requests)
}

// Context returns what a request for text sends: the committed messages
// plus text as a new user message. Nothing is committed.
func (s *Session) Context(text string) []llm.Message {
	s.mu.Lock()
	defer s.mu.Unlock()
	msgs := make([]llm.Message, 0, len(s.conv.Messages)+1)
	msgs = append(msgs, s.conv.Messages...)
	return append(msgs, llm.Message{Role: llm.RoleUser, Content: text})
}

// Commit adds a completed exchange (text and the answer in resp) and its
// reported usage, then saves the conversation. A save error is returned as
// a warning: the exchange stays in memory.
func (s *Session) Commit(text string, resp llm.Response) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.conv.Messages = append(s.conv.Messages,
		llm.Message{Role: llm.RoleUser, Content: text},
		llm.Message{Role: llm.RoleAssistant, Content: resp.Text})
	s.recordUsage(resp, storage.StatusCompleted)
	return s.save()
}

// RecordIncomplete keeps the reported usage of an answer that stopped
// early: it was billed, but it is not added to the messages. The
// conversation is saved if it already has messages or a file. Without
// reported usage nothing happens.
func (s *Session) RecordIncomplete(resp llm.Response) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if resp.Usage == nil {
		return nil
	}
	s.recordUsage(resp, storage.StatusIncomplete)
	if len(s.conv.Messages) == 0 && !s.onDisk {
		return nil
	}
	return s.save()
}

// recordUsage adds one usage record, only if usage was reported.
func (s *Session) recordUsage(resp llm.Response, status string) {
	if resp.Usage == nil {
		return
	}
	s.conv.Requests = append(s.conv.Requests, storage.UsageRecord{
		At: s.clock(), Provider: s.conv.Provider, Model: s.conv.Model,
		Status: status, ServiceTier: resp.ServiceTier, Usage: *resp.Usage,
	})
}

// Reset starts a new, empty conversation for the same selection and saves
// it right away, so it is the one restored next time; the previous file is
// left unchanged. An empty conversation (no messages and no usage) is
// reused. A save error is returned as a warning.
func (s *Session) Reset() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.conv.Messages) == 0 && len(s.conv.Requests) == 0 {
		return nil
	}
	s.startNew()
	return s.save()
}

// Switch continues with another provider/model. The current conversation
// is parked (it is already saved after each exchange); the target's
// conversation comes back from this run's parked ones, else the latest
// matching file, which is not rewritten. Without either, a new
// conversation starts and is saved right away (empty) so the selection is
// restored next time. History never moves between selections. A save
// error is returned as a warning.
func (s *Session) Switch(provider, model string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	target := storage.Selection{Provider: provider, Model: model}
	current := storage.Selection{Provider: s.conv.Provider, Model: s.conv.Model}
	if target == current {
		return nil
	}
	if len(s.conv.Messages) > 0 || len(s.conv.Requests) > 0 {
		s.parked[current] = parkedConversation{s.conv, s.onDisk}
	}
	if p, ok := s.parked[target]; ok {
		delete(s.parked, target)
		s.conv, s.onDisk = p.conv, p.onDisk
	} else if stored, found := s.loadLatest(target); found {
		s.conv.Model = model // for a stored file without a model
		s.adopt(stored)
	} else {
		// An empty conversation that is not on disk yet is reused; anything
		// else (e.g. the empty one Reset saved) stays with its selection.
		if len(s.conv.Messages) > 0 || len(s.conv.Requests) > 0 || s.onDisk {
			s.startNew()
		}
		s.restored = false
		s.conv.Provider, s.conv.Model = provider, model
		return s.save()
	}
	s.restored = len(s.conv.Messages) > 0
	s.conv.Provider, s.conv.Model = provider, model
	return s.saveActive()
}

func (s *Session) loadLatest(sel storage.Selection) (storage.Conversation, bool) {
	if s.store == nil {
		return storage.Conversation{}, false
	}
	// Warnings about invalid files were shown at startup.
	c, found, _ := s.store.LoadLatest(matcher(sel.Provider, sel.Model))
	return c, found
}

// SavedCount returns the number of conversation files (0 without
// history), e.g. to ask before Cleanup.
func (s *Session) SavedCount() (int, error) {
	if s.store == nil {
		return 0, nil
	}
	return s.store.CountConversations()
}

// Cleanup deletes all saved conversation files, forgets the parked
// conversations and starts a new, unsaved conversation. Only call it after
// the user confirmed. active.json, settings and other files are kept. On
// an error, the files deleted so far are reported in it and the session is
// unchanged.
func (s *Session) Cleanup() (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.store == nil {
		return 0, nil
	}
	n, err := s.store.DeleteAll()
	if err != nil {
		return n, err
	}
	clear(s.parked)
	s.startNew()
	s.restored = false
	return n, nil
}

// save writes the conversation with a new updated_at, then active.json if
// the selection changed. Without history it does nothing.
func (s *Session) save() error {
	if s.store == nil {
		return nil
	}
	s.conv.UpdatedAt = s.clock()
	if err := s.store.Save(s.conv); err != nil {
		return err
	}
	s.onDisk = true
	return s.saveActive()
}

// saveActive records the selection in active.json when it changed.
func (s *Session) saveActive() error {
	sel := storage.Selection{Provider: s.conv.Provider, Model: s.conv.Model}
	if s.store == nil || sel.Provider == "" || sel.Model == "" || sel == s.activeSaved {
		return nil
	}
	if err := s.store.SaveActive(sel); err != nil {
		return err
	}
	s.activeSaved = sel
	return nil
}
