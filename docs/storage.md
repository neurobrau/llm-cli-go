# Storage layer (`internal/storage`)

The Go implementation of NeuroBräu's local files, compatible with the Python
0.3.0 edition. The formats are defined in [`tui-spec.md`](tui-spec.md)
sections 11 (conversations, `active.json`), 13 (`settings.json`) and 15
(usage records); this page describes the Go API and its guarantees.

> Used by the interactive chat through `internal/session` (restore at
> startup, save after each completed answer, `/reset`, `/cleanup`);
> settings are only read. One-shot prompts neither read nor write any of
> these files.

## Files

| File | Location | Written |
|---|---|---|
| conversations | `$XDG_STATE_HOME/neurobrau/conversations/<id>.json` (absolute XDG only, else `~/.local/state/...`) | 2-space indented JSON, keys in spec order, trailing newline |
| active selection | `$XDG_STATE_HOME/neurobrau/active.json`, next to `conversations/` | one line, `{"schema_version": 1, "provider": …, "model": …}` |
| settings | `$XDG_CONFIG_HOME/neurobrau/settings.json` (absolute XDG only, else `~/.config/...`) | 2-space indented JSON, providers/models sorted by name |

All writes are atomic: a temp file `.<name>.<random>.tmp` (mode `0600`) in
the same directory is written, fsynced and renamed over the target, then the
directory is fsynced; the temp file is removed on any error. The app
directories (`neurobrau`, `conversations`) are created or tightened to
`0700`; directories above them are never changed.

## API

```go
paths, err := storage.ResolvePaths(os.Getenv, home) // Settings, Conversations, Active

history := storage.NewHistoryStore(paths.Conversations)
latest, found, warnings := history.LoadLatest(func(c storage.Conversation) bool {
    return c.Matches(provider, model) // "" = any; no provider = openai; no model = any model
})
err = history.Save(conversation)               // <id>.json, atomic, 0600
sel, found, warning := history.LoadActive()    // active.json; invalid → warning, file kept
err = history.SaveActive(storage.Selection{Provider: "ollama", Model: "qwen3.5:4b"})
n, err := history.CountConversations()         // for a /cleanup confirmation
n, err = history.DeleteAll()                   // only *.json files in conversations/

settingsStore := storage.NewSettingsStore(paths.Settings)
settings, warnings := settingsStore.Load()     // invalid file → defaults, store not writable
err = settingsStore.Save(settings)             // refuses to overwrite a file it could not load

id, err := storage.NewID()                     // random version-4 UUID
data, err := storage.EncodeConversation(c)     // also DecodeConversation, EncodeSettings, DecodeSettings
```

- `Conversation` holds `llm.Message`s (user/assistant only) and
  `UsageRecord`s whose counts are `llm.Usage` (nil = not reported).
- `ModelPreferences` keeps the settings semantics: `Reasoning` `""` = not
  set, `"default"` = send no parameter; `MaxOutputTokensSet` with
  `MaxOutputTokens` 0 = JSON `null` (model default), absent = not set.
- Nothing runs implicitly. A caller that must not touch history
  (`--no-history`) simply never creates a `HistoryStore`. `LoadLatest`
  writes no file; like the Python edition it tightens existing app
  directories to `0700`.
- Invalid files are reported as warnings
  (`Ignoring unreadable conversation file <path>: <reason>. The file was left unchanged.`)
  and never modified; I/O errors read
  `Could not <read|save|delete> conversation history at <path>: <reason>.`

## Python interoperability

`internal/storage/testdata/python/` holds files written by the Python 0.3.0
serializers from synthetic data. Go reads them and writes them back byte for
byte (`fixtures_test.go`). They were generated with:

```bash
python3 -B scripts/python_storage_parity.py generate ../llm-cli-python/src internal/storage/testdata/python
```

The reverse direction (Python reading Go-written files, including a
Python conversation that Go continued, recorded usage for and reset) is an
optional development check, skipped by `go test ./...`:

```bash
NEUROBRAU_PYTHON_SRC=$PWD/../llm-cli-python/src go test -run PythonReads ./internal/storage ./internal/session
```

Both use temp directories only and never write into the Python repository
(bytecode caching is disabled).

### Known differences

| Case | Python 0.3.0 | Go |
|---|---|---|
| Timestamps | `datetime.fromisoformat` also accepts non-RFC 3339 forms (space instead of `T`, basic format) | strict RFC 3339 (any offset, any fraction digits) |
| `"schema_version": true` | accepted (`True == 1`) | rejected; `1.0` is accepted by both |
| Integers beyond 64 bits in counts/limits | accepted | rejected (record or entry skipped) |
| `NaN`/`Infinity` literals | parsed | invalid JSON |
| U+2028/U+2029 in text | written as is | written as ` `/` ` (same JSON value) |
| `active.json` with an unknown provider | read; the CLI then silently uses defaults | warning `Ignoring invalid <path>.` (the spec only says it is ignored) |
| `provider` in a conversation | any non-empty string is read and written | any non-empty string is read; only `openai`/`ollama` are written |
| `DeleteAll` with a directory named `*.json` | fails on it | skips it |
| Warning order for invalid settings entries | file order | sorted by provider/model |
| Error reasons for malformed JSON | Python's decoder messages | Go's decoder messages |

Spec vs. Python source: the spec shows `active.json` on one line without
saying so; Go follows the Python writer exactly. The spec says a conversation
`id` is also its file name; neither implementation checks that on reading.
