# NeuroBräu Go

An idiomatic Go implementation of the NeuroBräu educational LLM command-line
client.

- Module: `github.com/neurobrau/llm-cli-go`
- Executable: `neurobrau`
- Version: `0.1.0-dev` (bootstrap)

## Relationship to the Python edition

The Python implementation, version 0.3.0
([`neurobrau/llm-cli-python`](https://github.com/neurobrau/llm-cli-python)),
defines the reference behavior. The language-neutral specification is copied
unchanged into [`docs/tui-spec.md`](docs/tui-spec.md); it governs the shared
JSON formats, file locations, provider request mappings, usage/pricing rules
and exit codes.

The Go edition is not a line-by-line translation, and its version numbers are
independent. **`0.1.0-dev` does not have feature parity with Python 0.3.0.**

## Current status (Go Task 004A — streaming one-shot prompts)

Working today: one-shot prompts (optionally **streamed** with `--stream`)
and a non-streaming, line-oriented **interactive chat** with **OpenAI** (Responses API, the
default) or a **local Ollama** server, with generation settings, saved
settings (read only) and conversations saved in the Python edition's
formats.

```bash
neurobrau "What is a neural network?"                 # OpenAI, gpt-6-luna
neurobrau --provider openai --model gpt-6-luna "Hello"
neurobrau --provider ollama --model qwen3.5:4b "Hello"
neurobrau --provider ollama "Hello"                   # first local model in `ollama list` order
neurobrau --reasoning-effort high --max-output-tokens 800 --answer-style brief --usage "Hello"
```

The answer goes to stdout with one final newline; errors, notes and token
usage go to stderr. Reasoning/thinking text is never printed as part of the
answer. Flags may come before or after the prompt.

### Generation flags

| Flag | Values | Without the flag |
|---|---|---|
| `--reasoning-effort EFFORT` | `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`, `model-default` | `none` where the model supports it (Ollama: thinking off) |
| `--max-output-tokens N` | positive integer | the model's default |
| `--answer-style STYLE` | `brief`, `normal`, `detailed` | `normal` (no extra instruction) |
| `--usage` | — | no usage line |
| `--stream` | — | the answer is printed when it is complete |

- **Reasoning, OpenAI:** a built-in table (from the Python reference) lists
  the confirmed efforts and output limits of known models; dated snapshots
  (`-YYYY-MM-DD`) use their base model's entry. `gpt-6-luna` gets
  `reasoning.effort = "none"` by default (its own default would be
  `medium`). Known models without `none` and non-reasoning models get no
  reasoning parameter; unknown models get none by default, and an explicit
  effort is passed through for the API to judge. `model-default` sends no
  reasoning parameter. If the API rejects the *built-in* default, the
  request is retried once without reasoning and the model is remembered for
  the run; a rejected *explicit* effort is reported (exit 1).
- **Reasoning, Ollama:** `none` turns thinking off and any other effort turns
  it on for models that only switch thinking off/on; models with named
  levels (from `/api/show`) take the level name. `model-default` is not
  available on off/on models. Thinking on a model without the capability is
  rejected.
- Values a model is known not to support, and `--max-output-tokens` above a
  known limit (OpenAI table; Ollama `context_length`), fail with exit 2
  before any chat request.
- Answer styles are sent as OpenAI `instructions` or a leading Ollama
  system message for this request only; `normal` adds nothing.
- `--usage` prints the reported counts, e.g.
  `Token usage: input=100, output=25, total=125`, or
  `Token usage: not reported`. Nothing is estimated.
- One-shot prompts use the flags and built-in defaults only: they neither
  read `settings.json` nor read or write conversation history. (The Python
  edition also applies saved settings to one-shot prompts.)

### Streaming (`--stream`)

```bash
neurobrau --stream "Explain transformers."
neurobrau --provider ollama --model qwen3.5:4b --stream --usage "Hello"
neurobrau "Explain neurons." --stream
```

The answer is written to stdout piece by piece as it is generated, then a
final newline. Reasoning/thinking is never printed (Ollama `thinking` and a
leading `<think>…</think>` block are hidden). Generation settings are the
same as without `--stream`. One-shot streaming never reads or writes
history or settings.

| Outcome | stderr | Exit |
|---|---|---|
| completed | with `--usage`: `Token usage: …`, and for Ollama `Duration: 2.50 s (load 1.00 s, generation 1.20 s)` | 0 |
| stopped early (token limit, content filter) | `neurobrau: warning: response incomplete: <reason>.` (+ usage with `--usage`) | 8 |
| failed, also after part of the answer | `neurobrau: error: …` (partial text stays on stdout) | 1, 3–7 by kind |
| Ctrl+C | `neurobrau: interrupted` (the connection is closed at once) | 130 |
| stdout closed by the reader (`\| head`) | nothing | 141 (SIGPIPE) |

- There is no time limit for a whole answer. A stream fails with a
  timeout (exit 6) only when no data arrives for 300 s (Ollama, including
  loading the model) or 600 s (OpenAI); time spent writing output does not
  count.
- OpenAI: up to 2 SDK retries before the stream starts (connection errors,
  HTTP 408/409/429/5xx), never after text arrived. If the API rejects the
  built-in reasoning default before any text, the request is sent once
  more without it; a rejected `--reasoning-effort` is reported.
- A stream that ends without its final event is an error (OpenAI: exit 1;
  Ollama: connection, exit 7).
- `--stream` without a prompt exits 2: the interactive chat does not stream
  yet (Task 004B).

### Interactive chat

```bash
neurobrau                          # plain chat (temporary: no fullscreen UI yet)
neurobrau --plain                  # plain chat
neurobrau --provider ollama --model qwen3.5:4b --plain
neurobrau --plain --no-history     # nothing is restored or saved
```

Without a prompt, `neurobrau` starts the line-oriented chat. In the Python
edition (and later in Go, Task 005) a terminal gets the fullscreen UI;
until then **every interactive start is the plain chat**, also on a
terminal. Redirected stdin/stdout work too (`printf 'Hi\n' | neurobrau`).

```text
NeuroBräu Go CLI 0.1.0-dev
Provider: openai | Model: gpt-6-luna
Type /help for commands, /exit to quit.

YOU > What is a neural network?
AI > A neural network is …
YOU >
```

Each request sends the whole saved conversation plus the new message. Only
a completed answer is added to the conversation (and saved); after an
error (`error: …` on stderr) or Ctrl+C the chat goes on with the
conversation unchanged. An answer that stops early (e.g. at the token
limit) is shown with `warning: Response incomplete: … The partial response
was not saved.` and is not part of the conversation; its reported usage is
kept as an `incomplete` record. `--usage` prints the token usage after each
answer.

| Input | Effect |
|---|---|
| `/help` | commands and keys |
| `/reset` | start a new conversation; the previous one stays saved (an empty conversation is reused) |
| `/cleanup` | delete all saved conversations after you type `yes` (any other answer, EOF or Ctrl+C cancels) |
| `/usage` | the conversation's token usage as reported by the API (no cost estimate yet) |
| `/exit` | leave the chat (exit code 0) |
| any other `/word` | `Unknown command: /word. Type /help for commands.` |
| Ctrl+D (EOF) / Ctrl+C at the prompt | exit with code 0 / 130 |
| Ctrl+C while waiting for an answer | cancel the request, back to the prompt |

Commands are a single word starting with `/` or `\` (`\reset` works too),
in any case; `/usr/bin is what?` is an ordinary message. Input is read
line by line: there is no in-app line editing or input history beyond
what the terminal itself offers.

#### Conversation history

| File | Location |
|---|---|
| conversations | `$XDG_STATE_HOME/neurobrau/conversations/<id>.json`, else `~/.local/state/neurobrau/conversations/` |
| active provider/model | `$XDG_STATE_HOME/neurobrau/active.json`, else `~/.local/state/neurobrau/active.json` |
| settings | `$XDG_CONFIG_HOME/neurobrau/settings.json`, else `~/.config/neurobrau/settings.json` |

XDG variables count only when they are absolute paths. Files are
**plaintext** (directories `0700`, files `0600`, written atomically) and
shared with the Python edition: either can continue the other's
conversations.

- Each provider/model has its own conversations; one is never continued
  with another model. At startup the latest matching conversation is
  restored: without `--provider`/`--model` the selection in `active.json`
  (else the most recent conversation of any provider); with `--provider`
  that provider's latest conversation and its model; with `--model` that
  exact model (provider default `openai`). Older files without a provider
  count as OpenAI; files without a model match any model.
- The conversation is saved after every completed answer, and
  `active.json` records the selection. Restoring alone writes nothing.
- Invalid or unreadable files are skipped with a warning and never changed;
  a failed save is a warning and the chat goes on with the conversation in
  memory.
- `--no-history`: no conversation or `active.json` file is read, written,
  created or deleted, and no permissions are changed; the chat, `/reset`
  and `/usage` work in memory.

#### Settings in the chat

`settings.json` (written by the Python edition's settings menus) is read at
chat startup; the Go chat never writes it. For the active provider/model:

1. command-line flags (`--reasoning-effort`, `--max-output-tokens`,
   `--answer-style`) for this run only; a value the model does not support
   exits with code 2 before the chat starts;
2. the saved per-model `reasoning` / `max_output_tokens` and the global
   `answer_style`; a saved value the model does not support is ignored with
   a `warning:` (e.g. `Saved reasoning 'High' is not available for …`);
3. the built-in defaults above.

An invalid `settings.json` gives a warning, the defaults are used, and the
file is left unchanged.

### OpenAI

- Uses the official OpenAI Go SDK and the Responses API, always with
  `store: false`. Default model: `gpt-6-luna`; `--model` accepts any model ID
  the API accepts.
- The API key is read from the `OPENAI_API_KEY` environment variable only.
  Set it in your shell before running a prompt; it is never written to disk
  or printed. Without it, OpenAI prompts and an OpenAI chat fail with exit
  code 3 (the chat before it starts); `--help`, `--version` and Ollama do
  not need it.
- The SDK also honors `OPENAI_BASE_URL` for a different endpoint.
- Chat requests: up to 2 SDK retries after connection errors or HTTP
  408/409/429/5xx, 600 s per attempt; timeouts and Ctrl+C are not retried.
- A non-streaming response that stops early (e.g. at the token limit) is
  printed as is and exits 0, as in the Python edition; a completed response
  with no text prints an empty line. Failed, refused and unfinished
  responses are errors (exit 1).
- Error text from the API is shortened, and anything that looks like an API
  key is replaced with `[redacted]`.

### Ollama

- Server: `OLLAMA_HOST` (`host`, `host:port` or URL), default
  `http://127.0.0.1:11434`; `0.0.0.0` means the local machine. No key needed.
- Only locally installed models are used. Cloud/remote models
  (`:cloud`, `-cloud`, `remote_host`, `remote_model`) are never selected,
  queried or sent requests; models without the `completion` capability are
  never chosen as the default. Nothing is downloaded: a missing model gives
  `Install it yourself with: ollama pull <model>`.
- Thinking models are asked not to think (`think: false`) unless
  `--reasoning-effort` says otherwise.

### Exit codes

| Invocation / outcome | Output | Exit code |
|---|---|---|
| `--help`, `-h` | usage on stdout | 0 |
| chat: `/exit`, Ctrl+D | — | 0 |
| chat: Ctrl+C at the prompt | `Goodbye.` | 130 |
| `--version` | `NeuroBräu Go CLI 0.1.0-dev` on stdout | 0 |
| successful prompt (also a non-streaming answer cut off at the token limit) | answer on stdout; with `--usage`, usage on stderr | 0 |
| `--stream` answer cut off (token limit, content filter) | partial answer; `neurobrau: warning: response incomplete: …` | 8 |
| stdout closed by the reader (e.g. `\| head`) | — | 141 (SIGPIPE) |
| API error, invalid reply, failed/refused/empty response, missing or cloud Ollama model | `neurobrau: error: …` | 1 |
| invalid arguments, unknown provider, empty prompt, setting the model does not support | `neurobrau: error: …` | 2 |
| `OPENAI_API_KEY` missing | `neurobrau: error: …` | 3 |
| authentication failed (HTTP 401/403) | `neurobrau: error: …` | 4 |
| rate limit or quota exceeded (HTTP 429) | `neurobrau: error: …` | 5 |
| timeout | `neurobrau: error: …` | 6 |
| connection failed | `neurobrau: error: …` | 7 |
| Ctrl+C during a one-shot request or chat startup | `neurobrau: interrupted` | 130 |

## Requirements

- Go 1.26 or newer (see `go.mod`)
- One direct dependency: the official OpenAI Go SDK
  (`github.com/openai/openai-go/v3`); no CGO

## Build, test, run

```bash
go build -o bin/neurobrau ./cmd/neurobrau
./bin/neurobrau --help
./bin/neurobrau --version

go test ./...      # no network, credentials or user state needed
go test -race ./...
go vet ./...
gofmt -l .        # must print nothing
```

## Layout

```text
cmd/neurobrau/   thin entry point: passes os.Args, stdout, stderr to cli.Run
internal/cli/    argument parsing, help, version, one-shot flow, chat startup,
                 exit codes
internal/chat/   plain chat loop, commands, generation settings resolution,
                 /usage report
internal/session/ active conversation: context, commit, restore, reset,
                 switch/park, cleanup (no UI, no network)
internal/llm/    provider-neutral types, Provider and StreamingProvider
                 interfaces, stream events, error kinds (llmtest: test helpers)
internal/ollama/ Ollama provider over net/http (/api/tags, /api/show, /api/chat)
internal/openai/ OpenAI provider over the official SDK (Responses API, Models API)
internal/storage/ Python-compatible JSON files: conversations, usage records,
                 active.json, settings.json
scripts/         development helpers (optional Python storage parity check)
docs/tui-spec.md authoritative, language-neutral specification (do not edit)
AGENTS.md        instructions for coding agents working on this repository
```

## Storage

`internal/storage` reads and writes the conversation, `active.json` and
`settings.json` formats shared with the Python edition (XDG locations,
atomic `0600` writes, `0700` app directories); the interactive chat uses
it through `internal/session`. See [`docs/storage.md`](docs/storage.md) for
the API, the Python fixtures and the optional parity checks.

## Roadmap (future — not implemented)

1. Streaming in the interactive chat — Task 004B.
2. Fullscreen retro TUI with provider/model menus and settings controls
   (F2–F6) — Task 005. The session layer already supports switching and
   parking conversations per provider/model.
3. Estimated cost in `/usage` (pricing catalog).

Anthropic support, LangChain/LangGraph, RAG, vector databases and autonomous
agents are out of scope for Phase 1.
