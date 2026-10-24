# NeuroBräu Go

An idiomatic Go implementation of the NeuroBräu educational LLM command-line
client.

- Module: `github.com/neurobrau/llm-cli-go`
- Executable: `neurobrau`
- Version: `0.4.0-dev` (release candidate of the 0.4 milestone; see the
  [release checklist](docs/release-checklist.md))

## Relationship to the Python edition

The Python implementation, version 0.3.0
([`neurobrau/llm-cli-python`](https://github.com/neurobrau/llm-cli-python)),
defines the reference behavior. The language-neutral specification is copied
unchanged into [`docs/tui-spec.md`](docs/tui-spec.md); it governs the shared
JSON formats, file locations, provider request mappings, usage/pricing rules
and exit codes.

The Go edition is not a line-by-line translation, and its version numbers are
independent. `0.4.0-dev` implements the Python 0.3.0 behavior; the remaining,
documented differences are listed in [`docs/parity.md`](docs/parity.md),
together with the optional checks that compare both editions.

Go development milestones (not releases or tags):

| Milestone | Scope |
|---|---|
| 0.1 | command line and one-shot prompts (OpenAI, Ollama) |
| 0.2 | interactive line-oriented chat with saved history |
| 0.3 | fullscreen DOS-style UI and streaming |
| 0.4 | working F2–F6 menus, saved settings and estimated costs |

## Current status (0.4.0-dev)

Working today: one-shot prompts and an **interactive chat** — the DOS-style
**fullscreen UI** on a terminal, or the line-oriented chat — all optionally
**streamed** with `--stream`, with **OpenAI** (Responses API, the default)
or a **local Ollama** server, with generation settings and conversations
saved in the Python edition's formats. In the fullscreen UI, **F2–F6**
switch model and provider and change the generation settings, which are
saved in `settings.json`. `/usage` and the settings panel show the
conversation's token usage and its **estimated** API cost (see
[Estimated cost](#estimated-cost)).

### Quick start

Build and run from the repository (no installation, no PATH changes):

```bash
go build -o bin/neurobrau ./cmd/neurobrau
./bin/neurobrau --version                                    # NeuroBräu Go CLI 0.4.0-dev
./bin/neurobrau --provider ollama "Say hello in five words." # one-shot
./bin/neurobrau --provider ollama --stream "Count to five."  # one-shot, streamed
./bin/neurobrau --provider ollama --stream                   # fullscreen UI (on a terminal)
./bin/neurobrau --provider ollama --plain                    # line-oriented chat
./bin/neurobrau --provider ollama --no-history               # no conversation files
```

In the fullscreen UI, F2 picks a model, F3 a provider, F4–F6 change
reasoning, max output tokens and answer style. **`--no-history` only stops
conversation files: F4–F6 choices are still saved in `settings.json`.** To
try everything without touching your real history or settings, use
throwaway directories:

```bash
d=$(mktemp -d)
HOME=$d XDG_STATE_HOME=$d/state XDG_CONFIG_HOME=$d/config ./bin/neurobrau --provider ollama --stream
```

OpenAI is the default provider and needs `OPENAI_API_KEY`; its requests are
billed by OpenAI. The examples below write `neurobrau` for `./bin/neurobrau`.

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
- `--stream` without a prompt starts the chat with streamed answers
  (fullscreen on a terminal, see below; plain chat otherwise).

#### Streaming in the plain chat

```bash
neurobrau --plain --stream         # or --stream with redirected stdin/stdout
neurobrau --provider ollama --model qwen3.5:4b --plain --stream --usage
```

`AI > ` appears with the first piece of the answer, then the text as it
arrives (an empty answer still prints `AI > `). Reasoning is never shown.
Only a completed answer is added to the conversation and saved, using the
assembled text; generation settings and saved preferences apply to every
request exactly as without `--stream`.

| Outcome | Shown | Conversation |
|---|---|---|
| completed | the answer; with `--usage`, usage (and Ollama durations) on stderr | exchange committed and saved; one usage record |
| stopped early | partial text, `warning: Response incomplete: <reason>. The partial response was not saved.` | unchanged; one `incomplete` usage record if usage was reported |
| failed | partial text, `error: <message>` (+ the same suffix if text was shown) | unchanged |
| Ctrl+C | `Request cancelled.` (+ the suffix if text was shown); back at `YOU > ` | unchanged; the connection is closed |

Ctrl+C at the prompt still exits with 130; `/exit` and Ctrl+D exit with 0.
A partial, failed or canceled answer is never sent as context later.

### Interactive chat

```bash
neurobrau                          # fullscreen UI on a terminal
neurobrau --stream                 # fullscreen, answers shown while generated
neurobrau --plain                  # line-oriented chat, also on a terminal
neurobrau --provider ollama --model qwen3.5:4b --plain
neurobrau --plain --no-history     # nothing is restored or saved
printf 'Hi\n' | neurobrau           # redirected: line-oriented chat
```

Without a prompt, `neurobrau` starts the **fullscreen UI** when stdin and
stdout are both terminals and `TERM` is set and not `dumb`; otherwise, and
always with `--plain`, it starts the **line-oriented chat**. A prompt
argument is always a one-shot request. `--stream` only decides whether
answers appear while they are generated, not which chat starts. Startup
problems (missing `OPENAI_API_KEY`, unreachable Ollama, an unsupported
setting) are reported before the terminal is switched to fullscreen.

#### Fullscreen UI

```text
+-[ NeuroBräu Go CLI 0.4.0-dev ]--------------------------+-[ LLM SETTINGS ]---+
| YOU > What is a neural network?                         | F2 MODEL           |
|                                                         |  qwen3.5:4b        |
| AI > A neural network is a model made of layers of      | F3 PROVIDER        |
|      connected units that learn weights from data.      |  Ollama            |
 …
+-[ Status ]----------------------------------------------+--------------------+
| DONE | 7.17s total | load 0.20s | generation 1.88s | 15.96 tok/s             |
+-[ Input ]--------------------------------------------------------------------+
| YOU >                                                                        |
+------------------------------------------------------------------------------+
 F1 /help | /reset /exit | F2-F6 settings | PgUp/PgDn scroll | Ctrl+C quit
```

Layout, colors, keys and messages follow `docs/tui-spec.md` (sections
1–9 and 14): the conversation on the left, the **LLM SETTINGS** panel
(model, provider, effective reasoning, max tokens, answer style,
conversation token totals) from 64 columns, a runtime status bar
(`READY`, `CONNECTING`, `GENERATING`, `DONE`, `CANCELLED`, `WARNING`,
`ERROR`) and the input line. Below 40x12 a "Terminal too small" notice is
shown until the window is larger again; nothing typed is lost.

- **Keys:** typing (UTF-8, e.g. `Grüß Gott`), Left/Right, Home/End,
  Ctrl+A/Ctrl+E, Backspace/Delete, Ctrl+U, Enter; Up/Down and PgUp/PgDn
  scroll; F1 help; Ctrl+D exits on empty input; Ctrl+C cancels a running
  request, or exits with 130 when idle.
- **Commands** are those of the plain chat (`/help`, `/reset`, `/cleanup`
  with a `yes` confirmation on the next line, `/usage`, `/exit`).
- **Requests run in the background:** typing, scrolling and resizing keep
  working. With `--stream` the answer appears as it arrives (reasoning only
  switches the status to `Thinking`); without it the status shows
  `GENERATING | Waiting for the complete response`. While a response is
  generated, Enter keeps your text and shows a warning, Ctrl+D does
  nothing, F1 / `/help` / `/usage` work, and `/exit` cancels and exits.
- **Ctrl+C during a request** marks it `CANCELLED`, keeps the partial text
  visible (never saved or sent again) and closes the connection in the
  background.
- **F2–F6** open the settings menus (below). `COST (EST.)` shows the
  conversation's [estimated cost](#estimated-cost): `$0.000259`, `~$…`
  (on assumptions), `>=$…` (some requests could not be priced),
  `unavailable`, `$0 (local)`, or `-` before the first request.
- **The terminal is always restored** (also after Ctrl+C, errors,
  internal failures, SIGTERM and SIGHUP; a running request is then
  cancelled and not saved). Only an uncatchable SIGKILL (or a crash of the
  terminal itself) can leave it in fullscreen mode; type `reset` then. `--transcript-on-exit` then prints the committed
  conversation to stdout (`YOU > …` / `AI > …`, blank lines between).
- `--usage` applies to the one-shot and plain modes; the fullscreen UI
  shows usage in its panel and with `/usage`.

#### Settings menus (F2–F6)

| Key | Menu | Effect |
|---|---|---|
| F2 | `SELECT <PROVIDER> MODEL` | the current provider's models (Ollama: local chat models with size/quantization, never cloud models; OpenAI: text models from the Models API), loaded once per run |
| F3 | `SELECT PROVIDER` | OpenAI or Ollama; switching first checks the provider (`CONNECTING \| Checking <Label>`: OpenAI needs `OPENAI_API_KEY` and an answering Models API, Ollama a server with a local model) and then uses its default model |
| F4 | `REASONING` | the model's confirmed values (OpenAI table, Ollama `/api/show`); `Model default` sends no parameter; `Off`/`On` for Ollama thinking models |
| F5 | `MAX OUTPUT TOKENS` | `Default`, 256–4096 (within the model's known limit) or `Custom...`, typed at a `TOKENS >` prompt (Esc cancels) |
| F6 | `ANSWER STYLE` | Brief, Normal or Detailed (for every provider and model) |

In a menu: Up/Down, PgUp/PgDn, Home/End move, Enter selects, Esc or Ctrl+D
closes it unchanged, Ctrl+C exits (130). Nothing typed reaches the input
line, and a draft you had typed is kept. Menus never open while a response
is generated or a cancelled one is still closing.

- **F2/F3 switch conversations:** every provider/model has its own. The
  target's conversation comes back (from this run, else the latest saved
  one, which is not rewritten), or a new one starts:
  `Now using Ollama / llama3.2:3b. Restored its conversation (4 messages).`
  Messages are never sent to another model. If the check fails
  (`OpenAI is not available: …`), nothing changes. Ctrl+C cancels a check
  or model discovery (`Cancelled.`); a late answer is ignored.
- **F4–F6 apply from the next message** and keep the conversation. Each
  confirmed change is saved at once in `settings.json` (per model for
  reasoning and max tokens, globally for the answer style), in the Python
  edition's format: `"reasoning": "default"` for Model default and
  `"max_output_tokens": null` for Default. Cancelling a menu or choosing
  the value that is already saved writes nothing. A choice replaces the
  matching command-line flag for the rest of the run.
- **Thinking off for good:** a saved `"reasoning": "on"` beats the built-in
  Off; choose F4 → Off once to save Off.
- **`--no-history` affects conversations only:** switching stays in memory
  (no conversation or `active.json` file is touched), but F4–F6 choices
  are still saved in `settings.json`.
- If `settings.json` could not be read at startup it is never overwritten:
  a change then applies for this run only, with
  `Warning: Settings not saved: …`. A failed save is a warning too.

#### Plain chat

```text
NeuroBräu Go CLI 0.4.0-dev
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
| `/usage` | the conversation's token usage as reported by the API and its [estimated cost](#estimated-cost) |
| `/exit` | leave the chat (exit code 0) |
| any other `/word` | `Unknown command: /word. Type /help for commands.` |
| Ctrl+D (EOF) / Ctrl+C at the prompt | exit with code 0 / 130 |
| Ctrl+C while waiting for an answer | cancel the request, back to the prompt |

Commands are a single word starting with `/` or `\` (`\reset` works too),
in any case; `/usr/bin is what?` is an ordinary message. Input is read
line by line: there is no in-app line editing or input history beyond
what the terminal itself offers.

#### Estimated cost

`/usage` (in both chats) and the panel's `COST (EST.)` price the
conversation's usage records with a small catalog verified on OpenAI's
pricing pages on **2026-10-22** (USD per 1M tokens):

| Model | Input | Cached input | Cache write | Output |
|---|---:|---:|---:|---:|
| `gpt-6-luna`, ≤ 272K input tokens | 0.10 | 0.01 | 0.125 | 0.50 |
| `gpt-6-luna`, > 272K input tokens (whole request) | 0.20 | 0.02 | 0.25 | 0.75 |
| `gpt-5.1` | 1.25 | 0.125 | as input | 10.00 |

```text
Conversation usage (1 request(s), as reported by the API):
Input tokens: 2,500
  of which cached: 1,600
  of which cache writes: 100
Output tokens: 300
  of which reasoning: 50
Total tokens: 2,800
Latest request input: 2,500 tokens (the whole history is sent, and billed, with every request)
Estimated cost: $0.000259 (ESTIMATED, not a billed amount; pricing 2026-10-22)
```

- **These are estimates, not billed amounts.** Your OpenAI invoice is
  authoritative; prices change, and this catalog is not updated live.
- Only reported token counts are used; nothing is estimated from text.
  Reasoning tokens are part of the output and are not charged twice;
  cached input and cache writes are priced as parts of the input.
- Service tiers scale the price (`gpt-6-luna`: flex and batch 50%, fast /
  priority 2x; `gpt-5.1`: default only). When the tier or the cache counts
  are not reported, the estimate is computed on stated assumptions and
  marked *approximately* (`~`); the reasons are listed in `/usage`.
- Other models, unpriced tiers, missing input/output counts, Ollama cloud
  models (`:cloud`, `-cloud`) and other providers are `unavailable`; a
  conversation with some such requests shows a lower bound (`at least`,
  `>=`).
- Local Ollama models have **$0 API cost** (`API cost: $0 (local Ollama
  inference)`) — that is no charge from an API provider, not free
  electricity or hardware.
- Every request re-sends the whole history, so earlier messages are billed
  again; incomplete answers are billed too and counted. Costs are computed
  when shown and never stored in conversation or settings files.
- Amounts are exact; only the display rounds (half up, 6 decimals below
  one cent, else 4). One-shot `--usage` prints token counts only.

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

`settings.json` (shared with the Python edition) is read at chat startup;
only the fullscreen UI's F4–F6 write it. For the active provider/model:

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
| `--version` | `NeuroBräu Go CLI 0.4.0-dev` on stdout | 0 |
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
| SIGTERM / SIGHUP to the fullscreen UI | request cancelled, terminal restored | 143 / 129 |
| SIGTERM / SIGHUP elsewhere | default: the process ends (no terminal state to restore) | 143 / 129 (by signal) |

## Requirements

- Go 1.26 or newer (see `go.mod`)
- Direct dependencies: the official OpenAI Go SDK
  (`github.com/openai/openai-go/v3`), `github.com/gdamore/tcell/v2` (the
  fullscreen UI's terminal handling, with a simulation screen for tests)
  and `golang.org/x/term` (terminal detection); no CGO

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

Optional parity checks against the read-only Python 0.3.0 sources (skipped
unless `NEUROBRAU_PYTHON_SRC` is set; temp files only) compare storage
formats, cost estimates and `/usage` text, and generation-settings
resolution with Python's; see [`docs/parity.md`](docs/parity.md). Some
import modules that need the reference's dependencies, so point
`NEUROBRAU_PYTHON` at its virtual environment:

```bash
NEUROBRAU_PYTHON_SRC=$PWD/../llm-cli-python/src \
NEUROBRAU_PYTHON=$PWD/../llm-cli-python/.venv/bin/python \
  go test -count=1 -run Python ./...
```

To try the fullscreen UI by hand without touching your real history or
settings, give it throwaway directories (this uses a local Ollama server,
which costs nothing; nothing is installed or pulled):

```bash
d=$(mktemp -d)
HOME=$d XDG_STATE_HOME=$d/state XDG_CONFIG_HOME=$d/config \
  ./bin/neurobrau --provider ollama --stream --transcript-on-exit
# F2 another model, F4 Off, F5 512, F6 Brief, Ctrl+D to quit, then:
cat "$d/config/neurobrau/settings.json"; ls "$d/state/neurobrau/conversations"
```

## Layout

```text
cmd/neurobrau/   thin entry point: passes os.Args, stdout, stderr to cli.Run
internal/cli/    argument parsing, help, version, one-shot flow, chat startup,
                 exit codes
internal/chat/   plain chat loop, commands, generation settings resolution,
                 /usage report (shared by both chats)
internal/pricing/ versioned pricing catalog, exact (math/big) cost estimates
                 and their display forms
internal/tui/    fullscreen UI: layout, wrapping, input editing, event loop,
                 background requests and checks, F2-F6 menus (tcell)
internal/session/ active conversation: context, commit, restore, reset,
                 switch/park, cleanup (no UI, no network)
internal/llm/    provider-neutral types, Provider and StreamingProvider
                 interfaces, stream events, error kinds (llmtest: test helpers)
internal/ollama/ Ollama provider over net/http (/api/tags, /api/show, /api/chat)
internal/openai/ OpenAI provider over the official SDK (Responses API, Models API)
internal/storage/ Python-compatible JSON files: conversations, usage records,
                 active.json, settings.json
scripts/         development helpers (optional Python storage, pricing and
                 settings-resolution parity checks)
docs/tui-spec.md authoritative, language-neutral specification (do not edit)
docs/parity.md   differences from Python 0.3.0 and how they are checked
docs/release-checklist.md  verification and owner-only release steps
AGENTS.md        instructions for coding agents working on this repository
```

## Storage

`internal/storage` reads and writes the conversation, `active.json` and
`settings.json` formats shared with the Python edition (XDG locations,
atomic `0600` writes, `0700` app directories); the interactive chat uses
it through `internal/session`. See [`docs/storage.md`](docs/storage.md) for
the API, the Python fixtures and the optional parity checks.

## Roadmap (future — not implemented)

1. Release 0.4.0 after the owner's review ([checklist](docs/release-checklist.md)).
2. A TypeScript edition with the same observable contracts.
3. Pricing catalog refreshes as separate, reviewed changes.

Anthropic support, LangChain/LangGraph, RAG, vector databases and autonomous
agents are out of scope for Phase 1.
