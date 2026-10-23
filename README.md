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

## Current status (Go Task 002C — one-shot generation settings)

Working today: non-streaming one-shot prompts to **OpenAI** (Responses API,
the default) or a **local Ollama** server, with generation settings.

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
- Settings are not saved yet: each flag applies to this invocation only.

### OpenAI

- Uses the official OpenAI Go SDK and the Responses API, always with
  `store: false`. Default model: `gpt-6-luna`; `--model` accepts any model ID
  the API accepts.
- The API key is read from the `OPENAI_API_KEY` environment variable only.
  Set it in your shell before running a prompt; it is never written to disk
  or printed. Without it, OpenAI prompts fail with exit code 3; `--help`,
  `--version` and Ollama prompts do not need it.
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
| `--help`, `-h`; no arguments (temporary, until interactive chat exists) | usage on stdout | 0 |
| `--version` | `NeuroBräu Go CLI 0.1.0-dev` on stdout | 0 |
| successful prompt (also a non-streaming answer cut off at the token limit) | answer on stdout; with `--usage`, usage on stderr | 0 |
| API error, invalid reply, failed/refused/empty response, missing or cloud Ollama model | `neurobrau: error: …` | 1 |
| invalid arguments, unknown provider, empty prompt, setting the model does not support | `neurobrau: error: …` | 2 |
| `OPENAI_API_KEY` missing | `neurobrau: error: …` | 3 |
| authentication failed (HTTP 401/403) | `neurobrau: error: …` | 4 |
| rate limit or quota exceeded (HTTP 429) | `neurobrau: error: …` | 5 |
| timeout | `neurobrau: error: …` | 6 |
| connection failed | `neurobrau: error: …` | 7 |
| Ctrl+C during a request | `neurobrau: interrupted` | 130 |

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
internal/cli/    argument parsing, help, version, one-shot flow, exit codes
internal/llm/    provider-neutral types, Provider interface, error kinds
internal/ollama/ Ollama provider over net/http (/api/tags, /api/show, /api/chat)
internal/openai/ OpenAI provider over the official SDK (Responses API, Models API)
docs/tui-spec.md authoritative, language-neutral specification (do not edit)
AGENTS.md        instructions for coding agents working on this repository
```

## Roadmap (future — not implemented)

1. Saved settings (`settings.json`), plain interactive chat and conversation
   persistence compatible with the Python edition's on-disk JSON.
2. Streaming output (`--stream`, including exit 8 for an incomplete streamed
   answer).
3. Fullscreen retro TUI.
4. Token usage accounting and estimated cost (pricing catalog).

Anthropic support, LangChain/LangGraph, RAG, vector databases and autonomous
agents are out of scope for Phase 1.
