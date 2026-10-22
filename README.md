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

## Current status (Go Task 001 — bootstrap)

Only the project skeleton is implemented:

| Invocation | Output | Exit code |
|---|---|---|
| `neurobrau --help`, `neurobrau -h` | usage on stdout | 0 |
| `neurobrau` (no arguments) | usage on stdout | 0 |
| `neurobrau --version` | `NeuroBräu Go CLI 0.1.0-dev` on stdout | 0 |
| unknown flag, positional argument, malformed flag value | `neurobrau: error: …` on stderr | 2 |

There is no chat, provider access, persistence or fullscreen UI yet. No API
key is needed, and none is read.

## Requirements

- Go 1.26 or newer (see `go.mod`)
- No third-party dependencies, no CGO

## Build, test, run

```bash
go build -o bin/neurobrau ./cmd/neurobrau
./bin/neurobrau --help
./bin/neurobrau --version

go test ./...
go vet ./...
gofmt -l .        # must print nothing
```

## Layout

```text
cmd/neurobrau/   thin entry point: passes os.Args, stdout, stderr to cli.Run
internal/cli/    argument parsing, help, version, exit codes (unit tested)
docs/tui-spec.md authoritative, language-neutral specification (do not edit)
AGENTS.md        instructions for coding agents working on this repository
```

## Roadmap (future — not implemented)

1. Providers: OpenAI Responses API and local Ollama, with model discovery.
2. One-shot prompts and plain interactive chat, with conversation persistence
   and settings compatible with the Python edition's on-disk JSON.
3. Streaming output (`--stream`).
4. Fullscreen retro TUI.
5. Token usage accounting and estimated cost (pricing catalog).

Anthropic support, LangChain/LangGraph, RAG, vector databases and autonomous
agents are out of scope for Phase 1.
