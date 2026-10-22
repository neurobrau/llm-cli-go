# NeuroBräu Go — Agent Instructions

## Project

NeuroBräu Go is an idiomatic Go implementation of the NeuroBräu LLM CLI. The Python 0.3.0 implementation (`neurobrau/llm-cli-python`) defines the reference behavior; the Go edition is **not** a line-by-line translation. A later TypeScript edition will use the same observable contracts.

- Go module: `github.com/neurobrau/llm-cli-go`
- Go executable: `neurobrau`
- Development machine: Fedora Linux, amd64, Bash, Ptyxis.
- Supported providers in Phase 1: OpenAI Responses API and local Ollama HTTP API. Anthropic is deferred.
- Phase 1 excludes LangChain/LangGraph, RAG, vector databases, and autonomous agents.

## Authoritative references

1. **`docs/tui-spec.md`** — language-neutral, authoritative specification, copied unchanged from `../llm-cli-python/docs/tui-spec.md`. Its interoperability sections cover JSON schemas, XDG paths, request mappings, usage/pricing rules, and failure semantics. Read the relevant sections before implementing those features.
2. `../llm-cli-python/` — reference implementation, **read only**. Do not modify Python files.
3. This repository's README and task instructions — scope and acceptance criteria.

If a reference is missing or inconsistent, report the issue instead of guessing a wire format or silently changing the contract. Prefer the reference specification over conceptual UI sketches.

## Engineering approach

- Use idiomatic, straightforward Go, not Python-like class hierarchies.
- Keep `cmd/neurobrau` thin; move testable logic into a small `internal/cli` package.
- Create packages when functionality warrants them. Do not pre-create empty `engine`, `providers`, `storage`, `usage`, or `ui` packages.
- Prefer the standard library for the bootstrap. Introduce third-party dependencies only in a task that needs them, with an explicit rationale.
- Use clear names, small functions, standard `error` values, `errors.Is` / `errors.As` when helpful, and explicit handling at boundaries.
- Future provider calls should take `context.Context`; use goroutines/channels for actual asynchronous needs, not by default.
- Future screen rendering must stay in a single controlled UI event loop. Keep HTTP/SDK logic and storage out of the TUI.
- Preserve user-facing behavior and on-disk JSON compatibility with Python. Do not read/write real conversation or settings files during tests.
- Do not implement speculative product features or unrelated refactors.

## Security and safety

- Do **not** read, print, inspect, copy, or log secrets, including `~/.config/neurobrau/credentials.sh` or `OPENAI_API_KEY`.
- Never put credentials into source code, test fixtures, command output, README examples, logs, or Git.
- No live provider requests in unit tests; use fakes/`httptest` in later tasks.
- Do not modify files outside the Go repository. The sibling Python repository is read-only.
- Do not read or modify actual user settings/history under `~/.config/neurobrau/` or `~/.local/state/neurobrau/`.
- Do **not** run `git commit`, `git push`, create tags/releases, or modify GitHub remotely. Leave version control actions for the user.

## Quality gates

Before reporting completion:

```bash
gofmt -l .
go vet ./...
go test ./...
go build -o bin/neurobrau ./cmd/neurobrau
git diff --check
git status --short
```

`gofmt -l .` must print nothing. Go module and imported dependency versions belong in `go.mod`; retain `go.sum` when dependencies introduce one. `go.sum` need not exist during standard-library-only bootstrap.

Tests should cover success and failure paths without the network, secrets, terminal UI, or user home directory. Report commands executed, test results, and any remaining limitations. Do not claim tests ran if they did not.

## Task scope

Implement only the current numbered task. For **Go Task 001**, deliver a minimal, buildable CLI with `--help`, `-h`, and `--version`, deterministic exit codes, basic documentation and tests. No API integrations, model discovery, streaming, persistence, full-screen UI, or additional CLI behavior yet.

Distinguish the Go bootstrap version (`0.1.0-dev`) from feature parity with Python 0.3.0; do not claim parity until the later tasks pass.
