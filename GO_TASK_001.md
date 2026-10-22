# NeuroBräu Go — Task 001: Project Bootstrap

## Objective

Bootstrap a clean, self-contained Go CLI project at `~/Developer/neurobrau/llm-cli-go` with a correct module, a minimal executable, CLI help/version behavior, tests, and documentation. This task is intentionally **not** an LLM integration task.

## Context and references

- GitHub organization: `https://github.com/neurobrau`
- Intended repository: `github.com/neurobrau/llm-cli-go` (the remote may not yet exist; do not create it).
- Module path: `github.com/neurobrau/llm-cli-go`.
- Reference Python repo (read-only): `~/Developer/neurobrau/llm-cli-python`, version 0.3.0.
- Authoritative specification: `docs/tui-spec.md`, copied without edits from `../llm-cli-python/docs/tui-spec.md` if not already present. Its future requirements **do not** need to be implemented in Task 001.
- Read and follow `AGENTS.md` before changing files.
- Host: Fedora 44, x86_64, Go 1.26.10, Bash.

## In scope

1. Initialize the Go module if not already initialized. Respect existing files if the user has run setup commands.
2. Create a minimal conventional structure:

   ```text
   cmd/neurobrau/main.go
   cmd/neurobrau/main_test.go       (only if it tests useful entrypoint behavior)
   internal/cli/cli.go
   internal/cli/cli_test.go
   docs/tui-spec.md                 (byte-for-byte copy of reference)
   AGENTS.md
   README.md
   .gitignore
   go.mod
   ```

   Do not create a `tests/` directory unless genuinely necessary; keep Go unit tests adjacent to the code. Do not create a LICENSE without the owner's license choice.
3. Implement:
   - `neurobrau --help` and `neurobrau -h`: readable usage/help to stdout, code 0.
   - `neurobrau --version`: `NeuroBräu Go CLI 0.1.0-dev` followed by a newline to stdout, code 0.
   - Unknown flag, unexpected positional argument, or malformed invocation: concise `neurobrau: error: ...` to stderr, code 2, not a stack trace.
   - Do not claim that unimplemented options, chatting, providers, or TUI are functional. Help may clearly identify them as planned in documentation, but not as currently supported flags.
   - Keep the entrypoint thin: `main` passes `os.Args[1:]`, stdout and stderr into a testable `cli.Run(args []string, stdout, stderr io.Writer) int` (or an equally straightforward design), and exits with the returned code.
   - Use standard-library `flag` or equivalent simple parsing; no external CLI framework for bootstrap.
4. Write useful tests, including successful help, `-h`, version, unknown option, unexpected positional args, correct stdout/stderr separation, and exit codes. Do not shell out to external providers.
5. Write a concise README with the project purpose, how it relates to Python 0.3.0, current Task 001 scope, module path, build/test/run instructions, and a clear roadmap (OpenAI/Ollama, plain chat/persistence, streaming, TUI, usage/pricing) labelled as **future**.
6. Add `.gitignore` for `/bin/`, the root `neurobrau` executable, IDE/OS clutter, and local credential/environment files. Do **not** ignore `go.mod` or `go.sum`.
7. Run `go mod tidy`; `go.sum` is not required if no external dependencies were added.

## Constraints

- No OpenAI SDK, Ollama, HTTP integration, real API requests, tcell, TUI, streaming, settings, conversation persistence, or pricing yet.
- No CGO requirement. Prefer standard library only.
- No reading `~/.config/neurobrau/credentials.sh`, API keys, or real XDG settings/history.
- Do not change the Python repository. Copy its spec **only** into this Go repository.
- No commits, pushes, tags, releases, GitHub actions, or remote repo creation.
- Do not delete or replace existing user work. If initialization has already happened, make only needed changes.
- Keep the code idiomatic Go, small, and testable. Do not create dummy empty packages for future tasks.

## Verification / acceptance criteria

Run and report results of:

```bash
gofmt -l .
go vet ./...
go test ./...
go build -o bin/neurobrau ./cmd/neurobrau
./bin/neurobrau --help
./bin/neurobrau --version
./bin/neurobrau --not-a-real-flag   # expected exit 2
./bin/neurobrau unexpected-input   # expected exit 2
git diff --check
git status --short
```

`gofmt -l .` should produce no output. Run help/version from a shell where `OPENAI_API_KEY` is unset, if convenient: both must work without credentials. Do not expose secret values while doing so. The binary should be an executable Linux ELF for amd64. Confirm `docs/tui-spec.md` is byte-identical to the Python copy (`cmp -s` when the sibling repo is present).

## Final report format

- Files created/changed and why.
- CLI behavior and exit codes.
- Exact verification commands and their outcomes, including test counts.
- Dependencies introduced (ideally none).
- Known issues or deviations from the reference.
- Recommendation for Go Task 002 **only after** Task 001 passes.

Stop after implementing and testing Task 001. Wait for review; do not commit or push.
