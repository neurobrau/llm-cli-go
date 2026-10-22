// Package cli implements the neurobrau command-line interface.
//
// It is kept independent of os.Args, os.Stdout and os.Exit so that it can be
// tested without spawning processes.
package cli

import (
	"errors"
	"flag"
	"fmt"
	"io"
)

const (
	// ProgName is the executable name used in usage and error messages.
	ProgName = "neurobrau"
	// Version is the Go edition's own version. It does not imply feature
	// parity with the Python reference implementation (0.3.0).
	Version = "0.1.0-dev"
)

// Exit codes (docs/tui-spec.md, section 16).
const (
	ExitOK    = 0
	ExitUsage = 2
)

const helpText = `usage: neurobrau [-h] [--version]

NeuroBräu Go CLI — educational multi-provider LLM command-line client.

This is a development bootstrap (` + Version + `). Chatting, providers and the
fullscreen UI are not implemented yet.

options:
  -h, --help  show this help message and exit
  --version   show program's version number and exit
`

// Run parses args (without the program name), writes output to stdout and
// stderr, and returns the process exit code.
func Run(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(ProgName, flag.ContinueOnError)
	// Errors and usage are reported by Run itself, not by the flag package.
	fs.SetOutput(io.Discard)
	fs.Usage = func() {}
	showVersion := fs.Bool("version", false, "show program's version number and exit")

	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			fmt.Fprint(stdout, helpText)
			return ExitOK
		}
		return usageError(stderr, err.Error())
	}
	if fs.NArg() > 0 {
		return usageError(stderr, fmt.Sprintf("unexpected argument: %q", fs.Arg(0)))
	}
	if *showVersion {
		fmt.Fprintf(stdout, "NeuroBräu Go CLI %s\n", Version)
		return ExitOK
	}
	fmt.Fprint(stdout, helpText)
	return ExitOK
}

func usageError(stderr io.Writer, msg string) int {
	fmt.Fprintf(stderr, "%s: error: %s\n", ProgName, msg)
	return ExitUsage
}
