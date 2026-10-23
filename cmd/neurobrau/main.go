// Command neurobrau is the NeuroBräu Go command-line client.
package main

import (
	"context"
	"os"
	"os/signal"

	"github.com/neurobrau/llm-cli-go/internal/cli"
)

func main() {
	os.Exit(run())
}

func run() int {
	// Ctrl+C is delivered to the CLI instead of killing the process: it
	// cancels a running request, or ends the chat when it is idle.
	interrupts := make(chan os.Signal, 1)
	signal.Notify(interrupts, os.Interrupt)
	defer signal.Stop(interrupts)
	return cli.Run(context.Background(), os.Args[1:], cli.Streams{
		Stdin:      os.Stdin,
		Stdout:     os.Stdout,
		Stderr:     os.Stderr,
		Interrupts: interrupts,
	})
}
