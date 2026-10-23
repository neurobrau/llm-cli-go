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
	// Ctrl+C cancels a running request instead of killing the process.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	return cli.Run(ctx, os.Args[1:], os.Stdout, os.Stderr)
}
