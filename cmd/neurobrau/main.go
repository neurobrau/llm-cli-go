// Command neurobrau is the NeuroBräu Go command-line client.
package main

import (
	"os"

	"github.com/neurobrau/llm-cli-go/internal/cli"
)

func main() {
	os.Exit(cli.Run(os.Args[1:], os.Stdout, os.Stderr))
}
