# Kapitsa as Xcode 26's chat model

Xcode 26 can use Kapitsa as a chat model provider for code questions.

## Setup

Tested on Xcode 26.6 (build 17F113), macOS 26.5. Kapitsa registered as a Locally Hosted chat provider and appeared in the Chat section alongside ChatGPT and Claude.

1. Start Kapitsa:

```bash
brew install nace-ai/tap/kapitsa
kapitsa serve --model 35b
```

2. Open **Xcode → Settings → Intelligence**. Scroll to the **Chat** section at the bottom.

3. Click **Add a Chat Provider...**. This opens the "Add a Model Provider" dialog.

4. Select **Locally Hosted**. Enter port `8000` and a description (e.g. "Kapitsa"). Click **Add**.

Kapitsa appears in the Chat section as "Kapitsa" with a "localhost" label.

## What the Intelligence settings show

The Intelligence pane has four sections, top to bottom:

- **Coding Intelligence**: header with a privacy link.
- **Agents**: Claude Agent, Codex, Gemini (built-in, with "Get" buttons). "Add an Agent..." registers a local ACP (Agent Client Protocol) executable.
- **Model Context Protocol**: toggle for external agents to use Xcode tools via MCP.
- **Chat**: ChatGPT in Xcode, Claude Sonnet & Opus, and any custom providers added through "Add a Chat Provider...". The dialog has two tabs: "Internet Hosted" (URL, API Key, API Key Header, Description) and "Locally Hosted" (port, Description).

## Which model to serve

| Mac RAM | Model | Command |
|---------|-------|---------|
| 16 GB | Qwen3.5-4B | `kapitsa serve --model 4b` |
| 24 GB | Qwen3.5-9B | `kapitsa serve --model 9b` |
| 32-48 GB | Qwen3.6-35B-A3B (quicker) | `kapitsa serve --model 35b` |
| 48 GB | Qwen3.8-27B (stronger reasoning) | `kapitsa serve --model 27b` |
| 128+ GB | Flash-Next 180B | `kapitsa serve --model flash-next` |

The 35B-A3B is a mixture-of-experts model with fast decode (351 tok/s on an M5 Max). The 27B is denser and the strongest coder of the three Qwen sizes.

## Using a coding agent from the terminal

`kapitsa opencode` starts opencode on a local Kapitsa model alongside Xcode through the terminal:

```bash
kapitsa opencode --model 35b
```

`kapitsa claude` and `kapitsa codex` do the same for Claude Code and Codex.

## Verify the server

```bash
curl http://127.0.0.1:8000/v1/models
```

This returns the loaded model. If Xcode lists "Kapitsa" in its Chat section and answers coding questions through it, the setup works.
