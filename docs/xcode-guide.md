# Kapitsa as Xcode 26's local model provider

Xcode 26 can use a local model for its coding assistant. Kapitsa serves the OpenAI-compatible API that Xcode expects.

> **Not tested in Xcode's UI.** The steps below are from Xcode 26.6's framework strings (`IDEIntelligenceChat.framework`). The API endpoint was verified: `kapitsa serve` responds to `/v1/models` and `/v1/chat/completions` with tool-call support. The Xcode Intelligence settings UI has not been walked through on this Mac.

## Requirements

- macOS 15.6 or later (Xcode 26 runs on macOS 15.6+)
- Xcode 26 beta or later
- Kapitsa installed: `brew install nace-ai/tap/kapitsa`

## Start the server

```bash
kapitsa serve --model 35b
```

This starts Kapitsa on `http://127.0.0.1:8000` with the Qwen3.6-35B-A3B model. On a 16 GB Mac, use `--model 9b` or `--model 4b`.

## Point Xcode at Kapitsa

1. Open **Xcode → Settings → Intelligence**.
2. Click **Add a Model Provider**.
3. Choose **Locally Hosted**.
4. Enter port `8000` (Kapitsa's default).
5. Xcode works best with models that support tool calling. Leave that toggle on: Kapitsa's OpenAI API supports tool calls.
6. Close Settings. Xcode now routes coding assistant requests through Kapitsa.

## Which models work best

| Mac RAM | Model | Command |
|---------|-------|---------|
| 16 GB | Qwen3.5-4B | `kapitsa serve --model 4b` |
| 24 GB | Qwen3.5-9B | `kapitsa serve --model 9b` |
| 32-48 GB | Qwen3.6-35B-A3B | `kapitsa serve --model 35b` |
| 48 GB | Qwen3.8-27B | `kapitsa serve --model 27b` |
| 64+ GB | Flash-Next 180B | `kapitsa serve --model flash-next` |

The 35B-A3B is the best fit for code completion: it is a mixture-of-experts model with fast decode (351 tok/s on an M5 Max) and good code quality. The 27B is denser and better at reasoning but slower to write.

For code completion latency, a faster model with a drafter (the 35B, 9B, 4B) gives quicker suggestions than a dense model without one.

## Verify

After starting the server, confirm Kapitsa is reachable:

```bash
curl http://127.0.0.1:8000/v1/models
```

This returns the loaded model. If Xcode lists the model and answers coding questions through it, the setup works.
