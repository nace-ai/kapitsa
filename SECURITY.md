# Security

Please report a vulnerability privately to **security@nace.ai**, not in a public issue. Include what you found, how
to reproduce it and the Kapitsa version (`kapitsa --version`). We will acknowledge your report and keep you informed
until it is fixed.

Kapitsa's server listens on every address and needs no API key by default, so that the Macs of a network find and use
each other's models (docs/usage.md, "Several Macs"): anyone who can reach the port can use the model. `--api-key`
requires a key for the APIs; `--private` keeps the server to `127.0.0.1`, invisible to the network.
