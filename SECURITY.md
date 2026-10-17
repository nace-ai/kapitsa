# Security

Please report a vulnerability privately to **security@nace.ai**, not in a public issue. Include what you found, how
to reproduce it and the Kapitsa version (`kapitsa --version`). We will acknowledge your report and keep you informed
until it is fixed.

Kapitsa's server listens on `127.0.0.1` and needs no API key by default. Serving a network (`--host 0.0.0.0`)
without `--api-key` lets anyone who can reach the port use the model.
