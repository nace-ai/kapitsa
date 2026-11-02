# Local runtime control

`kapitsa serve` publishes local state for native clients, independently of the
terminal display. The authenticated control endpoint is separate from the model's
OpenAI and Anthropic APIs. `--name ID` selects a stable instance identity; the
default is `serve-PORT`. IDs contain 1–64 letters, digits, `-` or `_`.

Discovery records live in `$KAPITSA_HOME/instances` (normally
`~/.cache/kapitsa/instances`). The directory is owner-only and each record is a
private `0600` file. Records include a loopback control URL, an instance ID, an
execution ID (`run_id`) and a private credential. Clients must not display the
credential. Starting two executions with the same instance ID is refused.

Control requests require `Authorization: Bearer TOKEN` and
`X-Kapitsa-Run-ID: RUN_ID`. An old execution's credentials cannot stop its
replacement.

| Request | Result |
| --- | --- |
| `GET /v1/status` | Lifecycle state, active request progress and last-request usage. |
| `GET /v1/events?after=N` | State plus events after cursor `N`; `reset` signals that bounded replay has a gap. |
| `POST /v1/stop` | Acknowledges the selected execution, then requests its shutdown. |

The helper `kapitsa-runtime` enters the same server as `kapitsa serve`. It reads
one protocol-v1 JSON configuration from standard input or `--config FILE`.
Managed executions use a private local server and outlive their launcher.
`--check` validates configuration without downloading or loading a model or
using Metal:

```bash
printf '%s\n' '{"protocol":1,"instance_id":"code","model":"27b","port":8100}' \
  | build/kapitsa-runtime --check
```
