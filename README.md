# Fennec

The fastest personal AI agent with collective intelligence.

A single Rust binary that runs as your persistent agent across Slack, Discord,
Telegram, WhatsApp, Email, an HTTP gateway, and a local CLI — sharing memory
and tools across every channel.

What makes Fennec different from a generic agent CLI:

- **Collective memory** (`Plurum`). Experiences are scrubbed for secrets and
  PII, then optionally published to a peer-learning network so other agents
  can search and benefit from solved problems. Your private data stays local;
  the redacted skill transfers.
- **Memory consolidation pipeline.** Daily summaries and core-fact extraction
  via the LLM, scrubbed and size-capped before persisting.
- **Snapshot / hydrate.** Resume an agent's working context across restarts
  without replaying the full conversation.
- **Reliable provider wrapper.** Multi-provider failover with per-provider
  cooldowns, exponential backoff, jitter, and an overall deadline.
- **Hardened by default.** Constant-time token comparison, prompt-injection
  guard, URL/SSRF guard, path sandbox, encrypted secrets at rest
  (ChaCha20-Poly1305).

## Status

Pre-1.0. Active development. The audit-driven security and correctness work
that landed across April 2026 is documented under
[`ARCHITECTURE.md`](ARCHITECTURE.md#security-posture).

## Install

### Quick install (Linux / macOS / WSL2)

```bash
curl -fsSL https://raw.githubusercontent.com/dunelabsco/fennec/main/scripts/install.sh | bash
```

The installer:
1. Verifies build tools (`git`, `gcc`, `make`, `pkg-config`, `libssl-dev`) and
   installs missing ones via the platform's package manager.
2. Installs Rust via `rustup` if not already present.
3. Builds Fennec in release mode (LTO, single codegen unit, panic-abort).
4. Symlinks the binary into `~/.local/bin/fennec`.
5. Creates `~/.fennec/` for config and data.

### Build from source

```bash
git clone https://github.com/dunelabsco/fennec.git
cd fennec
cargo install --path . --locked
```

The binary ends up in `~/.cargo/bin/fennec`. Add it to `$PATH` if it isn't.

### Update

```bash
~/.fennec/scripts/update.sh
```

Or rebuild from a fresh clone.

## First run

```bash
fennec onboard      # interactive setup wizard — provider, API key, channels
fennec doctor       # verify everything is wired up
fennec agent        # interactive chat in the terminal
```

`fennec onboard` writes `~/.fennec/config.toml`. Re-run with `--force` to
overwrite.

## Commands

| Command | What it does |
|---|---|
| `fennec agent` | Interactive chat session. `--message <text>` for single-shot. `--model <id>` to override. |
| `fennec gateway` | Start the multi-channel server. Runs all configured channels, the HTTP gateway, the cron scheduler, and the heartbeat loop together. |
| `fennec onboard` | Interactive setup wizard. `--force` overwrites an existing config. |
| `fennec login` | OAuth flow. Default: Anthropic (PKCE). `--provider copilot` runs the GitHub device-code flow for the Copilot provider. |
| `fennec doctor` | Self-diagnostic — verifies provider reachability, API key validity, memory DB schema, Plurum connectivity, skill loading, and channel config. |
| `fennec status` | Print version and quick status. |

All commands accept `--config-dir <path>` to override `$FENNEC_HOME`
(defaults to `~/.fennec`).

## Channels

| Channel | File | Auth | Streaming-edit support |
|---|---|---|---|
| CLI | `src/channels/cli.rs` | none (local stdin) | inline |
| Slack | `src/channels/slack.rs` | bot token + app token | partial-message edit |
| Discord | `src/channels/discord.rs` | bot token | partial-message edit |
| Telegram | `src/channels/telegram.rs` | bot token | passive |
| WhatsApp | `src/channels/whatsapp.rs` | access token + verify token + app secret | passive |
| Email | `src/channels/email.rs` | IMAP + SMTP creds | passive |

Channels are configured in `~/.fennec/config.toml` under `[channels.*]`. Tokens
can be stored as plaintext or encrypted via `SecretStore` (the `enc2:` prefix);
both paths work.

## Providers

| Provider | Streaming | Thinking-mode |
|---|---|---|
| Anthropic | SSE | Extended thinking, budget tokens |
| OpenAI | chunked | `reasoning_effort` (o1 family) |
| Google Gemini | SSE (`alt=sse`) | `thinkingConfig` budget (2.5 family) |
| Gemini (Cloud Code) | SSE (`alt=sse`) | `thinkingConfig` budget (2.5 family) |
| OpenAI Codex (Responses API) | SSE | `reasoning.effort` (gpt-5 / codex) |
| Azure OpenAI / Foundry | chunked | `reasoning_effort` (o-series / gpt-5) |
| AWS Bedrock | event-stream (Converse) | temperature fallback |
| GitHub Copilot | chunked | `reasoning_effort` (per underlying model) |
| Ollama | ND-JSON | temperature fallback |
| OpenRouter | passes through | passes through to underlying model |
| Kimi / Moonshot | OpenAI-shaped | temperature fallback |

Switch providers by setting `provider.name` in config; no code changes. The
`reliable_provider` wrapper (in `src/providers/reliable.rs`) lets you list a
fallback chain with cooldowns and an overall deadline.

Anthropic specifically supports OAuth via `fennec login`; other providers use
`provider.api_key` (encrypted at rest) or the equivalent env var
(`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, `OPENROUTER_API_KEY`,
`KIMI_API_KEY`).

Gemini has two flavors: `gemini` uses a `GEMINI_API_KEY`, while `gemini-cloudcode`
signs in with your Google account (`fennec login --provider gemini-cloudcode`)
for the Cloud Code Assist free tier — no API key, generous personal quota. The
login runs a loopback OAuth flow (with a paste fallback for headless/SSH hosts)
and discovers your Code Assist project automatically.

Set `provider.name = "codex"` to use OpenAI's Responses API (`/v1/responses`,
for gpt-5 / Codex models) instead of Chat Completions; it authenticates with the
same `OPENAI_API_KEY`.

**Azure OpenAI / Foundry** (`provider.name = "azure"`) is configured via
`config.toml`: set `provider.base_url` to your resource endpoint
(`https://<resource>.openai.azure.com`) and `provider.model` to the model family
(`gpt-5`, `o3-mini`, …). The deployment used for routing defaults to that model
name; set `AZURE_DEPLOYMENT` when your deployment is named differently. Auth is
auto-detected — an `AZURE_OPENAI_API_KEY` (or `provider.api_key`) uses key auth;
otherwise it goes keyless via Microsoft Entra ID, either through
`AZURE_TENANT_ID`/`AZURE_CLIENT_ID`/`AZURE_CLIENT_SECRET` (service principal) or
the Azure CLI (`az login`). Override `AZURE_OPENAI_API_VERSION` /
`AZURE_OPENAI_SCOPE` if needed.

**AWS Bedrock** (`provider.name = "bedrock"`) uses the Converse API with
SigV4-signed requests — no AWS SDK dependency. Set `provider.model` to the
Bedrock model or inference-profile id (e.g.
`anthropic.claude-3-5-sonnet-20241022-v2:0` or `us.anthropic.claude-…`).
Credentials resolve through a chain (env static keys → web-identity / EKS IRSA
via `AWS_WEB_IDENTITY_TOKEN_FILE` + `AWS_ROLE_ARN` → named profile from
`~/.aws/credentials` (`AWS_PROFILE`) → EC2/EKS instance role via IMDSv2); region
from `AWS_REGION` / `AWS_DEFAULT_REGION` (default `us-east-1`). SSO /
assume-role profiles are a follow-up.
**GitHub Copilot** (`provider.name = "copilot"`) uses the OpenAI-compatible
Copilot chat API. It needs a Copilot-enabled GitHub OAuth token (`gho_`/`ghu_`,
not a classic PAT), resolved from `COPILOT_GITHUB_TOKEN` / `GH_TOKEN` /
`GITHUB_TOKEN`, then `gh auth token`, then a token saved by
`fennec login --provider copilot` (GitHub device-code flow). The provider
exchanges it for a short-lived Copilot token automatically. Set
`provider.model` to a Copilot model id (`gpt-4o`, `o1`, `claude-3.5-sonnet`, …).

## Tools

Fennec ships ~27 built-in tools across files, shell, web, vision, voice,
code execution, image generation, memory, collective, cron, todo, git,
delegate, browser, MCP bridge, skills, and more. See
[`ARCHITECTURE.md`](ARCHITECTURE.md#tools) for the full list and
short descriptions.

External tools are loaded dynamically from MCP (Model Context Protocol)
servers configured in `~/.fennec/config.toml`. Both stdio and HTTP MCP
transports are supported. MCP tool names are namespaced
(`mcp_<server>_<tool>`) so a hostile server can't shadow a built-in.

## Skills

Skills are markdown files with YAML frontmatter that get pre-rendered into a
prompt fragment and injected into the agent's system prompt. 76 skills ship
in [`skills/`](skills/) covering integrations like 1Password, Apple Calendar,
GitHub, Notion, Linear, Slack, Spotify, Stripe, Zotero, etc.

Skill loading is hardened: symlink-rejected, 1 MiB per-file size cap, YAML
parsing with a maintained fork of `serde_yaml`.

Two activation modes:
- `always: true` — included on every turn.
- on-demand — surfaced via the `skills` tool when the agent decides it needs
  the integration's instructions.

## Memory

Three layers, all SQLite-backed:

1. **Conversation history** — in-memory `Vec<ChatMessage>` per session.
2. **Durable memory** — keyed key/content with FTS5 keyword search and
   optional embedding-based vector search (cosine + BM25 hybrid). Embeddings
   are optional; `NoopEmbedding` keeps the system local-only.
3. **Collective memory** — opt-in publication of redacted experiences to the
   Plurum peer-learning network. Search returns ranked results
   (High / Partial / None confidence) injected into the agent's context.

Daily summaries and core-fact extraction run via a consolidation pipeline
that scrubs both input and output through `collective::scrub` before storage.

## Security posture

- **Secrets at rest** — ChaCha20-Poly1305 with a 32-byte key generated on
  first run and stored at `~/.fennec/.key` (mode 0600). Keys are zeroized on
  drop.
- **Channel tokens** — every channel secret routes through `SecretStore::decrypt`
  with the same fail-closed pattern as `provider.api_key`. Plaintext
  passthrough is supported for legacy configs.
- **Prompt-injection guard** — `src/security/prompt_guard.rs` scans inputs
  with regex/RegexSet against a category-tagged pattern library; results are
  `Safe` / `Suspicious(categories, score)` / `Blocked(reason)`.
- **URL guard** — SSRF prevention; blocks private IPs, localhost, link-local.
- **Path sandbox** — symlink-aware whitelist enforcement for file ops.
- **Constant-time comparison** — `subtle` crate for bearer-token equality.
- **Anthropic OAuth** — PKCE flow, state parameter validated, `fs2` advisory
  lock around refresh.

See [`ARCHITECTURE.md`](ARCHITECTURE.md#security-posture) for module-level
detail.

## Configuration

`~/.fennec/config.toml`. Generated by `fennec onboard`. Top-level sections:

- `[identity]` — agent name, user ID
- `[provider]` — name, model, API key (or `enc2:` encrypted blob), base URL
- `[memory]` — DB path, embedding provider, retention policy
- `[security]` — prompt-guard sensitivity, action (Block / Warn / Sanitize)
- `[agent]` — max tool iterations, system-prompt overrides
- `[channels.*]` — per-channel auth and behavior
- `[gateway]` — HTTP host/port, auth token, body cap
- `[cron]` — store path, enabled flag
- `[collective]` — Plurum API key, base URL, enabled flag

A future doc will provide a full key reference; for now `fennec onboard`
walks you through every required field.

## Development

```bash
cargo build              # debug build
cargo test               # full test suite (33 integration tests + unit tests)
cargo clippy             # lint
cargo fmt                # format
```

CI runs `cargo test`, `cargo fmt --check`, `cargo clippy`, and `cargo audit`
on every PR. See [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

For deeper internals, read [`ARCHITECTURE.md`](ARCHITECTURE.md).

## License

MIT. See `Cargo.toml`.

## Repository

- Upstream: <https://github.com/dunelabsco/fennec>
- Issues: <https://github.com/dunelabsco/fennec/issues>
