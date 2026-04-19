# Build Bridge

Device client for [Build](https://getbuild.ing). Connects to the Build server over E2E encrypted WebSocket, manages agent processes (Claude Code, Codex CLI), and provides a control CLI for operations.

## Install

```bash
curl -sSf getbuild.ing | sh
```

The installer is self-contained: it installs `uv` if needed, clones this repo and [`build-secure-transport`](https://github.com/ZechCodes/build-secure-transport) into `~/.local/share/build/`, sets up a `build` launcher in `~/.local/bin/`, and optionally configures auto-start on login. Everything lives under your home directory — no `sudo`, no `/usr/local` writes.

### From source

```bash
uv sync
```

## CLI

```
build start [--url URL] [--agent-port PORT] [--reset] [--no-watchdog]
build stop [--keep-agents]
build restart [--keep-agents]
build status
build agents
build agent-stop <channel>
build agent-restart <channel>
```

### Commands

| Command | Description |
|---------|-------------|
| `start` | Start the device client daemon (default if no command given) |
| `stop` | Graceful shutdown. Stops all agents unless `--keep-agents` is passed |
| `restart` | Stop and restart. Agents are restarted unless `--keep-agents` is passed |
| `status` | Show daemon status: PID, uptime, relay connection, agent count |
| `agents` | List running agents with channel, harness, model, PID |
| `agent-stop` | Stop a specific agent by channel ID (or prefix) |
| `agent-restart` | Restart a specific agent by channel ID (or prefix) |

### Options

| Flag | Command | Description |
|------|---------|-------------|
| `--url` | `start` | Build server URL (default: `https://getbuild.ing`, or `$BUILD_URL`) |
| `--agent-port` | `start` | Agent WebSocket port (default: 9783, or `$BUILD_AGENT_PORT`) |
| `--reset` | `start` | Re-run device authorization flow |
| `--no-watchdog` | `start` | Run without crash recovery (exits on error instead of restarting) |
| `--keep-agents` | `stop`, `restart` | Leave agent processes running through the stop/restart |

### Process management

The daemon enforces a singleton — only one instance can run at a time. A second `build start` will fail with an error.

On crash, the watchdog automatically restarts the daemon with exponential backoff (2s to 60s). The backoff resets after 60 seconds of healthy running.

`build stop --keep-agents` is useful for upgrading the client without interrupting running agents. The agents will reconnect when the daemon restarts.

### Examples

```bash
# Start and leave running
build start

# Check what's running
build status
build agents

# Restart without killing agents
build restart --keep-agents

# Stop a specific agent
build agent-stop 8c461d72

# Stop everything
build stop
```

## Harnesses

Agent harness definitions live in `build_bridge/harnesses/*.json`. Each file defines a harness with its models:

```json
{
  "id": "claude-code",
  "name": "Claude Code",
  "binary": "claude",
  "default_model": "claude-sonnet-4-6",
  "models": [
    {"id": "claude-sonnet-4-6", "name": "Claude Sonnet 4.6", "provider": "anthropic"}
  ]
}
```

To add a new harness, create a JSON file and a corresponding runtime module in `build_bridge/`, then register the module in `agent_spawner.py`'s `_HARNESS_MODULES` dict.

## Development

```bash
uv run pytest tests/ -v
```

## License

MIT
