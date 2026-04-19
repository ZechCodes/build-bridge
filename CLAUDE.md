# Build Bridge

## CRITICAL: Managing the Build Bridge Daemon

**DO NOT run `build stop` or `python -m build_bridge.cli stop` unless explicitly asked by the user.**

Build Bridge is a long-running daemon that manages agent connections. Stopping it disconnects ALL agents across ALL channels — including yours.

### Safe operations
- `build restart` — restarts the daemon, agents reconnect automatically
- `build status` — check if the daemon is running
- `build agents` — list running agents

### Dangerous operations (require explicit user request)
- `build stop` — **kills the daemon and ALL agents**, nothing reconnects
- Killing processes manually (`kill`, `pkill`, etc.)

### If you need to restart the daemon
Always use `build restart`, never `build stop` followed by `build start`. The restart command preserves agent channel state so agents respawn automatically.

## Project Structure

- `build_bridge/cli.py` — CLI entry point and `async_main`
- `build_bridge/daemon.py` — Singleton process manager, watchdog, control socket
- `build_bridge/e2ee.py` — E2EE handler, terminal exec, interactions
- `build_bridge/agent_wrapper.py` — BAP protocol wrapper for agent processes
- `build_bridge/agent_server.py` — Local WebSocket server for agent connections
- `build_bridge/agent_spawner.py` — Agent subprocess management
- `build_bridge/build_agent.py` — Claude Code harness runtime
- `build_bridge/codex_agent.py` — Codex CLI harness runtime
- `build_bridge/harnesses/*.json` — Harness model definitions

## Running Tests

```bash
uv run pytest tests/ -v
```
