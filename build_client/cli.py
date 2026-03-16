"""CLI entry point for the Build device client.

Usage:
    build-device                 # Run with default settings
    build-device --url https://getbuild.ing
    build-device --reset         # Re-run auth flow
    build-device --agent-port 9783  # Custom agent server port
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

from build_client.agent_server import AgentServer, DEFAULT_AGENT_PORT
from build_client.agent_spawner import AgentSpawner
from build_client.agent_store import AgentStore
from build_client.auth import device_auth_flow
from build_client.config import DEFAULT_CONFIG_PATH, load_config
from build_client.e2ee import E2EEHandler
from build_client.storage import MessageStore
from build_client.ws import run_connection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://getbuild.ing"


async def async_main(base_url: str, reset: bool = False, agent_port: int = DEFAULT_AGENT_PORT) -> None:
    """Main async entry point."""
    config = None if reset else load_config()

    if not config:
        config = await device_auth_flow(base_url)

    log.info("Device: %s (%s)", config.device_name, config.device_id)
    log.info("Server: %s", config.base_url)

    # Initialize stores.
    store = MessageStore()
    agent_store = AgentStore()

    # Initialize E2EE handler.
    handler = E2EEHandler(config, store)

    # Initialize agent server with E2EE broadcast callback.
    agent_server = AgentServer(
        store=agent_store,
        broadcast=handler.broadcast_to_sessions,
        port=agent_port,
    )

    # Initialize agent spawner for managing agent worker processes.
    agent_spawner = AgentSpawner(
        store=agent_store,
        agent_port=agent_port,
    )

    # Wire the agent server and spawner into the E2EE handler.
    handler.set_agent_server(agent_server)
    handler.set_agent_spawner(agent_spawner)

    try:
        # Start the local agent WS server.
        await agent_server.start()
        log.info("Agent server ready on port %s", agent_port)

        # Run the relay connection (blocking reconnect loop).
        await run_connection(config, e2e_handler=handler.handle_message)
    except KeyboardInterrupt:
        pass
    finally:
        await agent_spawner.stop_all()
        await agent_server.stop()
        agent_store.close()
        store.close()

    log.info("Disconnected.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build device client — connects to the Build server.",
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("BUILD_URL", DEFAULT_BASE_URL),
        help=f"Build server URL (default: {DEFAULT_BASE_URL}, or $BUILD_URL)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Re-run the device authorization flow (discard saved config).",
    )
    parser.add_argument(
        "--agent-port",
        type=int,
        default=int(os.environ.get("BUILD_AGENT_PORT", str(DEFAULT_AGENT_PORT))),
        help=f"Port for the local agent WebSocket server (default: {DEFAULT_AGENT_PORT})",
    )
    args = parser.parse_args()

    try:
        asyncio.run(async_main(args.url, reset=args.reset, agent_port=args.agent_port))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
