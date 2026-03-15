"""CLI entry point for the Build device client.

Usage:
    build-device                 # Run with default settings
    build-device --url https://getbuild.ing
    build-device --reset         # Re-run auth flow
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

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


async def async_main(base_url: str, reset: bool = False) -> None:
    """Main async entry point."""
    config = None if reset else load_config()

    if not config:
        config = await device_auth_flow(base_url)

    log.info("Device: %s (%s)", config.device_name, config.device_id)
    log.info("Server: %s", config.base_url)

    # Initialize local message store and E2EE handler.
    store = MessageStore()
    handler = E2EEHandler(config, store)

    try:
        await run_connection(config, e2e_handler=handler.handle_message)
    except KeyboardInterrupt:
        pass
    finally:
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
    args = parser.parse_args()

    try:
        asyncio.run(async_main(args.url, reset=args.reset))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
