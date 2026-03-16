#!/usr/bin/env python3
"""Build Agent — Claude Code integration via Agent SDK + BAP wrapper.

Launches Claude Code as an SDK-managed agent with Chat MCP tools (read_unread,
send) and BAP protocol hooks that stream activity, tool use, and chat responses
to the device client.

The device client spawns this process per channel. It:
1. Connects to the device client's local WS server via AgentWrapper
2. Registers Chat MCP tools (read_unread, send) as in-process MCP
3. Hooks into SDK lifecycle events to emit BAP protocol messages
4. Runs the agent message loop, injecting user messages at breakpoints

Usage:
    build-agent                              # Connect to device client, wait for messages
    build-agent --port 9783                  # Custom device client port
    build-agent --prompt "Fix the auth bug"  # Start with initial prompt
    build-agent --model claude-sonnet-4-20250514  # Specify model
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    HookMatcher,
    TextBlock,
    ToolUseBlock,
    create_sdk_mcp_server,
    tool,
)

from build_client.agent_wrapper import AgentWrapper, CHAT_MCP_TOOLS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SDK monkey-patch: prevent premature stdin closure
# The SDK closes stdin after the first assistant message arrives, but hooks
# need the control stream open for the entire query duration.
# ---------------------------------------------------------------------------
from claude_agent_sdk._internal import query as _sdk_query  # noqa: E402


async def _patched_wait_for_result_and_end_input(self) -> None:
    """No-op replacement — keep stdin open for hooks throughout the query."""
    if self.sdk_mcp_servers or self.hooks:
        log.debug("Keeping stdin open for hook callbacks (patched)")
    else:
        await self.transport.end_input()


_sdk_query.Query.wait_for_result_and_end_input = _patched_wait_for_result_and_end_input

# ---------------------------------------------------------------------------
# Chat context prompt — instructs the agent to use Chat MCP tools
# ---------------------------------------------------------------------------

CHAT_CONTEXT = (
    "IMPORTANT: You are communicating with the user through a chat interface.\n"
    "- The user CANNOT see your text responses. The ONLY way to communicate "
    "with the user is by calling the mcp__build_chat__send tool.\n"
    "- When notified of unread messages, call mcp__build_chat__read_unread "
    "to read them.\n"
    "- Keep messages concise and natural. Don't narrate your thought process.\n"
    "- For complex tasks, use planning mode.\n\n"
)


# ---------------------------------------------------------------------------
# MCP tool creation
# ---------------------------------------------------------------------------


def make_chat_tools(wrapper: AgentWrapper) -> list:
    """Create SDK MCP tools backed by the wrapper's ChatMCP handlers."""

    @tool(
        "read_unread",
        "Read unread messages from the user. Returns any queued messages. "
        "Call this when notified of unread messages.",
        {},
    )
    async def read_unread(args):
        try:
            result = await wrapper.chat_mcp.handle_read_unread()
            return {"content": [{"type": "text", "text": json.dumps(result)}]}
        except Exception as e:
            return {"content": [{"type": "text", "text": f"Error: {e}"}], "is_error": True}

    @tool(
        "send",
        "Send a message to the user. Use this to communicate with the user "
        "instead of outputting text directly.",
        {"message": str},
    )
    async def send(args):
        try:
            result = await wrapper.chat_mcp.handle_send(args["message"])
            return {"content": [{"type": "text", "text": json.dumps(result)}]}
        except Exception as e:
            return {"content": [{"type": "text", "text": f"Error: {e}"}], "is_error": True}

    return [read_unread, send]


# ---------------------------------------------------------------------------
# SDK hook factories
# ---------------------------------------------------------------------------


def _describe_tool(tool_name: str, tool_input: dict) -> str:
    """Generate a human-readable description of what the tool is doing."""
    descriptions = {
        "Read": lambda i: f"Reading {os.path.basename(i.get('file_path', ''))}",
        "Edit": lambda i: f"Editing {os.path.basename(i.get('file_path', ''))}",
        "Write": lambda i: f"Writing {os.path.basename(i.get('file_path', ''))}",
        "Bash": lambda i: f"Running: {i.get('command', '')[:60]}",
        "Glob": lambda i: f"Searching for {i.get('pattern', '')}",
        "Grep": lambda i: f"Searching for '{i.get('pattern', '')}'",
        "Agent": lambda i: f"Agent: {i.get('description', '')}",
        "WebFetch": lambda _: "Fetching web page",
        "WebSearch": lambda i: f"Searching: {i.get('query', '')}",
    }
    fn = descriptions.get(tool_name)
    return fn(tool_input) if fn else f"Using {tool_name}"


def make_pre_tool_hook(wrapper: AgentWrapper):
    """PreToolUse hook — emit tool.use to device client."""

    async def hook(input_data, tool_use_id, context):
        tool_name = input_data.get("tool_name", "")
        tool_input = input_data.get("tool_input", {})

        # Skip our own Chat MCP tools (§8.4).
        if tool_name.startswith("mcp__build_chat__"):
            return {"continue_": True}

        # Emit tool.use via BAP.
        await wrapper.emit_tool_use(
            tool_use_id or f"tu_{id(input_data)}",
            tool_name,
            tool_input,
        )

        return {"continue_": True}

    return hook


def make_post_tool_hook(wrapper: AgentWrapper):
    """PostToolUse hook — emit tool.result to device client."""

    async def hook(input_data, tool_use_id, context):
        tool_name = input_data.get("tool_name", "")

        # Skip Chat MCP tools (§8.4).
        if tool_name.startswith("mcp__build_chat__"):
            return {}

        # The SDK doesn't directly expose the tool result in PostToolUse,
        # so we emit a completion signal. The full result content isn't
        # available here — emit a descriptive result instead.
        await wrapper.emit_tool_result(
            tool_use_id or f"tu_{id(input_data)}",
            _describe_tool(tool_name, input_data.get("tool_input", {})) + " — done",
            is_error=False,
        )

        return {}

    return hook


def make_stop_hook(wrapper: AgentWrapper):
    """Stop hook — emit activity.end to device client."""

    async def hook(input_data, tool_use_id, context):
        await wrapper.emit_activity_end("waiting")
        return {}

    return hook


def make_pre_compact_hook(wrapper: AgentWrapper):
    """PreCompact hook — emit activity delta about compaction."""

    async def hook(input_data, tool_use_id, context):
        trigger = input_data.get("trigger", "auto")
        await wrapper.emit_activity_delta("text", f"Compacting context ({trigger})...")
        return {}

    return hook


# ---------------------------------------------------------------------------
# Agent options builder
# ---------------------------------------------------------------------------


def build_agent_options(wrapper: AgentWrapper) -> ClaudeAgentOptions:
    """Build ClaudeAgentOptions with BAP hooks and Chat MCP tools."""
    tools = make_chat_tools(wrapper)
    mcp_server = create_sdk_mcp_server("build_chat", tools=tools)

    return ClaudeAgentOptions(
        setting_sources=["project"],
        permission_mode="bypassPermissions",
        mcp_servers={"build_chat": mcp_server},
        hooks={
            "PreToolUse": [HookMatcher(hooks=[make_pre_tool_hook(wrapper)])],
            "PostToolUse": [HookMatcher(hooks=[make_post_tool_hook(wrapper)])],
            "PreCompact": [HookMatcher(hooks=[make_pre_compact_hook(wrapper)])],
            "Stop": [HookMatcher(hooks=[make_stop_hook(wrapper)])],
        },
    )


# ---------------------------------------------------------------------------
# Response message handler
# ---------------------------------------------------------------------------


async def handle_response_message(
    message: Any,
    wrapper: AgentWrapper,
) -> str | None:
    """Process SDK response messages and emit activity deltas.

    AssistantMessage text blocks are internal reasoning — emitted as
    activity.delta (not chat.response, which only comes from the send tool).

    Returns the text content if any.
    """
    if isinstance(message, AssistantMessage):
        text_parts = []
        for block in message.content:
            if isinstance(block, TextBlock):
                text_parts.append(block.text)
                await wrapper.emit_activity_delta("text", block.text)
            elif isinstance(block, ToolUseBlock):
                # ToolUseBlock in content — the PreToolUse hook handles emission,
                # but we can emit thinking about tool use as activity.
                pass

        return "\n".join(text_parts) if text_parts else None
    return None


# ---------------------------------------------------------------------------
# Main agent loop
# ---------------------------------------------------------------------------


async def run_agent(
    port: int = 9783,
    host: str = "127.0.0.1",
    model: str = "claude-sonnet-4-20250514",
    initial_prompt: str | None = None,
    agent_id: str | None = None,
    working_directory: str | None = None,
) -> None:
    """Main agent loop — connects to device client and runs Claude Code."""

    # Change to working directory if specified.
    if working_directory and os.path.isdir(working_directory):
        os.chdir(working_directory)
        log.info("Working directory: %s", working_directory)

    # Create wrapper and connect to device client.
    cancel_event = asyncio.Event()

    async def on_cancel():
        cancel_event.set()

    wrapper = AgentWrapper(
        port=port,
        host=host,
        harness="claude-code",
        model=model,
        agent_id=agent_id,
        on_cancel=on_cancel,
    )

    try:
        config = await wrapper.connect()
    except ConnectionError as e:
        log.error("Failed to connect to device client: %s", e)
        return

    log.info("Connected to channel %s", config.channel_id)

    # Build chat_instructions into the system context.
    chat_context = CHAT_CONTEXT
    if config.chat_instructions:
        chat_context += f"{config.chat_instructions}\n\n"

    # Start the wrapper receive loop (handles incoming chat.message, etc.)
    receive_task = asyncio.create_task(wrapper.run())

    next_prompt: str | None = initial_prompt

    try:
        # Outer loop: each iteration is a fresh agent context.
        while True:
            options = build_agent_options(wrapper)
            reset_reason: str | None = None

            async with ClaudeSDKClient(options=options) as client:
                # Build initial prompt.
                if next_prompt:
                    prompt = chat_context + next_prompt
                else:
                    # Check for any queued messages from before we connected.
                    if wrapper.chat_mcp.has_unread:
                        notification = wrapper.chat_mcp.build_unread_notification()
                        prompt = chat_context + (notification or "Check in with the user.")
                    else:
                        prompt = chat_context + "You are online. Check in with the user using the send tool."
                next_prompt = None

                # Initial query.
                log.info("Sending initial prompt to agent")
                await client.query(prompt)
                async for message in client.receive_response():
                    await handle_response_message(message, wrapper)

                # Message loop — wait for user messages and inject them.
                while True:
                    # Wait for a new user message (queued by wrapper.run() from chat.message).
                    has_msg = await wrapper.chat_mcp.wait_for_unread(timeout=5.0)

                    if cancel_event.is_set():
                        cancel_event.clear()
                        log.info("Cancel received, breaking inner loop")
                        break

                    if not wrapper.is_connected:
                        log.warning("Lost connection to device client")
                        break

                    if not has_msg:
                        # No message — emit activity ping to keep browser informed.
                        # Only if the agent is between turns (not mid-work).
                        continue

                    # Build the unread notification and inject it.
                    notification = wrapper.chat_mcp.build_unread_notification()
                    if notification:
                        await client.query(notification)
                        async for message in client.receive_response():
                            await handle_response_message(message, wrapper)

            # Client exited — log and decide whether to loop.
            log.info("Agent context ended (%s)", reset_reason)

            if not wrapper.is_connected:
                break

    except Exception as e:
        log.error("Agent error: %s", e, exc_info=True)
        try:
            await wrapper.chat_mcp.handle_send(f"Agent error: {e}")
        except Exception:
            pass
    finally:
        receive_task.cancel()
        try:
            await receive_task
        except asyncio.CancelledError:
            pass
        await wrapper.disconnect("completed")
        log.info("Agent stopped")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build Agent — Claude Code + BAP wrapper")
    parser.add_argument("prompt", nargs="*", help="Initial prompt")
    parser.add_argument("--port", type=int, default=int(os.environ.get("BUILD_AGENT_PORT", "9783")),
                        help="Device client agent server port (default: 9783)")
    parser.add_argument("--host", default=os.environ.get("BUILD_AGENT_HOST", "127.0.0.1"),
                        help="Device client agent server host (default: 127.0.0.1)")
    parser.add_argument("--model", default=os.environ.get("BUILD_AGENT_MODEL", "claude-sonnet-4-20250514"),
                        help="Model to use")
    parser.add_argument("--agent-id", default=os.environ.get("BUILD_AGENT_ID"),
                        help="Agent ID (for reconnection)")
    parser.add_argument("--working-directory", default=os.environ.get("BUILD_WORKING_DIR"),
                        help="Working directory for the agent")
    args = parser.parse_args()

    initial_prompt = " ".join(args.prompt) if args.prompt else None

    asyncio.run(run_agent(
        port=args.port,
        host=args.host,
        model=args.model,
        initial_prompt=initial_prompt,
        agent_id=args.agent_id,
        working_directory=args.working_directory,
    ))


if __name__ == "__main__":
    main()
