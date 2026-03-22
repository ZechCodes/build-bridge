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
import uuid
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
try:
    from claude_agent_sdk._internal import query as _sdk_query  # noqa: E402

    if not hasattr(_sdk_query.Query, "wait_for_result_and_end_input"):
        raise AttributeError(
            "SDK internals changed — 'Query.wait_for_result_and_end_input' not found. "
            "The monkey-patch in build_agent.py needs to be updated for this SDK version."
        )

    async def _patched_wait_for_result_and_end_input(self) -> None:
        """No-op replacement — keep stdin open for hooks throughout the query."""
        if self.sdk_mcp_servers or self.hooks:
            log.debug("Keeping stdin open for hook callbacks (patched)")
        else:
            await self.transport.end_input()

    _sdk_query.Query.wait_for_result_and_end_input = _patched_wait_for_result_and_end_input

except (ImportError, AttributeError) as _patch_err:
    log.warning(
        "Failed to apply SDK stdin monkey-patch: %s. "
        "Hooks may not work correctly if the SDK closes stdin early.",
        _patch_err,
    )

# ---------------------------------------------------------------------------
# Chat context prompt — instructs the agent to use Chat MCP tools
# ---------------------------------------------------------------------------

CHAT_CONTEXT = (
    "IMPORTANT: You are communicating with the user through a remote chat interface.\n"
    "- The user CANNOT see your text responses. The ONLY way to communicate "
    "with the user is by calling the mcp__build_chat__send tool.\n"
    "- When notified of unread messages, call mcp__build_chat__read_unread "
    "to read them.\n"
    "- Keep messages concise and natural. Don't narrate your thought process.\n"
    "- AskUserQuestion works through the remote UI — use it normally when you "
    "need user input. The user will see a prompt in their browser.\n"
    "- Plan mode (EnterPlanMode/ExitPlanMode) works through the remote UI — "
    "plan approval is handled by the browser interface.\n\n"
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
            messages = result.get("messages", [])

            # Build MCP content blocks — inline images as actual image blocks
            # so the model can see them natively.
            content_blocks: list[dict[str, Any]] = []
            for msg in messages:
                msg_content = msg.get("content", "")
                if isinstance(msg_content, str):
                    content_blocks.append({
                        "type": "text",
                        "text": f"[{msg['role']}] {msg_content}",
                    })
                elif isinstance(msg_content, list):
                    # Multimodal content blocks — convert Anthropic image format
                    # to MCP ImageContent format for the SDK.
                    for block in msg_content:
                        if block.get("type") == "text":
                            content_blocks.append({
                                "type": "text",
                                "text": f"[{msg['role']}] {block['text']}",
                            })
                        elif block.get("type") == "image":
                            # Convert Anthropic format → MCP format.
                            # Anthropic: {"type": "image", "source": {"type": "base64", "media_type": "...", "data": "..."}}
                            # MCP:       {"type": "image", "data": "...", "mimeType": "..."}
                            source = block.get("source", {})
                            content_blocks.append({
                                "type": "image",
                                "data": source.get("data", ""),
                                "mimeType": source.get("media_type", "image/png"),
                            })
                        elif block.get("type") == "image_url":
                            # OpenAI format — extract base64 data from data URL.
                            image_url = block.get("image_url", {}).get("url", "")
                            if image_url.startswith("data:"):
                                # Parse "data:<mime>;base64,<data>"
                                header, _, b64data = image_url.partition(",")
                                mime = header.split(";")[0].replace("data:", "")
                                content_blocks.append({
                                    "type": "image",
                                    "data": b64data,
                                    "mimeType": mime or "image/png",
                                })
                            else:
                                content_blocks.append({
                                    "type": "text",
                                    "text": f"[{msg['role']}] [Image: {image_url}]",
                                })
                        else:
                            content_blocks.append(block)

            if not content_blocks:
                content_blocks = [{"type": "text", "text": json.dumps(result)}]

            return {"content": content_blocks}
        except (ConnectionError, RuntimeError, OSError) as e:
            log.warning("read_unread tool error: %s", e)
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
        except (ConnectionError, RuntimeError, OSError) as e:
            log.warning("send tool error: %s", e)
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
        "Bash": lambda i: i.get("description") or f"Running: {i.get('command', '')[:60]}",
        "Glob": lambda i: f"Searching for {i.get('pattern', '')}",
        "Grep": lambda i: f"Searching for '{i.get('pattern', '')}'",
        "Agent": lambda i: f"Agent: {i.get('description', '')}",
        "WebFetch": lambda _: "Fetching web page",
        "WebSearch": lambda i: f"Searching: {i.get('query', '')}",
    }
    fn = descriptions.get(tool_name)
    return fn(tool_input) if fn else f"Using {tool_name}"


def make_pre_tool_hook(wrapper: AgentWrapper):
    """PreToolUse hook — emit tool.use and intercept interactive tools."""

    async def hook(input_data, tool_use_id, context):
        tool_name = input_data.get("tool_name", "")
        tool_input = input_data.get("tool_input", {})

        # Skip our own Chat MCP tools (§8.4).
        if tool_name.startswith("mcp__build_chat__"):
            return {"continue_": True}

        # --- Interactive tool interception ---

        if tool_name == "AskUserQuestion":
            log.info("AskUserQuestion tool_input keys: %s", list(tool_input.keys()))
            # AskUserQuestion uses "questions" (plural) — an array of question
            # objects, each with: question, header, options, multiSelect.
            questions_raw = tool_input.get("questions", [])
            if not questions_raw:
                # Fallback: maybe a single question object at top level.
                questions_raw = [tool_input]

            # Build a combined question text and options from all questions.
            question_parts = []
            options = []
            for q in questions_raw:
                header = q.get("header", "")
                qtext = q.get("question", "")
                if header and qtext:
                    question_parts.append(f"**{header}**: {qtext}")
                elif qtext:
                    question_parts.append(qtext)

                for opt in q.get("options", []):
                    if isinstance(opt, dict):
                        label = opt.get("label", "")
                        desc = opt.get("description", "")
                        opt_id = label  # Use label as ID since there's no explicit id.
                        display = f"{label} — {desc}" if desc else label
                        options.append({"id": opt_id, "label": display})
                    elif isinstance(opt, str):
                        options.append({"id": opt, "label": opt})

            question = "\n\n".join(question_parts)
            log.info(
                "Intercepting AskUserQuestion: question=%r, %d options",
                question[:80], len(options),
            )
            result = await wrapper.request_interaction(
                interaction_id=f"int_{uuid.uuid4().hex[:16]}",
                question=question,
                kind="question",
                options=options,
                allow_freeform=True,
            )
            if result.get("cancelled") or result.get("timed_out"):
                return {
                    "continue_": True,
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "additionalContext": "The user did not respond to your question in time.",
                    },
                }
            answer = result.get("freeform_response") or result.get("selected_option", "")
            return {
                "continue_": True,
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "additionalContext": f"The user has already responded via the remote UI. Their answer: {answer}",
                },
            }

        if tool_name == "ExitPlanMode":
            log.info("Intercepting ExitPlanMode for plan approval")
            log.info("ExitPlanMode tool_input: %s", json.dumps(tool_input, default=str)[:500])
            await wrapper.emit_state_update(plan_mode=False)

            # Extract plan content from tool_input (injected by normalizeToolInput)
            # or read from the plan file on disk.
            plan_content = tool_input.get("plan", "")
            if not plan_content:
                # Try to find plan file in .claude/plans/
                import glob
                plan_files = sorted(
                    glob.glob(os.path.join(os.getcwd(), ".claude", "plans", "*.md")),
                    key=os.path.getmtime,
                    reverse=True,
                )
                if not plan_files:
                    plan_files = sorted(
                        glob.glob(os.path.join(os.path.expanduser("~"), ".claude", "plans", "*.md")),
                        key=os.path.getmtime,
                        reverse=True,
                    )
                if plan_files:
                    try:
                        with open(plan_files[0]) as f:
                            plan_content = f.read()
                        log.info("Read plan from %s (%d chars)", plan_files[0], len(plan_content))
                    except Exception as exc:
                        log.warning("Failed to read plan file: %s", exc)

            result = await wrapper.request_interaction(
                interaction_id=f"int_{uuid.uuid4().hex[:16]}",
                question="Review and approve the plan?",
                kind="plan_review",
                options=[
                    {"id": "approve", "label": "Approve"},
                    {"id": "reject", "label": "Reject"},
                ],
                allow_freeform=True,
                plan=plan_content or None,
            )
            if result.get("selected_option") == "approve":
                return {"continue_": True}
            feedback = result.get("freeform_response") or "Plan rejected by user"
            return {
                "continue_": False,
                "systemMessage": f"The user rejected the plan: {feedback}",
            }

        if tool_name == "EnterPlanMode":
            await wrapper.emit_state_update(plan_mode=True)
            # Fall through to normal emit + continue.

        # --- Standard tool.use emission ---

        await wrapper.emit_tool_use(
            tool_use_id or f"tu_{id(input_data)}",
            tool_name,
            tool_input,
        )

        return {"continue_": True}

    return hook


_MAX_TOOL_RESULT_CHARS = 16_000  # Keep results under ~16 KB to stay within relay limits.


def _extract_tool_content(tool_response) -> tuple[str, bool]:
    """Extract text content and error flag from a tool_response value.

    tool_response can be a string, a list of content blocks
    (``[{"type": "text", "text": "..."}]``), or ``None``.  It may also
    be a dict wrapper around content blocks or a plain text value.
    """
    if tool_response is None:
        return "", False

    is_error = False
    if isinstance(tool_response, str):
        text = tool_response
    elif isinstance(tool_response, list):
        parts: list[str] = []
        for block in tool_response:
            if isinstance(block, dict):
                # Handle {"type": "text", "text": "..."} content blocks.
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                # Also handle blocks that have "content" instead of "text".
                elif "content" in block and isinstance(block["content"], str):
                    parts.append(block["content"])
                # Fallback: if no known text field, stringify the block.
                elif "text" not in block and "type" not in block:
                    parts.append(str(block))
                if block.get("is_error"):
                    is_error = True
            elif isinstance(block, str):
                parts.append(block)
            elif hasattr(block, "text"):
                # Handle SDK model objects with a .text attribute.
                parts.append(str(block.text))
            else:
                parts.append(str(block))
        text = "\n".join(parts)
    elif isinstance(tool_response, dict):
        is_error = bool(tool_response.get("is_error"))

        # Check for standard content block wrappers first.
        raw = tool_response.get("text", "") or tool_response.get("content", "")
        if raw:
            # If content is a list of blocks, recurse.
            if isinstance(raw, list):
                text, nested_err = _extract_tool_content(raw)
                is_error = is_error or nested_err
            elif isinstance(raw, str):
                text = raw
            else:
                text = str(raw)
        # Handle CLI tool-specific dict formats (Bash, Read, Grep, etc.).
        elif "stdout" in tool_response:
            # Bash tool: {stdout, stderr, interrupted, ...}
            parts = []
            if tool_response.get("stdout"):
                parts.append(tool_response["stdout"])
            if tool_response.get("stderr"):
                parts.append(f"stderr: {tool_response['stderr']}")
            text = "\n".join(parts)
            is_error = bool(tool_response.get("interrupted"))
        elif "filenames" in tool_response:
            # Glob tool: {filenames, numFiles, durationMs, truncated}
            # Grep (files_with_matches mode): {mode, filenames, numFiles}
            filenames = tool_response.get("filenames", [])
            content = tool_response.get("content", "")
            if content:
                # Grep content mode — has actual matched lines.
                text = content
            elif filenames:
                text = "\n".join(filenames)
            else:
                num = tool_response.get("numFiles", 0)
                text = f"No matches ({num} files)"
        elif tool_response.get("type") == "text" and "file" in tool_response:
            # Read tool: {type: "text", file: {filePath, content}}
            file_info = tool_response["file"]
            file_content = file_info.get("content", "")
            text = file_content
        elif "filePath" in tool_response and "oldString" in tool_response:
            # Edit tool: {filePath, oldString, newString, ...}
            fp = tool_response.get("filePath", "")
            text = f"Edited {os.path.basename(fp)}"
        elif "status" in tool_response and "prompt" in tool_response:
            # Agent tool: {status, prompt, agentId}
            status = tool_response.get("status", "")
            text = f"Agent {status}"
        elif "matches" in tool_response:
            # ToolSearch: {matches, query, total_deferred_tools}
            matches = tool_response.get("matches", [])
            text = f"Found: {', '.join(matches)}" if matches else "No matches"
        else:
            # Fallback: JSON-serialize the dict so we don't lose data.
            import json as _json
            text = _json.dumps(tool_response, indent=2, default=str)
    elif hasattr(tool_response, "text"):
        # Handle SDK model objects with a .text attribute.
        text = str(tool_response.text)
    else:
        text = str(tool_response)

    # Truncate oversized results so they don't blow the relay envelope limit.
    if isinstance(text, str) and len(text) > _MAX_TOOL_RESULT_CHARS:
        text = text[:_MAX_TOOL_RESULT_CHARS] + "\n…truncated"

    return text, is_error


def make_post_tool_hook(wrapper: AgentWrapper):
    """PostToolUse hook — emit tool.result to device client."""

    async def hook(input_data, tool_use_id, context):
        tool_name = input_data.get("tool_name", "")

        # Skip Chat MCP tools (§8.4).
        if tool_name.startswith("mcp__build_chat__"):
            return {}

        tool_response = input_data.get("tool_response")

        # Log what the SDK actually sends so we can diagnose missing results.
        if tool_response is None:
            log.warning(
                "PostToolUse %s (id=%s): tool_response is None. input_data keys: %s",
                tool_name, tool_use_id, list(input_data.keys()),
            )
        else:
            resp_type = type(tool_response).__name__
            resp_preview = str(tool_response)[:200]
            log.info(
                "PostToolUse %s (id=%s): tool_response type=%s preview=%s",
                tool_name, tool_use_id, resp_type, resp_preview,
            )

        content, is_error = _extract_tool_content(tool_response)

        await wrapper.emit_tool_result(
            tool_use_id or f"tu_{id(input_data)}",
            content or "",
            is_error=is_error,
            tool_name=tool_name,
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


def _stderr_logger(line: str) -> None:
    """Log Claude CLI stderr output."""
    line = line.rstrip()
    if line:
        log.info("[claude-stderr] %s", line)


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
        stderr=_stderr_logger,
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
                exit_reason = "context_ended"
                while True:
                    # Wait for a new user message (queued by wrapper.run() from chat.message).
                    has_msg = await wrapper.chat_mcp.wait_for_unread(timeout=5.0)

                    if cancel_event.is_set():
                        cancel_event.clear()
                        exit_reason = "cancelled"
                        log.info("Cancel received, breaking inner loop")
                        break

                    if not wrapper.is_connected:
                        exit_reason = "disconnected"
                        log.warning("Lost connection to device client")
                        break

                    if not has_msg:
                        # No message — emit activity ping to keep browser informed.
                        # Only if the agent is between turns (not mid-work).
                        continue

                    # Atomically check and build notification to avoid race
                    # with concurrent handle_read_unread draining the queue.
                    notification = await wrapper.chat_mcp.drain_unread_notification()
                    if notification:
                        await client.query(notification)
                        async for message in client.receive_response():
                            await handle_response_message(message, wrapper)

            # Client exited — log and decide whether to loop.
            log.info("Agent context ended (%s)", exit_reason)

            if not wrapper.is_connected:
                break

    except Exception as e:
        log.error("Agent error: %s", e, exc_info=True)
        try:
            await wrapper.chat_mcp.handle_send(
                "Something went wrong. The agent encountered an internal error "
                "and may need to restart."
            )
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
