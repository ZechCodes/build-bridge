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
from pathlib import Path
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    HookMatcher,
    PermissionResultAllow,
    PermissionResultDeny,
    TextBlock,
    ToolPermissionContext,
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
    "You have access to 'send' and 'read_unread' MCP tools for communicating "
    "with the user. Use 'read_unread' to check for user messages and 'send' "
    "to reply. Do not output user-facing text directly — always use the send tool.\n\n"
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

            # Build MCP content blocks. For images, reference the file path
            # instead of inlining base64 — Claude Code can read images natively
            # via its Read tool, and inlining large base64 payloads can exceed
            # the SDK's stdio buffer limit (~10MB).
            content_blocks: list[dict[str, Any]] = []
            for msg in messages:
                msg_content = msg.get("content", "")
                if isinstance(msg_content, str):
                    content_blocks.append({
                        "type": "text",
                        "text": f"[{msg['role']}] {msg_content}",
                    })
                elif isinstance(msg_content, list):
                    for block in msg_content:
                        if block.get("type") == "text":
                            content_blocks.append({
                                "type": "text",
                                "text": f"[{msg['role']}] {block['text']}",
                            })
                        elif block.get("type") in ("image", "image_url"):
                            # Instead of inlining base64, tell the agent the file path.
                            # The ChatMCP already notes file paths for non-image attachments;
                            # for images, the path is in the attachment metadata.
                            content_blocks.append({
                                "type": "text",
                                "text": f"[{msg['role']}] [Image attached — use the Read tool to view it]",
                            })
                        else:
                            content_blocks.append(block)

            if not content_blocks:
                content_blocks = [{"type": "text", "text": json.dumps(result)}]

            # Remind the agent to use the send tool to respond.
            content_blocks.append({
                "type": "text",
                "text": (
                    "REMINDER: The user cannot see your text responses. "
                    "Use mcp__build_chat__send to communicate what you're "
                    "going to do and what you've done."
                ),
            })

            return {"content": content_blocks}
        except (ConnectionError, RuntimeError, OSError) as e:
            log.warning("read_unread tool error: %s", e)
            return {"content": [{"type": "text", "text": f"Error: {e}"}], "is_error": True}

    @tool(
        "send",
        "Send a message to the user. Use this to communicate with the user "
        "instead of outputting text directly.\n\n"
        "You can embed file snippets and diffs in messages:\n"
        "- [[file]](path/to/file) — embed entire file with syntax highlighting\n"
        "- [[file 8:18]](path/to/file) — embed lines 8-18\n"
        "- [[diff]](path/to/file) — embed git diff for file\n"
        "- [[diff]](file1|file2) — diff two files\n"
        "Paths are resolved relative to the channel working directory (not your shell cwd). "
        "Use absolute paths if you've cd'd elsewhere. "
        "Embeds render with syntax highlighting in the browser.",
        {
            "type": "object",
            "properties": {
                "message": {"type": "string"},
                "suggested_actions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional 2-3 short action labels shown as clickable buttons below the message (e.g. ['yes', 'no']). Clicking one sends it as a user message.",
                },
            },
            "required": ["message"],
        },
    )
    async def send(args):
        try:
            # Resolve relative embed paths to absolute using the agent's cwd,
            # since the agent may have cd'd away from the channel working dir.
            import re as _re
            _embed_path_re = _re.compile(r"(\[\[(?:file(?:\s+\d+:\d+)?|diff)\]\])\(([^)]+)\)")
            def _abs_paths(m):
                tag, raw = m.group(1), m.group(2)
                if "|" in raw:
                    parts = [str(Path(p.strip()).resolve()) for p in raw.split("|", 1)]
                    return f"{tag}({parts[0]}|{parts[1]})"
                return f"{tag}({Path(raw.strip()).resolve()})"
            message = _embed_path_re.sub(_abs_paths, args["message"])
            result = await wrapper.chat_mcp.handle_send(
                message,
                suggested_actions=args.get("suggested_actions"),
            )
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
            # AskUserQuestion uses "questions" (plural) — an array of question
            # objects, each with: question, header, options, multiSelect.
            questions_raw = tool_input.get("questions", [])
            if not questions_raw:
                questions_raw = [tool_input]

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
                        opt_id = label
                        display = f"{label} — {desc}" if desc else label
                        options.append({"id": opt_id, "label": display})
                    elif isinstance(opt, str):
                        options.append({"id": opt, "label": opt})

            question = "\n\n".join(question_parts)
            log.info("Intercepting AskUserQuestion: %s (%d options)", question[:80], len(options))

            result = await wrapper.request_interaction(
                interaction_id=f"int_{uuid.uuid4().hex[:16]}",
                question=question,
                kind="question",
                options=options,
                allow_freeform=True,
            )
            if result.get("cancelled"):
                reason = "Question cancelled — the user sent a new message. Read it with read_unread."
            else:
                answer = result.get("freeform_response") or result.get("selected_option", "")
                reason = f"User answered: {answer}"

            # Use permissionDecision: deny with the answer as reason.
            # Claude sees the deny reason as the tool result.
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": reason,
                },
            }

        # Skip EnterPlanMode/ExitPlanMode here — handled by can_use_tool.
        if tool_name in ("EnterPlanMode", "ExitPlanMode"):
            return {"continue_": True}

        # --- Standard tool.use emission ---

        await wrapper.emit_tool_use(
            tool_use_id or f"tu_{id(input_data)}",
            tool_name,
            tool_input,
        )

        # Inject unread message reminder so agent checks messages promptly.
        result: dict = {"continue_": True}
        unread = wrapper.chat_mcp.unread_count
        if unread > 0:
            noun = "message" if unread == 1 else "messages"
            result["systemMessage"] = (
                f"You have {unread} unread {noun} from the user. "
                f"Use the read_unread tool to read it."
            )
        return result

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

        try:
            await wrapper.emit_tool_result(
                tool_use_id or f"tu_{id(input_data)}",
                content or "",
                is_error=is_error,
                tool_name=tool_name,
            )
        except Exception:
            log.exception(
                "PostToolUse failed to emit result for %s (id=%s)",
                tool_name, tool_use_id,
            )

        # Inject unread message reminder via systemMessage so the agent
        # is aware of pending user messages during long tool chains.
        unread = wrapper.chat_mcp.unread_count
        if unread > 0:
            return {
                "systemMessage": (
                    f"REMINDER: The user cannot see your text responses. "
                    f"Use mcp__build_chat__send to communicate what you're going "
                    f"to do and what you've done."
                ),
            }

        return {}

    return hook


def make_stop_hook(wrapper: AgentWrapper):
    """Stop hook — emit activity.end to device client."""

    async def hook(input_data, tool_use_id, context):
        await wrapper.emit_activity_end("waiting")
        return {}

    return hook


def make_pre_compact_hook(wrapper: AgentWrapper):
    """PreCompact hook — emit activity delta and inject chat reminder."""

    _CHAT_REMINDER = (
        "\n\nIMPORTANT: You are communicating through a remote chat interface. "
        "The user CANNOT see your text responses. The ONLY way to communicate "
        "with the user is by calling the mcp__build_chat__send tool. "
        "Always use mcp__build_chat__send to tell the user what you're going "
        "to do and what you've done. "
        "When you see REMINDER notes about unread messages, read them promptly "
        "with mcp__build_chat__read_unread."
    )

    async def hook(input_data, tool_use_id, context):
        trigger = input_data.get("trigger", "auto")
        await wrapper.emit_activity_delta("text", f"Compacting context ({trigger})...")

        # Inject chat instructions so they survive compaction.
        existing = input_data.get("custom_instructions") or ""
        reminder = _CHAT_REMINDER
        config = wrapper.config
        if config and config.working_directory:
            reminder += (
                f"\n\nChannel working directory: {config.working_directory}\n"
                "File paths in [[file]] and [[diff]] embeds are resolved relative to this directory. "
                "Use absolute paths or paths relative to this directory, not your shell's cwd."
            )
        input_data["custom_instructions"] = existing + reminder

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


def _read_plan_file(tool_input: dict) -> str:
    """Read plan content for ExitPlanMode approval.

    Checks tool_input['plan'] (injected by normalizeToolInput), then falls
    back to the most recently modified file in ~/.claude/plans/.
    """
    plan = tool_input.get("plan", "")
    if plan:
        return plan

    from pathlib import Path
    plans_dir = Path.home() / ".claude" / "plans"
    if plans_dir.is_dir():
        plan_files = sorted(plans_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
        if plan_files:
            try:
                content = plan_files[0].read_text().strip()
                if content:
                    log.info("Read plan from %s (%d chars)", plan_files[0], len(content))
                    return content
            except OSError as exc:
                log.warning("Failed to read plan file: %s", exc)

    return "The agent has finished designing a plan and is requesting approval."


def make_can_use_tool(wrapper: AgentWrapper):
    """Create a can_use_tool callback for ExitPlanMode/EnterPlanMode approval.

    The CLI dispatches ExitPlanMode through can_use_tool — NOT through
    PreToolUse hooks. This callback handles plan approval via the interaction
    system and auto-allows everything else.
    """

    async def callback(
        tool_name: str,
        tool_input: dict,
        context: ToolPermissionContext,
    ) -> PermissionResultAllow | PermissionResultDeny:
        if tool_name == "EnterPlanMode":
            await wrapper.emit_state_update(plan_mode=True)
            return PermissionResultAllow()

        if tool_name != "ExitPlanMode":
            return PermissionResultAllow()

        log.info("Intercepted ExitPlanMode (can_use_tool)")
        await wrapper.emit_state_update(plan_mode=False)

        plan_content = _read_plan_file(tool_input)

        result = await wrapper.request_interaction(
            interaction_id=f"int_{uuid.uuid4().hex[:16]}",
            question="Review and approve the plan?",
            kind="plan_review",
            options=[
                {"id": "approve", "label": "Approve"},
                {"id": "reject", "label": "Reject"},
            ],
            allow_freeform=True,
            plan=plan_content,
        )

        if result.get("selected_option") == "approve":
            return PermissionResultAllow()

        reason = result.get("freeform_response") or "User rejected the plan."
        return PermissionResultDeny(message=reason)

    return callback


def build_agent_options(
    wrapper: AgentWrapper,
    system_prompt_append: str = "",
) -> ClaudeAgentOptions:
    """Build ClaudeAgentOptions with BAP hooks and Chat MCP tools.

    If *system_prompt_append* is provided it is appended to the default
    Claude Code system prompt via the ``preset`` mechanism so that chat
    instructions and session history appear in the system prompt rather
    than in a user message.
    """
    tools = make_chat_tools(wrapper)
    mcp_server = create_sdk_mcp_server("build_chat", tools=tools)

    system_prompt = (
        {"type": "preset", "preset": "claude_code", "append": system_prompt_append}
        if system_prompt_append
        else None
    )

    return ClaudeAgentOptions(
        setting_sources=["project"],
        permission_mode="default",
        can_use_tool=make_can_use_tool(wrapper),
        mcp_servers={"build_chat": mcp_server},
        hooks={
            "PreToolUse": [HookMatcher(hooks=[make_pre_tool_hook(wrapper)])],
            "PostToolUse": [HookMatcher(hooks=[make_post_tool_hook(wrapper)])],
            "PreCompact": [HookMatcher(hooks=[make_pre_compact_hook(wrapper)])],
            "Stop": [HookMatcher(hooks=[make_stop_hook(wrapper)])],
        },
        stderr=_stderr_logger,
        system_prompt=system_prompt,
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
    active_client: list[ClaudeSDKClient | None] = [None]

    async def on_cancel():
        cancel_event.set()
        client = active_client[0]
        if client:
            try:
                await client.interrupt()
            except Exception:
                pass

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
    if config.working_directory:
        chat_context += (
            f"Channel working directory: {config.working_directory}\n"
            "File paths in [[file]] and [[diff]] embeds are resolved relative to this directory. "
            "Use absolute paths or paths relative to this directory, not your shell's cwd.\n\n"
        )
    if config.chat_instructions:
        chat_context += f"{config.chat_instructions}\n\n"

    # Start the wrapper receive loop (handles incoming chat.message, etc.)
    receive_task = asyncio.create_task(wrapper.run())

    next_prompt: str | None = initial_prompt

    # Build recent history summary for context across sessions.
    def _build_history_context() -> str:
        if not config:
            return ""

        sections: list[str] = []

        # Include recent chat messages so the agent has conversation context.
        if config.chat_history:
            recent_chat = config.chat_history[-20:]
            if recent_chat:
                chat_lines = ["Recent conversation history:"]
                for msg in recent_chat:
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    if len(content) > 300:
                        content = content[:300] + "..."
                    chat_lines.append(f"[{role}] {content}")
                sections.append("\n".join(chat_lines))

        # Include recent activity (tool use, reasoning) so the agent retains
        # context about what it has already done.
        if config.activity_history:
            recent_activity = config.activity_history[-50:]
            if recent_activity:
                activity_lines = ["Recent tool and reasoning activity:"]
                for entry in recent_activity:
                    entry_type = entry.get("type", "")
                    if entry_type == "tool_use":
                        name = entry.get("name", "tool")
                        tool_input = entry.get("input", {})
                        # Summarize input to keep context concise.
                        input_summary = json.dumps(tool_input, default=str)
                        if len(input_summary) > 200:
                            input_summary = input_summary[:200] + "..."
                        activity_lines.append(f"[tool_use] {name}: {input_summary}")
                    elif entry_type == "tool_result":
                        content = entry.get("content", "")
                        is_error = entry.get("is_error", False)
                        if isinstance(content, str) and len(content) > 200:
                            content = content[:200] + "..."
                        status = "error" if is_error else "ok"
                        activity_lines.append(f"[tool_result ({status})] {content}")
                    elif entry_type == "text":
                        text = entry.get("text", "") or entry.get("delta", {}).get("text", "")
                        if text:
                            if len(text) > 200:
                                text = text[:200] + "..."
                            activity_lines.append(f"[reasoning] {text}")
                sections.append("\n".join(activity_lines))

        if not sections:
            return ""
        return "\n\n".join(sections) + "\n\n"

    try:
        # Outer loop: each iteration is a fresh agent context.
        while True:
            # Inject chat context and session history into the system
            # prompt so the agent retains context across restarts and
            # context-window resets.
            history_ctx = _build_history_context()
            system_append = chat_context
            if history_ctx:
                system_append += history_ctx

            options = build_agent_options(wrapper, system_prompt_append=system_append)

            async with ClaudeSDKClient(options=options) as client:
                active_client[0] = client
                # Build initial prompt (chat context / history now live in
                # the system prompt, so the user message is just the trigger).
                if next_prompt:
                    prompt = next_prompt
                else:
                    # Yield to the event loop so forwarded messages from
                    # the server have a chance to be queued by wrapper.run().
                    await asyncio.sleep(0)
                    # Check for any queued messages from before we connected.
                    if wrapper.chat_mcp.has_unread:
                        notification = wrapper.chat_mcp.build_unread_notification()
                        prompt = notification or "Check in with the user."
                    else:
                        # No messages — just notify browser and wait.
                        await wrapper.emit_system_message("Agent is ready.")
                        prompt = None
                next_prompt = None

                # Initial query (skip if no prompt — agent is idle).
                if prompt:
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
                        continue

                    # Atomically check and build notification to avoid race
                    # with concurrent handle_read_unread draining the queue.
                    notification = await wrapper.chat_mcp.drain_unread_notification()
                    if notification:
                        await client.query(notification)
                        async for message in client.receive_response():
                            await handle_response_message(message, wrapper)

            active_client[0] = None
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
