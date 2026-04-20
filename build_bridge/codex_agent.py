#!/usr/bin/env python3
"""Build Agent — Codex CLI integration via app-server + BAP wrapper."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from build_bridge.agent_wrapper import AgentWrapper
from build_bridge.build_chat_bridge import BuildChatBridgeServer
from build_bridge.codex_app_server import CodexAppServerClient, CodexAppServerError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
# Also log to a file for debugging (spawner truncates stderr).
try:
    _log_dir = Path.home() / ".config" / "build" / "logs"
    _log_dir.mkdir(parents=True, exist_ok=True)
    _fh = logging.FileHandler(_log_dir / "codex_agent.log")
    _fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(_fh)
except Exception:
    pass
log = logging.getLogger(__name__)

CHAT_CONTEXT = (
    "IMPORTANT: You are connected to the user through Build's remote chat bridge.\n"
    "- The user CANNOT see your raw harness output. Your normal text replies go nowhere.\n"
    "- The ONLY way to reply to the user is by calling the build-chat MCP server's `send` tool.\n"
    "- After calling `read_unread` you MUST call `send` at least once before ending the turn — "
    "even a short acknowledgment. Never end a turn silently.\n"
    "- Use `send` while you're working to keep the user looped in on progress, important findings, and decisions.\n"
    "- Keep user-visible replies concise and natural.\n"
    "- Treat planning mode instructions from the system as authoritative.\n"
)

PING_INTERVAL_S = 5.0
HANG_THRESHOLD_S = 120.0
WATCHDOG_INTERVAL_S = 15.0
QUOTA_ERROR_DEDUP_S = 60.0
CHAT_SERVER_NAME = "build-chat"
CHAT_TOOL_NAMES = frozenset({"read_unread", "send"})


def _build_history_context(chat_history: list[dict[str, Any]]) -> str:
    if not chat_history:
        return ""

    recent = chat_history[-20:]
    lines = ["Recent conversation history:"]
    for message in recent:
        role = message.get("role", "unknown")
        content = str(message.get("content", ""))
        if len(content) > 300:
            content = content[:300] + "..."
        lines.append(f"[{role}] {content}")
    return "\n".join(lines) + "\n\n"


def _render_plan_text(explanation: str | None, plan: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    if explanation:
        lines.append(explanation)
        lines.append("")
    for step in plan:
        status = step.get("status", "pending")
        status_text = {
            "completed": "[x]",
            "inProgress": "[>]",
            "pending": "[ ]",
        }.get(status, "[ ]")
        lines.append(f"{status_text} {step.get('step', '')}".rstrip())
    return "\n".join(lines).strip()


def _codex_input(text: str) -> list[dict[str, Any]]:
    return [{"type": "text", "text": text, "text_elements": []}]


def _is_chat_bridge_tool(item: dict[str, Any]) -> bool:
    return (
        item.get("type") == "mcpToolCall"
        and item.get("server") == CHAT_SERVER_NAME
        and item.get("tool") in CHAT_TOOL_NAMES
    )


def _tool_name_for_item(item: dict[str, Any]) -> str | None:
    item_type = item.get("type")
    if item_type == "commandExecution":
        return "Bash"
    if item_type == "fileChange":
        return "ApplyPatch"
    if item_type == "mcpToolCall":
        return f"mcp:{item.get('server')}:{item.get('tool')}"
    if item_type == "dynamicToolCall":
        return str(item.get("tool", "DynamicTool"))
    if item_type == "webSearch":
        return "WebSearch"
    if item_type == "imageView":
        return "ViewImage"
    return None


def _tool_input_for_item(item: dict[str, Any]) -> dict[str, Any]:
    item_type = item.get("type")
    if item_type == "commandExecution":
        return {
            "command": item.get("command"),
            "cwd": item.get("cwd"),
            "commandActions": item.get("commandActions"),
        }
    if item_type == "fileChange":
        return {"changes": item.get("changes", [])}
    if item_type == "mcpToolCall":
        return {
            "server": item.get("server"),
            "tool": item.get("tool"),
            "arguments": item.get("arguments"),
        }
    if item_type == "dynamicToolCall":
        return {"arguments": item.get("arguments")}
    if item_type == "webSearch":
        return {"query": item.get("query"), "action": item.get("action")}
    if item_type == "imageView":
        return {"path": item.get("path")}
    return {}


def _text_from_dynamic_content(content_items: list[dict[str, Any]] | None) -> str:
    if not content_items:
        return ""
    parts = []
    for item in content_items:
        if item.get("type") == "inputText":
            parts.append(item.get("text", ""))
        elif item.get("type") == "inputImage":
            parts.append(f"[image] {item.get('imageUrl', '')}")
    return "\n".join(part for part in parts if part)


def _text_from_agent_message(item: dict[str, Any]) -> str:
    """Extract plain text from an agentMessage item's content.

    Codex ships agent replies as either:
      - a `text` string field, or
      - a `content` array with {"type": "text", "text": "..."} blocks.
    """
    direct = item.get("text")
    if isinstance(direct, str) and direct:
        return direct
    content = item.get("content")
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") in ("text", "output_text") and block.get("text"):
            parts.append(str(block.get("text")))
    return "\n".join(parts)


def _tool_result_from_item(item: dict[str, Any], buffered_output: str) -> tuple[str, bool]:
    item_type = item.get("type")
    if item_type == "commandExecution":
        text = buffered_output or item.get("aggregatedOutput") or ""
        exit_code = item.get("exitCode")
        status = item.get("status")
        is_error = status == "failed" or (exit_code not in (None, 0))
        return text, is_error
    if item_type == "fileChange":
        if buffered_output:
            return buffered_output, False
        return json.dumps(item.get("changes", []), indent=2), False
    if item_type == "mcpToolCall":
        error = item.get("error")
        if error:
            return str(error.get("message", error)), True
        result = item.get("result") or {}
        content = result.get("content", [])
        text = "\n".join(json.dumps(part, default=str) for part in content) if content else ""
        return text, item.get("status") == "failed"
    if item_type == "dynamicToolCall":
        return _text_from_dynamic_content(item.get("contentItems")), not bool(item.get("success", True))
    if item_type == "webSearch":
        return item.get("query", ""), False
    if item_type == "imageView":
        return item.get("path", ""), False
    return "", False


def _write_codex_config(
    *,
    codex_dir: Path,
    bridge_socket: str,
    bridge_token: str,
    trusted_project: str | None,
) -> None:
    config_lines = [
        f"[mcp_servers.{CHAT_SERVER_NAME}]",
        f'command = "{sys.executable}"',
        'args = ["-m", "build_bridge.build_chat_bridge"]',
        "",
        f"[mcp_servers.{CHAT_SERVER_NAME}.env]",
        f'BUILD_CHAT_BRIDGE_SOCKET = "{bridge_socket}"',
        f'BUILD_CHAT_BRIDGE_TOKEN = "{bridge_token}"',
        "",
    ]
    if trusted_project:
        config_lines.extend([
            f'[projects."{trusted_project}"]',
            'trust_level = "trusted"',
            "",
        ])
    (codex_dir / "config.toml").write_text("\n".join(config_lines), encoding="utf-8")


def create_isolated_codex_home(
    *,
    bridge_socket: str,
    bridge_token: str,
    trusted_project: str | None = None,
    channel_id: str | None = None,
) -> Path:
    # Use a persistent directory keyed by channel ID so Codex rollout files
    # and session state survive agent restarts, enabling thread/resume.
    if channel_id:
        runtime_home = Path.home() / ".config" / "build" / "codex-homes" / channel_id
    else:
        runtime_home = Path(tempfile.mkdtemp(prefix="build-codex-home-"))
    codex_dir = runtime_home / ".codex"
    codex_dir.mkdir(parents=True, exist_ok=True)

    auth_source = Path.home() / ".codex" / "auth.json"
    if auth_source.exists():
        shutil.copy2(auth_source, codex_dir / "auth.json")

    _write_codex_config(
        codex_dir=codex_dir,
        bridge_socket=bridge_socket,
        bridge_token=bridge_token,
        trusted_project=trusted_project,
    )
    return runtime_home


class CodexHarnessRuntime:
    def __init__(
        self,
        *,
        wrapper: AgentWrapper,
        client: CodexAppServerClient,
        config: Any,
        model: str,
        initial_prompt: str | None,
        working_directory: str,
        effort: str = "",
        resume_thread_id: str | None = None,
        auto_approve_tools: bool = False,
    ) -> None:
        self.wrapper = wrapper
        self.client = client
        self.config = config
        self.model = model
        self.effort = effort
        self.initial_prompt = initial_prompt
        self.working_directory = working_directory
        self.resume_thread_id = resume_thread_id
        # Prefer the wrapper's configured value (DB-backed, authoritative).
        # Fall back to the CLI arg for the first fraction of a second before
        # wrapper.connect() returns.
        self.auto_approve_tools: bool = bool(
            getattr(config, "auto_approve_tools", False) or auto_approve_tools
        )
        self.thread_id: str | None = None
        self.turn_id: str | None = None
        self.plan_mode = False
        self.latest_plan_text: str | None = None
        self.latest_plan_turn_id: str | None = None
        self._plan_review_task: asyncio.Task | None = None
        self._tool_outputs: dict[str, list[str]] = {}
        self._tool_names: dict[str, str] = {}
        self._cancel_event = asyncio.Event()
        self._last_activity = time.monotonic()
        self._unread_notified = False
        self._last_quota_error_ts = 0.0
        self._hang_detected = False
        # Signatures of (tool_name, command-or-path) that the user has
        # approved "for session" — repeat requests within the same agent
        # lifetime auto-accept without prompting again.
        self._session_approved: set[str] = set()
        # Tracks whether Codex called the build-chat `send` MCP tool during
        # the current turn. If it didn't, we fall back to emitting the
        # terminal `agentMessage` content as the chat response so the user
        # sees Codex's reply even when the model skips the tool call.
        self._turn_used_send: bool = False

    async def run(self) -> None:
        self._register_handlers()
        log.info("Starting Codex app-server...")
        await self.client.start()
        log.info("Codex app-server started, initializing...")
        await self.client.initialize(client_name="build-bridge", client_version="0.1.0")
        log.info("Codex app-server initialized")

        # Try to resume existing thread or start a new one.
        if self.resume_thread_id:
            try:
                log.info("Resuming Codex thread %s...", self.resume_thread_id[:8])
                await self.client.send_request("thread/resume", {
                    "threadId": self.resume_thread_id,
                    "cwd": self.working_directory,
                    "model": self.model,
                })
                self.thread_id = self.resume_thread_id
                log.info("Resumed Codex thread %s", self.thread_id[:8])
            except Exception as exc:
                log.warning("Thread resume failed, starting fresh: %s", exc)
                self.resume_thread_id = None
                await self.wrapper.emit_system_message("Context cleared — could not resume previous session.")
                self.thread_id = None

        if not self.thread_id:
            log.info("Starting new Codex thread...")
            thread_params: dict[str, Any] = {
                "cwd": self.working_directory,
                "model": self.model,
                "approvalPolicy": "on-request",
                "approvalsReviewer": "user",
                "sandbox": "workspace-write",
                "baseInstructions": self.config.system_prompt or None,
                "developerInstructions": self._developer_instructions(),
                "ephemeral": False,
                "experimentalRawEvents": False,
                "persistExtendedHistory": True,
            }
            if self.effort:
                thread_params["effort"] = self.effort
            thread = await self.client.send_request("thread/start", thread_params)
            self.thread_id = thread["thread"]["id"]

        # Persist thread_id as resume cursor.
        try:
            await self.wrapper.emit_resume_cursor(self.thread_id)
        except Exception:
            log.debug("Failed to persist resume cursor", exc_info=True)

        initial = self._initial_turn_text()
        if initial:
            await self._start_or_steer(initial)
        else:
            await self.wrapper.emit_system_message("Agent is ready.")

        ping_task = asyncio.create_task(self._ping_loop())
        watchdog_task = asyncio.create_task(self._hang_watchdog_loop())
        try:
            while self.wrapper.is_connected:
                if self._cancel_event.is_set():
                    await self._interrupt_turn()
                    self._cancel_event.clear()
                    continue

                # Race notification wait against cancel event so cancel is
                # processed immediately rather than after the timeout expires.
                notification_task = asyncio.create_task(
                    self._next_notification_text(timeout=1.0)
                )
                cancel_wait = asyncio.create_task(self._cancel_event.wait())
                done, _ = await asyncio.wait(
                    {notification_task, cancel_wait},
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if cancel_wait in done:
                    notification_task.cancel()
                    try:
                        await notification_task
                    except asyncio.CancelledError:
                        pass
                    continue  # Loop back to process cancel at top.
                else:
                    cancel_wait.cancel()

                notification = notification_task.result()
                if not notification:
                    continue
                await self._start_or_steer(notification)
        finally:
            for task in (ping_task, watchdog_task):
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    def cancel(self) -> None:
        self._cancel_event.set()

    async def shutdown(self) -> None:
        if self._plan_review_task:
            self._plan_review_task.cancel()
            try:
                await self._plan_review_task
            except asyncio.CancelledError:
                pass
        await self.client.stop()

    def _register_handlers(self) -> None:
        self.client.on_notification("turn/started", self._on_turn_started)
        self.client.on_notification("turn/completed", self._on_turn_completed)
        self.client.on_notification("item/agentMessage/delta", self._on_agent_message_delta)
        self.client.on_notification("item/reasoning/textDelta", self._on_reasoning_delta)
        self.client.on_notification("item/reasoning/summaryTextDelta", self._on_reasoning_delta)
        self.client.on_notification("item/plan/delta", self._on_plan_delta)
        self.client.on_notification("turn/plan/updated", self._on_turn_plan_updated)
        self.client.on_notification("item/started", self._on_item_started)
        self.client.on_notification("item/completed", self._on_item_completed)
        self.client.on_notification("item/commandExecution/outputDelta", self._on_command_output_delta)
        self.client.on_notification("item/fileChange/outputDelta", self._on_file_change_output_delta)
        self.client.on_notification("item/mcpToolCall/progress", self._on_mcp_progress)
        self.client.on_notification("error", self._on_error)
        self.client.on_notification("account/rateLimits/updated", self._on_rate_limits_updated)

        self.client.on_request("item/commandExecution/requestApproval", self._on_command_approval)
        self.client.on_request("item/fileChange/requestApproval", self._on_file_change_approval)
        self.client.on_request("item/permissions/requestApproval", self._on_permissions_approval)
        self.client.on_request("item/tool/requestUserInput", self._on_tool_request_user_input)
        self.client.on_request("mcpServer/elicitation/request", self._on_mcp_elicitation)

    def _developer_instructions(self) -> str:
        instructions = CHAT_CONTEXT
        if self.config.chat_instructions:
            instructions += "\n" + self.config.chat_instructions
        return instructions.strip()

    def _initial_turn_text(self) -> str | None:
        history = _build_history_context(self.config.chat_history)
        if self.initial_prompt:
            return history + self.initial_prompt
        if self.wrapper.chat_mcp.has_unread:
            unread = self.wrapper.chat_mcp.build_unread_notification() or ""
            return history + unread
        return None  # no turn needed — wait for messages

    async def _next_notification_text(self, timeout: float) -> str | None:
        parts: list[str] = []

        # Check if previously notified unread messages have been drained.
        if self._unread_notified and not self.wrapper.chat_mcp.has_unread:
            self._unread_notified = False

        if not self._unread_notified:
            # Wait for new unread messages (blocks up to timeout).
            has_unread = await self.wrapper.chat_mcp.wait_for_unread(timeout=timeout)
            if has_unread:
                unread = await self.wrapper.chat_mcp.drain_unread_notification()
                if unread:
                    parts.append(unread)
                    self._unread_notified = True
        elif not parts:
            # Already notified, queue not yet drained — just sleep to avoid busy-wait.
            await asyncio.sleep(timeout)
            return None

        text = "\n\n".join(part for part in parts if part).strip()
        return text or None

    async def _start_or_steer(self, text: str) -> None:
        if not self.thread_id:
            return

        # Apply pending per-turn settings from the wrapper.
        if self.wrapper.pending_model and self.wrapper.pending_model != self.model:
            self.model = self.wrapper.pending_model
            log.info("Model switched to %s", self.model)
        self.wrapper.pending_model = None
        if self.wrapper.pending_effort and self.wrapper.pending_effort != self.effort:
            self.effort = self.wrapper.pending_effort
            log.info("Effort changed to %s", self.effort)
        self.wrapper.pending_effort = None
        if self.wrapper.pending_plan_mode is not None and self.wrapper.pending_plan_mode != self.plan_mode:
            self.plan_mode = self.wrapper.pending_plan_mode
            await self.wrapper.emit_state_update(plan_mode=self.plan_mode)
            log.info("Plan mode set to %s", self.plan_mode)
        self.wrapper.pending_plan_mode = None

        params: dict[str, Any] = {
            "threadId": self.thread_id,
            "input": _codex_input(text),
            "cwd": self.working_directory,
            "approvalPolicy": "on-request",
            "approvalsReviewer": "user",
            "sandboxPolicy": {
                "type": "workspaceWrite",
                "writableRoots": [self.working_directory],
                "readOnlyAccess": {"type": "fullAccess"},
                "networkAccess": True,
                "excludeTmpdirEnvVar": False,
                "excludeSlashTmp": False,
            },
            "model": self.model,
        }
        if self.effort:
            params["effort"] = self.effort
        if self.plan_mode:
            params["collaborationMode"] = {
                "mode": "plan",
                "settings": {"model": self.model},
            }

        self._last_activity = time.monotonic()
        if self.turn_id:
            try:
                await self.client.send_request("turn/steer", {
                    "threadId": self.thread_id,
                    "expectedTurnId": self.turn_id,
                    "input": params["input"],
                })
                return
            except CodexAppServerError:
                self.turn_id = None

        # turn/start can run for minutes — rely on _hang_watchdog_loop for liveness.
        await self.client.send_request("turn/start", params, timeout=None)

    async def _interrupt_turn(self) -> None:
        if not self.thread_id or not self.turn_id:
            return
        try:
            await self.client.send_request("turn/interrupt", {
                "threadId": self.thread_id,
                "turnId": self.turn_id,
            }, timeout=10.0)
        except CodexAppServerError as exc:
            log.warning("Failed to interrupt Codex turn: %s", exc)

    async def _ping_loop(self) -> None:
        while True:
            await asyncio.sleep(PING_INTERVAL_S)
            if self.turn_id and time.monotonic() - self._last_activity >= PING_INTERVAL_S:
                await self.wrapper.emit_activity_ping()

    async def _hang_watchdog_loop(self) -> None:
        """Detect hung Codex subprocess during active turns and kill it.

        Exits the agent process with a non-zero code so agent_spawner's
        auto-restart fires. Killing just the Codex subprocess is not enough:
        the agent process's run() loop would keep running with a dead client,
        and run_agent() catches the downstream CodexAppServerError as a clean
        exit — auto-restart only triggers on non-zero process exit.
        """
        while True:
            await asyncio.sleep(WATCHDOG_INTERVAL_S)
            if not self.turn_id:
                continue
            idle = time.monotonic() - self._last_activity
            if idle > HANG_THRESHOLD_S:
                log.error(
                    "Codex subprocess hung: no activity for %.1fs during turn %s. Exiting for auto-restart.",
                    idle, self.turn_id,
                )
                self._dump_hang_snapshot(idle)
                self._hang_detected = True
                try:
                    await self.client.stop()
                except Exception:
                    log.exception("Error stopping hung Codex client")
                try:
                    await self.wrapper.emit_system_message(
                        "Codex subprocess stopped responding — restarting agent."
                    )
                except Exception:
                    log.debug("Failed to emit hang notice", exc_info=True)
                os._exit(1)

    def _dump_hang_snapshot(self, idle: float) -> None:
        """Dump runtime + recent message state when a hang is detected."""
        try:
            pending = self.client.pending_requests()
        except Exception:
            pending = []
        log.error(
            "Hang snapshot: thread_id=%s turn_id=%s idle=%.1fs pending_requests=%s model=%s plan_mode=%s",
            self.thread_id, self.turn_id, idle, pending, self.model, self.plan_mode,
        )
        try:
            recent = self.client.recent_messages()
        except Exception:
            log.exception("Failed to collect recent messages")
            return
        if not recent:
            log.error("Hang snapshot: ring buffer empty")
            return
        now = time.monotonic()
        log.error("Hang snapshot: last %d app-server messages (newest first):", len(recent))
        for ts, message in reversed(recent[-30:]):
            age = now - ts
            try:
                method = message.get("method") or "(response)"
                params = message.get("params") or {}
                blob = json.dumps(params, default=str)[:500]
                log.error("  -%.1fs method=%s id=%s params=%s", age, method, message.get("id"), blob)
            except Exception:
                log.error("  -%.1fs raw=%r", age, message)

    async def _on_turn_started(self, params: dict[str, Any]) -> None:
        turn = params.get("turn", {})
        self.turn_id = turn.get("id")
        self._last_activity = time.monotonic()
        self._turn_used_send = False

    async def _on_turn_completed(self, params: dict[str, Any]) -> None:
        turn = params.get("turn", {})
        self.turn_id = None
        self._last_activity = time.monotonic()
        self._unread_notified = False
        reason = "error" if turn.get("status") == "failed" else "waiting"
        await self.wrapper.emit_activity_end(reason)

        if self.plan_mode and self.latest_plan_text and self.latest_plan_turn_id == turn.get("id"):
            if not self._plan_review_task or self._plan_review_task.done():
                self._plan_review_task = asyncio.create_task(self._handle_plan_review())

    async def _handle_plan_review(self) -> None:
        result = await self.wrapper.request_interaction(
            interaction_id=f"int_{uuid.uuid4().hex[:16]}",
            question="Review and approve the plan?",
            kind="plan_review",
            options=[
                {"id": "approve", "label": "Approve"},
                {"id": "reject", "label": "Reject"},
            ],
            allow_freeform=True,
            plan=self.latest_plan_text,
        )

        if result.get("cancelled"):
            return

        if result.get("selected_option") == "approve":
            self.plan_mode = False
            await self.wrapper.emit_state_update(plan_mode=False)
            await self._start_or_steer("The user approved the plan. Proceed with the implementation.")
            return

        feedback = result.get("freeform_response") or "Revise the plan based on the user's feedback."
        await self._start_or_steer(f"The user rejected the plan. Revise it.\n\nFeedback: {feedback}")

    async def _on_agent_message_delta(self, params: dict[str, Any]) -> None:
        self._last_activity = time.monotonic()
        delta = params.get("delta", "")
        if delta:
            await self.wrapper.emit_activity_delta("text", delta)

    async def _on_reasoning_delta(self, params: dict[str, Any]) -> None:
        self._last_activity = time.monotonic()
        delta = params.get("delta", "")
        if delta:
            await self.wrapper.emit_activity_delta("text", delta)

    async def _on_plan_delta(self, params: dict[str, Any]) -> None:
        self._last_activity = time.monotonic()
        delta = params.get("delta", "")
        if delta:
            await self.wrapper.emit_activity_delta("text", delta)

    async def _on_turn_plan_updated(self, params: dict[str, Any]) -> None:
        plan = params.get("plan", []) or []
        self.latest_plan_text = _render_plan_text(
            params.get("explanation"),
            plan,
        )
        self.latest_plan_turn_id = params.get("turnId")

        # Mirror Codex's plan into the same shape Claude Code's TodoWrite tool
        # uses, so the dashboard's sidebar Tasks section populates identically
        # across harnesses. Emitted as a synthetic TodoWrite tool_use — no new
        # BAP event type needed, and the browser treats it as a normal todo
        # update (see channelTodos handling in dashboard.html).
        turn_id = self.latest_plan_turn_id or "turn"
        todos = []
        for i, step in enumerate(plan):
            raw_status = (step.get("status") or "pending").lower()
            status = {
                "completed": "completed",
                "inprogress": "in_progress",
                "in_progress": "in_progress",
                "pending": "pending",
            }.get(raw_status, "pending")
            todos.append({
                "id": f"{turn_id}:{i}",
                "content": step.get("step", "") or "",
                "status": status,
            })
        try:
            await self.wrapper.emit_tool_use(
                f"codex-plan-{turn_id}-{uuid.uuid4().hex[:8]}",
                "TodoWrite",
                {"todos": todos},
            )
        except Exception:
            log.debug("Failed to emit synthetic TodoWrite for Codex plan", exc_info=True)

    async def _on_item_started(self, params: dict[str, Any]) -> None:
        self._last_activity = time.monotonic()
        item = params.get("item", {})
        if _is_chat_bridge_tool(item):
            return

        tool_name = _tool_name_for_item(item)
        if not tool_name:
            return

        item_id = item.get("id")
        if not item_id:
            return

        self._tool_names[item_id] = tool_name
        self._tool_outputs.setdefault(item_id, [])
        await self.wrapper.emit_tool_use(item_id, tool_name, _tool_input_for_item(item))

    async def _on_item_completed(self, params: dict[str, Any]) -> None:
        self._last_activity = time.monotonic()
        item = params.get("item", {})

        # Track when Codex calls our own build-chat `send` tool so we can
        # decide whether the terminal agentMessage needs fallback surfacing.
        if _is_chat_bridge_tool(item) and item.get("tool") == "send":
            self._turn_used_send = True

        if _is_chat_bridge_tool(item):
            return

        # agentMessage is Codex's direct text response. Modern Codex emits
        # the full content in item/completed with no streaming deltas, so
        # if we don't read it here we lose the reply entirely. Surface it
        # to the chat when the model didn't use the send tool this turn.
        if item.get("type") == "agentMessage":
            text = _text_from_agent_message(item)
            log.info(
                "agentMessage completed: used_send=%s text_len=%d keys=%s raw=%s",
                self._turn_used_send, len(text), list(item.keys()),
                json.dumps(item, default=str)[:600],
            )
            if not self._turn_used_send and text.strip():
                log.info(
                    "Surfacing agentMessage fallback (%d chars) — no send call this turn",
                    len(text),
                )
                try:
                    await self.wrapper.chat_mcp.handle_send(text)
                except Exception:
                    log.exception("Failed to surface agentMessage fallback")
            return

        item_id = item.get("id")
        tool_name = self._tool_names.pop(item_id, None) if item_id else None
        if not item_id or not tool_name:
            return

        buffered = "".join(self._tool_outputs.pop(item_id, []))
        content, is_error = _tool_result_from_item(item, buffered)
        await self.wrapper.emit_tool_result(
            item_id,
            content,
            is_error=is_error,
            tool_name=tool_name,
        )

    async def _on_command_output_delta(self, params: dict[str, Any]) -> None:
        item_id = params.get("itemId")
        delta = params.get("delta", "")
        if item_id and delta:
            self._tool_outputs.setdefault(item_id, []).append(delta)

    async def _on_file_change_output_delta(self, params: dict[str, Any]) -> None:
        item_id = params.get("itemId")
        delta = params.get("delta", "")
        if item_id and delta:
            self._tool_outputs.setdefault(item_id, []).append(delta)

    async def _on_mcp_progress(self, params: dict[str, Any]) -> None:
        message = params.get("message", "")
        self._last_activity = time.monotonic()
        if message:
            await self.wrapper.emit_activity_delta("text", message)

    async def _on_rate_limits_updated(self, params: dict[str, Any]) -> None:
        rate_limits = params.get("rateLimits", {})
        for window_key in ("primary", "secondary"):
            window = rate_limits.get(window_key)
            if not window:
                continue
            if window.get("usedPercent", 0) >= 100:
                now = time.monotonic()
                if now - self._last_quota_error_ts < QUOTA_ERROR_DEDUP_S:
                    return
                self._last_quota_error_ts = now
                resets_at = window.get("resetsAt")
                if resets_at:
                    from datetime import datetime, timezone
                    reset_time = datetime.fromtimestamp(resets_at, tz=timezone.utc).astimezone()
                    msg = f"Usage limit reached. Resets at {reset_time.strftime('%I:%M %p')}."
                else:
                    msg = "Usage limit reached."
                log.warning("Codex quota exhausted: %s", msg)
                await self.wrapper._send_error(
                    code="quota_exceeded",
                    message=msg,
                    fatal=True,
                )
                return

    async def _on_error(self, params: dict[str, Any]) -> None:
        error = params.get("error", params)
        message = error.get("message", str(error)) if isinstance(error, dict) else str(error)

        log.error("Codex app-server error: %s", message)

        msg_lower = message.lower()

        # Detect auth errors and surface them to the user.
        auth_keywords = ("refresh token", "sign in again", "token_expired", "401 Unauthorized", "authentication")
        if any(kw.lower() in msg_lower for kw in auth_keywords):
            await self.wrapper._send_error(
                code="auth_expired",
                message="OpenAI authentication expired. Run `codex auth` in your terminal to sign in again.",
                fatal=True,
            )
            return

        # Detect usage limit errors (e.g. from Codex's internal compact task)
        # and treat like quota — dedup on a time window so transient storms
        # don't spam, but later unrelated errors aren't permanently silenced.
        usage_keywords = ("usage limit", "hit your usage limit", "upgrade to pro", "purchase more credits")
        if any(kw in msg_lower for kw in usage_keywords):
            now = time.monotonic()
            if now - self._last_quota_error_ts < QUOTA_ERROR_DEDUP_S:
                return
            self._last_quota_error_ts = now
            await self.wrapper._send_error(
                code="quota_exceeded",
                message=message,
                fatal=True,
            )
            return

        await self.wrapper._send_error(
            code="codex_error",
            message=message,
            fatal=False,
        )

    async def _on_command_approval(self, _request_id: str | int, params: dict[str, Any]) -> dict[str, Any]:
        log.info("Command approval params: %s", json.dumps(params, default=str)[:500])
        command = params.get("command") or ""

        # Channel is in auto-approve mode — rubber-stamp everything.
        if self.auto_approve_tools:
            return {"decision": "accept"}

        # Session allowlist: user already said "Always Approve" for this
        # command during this agent's lifetime.
        sig = f"cmd:{command}"
        if sig in self._session_approved:
            return {"decision": "accept"}

        reason = params.get("reason") or ""
        cwd = params.get("cwd") or ""
        parts = ["**Run command?**"]
        if command:
            parts.append(f"```\n{command}\n```")
        if cwd:
            parts.append(f"Working directory: `{cwd}`")
        if reason:
            parts.append(f"Reason: {reason}")
        question = "\n\n".join(parts)
        result = await self.wrapper.request_interaction(
            interaction_id=f"int_{uuid.uuid4().hex[:16]}",
            question=question,
            kind="approval",
            options=[
                {"id": "accept", "label": "Approve"},
                {"id": "acceptForSession", "label": "Approve for session"},
                {"id": "decline", "label": "Reject"},
            ],
            allow_freeform=False,
        )
        selected = result.get("selected_option") or "decline"
        if selected == "acceptForSession":
            self._session_approved.add(sig)
        return {"decision": selected}

    async def _on_file_change_approval(self, _request_id: str | int, params: dict[str, Any]) -> dict[str, Any]:
        log.info("File change approval params: %s", json.dumps(params, default=str)[:1000])
        changes = params.get("changes") or []

        if self.auto_approve_tools:
            return {"decision": "accept"}

        # Session allowlist — key on the set of file paths being changed.
        paths = sorted({(c.get("filePath") or c.get("path") or "") for c in changes if c.get("filePath") or c.get("path")})
        sig = f"file:{','.join(paths)}"
        if sig and sig in self._session_approved:
            return {"decision": "accept"}

        reason = params.get("reason") or ""
        parts = ["**Approve file changes?**"]
        if reason:
            parts.append(reason)
        for change in changes[:10]:  # Limit to 10 files.
            path = change.get("filePath") or change.get("path") or ""
            diff = change.get("diff") or change.get("patch") or ""
            if path:
                if diff:
                    parts.append(f"`{path}`\n```diff\n{diff[:500]}\n```")
                else:
                    parts.append(f"`{path}`")
        question = "\n\n".join(parts)
        result = await self.wrapper.request_interaction(
            interaction_id=f"int_{uuid.uuid4().hex[:16]}",
            question=question,
            kind="approval",
            options=[
                {"id": "accept", "label": "Approve"},
                {"id": "acceptForSession", "label": "Approve for session"},
                {"id": "decline", "label": "Reject"},
            ],
            allow_freeform=False,
        )
        selected = result.get("selected_option") or "decline"
        if selected == "acceptForSession" and sig:
            self._session_approved.add(sig)
        return {"decision": selected}

    async def _on_permissions_approval(self, _request_id: str | int, params: dict[str, Any]) -> dict[str, Any]:
        log.info("Permissions approval params: %s", json.dumps(params, default=str)[:500])

        if self.auto_approve_tools:
            return {"permissions": params.get("permissions", {}), "scope": "session"}

        # Session allowlist — key on the set of permission names requested.
        perm_keys = sorted((params.get("permissions") or {}).keys())
        sig = f"perm:{','.join(perm_keys)}" if perm_keys else ""
        if sig and sig in self._session_approved:
            return {"permissions": params.get("permissions", {}), "scope": "session"}

        question = params.get("reason") or "Approve requested permissions?"
        result = await self.wrapper.request_interaction(
            interaction_id=f"int_{uuid.uuid4().hex[:16]}",
            question=question,
            kind="approval",
            options=[
                {"id": "accept", "label": "Approve"},
                {"id": "acceptForSession", "label": "Approve for session"},
                {"id": "decline", "label": "Reject"},
            ],
            allow_freeform=False,
        )
        selected = result.get("selected_option")
        if selected in ("accept", "acceptForSession"):
            if selected == "acceptForSession" and sig:
                self._session_approved.add(sig)
            return {"permissions": params.get("permissions", {}), "scope": "session"}
        return {"permissions": {"network": None, "fileSystem": None}, "scope": "turn"}

    async def _on_tool_request_user_input(self, _request_id: str | int, params: dict[str, Any]) -> dict[str, Any]:
        answers: dict[str, dict[str, list[str]]] = {}
        for question in params.get("questions", []):
            options = [
                {"id": opt.get("label", ""), "label": opt.get("label", "")}
                for opt in (question.get("options") or [])
            ]
            result = await self.wrapper.request_interaction(
                interaction_id=f"int_{uuid.uuid4().hex[:16]}",
                question=question.get("question", ""),
                kind="question",
                options=options,
                allow_freeform=bool(question.get("isOther")),
            )
            value = result.get("selected_option") or result.get("freeform_response") or ""
            answers[question.get("id", str(uuid.uuid4()))] = {"answers": [value] if value else []}
        return {"answers": answers}

    async def _on_mcp_elicitation(self, _request_id: str | int, params: dict[str, Any]) -> dict[str, Any]:
        meta = params.get("_meta") or {}
        server_name = params.get("serverName", "")
        approval_kind = meta.get("codex_approval_kind")
        persist_options = meta.get("persist") or []

        def _accept_with_persist() -> dict[str, Any]:
            persist = "always" if "always" in persist_options else ("session" if "session" in persist_options else None)
            content: dict[str, Any] = {}
            if persist:
                content["persist"] = persist
            return {"action": "accept", "content": content, "_meta": meta}

        # Codex CLI's new mcp_tool_call approval flow routes through elicitation.
        # Our build-chat bridge is an internal MCP server — always auto-approve
        # its tool calls regardless of channel flag (it's our own server and
        # needs no user confirmation).
        if server_name == CHAT_SERVER_NAME and approval_kind == "mcp_tool_call":
            log.info("Auto-approving build-chat mcp_tool_call")
            return _accept_with_persist()

        # Third-party MCP tool calls follow the channel flag.
        if approval_kind == "mcp_tool_call":
            if self.auto_approve_tools:
                return _accept_with_persist()
            tool_params = meta.get("tool_params") or {}
            tool_name = meta.get("tool_name") or tool_params.get("tool") or ""
            sig = f"mcp:{server_name}:{tool_name}"
            if sig in self._session_approved:
                return _accept_with_persist()
            tool_desc = meta.get("tool_description") or ""
            display_params = meta.get("tool_params_display") or []
            parts = [f"**Allow MCP server `{server_name}` to run tool `{tool_name}`?**"]
            if tool_desc:
                parts.append(tool_desc)
            for p in display_params[:10]:
                parts.append(f"`{p.get('display_name') or p.get('name')}`: {p.get('value')}")
            question = "\n\n".join(parts)
            result = await self.wrapper.request_interaction(
                interaction_id=f"int_{uuid.uuid4().hex[:16]}",
                question=question,
                kind="approval",
                options=[
                    {"id": "accept", "label": "Approve"},
                    {"id": "acceptForSession", "label": "Approve for session"},
                    {"id": "decline", "label": "Reject"},
                ],
                allow_freeform=False,
            )
            selected = result.get("selected_option")
            if selected == "acceptForSession":
                self._session_approved.add(sig)
                return _accept_with_persist()
            if selected == "accept":
                return _accept_with_persist()
            return {"action": "decline", "content": None, "_meta": meta}

        # Non-tool-call elicitations (real user input forms) — always route
        # to the user regardless of auto-approve.
        message = params.get("message", "MCP server is requesting input.")
        result = await self.wrapper.request_interaction(
            interaction_id=f"int_{uuid.uuid4().hex[:16]}",
            question=message,
            kind="question",
            options=[
                {"id": "accept", "label": "Approve"},
                {"id": "decline", "label": "Reject"},
            ],
            allow_freeform=True,
        )
        if result.get("selected_option") == "accept":
            content_any: Any = None
            freeform = result.get("freeform_response")
            if freeform:
                try:
                    content_any = json.loads(freeform)
                except json.JSONDecodeError:
                    content_any = {"value": freeform}
            return {"action": "accept", "content": content_any, "_meta": meta}
        return {"action": "decline", "content": None, "_meta": meta}


async def run_agent(
    port: int = 9783,
    host: str = "127.0.0.1",
    model: str = "gpt-5.4",
    initial_prompt: str | None = None,
    agent_id: str | None = None,
    working_directory: str | None = None,
    effort: str = "",
    resume_session: str | None = None,
    auto_approve_tools: bool = False,
) -> None:
    workdir = working_directory or os.getcwd()
    if os.path.isdir(workdir):
        os.chdir(workdir)
        log.info("Working directory: %s", workdir)

    # Runtime reference — set once created, used by on_cancel for immediate
    # interrupt without polling.
    _runtime_ref: list[CodexHarnessRuntime | None] = [None]

    async def on_cancel() -> None:
        rt = _runtime_ref[0]
        if rt:
            rt.cancel()

    wrapper = AgentWrapper(
        port=port,
        host=host,
        harness="codex",
        model=model,
        agent_id=agent_id,
        on_cancel=on_cancel,
    )

    runtime_home: Path | None = None
    bridge_dir: Path | None = None
    bridge: BuildChatBridgeServer | None = None
    client: CodexAppServerClient | None = None
    runtime: CodexHarnessRuntime | None = None
    receive_task: asyncio.Task | None = None
    run_task: asyncio.Task | None = None

    try:
        config = await wrapper.connect()
        receive_task = asyncio.create_task(wrapper.run())

        bridge_token = uuid.uuid4().hex
        bridge_dir = Path(tempfile.mkdtemp(prefix="build-chat-bridge-"))
        bridge = BuildChatBridgeServer(
            chat_mcp=wrapper.chat_mcp,
            socket_path=str(bridge_dir / "bridge.sock"),
            token=bridge_token,
        )
        await bridge.start()

        runtime_home = create_isolated_codex_home(
            bridge_socket=bridge.socket_path,
            bridge_token=bridge_token,
            trusted_project=workdir,
            channel_id=wrapper.channel_id,
        )

        env = os.environ.copy()
        env["HOME"] = str(runtime_home)
        env["BUILD_AGENT_PORT"] = str(port)
        env["BUILD_AGENT_HOST"] = host

        client = CodexAppServerClient(env=env, cwd=workdir)
        runtime = CodexHarnessRuntime(
            wrapper=wrapper,
            client=client,
            config=config,
            model=model,
            initial_prompt=initial_prompt,
            working_directory=workdir,
            effort=effort,
            resume_thread_id=resume_session,
            auto_approve_tools=auto_approve_tools,
        )
        _runtime_ref[0] = runtime

        log.info("Starting Codex runtime...")
        run_task = asyncio.create_task(runtime.run())
        await run_task
        log.info("Codex runtime completed")

    except Exception as exc:
        log.error("Codex agent error: %s", exc, exc_info=True)
        try:
            await wrapper.chat_mcp.handle_send(
                "Something went wrong. The Codex agent encountered an internal error and may need to restart."
            )
        except Exception:
            pass
    finally:
        log.info("Codex agent shutting down, cleaning up...")
        try:
            async with asyncio.timeout(10):
                if runtime:
                    await runtime.shutdown()
                if receive_task:
                    receive_task.cancel()
                    try:
                        await receive_task
                    except asyncio.CancelledError:
                        pass
                if bridge:
                    await bridge.stop()
        except TimeoutError:
            log.warning("Cleanup timed out after 10s")
        # Only clean up temp homes (no channel_id); persistent homes are kept
        # so Codex rollout files survive restarts for thread/resume.
        if runtime_home and str(runtime_home).startswith(tempfile.gettempdir()):
            shutil.rmtree(runtime_home, ignore_errors=True)
        if bridge_dir:
            shutil.rmtree(bridge_dir, ignore_errors=True)
        await wrapper.disconnect("completed")
        log.info("Codex agent stopped")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Agent — Codex CLI + BAP wrapper")
    parser.add_argument("prompt", nargs="*", help="Initial prompt")
    parser.add_argument("--port", type=int, default=int(os.environ.get("BUILD_AGENT_PORT", "9783")))
    parser.add_argument("--host", default=os.environ.get("BUILD_AGENT_HOST", "127.0.0.1"))
    parser.add_argument("--model", default=os.environ.get("BUILD_AGENT_MODEL", "gpt-5.4"))
    parser.add_argument("--agent-id", default=os.environ.get("BUILD_AGENT_ID"))
    parser.add_argument("--working-directory", default=os.environ.get("BUILD_WORKING_DIR"))
    parser.add_argument("--effort", default=os.environ.get("BUILD_AGENT_EFFORT", ""))
    parser.add_argument("--resume-session", default=None,
                        help="Thread ID to resume from a previous session")
    parser.add_argument("--auto-approve-tools", action="store_true",
                        help="Skip user approval for tool calls (channel setting)")
    args = parser.parse_args()

    initial_prompt = " ".join(args.prompt) if args.prompt else None
    asyncio.run(run_agent(
        port=args.port,
        host=args.host,
        model=args.model,
        initial_prompt=initial_prompt,
        agent_id=args.agent_id,
        working_directory=args.working_directory,
        effort=args.effort or "",
        resume_session=args.resume_session,
        auto_approve_tools=args.auto_approve_tools,
    ))


if __name__ == "__main__":
    main()
