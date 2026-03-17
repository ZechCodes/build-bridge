"""Tests for build_client.chat_mcp — Chat MCP tool handlers and message queue."""

from __future__ import annotations

import asyncio

import pytest

from build_client.chat_mcp import ChatMCP


class TestMessageQueue:
    async def test_queue_message(self):
        chat = ChatMCP()
        await chat.queue_message("Hello")
        assert chat.unread_count == 1
        assert chat.has_unread

    def test_empty_queue(self):
        chat = ChatMCP()
        assert chat.unread_count == 0
        assert not chat.has_unread

    async def test_queue_multiple(self):
        chat = ChatMCP()
        await chat.queue_message("First")
        await chat.queue_message("Second")
        await chat.queue_message("Third")
        assert chat.unread_count == 3

    async def test_custom_role_and_timestamp(self):
        chat = ChatMCP()
        await chat.queue_message("Hello", role="system", timestamp="2026-01-01T00:00:00Z")
        assert chat.unread_count == 1


class TestReadUnread:
    async def test_read_returns_messages(self):
        chat = ChatMCP()
        await chat.queue_message("Hello agent")
        await chat.queue_message("Fix the bug")

        result = await chat.handle_read_unread()
        assert len(result["messages"]) == 2
        assert result["messages"][0]["content"] == "Hello agent"
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][1]["content"] == "Fix the bug"

    async def test_read_clears_queue(self):
        chat = ChatMCP()
        await chat.queue_message("Hello")

        await chat.handle_read_unread()
        assert chat.unread_count == 0
        assert not chat.has_unread

    async def test_subsequent_read_empty(self):
        chat = ChatMCP()
        await chat.queue_message("Hello")

        await chat.handle_read_unread()
        result = await chat.handle_read_unread()
        assert result["messages"] == []

    async def test_read_empty_queue(self):
        chat = ChatMCP()
        result = await chat.handle_read_unread()
        assert result["messages"] == []

    async def test_new_messages_after_read(self):
        chat = ChatMCP()
        await chat.queue_message("First")
        await chat.handle_read_unread()

        await chat.queue_message("Second")
        result = await chat.handle_read_unread()
        assert len(result["messages"]) == 1
        assert result["messages"][0]["content"] == "Second"

    async def test_messages_have_timestamp(self):
        chat = ChatMCP()
        await chat.queue_message("Hello", timestamp="2026-03-15T12:00:00+00:00")

        result = await chat.handle_read_unread()
        assert result["messages"][0]["timestamp"] == "2026-03-15T12:00:00+00:00"

    async def test_messages_get_auto_timestamp(self):
        chat = ChatMCP()
        await chat.queue_message("Hello")

        result = await chat.handle_read_unread()
        # Should have a timestamp even without explicit one.
        assert "T" in result["messages"][0]["timestamp"]


class TestSend:
    async def test_send_calls_callback(self):
        sent_messages = []

        async def on_send(message: str):
            sent_messages.append(message)

        chat = ChatMCP(on_send=on_send)
        result = await chat.handle_send("I'll fix the bug.")

        assert result == {"status": "sent"}
        assert sent_messages == ["I'll fix the bug."]

    async def test_send_without_callback(self):
        chat = ChatMCP()
        # Should not raise.
        result = await chat.handle_send("Hello")
        assert result == {"status": "sent"}

    async def test_multiple_sends(self):
        sent_messages = []

        async def on_send(message: str):
            sent_messages.append(message)

        chat = ChatMCP(on_send=on_send)
        await chat.handle_send("First response")
        await chat.handle_send("Second response")

        assert len(sent_messages) == 2


class TestNotificationInjection:
    def test_no_unread(self):
        chat = ChatMCP()
        assert chat.build_unread_notification() is None

    async def test_one_unread(self):
        chat = ChatMCP()
        await chat.queue_message("Hello")
        notification = chat.build_unread_notification()
        assert notification is not None
        assert "1 unread message" in notification
        assert "read_unread" in notification

    async def test_multiple_unread(self):
        chat = ChatMCP()
        await chat.queue_message("Hello")
        await chat.queue_message("Also this")
        notification = chat.build_unread_notification()
        assert "2 unread messages" in notification
        assert "read_unread" in notification


class TestDrainUnreadNotification:
    async def test_no_unread(self):
        chat = ChatMCP()
        assert await chat.drain_unread_notification() is None

    async def test_one_unread(self):
        chat = ChatMCP()
        await chat.queue_message("Hello")
        notification = await chat.drain_unread_notification()
        assert notification is not None
        assert "1 unread message" in notification

    async def test_multiple_unread(self):
        chat = ChatMCP()
        await chat.queue_message("Hello")
        await chat.queue_message("Also this")
        notification = await chat.drain_unread_notification()
        assert "2 unread messages" in notification


class TestWaitForUnread:
    async def test_returns_immediately_if_has_unread(self):
        chat = ChatMCP()
        await chat.queue_message("Hello")
        result = await chat.wait_for_unread(timeout=0.1)
        assert result is True

    async def test_timeout_when_no_messages(self):
        chat = ChatMCP()
        result = await chat.wait_for_unread(timeout=0.05)
        assert result is False

    async def test_wakes_when_message_arrives(self):
        chat = ChatMCP()

        async def queue_after_delay():
            await asyncio.sleep(0.05)
            await chat.queue_message("Delayed message")

        task = asyncio.create_task(queue_after_delay())
        result = await chat.wait_for_unread(timeout=1.0)
        assert result is True
        await task


class TestStdioServer:
    def test_create_stdio_server(self):
        """Test that create_stdio_server returns a FastMCP instance."""
        chat = ChatMCP()
        server = chat.create_stdio_server()
        assert server is not None
        assert server.name == "build-chat"
