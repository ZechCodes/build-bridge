# Build Agent Protocol Specification

**Version:** 1.0-draft
**Date:** 2026-03-15
**Status:** Draft

-----

## 1. Overview

The Build Agent Protocol (BAP) defines the communication layer between the Build device client and individual agent processes. Each agent represents a single conversation, managed by the device client as a **channel**. The device client is the central hub: it owns the database, manages all agent lifecycles, and relays messages between agents and browser clients via the Build relay server.

Communication uses WebSocket connections. Each agent process maintains exactly one WebSocket connection to the device client. The connection **is** the channel — there is no multiplexing. One connection, one agent, one conversation.

### 1.1 Design Principles

- **Connection-scoped identity.** An agent’s identity and conversation context are established at connection time. Subsequent messages carry no routing or conversation identifiers — the socket itself is the binding.
- **Correlation over conversation.** Request/response pairs are linked by message IDs, not by session or conversation references.
- **Chat/activity separation.** The user-facing conversation (chat) is distinct from the agent’s internal reasoning and tool use (activity). Chat is mediated through MCP tools; activity is the raw harness output stream. The browser receives both but presents them differently.
- **Namespace extensibility.** Message types are dot-namespaced (`agent.*`, `chat.*`, `activity.*`, `tool.*`). New namespaces and message types can be introduced without breaking existing agents.
- **Capability negotiation.** Agents declare what they support at connection time. The device client never sends message types the agent hasn’t claimed.
- **Full fidelity.** All message content is sent in its entirety over the wire. No truncation, summarization, or transformation. The device client stores and forwards complete payloads to browser clients.
- **Agent-owned execution.** Agents are existing coding harnesses (Claude Code, Codex, Open Code, etc.) that manage their own tool execution loop internally. The device client observes and records tool activity — it does not execute tools on behalf of agents.

### 1.2 Architecture Context

```
┌─────────────────────────────────┐
│          Agent Process          │
│                                 │
│  ┌───────────┐  ┌───────────┐  │
│  │  Harness  │  │  Chat MCP │  │       ┌──────────────────┐
│  │ (Claude   │◄►│  Server   │  │       │                  │
│  │  Code,    │  └─────┬─────┘  │       │                  │         ┌──────────┐
│  │  Codex,   │        │        │  WS   │   Device Client  │  WS     │ Browser  │
│  │  etc.)    │    ┌───┴───┐   ─┼──────►│                  │◄───────►│ Client   │
│  │           │◄──►│Wrapper│    │       │   - Channel mgr  │  (via   │          │
│  └───────────┘    └───────┘   ─┼──────►│   - Database     │  relay) │          │
│                                │       │   - Relay conn   │         └──────────┘
└─────────────────────────────────┘       └──────────────────┘
```

Each agent process contains three components:

- **Harness**: The existing coding agent (Claude Code, Codex, Open Code, etc.) that manages its own model calls, tool execution, and reasoning loop.
- **Chat MCP Server**: A local MCP server providing `send` and `read_unread` tools. The harness uses these tools to communicate with the user, treating the interaction as a chat conversation rather than a raw model prompt/response.
- **Wrapper**: The Build integration layer. It manages the WebSocket connection to the device client, serves the Chat MCP, intercepts harness output for the activity stream, and injects unread message notifications into the harness loop at appropriate points.

The device client:

- Spawns and manages agent processes.
- Maintains one WebSocket connection per agent (per channel).
- Owns the SQLite database storing all conversation history, tool use records, activity, and channel metadata.
- Handles all communication with browser clients through the relay server.
- Receives chat responses and tool activity from agents, stores them, and relays them to the browser.
- Does not execute tools — agents are self-contained harnesses that manage their own tool execution loop.

### 1.3 The Chat Abstraction

Build separates the **user-facing conversation** from the **agent’s internal reasoning**. This separation is fundamental to the protocol.

**Without the abstraction**, a user message goes directly into the model’s context and the model’s streamed output goes directly to the user. The user sees raw reasoning, tool calls, retries, and dead ends.

**With the abstraction**, the harness’s internal chat loop is its own workspace. The user converses with the agent through MCP-mediated messaging:

1. The user sends a message via the browser UI.
1. The device client delivers it to the agent wrapper as a `chat.message`.
1. The wrapper queues the message and injects an unread notification into the harness loop at an appropriate breakpoint (e.g., between model turns, after a tool result).
1. The harness calls `read_unread` (Chat MCP tool) to receive the user’s message.
1. The harness does its work — reading files, editing code, running commands, reasoning. All of this streams to the device client as **activity** events.
1. When the harness has something to say to the user, it calls `send` (Chat MCP tool).
1. The wrapper captures the `send` call and emits a `chat.response` protocol message.
1. The device client stores it and relays it to the browser as the agent’s reply.

The browser UI presents two views:

- **Chat view**: The clean conversation thread — user messages and agent replies only.
- **Activity view**: The full stream of what the agent is doing — text output, tool invocations, tool results, thinking.

-----

## 2. Transport

### 2.1 WebSocket Connection

Agents connect to the device client at a well-known local address:

```
ws://localhost:{port}/agent
```

The port is provided to the agent process as a startup argument or environment variable. TLS is not required for local connections.

### 2.2 Message Framing

All messages are UTF-8 encoded JSON sent as WebSocket text frames. Binary frames are reserved for future use and MUST be ignored by implementations that do not recognize them.

### 2.3 Health Checking

Connection health is monitored using native WebSocket ping/pong frames. The device client sends pings at a regular interval (recommended: 30 seconds). An agent that fails to respond to 3 consecutive pings is considered dead and its channel is marked accordingly.

Implementations MUST NOT define application-level heartbeat messages. Use the WebSocket protocol’s built-in mechanism.

### 2.4 Reconnection

When an agent detects a connection drop, it SHOULD attempt to reconnect using exponential backoff:

- Initial delay: 500ms
- Maximum delay: 30s
- Backoff factor: 2x
- Jitter: ±25%

On reconnection, the agent sends `agent.hello` with `"reconnect": true` to signal that it is resuming a previous session (see §4.1).

-----

## 3. Message Envelope

Every message on the wire uses the following envelope:

```json
{
  "v": 1,
  "id": "msg_01HX7V3JKBM9QRS2FY8T0N5P",
  "ref": null,
  "type": "chat.message",
  "payload": {}
}
```

|Field    |Type       |Required|Description                                                                    |
|---------|-----------|--------|-------------------------------------------------------------------------------|
|`v`      |integer    |Yes     |Protocol version. Currently `1`.                                               |
|`id`     |string     |Yes     |Unique message identifier. MUST be a ULID or equivalent sortable unique ID.    |
|`ref`    |string/null|Yes     |The `id` of the message this is responding to. `null` for unsolicited messages.|
|`type`   |string     |Yes     |Dot-namespaced message type. See §4–§7 for defined types.                      |
|`payload`|object     |Yes     |Type-specific data. May be an empty object `{}` but MUST be present.           |

### 3.1 Message Identity

Message IDs (`id`) MUST be unique within a connection’s lifetime. ULIDs are recommended because they are sortable by generation time and have negligible collision probability without coordination.

### 3.2 Correlation

The `ref` field links a response to the message that prompted it. This is the sole mechanism for request/response pairing. Multiple messages MAY reference the same `id` (e.g., a stream of `activity.delta` messages all referencing a single `chat.message`).

Messages that are not responses to a specific prior message set `ref` to `null`.

### 3.3 Version Handling

If an agent or device client receives a message with a `v` higher than it supports, it SHOULD send an `agent.error` and MAY close the connection. Messages with an unrecognized `type` but a supported `v` SHOULD be silently ignored (not cause disconnection).

-----

## 4. Agent Namespace (`agent.*`)

The agent namespace governs connection lifecycle: handshake, configuration, shutdown, and errors.

### 4.1 `agent.hello`

**Direction:** Agent → Device Client
**When:** Immediately after WebSocket connection is established. This MUST be the first message sent by the agent.

```json
{
  "type": "agent.hello",
  "payload": {
    "agent_id": "agt_x7f2",
    "harness": "claude-code",
    "capabilities": ["chat", "activity", "tools"],
    "model": "claude-sonnet-4-20250514",
    "reconnect": false
  }
}
```

|Field         |Type    |Required|Description                                                                                  |
|--------------|--------|--------|---------------------------------------------------------------------------------------------|
|`agent_id`    |string  |Yes     |Stable identifier for this agent. Used for reconnection matching and database association.   |
|`harness`     |string  |Yes     |The coding harness being wrapped (e.g., `"claude-code"`, `"codex"`, `"open-code"`).          |
|`capabilities`|string[]|Yes     |List of capability identifiers the agent supports. See §4.1.1.                               |
|`model`       |string  |Yes     |The model identifier the agent is using.                                                     |
|`reconnect`   |boolean |Yes     |`true` if this is a reconnection to a previously established channel. `false` for new agents.|

#### 4.1.1 Defined Capabilities

|Capability|Description                         |Required Namespaces|
|----------|------------------------------------|-------------------|
|`chat`    |MCP-mediated user conversation.     |`chat.*`           |
|`activity`|Raw harness output streaming.       |`activity.*`       |
|`tools`   |Agent reports tool execution events.|`tool.*`           |

New capabilities MAY be added in future versions. An agent SHOULD ignore capability identifiers it does not recognize. The device client MUST NOT send messages from a namespace whose corresponding capability was not declared by the agent.

### 4.2 `agent.configured`

**Direction:** Device Client → Agent
**When:** In response to `agent.hello`. The `ref` field references the `agent.hello` message.

```json
{
  "type": "agent.configured",
  "payload": {
    "channel_id": "ch_9k3m",
    "system_prompt": "You are working on the Build project...",
    "chat_instructions": "You have access to send and read_unread MCP tools for communicating with the user. When you have unread messages, read them. When you have something to report or need input, use send. Keep your chat messages concise and focused on what the user needs to know.",
    "history": {
      "chat": [
        {
          "role": "user",
          "content": "Refactor the auth module."
        },
        {
          "role": "assistant",
          "content": "Done — I've extracted the auth logic into an AuthService class with dependency injection. The tests all pass."
        }
      ],
      "activity": [
        {
          "type": "text",
          "text": "I'll start by reading the current auth module..."
        },
        {
          "type": "tool_use",
          "id": "tu_abc",
          "name": "read_file",
          "input": { "path": "/src/auth.py" }
        },
        {
          "type": "tool_result",
          "tool_use_id": "tu_abc",
          "content": "class Auth:\n    ...",
          "is_error": false
        }
      ]
    }
  }
}
```

|Field              |Type  |Required|Description                                                                                                                                                                                     |
|-------------------|------|--------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|`channel_id`       |string|Yes     |The device client’s internal channel identifier for this agent. Informational — the agent does not use this for routing.                                                                        |
|`system_prompt`    |string|Yes     |Additional system prompt context from the user, injected via the Build UI. The wrapper may prepend or append this to the harness’s own system prompt. May be empty string if none is configured.|
|`chat_instructions`|string|Yes     |Instructions for the agent on how to use the Chat MCP tools. The wrapper injects this into the harness’s system prompt.                                                                         |
|`history`          |object|Yes     |Prior conversation data for this channel. Contains `chat` and `activity` arrays. See §4.2.1.                                                                                                    |

#### 4.2.1 History Format

The `history` object contains two arrays:

**`history.chat`** — The clean user↔agent conversation. Each entry has `role` (`"user"` or `"assistant"`) and `content` (string). This is what the wrapper uses to provide conversational context to the harness.

```json
{
  "role": "user",
  "content": "Refactor the auth module."
}
```

```json
{
  "role": "assistant",
  "content": "Done — I've extracted the auth logic into an AuthService class."
}
```

**`history.activity`** — The raw harness activity log. A flat array of activity entries in chronological order. Each entry has a `type` field:

- `"text"` — Harness text output, with a `text` field.
- `"thinking"` — Harness thinking output, with a `text` field.
- `"tool_use"` — Tool invocation, with `id`, `name`, and `input` fields.
- `"tool_result"` — Tool execution result, with `tool_use_id`, `content`, and `is_error` fields.

The wrapper uses the chat history to reconstruct conversational context. The activity history is available for display in the browser UI but is not typically injected back into the harness’s context (the harness maintains its own internal conversation state).

### 4.3 `agent.shutdown`

**Direction:** Device Client → Agent
**When:** The device client is requesting the agent to shut down gracefully.

```json
{
  "type": "agent.shutdown",
  "payload": {
    "reason": "user_closed"
  }
}
```

|Field   |Type  |Required|Description                         |
|--------|------|--------|------------------------------------|
|`reason`|string|Yes     |Why the shutdown is being requested.|

Defined reasons:

|Reason           |Description                                         |
|-----------------|----------------------------------------------------|
|`user_closed`    |The user explicitly closed this conversation.       |
|`client_shutdown`|The device client itself is shutting down.          |
|`timeout`        |The agent has been idle beyond the configured limit.|
|`error`          |The channel is being terminated due to an error.    |

On receiving `agent.shutdown`, the agent SHOULD:

1. Cancel any in-progress work.
1. Send `agent.goodbye` (see §4.4).
1. Close the WebSocket connection.

If the agent does not close the connection within 5 seconds of receiving `agent.shutdown`, the device client MAY forcibly terminate the connection.

### 4.4 `agent.goodbye`

**Direction:** Agent → Device Client
**When:** The agent is confirming shutdown, or the agent is voluntarily disconnecting.

```json
{
  "type": "agent.goodbye",
  "payload": {
    "reason": "shutdown_ack"
  }
}
```

|Field   |Type  |Required|Description                    |
|--------|------|--------|-------------------------------|
|`reason`|string|Yes     |Why the agent is disconnecting.|

Defined reasons:

|Reason        |Description                                              |
|--------------|---------------------------------------------------------|
|`shutdown_ack`|Acknowledging a device client `agent.shutdown`.          |
|`completed`   |The agent has finished its work and is exiting.          |
|`error`       |The agent encountered a fatal error and is disconnecting.|

### 4.5 `agent.error`

**Direction:** Bidirectional
**When:** A non-fatal error has occurred that the other side should be aware of.

```json
{
  "type": "agent.error",
  "payload": {
    "code": "invalid_message",
    "message": "Received message with unrecognized type: foo.bar",
    "fatal": false
  }
}
```

|Field    |Type   |Required|Description                                                        |
|---------|-------|--------|-------------------------------------------------------------------|
|`code`   |string |Yes     |Machine-readable error code.                                       |
|`message`|string |Yes     |Human-readable error description.                                  |
|`fatal`  |boolean|Yes     |If `true`, the sender will close the connection after sending this.|

Defined error codes:

|Code                 |Description                                                            |
|---------------------|-----------------------------------------------------------------------|
|`invalid_message`    |Message could not be parsed or is malformed.                           |
|`unknown_type`       |The message type is not recognized.                                    |
|`protocol_violation` |A protocol constraint was violated (e.g., no `agent.hello` sent first).|
|`capability_mismatch`|A message was received for an undeclared capability.                   |
|`version_mismatch`   |The protocol version is not supported.                                 |
|`internal_error`     |An unexpected internal error occurred.                                 |
|`reconnect_failed`   |Reconnection was attempted but the channel could not be restored.      |

-----

## 5. Chat Namespace (`chat.*`)

The chat namespace handles the **user-facing conversation** — the clean message thread the user sees in the browser. Chat messages are mediated through the Chat MCP tools: user messages are delivered to the agent and read via MCP `read_unread`, agent replies are sent via MCP `send`.

All agents declaring the `chat` capability MUST support these message types.

### 5.1 `chat.message`

**Direction:** Device Client → Agent
**When:** The user has sent a message through the browser client.

```json
{
  "type": "chat.message",
  "payload": {
    "role": "user",
    "content": "Refactor the auth module to use dependency injection."
  }
}
```

|Field    |Type        |Required|Description                                                                                                           |
|---------|------------|--------|----------------------------------------------------------------------------------------------------------------------|
|`role`   |string      |Yes     |The message role. Currently always `"user"`.                                                                          |
|`content`|string/array|Yes     |The message content. A string for text-only messages, or an array of content blocks for multimodal input (see §5.1.1).|

On receiving `chat.message`, the wrapper:

1. Queues the message internally for the Chat MCP `read_unread` tool.
1. Injects an unread message notification into the harness loop at the next appropriate breakpoint (e.g., between model turns, after a tool result completes, or as a new prompt if the harness is idle).

The notification injection is wrapper-managed. The protocol delivers the message; the wrapper decides when and how to surface it to the harness. System prompt instructions (from `chat_instructions` in `agent.configured`) tell the harness to use the MCP tools to read and respond.

#### 5.1.1 Multimodal Content Blocks

When `content` is an array, each element is a content block:

**Text block:**

```json
{
  "type": "text",
  "text": "What does this image show?"
}
```

**Image block:**

```json
{
  "type": "image",
  "source": {
    "type": "base64",
    "media_type": "image/png",
    "data": "iVBORw0KGgo..."
  }
}
```

**Document block:**

```json
{
  "type": "document",
  "source": {
    "type": "base64",
    "media_type": "application/pdf",
    "data": "JVBERi0xLjQ..."
  }
}
```

The content block format mirrors the Anthropic Messages API. Additional block types MAY be added in future versions.

### 5.2 `chat.response`

**Direction:** Agent → Device Client
**When:** The agent has called the Chat MCP `send` tool. The wrapper captures the `send` call and emits this protocol message.

```json
{
  "type": "chat.response",
  "ref": "msg_01CHATMESSAGE",
  "payload": {
    "content": "Done — I've extracted the auth logic into an AuthService class with dependency injection. All 47 tests pass. Want me to update the integration tests too?"
  }
}
```

|Field    |Type  |Required|Description                                                                                                   |
|---------|------|--------|--------------------------------------------------------------------------------------------------------------|
|`content`|string|Yes     |The agent’s message to the user. This is the text content from the Chat MCP `send` call, sent in its entirety.|

The `ref` field references the most recent `chat.message` this response relates to, or `null` if the agent is sending an unsolicited message (e.g., proactively reporting progress without a user prompt).

A single `chat.message` may produce zero, one, or multiple `chat.response` messages. The agent may send several updates as it works, or may complete silently with no chat response if the task requires no user communication.

### 5.3 `chat.cancel`

**Direction:** Device Client → Agent
**When:** The user has requested cancellation of the agent’s current work.

```json
{
  "type": "chat.cancel",
  "payload": {}
}
```

On receiving `chat.cancel`, the agent SHOULD:

1. Abort any in-progress harness work as quickly as possible.
1. Send a final `activity.end` with `"reason": "cancelled"` (see §6.3).

The agent MAY send a `chat.response` acknowledging the cancellation before stopping, or may stop silently.

-----

## 6. Activity Namespace (`activity.*`)

The activity namespace carries the **raw harness output stream** — everything the agent’s internal reasoning loop produces. This includes streamed text, thinking, and is the full picture of what the agent is doing. The browser UI uses this to render an activity view alongside the clean chat.

Activity messages are observational. They flow from agent to device client only. The device client stores and relays them to the browser but takes no action on them.

All agents declaring the `activity` capability MUST support these message types.

### 6.1 `activity.delta`

**Direction:** Agent → Device Client
**When:** The harness is streaming output.

```json
{
  "type": "activity.delta",
  "payload": {
    "delta": {
      "type": "text",
      "text": "I'll start by reviewing the current auth module structure. "
    },
    "index": 0
  }
}
```

|Field  |Type   |Required|Description                                                                                       |
|-------|-------|--------|--------------------------------------------------------------------------------------------------|
|`delta`|object |Yes     |The content delta. See §6.1.1 for delta types.                                                    |
|`index`|integer|Yes     |Monotonically increasing sequence number starting at 0 for each harness turn. Used to detect gaps.|

#### 6.1.1 Delta Types

**Text delta:**

```json
{
  "type": "text",
  "text": "partial text content..."
}
```

**Thinking delta:**

```json
{
  "type": "thinking",
  "text": "Let me consider the architecture..."
}
```

Additional delta types MAY be introduced in future versions. Consumers that encounter an unrecognized delta type SHOULD treat it as opaque text content.

### 6.2 `activity.ping`

**Direction:** Agent → Device Client
**When:** The agent is actively working but has no new streamed output. This keeps the browser UI informed that the agent hasn’t stalled.

```json
{
  "type": "activity.ping",
  "payload": {}
}
```

The wrapper SHOULD send `activity.ping` at a regular interval (recommended: every 5 seconds) when the harness is processing but not producing output (e.g., during a long tool execution, waiting for a model response, etc.). This enables the browser to display a “working…” indicator.

### 6.3 `activity.end`

**Direction:** Agent → Device Client
**When:** The agent has completed a unit of work. This signals the end of a harness turn or the agent becoming idle.

```json
{
  "type": "activity.end",
  "payload": {
    "reason": "complete",
    "usage": {
      "input_tokens": 1200,
      "output_tokens": 340,
      "cache_creation_input_tokens": 0,
      "cache_read_input_tokens": 800
    }
  }
}
```

|Field   |Type  |Required|Description                                                |
|--------|------|--------|-----------------------------------------------------------|
|`reason`|string|Yes     |Why the activity ended.                                    |
|`usage` |object|No      |Token usage for this unit of work. All fields are integers.|

Defined reasons:

|Reason     |Description                                                                                   |
|-----------|----------------------------------------------------------------------------------------------|
|`complete` |The agent finished its current work and is idle.                                              |
|`cancelled`|The work was cancelled by a `chat.cancel` message.                                            |
|`error`    |The work was terminated due to an error.                                                      |
|`limit`    |A limit was reached (token limit, rate limit, etc.).                                          |
|`waiting`  |The agent has paused and is waiting for user input (sent a `chat.response` asking a question).|

-----

## 7. Tool Namespace (`tool.*`)

The tool namespace handles tool use reporting. Build wraps existing coding agent harnesses that manage their own tool execution loop internally. When the agent uses a tool, it reports the invocation and result to the device client as events.

The Chat MCP tools (`send`, `read_unread`) are **excluded** from the tool namespace. The wrapper handles them internally and translates them into `chat.*` protocol messages. Only the agent’s work tools (file operations, shell commands, search, etc.) appear as `tool.*` events.

Agents MUST declare the `tools` capability in `agent.hello` to use this namespace.

### 7.1 Tool Use Flow

The tool execution loop runs entirely inside the agent harness. The protocol carries event notifications to the device client:

```
Agent (internal)                  Protocol Messages              Device Client
     │                                                                │
     │  model returns tool_use                                        │
     │  ─────────────────────►  tool.use  ──────────────────────────►│  (stores, relays to browser)
     │                                                                │
     │  agent executes tool                                           │
     │  locally (browser UI                                           │
     │  shows "running...")                                           │
     │                                                                │
     │  execution completes   ►  tool.result  ──────────────────────►│  (stores, relays to browser)
     │                                                                │
     │  agent feeds result                                            │
     │  back to model                                                 │
     │                                                                │
     │  model returns text                                            │
     │  ─────────────────────►  activity.delta  ────────────────────►│
     │                                                                │
     │  (loop may repeat)                                             │
     │                                                                │
```

A single agent turn may involve zero, one, or many tool use cycles. The device client never sends tool-related messages to the agent — all `tool.*` messages flow from agent to device client.

### 7.2 `tool.use`

**Direction:** Agent → Device Client
**When:** The model has produced a tool use block and the agent is about to execute the tool. Sent immediately before execution begins.

```json
{
  "type": "tool.use",
  "payload": {
    "tool_use_id": "tu_01HX7V4ABCD",
    "name": "edit_file",
    "input": {
      "path": "/src/auth.py",
      "content": "import inject\n\nclass AuthService:\n    ..."
    }
  }
}
```

|Field        |Type  |Required|Description                                                                                                                                                 |
|-------------|------|--------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
|`tool_use_id`|string|Yes     |Unique identifier for this tool invocation. Typically the `id` from the model’s tool_use content block. Used to correlate with the subsequent `tool.result`.|
|`name`       |string|Yes     |The name of the tool being invoked (e.g., `edit_file`, `bash`, `read_file`).                                                                                |
|`input`      |object|Yes     |The full tool input parameters as produced by the model. Sent in its entirety — no transformation or truncation.                                            |

The agent MUST send `tool.use` before executing the tool. This ensures the browser UI can display the invocation immediately and provides a real-time view of agent activity. The harness wrapper is responsible for hooking into the harness’s tool execution lifecycle to emit `tool.use` at the point of invocation, prior to execution.

### 7.3 `tool.result`

**Direction:** Agent → Device Client
**When:** The agent has finished executing a tool and has the result.

```json
{
  "type": "tool.result",
  "ref": "msg_01TOOLUSE",
  "payload": {
    "tool_use_id": "tu_01HX7V4ABCD",
    "content": "File /src/auth.py updated successfully (234 lines).",
    "is_error": false
  }
}
```

|Field        |Type        |Required|Description                                                                                                                                                     |
|-------------|------------|--------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
|`tool_use_id`|string      |Yes     |The `tool_use_id` from the corresponding `tool.use` message.                                                                                                    |
|`content`    |string/array|Yes     |The tool output. A string for text results, or an array of content blocks for rich results (see §7.3.1). Sent in its entirety — no transformation or truncation.|
|`is_error`   |boolean     |Yes     |`true` if the tool execution failed.                                                                                                                            |

The `ref` field references the corresponding `tool.use` message.

#### 7.3.1 Rich Tool Results

When `content` is an array, it may contain:

**Text block:**

```json
{
  "type": "text",
  "text": "File contents:\n..."
}
```

**Image block:**

```json
{
  "type": "image",
  "source": {
    "type": "base64",
    "media_type": "image/png",
    "data": "..."
  }
}
```

This allows tools like screenshot capture or diagram generation to return visual content to the browser UI.

### 7.4 Parallel Tool Use

A model may produce multiple tool use blocks in a single response. The agent sends each `tool.use` before beginning execution, then reports results as they complete. The device client matches them by `tool_use_id`:

```
Agent → Device Client:  tool.use     (tool_use_id: "tu_1", name: "read_file")
Agent → Device Client:  tool.use     (tool_use_id: "tu_2", name: "read_file")
  (agent executes both tools)
Agent → Device Client:  tool.result  (tool_use_id: "tu_2", content: "...")  ← order not guaranteed
Agent → Device Client:  tool.result  (tool_use_id: "tu_1", content: "...")
```

All `tool.use` messages for a parallel batch MUST be sent before any `tool.result` messages for that batch. Results MAY arrive in any order.

-----

## 8. Chat MCP Server

The Chat MCP server is a local MCP server running inside the agent process, managed by the wrapper. It provides the tools the harness uses to communicate with the user. The MCP server is connected to the harness via stdio transport.

### 8.1 Tool: `read_unread`

Returns any unread messages from the user.

**Input schema:**

```json
{
  "type": "object",
  "properties": {},
  "required": []
}
```

**Output:** An array of unread messages, each with `role`, `content`, and `timestamp`:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Refactor the auth module to use dependency injection.",
      "timestamp": "2026-03-15T14:30:00Z"
    }
  ]
}
```

If there are no unread messages, returns `{ "messages": [] }`.

Once messages are returned by `read_unread`, they are marked as read. Subsequent calls return only new messages.

### 8.2 Tool: `send`

Sends a message to the user.

**Input schema:**

```json
{
  "type": "object",
  "properties": {
    "message": {
      "type": "string",
      "description": "The message to send to the user."
    }
  },
  "required": ["message"]
}
```

**Output:** Confirmation of delivery:

```json
{
  "status": "sent"
}
```

When the wrapper receives a `send` call:

1. It returns the confirmation to the harness via MCP.
1. It emits a `chat.response` protocol message to the device client with the message content.

### 8.3 Unread Notification Injection

The wrapper is responsible for injecting unread message notifications into the harness loop. The injection mechanism is harness-specific, but the general pattern is:

- **If the harness is idle** (waiting for input): Inject the notification as a new user turn prompt, e.g., `"You have 1 unread message from the user. Use the read_unread tool to read it."`
- **If the harness is mid-turn** (actively generating or executing tools): Queue the notification and inject it at the next natural breakpoint — after the current tool result, or between model turns.
- **If the harness supports interrupt signals**: The wrapper may signal the harness to check for unread messages at the next opportunity.

The exact wording and timing of injections is configured via `chat_instructions` in `agent.configured` and MAY be tuned per harness type.

### 8.4 Chat MCP and the Tool Namespace

Chat MCP tool calls (`send`, `read_unread`) are **not** reported in the `tool.*` namespace. They are an internal mechanism of the chat abstraction, not work the agent is performing. The wrapper filters them out of the tool event stream:

- `send` → translated into `chat.response`
- `read_unread` → handled internally by the wrapper, no protocol message emitted

This keeps the tool namespace clean for the browser’s activity view. If debugging visibility into Chat MCP calls is needed, a future `debug.*` namespace could expose them.

-----

## 9. Database Model

The device client maintains the following data for each channel, stored in its SQLite database.

### 9.1 Channels Table

|Column         |Type   |Description                                         |
|---------------|-------|----------------------------------------------------|
|`id`           |TEXT PK|Channel identifier (e.g., `ch_9k3m`).               |
|`agent_id`     |TEXT   |The agent identifier for this channel.              |
|`harness`      |TEXT   |The harness type (e.g., `claude-code`, `codex`).    |
|`model`        |TEXT   |The model the agent is using.                       |
|`system_prompt`|TEXT   |User-provided system prompt for this channel.       |
|`status`       |TEXT   |Channel status: `active`, `idle`, `closed`, `error`.|
|`created_at`   |TEXT   |ISO 8601 timestamp.                                 |
|`updated_at`   |TEXT   |ISO 8601 timestamp of last activity.                |

### 9.2 Chat Messages Table

Stores the clean user↔agent conversation thread.

|Column      |Type   |Description                                                      |
|------------|-------|-----------------------------------------------------------------|
|`id`        |TEXT PK|Message identifier (the protocol `id`).                          |
|`channel_id`|TEXT FK|The channel this message belongs to.                             |
|`role`      |TEXT   |`user` or `assistant`.                                           |
|`content`   |TEXT   |The message content (string or JSON-encoded content block array).|
|`created_at`|TEXT   |ISO 8601 timestamp.                                              |

User messages are inserted when the device client receives them from the browser. Assistant messages are inserted when the device client receives a `chat.response` from the agent.

### 9.3 Activity Log Table

Stores the raw harness output stream.

|Column      |Type   |Description                                                        |
|------------|-------|-------------------------------------------------------------------|
|`id`        |TEXT PK|Auto-generated identifier.                                         |
|`channel_id`|TEXT FK|The channel this activity belongs to.                              |
|`type`      |TEXT   |Activity entry type: `text`, `thinking`, `tool_use`, `tool_result`.|
|`data`      |TEXT   |JSON-encoded activity data. Full content, no truncation.           |
|`created_at`|TEXT   |ISO 8601 timestamp.                                                |

Activity entries are inserted from `activity.delta`, `tool.use`, and `tool.result` events.

### 9.4 Tool Uses Table

Stores structured tool use records for querying and display.

|Column        |Type   |Description                                                                                       |
|--------------|-------|--------------------------------------------------------------------------------------------------|
|`id`          |TEXT PK|The `tool_use_id`.                                                                                |
|`channel_id`  |TEXT FK|The channel this tool use belongs to.                                                             |
|`name`        |TEXT   |Tool name.                                                                                        |
|`input`       |TEXT   |JSON-encoded tool input. Stored in full — no truncation.                                          |
|`output`      |TEXT   |JSON-encoded tool result content. Stored in full — no truncation. NULL if result not yet received.|
|`is_error`    |BOOLEAN|Whether the tool execution resulted in an error. NULL if result not yet received.                 |
|`created_at`  |TEXT   |ISO 8601 timestamp of the `tool.use` event.                                                       |
|`completed_at`|TEXT   |ISO 8601 timestamp of the `tool.result` event. NULL if not yet received.                          |

### 9.5 History Reconstruction

To populate `agent.configured` history on reconnection:

**Chat history:** Query chat messages for the channel, ordered by `created_at`. Returns the clean conversation thread.

**Activity history:** Query activity log entries for the channel, ordered by `created_at`. Returns the full activity stream including text output, tool use, and tool results.

Both are sent in their entirety. The wrapper uses chat history for conversational context and activity history for browser display.

-----

## 10. Message Flow Examples

### 10.1 Simple Conversation

```
Agent                          Device Client              Browser
  │                                  │                       │
  │──── agent.hello ────────────────►│                       │
  │◄─── agent.configured ───────────│                       │
  │                                  │                       │
  │                                  │◄── user message ──────│
  │◄─── chat.message ───────────────│                       │
  │                                  │                       │
  │  (wrapper queues message,        │                       │
  │   injects unread notification)   │                       │
  │                                  │                       │
  │──── activity.delta (0) ────────►│──── activity ────────►│
  │  "Let me read the user message"  │                       │
  │                                  │                       │
  │  (harness calls read_unread MCP) │                       │
  │  (wrapper returns user message)  │                       │
  │                                  │                       │
  │──── activity.delta (1) ────────►│──── activity ────────►│
  │  "The user wants DI refactoring" │                       │
  │                                  │                       │
  │  (harness calls send MCP)        │                       │
  │                                  │                       │
  │──── chat.response ─────────────►│──── chat msg ────────►│
  │  "I'll refactor the auth module" │                       │
  │                                  │                       │
  │──── activity.end (complete) ───►│──── activity ────────►│
  │                                  │                       │
```

### 10.2 Conversation with Tool Use

```
Agent                          Device Client              Browser
  │                                  │                       │
  │◄─── chat.message ───────────────│  "Add logging"        │
  │                                  │                       │
  │  (wrapper injects notification)  │                       │
  │  (harness reads unread via MCP)  │                       │
  │                                  │                       │
  │──── activity.delta (0) ────────►│──── activity ────────►│
  │  "I'll read the file first."     │                       │
  │                                  │                       │
  │──── tool.use ──────────────────►│──── activity ────────►│
  │  (read_file, /src/auth.py)       │  (browser shows tool) │
  │                                  │                       │
  │  (agent executes read_file)      │                       │
  │                                  │                       │
  │──── tool.result ───────────────►│──── activity ────────►│
  │  (file contents)                 │                       │
  │                                  │                       │
  │──── activity.delta (1) ────────►│──── activity ────────►│
  │  "I see the auth module, adding" │                       │
  │                                  │                       │
  │──── tool.use ──────────────────►│──── activity ────────►│
  │  (edit_file, /src/auth.py)       │  (browser shows tool) │
  │                                  │                       │
  │  (agent executes edit_file)      │                       │
  │                                  │                       │
  │──── tool.result ───────────────►│──── activity ────────►│
  │  (success)                       │                       │
  │                                  │                       │
  │  (harness calls send MCP)        │                       │
  │                                  │                       │
  │──── chat.response ─────────────►│──── chat msg ────────►│
  │  "Added logging to auth.py."     │                       │
  │                                  │                       │
  │──── activity.end (complete) ───►│──── activity ────────►│
  │                                  │                       │
```

### 10.3 User Message During Active Work

```
Agent                          Device Client              Browser
  │                                  │                       │
  │  (agent is mid-turn, editing     │                       │
  │   files after a prior message)   │                       │
  │                                  │                       │
  │──── tool.use ──────────────────►│                       │
  │  (edit_file)                     │                       │
  │                                  │◄── user message ──────│
  │◄─── chat.message ───────────────│  "Also fix the tests" │
  │                                  │                       │
  │  (wrapper queues message,        │                       │
  │   waits for current tool to      │                       │
  │   finish before injecting)       │                       │
  │                                  │                       │
  │──── tool.result ───────────────►│                       │
  │                                  │                       │
  │  (wrapper injects: "You have     │                       │
  │   1 unread message")             │                       │
  │                                  │                       │
  │  (harness calls read_unread MCP) │                       │
  │  (harness adjusts plan to        │                       │
  │   include test fixes)            │                       │
  │                                  │                       │
  │──── chat.response ─────────────►│──── chat msg ────────►│
  │  "Got it, I'll fix tests too."   │                       │
  │                                  │                       │
```

### 10.4 Reconnection After Device Client Restart

```
Agent                          Device Client (new process)
  │                                  │
  │  (connection drops)              │
  │  (exponential backoff...)        │
  │                                  │
  │──── agent.hello ────────────────►│  (reconnect: true, agent_id: "agt_x7f2")
  │                                  │
  │  (device client looks up         │
  │   agent_id in database,          │
  │   finds existing channel)        │
  │                                  │
  │◄─── agent.configured ───────────│  (chat + activity history from DB)
  │                                  │
  │  (agent resumes, ready for       │
  │   new messages)                  │
  │                                  │
```

### 10.5 Cancellation

```
Agent                          Device Client              Browser
  │                                  │                       │
  │  (agent working on a task)       │                       │
  │──── tool.use ──────────────────►│                       │
  │                                  │                       │
  │                                  │◄── user cancels ──────│
  │◄─── chat.cancel ────────────────│                       │
  │                                  │                       │
  │  (agent aborts current work)     │                       │
  │                                  │                       │
  │──── activity.end (cancelled) ──►│──── activity ────────►│
  │                                  │                       │
```

-----

## 11. Extensibility

### 11.1 Adding New Namespaces

New functionality is added by defining new namespaces and corresponding capabilities. For example, a future direct filesystem browsing feature (independent of agent tools) would:

1. Define capability `"filesystem"`.
1. Define message types: `fs.list`, `fs.read`, `fs.write`, `fs.watch`.
1. Agents that support it include `"filesystem"` in their `capabilities` array.
1. The device client only sends `fs.*` messages to agents that declared the capability.

### 11.2 Adding New Message Types to Existing Namespaces

New message types can be added to existing namespaces without a version bump. Receivers that encounter an unrecognized type within a known namespace SHOULD silently ignore it (not error).

### 11.3 Payload Extension

Payloads may gain new fields over time. Implementations MUST ignore unrecognized fields in payloads (do not error on unexpected keys). Required fields will never be removed from a payload within the same protocol version.

### 11.4 Protocol Version Bumps

The `v` field is incremented only for breaking changes:

- Removing a required field.
- Changing the semantics of an existing message type.
- Changing the envelope structure.

Additive changes (new namespaces, new types, new optional fields) do not require a version bump.

### 11.5 Chat MCP Extension

Additional Chat MCP tools MAY be introduced in future versions. Potential candidates:

- `send_image` — Send an image to the user (e.g., a generated diagram or screenshot).
- `send_file` — Attach a file to the chat.
- `ask` — Send a message and block until the user replies (synchronous conversation).
- `typing` — Indicate the agent is composing a response (for browser UI).

New Chat MCP tools follow the same pattern: the wrapper handles them internally and translates them into appropriate protocol messages.

-----

## 12. Security Considerations

### 12.1 Local Transport

The agent-to-device-client connection is local (localhost). No authentication is applied at the WebSocket layer. Process-level isolation (the agent runs as a separate process managed by the device client) provides the trust boundary.

### 12.2 Agent Execution Boundary

Agents are existing coding harnesses that execute tools within their own process. The device client does not sandbox or gate tool execution — the harness’s own permission model applies (e.g., Claude Code’s allow/deny lists, Codex’s sandbox). The device client’s role is to record all tool activity for audit and relay it to the browser UI.

Future protocol extensions MAY introduce a `tool.approval` message type allowing the device client to request user confirmation before the agent proceeds with a tool execution, but this is out of scope for v1.

### 12.3 Chat MCP Trust Boundary

The Chat MCP server runs inside the agent process and is managed by the wrapper. It is not exposed outside the process. The harness connects to it via stdio transport and treats it as a trusted tool provider. Messages delivered via `read_unread` contain user-provided content and should be treated as untrusted input by the harness, consistent with how harnesses treat user prompts generally.

### 12.4 End-to-End Encryption

The agent protocol operates within a single device. E2EE applies to the separate device-client-to-relay-server-to-browser-client path, which is outside the scope of this specification.

-----

## Appendix A: Message Type Reference

|Type              |Direction     |Capability|Description                       |
|------------------|--------------|----------|----------------------------------|
|`agent.hello`     |Agent → Client|—         |Connection handshake.             |
|`agent.configured`|Client → Agent|—         |Configuration response.           |
|`agent.shutdown`  |Client → Agent|—         |Graceful shutdown request.        |
|`agent.goodbye`   |Agent → Client|—         |Shutdown acknowledgment.          |
|`agent.error`     |Bidirectional |—         |Error notification.               |
|`chat.message`    |Client → Agent|`chat`    |User message delivered to agent.  |
|`chat.response`   |Agent → Client|`chat`    |Agent reply via Chat MCP send.    |
|`chat.cancel`     |Client → Agent|`chat`    |Cancel agent’s current work.      |
|`activity.delta`  |Agent → Client|`activity`|Streaming harness output.         |
|`activity.ping`   |Agent → Client|`activity`|Working indicator (no new output).|
|`activity.end`    |Agent → Client|`activity`|Work unit completed.              |
|`tool.use`        |Agent → Client|`tools`   |Tool invocation event.            |
|`tool.result`     |Agent → Client|`tools`   |Tool execution result event.      |

## Appendix B: Error Code Reference

|Code                 |Description                                 |
|---------------------|--------------------------------------------|
|`invalid_message`    |Message could not be parsed or is malformed.|
|`unknown_type`       |The message type is not recognized.         |
|`protocol_violation` |A protocol constraint was violated.         |
|`capability_mismatch`|Message received for undeclared capability. |
|`version_mismatch`   |Protocol version not supported.             |
|`internal_error`     |Unexpected internal error.                  |
|`reconnect_failed`   |Reconnection could not restore the channel. |

## Appendix C: Chat MCP Tool Reference

|Tool         |Direction        |Description                |
|-------------|-----------------|---------------------------|
|`read_unread`|Harness → Wrapper|Read queued user messages. |
|`send`       |Harness → Wrapper|Send a message to the user.|

These tools are served by the wrapper’s local MCP server and are not visible in the `tool.*` protocol namespace.