# Build Relay Protocol v1 Reference

**Status:** draft reference
**Scope:** browser/device application protocol carried through the relay
**Supersedes:** `RELAY_PROTOCOL.md` v0 after migration

This document defines a clean v1 protocol for the Build browser to talk to a
Build device through the relay. It is intentionally not a rename of the v0
`payload.action` API. The goal is a stable set of domain interfaces with
predictable request/response behavior, explicit events, and room for growth.

The relay remains an untrusted router for encrypted traffic. It should know how
to route frames, enforce size limits, and report delivery state. It should not
know application payload contents.

---

## 1. Design Goals

- One envelope for requests, responses, events, and stream chunks.
- Every request has a caller-generated `id`; every response references it.
- Every request completes with exactly one terminal response.
- Long-running operations use progress events or stream chunks, not ad hoc
  partial result shapes.
- Broadcasts are explicit events, not response-shaped side effects.
- Domain APIs model what the UI needs: channels, messages, agents, activity,
  interactions, terminal, files, uploads, URL fetch, and complications.
- Validation errors, authorization errors, missing resources, and handler
  failures use the same error object.
- The transport and E2EE session protocol are separate from the application
  API.

---

## 2. Layering

### 2.1 Relay Transport

The device maintains a signed WebSocket connection to the relay. The browser
does not connect directly to the device.

Relay-visible carrier messages:

```jsonc
// Browser/device session setup.
{ "type": "session.init", "session_id": "<uuid>", "init": { ... } }
{ "type": "session.accept", "session_id": "<uuid>", "accept": { ... } }

// Encrypted application frame.
{ "type": "session.frame", "session_id": "<uuid>", "frame": "<base64 ciphertext>" }

// Session teardown.
{ "type": "session.close", "session_id": "<uuid>", "reason": "<string>?" }
```

The relay only validates carrier shape, device ownership, active session
routing, and encrypted frame size.

### 2.2 E2EE Session

The E2EE handshake continues to use X25519 session establishment and encrypted
AES-GCM frames. The decrypted frame is the v1 application envelope below.

Browser-to-device application frames MUST have:

```jsonc
{
  "v": 1,
  "kind": "request" | "close",
  "id": "<id>",
  "type": "<domain.method>",
  "target": { "device_id": "<uuid>", "channel_id": "<uuid>?" },
  "payload": {},
  "meta": {}
}
```

Device-to-browser application frames MUST have:

```jsonc
{
  "v": 1,
  "kind": "response" | "event" | "stream" | "close",
  "id": "<id>",
  "ref": "<request id>?",
  "type": "<domain.method|domain.event>",
  "target": { "device_id": "<uuid>", "channel_id": "<uuid>?" },
  "payload": {},
  "error": null,
  "meta": {}
}
```

Required invariants:

- `v` is required and MUST equal `1`.
- `kind` is required.
- `id` is required on every frame and unique within a session.
- `ref` is required on `response` and `stream` frames.
- `type` is required on every non-close frame.
- `target.device_id` is required.
- `target.channel_id` is required for channel-scoped APIs.
- `payload` is always an object. Use `{}` when empty.
- `error` is `null` on success and an error object on failure responses.
- Receivers MUST ignore unknown fields.
- Receivers MUST reject unknown `v`, `kind`, missing required fields, and
  invalid target shape with a protocol error response when possible.

### 2.3 Common Metadata

`meta` is optional per-field but always present as an object.

Common request metadata:

```jsonc
{
  "sent_at": "<iso8601>?",
  "trace_id": "<id>?",
  "strict_agent_delivery": false,
  "timeout_ms": 30000
}
```

Common response/event metadata:

```jsonc
{
  "sent_at": "<iso8601>?",
  "trace_id": "<id>?",
  "seq": 0
}
```

`trace_id` is copied from request to response when present. `seq` is only used
for `stream` frames.

---

## 3. Request/Response Semantics

### 3.1 Unary Requests

Most operations are unary:

```jsonc
// Request
{
  "v": 1,
  "kind": "request",
  "id": "req_01J...",
  "type": "channel.list",
  "target": { "device_id": "<uuid>" },
  "payload": {},
  "meta": {}
}

// Response
{
  "v": 1,
  "kind": "response",
  "id": "res_01J...",
  "ref": "req_01J...",
  "type": "channel.list",
  "target": { "device_id": "<uuid>" },
  "payload": { "channels": [] },
  "error": null,
  "meta": {}
}
```

Every request MUST receive one terminal `response` frame unless the session
closes first.

### 3.2 Streaming Requests

Streaming operations emit zero or more `stream` frames and exactly one terminal
`response` frame.

```jsonc
{
  "v": 1,
  "kind": "stream",
  "id": "str_01J...",
  "ref": "req_01J...",
  "type": "terminal.exec.output",
  "target": { "device_id": "<uuid>", "channel_id": "<uuid>" },
  "payload": { "stdout": "..." },
  "error": null,
  "meta": { "seq": 3 }
}
```

`meta.seq` is monotonic per request and starts at `0`. Missing stream chunks
are detectable. The terminal `response` carries final status and no `done`
boolean is needed.

### 3.3 Cancellation

Any in-flight request can be cancelled by id. Cancellation is itself a normal
request and therefore gets its own terminal response:

```jsonc
{
  "v": 1,
  "kind": "request",
  "id": "can_01J...",
  "type": "request.cancel",
  "target": { "device_id": "<uuid>" },
  "payload": { "request_id": "req_01J...", "reason": "user" },
  "meta": {}
}
```

The cancellation request completes with `{ "cancelled": true }` if the target
request was found and cancellation was signalled. The original request also
completes with a terminal response whose error code is `cancelled` unless the
operation had already completed.

### 3.4 Events

Events are device-initiated or relay-initiated notifications. They never
complete a request.

```jsonc
{
  "v": 1,
  "kind": "event",
  "id": "evt_01J...",
  "type": "message.created",
  "target": { "device_id": "<uuid>", "channel_id": "<uuid>" },
  "payload": { "message": { ... } },
  "error": null,
  "meta": { "occurred_at": "2026-05-15T21:00:00Z" }
}
```

Events SHOULD be idempotent. Consumers SHOULD dedupe by `id`.

---

## 4. Error Model

All failed terminal responses use:

```jsonc
{
  "code": "invalid_request",
  "message": "channel_id is required",
  "details": {
    "field": "target.channel_id"
  },
  "retryable": false
}
```

Standard error codes:

| Code | Meaning |
|------|---------|
| `invalid_frame` | Envelope is malformed or violates v1 invariants. |
| `invalid_request` | Payload is malformed for the requested method. |
| `unknown_method` | `type` is not implemented by the receiver. |
| `not_found` | Referenced channel/message/file/session was not found. |
| `conflict` | Request conflicts with current state. |
| `cancelled` | Request was cancelled. |
| `agent_unavailable` | Request requires an agent but none is available. |
| `permission_denied` | Operation is not authorized by policy. |
| `payload_too_large` | Request or response exceeds frame limits. |
| `internal` | Unexpected handler failure. |
| `timeout` | Operation timed out. |

Rules:

- A failed request still receives `kind: "response"` and `ref` to the request.
- `payload` on failed responses SHOULD be `{}`.
- Handlers MUST NOT fail silently.
- Events MAY carry error state in their payload when the event itself is about
  an error, but request failures MUST use the response error object.

---

## 5. Domain Interfaces

Method names use `<domain>.<verb>` for requests and `<domain>.<event>` for
events. Domains are independently extensible.

### 5.0 Protocol

#### `protocol.hello`

Target: device

`protocol.hello` SHOULD be the first v1 request after session accept. Until it
completes, the device SHOULD reject non-protocol requests with
`invalid_request`.

Request:

```jsonc
{
  "client": { "name": "build-web", "version": "<string>" },
  "accept_versions": [1],
  "features": ["streaming", "uploads.v2", "projects.v1", "project.create", "project.repo.list", "plans.v1", "worktree.create", "worktree.snapshot"]
}
```

Response:

```jsonc
{
  "version": 1,
  "features": ["streaming", "uploads.v2", "projects.v1", "project.create", "project.repo.list", "plans.v1", "worktree.create", "worktree.snapshot", "dashboard.snapshot"],
  "limits": { "max_encrypted_frame_bytes": 262144 }
}
```

#### `request.cancel`

Target: device, or channel when cancelling a channel-scoped request

Request:

```jsonc
{ "request_id": "<request id>", "reason": "user" }
```

Response:

```jsonc
{ "cancelled": true }
```

### 5.1 Projects

Projects are durable workspace/repository containers. Worktrees are durable
children of projects. A worktree may have a `channel_id` when an agent
conversation is attached, but channels remain the chat/control primitive rather
than the project identity.

#### `project.list`

Target: device

Request:

```jsonc
{}
```

Response:

```jsonc
{
  "projects": [
    {
      "id": "<project id>",
      "name": "<repository or workspace>",
      "root_path": "<absolute path>?",
      "repo": "<string>",
      "default_branch": "main",
      "color": "#6366f1",
      "created_at": 1715788800.0,
      "updated_at": 1715788800.0,
      "worktree_count": 2
    }
  ]
}
```

#### `project.create`

Target: device

Request:

```jsonc
{
  "name": "<project name>",
  "root_path": "<absolute project directory>"
}
```

Response:

```jsonc
{
  "project": {
    "id": "<project id>",
    "name": "<project name>",
    "root_path": "<absolute project directory>",
    "repo": "<directory or root git repo name>",
    "default_branch": "main",
    "color": "#6366f1",
    "created_at": 1715788800.0,
    "updated_at": 1715788800.0,
    "worktree_count": 0
  }
}
```

#### `project.repo.list`

Target: device

Request:

```jsonc
{ "project_id": "<project id>" }
```

Response:

```jsonc
{
  "project_id": "<project id>",
  "root_path": "<absolute project directory>",
  "repos": [
    {
      "id": "<repo id>",
      "name": "owner/repo or directory name",
      "path": "<absolute git repository root>",
      "relative_path": "." | "services/api",
      "branch": "main",
      "upstream": "origin/main?",
      "remote": "owner/repo?",
      "is_root": false
    }
  ],
  "error": null
}
```

#### `worktree.list`

Target: device

Request:

```jsonc
{ "project_id": "<project id>?" }
```

Response:

```jsonc
{
  "worktrees": [
    {
      "id": "<worktree id>",
      "project_id": "<project id>",
      "channel_id": "<channel id>?",
      "name": "<string>",
      "path": "<absolute path>?",
      "branch": "<string>",
      "base_ref": "<git ref>?",
      "head_ref": "<git ref>?",
      "status": "working" | "idle" | "blocked" | "closed",
      "created_at": 1715788800.0,
      "updated_at": 1715788800.0
    }
  ]
}
```

#### `worktree.create`

Target: device

Request:

```jsonc
{
  "project_id": "<project id>",
  "name": "<display name>?",
  "repo_path": "<absolute or project-relative git repository root>?",
  "branch": "agents/<branch>?",
  "path": "<absolute path>?",
  "base_ref": "main?",
  "create_git_worktree": false,
  "agent": {
    "harness": "codex?",
    "model": "<model id>?",
    "effort": "<effort>?",
    "system_prompt": "<prompt>?",
    "auto_approve_tools": false
  }
}
```

Response:

```jsonc
{
  "project": { "id": "<project id>", "name": "<string>" },
  "worktree": {
    "id": "<worktree id>",
    "project_id": "<project id>",
    "channel_id": "<channel id>",
    "name": "<display name>",
    "path": "<absolute path>?",
    "branch": "agents/<branch>",
    "status": "idle"
  },
  "plan": { "id": "<plan id>", "worktree_id": "<worktree id>", "channel_id": "<channel id>" },
  "channel": { "id": "<channel id>", "name": "<display name>" },
  "agent": { "agent_id": "<agent id>", "harness": "codex", "model": "<model id>", "pid": 12345 },
  "agent_error": null,
  "git": { "created": false, "error": null }
}
```

If `create_git_worktree` is true and `git worktree add` fails, the device
returns `failed_precondition` and does not create a channel or start an agent.

#### `worktree.snapshot`

Target: device

Request:

```jsonc
{ "worktree_id": "<worktree id>" }
```

Response:

```jsonc
{
  "schema": "worktree.snapshot.v1",
  "worktree": { "id": "<worktree id>", "project_id": "<project id>" },
  "project": { "id": "<project id>", "name": "<string>" },
  "agent": {
    "harness": "codex",
    "model": "gpt-5.2",
    "effort": "high",
    "auto_approve_tools": false,
    "working_directory": "<absolute path>",
    "status": "active",
    "is_running": true
  },
  "workspace": "<absolute path>?",
  "git": {
    "repo_path": ".",
    "branch": "main",
    "upstream": "origin/main",
    "staged": 0,
    "unstaged": 2,
    "commits": [
      { "sha": "abc1234", "full_sha": "<sha>", "message": "<subject>", "time": "4m" }
    ],
    "error": null
  },
  "files": [
    { "path": "src/app.ts", "status": "M", "add": 12, "del": 3 }
  ],
  "diffs": [
    {
      "file": "src/app.ts",
      "add": 12,
      "del": 3,
      "lines": [
        { "type": "ctx", "text": "unchanged line" },
        { "type": "add", "text": "new line" },
        { "type": "del", "text": "old line" }
      ]
    }
  ],
  "tests": []
}
```

#### `plan.list`

Target: device

Request:

```jsonc
{
  "project_id": "<project id>?",
  "worktree_id": "<worktree id>?"
}
```

Response:

```jsonc
{
  "plans": [
    {
      "id": "<plan id>",
      "project_id": "<project id>",
      "worktree_id": "<worktree id>?",
      "channel_id": "<channel id>?",
      "title": "<string>",
      "status": "draft" | "queued" | "in-progress" | "review" | "done",
      "body": "<markdown>",
      "step_count": 4,
      "done_step_count": 1,
      "model": "<string>?",
      "created_at": 1715788800.0,
      "updated_at": 1715788800.0
    }
  ]
}
```

### 5.2 Dashboard

#### `dashboard.snapshot`

Target: device

Request:

```jsonc
{}
```

Response:

```jsonc
{
  "schema": "dashboard.snapshot.v1",
  "source": "projects",
  "generated_at": 1715788800.0,
  "projects": [
    {
      "id": "<slug>",
      "name": "<repository or workspace>",
      "description": "<string>",
      "repo": "<string>",
      "branch": "<string>",
      "color": "#6366f1",
      "root_path": "<absolute path>?",
      "needsYou": 0,
      "runningAgents": 1,
      "queued": 0,
      "lastActive": "2m",
      "worktrees": [
        {
          "id": "<worktree id>",
          "project_id": "<project id>",
          "channel_id": "<channel id>?",
          "branch": "<git branch or workspace>",
          "plan": "plan-<worktree prefix>",
          "model": "<model>",
          "agent": "<display agent>",
          "status": "working" | "idle" | "blocked",
          "summary": "<channel name>",
          "device": "<device id>",
          "workspace": "<absolute path>?",
          "pct": 40,
          "files": 0,
          "add": 0,
          "del": 0,
          "updated": "2m"
        }
      ],
      "plans": [
        {
          "id": "<plan id>",
          "project_id": "<project id>",
          "worktree_id": "<worktree id>?",
          "channel_id": "<channel id>?",
          "title": "<plan title>",
          "status": "draft" | "queued" | "in-progress" | "review" | "done",
          "steps": 1,
          "doneSteps": 0,
          "model": "<model>",
          "updated": "2m"
        }
      ],
      "activity": ["<string>"]
    }
  ]
}
```

This snapshot reads project/worktree/plan primitives, then joins current
channel and agent metadata for live status. Existing channel-only data is
lazily migrated into one project/worktree/plan graph per workspace.

### 5.3 Channels

#### `channel.list`

Target: device

Request:

```jsonc
{}
```

Response:

```jsonc
{
  "channels": [
    {
      "id": "<uuid>",
      "name": "<string>",
      "created_at": "<iso8601>",
      "agent": {
        "harness": "<string>",
        "display_name": "<string>",
        "model": "<string>",
        "effort": "<string>?",
        "working_directory": "<string>?",
        "plan_mode": false,
        "auto_approve_tools": false,
        "status": "running" | "idle" | "stopped" | "error",
        "last_seen_at": "<iso8601>?"
      }
    }
  ],
  "device": {
    "cwd": "<absolute path>"
  }
}
```

#### `channel.create`

Target: device

Request:

```jsonc
{
  "name": "<string>",
  "agent": {
    "harness": "<string>?",
    "model": "<string>?",
    "effort": "<string>?",
    "system_prompt": "<string>?",
    "working_directory": "<string>?",
    "auto_approve_tools": false
  }
}
```

Response:

```jsonc
{ "channel": { ... }, "agent_started": true, "worker": { "pid": 1234 } }
```

#### `channel.update`

Target: channel

Request:

```jsonc
{
  "name": "<string>?",
  "agent": {
    "harness": "<string>?",
    "model": "<string>?",
    "effort": "<string>?",
    "working_directory": "<string>?",
    "auto_approve_tools": false
  }
}
```

Response:

```jsonc
{ "channel": { ... }, "restart_required": false }
```

#### `channel.delete`

Target: channel

Request:

```jsonc
{ "stop_agent": true }
```

Response:

```jsonc
{ "deleted": true }
```

Events:

- `channel.created`
- `channel.updated`
- `channel.deleted`
- `channel.presence_changed`

### 5.2 Messages

#### `message.list`

Target: channel

Request:

```jsonc
{
  "limit": 50,
  "before": "<cursor>?"
}
```

Response:

```jsonc
{
  "messages": [
    {
      "id": "<uuid>",
      "sender": "user" | "agent" | "system",
      "display_name": "<string>?",
      "content": "<string>",
      "attachments": [],
      "created_at": "<iso8601>",
      "delivered_at": "<iso8601>?",
      "read_at": "<iso8601>?",
      "metadata": {}
    }
  ],
  "next_cursor": "<cursor>?"
}
```

#### `message.send`

Target: channel

Request:

```jsonc
{
  "message_id": "<uuid>",
  "content": "<string>",
  "attachments": [
    { "upload_id": "<uuid>", "filename": "<string>", "mime_type": "<string>", "size": 123 }
  ],
  "agent_options": {
    "model": "<string>?",
    "effort": "<string>?",
    "plan_mode": false
  }
}
```

Response:

```jsonc
{
  "message": { ... },
  "agent_delivery": "accepted" | "queued" | "unavailable"
}
```

If the message reaches the device store but cannot reach the agent, the
response succeeds with `agent_delivery: "unavailable"` unless the client
requested strict agent delivery in `meta`.

Events:

- `message.created`
- `message.updated`
- `message.read`
- `message.delivery_changed`

#### `message.mark_read`

Target: channel

Request:

```jsonc
{ "message_ids": ["<uuid>"] }
```

Response:

```jsonc
{ "read": ["<uuid>"] }
```

#### `message.mark_seen`

Target: channel

Request:

```jsonc
{ "seen_at": "<iso8601>" }
```

Response:

```jsonc
{ "seen_at": "<iso8601>" }
```

### 5.3 Agents

#### `agent.harnesses`

Target: device

Request:

```jsonc
{}
```

Response:

```jsonc
{
  "harnesses": [
    {
      "id": "codex",
      "display_name": "Codex",
      "installed": true,
      "default_model": "gpt-5.4",
      "models": [],
      "efforts": []
    }
  ]
}
```

#### `agent.workers`

Target: device

Request:

```jsonc
{}
```

Response:

```jsonc
{
  "workers": [
    {
      "channel_id": "<uuid>",
      "agent_id": "<string>",
      "harness": "<string>",
      "model": "<string>",
      "pid": 1234,
      "status": "running" | "stopped"
    }
  ]
}
```

#### `agent.start`

Target: channel

Request:

```jsonc
{
  "harness": "<string>",
  "model": "<string>?",
  "effort": "<string>?",
  "system_prompt": "<string>?",
  "working_directory": "<string>?"
}
```

Response:

```jsonc
{
  "agent_id": "<string>",
  "status": "running",
  "worker": { "pid": 1234 }
}
```

#### `agent.stop`

Target: channel

Request:

```jsonc
{ "mode": "graceful" | "force", "timeout_ms": 3000 }
```

Response:

```jsonc
{ "was_running": true, "stopped": true, "forced": false }
```

#### `agent.restart`

Target: channel

Request:

```jsonc
{ "preserve_session": true }
```

Response:

```jsonc
{ "agent_id": "<string>", "status": "running", "worker": { "pid": 1234 } }
```

#### `agent.cancel_turn`

Target: channel

Request:

```jsonc
{ "reason": "user" }
```

Response:

```jsonc
{ "sent": true }
```

Events:

- `agent.connected`
- `agent.disconnected`
- `agent.status_changed`
- `agent.error`
- `agent.state_changed`

### 5.4 Activity

#### `activity.list`

Target: channel

Request:

```jsonc
{ "limit": 200, "before": "<cursor>?" }
```

Response:

```jsonc
{
  "entries": [
    {
      "id": "<uuid>",
      "kind": "text" | "tool_use" | "tool_result" | "end",
      "payload": {},
      "created_at": "<iso8601>"
    }
  ],
  "next_cursor": "<cursor>?"
}
```

Events:

- `activity.delta`
- `activity.ping`
- `activity.end`
- `tool.used`
- `tool.completed`
- `task_plan.updated`
- `workspace.changed`

BAP events SHOULD be normalized into these browser-facing event types. The
browser should not need to know every BAP message type.

### 5.5 Interactions

#### `interaction.respond`

Target: channel

Request:

```jsonc
{
  "interaction_id": "<uuid>",
  "response": {
    "selected_option": "<string>?",
    "selected_options": ["<string>"],
    "freeform": "<string>?",
    "step_answers": [
      { "id": "<string>?", "header": "<string>?", "answer": "<string>" }
    ]
  }
}
```

Response:

```jsonc
{ "accepted": true }
```

Events:

- `interaction.requested`
- `interaction.resolved`
- `interaction.cancelled`

Interaction request event payload:

```jsonc
{
  "interaction": {
    "id": "<uuid>",
    "kind": "question" | "approval" | "plan_review",
    "question": "<string>",
    "options": [{ "id": "<string>", "label": "<string>", "description": "<string>?" }],
    "allow_freeform": true,
    "multiselect": false,
    "plan": "<string>?",
    "questions": []
  }
}
```

### 5.6 Sessions

#### `session.reset_agent_context`

Target: channel

Request:

```jsonc
{ "restart_agent": true }
```

Response:

```jsonc
{ "session_start_at": "<iso8601>", "agent_restarted": true }
```

#### `session.compact_agent_context`

Target: channel

Request:

```jsonc
{ "restart_agent": true }
```

Response:

```jsonc
{ "summary_message_id": "<uuid>", "session_start_at": "<iso8601>" }
```

Events:

- `session.compaction_started`
- `session.reset`

### 5.7 Terminal

#### `terminal.exec`

Target: channel

Streaming request.

Request:

```jsonc
{
  "command": "<string>",
  "cwd": "<absolute or channel-relative path>?",
  "env": {},
  "timeout_ms": 0
}
```

Stream type: `terminal.exec.output`

```jsonc
{ "stdout": "<string>?", "stderr": "<string>?", "cwd": "<string>?" }
```

Terminal response:

```jsonc
{
  "exit_code": 0,
  "signal": null,
  "cwd": "<final cwd>",
  "duration_ms": 1234
}
```

#### `terminal.kill`

Target: channel

Request:

```jsonc
{ "request_id": "<terminal.exec request id>?" }
```

Response:

```jsonc
{ "killed": true }
```

#### `terminal.complete`

Target: channel

Request:

```jsonc
{ "line": "<string>", "cursor": 12, "cwd": "<string>?" }
```

Response:

```jsonc
{ "completions": ["<string>"] }
```

### 5.8 Files

All file paths are resolved relative to the channel working directory unless
`base` is supplied. The device MUST reject paths that escape the allowed root.

#### `file.tree`

Target: channel

Request:

```jsonc
{
  "path": ".",
  "depth": 2,
  "include_git": true,
  "limit": 1000
}
```

Response:

```jsonc
{
  "root": "<absolute path>",
  "entries": [
    {
      "path": "<relative path>",
      "type": "file" | "directory" | "symlink",
      "size": 123,
      "modified_at": "<iso8601>?",
      "git": { "status": "modified" }
    }
  ],
  "truncated": false
}
```

#### `file.read`

Target: channel

Streaming for large/binary content.

Request:

```jsonc
{ "path": "<relative path>", "offset": 0, "limit": 100000, "encoding": "utf8" | "base64" }
```

Stream type: `file.read.chunk`

```jsonc
{ "offset": 0, "data": "<string>", "encoding": "utf8" | "base64" }
```

Terminal response:

```jsonc
{
  "path": "<relative path>",
  "size": 123,
  "mime_type": "<string>",
  "encoding": "utf8" | "base64",
  "truncated": false
}
```

#### `file.diff`

Target: channel

Request:

```jsonc
{
  "path": "<relative path>",
  "repo_path": "<relative path>?",
  "staged": false,
  "base_ref": "<ref>?",
  "head_ref": "<ref>?"
}
```

Response:

```jsonc
{ "path": "<relative path>", "diff": "<unified diff>", "truncated": false }
```

#### `file.changes`

Target: channel

Request:

```jsonc
{ "repo_path": ".", "base_ref": "<ref>?", "head_ref": "<ref>?" }
```

Response:

```jsonc
{ "files": [{ "path": "<string>", "status": "modified", "stats": {} }] }
```

#### `file.commits`

Target: channel

Request:

```jsonc
{ "repo_path": ".", "limit": 50 }
```

Response:

```jsonc
{ "commits": [{ "sha": "<sha>", "subject": "<string>", "author": "<string>", "date": "<iso8601>" }] }
```

Events:

- `file.changed`
- `file.tree_changed`

### 5.9 Uploads

Uploads are explicit resources rather than chunk frames coupled to chat
messages.

#### `upload.create`

Target: channel

Request:

```jsonc
{
  "filename": "<string>",
  "mime_type": "<string>",
  "size": 123,
  "sha256": "<hex>",
  "destination": { "kind": "scratch" | "workspace", "path": "<relative path>?" }
}
```

Response:

```jsonc
{
  "upload_id": "<uuid>",
  "chunk_size": 65536,
  "accepted": true
}
```

#### `upload.write_chunk`

Target: channel

Request:

```jsonc
{ "upload_id": "<uuid>", "index": 0, "data": "<base64>" }
```

Response:

```jsonc
{ "upload_id": "<uuid>", "index": 0, "received": true }
```

#### `upload.complete`

Target: channel

Request:

```jsonc
{ "upload_id": "<uuid>" }
```

Response:

```jsonc
{ "upload": { "id": "<uuid>", "path": "<absolute path>", "sha256": "<hex>" } }
```

Events:

- `upload.completed`

### 5.10 URL Fetch

#### `url.fetch`

Target: device

Request:

```jsonc
{
  "url": "https://example.com",
  "method": "GET" | "POST",
  "headers": {},
  "body": "<string>?",
  "tab_id": "<string>?"
}
```

Response:

```jsonc
{
  "url": "<final url>",
  "status": 200,
  "headers": {},
  "mime_type": "<string>",
  "encoding": "utf8" | "base64",
  "body": "<string>",
  "truncated": false
}
```

The device owns cookie jars keyed by `tab_id`. The response body MUST be
bounded and may be truncated.

### 5.11 Complications

#### `complication.list`

Target: channel

Request:

```jsonc
{}
```

Response:

```jsonc
{ "complications": [{ "id": "<string>", "kind": "<string>", "data": {}, "options": [] }] }
```

#### `complication.invoke`

Target: channel

Request:

```jsonc
{ "complication_id": "<string>", "option_id": "<string>", "params": {} }
```

Response:

```jsonc
{ "accepted": true }
```

Events:

- `complication.updated`
- `complication.removed`

---

## 6. Event Payload Reference

### 6.1 Agent Events

Browser-facing agent events SHOULD be normalized:

| Event | Payload |
|-------|---------|
| `agent.connected` | `{ agent_id, harness, model, capabilities }` |
| `agent.disconnected` | `{ agent_id, reason? }` |
| `agent.status_changed` | `{ status }` |
| `agent.error` | `{ code, message, fatal }` |
| `agent.state_changed` | `{ plan_mode?, resume_cursor?, read_message_ids? }` |

### 6.2 Activity Events

| Event | Payload |
|-------|---------|
| `activity.delta` | `{ delta: { type, text? }, index }` |
| `activity.ping` | `{}` |
| `activity.end` | `{ reason, usage? }` |
| `tool.used` | `{ tool_use_id, name, input, created_at }` |
| `tool.completed` | `{ tool_use_id, content, is_error, completed_at }` |
| `task_plan.updated` | `{ todos: [{ id, content, status }] }` |
| `workspace.changed` | `{ paths: ["<relative path>"] }` |

---

## 7. Validation Requirements

Every method MUST have:

- A request payload schema.
- A response payload schema.
- An error behavior.
- A test that invalid required fields return `invalid_request`.
- A test that unknown methods return `unknown_method`.
- A test that every request path emits exactly one terminal response.

Validation should happen before handler side effects. Business-rule errors
(`not_found`, `conflict`, `agent_unavailable`) are handler errors but still use
the same terminal response shape.

---

## 8. Size Limits and Chunking

Default encrypted frame limit: 256 KiB.

Rules:

- A sender MUST NOT emit an encrypted frame above the relay limit.
- Large file reads, terminal output, and uploads use streaming frames.
- List endpoints MUST support `limit` and SHOULD support cursors.
- Responses that can still be too large MUST set `truncated: true` or fail with
  `payload_too_large`; silent truncation is forbidden.

---

## 9. Compatibility Strategy

The device can support v0 and v1 in parallel during migration:

1. Browser sends `protocol.hello` after session accept.
2. Device responds with supported versions and feature flags.
3. If v1 is available, browser uses v1 envelopes.
4. If v1 is unavailable, browser falls back to v0 `payload.action` frames.

The `protocol.hello` payload is defined in §5.0.

---

## 10. v0 Migration Map

| v0 action | v1 method/event |
|-----------|-----------------|
| persisted project/worktree/plan graph + agent metadata | `project.list`, `project.create`, `project.repo.list`, `worktree.list`, `worktree.create`, `worktree.snapshot`, `plan.list`, `dashboard.snapshot` |
| `list_channels` | `channel.list` |
| `create_channel` | `channel.create` |
| `rename_channel` | `channel.update` |
| `update_channel` | `channel.update` |
| `delete_channel` | `channel.delete` |
| `get_messages` | `message.list` |
| `message` | `message.send` + `message.created` |
| `retry_message` | `message.send` with same `message_id`, or `message.retry` if needed |
| `mark_read` | `message.mark_read` |
| `mark_seen` | `message.mark_seen` |
| `get_activity` | `activity.list` |
| `list_harnesses` | `agent.harnesses` |
| `list_workers` | `agent.workers` if still needed |
| `start_agent` | `agent.start` |
| `stop_agent` | `agent.stop` |
| `cancel` | `agent.cancel_turn` |
| `restart_agent` | `agent.restart` |
| `interaction_response` | `interaction.respond` |
| `reset_session` | `session.reset_agent_context` |
| `compact_session` | `session.compact_agent_context` |
| `get_complications` | `complication.list` |
| `complication:action` | `complication.invoke` |
| `terminal_exec` | `terminal.exec` |
| `terminal_kill` | `terminal.kill` |
| `terminal_complete` | `terminal.complete` |
| `files_list` | `file.tree` |
| `files_changes` | `file.changes` |
| `files_commits` | `file.commits` |
| `file_read` | `file.read` |
| `file_diff` | `file.diff` |
| `url_fetch` | `url.fetch` |
| `upload_chunk` | `upload.create` + `upload.write_chunk` |
| `upload_complete` | `upload.complete` |
| `agent_event` | normalized `agent.*`, `activity.*`, `tool.*`, `interaction.*`, `workspace.*` events |
| `system_message` | `message.created` with `sender: "system"` |
| `complication:update` | `complication.updated` |
| `complication:remove` | `complication.removed` |

---

## 11. Open Decisions

- Whether to use ULID, UUIDv7, or existing UUID strings for frame IDs.
- Whether strict agent delivery should be the default for `message.send`.
- Whether agent/BAP event names should pass through raw in a debug namespace.
- Whether upload destination `workspace` should require explicit user approval.
- Whether `url.fetch` belongs in this protocol long term or should move behind
  a browser-originated capability gate.
