"""URL namespace — RELAY_PROTOCOL.md §5.10.

HTTP proxy primarily used for previewing localhost from the browser. Per-tab
cookie isolation; body capped at 500 KB; 15 s timeout.

The per-tab cookie jar is persisted on the facade (`facade._cookie_jars`)
so it survives across requests within the same daemon.
"""

from __future__ import annotations

import base64
import logging
from typing import Any

import httpx

from build_bridge import relay_protocol as proto
from build_bridge.relay_handlers.context import HandlerContext
from build_bridge.relay_session import ActiveSession

log = logging.getLogger(__name__)


_MAX_URL_FETCH = 500_000  # 500 KB max response body


async def handle_url_fetch(
    ctx: HandlerContext,
    session: ActiveSession,
    payload: dict[str, Any],
    ws: Any,
) -> None:
    """Fetch a URL and return the response content (text or base64 binary)."""
    # validator: url required (non-empty).
    url = payload["url"]
    request_id = payload.get("request_id", "")
    tab_id = payload.get("tab_id", "")
    method = (payload.get("method") or "GET").upper()
    body = payload.get("body")
    req_content_type = payload.get("content_type")

    # Per-tab cookie jar (persisted on the facade between calls).
    jars = ctx.cookie_jars
    if tab_id and tab_id not in jars:
        jars[tab_id] = httpx.Cookies()
    cookies = jars.get(tab_id) if tab_id else None

    try:
        async with httpx.AsyncClient(
            follow_redirects=True, timeout=15, cookies=cookies,
        ) as client:
            if method == "POST":
                headers: dict[str, str] = {}
                if req_content_type:
                    headers["content-type"] = req_content_type
                resp = await client.post(url, content=body, headers=headers)
            else:
                resp = await client.get(url)
            # Persist response cookies back to the tab jar.
            if tab_id and cookies is not None:
                cookies.update(resp.cookies)

        content_type = resp.headers.get("content-type", "")
        final_url = str(resp.url)
        is_text = any(t in content_type for t in (
            "text/", "javascript", "json", "xml", "svg",
        ))

        if is_text:
            text_body = resp.text[:_MAX_URL_FETCH]
            truncated = len(resp.text) > _MAX_URL_FETCH
            await ctx.send_frame(session, ws, payload={
                "action": proto.URL_FETCH_RESULT,
                "request_id": request_id,
                "url": url,
                "final_url": final_url,
                "status": resp.status_code,
                "content_type": content_type,
                "content": text_body,
                "is_binary": False,
                "truncated": truncated,
            })
        else:
            raw = resp.content[:_MAX_URL_FETCH]
            b64 = base64.b64encode(raw).decode()
            await ctx.send_frame(session, ws, payload={
                "action": proto.URL_FETCH_RESULT,
                "request_id": request_id,
                "url": url,
                "final_url": final_url,
                "status": resp.status_code,
                "content_type": content_type,
                "content": b64,
                "is_binary": True,
                "truncated": len(resp.content) > _MAX_URL_FETCH,
            })

    except Exception as exc:
        await ctx.send_frame(session, ws, payload={
            "action": proto.URL_FETCH_RESULT,
            "request_id": request_id,
            "url": url,
            "error": str(exc),
        })
