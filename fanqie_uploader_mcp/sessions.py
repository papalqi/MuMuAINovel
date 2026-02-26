from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Dict, Optional

from playwright.async_api import Browser, BrowserContext, Page

from fanqie_uploader_mcp.settings import LOGIN_SESSION_TTL_SECONDS
from fanqie_uploader_mcp.utils import new_id


@dataclass
class LoginSession:
    session_id: str
    account_key: str
    browser: Browser
    context: BrowserContext
    page: Page
    created_at: float
    trace_enabled: bool = False


_sessions: Dict[str, LoginSession] = {}
_lock = asyncio.Lock()


async def put_session(sess: LoginSession) -> None:
    async with _lock:
        _sessions[sess.session_id] = sess


async def get_session(session_id: str) -> Optional[LoginSession]:
    async with _lock:
        return _sessions.get(session_id)


async def pop_session(session_id: str) -> Optional[LoginSession]:
    async with _lock:
        return _sessions.pop(session_id, None)


async def list_sessions() -> Dict[str, Dict]:
    now = time.time()
    async with _lock:
        return {
            k: {
                "session_id": v.session_id,
                "account_key": v.account_key,
                "age_seconds": round(now - v.created_at, 1),
                "trace_enabled": v.trace_enabled,
            }
            for k, v in _sessions.items()
        }


async def cleanup_expired_sessions() -> int:
    """
    清理超时登录会话，避免浏览器句柄泄漏。
    """
    now = time.time()
    expired: Dict[str, LoginSession] = {}
    async with _lock:
        for sid, sess in list(_sessions.items()):
            if now - sess.created_at > LOGIN_SESSION_TTL_SECONDS:
                expired[sid] = sess
                _sessions.pop(sid, None)

    for sess in expired.values():
        try:
            await sess.context.close()
        except Exception:
            pass
        try:
            await sess.browser.close()
        except Exception:
            pass
    return len(expired)


def new_session_id() -> str:
    return new_id("fq_login")

