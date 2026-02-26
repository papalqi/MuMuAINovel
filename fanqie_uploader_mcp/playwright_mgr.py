from __future__ import annotations

import asyncio
from typing import Optional

from playwright.async_api import Playwright, async_playwright


_pw: Optional[Playwright] = None
_lock = asyncio.Lock()


async def get_playwright() -> Playwright:
    global _pw
    async with _lock:
        if _pw is None:
            _pw = await async_playwright().start()
        return _pw


async def stop_playwright() -> None:
    global _pw
    async with _lock:
        if _pw is not None:
            await _pw.stop()
            _pw = None

