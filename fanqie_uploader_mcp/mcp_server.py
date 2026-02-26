from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP  # type: ignore
from playwright.async_api import Browser, BrowserContext, Page

from fanqie_uploader_mcp import __version__
from fanqie_uploader_mcp.fanqie import (
    _launch_browser,
    _new_context,
    _screenshot,
    check_logged_in,
    login_open_page,
    open_authed_page,
    save_storage_state,
    work_create_draft_ui,
    chapter_upsert_draft_ui,
)
from fanqie_uploader_mcp.sessions import (
    LoginSession,
    cleanup_expired_sessions,
    get_session,
    list_sessions,
    new_session_id,
    pop_session,
    put_session,
)
from fanqie_uploader_mcp.settings import DEFAULT_HEADLESS, DEFAULT_TRACE, FANQIE_LOGIN_URL, STATE_DIR


def _new_mcp() -> FastMCP:
    instructions = (
        "Fanqie Uploader MCP：用于番茄作家后台自动化（Playwright）。"
        "推荐流程：login_start -> 展示二维码 -> login_poll 保存登录态 -> work_create_draft/chapter_upsert_draft。"
    )
    try:
        return FastMCP(
            name="fanqie-uploader-mcp",
            instructions=instructions,
            stateless_http=True,
            json_response=True,
        )
    except TypeError:
        # 兼容旧版 FastMCP 构造签名
        try:
            return FastMCP(name="fanqie-uploader-mcp", instructions=instructions)
        except TypeError:
            return FastMCP("fanqie-uploader-mcp")


fanqie_mcp: FastMCP = _new_mcp()


@fanqie_mcp.tool()
async def fanqie_health() -> Dict[str, Any]:
    """健康检查。"""
    cleaned = await cleanup_expired_sessions()
    return {
        "ok": True,
        "service": "fanqie-uploader-mcp",
        "version": __version__,
        "login_url": FANQIE_LOGIN_URL,
        "state_dir": str(STATE_DIR),
        "sessions_cleaned": cleaned,
        "sessions": await list_sessions(),
        "ts": int(time.time()),
    }


@fanqie_mcp.tool()
async def fanqie_login_start(
    account_key: str = "default",
    headless: Optional[bool] = None,
    trace: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    打开番茄作家后台登录页并截图（用于扫码登录）。

    返回：
    - session_id：用于后续 login_poll / login_cancel
    - screenshot_base64：当前页面截图（建议前端直接展示给用户扫码）
    """
    browser: Browser | None = None
    context: BrowserContext | None = None
    page: Page | None = None
    try:
        browser = await _launch_browser(headless=headless)
        context = await _new_context(browser)

        if trace is None:
            trace = DEFAULT_TRACE
        if trace:
            await context.tracing.start(screenshots=True, snapshots=True, sources=False)

        page = await context.new_page()
        await login_open_page(page)

        shot = await _screenshot(page, tag=f"login_start_{account_key}", full_page=True)

        sid = new_session_id()
        await put_session(
            LoginSession(
                session_id=sid,
                account_key=account_key,
                browser=browser,
                context=context,
                page=page,
                created_at=time.time(),
                trace_enabled=bool(trace),
            )
        )

        return {
            "ok": True,
            "session_id": sid,
            "account_key": account_key,
            "headless": DEFAULT_HEADLESS if headless is None else bool(headless),
            "trace": bool(trace),
            "url": page.url,
            "message": "请扫码登录，然后反复调用 fanqie_login_poll 直到 logged_in=true。",
            **shot,
        }
    except Exception as e:
        # 失败时，尽量带回截图
        res = {"ok": False, "error": str(e), "account_key": account_key}
        if page is not None:
            res.update(await _screenshot(page, tag="login_start_failed", full_page=True))

        # 清理句柄
        try:
            if context is not None:
                await context.close()
        except Exception:
            pass
        try:
            if browser is not None:
                await browser.close()
        except Exception:
            pass
        return res


@fanqie_mcp.tool()
async def fanqie_login_poll(
    session_id: str,
    save_state: bool = True,
    include_screenshot: bool = False,
) -> Dict[str, Any]:
    """
    轮询登录状态。

    - 若已登录：保存 storage_state 文件并关闭会话（默认）
    - 若未登录：返回 logged_in=false（可选返回当前截图）
    """
    sess = await get_session(session_id)
    if not sess:
        return {"ok": False, "error": f"session_id 不存在或已过期: {session_id}"}

    ok, info = await check_logged_in(sess.context)
    if not ok:
        res = {
            "ok": True,
            "logged_in": False,
            "session_id": session_id,
            "account_key": sess.account_key,
            "info": info,
        }
        if include_screenshot:
            res.update(await _screenshot(sess.page, tag=f"login_poll_{sess.account_key}", full_page=True))
        return res

    # 已登录：保存状态 -> 关闭 session
    storage_state_path = None
    if save_state:
        storage_state_path = await save_storage_state(sess.context, sess.account_key)

    # 从 registry 移除并关闭资源
    sess2 = await pop_session(session_id)
    if sess2:
        try:
            if sess2.trace_enabled:
                # trace 文件名包含 session_id，便于定位
                from fanqie_uploader_mcp.settings import TRACE_DIR
                from fanqie_uploader_mcp.utils import now_ts

                trace_path = TRACE_DIR / f"{now_ts()}_{session_id}.trace.zip"
                await sess2.context.tracing.stop(path=str(trace_path))
        except Exception:
            pass
        try:
            await sess2.context.close()
        except Exception:
            pass
        try:
            await sess2.browser.close()
        except Exception:
            pass

    return {
        "ok": True,
        "logged_in": True,
        "session_id": session_id,
        "account_key": sess.account_key,
        "storage_state_path": storage_state_path,
        "info": info,
        "message": "登录成功，已保存登录态。后续可调用 work_create_draft/chapter_upsert_draft。",
    }


@fanqie_mcp.tool()
async def fanqie_login_cancel(session_id: str) -> Dict[str, Any]:
    """取消/关闭登录会话。"""
    sess = await pop_session(session_id)
    if not sess:
        return {"ok": True, "session_id": session_id, "closed": False, "message": "session 不存在或已被回收"}
    try:
        await sess.context.close()
    except Exception:
        pass
    try:
        await sess.browser.close()
    except Exception:
        pass
    return {"ok": True, "session_id": session_id, "closed": True}


@fanqie_mcp.tool()
async def fanqie_debug_open(
    account_key: str,
    url: str,
    headless: Optional[bool] = None,
    trace: Optional[bool] = None,
) -> Dict[str, Any]:
    """用已保存登录态打开指定 URL 并截图（用于排障/选择器迭代）。"""
    browser = context = page = None
    try:
        browser, context, page = await open_authed_page(account_key=account_key, url=url, headless=headless, trace=trace)
        shot = await _screenshot(page, tag=f"debug_open_{account_key}", full_page=True)
        return {"ok": True, "url": page.url, **shot}
    except Exception as e:
        res = {"ok": False, "error": str(e), "account_key": account_key, "url": url}
        if page is not None:
            res.update(await _screenshot(page, tag="debug_open_failed", full_page=True))
        return res
    finally:
        if context is not None:
            try:
                if trace:
                    from fanqie_uploader_mcp.settings import TRACE_DIR
                    from fanqie_uploader_mcp.utils import now_ts

                    trace_path = TRACE_DIR / f"{now_ts()}_debug_open.trace.zip"
                    await context.tracing.stop(path=str(trace_path))
            except Exception:
                pass
        if context is not None:
            try:
                await context.close()
            except Exception:
                pass
        if browser is not None:
            try:
                await browser.close()
            except Exception:
                pass


@fanqie_mcp.tool()
async def fanqie_work_create_draft(
    account_key: str,
    title: str,
    category: str,
    intro: str,
    tags: Optional[List[str]] = None,
    headless: Optional[bool] = None,
    trace: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    创建作品（自动填：书名/分类/简介；失败返回截图便于调整）。
    """
    return await work_create_draft_ui(
        account_key=account_key,
        title=title,
        category=category,
        intro=intro,
        tags=tags,
        headless=headless,
        trace=trace,
    )


@fanqie_mcp.tool()
async def fanqie_chapter_upsert_draft(
    account_key: str,
    work_id: str,
    chapter_number: int,
    title: str,
    content: str,
    overwrite: bool = True,
    headless: Optional[bool] = None,
    trace: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    新增/覆盖章节，并保存草稿。

    overwrite=true 时，会尝试定位“第X章”进入编辑；若无法定位则退化为新建章节。
    """
    return await chapter_upsert_draft_ui(
        account_key=account_key,
        work_id=work_id,
        chapter_number=int(chapter_number),
        title=title,
        content=content,
        overwrite=bool(overwrite),
        headless=headless,
        trace=trace,
    )

