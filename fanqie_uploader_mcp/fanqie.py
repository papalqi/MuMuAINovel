from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from playwright.async_api import Browser, BrowserContext, Error, Page

from fanqie_uploader_mcp.playwright_mgr import get_playwright
from fanqie_uploader_mcp.settings import (
    DEFAULT_HEADLESS,
    DEFAULT_TRACE,
    FANQIE_BASE_URL,
    FANQIE_LOGIN_URL,
    PW_ACTION_TIMEOUT_MS,
    PW_DEVICE_SCALE_FACTOR,
    PW_NAV_TIMEOUT_MS,
    PW_VIEWPORT_HEIGHT,
    PW_VIEWPORT_WIDTH,
    SHOT_DIR,
    STATE_DIR,
    TRACE_DIR,
)
from fanqie_uploader_mcp.utils import (
    b64encode_bytes,
    now_ts,
    parse_chapter_id_from_url,
    parse_work_id_from_url,
    safe_filename,
    write_bytes,
)


def state_path(account_key: str) -> Path:
    key = safe_filename(account_key or "default", max_len=60)
    return STATE_DIR / f"{key}.storage_state.json"


async def _launch_browser(headless: Optional[bool] = None) -> Browser:
    pw = await get_playwright()
    return await pw.chromium.launch(
        headless=DEFAULT_HEADLESS if headless is None else bool(headless),
        args=[
            "--disable-dev-shm-usage",
            "--no-sandbox",
            "--disable-blink-features=AutomationControlled",
        ],
    )


async def _new_context(browser: Browser, *, storage_state_path: Optional[Path] = None) -> BrowserContext:
    ctx = await browser.new_context(
        viewport={"width": PW_VIEWPORT_WIDTH, "height": PW_VIEWPORT_HEIGHT},
        device_scale_factor=PW_DEVICE_SCALE_FACTOR,
        locale="zh-CN",
        storage_state=str(storage_state_path) if storage_state_path and storage_state_path.exists() else None,
    )
    ctx.set_default_navigation_timeout(PW_NAV_TIMEOUT_MS)
    ctx.set_default_timeout(PW_ACTION_TIMEOUT_MS)
    return ctx


async def _screenshot(page: Page, *, tag: str, full_page: bool = True) -> Dict[str, Any]:
    try:
        data = await page.screenshot(full_page=full_page, type="png")
        fn = f"{now_ts()}_{safe_filename(tag)}.png"
        path = SHOT_DIR / fn
        write_bytes(path, data)
        return {
            "ok": True,
            "screenshot_file": fn,
            "screenshot_path": str(path),
            "screenshot_url_path": f"/shots/{fn}",
            "screenshot_base64": b64encode_bytes(data),
            "screenshot_mime": "image/png",
        }
    except Exception as e:
        return {"ok": False, "error": f"screenshot_failed: {e}"}


async def check_logged_in(context: BrowserContext) -> Tuple[bool, Dict[str, Any]]:
    """
    使用稳定接口判断登录状态（比 DOM 文案更可靠）
    - 未登录：{"code": -1, "message": "请先登录"}
    - 已登录：通常 code==0 且 data 非空（具体结构以实际为准）
    """
    try:
        resp = await context.request.get(f"{FANQIE_BASE_URL}/api/user/info/v2", timeout=15_000)
        data = await resp.json()
        code = data.get("code")
        if code == 0:
            return True, data
        return False, data
    except Exception as e:
        return False, {"error": str(e)}


async def login_open_page(page: Page) -> None:
    await page.goto(FANQIE_LOGIN_URL, wait_until="domcontentloaded")
    # 给站点一点时间渲染
    await page.wait_for_timeout(1200)

    # 若存在“扫码登录”tab，切换过去（更适合自动化拿二维码）
    try:
        tab = page.get_by_text("扫码登录")
        if await tab.count():
            await tab.first.click()
            await page.wait_for_timeout(1200)
    except Exception:
        pass


async def save_storage_state(context: BrowserContext, account_key: str) -> str:
    p = state_path(account_key)
    p.parent.mkdir(parents=True, exist_ok=True)
    await context.storage_state(path=str(p))
    return str(p)


async def open_authed_page(
    *,
    account_key: str,
    url: str,
    headless: Optional[bool] = None,
    trace: Optional[bool] = None,
) -> Tuple[Browser, BrowserContext, Page]:
    """
    使用已保存的登录态打开页面。
    调用方负责 close。
    """
    st = state_path(account_key)
    if not st.exists():
        raise RuntimeError(f"account_key={account_key} 未找到登录态文件，请先 login_start/login_poll: {st}")

    browser = await _launch_browser(headless=headless)
    context = await _new_context(browser, storage_state_path=st)

    # 再次确认是否登录有效
    ok, _ = await check_logged_in(context)
    if not ok:
        await context.close()
        await browser.close()
        raise RuntimeError(f"account_key={account_key} 登录态已失效，请重新扫码登录")

    if trace is None:
        trace = DEFAULT_TRACE
    if trace:
        await context.tracing.start(screenshots=True, snapshots=True, sources=False)

    page = await context.new_page()
    await page.goto(url, wait_until="domcontentloaded")
    await page.wait_for_timeout(800)
    return browser, context, page


# ========================= 作品创建（UI 尝试） =========================

async def work_create_draft_ui(
    *,
    account_key: str,
    title: str,
    category: str,
    intro: str,
    tags: Optional[List[str]] = None,
    headless: Optional[bool] = None,
    trace: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    通过 UI 自动化创建作品。
    说明：该流程对站点 UI 变化敏感；失败会返回截图，便于迭代。
    """
    tags = tags or []
    browser = context = page = None
    trace_path = None
    try:
        browser, context, page = await open_authed_page(
            account_key=account_key,
            url=f"{FANQIE_BASE_URL}/writer/zone/create",
            headless=headless,
            trace=trace,
        )

        # 1) 填书名/标题
        # 兼容：input / textarea / 自定义组件
        title_locators = [
            page.get_by_placeholder(re.compile(r"(书名|作品名|作品名称|小说名|标题)")),
            page.get_by_label(re.compile(r"(书名|作品名|作品名称|小说名|标题)")),
        ]
        filled = False
        for loc in title_locators:
            if await loc.count():
                await loc.first.click()
                await loc.first.fill(title)
                filled = True
                break
        if not filled:
            raise RuntimeError("未找到书名/标题输入框（placeholder/label 未命中）")

        # 2) 选择分类（尽力：找 combobox / 下拉）
        # 如果找不到，先不阻塞（平台可能允许默认值）
        try:
            combo = page.get_by_role("combobox", name=re.compile(r"(题材|分类|类型)"))
            if await combo.count():
                await combo.first.click()
                opt = page.get_by_role("option", name=re.compile(re.escape(category)))
                if await opt.count():
                    await opt.first.click()
        except Exception:
            pass

        # 3) 简介
        intro_locators = [
            page.get_by_placeholder(re.compile(r"(简介|内容简介)")),
            page.get_by_label(re.compile(r"(简介|内容简介)")),
        ]
        intro_ok = False
        for loc in intro_locators:
            if await loc.count():
                await loc.first.click()
                await loc.first.fill(intro)
                intro_ok = True
                break
        if not intro_ok:
            # 有些页面简介是可选或在下一步；这里不强制
            pass

        # 4) 点击创建/保存（尽力匹配）
        btn_candidates = [
            page.get_by_role("button", name=re.compile(r"^(创建|保存|下一步)")),
            page.get_by_text(re.compile(r"^(创建|保存|下一步)$")),
        ]
        clicked = False
        for b in btn_candidates:
            if await b.count():
                await b.first.click()
                clicked = True
                break
        if not clicked:
            raise RuntimeError("未找到创建/保存按钮")

        # 等待跳转/渲染
        await page.wait_for_timeout(1500)

        work_id = parse_work_id_from_url(page.url) or parse_work_id_from_url(await page.evaluate("location.href"))
        if not work_id:
            # 尝试从页面上抓取（弱匹配）
            html = await page.content()
            m = re.search(r"article/(\\d{10,})", html)
            if m:
                work_id = m.group(1)

        shot = await _screenshot(page, tag=f"work_create_{title}", full_page=True)

        return {
            "ok": True,
            "work_id": work_id,
            "work_url": page.url,
            "message": "创建流程已提交（若 work_id 为空，说明未成功解析跳转 URL，需要根据截图调整流程）",
            "debug": {
                "category": category,
                "tags": tags,
                "title_filled": True,
                "intro_filled": intro_ok,
                "clicked": clicked,
            },
            **({"screenshot_path": shot.get("screenshot_path"), "screenshot_base64": shot.get("screenshot_base64"), "screenshot_mime": shot.get("screenshot_mime")} if shot.get("ok") else {}),
        }

    except Exception as e:
        err = {"ok": False, "error": str(e)}
        if page is not None:
            err.update(await _screenshot(page, tag="work_create_failed", full_page=True))
        return err
    finally:
        if context is not None:
            try:
                if trace:
                    trace_path = TRACE_DIR / f"{now_ts()}_work_create.trace.zip"
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


# ========================= 章节 upsert（UI 尝试） =========================

async def chapter_upsert_draft_ui(
    *,
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
    新增/覆盖章节并保存草稿（UI 自动化）。

    注意：
    - “覆盖”非常依赖平台是否允许编辑已发布章节；本实现仅做 UI 层尝试。
    - 真正可靠的覆盖，需要持久化 fanqie_chapter_id（后续接入 MuMu 时会做映射表）。
    """
    browser = context = page = None
    try:
        browser, context, page = await open_authed_page(
            account_key=account_key,
            url=f"{FANQIE_BASE_URL}/writer/zone/article/{work_id}",
            headless=headless,
            trace=trace,
        )

        # 1) 进入“编辑/新建章节”
        entered_editor = False

        if overwrite:
            # 尽力：按“第X章”定位
            try:
                row = page.get_by_text(re.compile(fr"第\\s*{chapter_number}\\s*章"))
                if await row.count():
                    await row.first.click()
                    # 找编辑按钮
                    edit_btn = page.get_by_role("button", name=re.compile(r"(编辑|修改|继续写)"))
                    if await edit_btn.count():
                        await edit_btn.first.click()
                        entered_editor = True
            except Exception:
                pass

        if not entered_editor:
            # 新建章节按钮
            new_btn = page.get_by_role("button", name=re.compile(r"(新建|新增|创建).*(章节|章)"))
            if not await new_btn.count():
                new_btn = page.get_by_text(re.compile(r"(新建|新增|创建).*(章节|章)"))
            if await new_btn.count():
                await new_btn.first.click()
                entered_editor = True

        if not entered_editor:
            raise RuntimeError("未能进入章节编辑器（未找到 新建章节/编辑 按钮）")

        await page.wait_for_timeout(1200)

        # 2) 填标题
        title_locators = [
            page.get_by_placeholder(re.compile(r"(章节标题|标题)")),
            page.get_by_label(re.compile(r"(章节标题|标题)")),
        ]
        title_ok = False
        for loc in title_locators:
            if await loc.count():
                await loc.first.click()
                await loc.first.fill(title)
                title_ok = True
                break
        if not title_ok:
            # 有些编辑器标题可能是普通 input，无 placeholder；兜底：找第一个 input
            inputs = page.locator("input")
            if await inputs.count():
                await inputs.first.click()
                await inputs.first.fill(title)
                title_ok = True

        # 3) 填正文
        body_ok = False
        # 优先 textarea
        ta = page.locator("textarea")
        if await ta.count():
            await ta.first.click()
            await ta.first.fill(content)
            body_ok = True
        else:
            # contenteditable
            ed = page.locator("[contenteditable='true']")
            if await ed.count():
                await ed.first.click()
                # Playwright 的 fill 对 contenteditable 也可用（比逐字 type 更快）
                await ed.first.fill(content)
                body_ok = True
            else:
                # 兜底：找 role=textbox 的最大块
                tb = page.get_by_role("textbox")
                if await tb.count():
                    await tb.first.click()
                    await tb.first.fill(content)
                    body_ok = True

        if not body_ok:
            raise RuntimeError("未找到正文输入区域（textarea/contenteditable/textbox 均未命中）")

        # 4) 保存草稿
        save_btn = page.get_by_role("button", name=re.compile(r"(保存草稿|保存)"))
        if not await save_btn.count():
            save_btn = page.get_by_text(re.compile(r"(保存草稿|保存)"))
        if not await save_btn.count():
            raise RuntimeError("未找到“保存草稿/保存”按钮")

        await save_btn.first.click()

        await page.wait_for_timeout(1500)

        chapter_id = parse_chapter_id_from_url(page.url)
        shot = await _screenshot(page, tag=f"chapter_upsert_{work_id}_{chapter_number}", full_page=True)
        return {
            "ok": True,
            "work_id": work_id,
            "chapter_number": chapter_number,
            "chapter_id": chapter_id,
            "chapter_url": page.url,
            "debug": {"title_ok": title_ok, "body_ok": body_ok, "overwrite": overwrite},
            **({"screenshot_path": shot.get("screenshot_path"), "screenshot_base64": shot.get("screenshot_base64"), "screenshot_mime": shot.get("screenshot_mime")} if shot.get("ok") else {}),
        }

    except Exception as e:
        err = {"ok": False, "error": str(e), "work_id": work_id, "chapter_number": chapter_number}
        if page is not None:
            err.update(await _screenshot(page, tag="chapter_upsert_failed", full_page=True))
        return err
    finally:
        if context is not None:
            try:
                if trace:
                    trace_path = TRACE_DIR / f"{now_ts()}_chapter_upsert.trace.zip"
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
