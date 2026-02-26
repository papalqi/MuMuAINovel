from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI

from fanqie_uploader_mcp.mcp_server import fanqie_mcp
from fanqie_uploader_mcp.playwright_mgr import stop_playwright
from fanqie_uploader_mcp.sessions import cleanup_expired_sessions
from fanqie_uploader_mcp.settings import SHOT_DIR, TRACE_DIR
from fastapi.staticfiles import StaticFiles


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时不需要预热 Playwright（延迟加载即可）
    # 但 FastMCP streamable-http 需要启动 session_manager.run()
    mcp_server_ctx = None
    try:
        session_manager = getattr(fanqie_mcp, "session_manager", None)
        if session_manager and hasattr(session_manager, "run"):
            mcp_server_ctx = session_manager.run()
            await mcp_server_ctx.__aenter__()
    except Exception:
        # 启动失败不阻塞应用启动（但 MCP 可能不可用）
        mcp_server_ctx = None

    yield

    # 先关闭 MCP session manager
    if mcp_server_ctx is not None:
        try:
            await mcp_server_ctx.__aexit__(None, None, None)
        except Exception:
            pass

    # 退出时清理
    try:
        await cleanup_expired_sessions()
    except Exception:
        pass
    try:
        await stop_playwright()
    except Exception:
        pass


app = FastAPI(title="fanqie-uploader-mcp", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health() -> Dict[str, Any]:
    cleaned = await cleanup_expired_sessions()
    return {"ok": True, "cleaned_sessions": cleaned}

# 便于调试：直接通过 HTTP 访问截图/trace 文件
app.mount("/shots", StaticFiles(directory=str(SHOT_DIR)), name="shots")
app.mount("/traces", StaticFiles(directory=str(TRACE_DIR)), name="traces")


# 挂载 MCP（与 MuMu 主服务一致：/mcp + 内部默认 /mcp -> 最终客户端连接 /mcp/mcp）
mcp_asgi_app = None
if hasattr(fanqie_mcp, "streamable_http_app"):
    mcp_asgi_app = fanqie_mcp.streamable_http_app()
elif hasattr(fanqie_mcp, "http_app"):
    mcp_asgi_app = fanqie_mcp.http_app()

if mcp_asgi_app is None:
    raise RuntimeError("FastMCP 未提供可挂载的 ASGI app（streamable_http_app/http_app）")

app.mount("/mcp", mcp_asgi_app)
