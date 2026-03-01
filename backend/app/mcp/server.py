"""MuMu MCP Server（对外暴露 MuMu UI 能力）

改造目标（V2）：
1. 不直接裸暴露 raw API 为主入口（避免“API 透传耦合”）
2. 提供动作化（action-based）工具，覆盖 UI 可操作能力
3. 保留 raw API 调用作为调试兜底，并默认关闭

核心模式：
- mumu_list_actions -> 发现可用动作
- mumu_call_action / mumu_call_action_sse -> 调用动作
- mumu_api_call / mumu_api_call_sse -> 仅 debug_proxy scope + 环境开关时可用
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import quote

import httpx
from fastapi.routing import APIRoute

from app.config import settings
from app.logger import get_logger
from app.user_manager import user_manager
from app.user_password import password_manager

logger = get_logger(__name__)


# ==================== MCP 可用性探测 ====================

try:
    from mcp.server.fastmcp import FastMCP  # type: ignore

    MCP_SERVER_AVAILABLE = True
except Exception as import_error:  # pragma: no cover
    FastMCP = None  # type: ignore
    MCP_SERVER_AVAILABLE = False
    logger.warning(f"⚠️ MCP Server 依赖不可用，已跳过对外 MCP 暴露: {import_error}")


# ==================== 环境开关 ====================

def _env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "on"}


def _raw_proxy_enabled() -> bool:
    # 默认关闭，防止直接暴露 raw API
    return _env_bool("MCP_RAW_API_ENABLED", False)


def _expose_admin_actions() -> bool:
    # 是否允许将 /api/admin 路由加入 action catalog
    return _env_bool("MCP_EXPOSE_ADMIN_ACTIONS", True)


# ==================== Token（无状态签名） ====================

TOKEN_PREFIX = "mumu"


def _get_token_secret() -> str:
    """获取 MCP token 签名密钥。"""
    secret = os.getenv("MCP_SERVER_SECRET")
    if secret:
        return secret

    if settings.LOCAL_AUTH_PASSWORD:
        return f"{settings.app_name}:{settings.LOCAL_AUTH_PASSWORD}:mcp"

    # 开发兜底（生产请务必设置 MCP_SERVER_SECRET）
    return f"{settings.app_name}:mcp-dev-secret"


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")


def _b64d(s: str) -> bytes:
    pad = "=" * ((4 - len(s) % 4) % 4)
    return base64.urlsafe_b64decode((s + pad).encode("utf-8"))


def issue_access_token(
    *,
    user_id: str,
    username: str,
    is_admin: bool,
    scopes: List[str],
    ttl_minutes: int = 120,
) -> str:
    """签发访问令牌。"""
    now = int(time.time())
    exp = now + max(1, int(ttl_minutes)) * 60
    payload = {
        "uid": user_id,
        "uname": username,
        "admin": bool(is_admin),
        "scopes": sorted(set(scopes)),
        "iat": now,
        "exp": exp,
    }
    payload_b64 = _b64e(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    sig = hmac.new(_get_token_secret().encode("utf-8"), payload_b64.encode("utf-8"), hashlib.sha256).digest()
    sig_b64 = _b64e(sig)
    return f"{TOKEN_PREFIX}.{payload_b64}.{sig_b64}"


def verify_access_token(token: str) -> Dict[str, Any]:
    """校验访问令牌并返回 payload。"""
    try:
        prefix, payload_b64, sig_b64 = token.split(".", 2)
    except ValueError as e:
        raise ValueError("token 格式无效") from e

    if prefix != TOKEN_PREFIX:
        raise ValueError("token 前缀无效")

    expected_sig = hmac.new(_get_token_secret().encode("utf-8"), payload_b64.encode("utf-8"), hashlib.sha256).digest()
    actual_sig = _b64d(sig_b64)
    if not hmac.compare_digest(expected_sig, actual_sig):
        raise ValueError("token 签名无效")

    payload = json.loads(_b64d(payload_b64).decode("utf-8"))
    exp = int(payload.get("exp", 0))
    if int(time.time()) >= exp:
        raise ValueError("token 已过期")

    if not payload.get("uid"):
        raise ValueError("token 缺少 uid")

    scopes = payload.get("scopes")
    if not isinstance(scopes, list):
        payload["scopes"] = []

    return payload


def _assert_scope(payload: Dict[str, Any], required: Set[str]) -> None:
    scopes = set(payload.get("scopes") or [])
    missing = [s for s in required if s not in scopes]
    if missing:
        raise PermissionError(f"缺少权限 scope: {', '.join(missing)}")


# ==================== 本地账号认证（复用现有账户体系） ====================

async def authenticate_local_user(username: str, password: str) -> Dict[str, Any]:
    """复用现有本地登录逻辑进行账号认证。"""
    if not settings.LOCAL_AUTH_ENABLED:
        raise ValueError("本地账户登录未启用")

    # 1) 优先匹配已存在用户（LinuxDO 或本地用户）
    all_users = await user_manager.get_all_users()
    target_user = None
    for user in all_users:
        pwd_username = await password_manager.get_username(user.user_id)
        if user.username == username or pwd_username == username:
            target_user = user
            break

    if target_user:
        has_pwd = await password_manager.has_password(target_user.user_id)
        if not has_pwd:
            raise ValueError("用户名或密码错误")
        ok = await password_manager.verify_password(target_user.user_id, password)
        if not ok:
            raise ValueError("用户名或密码错误")
        return target_user.dict()

    # 2) 回退到 .env 管理员账号
    if not settings.LOCAL_AUTH_USERNAME or not settings.LOCAL_AUTH_PASSWORD:
        raise ValueError("用户名或密码错误")

    if username != settings.LOCAL_AUTH_USERNAME or password != settings.LOCAL_AUTH_PASSWORD:
        raise ValueError("用户名或密码错误")

    user_id = f"local_{hashlib.md5(username.encode()).hexdigest()[:16]}"
    user = await user_manager.get_user(user_id)
    if not user:
        user = await user_manager.create_or_update_from_linuxdo(
            linuxdo_id=user_id,
            username=username,
            display_name=settings.LOCAL_AUTH_DISPLAY_NAME,
            avatar_url=None,
            trust_level=9,
        )
        await password_manager.set_password(user.user_id, username, password)
    else:
        ok = await password_manager.verify_password(user.user_id, password)
        if not ok:
            raise ValueError("用户名或密码错误")

    return user.dict()


# ==================== FastAPI 内部转发工具 ====================

_METHODS_ALLOWED = {"GET", "POST", "PUT", "PATCH", "DELETE"}
_PATH_PARAM_RE = re.compile(r"{([^}:]+)(?::[^}]+)?}")


def _normalize_path(path: str) -> str:
    if not path:
        raise ValueError("path 不能为空")
    return path if path.startswith("/") else f"/{path}"


def _validate_path_for_proxy(path: str) -> None:
    # 防止误调挂载应用和静态资源
    if path.startswith("/api/"):
        return
    if path in ("/health", "/health/db-sessions", "/openapi.json"):
        return
    raise ValueError("仅允许访问 /api/**、/health、/openapi.json")


def _safe_response_headers(headers: httpx.Headers) -> Dict[str, str]:
    keep = {"content-type", "content-disposition", "cache-control", "x-request-id"}
    out: Dict[str, str] = {}
    for k, v in headers.items():
        lk = k.lower()
        if lk in keep:
            out[lk] = v
    return out


def _decode_response_content(resp: httpx.Response, max_text_chars: int = 120_000) -> Tuple[str, Any, bool]:
    content_type = (resp.headers.get("content-type") or "").lower()

    if "application/json" in content_type:
        try:
            return "json", resp.json(), False
        except Exception:
            text = resp.text
            if len(text) > max_text_chars:
                return "text", text[:max_text_chars], True
            return "text", text, False

    if content_type.startswith("text/") or "application/xml" in content_type:
        text = resp.text
        if len(text) > max_text_chars:
            return "text", text[:max_text_chars], True
        return "text", text, False

    raw = resp.content
    b64 = base64.b64encode(raw).decode("ascii")
    truncated = False
    if len(b64) > max_text_chars:
        b64 = b64[:max_text_chars]
        truncated = True
    return "binary_base64", b64, truncated


async def _internal_api_request(
    *,
    user_id: str,
    method: str,
    path: str,
    query: Optional[Dict[str, Any]] = None,
    body: Optional[Any] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout_seconds: float = 120.0,
) -> Dict[str, Any]:
    method_u = method.upper().strip()
    if method_u not in _METHODS_ALLOWED:
        raise ValueError(f"method 不支持: {method_u}")

    p = _normalize_path(path)
    _validate_path_for_proxy(p)

    from app.main import app as fastapi_app  # 延迟导入，避免循环引用

    transport = httpx.ASGITransport(app=fastapi_app)
    started = time.perf_counter()

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://mumu.internal",
        timeout=timeout_seconds,
        follow_redirects=True,
    ) as client:
        req_kwargs: Dict[str, Any] = {
            "params": query or None,
            "cookies": {"user_id": user_id},
            "headers": headers or None,
        }
        if body is not None and method_u in {"POST", "PUT", "PATCH", "DELETE"}:
            if isinstance(body, (dict, list)):
                req_kwargs["json"] = body
            elif isinstance(body, str):
                req_kwargs["content"] = body.encode("utf-8")
            else:
                req_kwargs["content"] = json.dumps(body, ensure_ascii=False).encode("utf-8")

        resp = await client.request(method_u, p, **req_kwargs)

    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    data_type, data, truncated = _decode_response_content(resp)
    return {
        "ok": resp.is_success,
        "status_code": resp.status_code,
        "elapsed_ms": elapsed_ms,
        "method": method_u,
        "path": p,
        "query": query or {},
        "headers": _safe_response_headers(resp.headers),
        "data_type": data_type,
        "truncated": truncated,
        "data": data,
    }


async def _internal_sse_request(
    *,
    user_id: str,
    method: str,
    path: str,
    query: Optional[Dict[str, Any]] = None,
    body: Optional[Any] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout_seconds: float = 3600.0,
    max_events: int = 300,
    max_data_chars: int = 300_000,
) -> Dict[str, Any]:
    method_u = method.upper().strip()
    if method_u not in _METHODS_ALLOWED:
        raise ValueError(f"method 不支持: {method_u}")

    p = _normalize_path(path)
    _validate_path_for_proxy(p)

    from app.main import app as fastapi_app  # 延迟导入，避免循环引用

    transport = httpx.ASGITransport(app=fastapi_app)
    started = time.perf_counter()

    events: List[Dict[str, Any]] = []
    event_count = 0
    truncated = False
    total_data_chars = 0
    current_event: Dict[str, Any] = {"event": "message", "data_lines": []}

    def _flush_current_event():
        nonlocal event_count, total_data_chars, truncated
        if not current_event.get("data_lines"):
            return
        data_raw = "\n".join(current_event.get("data_lines", []))
        total_data_chars += len(data_raw)
        if total_data_chars > max_data_chars:
            truncated = True
            return

        parsed: Any = data_raw
        if data_raw:
            try:
                parsed = json.loads(data_raw)
            except Exception:
                parsed = data_raw

        events.append(
            {
                "event": current_event.get("event") or "message",
                "id": current_event.get("id"),
                "retry": current_event.get("retry"),
                "data": parsed,
            }
        )
        event_count += 1

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://mumu.internal",
        timeout=timeout_seconds,
        follow_redirects=True,
    ) as client:
        req_kwargs: Dict[str, Any] = {
            "params": query or None,
            "cookies": {"user_id": user_id},
            "headers": headers or None,
        }
        if body is not None and method_u in {"POST", "PUT", "PATCH", "DELETE"}:
            if isinstance(body, (dict, list)):
                req_kwargs["json"] = body
            elif isinstance(body, str):
                req_kwargs["content"] = body.encode("utf-8")
            else:
                req_kwargs["content"] = json.dumps(body, ensure_ascii=False).encode("utf-8")

        async with client.stream(method_u, p, **req_kwargs) as resp:
            status_code = resp.status_code
            resp_headers = _safe_response_headers(resp.headers)
            if status_code >= 400:
                text = await resp.aread()
                elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
                return {
                    "ok": False,
                    "status_code": status_code,
                    "elapsed_ms": elapsed_ms,
                    "method": method_u,
                    "path": p,
                    "headers": resp_headers,
                    "data_type": "text",
                    "truncated": len(text) > 120_000,
                    "data": text.decode("utf-8", errors="replace")[:120_000],
                    "event_count": 0,
                    "events": [],
                }

            async for line in resp.aiter_lines():
                if line is None:
                    continue
                if line == "":
                    _flush_current_event()
                    if truncated or event_count >= max_events:
                        truncated = True
                        break
                    current_event = {"event": "message", "data_lines": []}
                    continue
                if line.startswith(":"):
                    continue
                if line.startswith("event:"):
                    current_event["event"] = line[6:].strip()
                    continue
                if line.startswith("data:"):
                    current_event.setdefault("data_lines", []).append(line[5:].lstrip())
                    continue
                if line.startswith("id:"):
                    current_event["id"] = line[3:].strip()
                    continue
                if line.startswith("retry:"):
                    current_event["retry"] = line[6:].strip()
                    continue

            _flush_current_event()

    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    return {
        "ok": True,
        "status_code": 200,
        "elapsed_ms": elapsed_ms,
        "method": method_u,
        "path": p,
        "headers": {"content-type": "text/event-stream"},
        "data_type": "sse_events",
        "truncated": truncated,
        "event_count": event_count,
        "events": events,
        "last_event": events[-1] if events else None,
    }


# ==================== Action Catalog（核心） ====================

_ACTION_CATALOG_CACHE: Optional[Dict[str, Dict[str, Any]]] = None
_ALIAS_CATALOG_CACHE: Optional[Dict[str, Dict[str, Any]]] = None


def _is_sse_route(method: str, path: str, summary: Optional[str]) -> bool:
    if method.upper() != "POST":
        return False
    s = (summary or "").lower()
    p = path.lower()
    return ("stream" in p) or ("sse" in s) or ("流式" in (summary or ""))


def _route_allowlisted(path: str) -> bool:
    if path in ("/health", "/health/db-sessions", "/openapi.json"):
        return True
    if not path.startswith("/api/"):
        return False
    if path.startswith("/api/auth/"):
        return False
    if path.startswith("/api/mcp/"):
        # 避免对外 MCP 再嵌套调用自身管理接口
        return False
    if path.startswith("/api/admin/") and not _expose_admin_actions():
        return False
    return True


def _action_slug(method: str, path: str) -> str:
    # /api/chapters/{chapter_id}/analysis -> api_chapters_chapter_id_analysis
    normalized = _PATH_PARAM_RE.sub(lambda m: m.group(1), path.strip("/"))
    normalized = normalized.replace("/", "_").replace("-", "_").replace(".", "_")
    normalized = re.sub(r"[^a-zA-Z0-9_]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    digest = hashlib.md5(f"{method}:{path}".encode("utf-8")).hexdigest()[:6]
    return f"{method.lower()}__{normalized}__{digest}" if normalized else f"{method.lower()}__root__{digest}"


def _required_scopes_for_action(meta: Dict[str, Any]) -> Set[str]:
    req: Set[str] = set()
    method = (meta.get("method") or "GET").upper()
    path = meta.get("path") or ""

    if method == "GET":
        req.add("read")
    else:
        req.add("write")

    if path.startswith("/api/admin/"):
        req.add("admin")
    return req


def _build_action_catalog() -> Dict[str, Dict[str, Any]]:
    from app.main import app as fastapi_app  # 延迟导入，避免循环引用

    catalog: Dict[str, Dict[str, Any]] = {}
    for route in fastapi_app.routes:
        if not isinstance(route, APIRoute):
            continue
        path = route.path or ""
        if not _route_allowlisted(path):
            continue

        methods = sorted([m for m in (route.methods or set()) if m not in {"HEAD", "OPTIONS"}])
        for method in methods:
            action = _action_slug(method, path)
            param_names = _PATH_PARAM_RE.findall(path)
            item = {
                "action": action,
                "method": method,
                "path": path,
                "path_params": param_names,
                "summary": route.summary,
                "name": route.name,
                "tags": route.tags or [],
                "is_sse": _is_sse_route(method, path, route.summary),
            }
            item["required_scopes"] = sorted(_required_scopes_for_action(item))
            catalog[action] = item

    return catalog


def _get_action_catalog(force_refresh: bool = False) -> Dict[str, Dict[str, Any]]:
    global _ACTION_CATALOG_CACHE
    if _ACTION_CATALOG_CACHE is None or force_refresh:
        _ACTION_CATALOG_CACHE = _build_action_catalog()
    return _ACTION_CATALOG_CACHE


def _find_action_by_method_path(
    catalog: Dict[str, Dict[str, Any]],
    *,
    method: str,
    path: str,
) -> Optional[str]:
    m = method.upper()
    for action, meta in catalog.items():
        if (meta.get("method") or "").upper() == m and meta.get("path") == path:
            return action
    return None


_ALIAS_CANDIDATES: List[Dict[str, str]] = [
    # 项目
    {"alias": "projects_list", "method": "GET", "path": "/api/projects", "description": "获取项目列表"},
    {"alias": "project_create", "method": "POST", "path": "/api/projects", "description": "创建项目"},
    {"alias": "project_get", "method": "GET", "path": "/api/projects/{project_id}", "description": "获取项目详情"},
    {"alias": "project_update", "method": "PUT", "path": "/api/projects/{project_id}", "description": "更新项目"},
    {"alias": "project_delete", "method": "DELETE", "path": "/api/projects/{project_id}", "description": "删除项目"},
    # 向导
    {"alias": "wizard_world_building", "method": "POST", "path": "/api/wizard-stream/world-building", "description": "流式生成世界观"},
    # 大纲
    {"alias": "outlines_list", "method": "GET", "path": "/api/outlines/project/{project_id}", "description": "获取项目大纲"},
    {"alias": "outline_generate_stream", "method": "POST", "path": "/api/outlines/generate-stream", "description": "流式生成/续写大纲"},
    {"alias": "outline_postcheck_stream", "method": "POST", "path": "/api/outlines/postcheck-stream", "description": "大纲后处理补全（角色/组织，流式）"},
    {"alias": "outline_batch_expand_stream", "method": "POST", "path": "/api/outlines/batch-expand-stream", "description": "批量展开大纲（流式）"},
    # 章节
    {"alias": "chapters_list", "method": "GET", "path": "/api/chapters/project/{project_id}", "description": "获取项目章节列表"},
    {"alias": "chapter_create", "method": "POST", "path": "/api/chapters", "description": "创建章节"},
    {"alias": "chapter_get", "method": "GET", "path": "/api/chapters/{chapter_id}", "description": "获取章节详情"},
    {"alias": "chapter_update", "method": "PUT", "path": "/api/chapters/{chapter_id}", "description": "更新章节"},
    {"alias": "chapter_generate_stream", "method": "POST", "path": "/api/chapters/{chapter_id}/generate-stream", "description": "流式创作章节"},
    {"alias": "chapter_analysis_status", "method": "GET", "path": "/api/chapters/{chapter_id}/analysis/status", "description": "查询章节分析状态"},
    {"alias": "chapter_analysis_get", "method": "GET", "path": "/api/chapters/{chapter_id}/analysis", "description": "获取章节分析结果"},
    {"alias": "chapter_analyze", "method": "POST", "path": "/api/chapters/{chapter_id}/analyze", "description": "手动触发章节分析"},
    {"alias": "chapter_batch_generate_start", "method": "POST", "path": "/api/chapters/project/{project_id}/batch-generate", "description": "启动批量生成"},
    {"alias": "chapter_batch_generate_status", "method": "GET", "path": "/api/chapters/batch-generate/{batch_id}/status", "description": "查询批量生成状态"},
    # 角色 / 设定
    {"alias": "characters_list", "method": "GET", "path": "/api/characters/project/{project_id}", "description": "获取项目角色"},
    {"alias": "careers_list", "method": "GET", "path": "/api/careers", "description": "获取职业列表"},
    {"alias": "styles_list", "method": "GET", "path": "/api/writing-styles", "description": "获取写作风格列表"},
    # 记忆 / 伏笔
    {"alias": "memories_stats", "method": "GET", "path": "/api/memories/projects/{project_id}/stats", "description": "获取记忆统计"},
    {"alias": "foreshadow_stats", "method": "GET", "path": "/api/foreshadows/projects/{project_id}/stats", "description": "获取伏笔统计"},
]


def _build_alias_catalog() -> Dict[str, Dict[str, Any]]:
    catalog = _get_action_catalog(force_refresh=False)
    aliases: Dict[str, Dict[str, Any]] = {}
    for c in _ALIAS_CANDIDATES:
        action = _find_action_by_method_path(catalog, method=c["method"], path=c["path"])
        if not action:
            continue
        meta = catalog[action]
        aliases[c["alias"]] = {
            "alias": c["alias"],
            "description": c["description"],
            "method": meta["method"],
            "path": meta["path"],
            "path_params": meta.get("path_params") or [],
            "action": action,
            "is_sse": bool(meta.get("is_sse")),
            "required_scopes": meta.get("required_scopes") or [],
            "tags": meta.get("tags") or [],
        }
    return aliases


def _get_alias_catalog(force_refresh: bool = False) -> Dict[str, Dict[str, Any]]:
    global _ALIAS_CATALOG_CACHE
    if _ALIAS_CATALOG_CACHE is None or force_refresh:
        _ALIAS_CATALOG_CACHE = _build_alias_catalog()
    return _ALIAS_CATALOG_CACHE


def _resolve_path(path_template: str, path_params: Optional[Dict[str, Any]]) -> str:
    params = path_params or {}
    missing: List[str] = []
    resolved = path_template

    for name in _PATH_PARAM_RE.findall(path_template):
        if name not in params:
            missing.append(name)
            continue
        value = quote(str(params[name]), safe="")
        resolved = re.sub(r"{%s(?::[^}]+)?}" % re.escape(name), value, resolved)

    if missing:
        raise ValueError(f"缺少路径参数: {', '.join(missing)}")
    return resolved


def _filter_actions_for_payload(payload: Dict[str, Any], catalog: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    scopes = set(payload.get("scopes") or [])
    rows: List[Dict[str, Any]] = []
    for _, meta in catalog.items():
        req = set(meta.get("required_scopes") or [])
        if req.issubset(scopes):
            rows.append(meta)
    rows.sort(key=lambda x: (x.get("path") or "", x.get("method") or ""))
    return rows


def _filter_aliases_for_payload(payload: Dict[str, Any], aliases: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    scopes = set(payload.get("scopes") or [])
    rows: List[Dict[str, Any]] = []
    for _, meta in aliases.items():
        req = set(meta.get("required_scopes") or [])
        if req.issubset(scopes):
            rows.append(meta)
    rows.sort(key=lambda x: (x.get("alias") or ""))
    return rows


def _ensure_raw_proxy_allowed(payload: Dict[str, Any]) -> None:
    if not _raw_proxy_enabled():
        raise PermissionError("raw API proxy 默认关闭（设置 MCP_RAW_API_ENABLED=true 可启用）")
    _assert_scope(payload, {"debug_proxy"})


async def _execute_action_call(
    *,
    payload: Dict[str, Any],
    action: str,
    path_params: Optional[Dict[str, Any]] = None,
    query: Optional[Dict[str, Any]] = None,
    body: Optional[Any] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout_seconds: float = 120.0,
    as_sse: bool = False,
    max_events: int = 300,
    max_data_chars: int = 300000,
) -> Dict[str, Any]:
    catalog = _get_action_catalog(force_refresh=False)
    meta = catalog.get(action)
    if not meta:
        return {"ok": False, "error": f"未知 action: {action}"}

    req = set(meta.get("required_scopes") or [])
    _assert_scope(payload, req)

    is_sse = bool(meta.get("is_sse"))
    if as_sse and not is_sse:
        return {"ok": False, "error": "该 action 非流式接口，请使用 mumu_call_action"}
    if (not as_sse) and is_sse:
        return {"ok": False, "error": "该 action 为流式接口，请使用 mumu_call_action_sse"}

    resolved_path = _resolve_path(meta["path"], path_params)
    if as_sse:
        return await _internal_sse_request(
            user_id=payload["uid"],
            method=meta["method"],
            path=resolved_path,
            query=query,
            body=body,
            headers=headers,
            timeout_seconds=timeout_seconds,
            max_events=max_events,
            max_data_chars=max_data_chars,
        )
    return await _internal_api_request(
        user_id=payload["uid"],
        method=meta["method"],
        path=resolved_path,
        query=query,
        body=body,
        headers=headers,
        timeout_seconds=timeout_seconds,
    )


# ==================== MCP Server 定义 ====================

mumu_mcp_server = None

if MCP_SERVER_AVAILABLE:
    _mcp_instructions = (
        "MuMu MCP：优先使用动作化工具 mumu_list_actions / mumu_call_action / mumu_call_action_sse。"
        "raw API 调用默认关闭，仅调试场景开启。"
    )
    try:
        mumu_mcp_server = FastMCP(
            name="MuMuAINovel MCP",
            instructions=_mcp_instructions,
            stateless_http=True,
            json_response=True,
        )
    except TypeError:
        try:
            mumu_mcp_server = FastMCP(name="MuMuAINovel MCP", instructions=_mcp_instructions)
        except TypeError:
            mumu_mcp_server = FastMCP("MuMuAINovel MCP")

    @mumu_mcp_server.tool()
    async def mumu_login_local(
        username: str,
        password: str,
        ttl_minutes: int = 120,
        request_debug_proxy: bool = False,
    ) -> Dict[str, Any]:
        """本地账号登录，返回 access_token。

        默认签发 read/write（管理员附带 admin）。
        若 request_debug_proxy=true 且 MCP_RAW_API_ENABLED=true 且用户为管理员，会附带 debug_proxy。
        """
        try:
            user = await authenticate_local_user(username=username, password=password)
            is_admin = bool(user.get("is_admin", False))
            scopes = ["read", "write"]
            if is_admin:
                scopes.append("admin")
            if request_debug_proxy and is_admin and _raw_proxy_enabled():
                scopes.append("debug_proxy")

            token = issue_access_token(
                user_id=user["user_id"],
                username=user.get("username") or username,
                is_admin=is_admin,
                scopes=scopes,
                ttl_minutes=ttl_minutes,
            )
            return {
                "success": True,
                "access_token": token,
                "expires_in_seconds": max(60, int(ttl_minutes) * 60),
                "scopes": sorted(set(scopes)),
                "raw_proxy_enabled": _raw_proxy_enabled(),
                "user": {
                    "user_id": user.get("user_id"),
                    "username": user.get("username"),
                    "display_name": user.get("display_name"),
                    "is_admin": is_admin,
                },
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @mumu_mcp_server.tool()
    async def mumu_verify_token(access_token: str) -> Dict[str, Any]:
        """验证 access_token。"""
        try:
            payload = verify_access_token(access_token)
            now = int(time.time())
            return {
                "valid": True,
                "user_id": payload.get("uid"),
                "username": payload.get("uname"),
                "is_admin": bool(payload.get("admin", False)),
                "scopes": payload.get("scopes", []),
                "expires_at": payload.get("exp"),
                "expires_in_seconds": max(0, int(payload.get("exp", 0)) - now),
            }
        except Exception as e:
            return {"valid": False, "error": str(e)}

    @mumu_mcp_server.tool()
    async def mumu_list_actions(
        access_token: str,
        force_refresh: bool = False,
        include_sse_only: bool = False,
        limit: int = 500,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """列出当前 token 可调用的 MuMu 动作（非 raw API）。"""
        try:
            payload = verify_access_token(access_token)
            catalog = _get_action_catalog(force_refresh=force_refresh)
            rows = _filter_actions_for_payload(payload, catalog)
            if include_sse_only:
                rows = [x for x in rows if x.get("is_sse")]

            total = len(rows)
            lim = max(1, min(int(limit), 2000))
            off = max(0, int(offset))
            page = rows[off : off + lim]

            return {
                "success": True,
                "total": total,
                "offset": off,
                "limit": lim,
                "items": page,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @mumu_mcp_server.tool()
    async def mumu_list_aliases(
        access_token: str,
        force_refresh: bool = False,
        include_sse_only: bool = False,
    ) -> Dict[str, Any]:
        """列出高频动作别名（更适合 Agent 直接调用）。"""
        try:
            payload = verify_access_token(access_token)
            _assert_scope(payload, {"read"})
            if force_refresh:
                _get_action_catalog(force_refresh=True)
            aliases = _get_alias_catalog(force_refresh=force_refresh)
            rows = _filter_aliases_for_payload(payload, aliases)
            if include_sse_only:
                rows = [x for x in rows if x.get("is_sse")]
            return {
                "success": True,
                "count": len(rows),
                "items": rows,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @mumu_mcp_server.tool()
    async def mumu_call_action(
        access_token: str,
        action: str,
        path_params: Optional[Dict[str, Any]] = None,
        query: Optional[Dict[str, Any]] = None,
        body: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout_seconds: float = 120.0,
    ) -> Dict[str, Any]:
        """调用一个非流式 MuMu 动作。"""
        try:
            payload = verify_access_token(access_token)
            return await _execute_action_call(
                payload=payload,
                action=action,
                path_params=path_params,
                query=query,
                body=body,
                headers=headers,
                timeout_seconds=timeout_seconds,
                as_sse=False,
            )
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @mumu_mcp_server.tool()
    async def mumu_call_action_sse(
        access_token: str,
        action: str,
        path_params: Optional[Dict[str, Any]] = None,
        query: Optional[Dict[str, Any]] = None,
        body: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout_seconds: float = 3600.0,
        max_events: int = 300,
        max_data_chars: int = 300000,
    ) -> Dict[str, Any]:
        """调用一个流式（SSE）MuMu 动作。"""
        try:
            payload = verify_access_token(access_token)
            return await _execute_action_call(
                payload=payload,
                action=action,
                path_params=path_params,
                query=query,
                body=body,
                headers=headers,
                timeout_seconds=timeout_seconds,
                as_sse=True,
                max_events=max_events,
                max_data_chars=max_data_chars,
            )
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @mumu_mcp_server.tool()
    async def mumu_call_alias(
        access_token: str,
        alias: str,
        path_params: Optional[Dict[str, Any]] = None,
        query: Optional[Dict[str, Any]] = None,
        body: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout_seconds: float = 120.0,
    ) -> Dict[str, Any]:
        """调用高频别名（非流式）。"""
        try:
            payload = verify_access_token(access_token)
            aliases = _get_alias_catalog(force_refresh=False)
            meta = aliases.get(alias)
            if not meta:
                return {"ok": False, "error": f"未知 alias: {alias}"}
            return await _execute_action_call(
                payload=payload,
                action=meta["action"],
                path_params=path_params,
                query=query,
                body=body,
                headers=headers,
                timeout_seconds=timeout_seconds,
                as_sse=False,
            )
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @mumu_mcp_server.tool()
    async def mumu_call_alias_sse(
        access_token: str,
        alias: str,
        path_params: Optional[Dict[str, Any]] = None,
        query: Optional[Dict[str, Any]] = None,
        body: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout_seconds: float = 3600.0,
        max_events: int = 300,
        max_data_chars: int = 300000,
    ) -> Dict[str, Any]:
        """调用高频别名（流式）。"""
        try:
            payload = verify_access_token(access_token)
            aliases = _get_alias_catalog(force_refresh=False)
            meta = aliases.get(alias)
            if not meta:
                return {"ok": False, "error": f"未知 alias: {alias}"}
            return await _execute_action_call(
                payload=payload,
                action=meta["action"],
                path_params=path_params,
                query=query,
                body=body,
                headers=headers,
                timeout_seconds=timeout_seconds,
                as_sse=True,
                max_events=max_events,
                max_data_chars=max_data_chars,
            )
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @mumu_mcp_server.tool()
    async def mumu_list_ui_routes(access_token: Optional[str] = None) -> Dict[str, Any]:
        """兼容工具：列出动作对应的 method/path（建议改用 mumu_list_actions）。"""
        try:
            if access_token:
                payload = verify_access_token(access_token)
                catalog = _get_action_catalog(force_refresh=False)
                rows = _filter_actions_for_payload(payload, catalog)
            else:
                # 未提供 token 时仅列基础列表（不做权限过滤）
                rows = list(_get_action_catalog(force_refresh=False).values())
                rows.sort(key=lambda x: (x.get("path") or "", x.get("method") or ""))

            return {
                "success": True,
                "route_count": len(rows),
                "routes": [
                    {
                        "action": x.get("action"),
                        "path": x.get("path"),
                        "methods": [x.get("method")],
                        "summary": x.get("summary"),
                        "tags": x.get("tags") or [],
                        "is_sse": bool(x.get("is_sse")),
                    }
                    for x in rows
                ],
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @mumu_mcp_server.tool()
    async def mumu_api_call(
        access_token: str,
        method: str,
        path: str,
        query: Optional[Dict[str, Any]] = None,
        body: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout_seconds: float = 120.0,
    ) -> Dict[str, Any]:
        """调试兜底：raw API 调用（默认关闭）。"""
        try:
            payload = verify_access_token(access_token)
            _ensure_raw_proxy_allowed(payload)
            return await _internal_api_request(
                user_id=payload["uid"],
                method=method,
                path=path,
                query=query,
                body=body,
                headers=headers,
                timeout_seconds=timeout_seconds,
            )
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @mumu_mcp_server.tool()
    async def mumu_api_call_sse(
        access_token: str,
        method: str,
        path: str,
        query: Optional[Dict[str, Any]] = None,
        body: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout_seconds: float = 3600.0,
        max_events: int = 300,
        max_data_chars: int = 300000,
    ) -> Dict[str, Any]:
        """调试兜底：raw SSE API 调用（默认关闭）。"""
        try:
            payload = verify_access_token(access_token)
            _ensure_raw_proxy_allowed(payload)
            return await _internal_sse_request(
                user_id=payload["uid"],
                method=method,
                path=path,
                query=query,
                body=body,
                headers=headers,
                timeout_seconds=timeout_seconds,
                max_events=max_events,
                max_data_chars=max_data_chars,
            )
        except Exception as e:
            return {"ok": False, "error": str(e)}
