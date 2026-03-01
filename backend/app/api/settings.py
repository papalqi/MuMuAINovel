"""
设置管理 API
"""
from fastapi import APIRouter, HTTPException, Request, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Dict, Any, List, Optional
from pathlib import Path
from pydantic import BaseModel
from datetime import datetime
import httpx
import json
import time

from app.database import get_db
from app.models.settings import Settings
from app.schemas.settings import (
    SettingsCreate, SettingsUpdate, SettingsResponse,
    APIKeyPreset, APIKeyPresetConfig, PresetCreateRequest,
    PresetUpdateRequest, PresetResponse, PresetListResponse
)
from app.user_manager import User
from app.logger import get_logger
from app.config import settings as app_settings, PROJECT_ROOT
from app.services.ai_service import AIService, create_user_ai_service, create_user_ai_service_with_mcp

logger = get_logger(__name__)

router = APIRouter(prefix="/settings", tags=["设置管理"])

# ==================== AI 路由（按任务选择不同模型/Provider） ====================
#
# 设计目标：
# - 允许用户在前端设置“不同 AI 请求类型使用不同 API 预设（provider/key/base_url/model）”
# - 路由信息存储在 settings.preferences(JSON) 中，避免新增数据库字段
# - 后端通过 task_key 解析路由，动态创建对应的 AIService（从而支持跨 provider 分流且使用不同 key）
#

AI_ROUTE_VERSION = "1.0"

# task_key 约定：只用于“路由选择”，不直接暴露为业务概念。
# 前端使用这些 key 显示配置项；后端在各 API 调用点选择对应 key。
AI_ROUTE_TASKS: list[dict[str, str]] = [
    {"key": "wizard_world_building", "label": "向导：世界观生成", "category": "向导"},
    {"key": "inspiration_options", "label": "灵感模式：生成选项", "category": "灵感"},
    {"key": "outline_generate", "label": "大纲：生成/续写", "category": "大纲"},
    {"key": "outline_expand", "label": "大纲：展开章节规划", "category": "大纲"},
    {"key": "chapter_generate", "label": "章节：生成正文", "category": "章节"},
    {"key": "chapter_regenerate", "label": "章节：重写/再生成", "category": "章节"},
    {"key": "chapter_analysis", "label": "章节：剧情分析/伏笔提取", "category": "章节"},
    {"key": "polish", "label": "文本：去味/润色", "category": "文本"},
    {"key": "character_generate", "label": "设定：生成角色", "category": "设定"},
    {"key": "organization_generate", "label": "设定：生成组织", "category": "设定"},
    {"key": "career_generate", "label": "设定：生成/补全职业", "category": "设定"},
]


class AIRouteTask(BaseModel):
    key: str
    label: str
    category: str


class AIRoutesResponse(BaseModel):
    version: str = AI_ROUTE_VERSION
    routes: Dict[str, Optional[str]]  # task_key -> preset_id (None 表示使用当前配置)
    tasks: List[AIRouteTask]


class AIRoutesUpdateRequest(BaseModel):
    routes: Dict[str, Optional[str]]


# ==================== 向量检索配置（Embedding / Rerank） ====================
#
# 设计目标：
# - 允许用户在前端 Settings 中配置：
#   1) 本地 embedding（现有 sentence-transformers） 或 远端 embedding（OpenAI 兼容接口）
#   2) 可选的远端 rerank（Cohere 兼容接口）
# - 配置存储在 settings.preferences(JSON) 中的 retrieval 字段，避免数据库改动
#

RETRIEVAL_CONFIG_VERSION = "1.0"


class RemoteEmbeddingConfig(BaseModel):
    """远端 Embedding 配置（OpenAI 兼容：/v1/embeddings）。"""

    # 允许字段名包含 `model_` 等受保护前缀
    model_config = {"protected_namespaces": ()}

    provider: str = "openai_compatible"
    api_key: Optional[str] = None
    api_base_url: Optional[str] = None  # 例如：https://api.openai.com/v1
    model: Optional[str] = None  # 例如：text-embedding-3-small / Qwen3-Embedding-8B
    timeout_s: int = 60


class EmbeddingConfig(BaseModel):
    """Embedding 配置。"""

    model_config = {"protected_namespaces": ()}

    backend: str = "local"  # local | remote
    remote: RemoteEmbeddingConfig = RemoteEmbeddingConfig()


class RemoteRerankConfig(BaseModel):
    """远端 Rerank 配置（Cohere 兼容：/v1/rerank）。"""

    model_config = {"protected_namespaces": ()}

    provider: str = "cohere_compatible"
    api_key: Optional[str] = None
    api_base_url: Optional[str] = None  # 例如：https://api.cohere.ai/v1
    model: Optional[str] = None
    timeout_s: int = 60
    top_k: int = 30  # 向量召回候选数量
    top_n: int = 10  # rerank 后返回数量
    min_score: Optional[float] = None  # 可选：过滤过低分


class RerankConfig(BaseModel):
    model_config = {"protected_namespaces": ()}

    enabled: bool = False
    remote: RemoteRerankConfig = RemoteRerankConfig()


class RetrievalConfigResponse(BaseModel):
    """检索配置响应。"""

    model_config = {"protected_namespaces": ()}

    version: str = RETRIEVAL_CONFIG_VERSION
    embedding: EmbeddingConfig = EmbeddingConfig()
    rerank: RerankConfig = RerankConfig()


class RetrievalConfigUpdateRequest(BaseModel):
    """检索配置更新请求。"""

    model_config = {"protected_namespaces": ()}

    embedding: EmbeddingConfig
    rerank: RerankConfig


class EmbeddingTestRequest(BaseModel):
    """Embedding 可用性检测请求（不落库，仅用于 UI 检测按钮）。"""

    model_config = {"protected_namespaces": ()}

    embedding: EmbeddingConfig


class EmbeddingTestResponse(BaseModel):
    """Embedding 可用性检测响应。"""

    model_config = {"protected_namespaces": ()}

    success: bool
    message: str
    backend: str
    response_time_ms: Optional[float] = None
    details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    suggestions: Optional[List[str]] = None


class RerankTestRequest(BaseModel):
    """Rerank 可用性检测请求（不落库，仅用于 UI 检测按钮）。"""

    model_config = {"protected_namespaces": ()}

    rerank: RerankConfig


class RerankTestResponse(BaseModel):
    """Rerank 可用性检测响应。"""

    model_config = {"protected_namespaces": ()}

    success: bool
    enabled: bool
    message: str
    response_time_ms: Optional[float] = None
    details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    suggestions: Optional[List[str]] = None


def _safe_load_preferences(preferences: Optional[str]) -> Dict[str, Any]:
    if not preferences:
        return {}
    try:
        raw = json.loads(preferences)
        return raw if isinstance(raw, dict) else {}
    except json.JSONDecodeError:
        return {}


def _get_presets_from_preferences(prefs: Dict[str, Any]) -> List[Dict[str, Any]]:
    api_presets = prefs.get("api_presets") or {}
    presets = api_presets.get("presets") or []
    return presets if isinstance(presets, list) else []


def _get_ai_routes_from_preferences(prefs: Dict[str, Any]) -> Dict[str, Optional[str]]:
    ai_routes = prefs.get("ai_routes") or {}
    routes = ai_routes.get("routes") if isinstance(ai_routes, dict) else None
    if not isinstance(routes, dict):
        return {}

    # 只保留 str/None，避免脏数据
    cleaned: Dict[str, Optional[str]] = {}
    for k, v in routes.items():
        if v is None:
            cleaned[str(k)] = None
        elif isinstance(v, str) and v.strip():
            cleaned[str(k)] = v.strip()
        else:
            cleaned[str(k)] = None
    return cleaned


def _upsert_ai_routes_to_preferences(
    prefs: Dict[str, Any],
    routes: Dict[str, Optional[str]],
) -> Dict[str, Any]:
    prefs = dict(prefs or {})
    prefs["ai_routes"] = {
        "version": AI_ROUTE_VERSION,
        "updated_at": datetime.now().isoformat(),
        "routes": routes,
    }
    return prefs


def _get_retrieval_from_preferences(prefs: Dict[str, Any]) -> Dict[str, Any]:
    retrieval = prefs.get("retrieval") if isinstance(prefs, dict) else None
    return retrieval if isinstance(retrieval, dict) else {}


def _upsert_retrieval_to_preferences(
    prefs: Dict[str, Any],
    retrieval_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    prefs = dict(prefs or {})
    cfg = dict(retrieval_cfg or {})
    cfg["version"] = RETRIEVAL_CONFIG_VERSION
    prefs["retrieval"] = cfg
    return prefs


def _resolve_preset_config(
    *,
    prefs: Dict[str, Any],
    preset_id: str,
) -> Optional[Dict[str, Any]]:
    if not preset_id:
        return None
    for preset in _get_presets_from_preferences(prefs):
        if str(preset.get("id")) == preset_id:
            cfg = preset.get("config")
            return cfg if isinstance(cfg, dict) else None
    return None


async def create_user_ai_service_for_task(
    *,
    user_id: str,
    db: AsyncSession,
    task_key: str,
    request_enable_mcp: Optional[bool] = None,
) -> AIService:
    """根据 task_key 路由创建对应 AIService（支持跨 provider + 不同 key）。

    路由优先级：
    1) preferences.ai_routes.routes[task_key] 指向的 preset config
    2) 回退到 Settings 主字段（当前配置）
    """
    from app.models.mcp_plugin import MCPPlugin

    settings = await get_user_settings(user_id, db)
    prefs = _safe_load_preferences(settings.preferences)
    routes = _get_ai_routes_from_preferences(prefs)

    preset_id = routes.get(task_key)
    preset_cfg = _resolve_preset_config(prefs=prefs, preset_id=preset_id) if preset_id else None

    # MCP 启用逻辑：与 get_user_ai_service 保持一致，再叠加 request_enable_mcp（如传入）
    mcp_result = await db.execute(select(MCPPlugin).where(MCPPlugin.user_id == user_id))
    mcp_plugins = mcp_result.scalars().all()
    enable_mcp_by_plugins = any(plugin.enabled for plugin in mcp_plugins) if mcp_plugins else False
    enable_mcp = enable_mcp_by_plugins
    if request_enable_mcp is not None:
        enable_mcp = enable_mcp and bool(request_enable_mcp)

    if preset_cfg:
        # 数值类型（例如 temperature=0.0）不能用 `or` 合并，否则 0.0 会被误判为“未设置”
        resolved_temperature = preset_cfg.get("temperature")
        if resolved_temperature is None:
            resolved_temperature = settings.temperature if settings.temperature is not None else app_settings.default_temperature

        resolved_max_tokens = preset_cfg.get("max_tokens")
        if resolved_max_tokens is None:
            resolved_max_tokens = settings.max_tokens if settings.max_tokens is not None else app_settings.default_max_tokens

        return create_user_ai_service_with_mcp(
            api_provider=str(preset_cfg.get("api_provider") or settings.api_provider),
            api_key=str(preset_cfg.get("api_key") or settings.api_key or ""),
            api_base_url=str(preset_cfg.get("api_base_url") or settings.api_base_url or ""),
            model_name=str(preset_cfg.get("llm_model") or settings.llm_model),
            temperature=float(resolved_temperature),
            max_tokens=int(resolved_max_tokens),
            user_id=user_id,
            db_session=db,
            system_prompt=settings.system_prompt,
            enable_mcp=enable_mcp,
        )

    # fallback：当前配置
    return create_user_ai_service_with_mcp(
        api_provider=settings.api_provider,
        api_key=settings.api_key,
        api_base_url=settings.api_base_url or "",
        model_name=settings.llm_model,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        user_id=user_id,
        db_session=db,
        system_prompt=settings.system_prompt,
        enable_mcp=enable_mcp,
    )


def read_env_defaults() -> Dict[str, Any]:
    """从.env文件读取默认配置（仅读取，不修改）"""
    return {
        "api_provider": app_settings.default_ai_provider,
        "api_key": app_settings.openai_api_key or app_settings.anthropic_api_key or "",
        "api_base_url": app_settings.openai_base_url or app_settings.anthropic_base_url or "",
        "llm_model": app_settings.default_model,
        "temperature": app_settings.default_temperature,
        "max_tokens": app_settings.default_max_tokens,
    }


def require_login(request: Request):
    """依赖：要求用户已登录"""
    if not hasattr(request.state, "user") or not request.state.user:
        raise HTTPException(status_code=401, detail="需要登录")
    return request.state.user


async def get_user_ai_service(
    user: User = Depends(require_login),
    db: AsyncSession = Depends(get_db)
) -> AIService:
    """
    依赖：获取当前用户的AI服务实例（支持MCP工具自动加载）
    
    从数据库读取用户设置并创建对应的AI服务。
    自动传递 user_id 和 db_session，使得 AIService 能够加载用户配置的MCP工具。
    根据用户的所有MCP插件状态决定是否启用MCP：如果有启用的插件则启用，否则禁用。
    """
    from app.models.mcp_plugin import MCPPlugin
    
    result = await db.execute(
        select(Settings).where(Settings.user_id == user.user_id)
    )
    settings = result.scalar_one_or_none()
    
    if not settings:
        # 如果用户没有设置，从.env读取并保存
        env_defaults = read_env_defaults()
        settings = Settings(
            user_id=user.user_id,
            **env_defaults
        )
        db.add(settings)
        await db.commit()
        await db.refresh(settings)
        logger.info(f"用户 {user.user_id} 首次使用AI服务，已从.env同步设置到数据库")
    
    # 查询用户的所有MCP插件状态
    mcp_result = await db.execute(
        select(MCPPlugin).where(MCPPlugin.user_id == user.user_id)
    )
    mcp_plugins = mcp_result.scalars().all()
    
    # 检查是否有启用的MCP插件
    enable_mcp = any(plugin.enabled for plugin in mcp_plugins) if mcp_plugins else False
    
    if mcp_plugins:
        enabled_count = sum(1 for p in mcp_plugins if p.enabled)
        logger.info(f"用户 {user.user_id} 有 {len(mcp_plugins)} 个MCP插件，{enabled_count} 个启用，{enable_mcp} 决定使用MCP")
    else:
        logger.debug(f"用户 {user.user_id} 没有配置MCP插件，禁用MCP")
    
    # ✅ 使用支持MCP的工厂函数创建AI服务实例
    # 传递 user_id 和 db_session，使得 AIService 能够自动加载用户配置的MCP工具
    return create_user_ai_service_with_mcp(
        api_provider=settings.api_provider,
        api_key=settings.api_key,
        api_base_url=settings.api_base_url or "",
        model_name=settings.llm_model,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        user_id=user.user_id,          # ✅ 传递 user_id
        db_session=db,                 # ✅ 传递 db_session
        system_prompt=settings.system_prompt,
        enable_mcp=enable_mcp,         # 根据MCP插件状态动态决定
    )


def get_user_ai_service_for_task(task_key: str):
    """依赖工厂：为指定 task_key 创建 AIService（支持按任务路由到不同预设）。

    用法示例：
        user_ai_service: AIService = Depends(get_user_ai_service_for_task("chapter_generate"))
    """

    async def _dep(
        user: User = Depends(require_login),
        db: AsyncSession = Depends(get_db),
    ) -> AIService:
        return await create_user_ai_service_for_task(
            user_id=user.user_id,
            db=db,
            task_key=task_key,
        )

    return _dep


@router.get("", response_model=SettingsResponse)
async def get_settings(
    user: User = Depends(require_login),
    db: AsyncSession = Depends(get_db)
):
    """
    获取当前用户的设置
    如果用户没有保存过设置，自动从.env创建并保存到数据库
    """
    result = await db.execute(
        select(Settings).where(Settings.user_id == user.user_id)
    )
    settings = result.scalar_one_or_none()
    
    if not settings:
        # 如果用户没有保存过设置，从.env读取默认配置并保存到数据库
        env_defaults = read_env_defaults()
        logger.info(f"用户 {user.user_id} 首次获取设置，自动从.env同步到数据库")
        
        # 创建新设置并保存到数据库
        settings = Settings(
            user_id=user.user_id,
            **env_defaults
        )
        db.add(settings)
        await db.commit()
        await db.refresh(settings)
        logger.info(f"用户 {user.user_id} 的设置已从.env同步到数据库")
    
    logger.info(f"用户 {user.user_id} 获取已保存的设置")
    return settings


@router.post("", response_model=SettingsResponse)
async def save_settings(
    data: SettingsCreate,
    user: User = Depends(require_login),
    db: AsyncSession = Depends(get_db)
):
    """
    创建或更新当前用户的设置（Upsert）
    如果设置已存在则更新，否则创建新设置
    仅保存到数据库
    
    注意：手动保存配置后会自动取消之前激活的预设状态，
    因为手动修改的配置可能与预设不一致
    """
    # 查找现有设置
    result = await db.execute(
        select(Settings).where(Settings.user_id == user.user_id)
    )
    settings = result.scalar_one_or_none()
    
    # 准备数据
    settings_dict = data.model_dump(exclude_unset=True)
    
    if settings:
        # 更新现有设置
        for key, value in settings_dict.items():
            setattr(settings, key, value)
        
        # 检查并取消预设激活状态
        # 因为用户手动修改了配置，可能与之前激活的预设不一致
        try:
            prefs = json.loads(settings.preferences or '{}')
            api_presets = prefs.get('api_presets', {'presets': [], 'version': '1.0'})
            presets = api_presets.get('presets', [])
            
            # 找到激活的预设并检查是否与当前保存的配置一致
            active_preset = next((p for p in presets if p.get('is_active')), None)
            if active_preset:
                preset_config = active_preset.get('config', {})
                # 检查配置是否发生变化
                config_changed = (
                    preset_config.get('api_provider') != settings_dict.get('api_provider', settings.api_provider) or
                    preset_config.get('api_key') != settings_dict.get('api_key', settings.api_key) or
                    preset_config.get('api_base_url') != settings_dict.get('api_base_url', settings.api_base_url) or
                    preset_config.get('llm_model') != settings_dict.get('llm_model', settings.llm_model) or
                    preset_config.get('temperature') != settings_dict.get('temperature', settings.temperature) or
                    preset_config.get('max_tokens') != settings_dict.get('max_tokens', settings.max_tokens)
                )
                
                if config_changed:
                    # 取消激活状态
                    active_preset['is_active'] = False
                    prefs['api_presets'] = api_presets
                    settings.preferences = json.dumps(prefs, ensure_ascii=False)
                    logger.info(f"用户 {user.user_id} 手动修改配置，已取消预设 {active_preset.get('name')} 的激活状态")
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"解析用户 {user.user_id} 的preferences失败: {e}")
        
        await db.commit()
        await db.refresh(settings)
        logger.info(f"用户 {user.user_id} 更新设置")
    else:
        # 创建新设置
        settings = Settings(
            user_id=user.user_id,
            **settings_dict
        )
        db.add(settings)
        await db.commit()
        await db.refresh(settings)
        logger.info(f"用户 {user.user_id} 创建设置")
    
    return settings


@router.put("", response_model=SettingsResponse)
async def update_settings(
    data: SettingsUpdate,
    user: User = Depends(require_login),
    db: AsyncSession = Depends(get_db)
):
    """
    更新当前用户的设置
    仅保存到数据库
    """
    result = await db.execute(
        select(Settings).where(Settings.user_id == user.user_id)
    )
    settings = result.scalar_one_or_none()
    
    if not settings:
        raise HTTPException(status_code=404, detail="设置不存在，请先创建设置")
    
    # 更新设置
    update_data = data.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(settings, key, value)
    
    await db.commit()
    await db.refresh(settings)
    logger.info(f"用户 {user.user_id} 更新设置")
    
    return settings


@router.delete("")
async def delete_settings(
    user: User = Depends(require_login),
    db: AsyncSession = Depends(get_db)
):
    """
    删除当前用户的设置
    """
    result = await db.execute(
        select(Settings).where(Settings.user_id == user.user_id)
    )
    settings = result.scalar_one_or_none()
    
    if not settings:
        raise HTTPException(status_code=404, detail="设置不存在")
    
    await db.delete(settings)
    await db.commit()
    logger.info(f"用户 {user.user_id} 删除设置")
    
    return {"message": "设置已删除", "user_id": user.user_id}


@router.get("/models")
async def get_available_models(
    api_key: str,
    api_base_url: str,
    provider: str = "openai"
):
    """
    从配置的 API 获取可用的模型列表
    
    Args:
        api_key: API 密钥
        api_base_url: API 基础 URL
        provider: API 提供商 (openai, anthropic, azure, custom)
    
    Returns:
        模型列表
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            if provider == "openai" or provider == "azure" or provider == "custom":
                # OpenAI 兼容接口获取模型列表
                url = f"{api_base_url.rstrip('/')}/models"
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                
                logger.info(f"正在从 {url} 获取模型列表")
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                
                data = response.json()
                models = []
                
                if "data" in data and isinstance(data["data"], list):
                    for model in data["data"]:
                        model_id = model.get("id", "")
                        # 返回所有模型，不进行过滤
                        if model_id:
                            models.append({
                                "value": model_id,
                                "label": model_id,
                                "description": model.get("description", "") or f"Created: {model.get('created', 'N/A')}"
                            })
                
                if not models:
                    raise HTTPException(
                        status_code=404,
                        detail="未能从 API 获取到可用的模型列表"
                    )
                
                logger.info(f"成功获取 {len(models)} 个模型")
                return {
                    "provider": provider,
                    "models": models,
                    "count": len(models)
                }
                
            elif provider == "anthropic":
                # Anthropic models API
                url = f"{api_base_url.rstrip('/')}/v1/models"
                headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                data = response.json()
                models = [{"value": m["id"], "label": m["id"], "description": m.get("display_name", "")} for m in data.get("data", [])]
                return {"provider": provider, "models": models, "count": len(models)}
            
            elif provider == "gemini":
                # Gemini models API
                url = f"{api_base_url.rstrip('/')}/models?key={api_key}"
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()
                models = []
                for m in data.get("models", []):
                    if "generateContent" in m.get("supportedGenerationMethods", []):
                        mid = m.get("name", "").replace("models/", "")
                        models.append({"value": mid, "label": m.get("displayName", mid), "description": ""})
                return {"provider": provider, "models": models, "count": len(models)}
            
            else:
                raise HTTPException(status_code=400, detail=f"不支持的提供商: {provider}")
            
    except httpx.HTTPStatusError as e:
        logger.error(f"获取模型列表失败 (HTTP {e.response.status_code}): {e.response.text}")
        raise HTTPException(
            status_code=400,
            detail=f"无法从 API 获取模型列表 (HTTP {e.response.status_code})"
        )
    except httpx.RequestError as e:
        logger.error(f"请求模型列表失败: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"无法连接到 API: {str(e)}"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取模型列表时发生错误: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"获取模型列表失败: {str(e)}"
        )


class ApiTestRequest(BaseModel):
    """API 测试请求模型"""
    api_key: str
    api_base_url: str
    provider: str
    llm_model: str
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


@router.post("/check-function-calling")
async def check_function_calling_support(data: ApiTestRequest):
    """
    检查模型是否支持 Function Calling（工具调用）
    
    基于业界最佳实践的测试方法：
    1. 发送包含工具定义的请求
    2. 检查响应的 finish_reason 是否为 "tool_calls"
    3. 验证响应中是否包含有效的 tool_calls 数据
    
    Args:
        data: 包含 API 配置的请求数据
    
    Returns:
        检测结果包含支持状态、详细信息和建议
    """
    api_key = data.api_key
    api_base_url = data.api_base_url
    provider = data.provider
    llm_model = data.llm_model
    
    try:
        start_time = time.time()
        
        # 定义一个简单的测试工具（天气查询）
        test_tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "获取指定城市的当前天气信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "城市名称，例如：北京、上海、深圳"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "温度单位"
                        }
                    },
                    "required": ["city"]
                }
            }
        }]
        
        # 测试提示：故意设计一个需要调用工具的问题
        test_prompt = "请告诉我北京现在的天气情况如何？"
        
        logger.info(f"🧪 开始检测 Function Calling 支持")
        logger.info(f"  - 提供商: {provider}")
        logger.info(f"  - 模型: {llm_model}")
        logger.info(f"  - 测试工具: get_weather")
        
        # 创建临时 AI 服务实例进行测试
        test_service = AIService(
            api_provider=provider,
            api_key=api_key,
            api_base_url=api_base_url,
            default_model=llm_model,
            default_temperature=0.3,  # 使用较低温度以获得更确定的行为
            default_max_tokens=200
        )
        
        # 发送带工具的测试请求
        response = await test_service.generate_text(
            prompt=test_prompt,
            provider=provider,
            model=llm_model,
            temperature=0.3,
            max_tokens=200,
            tools=test_tools,
            tool_choice="auto",  # 让模型自动决定是否使用工具
            auto_mcp=False  # 禁用 MCP 自动加载
        )
        
        end_time = time.time()
        response_time = round((end_time - start_time) * 1000, 2)
        
        # 分析响应以确定是否支持 Function Calling
        supported = False
        finish_reason = None
        tool_calls = None
        response_content = None
        
        if isinstance(response, dict):
            # 检查 finish_reason（OpenAI 标准）
            finish_reason = response.get("finish_reason")
            
            # 检查是否有 tool_calls
            if "tool_calls" in response and response["tool_calls"]:
                supported = True
                tool_calls = response["tool_calls"]
                logger.info(f"✅ 检测到工具调用: {len(tool_calls)} 个")
            
            # 记录返回的内容（如果有）
            if "content" in response:
                response_content = response["content"]
        elif isinstance(response, str):
            # 如果只返回字符串，说明不支持工具调用
            response_content = response
        
        logger.info(f"  - 响应时间: {response_time}ms")
        logger.info(f"  - finish_reason: {finish_reason}")
        logger.info(f"  - 支持状态: {'✅ 支持' if supported else '❌ 不支持'}")
        
        # 构建详细的返回信息
        result = {
            "success": True,
            "supported": supported,
            "message": "✅ 模型支持 Function Calling" if supported else "❌ 模型不支持 Function Calling",
            "response_time_ms": response_time,
            "provider": provider,
            "model": llm_model,
            "details": {
                "finish_reason": finish_reason,
                "has_tool_calls": bool(tool_calls),
                "tool_call_count": len(tool_calls) if tool_calls else 0,
                "test_tool": "get_weather",
                "test_prompt": test_prompt,
                "response_type": "tool_calls" if supported else "text"
            }
        }
        
        # 添加工具调用详情
        if tool_calls:
            result["tool_calls"] = tool_calls
            result["suggestions"] = [
                "✅ 该模型支持 Function Calling，可以正常使用 MCP 插件",
                "建议：启用需要的 MCP 插件以扩展 AI 能力",
                "提示：测试成功检测到工具调用，模型能够正确解析和使用外部工具"
            ]
        else:
            result["response_preview"] = response_content[:200] if response_content else None
            result["suggestions"] = [
                "❌ 该模型不支持 Function Calling，无法使用 MCP 插件功能",
                "建议：更换支持工具调用的模型",
                "推荐模型：GPT-4 系列、GPT-4-turbo、Claude 3 Opus/Sonnet、Gemini 1.5 Pro 等",
                "说明：模型返回了文本回复而非工具调用，表明不支持该功能"
            ]
        
        return result
        
    except ValueError as e:
        error_msg = str(e)
        logger.error(f"❌ Function Calling 检测配置错误: {error_msg}")
        return {
            "success": False,
            "supported": False,
            "message": "配置错误",
            "error": error_msg,
            "error_type": "ConfigurationError",
            "suggestions": [
                "请检查 API Key 是否正确",
                "请确认 API Base URL 格式是否正确",
                "请验证所选提供商与配置是否匹配"
            ]
        }
        
    except TimeoutError as e:
        error_msg = str(e)
        logger.error(f"❌ Function Calling 检测超时: {error_msg}")
        return {
            "success": False,
            "supported": None,
            "message": "检测超时",
            "error": error_msg,
            "error_type": "TimeoutError",
            "suggestions": [
                "请检查网络连接是否正常",
                "请确认 API 服务是否可访问",
                "建议：稍后重试或使用其他网络环境"
            ]
        }
        
    except Exception as e:
        error_msg = str(e)
        error_type = type(e).__name__
        
        logger.error(f"❌ Function Calling 检测失败: {error_msg}")
        logger.error(f"  - 错误类型: {error_type}")
        
        # 智能分析错误原因
        suggestions = []
        if "tool" in error_msg.lower() or "function" in error_msg.lower():
            suggestions = [
                "该模型可能不支持 Function Calling 功能",
                "API 返回了与工具调用相关的错误",
                "建议：更换支持工具调用的模型或联系 API 提供商"
            ]
        elif "unauthorized" in error_msg.lower() or "401" in error_msg:
            suggestions = [
                "API Key 认证失败",
                "请检查 API Key 是否正确且有效",
                "请确认 API Key 是否有足够的权限"
            ]
        elif "not found" in error_msg.lower() or "404" in error_msg:
            suggestions = [
                "模型不存在或不可用",
                "请检查模型名称是否正确",
                "请确认该模型在当前 API 中是否可用"
            ]
        else:
            suggestions = [
                "检测过程中遇到未知错误",
                "建议：检查所有配置参数是否正确",
                "提示：查看详细错误信息以获取更多线索"
            ]
        
        return {
            "success": False,
            "supported": False,
            "message": "Function Calling 检测失败",
            "error": error_msg,
            "error_type": error_type,
            "suggestions": suggestions
        }


@router.post("/test")
async def test_api_connection(data: ApiTestRequest):
    """
    测试 API 连接和配置是否正确
    
    Args:
        data: 包含 API 配置的请求数据（包括 temperature 和 max_tokens）
    
    Returns:
        测试结果包含状态、响应时间和详细信息
    """
    api_key = data.api_key
    api_base_url = data.api_base_url
    provider = data.provider
    llm_model = data.llm_model
    # 使用前端传递的参数，如果未传递则使用默认值
    temperature = data.temperature if data.temperature is not None else 0.7
    max_tokens = data.max_tokens if data.max_tokens is not None else 2000
    import time
    
    try:
        start_time = time.time()
        
        # 创建临时 AI 服务实例，使用前端传递的参数
        test_service = AIService(
            api_provider=provider,
            api_key=api_key,
            api_base_url=api_base_url,
            default_model=llm_model,
            default_temperature=temperature,
            default_max_tokens=max_tokens
        )
        
        # 发送简单的测试请求
        test_prompt = "请用一句话回复：测试成功"
        
        logger.info(f"🧪 开始测试 API 连接")
        logger.info(f"  - 提供商: {provider}")
        logger.info(f"  - 模型: {llm_model}")
        logger.info(f"  - Base URL: {api_base_url}")
        logger.info(f"  - Temperature: {temperature}")
        logger.info(f"  - Max Tokens: {max_tokens}")
        
        response = await test_service.generate_text(
            prompt=test_prompt,
            provider=provider,
            model=llm_model,
            temperature=temperature,
            max_tokens=max_tokens,
            auto_mcp=False  # 测试时不加载MCP工具
        )
        
        end_time = time.time()
        response_time = round((end_time - start_time) * 1000, 2)  # 转换为毫秒
        
        logger.info(f"✅ API 测试成功")
        logger.info(f"  - 响应时间: {response_time}ms")
        
        # 安全地处理响应内容（确保是字符串）
        response_str = str(response) if response else 'N/A'
        logger.info(f"  - 响应内容: {response_str[:100]}")
        
        return {
            "success": True,
            "message": "API 连接测试成功",
            "response_time_ms": response_time,
            "provider": provider,
            "model": llm_model,
            "response_preview": response_str[:100] if len(response_str) > 100 else response_str,
            "details": {
                "api_available": True,
                "model_accessible": True,
                "response_valid": bool(response),
                "temperature": temperature,
                "max_tokens": max_tokens
            }
        }
        
    except ValueError as e:
        # 配置错误
        error_msg = str(e)
        logger.error(f"❌ API 配置错误: {error_msg}")
        return {
            "success": False,
            "message": "API 配置错误",
            "error": error_msg,
            "error_type": "ConfigurationError",
            "suggestions": [
                "请检查 API Key 是否正确",
                "请确认 API Base URL 格式正确",
                "请验证所选提供商是否匹配"
            ]
        }
        
    except TimeoutError as e:
        # 超时错误
        error_msg = str(e)
        logger.error(f"❌ API 请求超时: {error_msg}")
        return {
            "success": False,
            "message": "API 请求超时",
            "error": error_msg,
            "error_type": "TimeoutError",
            "suggestions": [
                "请检查网络连接",
                "请确认 API Base URL 是否可访问",
                "如果使用代理，请检查代理设置"
            ]
        }
        
    except Exception as e:
        # 其他错误
        error_msg = str(e)
        error_type = type(e).__name__
        
        logger.error(f"❌ API 测试失败: {error_msg}")
        logger.error(f"  - 错误类型: {error_type}")
        
        # 分析错误原因并提供建议
        suggestions = []
        if "blocked" in error_msg.lower():
            suggestions = [
                "请求被 API 提供商阻止",
                "可能原因：API Key 被限制或地区限制",
                "建议：检查 API Key 状态和账户余额",
                "建议：尝试更换 API Base URL 或使用代理"
            ]
        elif "unauthorized" in error_msg.lower() or "401" in error_msg:
            suggestions = [
                "API Key 认证失败",
                "建议：检查 API Key 是否正确",
                "建议：确认 API Key 是否过期"
            ]
        elif "not found" in error_msg.lower() or "404" in error_msg:
            suggestions = [
                "API 端点不存在或模型不可用",
                "建议：检查 API Base URL 是否正确",
                "建议：确认模型名称是否正确"
            ]
        elif "rate limit" in error_msg.lower() or "429" in error_msg:
            suggestions = [
                "API 请求频率超限",
                "建议：稍后重试",
                "建议：升级 API 套餐"
            ]
        elif "insufficient" in error_msg.lower() or "quota" in error_msg.lower():
            suggestions = [
                "API 配额不足",
                "建议：检查账户余额",
                "建议：充值或升级套餐"
            ]
        else:
            suggestions = [
                "请检查所有配置参数是否正确",
                "请确认网络连接正常",
                "请查看详细错误信息"
            ]
        
        return {
            "success": False,
            "message": "API 测试失败",
            "error": error_msg,
            "error_type": error_type,
            "suggestions": suggestions
        }


# ========== API配置预设管理（零数据库改动方案）==========

async def get_user_settings(user_id: str, db: AsyncSession) -> Settings:
    """获取用户settings，如果不存在则创建"""
    result = await db.execute(
        select(Settings).where(Settings.user_id == user_id)
    )
    settings = result.scalar_one_or_none()
    
    if not settings:
        # 创建默认设置
        env_defaults = read_env_defaults()
        settings = Settings(
            user_id=user_id,
            **env_defaults,
            preferences='{}'  # 初始化为空JSON
        )
        db.add(settings)
        await db.commit()
        await db.refresh(settings)
        logger.info(f"用户 {user_id} 首次访问，已创建默认设置")
    
    return settings


@router.get("/presets", response_model=PresetListResponse)
async def get_presets(
    user: User = Depends(require_login),
    db: AsyncSession = Depends(get_db)
):
    """
    获取所有API配置预设
    
    从preferences字段读取预设列表
    """
    settings = await get_user_settings(user.user_id, db)
    
    # 解析preferences
    try:
        prefs = json.loads(settings.preferences or '{}')
    except json.JSONDecodeError:
        logger.warning(f"用户 {user.user_id} 的preferences字段JSON格式错误，重置为空")
        prefs = {}
    
    api_presets = prefs.get('api_presets', {'presets': [], 'version': '1.0'})
    presets = api_presets.get('presets', [])
    
    # 找到激活的预设
    active_preset_id = next(
        (p['id'] for p in presets if p.get('is_active')),
        None
    )
    
    logger.info(f"用户 {user.user_id} 获取预设列表，共 {len(presets)} 个")
    
    return {
        "presets": presets,
        "total": len(presets),
        "active_preset_id": active_preset_id
    }


@router.post("/presets", response_model=PresetResponse)
async def create_preset(
    data: PresetCreateRequest,
    user: User = Depends(require_login),
    db: AsyncSession = Depends(get_db)
):
    """
    创建新预设
    
    将预设添加到preferences字段的JSON中
    """
    settings = await get_user_settings(user.user_id, db)
    
    # 解析preferences
    try:
        prefs = json.loads(settings.preferences or '{}')
    except json.JSONDecodeError:
        prefs = {}
    
    api_presets = prefs.get('api_presets', {'presets': [], 'version': '1.0'})
    presets = api_presets.get('presets', [])
    
    # 创建新预设
    new_preset = {
        "id": f"preset_{int(datetime.now().timestamp() * 1000)}",
        "name": data.name,
        "description": data.description,
        "is_active": False,
        "created_at": datetime.now().isoformat(),
        "config": data.config.model_dump()
    }
    
    presets.append(new_preset)
    
    # 保存回preferences
    api_presets['presets'] = presets
    prefs['api_presets'] = api_presets
    settings.preferences = json.dumps(prefs, ensure_ascii=False)
    
    await db.commit()
    
    logger.info(f"用户 {user.user_id} 创建预设: {data.name}")
    return new_preset


@router.put("/presets/{preset_id}", response_model=PresetResponse)
async def update_preset(
    preset_id: str,
    data: PresetUpdateRequest,
    user: User = Depends(require_login),
    db: AsyncSession = Depends(get_db)
):
    """
    更新预设
    
    在preferences字段的JSON中更新指定预设
    """
    settings = await get_user_settings(user.user_id, db)
    
    # 解析preferences
    try:
        prefs = json.loads(settings.preferences or '{}')
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="配置数据格式错误")
    
    api_presets = prefs.get('api_presets', {'presets': [], 'version': '1.0'})
    presets = api_presets.get('presets', [])
    
    # 找到并更新预设
    target_preset = next((p for p in presets if p['id'] == preset_id), None)
    if not target_preset:
        raise HTTPException(status_code=404, detail="预设不存在")
    
    # 更新字段
    if data.name is not None:
        target_preset['name'] = data.name
    if data.description is not None:
        target_preset['description'] = data.description
    if data.config is not None:
        target_preset['config'] = data.config.model_dump()
    
    # 保存回preferences
    prefs['api_presets'] = api_presets
    settings.preferences = json.dumps(prefs, ensure_ascii=False)
    
    await db.commit()
    
    logger.info(f"用户 {user.user_id} 更新预设: {preset_id}")
    return target_preset


@router.delete("/presets/{preset_id}")
async def delete_preset(
    preset_id: str,
    user: User = Depends(require_login),
    db: AsyncSession = Depends(get_db)
):
    """
    删除预设
    
    从preferences字段的JSON中删除指定预设
    """
    settings = await get_user_settings(user.user_id, db)
    
    # 解析preferences
    try:
        prefs = json.loads(settings.preferences or '{}')
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="配置数据格式错误")
    
    api_presets = prefs.get('api_presets', {'presets': [], 'version': '1.0'})
    presets = api_presets.get('presets', [])
    
    # 找到预设
    target_preset = next((p for p in presets if p['id'] == preset_id), None)
    if not target_preset:
        raise HTTPException(status_code=404, detail="预设不存在")
    
    # 检查是否是激活的预设
    if target_preset.get('is_active'):
        raise HTTPException(status_code=400, detail="无法删除激活中的预设，请先激活其他预设")
    
    # 删除预设
    presets = [p for p in presets if p['id'] != preset_id]
    
    # 保存回preferences
    api_presets['presets'] = presets
    prefs['api_presets'] = api_presets
    settings.preferences = json.dumps(prefs, ensure_ascii=False)
    
    await db.commit()
    
    logger.info(f"用户 {user.user_id} 删除预设: {preset_id}")
    return {"message": "预设已删除", "preset_id": preset_id}


@router.post("/presets/{preset_id}/activate")
async def activate_preset(
    preset_id: str,
    user: User = Depends(require_login),
    db: AsyncSession = Depends(get_db)
):
    """
    激活预设
    
    将预设的配置应用到Settings主字段
    """
    settings = await get_user_settings(user.user_id, db)
    
    # 解析preferences
    try:
        prefs = json.loads(settings.preferences or '{}')
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="配置数据格式错误")
    
    api_presets = prefs.get('api_presets', {'presets': [], 'version': '1.0'})
    presets = api_presets.get('presets', [])
    
    # 找到目标预设
    target_preset = next((p for p in presets if p['id'] == preset_id), None)
    if not target_preset:
        raise HTTPException(status_code=404, detail="预设不存在")
    
    # 应用配置到Settings主字段
    config = target_preset['config']
    settings.api_provider = config['api_provider']
    settings.api_key = config['api_key']
    settings.api_base_url = config.get('api_base_url')
    settings.llm_model = config['llm_model']
    settings.temperature = config['temperature']
    settings.max_tokens = config['max_tokens']
    
    # 更新所有预设的is_active状态
    for preset in presets:
        preset['is_active'] = (preset['id'] == preset_id)
    
    # 保存回preferences
    prefs['api_presets'] = api_presets
    settings.preferences = json.dumps(prefs, ensure_ascii=False)
    
    await db.commit()
    
    logger.info(f"用户 {user.user_id} 激活预设: {target_preset['name']}")
    return {
        "message": "预设已激活",
        "preset_id": preset_id,
        "preset_name": target_preset['name']
    }


@router.get("/ai-routes", response_model=AIRoutesResponse)
async def get_ai_routes(
    user: User = Depends(require_login),
    db: AsyncSession = Depends(get_db),
):
    """获取 AI 任务路由配置（task_key -> preset_id）。"""
    settings = await get_user_settings(user.user_id, db)
    prefs = _safe_load_preferences(settings.preferences)
    routes = _get_ai_routes_from_preferences(prefs)

    # 确保所有已定义 task_key 都返回（未配置则 None）
    full_routes: Dict[str, Optional[str]] = {
        item["key"]: routes.get(item["key"])
        for item in AI_ROUTE_TASKS
    }

    return AIRoutesResponse(
        version=AI_ROUTE_VERSION,
        routes=full_routes,
        tasks=[AIRouteTask(**item) for item in AI_ROUTE_TASKS],
    )


@router.put("/ai-routes", response_model=AIRoutesResponse)
async def update_ai_routes(
    data: AIRoutesUpdateRequest,
    user: User = Depends(require_login),
    db: AsyncSession = Depends(get_db),
):
    """更新 AI 任务路由配置。

    - routes 的 value 为 preset_id 或 None（None 表示使用当前配置）
    - preset_id 必须存在于用户的 api_presets 中
    """
    settings = await get_user_settings(user.user_id, db)
    prefs = _safe_load_preferences(settings.preferences)

    # 校验 preset_id 是否存在
    presets = _get_presets_from_preferences(prefs)
    preset_ids = {str(p.get("id")) for p in presets if p.get("id")}

    for task_key, preset_id in (data.routes or {}).items():
        if preset_id is None:
            continue
        if not isinstance(preset_id, str) or not preset_id.strip():
            continue
        if preset_id not in preset_ids:
            raise HTTPException(status_code=400, detail=f"路由配置引用了不存在的预设: {preset_id} (task={task_key})")

    # 只保存已定义任务的配置（忽略未知 key，避免前后端版本差异造成污染）
    normalized_routes: Dict[str, Optional[str]] = {}
    for item in AI_ROUTE_TASKS:
        key = item["key"]
        val = (data.routes or {}).get(key)
        if isinstance(val, str) and val.strip():
            normalized_routes[key] = val.strip()
        else:
            normalized_routes[key] = None

    prefs = _upsert_ai_routes_to_preferences(prefs, normalized_routes)
    settings.preferences = json.dumps(prefs, ensure_ascii=False)
    await db.commit()

    return AIRoutesResponse(
        version=AI_ROUTE_VERSION,
        routes=normalized_routes,
        tasks=[AIRouteTask(**item) for item in AI_ROUTE_TASKS],
    )


@router.get("/retrieval", response_model=RetrievalConfigResponse)
async def get_retrieval_config(
    user: User = Depends(require_login),
    db: AsyncSession = Depends(get_db),
):
    """获取向量检索配置（Embedding / Rerank）。"""
    settings = await get_user_settings(user.user_id, db)
    prefs = _safe_load_preferences(settings.preferences)
    raw = _get_retrieval_from_preferences(prefs)
    try:
        return RetrievalConfigResponse(**raw)
    except Exception:
        # 配置损坏时回退到默认
        return RetrievalConfigResponse()


@router.put("/retrieval", response_model=RetrievalConfigResponse)
async def update_retrieval_config(
    data: RetrievalConfigUpdateRequest,
    user: User = Depends(require_login),
    db: AsyncSession = Depends(get_db),
):
    """更新向量检索配置（Embedding / Rerank）。"""
    settings = await get_user_settings(user.user_id, db)
    prefs = _safe_load_preferences(settings.preferences)

    cfg = RetrievalConfigResponse(
        version=RETRIEVAL_CONFIG_VERSION,
        embedding=data.embedding,
        rerank=data.rerank,
    )

    prefs = _upsert_retrieval_to_preferences(prefs, cfg.model_dump())
    settings.preferences = json.dumps(prefs, ensure_ascii=False)
    await db.commit()

    # 返回写入后的最终值（带默认补齐）
    raw = _get_retrieval_from_preferences(prefs)
    try:
        return RetrievalConfigResponse(**raw)
    except Exception:
        return RetrievalConfigResponse()




@router.post("/retrieval/test-embedding", response_model=EmbeddingTestResponse)
async def test_embedding_config(
    data: EmbeddingTestRequest,
    user: User = Depends(require_login),
    db: AsyncSession = Depends(get_db),
):
    """检测 Embedding 配置是否可用（用于前端 Settings 检测按钮）。

    - backend=local：检查本地 sentence-transformers 模型是否已加载
    - backend=remote：调用 OpenAI 兼容 /v1/embeddings 做一次最小请求并校验返回格式

    注意：该接口不写入 settings.preferences，仅用于检测。
    """
    from app.services.memory_service import memory_service

    settings = await get_user_settings(user.user_id, db)

    embedding = data.embedding
    backend = str(getattr(embedding, "backend", None) or "local").lower()

    if backend != "remote":
        ok = getattr(memory_service, "embedding_model", None) is not None
        if ok:
            return EmbeddingTestResponse(
                success=True,
                backend="local",
                message="本地 Embedding 模型已加载，可正常使用",
                details={
                    "model_loaded": True,
                    "hint": "当前为本地 embedding。若想避免本地模型体积/依赖，可切换到远端 embedding。",
                },
            )
        return EmbeddingTestResponse(
            success=False,
            backend="local",
            message="本地 Embedding 模型未加载（将导致记忆检索失败）",
            error="Local embedding model not loaded",
            error_type="LocalEmbeddingNotLoaded",
            suggestions=[
                "请确认 embedding 模型文件已存在（例如 paraphrase-multilingual-MiniLM-L12-v2）",
                "如果你使用的是 Docker 镜像版本：请确认镜像内已包含 embedding 模型文件，或按项目说明下载模型",
                "也可以在设置中将 Embedding 后端切换为「远端（OpenAI 兼容 /v1/embeddings）」",
            ],
        )

    remote = embedding.remote
    provider = str(getattr(remote, "provider", None) or "openai_compatible")
    model = getattr(remote, "model", None) or None

    # 允许留空复用当前 LLM 配置（与 MemoryService 行为保持一致）
    api_base_url = getattr(remote, "api_base_url", None) or (settings.api_base_url if settings else None)
    api_key = getattr(remote, "api_key", None) or (settings.api_key if settings else None) or ""
    timeout_s = int(getattr(remote, "timeout_s", None) or 60)

    used_fallback_base_url = not bool(getattr(remote, "api_base_url", None))
    used_fallback_api_key = not bool(getattr(remote, "api_key", None))

    if not api_base_url or not model:
        return EmbeddingTestResponse(
            success=False,
            backend="remote",
            message="远端 Embedding 配置不完整",
            error="api_base_url / model 不能为空（api_base_url 允许留空复用当前配置，但当前配置也为空）",
            error_type="ConfigurationError",
            suggestions=[
                "请填写 Embedding 模型名称（model）",
                "请填写 Embedding API 地址（api_base_url），或先在「当前配置」中填写 API 地址以供复用",
            ],
        )

    if provider != "openai_compatible":
        provider = "openai_compatible"

    url = memory_service._build_openai_compatible_url(str(api_base_url), "embeddings")
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {"model": model, "input": ["你好，Embedding 测试"], "encoding_format": "float"}

    start = time.time()
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data_json = resp.json()

        cost_ms = round((time.time() - start) * 1000, 2)

        items = data_json.get("data") or []
        if not isinstance(items, list) or not items:
            raise ValueError("返回缺少 data 数组或 data 为空")

        first = items[0] if isinstance(items[0], dict) else None
        if not first:
            raise ValueError("返回 data[0] 不是对象")

        emb = first.get("embedding")
        if not isinstance(emb, list) or not emb:
            raise ValueError("返回 data[0].embedding 不是非空 list")

        try:
            _ = float(emb[0])
        except Exception:
            raise ValueError("返回 embedding 不是数值列表")

        return EmbeddingTestResponse(
            success=True,
            backend="remote",
            message="远端 Embedding 检测通过",
            response_time_ms=cost_ms,
            details={
                "api_base_url": api_base_url,
                "endpoint": url,
                "model": model,
                "vector_dim": len(emb),
                "preview": emb[:8],
                "used_fallback": {
                    "api_base_url": used_fallback_base_url,
                    "api_key": used_fallback_api_key,
                },
            },
        )

    except httpx.HTTPStatusError as e:
        status = e.response.status_code if e.response else None
        raw = ""
        try:
            raw = e.response.text if e.response else ""
        except Exception:
            raw = ""

        suggestions: list[str] = []
        if status in (401, 403):
            suggestions = ["API Key 认证失败：请检查 Embedding API Key 是否正确/有权限。"]
        elif status == 404:
            suggestions = ["接口或模型不存在：请检查 api_base_url 是否包含 /v1，以及 model 名称是否正确。"]
        elif status == 400:
            suggestions = [
                "请求参数不符合该服务的 embeddings 接口要求。",
                "建议检查：model 名称、是否需要 encoding_format、input 是否支持数组。",
            ]
        elif status == 429:
            suggestions = ["触发限流（429）：请降低并发/频率或稍后重试，或更换额度更高的服务。"]
        else:
            suggestions = ["服务端返回错误：请查看错误详情或稍后重试。"]

        return EmbeddingTestResponse(
            success=False,
            backend="remote",
            message="远端 Embedding 检测失败",
            error=f"HTTP {status}: {raw[:500]}",
            error_type="HTTPStatusError",
            suggestions=suggestions,
        )

    except httpx.RequestError as e:
        return EmbeddingTestResponse(
            success=False,
            backend="remote",
            message="远端 Embedding 检测失败（网络/连接错误）",
            error=str(e),
            error_type="RequestError",
            suggestions=[
                "请检查网络是否可达该 Embedding 服务",
                "请检查 api_base_url 是否正确（是否需要走代理）",
                "若是自建服务：确认端口、TLS、域名解析正常",
            ],
        )

    except Exception as e:
        return EmbeddingTestResponse(
            success=False,
            backend="remote",
            message="远端 Embedding 检测失败（未知错误）",
            error=str(e),
            error_type=type(e).__name__,
            suggestions=["请检查配置是否正确，或查看后端日志 app.log 获取更多信息。"],
        )


@router.post("/retrieval/test-rerank", response_model=RerankTestResponse)
async def test_rerank_config(
    data: RerankTestRequest,
    user: User = Depends(require_login),
    db: AsyncSession = Depends(get_db),
):
    """检测 Rerank 配置是否可用（用于前端 Settings 检测按钮）。

    - enabled=false：直接返回提示（不报错）
    - enabled=true：调用 Cohere 兼容 /v1/rerank 做一次最小请求并校验返回格式

    注意：该接口不写入 settings.preferences，仅用于检测。
    """
    from app.services.memory_service import memory_service

    settings = await get_user_settings(user.user_id, db)

    rerank = data.rerank
    enabled = bool(getattr(rerank, "enabled", False))
    if not enabled:
        return RerankTestResponse(
            success=True,
            enabled=False,
            message="Rerank 未启用，无需检测",
            details={"enabled": False},
        )

    remote = rerank.remote
    provider = str(getattr(remote, "provider", None) or "cohere_compatible")
    model = getattr(remote, "model", None) or None
    api_base_url = getattr(remote, "api_base_url", None) or (settings.api_base_url if settings else None)
    api_key = getattr(remote, "api_key", None) or (settings.api_key if settings else None) or ""
    timeout_s = int(getattr(remote, "timeout_s", None) or 60)
    top_n_cfg = int(getattr(remote, "top_n", None) or 10)

    used_fallback_base_url = not bool(getattr(remote, "api_base_url", None))
    used_fallback_api_key = not bool(getattr(remote, "api_key", None))

    if not api_base_url or not model:
        return RerankTestResponse(
            success=False,
            enabled=True,
            message="远端 Rerank 配置不完整",
            error="api_base_url / model 不能为空（api_base_url 允许留空复用当前配置，但当前配置也为空）",
            error_type="ConfigurationError",
            suggestions=[
                "请填写 Rerank 模型名称（model）",
                "请填写 Rerank API 地址（api_base_url），或先在「当前配置」中填写 API 地址以供复用",
            ],
        )

    if provider != "cohere_compatible":
        provider = "cohere_compatible"

    url = memory_service._build_openai_compatible_url(str(api_base_url), "rerank")
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    query = "如何在2005年的电脑上提升程序运行速度？"
    documents = [
        "给内存加到512MB以上，关闭后台程序，减少磁盘碎片。",
        "买一台更好的显卡可以让打字更快。",
        "降低算法复杂度，避免不必要的IO，使用缓存。",
    ]
    top_n = max(1, min(top_n_cfg, len(documents)))

    payload = {"model": model, "query": query, "documents": documents, "top_n": top_n}

    start = time.time()
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data_json = resp.json()

        cost_ms = round((time.time() - start) * 1000, 2)

        results = data_json.get("results") or data_json.get("data") or []
        if not isinstance(results, list) or not results:
            raise ValueError("返回缺少 results 数组或 results 为空")

        parsed: list[dict[str, Any]] = []
        for r in results[: min(5, len(results))]:
            if not isinstance(r, dict):
                continue
            idx = r.get("index")
            score = r.get("relevance_score", r.get("score"))
            try:
                idx_i = int(idx)
            except Exception:
                continue
            try:
                score_f = float(score) if score is not None else None
            except Exception:
                score_f = None
            parsed.append({"index": idx_i, "score": score_f})

        if not parsed:
            raise ValueError("results 解析失败：缺少 index / relevance_score")

        return RerankTestResponse(
            success=True,
            enabled=True,
            message="远端 Rerank 检测通过",
            response_time_ms=cost_ms,
            details={
                "api_base_url": api_base_url,
                "endpoint": url,
                "model": model,
                "top_n": top_n,
                "sample_query": query,
                "sample_docs_count": len(documents),
                "top_results_preview": parsed,
                "used_fallback": {
                    "api_base_url": used_fallback_base_url,
                    "api_key": used_fallback_api_key,
                },
            },
        )

    except httpx.HTTPStatusError as e:
        status = e.response.status_code if e.response else None
        raw = ""
        try:
            raw = e.response.text if e.response else ""
        except Exception:
            raw = ""

        suggestions: list[str] = []
        if status in (401, 403):
            suggestions = ["API Key 认证失败：请检查 Rerank API Key 是否正确/有权限。"]
        elif status == 404:
            suggestions = ["接口或模型不存在：请检查 api_base_url 是否包含 /v1，以及 model 名称是否正确。"]
        elif status == 400:
            suggestions = [
                "请求参数不符合该服务的 rerank 接口要求。",
                "建议检查：model 名称、top_n 是否合理、documents 是否为字符串数组。",
            ]
        elif status == 429:
            suggestions = ["触发限流（429）：请降低并发/频率或稍后重试，或更换额度更高的服务。"]
        else:
            suggestions = ["服务端返回错误：请查看错误详情或稍后重试。"]

        return RerankTestResponse(
            success=False,
            enabled=True,
            message="远端 Rerank 检测失败",
            error=f"HTTP {status}: {raw[:500]}",
            error_type="HTTPStatusError",
            suggestions=suggestions,
        )

    except httpx.RequestError as e:
        return RerankTestResponse(
            success=False,
            enabled=True,
            message="远端 Rerank 检测失败（网络/连接错误）",
            error=str(e),
            error_type="RequestError",
            suggestions=[
                "请检查网络是否可达该 Rerank 服务",
                "请检查 api_base_url 是否正确（是否需要走代理）",
                "若是自建服务：确认端口、TLS、域名解析正常",
            ],
        )

    except Exception as e:
        return RerankTestResponse(
            success=False,
            enabled=True,
            message="远端 Rerank 检测失败（未知错误）",
            error=str(e),
            error_type=type(e).__name__,
            suggestions=["请检查配置是否正确，或查看后端日志 app.log 获取更多信息。"],
        )
@router.post("/presets/{preset_id}/test")
async def test_preset(
    preset_id: str,
    user: User = Depends(require_login),
    db: AsyncSession = Depends(get_db)
):
    """
    测试预设的API连接
    """
    settings = await get_user_settings(user.user_id, db)
    
    # 解析preferences
    try:
        prefs = json.loads(settings.preferences or '{}')
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="配置数据格式错误")
    
    api_presets = prefs.get('api_presets', {'presets': [], 'version': '1.0'})
    presets = api_presets.get('presets', [])
    
    # 找到预设
    target_preset = next((p for p in presets if p['id'] == preset_id), None)
    if not target_preset:
        raise HTTPException(status_code=404, detail="预设不存在")
    
    # 使用现有的test_api_connection逻辑
    # 确保传递完整参数，与当前配置测试保持一致
    config = target_preset['config']
    test_request = ApiTestRequest(
        api_key=config['api_key'],
        api_base_url=config.get('api_base_url', ''),
        provider=config['api_provider'],
        llm_model=config['llm_model'],
        temperature=config.get('temperature'),   # 使用预设中的温度参数
        max_tokens=config.get('max_tokens')      # 使用预设中的最大tokens参数
    )
    
    logger.info(f"用户 {user.user_id} 测试预设: {target_preset['name']}")
    return await test_api_connection(test_request)


@router.post("/presets/from-current", response_model=PresetResponse)
async def create_preset_from_current(
    name: str,
    description: Optional[str] = None,
    user: User = Depends(require_login),
    db: AsyncSession = Depends(get_db)
):
    """
    从当前配置创建新预设
    
    快捷方式：将当前激活的配置保存为新预设
    """
    settings = await get_user_settings(user.user_id, db)
    
    # 从当前Settings主字段读取配置
    current_config = APIKeyPresetConfig(
        api_provider=settings.api_provider,
        api_key=settings.api_key,
        api_base_url=settings.api_base_url,
        llm_model=settings.llm_model,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens
    )
    
    # 创建预设
    create_request = PresetCreateRequest(
        name=name,
        description=description,
        config=current_config
    )
    
    logger.info(f"用户 {user.user_id} 从当前配置创建预设: {name}")
    return await create_preset(create_request, user, db)
