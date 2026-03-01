"""项目创建向导流式API - 使用SSE避免超时

⚠️ 说明（2026-03）
- 向导只负责**生成世界观**（world-building）。
- 职业/角色/大纲/地点列表等内容不再由向导生成，改为在项目内按需调用各自的业务 API：
  - 职业体系：`/api/careers/generate-system`
  - 角色卡：`/api/characters/generate-stream`
  - 大纲：`/api/outlines/generate-stream`
  - 大纲展开成章节：`/api/outlines/batch-expand-stream` / `/{outline_id}/expand-stream`
"""

from __future__ import annotations

import json
from typing import Any, AsyncGenerator, Dict

from fastapi import APIRouter, Depends, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.settings import get_user_ai_service_for_task
from app.database import get_db
from app.logger import get_logger
from app.models.project import Project
from app.models.project_default_style import ProjectDefaultStyle
from app.models.writing_style import WritingStyle
from app.services.ai_service import AIService
from app.services.prompt_service import PromptService
from app.utils.sse_response import SSEResponse, WizardProgressTracker, create_sse_response

router = APIRouter(prefix="/wizard-stream", tags=["项目创建向导(流式)"])
logger = get_logger(__name__)


async def world_building_generator(
    data: Dict[str, Any],
    db: AsyncSession,
    user_ai_service: AIService,
) -> AsyncGenerator[str, None]:
    """世界构建流式生成器（只生成世界观并创建项目）"""

    db_committed = False
    tracker = WizardProgressTracker("世界观")

    try:
        yield await tracker.start()

        title = data.get("title")
        description = data.get("description")
        theme = data.get("theme")
        genre = data.get("genre")
        narrative_perspective = data.get("narrative_perspective")
        target_words = data.get("target_words")
        chapter_count = data.get("chapter_count")
        character_count = data.get("character_count")
        outline_mode = data.get("outline_mode", "one-to-many")
        provider = data.get("provider")
        model = data.get("model")
        enable_mcp = data.get("enable_mcp", True)  # 预留：模板/插件可能会用到
        user_id = data.get("user_id")

        if not title or not description or not theme or not genre:
            yield await tracker.error("title、description、theme 和 genre 是必需的参数", 400)
            return

        yield await tracker.preparing("准备AI提示词...")
        template = await PromptService.get_template("WORLD_BUILDING", user_id, db)
        base_prompt = PromptService.format_prompt(
            template,
            title=title,
            theme=theme,
            genre=genre or "通用类型",
            description=description or "暂无简介",
        )

        if user_id:
            user_ai_service.user_id = user_id
            user_ai_service.db_session = db

        MAX_WORLD_RETRIES = 3
        world_retry_count = 0
        world_generation_success = False
        world_data: Dict[str, Any] = {}
        estimated_total = 1000

        while world_retry_count < MAX_WORLD_RETRIES and not world_generation_success:
            try:
                if world_retry_count > 0:
                    tracker.reset_generating_progress()

                yield await tracker.generating(
                    current_chars=0,
                    estimated_total=estimated_total,
                    retry_count=world_retry_count,
                    max_retries=MAX_WORLD_RETRIES,
                )

                accumulated_text = ""
                chunk_count = 0

                async for chunk in user_ai_service.generate_text_stream(
                    prompt=base_prompt,
                    provider=provider,
                    model=model,
                    tool_choice="required",
                ):
                    chunk_count += 1
                    accumulated_text += chunk

                    yield await tracker.generating_chunk(chunk)

                    current_len = len(accumulated_text)
                    if chunk_count % 10 == 0:
                        yield await tracker.generating(
                            current_chars=current_len,
                            estimated_total=estimated_total,
                            retry_count=world_retry_count,
                            max_retries=MAX_WORLD_RETRIES,
                        )

                    if chunk_count % 20 == 0:
                        yield await tracker.heartbeat()

                if not accumulated_text or not accumulated_text.strip():
                    logger.warning(f"⚠️ AI返回空世界观（尝试{world_retry_count+1}/{MAX_WORLD_RETRIES}）")
                    world_retry_count += 1
                    if world_retry_count < MAX_WORLD_RETRIES:
                        yield await tracker.retry(world_retry_count, MAX_WORLD_RETRIES, "AI返回为空")
                        continue
                    logger.error("❌ 世界观生成多次返回空响应")
                    world_data = {
                        "time_period": "AI多次返回为空，请稍后重试",
                        "location": "AI多次返回为空，请稍后重试",
                        "atmosphere": "AI多次返回为空，请稍后重试",
                        "rules": "AI多次返回为空，请稍后重试",
                    }
                    world_generation_success = True
                    break

                yield await tracker.parsing("解析世界观数据...")

                try:
                    logger.info(f"🔍 开始清洗JSON，原始长度: {len(accumulated_text)}")
                    logger.info(f"   原始内容预览: {accumulated_text[:300]}...")

                    cleaned_text = user_ai_service._clean_json_response(accumulated_text)
                    logger.info(f"✅ JSON清洗完成，清洗后长度: {len(cleaned_text)}")
                    logger.info(f"   清洗后预览: {cleaned_text[:300]}...")

                    world_data = json.loads(cleaned_text)
                    logger.info(f"✅ 世界观JSON解析成功（尝试{world_retry_count+1}/{MAX_WORLD_RETRIES}）")
                    world_generation_success = True
                except json.JSONDecodeError as e:
                    logger.error(f"❌ 世界构建JSON解析失败（尝试{world_retry_count+1}/{MAX_WORLD_RETRIES}）: {e}")
                    logger.error(f"   原始内容长度: {len(accumulated_text)}")
                    logger.error(f"   原始内容预览: {accumulated_text[:200]}")
                    world_retry_count += 1
                    if world_retry_count < MAX_WORLD_RETRIES:
                        yield await tracker.retry(world_retry_count, MAX_WORLD_RETRIES, "JSON解析失败")
                        continue
                    world_data = {
                        "time_period": "AI返回格式错误，请重试",
                        "location": "AI返回格式错误，请重试",
                        "atmosphere": "AI返回格式错误，请重试",
                        "rules": "AI返回格式错误，请重试",
                    }
                    world_generation_success = True
            except Exception as e:
                logger.error(
                    f"❌ 世界构建生成异常（尝试{world_retry_count+1}/{MAX_WORLD_RETRIES}）: {type(e).__name__}: {e}"
                )
                world_retry_count += 1
                if world_retry_count < MAX_WORLD_RETRIES:
                    yield await tracker.retry(world_retry_count, MAX_WORLD_RETRIES, "生成异常")
                    continue
                logger.error(
                    "   accumulated_text 长度: %s",
                    len(accumulated_text) if "accumulated_text" in locals() else "N/A",
                )
                raise

        yield await tracker.saving("保存世界观到数据库...")

        if not user_id:
            yield await SSEResponse.send_error("用户ID缺失，无法创建项目", 401)
            return

        project = Project(
            user_id=user_id,
            title=title,
            description=description,
            theme=theme,
            genre=genre,
            world_time_period=world_data.get("time_period"),
            world_location=world_data.get("location"),
            world_atmosphere=world_data.get("atmosphere"),
            world_rules=world_data.get("rules"),
            narrative_perspective=narrative_perspective,
            target_words=target_words,
            chapter_count=chapter_count,
            character_count=character_count,
            outline_mode=outline_mode,
            # ✅ 新策略：创建项目时只初始化世界观即可视为“可进入项目”
            wizard_status="completed",
            wizard_step=1,
            status="planning",
        )
        db.add(project)
        await db.commit()
        await db.refresh(project)

        # 自动设置默认写作风格为第一个全局预设风格
        try:
            result = await db.execute(
                select(WritingStyle).where(
                    WritingStyle.user_id.is_(None),
                    WritingStyle.order_index == 1,
                ).limit(1)
            )
            first_style = result.scalar_one_or_none()
            if first_style:
                default_style = ProjectDefaultStyle(project_id=project.id, style_id=first_style.id)
                db.add(default_style)
                await db.commit()
                logger.info(f"为项目 {project.id} 自动设置默认风格: {first_style.name}")
            else:
                logger.warning(
                    f"未找到order_index=1的全局预设风格，项目 {project.id} 未设置默认风格"
                )
        except Exception as e:
            logger.warning(f"设置默认写作风格失败: {e}，不影响项目创建")

        project.wizard_step = 1
        project.wizard_status = "completed"
        await db.commit()

        db_committed = True

        yield await tracker.complete()
        yield await tracker.result(
            {
                "project_id": project.id,
                "time_period": world_data.get("time_period"),
                "location": world_data.get("location"),
                "atmosphere": world_data.get("atmosphere"),
                "rules": world_data.get("rules"),
            }
        )
        yield await tracker.done()

        logger.info(f"✅ 世界观生成完成，项目ID: {project.id}")

    except GeneratorExit:
        logger.warning("世界构建生成器被提前关闭")
        if not db_committed and db.in_transaction():
            await db.rollback()
            logger.info("世界构建事务已回滚（GeneratorExit）")
    except Exception as e:
        logger.error(f"世界构建流式生成失败: {str(e)}")
        if not db_committed and db.in_transaction():
            await db.rollback()
            logger.info("世界构建事务已回滚（异常）")
        yield await tracker.error(f"生成失败: {str(e)}")


@router.post("/world-building", summary="流式生成世界构建")
async def generate_world_building_stream(
    request: Request,
    data: Dict[str, Any],
    db: AsyncSession = Depends(get_db),
    user_ai_service: AIService = Depends(get_user_ai_service_for_task("wizard_world_building")),
):
    """使用SSE流式生成世界构建，避免超时。"""
    if hasattr(request.state, "user_id"):
        data["user_id"] = request.state.user_id
    return create_sse_response(world_building_generator(data, db, user_ai_service))


async def world_building_regenerate_generator(
    project_id: str,
    data: Dict[str, Any],
    db: AsyncSession,
    user_ai_service: AIService,
) -> AsyncGenerator[str, None]:
    """世界观重新生成流式生成器（不落库，仅返回预览结果）"""

    db_committed = False
    tracker = WizardProgressTracker("世界观")

    try:
        yield await tracker.start("开始重新生成世界观...")

        yield await tracker.loading("加载项目信息...")
        result = await db.execute(select(Project).where(Project.id == project_id))
        project = result.scalar_one_or_none()
        if not project:
            yield await tracker.error("项目不存在", 404)
            return

        provider = data.get("provider")
        model = data.get("model")
        enable_mcp = data.get("enable_mcp", True)  # 预留
        user_id = data.get("user_id")

        yield await tracker.preparing("准备AI提示词...")
        template = await PromptService.get_template("WORLD_BUILDING", user_id, db)
        base_prompt = PromptService.format_prompt(
            template,
            title=project.title,
            theme=project.theme or "未设定",
            genre=project.genre or "通用",
            description=project.description or "暂无简介",
        )

        if user_id:
            user_ai_service.user_id = user_id
            user_ai_service.db_session = db

        MAX_WORLD_RETRIES = 3
        world_retry_count = 0
        world_generation_success = False
        world_data: Dict[str, Any] = {}
        estimated_total = 1000

        while world_retry_count < MAX_WORLD_RETRIES and not world_generation_success:
            try:
                if world_retry_count > 0:
                    tracker.reset_generating_progress()

                yield await tracker.generating(
                    current_chars=0,
                    estimated_total=estimated_total,
                    message="重新生成世界观",
                    retry_count=world_retry_count,
                    max_retries=MAX_WORLD_RETRIES,
                )

                accumulated_text = ""
                chunk_count = 0

                async for chunk in user_ai_service.generate_text_stream(
                    prompt=base_prompt,
                    provider=provider,
                    model=model,
                    tool_choice="required",
                ):
                    chunk_count += 1
                    accumulated_text += chunk

                    yield await tracker.generating_chunk(chunk)

                    current_len = len(accumulated_text)
                    if chunk_count % 10 == 0:
                        yield await tracker.generating(
                            current_chars=current_len,
                            estimated_total=estimated_total,
                            message="重新生成世界观",
                            retry_count=world_retry_count,
                            max_retries=MAX_WORLD_RETRIES,
                        )

                    if chunk_count % 20 == 0:
                        yield await tracker.heartbeat()

                if not accumulated_text or not accumulated_text.strip():
                    logger.warning(f"⚠️ AI返回空世界观（尝试{world_retry_count+1}/{MAX_WORLD_RETRIES}）")
                    world_retry_count += 1
                    if world_retry_count < MAX_WORLD_RETRIES:
                        yield await tracker.retry(world_retry_count, MAX_WORLD_RETRIES, "AI返回为空")
                        continue
                    logger.error("❌ 世界观重新生成多次返回空响应")
                    world_data = {
                        "time_period": "AI多次返回为空，请稍后重试",
                        "location": "AI多次返回为空，请稍后重试",
                        "atmosphere": "AI多次返回为空，请稍后重试",
                        "rules": "AI多次返回为空，请稍后重试",
                    }
                    world_generation_success = True
                    break

                yield await tracker.parsing("解析AI返回结果...")

                try:
                    logger.info(f"🔍 开始清洗JSON，原始长度: {len(accumulated_text)}")
                    cleaned_text = user_ai_service._clean_json_response(accumulated_text)
                    logger.info(f"✅ JSON清洗完成，清洗后长度: {len(cleaned_text)}")

                    world_data = json.loads(cleaned_text)
                    logger.info(
                        f"✅ 世界观重新生成JSON解析成功（尝试{world_retry_count+1}/{MAX_WORLD_RETRIES}）"
                    )
                    world_generation_success = True
                except json.JSONDecodeError as e:
                    logger.error(f"❌ 世界构建JSON解析失败（尝试{world_retry_count+1}/{MAX_WORLD_RETRIES}）: {e}")
                    logger.error(f"   原始内容长度: {len(accumulated_text)}")
                    logger.error(f"   原始内容预览: {accumulated_text[:200]}")
                    world_retry_count += 1
                    if world_retry_count < MAX_WORLD_RETRIES:
                        yield await tracker.retry(world_retry_count, MAX_WORLD_RETRIES, "JSON解析失败")
                        continue
                    world_data = {
                        "time_period": "AI返回格式错误，请重试",
                        "location": "AI返回格式错误，请重试",
                        "atmosphere": "AI返回格式错误，请重试",
                        "rules": "AI返回格式错误，请重试",
                    }
                    world_generation_success = True
            except Exception as e:
                logger.error(
                    f"❌ 世界观重新生成异常（尝试{world_retry_count+1}/{MAX_WORLD_RETRIES}）: {type(e).__name__}: {e}"
                )
                world_retry_count += 1
                if world_retry_count < MAX_WORLD_RETRIES:
                    yield await tracker.retry(world_retry_count, MAX_WORLD_RETRIES, "生成异常")
                    continue
                logger.error(
                    "   accumulated_text 长度: %s",
                    len(accumulated_text) if "accumulated_text" in locals() else "N/A",
                )
                raise

        yield await tracker.saving("生成完成，等待用户确认...", 0.5)
        yield await tracker.complete()
        yield await tracker.result(
            {
                "time_period": world_data.get("time_period"),
                "location": world_data.get("location"),
                "atmosphere": world_data.get("atmosphere"),
                "rules": world_data.get("rules"),
            }
        )
        yield await tracker.done()

    except GeneratorExit:
        logger.warning("世界观重新生成器被提前关闭")
        if not db_committed and db.in_transaction():
            await db.rollback()
            logger.info("世界观重新生成事务已回滚（GeneratorExit）")
    except Exception as e:
        logger.error(f"世界观重新生成失败: {str(e)}")
        if not db_committed and db.in_transaction():
            await db.rollback()
            logger.info("世界观重新生成事务已回滚（异常）")
        yield await tracker.error(f"生成失败: {str(e)}")


@router.post("/world-building/{project_id}/regenerate", summary="流式重新生成世界观")
async def regenerate_world_building_stream(
    project_id: str,
    request: Request,
    data: Dict[str, Any],
    db: AsyncSession = Depends(get_db),
    user_ai_service: AIService = Depends(get_user_ai_service_for_task("wizard_world_building")),
):
    """使用SSE流式重新生成世界观（不落库，仅返回预览结果）。"""
    if hasattr(request.state, "user_id"):
        data["user_id"] = request.state.user_id
    return create_sse_response(world_building_regenerate_generator(project_id, data, db, user_ai_service))

