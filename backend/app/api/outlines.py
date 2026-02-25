"""大纲管理API"""
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, delete
from typing import List, AsyncGenerator, Dict, Any, Optional
import json

from app.database import get_db
from app.api.common import verify_project_access
from app.models.outline import Outline
from app.models.project import Project
from app.models.chapter import Chapter
from app.models.character import Character
from app.models.relationship import CharacterRelationship, Organization, OrganizationMember
from app.models.generation_history import GenerationHistory
from app.schemas.outline import (
    OutlineCreate,
    OutlineUpdate,
    OutlineResponse,
    OutlineListResponse,
    OutlineGenerateRequest,
    OutlineExpansionRequest,
    OutlineExpansionResponse,
    BatchOutlineExpansionRequest,
    BatchOutlineExpansionResponse,
    CreateChaptersFromPlansRequest,
    CreateChaptersFromPlansResponse
)
from app.services.ai_service import AIService
from app.services.prompt_service import prompt_service, PromptService
from app.services.memory_service import memory_service
from app.services.plot_expansion_service import PlotExpansionService
from app.services.foreshadow_service import foreshadow_service
from app.services.memory_service import memory_service
from app.logger import get_logger
from app.api.settings import get_user_ai_service_for_task
from app.utils.sse_response import SSEResponse, create_sse_response, WizardProgressTracker

router = APIRouter(prefix="/outlines", tags=["大纲管理"])
logger = get_logger(__name__)


def _build_chapters_brief(outlines: List[Outline], max_recent: int = 20) -> str:
    """构建章节概览字符串"""
    target = outlines[-max_recent:] if len(outlines) > max_recent else outlines
    return "\n".join([f"第{o.order_index}章《{o.title}》" for o in target])


def _build_characters_info(characters: List[Character]) -> str:
    """构建角色信息字符串"""
    return "\n".join([
        f"- {char.name} ({'组织' if char.is_organization else '角色'}, {char.role_type}): "
        f"{char.personality[:100] if char.personality else '暂无描述'}"
        for char in characters
    ])


def _extract_created_names(items: Optional[List[Any]]) -> List[str]:
    """从对象列表中提取名称，兼容 ORM 对象 / dict / str。"""
    if not items:
        return []

    names: List[str] = []
    for item in items:
        name = None
        if isinstance(item, dict):
            name = item.get("name")
        elif isinstance(item, str):
            name = item
        else:
            name = getattr(item, "name", None)
        if isinstance(name, str) and name.strip():
            names.append(name.strip())
    return names


def _normalize_postcheck_item(
    *,
    section: str,
    raw_result: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    规范化后处理结果（角色/组织）。

    section:
      - characters
      - organizations
    """
    raw = raw_result or {}

    created_key = "created_characters" if section == "characters" else "created_organizations"
    created_items = raw.get(created_key) or []
    created_names = _extract_created_names(created_items)
    missing_names = raw.get("missing_names") or []
    success = bool(raw.get("success", True))
    error = raw.get("error")

    status = "success" if success else "warning"

    return {
        "status": status,
        "success": success,
        "created_count": int(raw.get("created_count", 0) or 0),
        "created_names": created_names,
        "missing_names": [x for x in missing_names if isinstance(x, str)],
        "error": str(error) if error else None,
    }


def _build_postcheck_summary(
    *,
    scope: str,
    char_result: Optional[Dict[str, Any]] = None,
    org_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """构建统一的后处理摘要，便于 UI 与 MCP 消费。"""
    include_characters = scope in {"all", "characters"}
    include_organizations = scope in {"all", "organizations"}

    characters = (
        _normalize_postcheck_item(section="characters", raw_result=char_result)
        if include_characters
        else None
    )
    organizations = (
        _normalize_postcheck_item(section="organizations", raw_result=org_result)
        if include_organizations
        else None
    )

    active_sections = [x for x in [characters, organizations] if x is not None]
    warning_sections = [x for x in active_sections if x.get("status") != "success"]

    if not active_sections:
        overall_status = "success"
    elif len(warning_sections) == len(active_sections):
        overall_status = "failed"
    elif warning_sections:
        overall_status = "partial_failed"
    else:
        overall_status = "success"

    total_created = sum(int(x.get("created_count", 0) or 0) for x in active_sections)

    return {
        "scope": scope,
        "overall_status": overall_status,
        "has_warning": bool(warning_sections),
        "total_created": total_created,
        "characters": characters,
        "organizations": organizations,
    }


def _extract_outline_data_for_postcheck(outlines: List[Outline]) -> List[Dict[str, Any]]:
    """
    从大纲记录中提取后处理所需结构数据（兼容 structure 异常场景）。
    """
    rows: List[Dict[str, Any]] = []
    for outline in outlines:
        parsed: Optional[Any] = None
        if outline.structure:
            try:
                parsed = json.loads(outline.structure)
            except Exception:
                logger.warning(f"大纲 {outline.id} 的 structure 解析失败，后处理将使用降级数据")

        if isinstance(parsed, dict):
            rows.append(parsed)
            continue
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    rows.append(item)
            if parsed:
                continue

        # 降级：至少保留标题/摘要，characters 留空
        rows.append(
            {
                "title": outline.title or f"第{outline.order_index}章",
                "summary": outline.content or "",
                "characters": [],
            }
        )
    return rows


@router.post("", response_model=OutlineResponse, summary="创建大纲")
async def create_outline(
    outline: OutlineCreate,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """创建新的章节大纲（one-to-one模式会自动创建对应章节）"""
    # 验证用户权限
    user_id = getattr(request.state, 'user_id', None)
    project = await verify_project_access(outline.project_id, user_id, db)
    
    # 创建大纲
    db_outline = Outline(**outline.model_dump())
    db.add(db_outline)
    await db.flush()  # 确保大纲有ID
    
    # 如果是one-to-one模式，自动创建对应的章节
    if project.outline_mode == 'one-to-one':
        chapter = Chapter(
            project_id=outline.project_id,
            title=db_outline.title,
            summary=db_outline.content,
            chapter_number=db_outline.order_index,
            sub_index=1,
            outline_id=None,  # one-to-one模式不关联outline_id
            status='pending',
            content=""
        )
        db.add(chapter)
        logger.info(f"一对一模式：为手动创建的大纲 {db_outline.title} (序号{db_outline.order_index}) 自动创建了对应章节")
    
    await db.commit()
    await db.refresh(db_outline)
    return db_outline


@router.get("", response_model=OutlineListResponse, summary="获取大纲列表")
async def get_outlines(
    project_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """获取指定项目的所有大纲（优化版：后端完全解析structure，构建标准JSON返回）"""
    # 验证用户权限
    user_id = getattr(request.state, 'user_id', None)
    await verify_project_access(project_id, user_id, db)
    
    # 获取总数
    count_result = await db.execute(
        select(func.count(Outline.id)).where(Outline.project_id == project_id)
    )
    total = count_result.scalar_one()
    
    # 获取大纲列表
    result = await db.execute(
        select(Outline)
        .where(Outline.project_id == project_id)
        .order_by(Outline.order_index)
    )
    outlines = result.scalars().all()
    
    # 🔧 优化：后端完全解析structure，提取所有字段填充到outline对象
    for outline in outlines:
        if outline.structure:
            try:
                structure_data = json.loads(outline.structure)
                
                # 从structure中提取所有字段填充到outline对象
                outline.title = structure_data.get("title", f"第{outline.order_index}章")
                outline.content = structure_data.get("summary") or structure_data.get("content", "")
                
                # structure字段保持不变，供前端使用其他字段（如characters、scenes等）
                
            except json.JSONDecodeError:
                logger.warning(f"解析大纲 {outline.id} 的structure失败")
                outline.title = f"第{outline.order_index}章"
                outline.content = "解析失败"
        else:
            # 没有structure的异常情况
            outline.title = f"第{outline.order_index}章"
            outline.content = "暂无内容"
    
    return OutlineListResponse(total=total, items=outlines)


@router.get("/project/{project_id}", response_model=OutlineListResponse, summary="获取项目的所有大纲")
async def get_project_outlines(
    project_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """获取指定项目的所有大纲（路径参数版本，兼容旧API）"""
    return await get_outlines(project_id, request, db)


@router.get("/{outline_id}", response_model=OutlineResponse, summary="获取大纲详情")
async def get_outline(
    outline_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """根据ID获取大纲详情"""
    result = await db.execute(
        select(Outline).where(Outline.id == outline_id)
    )
    outline = result.scalar_one_or_none()
    
    if not outline:
        raise HTTPException(status_code=404, detail="大纲不存在")
    
    # 验证用户权限
    user_id = getattr(request.state, 'user_id', None)
    await verify_project_access(outline.project_id, user_id, db)
    
    return outline


@router.put("/{outline_id}", response_model=OutlineResponse, summary="更新大纲")
async def update_outline(
    outline_id: str,
    outline_update: OutlineUpdate,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """更新大纲信息并同步更新structure字段和关联章节"""
    result = await db.execute(
        select(Outline).where(Outline.id == outline_id)
    )
    outline = result.scalar_one_or_none()
    
    if not outline:
        raise HTTPException(status_code=404, detail="大纲不存在")
    
    # 验证用户权限
    user_id = getattr(request.state, 'user_id', None)
    project = await verify_project_access(outline.project_id, user_id, db)
    
    # 更新字段
    update_data = outline_update.model_dump(exclude_unset=True)
    
    # 🔧 特殊处理：如果直接传递了structure字段，优先使用它
    if 'structure' in update_data:
        # 直接使用前端传递的structure（前端已经处理好了完整的JSON）
        outline.structure = update_data['structure']
        logger.info(f"直接更新大纲 {outline_id} 的structure字段")
        # 从update_data中移除structure，避免后续重复处理
        structure_updated = True
        del update_data['structure']
    else:
        structure_updated = False
    
    # 更新其他字段
    for field, value in update_data.items():
        setattr(outline, field, value)
    
    # 如果没有直接更新structure，但修改了content或title，则同步更新structure字段
    if not structure_updated and ('content' in update_data or 'title' in update_data):
        try:
            # 尝试解析现有的structure
            if outline.structure:
                structure_data = json.loads(outline.structure)
            else:
                structure_data = {}
            
            # 更新structure中的对应字段
            if 'title' in update_data:
                structure_data['title'] = outline.title
            if 'content' in update_data:
                structure_data['summary'] = outline.content
                structure_data['content'] = outline.content
            
            # 保存更新后的structure
            outline.structure = json.dumps(structure_data, ensure_ascii=False)
            logger.info(f"同步更新大纲 {outline_id} 的structure字段")
        except json.JSONDecodeError:
            logger.warning(f"大纲 {outline_id} 的structure字段格式错误，跳过更新")
    
    # 🔧 传统模式（one-to-one）：同步更新关联章节的标题
    if 'title' in update_data and project.outline_mode == 'one-to-one':
        try:
            # 查找对应的章节（通过chapter_number匹配order_index）
            chapter_result = await db.execute(
                select(Chapter).where(
                    Chapter.project_id == outline.project_id,
                    Chapter.chapter_number == outline.order_index
                )
            )
            chapter = chapter_result.scalar_one_or_none()
            
            if chapter:
                # 同步更新章节标题
                chapter.title = outline.title
                logger.info(f"一对一模式：同步更新章节 {chapter.id} 的标题为 '{outline.title}'")
            else:
                logger.debug(f"一对一模式：未找到对应的章节（chapter_number={outline.order_index}）")
        except Exception as e:
            logger.error(f"同步更新章节标题失败: {str(e)}")
            # 不阻断大纲更新流程，仅记录错误
    
    await db.commit()
    await db.refresh(outline)
    return outline


@router.delete("/{outline_id}", summary="删除大纲")
async def delete_outline(
    outline_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """删除大纲，同时删除该大纲对应的所有章节和相关的伏笔数据"""
    result = await db.execute(
        select(Outline).where(Outline.id == outline_id)
    )
    outline = result.scalar_one_or_none()
    
    if not outline:
        raise HTTPException(status_code=404, detail="大纲不存在")
    
    # 验证用户权限
    user_id = getattr(request.state, 'user_id', None)
    project = await verify_project_access(outline.project_id, user_id, db)
    
    project_id = outline.project_id
    deleted_order = outline.order_index
    
    # 获取要删除的章节并计算总字数
    deleted_word_count = 0
    deleted_foreshadow_count = 0
    if project.outline_mode == 'one-to-one':
        # one-to-one模式：通过chapter_number获取对应章节
        chapters_result = await db.execute(
            select(Chapter).where(
                Chapter.project_id == project_id,
                Chapter.chapter_number == outline.order_index
            )
        )
        chapters_to_delete = chapters_result.scalars().all()
        deleted_word_count = sum(ch.word_count or 0 for ch in chapters_to_delete)
        
        # 🔮 清理章节相关的伏笔数据和向量记忆
        for chapter in chapters_to_delete:
            try:
                # 清理向量数据库中的记忆数据
                await memory_service.delete_chapter_memories(
                    user_id=user_id,
                    project_id=project_id,
                    chapter_id=chapter.id
                )
                logger.info(f"✅ 已清理章节 {chapter.id[:8]} 的向量记忆数据")
            except Exception as e:
                logger.warning(f"⚠️ 清理章节 {chapter.id[:8]} 向量记忆失败: {str(e)}")
            
            try:
                # 清理伏笔数据（分析来源的伏笔）
                foreshadow_result = await foreshadow_service.delete_chapter_foreshadows(
                    db=db,
                    project_id=project_id,
                    chapter_id=chapter.id,
                    only_analysis_source=True
                )
                deleted_foreshadow_count += foreshadow_result.get('deleted_count', 0)
                if foreshadow_result.get('deleted_count', 0) > 0:
                    logger.info(f"🔮 已清理章节 {chapter.id[:8]} 的 {foreshadow_result['deleted_count']} 个伏笔数据")
            except Exception as e:
                logger.warning(f"⚠️ 清理章节 {chapter.id[:8]} 伏笔数据失败: {str(e)}")
        
        # 删除章节
        delete_result = await db.execute(
            delete(Chapter).where(
                Chapter.project_id == project_id,
                Chapter.chapter_number == outline.order_index
            )
        )
        deleted_chapters_count = delete_result.rowcount
        logger.info(f"一对一模式：删除大纲 {outline_id}（序号{outline.order_index}），同时删除了第{outline.order_index}章（{deleted_chapters_count}个章节，{deleted_word_count}字，{deleted_foreshadow_count}个伏笔）")
    else:
        # one-to-many模式：通过outline_id获取关联章节
        chapters_result = await db.execute(
            select(Chapter).where(Chapter.outline_id == outline_id)
        )
        chapters_to_delete = chapters_result.scalars().all()
        deleted_word_count = sum(ch.word_count or 0 for ch in chapters_to_delete)
        
        # 🔮 清理章节相关的伏笔数据和向量记忆
        for chapter in chapters_to_delete:
            try:
                # 清理向量数据库中的记忆数据
                await memory_service.delete_chapter_memories(
                    user_id=user_id,
                    project_id=project_id,
                    chapter_id=chapter.id
                )
                logger.info(f"✅ 已清理章节 {chapter.id[:8]} 的向量记忆数据")
            except Exception as e:
                logger.warning(f"⚠️ 清理章节 {chapter.id[:8]} 向量记忆失败: {str(e)}")
            
            try:
                # 清理伏笔数据（分析来源的伏笔）
                foreshadow_result = await foreshadow_service.delete_chapter_foreshadows(
                    db=db,
                    project_id=project_id,
                    chapter_id=chapter.id,
                    only_analysis_source=True
                )
                deleted_foreshadow_count += foreshadow_result.get('deleted_count', 0)
                if foreshadow_result.get('deleted_count', 0) > 0:
                    logger.info(f"🔮 已清理章节 {chapter.id[:8]} 的 {foreshadow_result['deleted_count']} 个伏笔数据")
            except Exception as e:
                logger.warning(f"⚠️ 清理章节 {chapter.id[:8]} 伏笔数据失败: {str(e)}")
        
        # 删除章节
        delete_result = await db.execute(
            delete(Chapter).where(Chapter.outline_id == outline_id)
        )
        deleted_chapters_count = delete_result.rowcount
        logger.info(f"一对多模式：删除大纲 {outline_id}，同时删除了 {deleted_chapters_count} 个关联章节（{deleted_word_count}字，{deleted_foreshadow_count}个伏笔）")
    
    # 更新项目字数
    if deleted_word_count > 0:
        project.current_words = max(0, project.current_words - deleted_word_count)
        logger.info(f"更新项目字数：减少 {deleted_word_count} 字")
    
    # 删除大纲
    await db.delete(outline)
    
    # 重新排序后续的大纲（序号-1）
    result = await db.execute(
        select(Outline).where(
            Outline.project_id == project_id,
            Outline.order_index > deleted_order
        )
    )
    subsequent_outlines = result.scalars().all()
    
    for o in subsequent_outlines:
        o.order_index -= 1
    
    # 如果是one-to-one模式，还需要重新排序后续章节的chapter_number
    if project.outline_mode == 'one-to-one':
        chapters_result = await db.execute(
            select(Chapter).where(
                Chapter.project_id == project_id,
                Chapter.chapter_number > deleted_order
            ).order_by(Chapter.chapter_number)
        )
        subsequent_chapters = chapters_result.scalars().all()
        
        for ch in subsequent_chapters:
            ch.chapter_number -= 1
        
        logger.info(f"一对一模式：重新排序了 {len(subsequent_chapters)} 个后续章节")
    
    await db.commit()
    
    return {
        "message": "大纲删除成功",
        "deleted_chapters": deleted_chapters_count,
        "deleted_foreshadows": deleted_foreshadow_count
    }




async def _build_outline_continue_context(
    project: Project,
    latest_outlines: List[Outline],
    characters: List[Character],
    chapter_count: int,
    plot_stage: str,
    story_direction: str,
    requirements: str,
    db: AsyncSession
) -> dict:
    """
    构建大纲续写上下文（简化版）
    
    包含内容：
    1. 项目基础信息：title, theme, genre, world_time_period, world_location,
       world_atmosphere, world_rules, narrative_perspective
    2. 最近10章的完整大纲structure（解析JSON转化为文本）
    3. 所有角色的全部信息
    4. 用户输入：chapter_count, plot_stage, story_direction, requirements
    
    Args:
        project: 项目对象
        latest_outlines: 所有已有大纲列表
        characters: 所有角色列表
        chapter_count: 要生成的章节数
        plot_stage: 情节阶段
        story_direction: 故事发展方向
        requirements: 其他要求
        
    Returns:
        包含上下文信息的字典
    """
    context = {
        'project_info': '',
        'recent_outlines': '',
        'characters_info': '',
        'user_input': '',
        'stats': {
            'total_outlines': len(latest_outlines),
            'recent_outlines_count': 0,
            'characters_count': len(characters)
        }
    }
    
    try:
        # 1. 项目基础信息
        project_info_parts = [
            f"【项目基础信息】",
            f"标题：{project.title}",
            f"主题：{project.theme or '未设定'}",
            f"类型：{project.genre or '未设定'}",
            f"时代背景：{project.world_time_period or '未设定'}",
            f"地点设定：{project.world_location or '未设定'}",
            f"氛围基调：{project.world_atmosphere or '未设定'}",
            f"世界规则：{project.world_rules or '未设定'}",
            f"叙事视角：{project.narrative_perspective or '第三人称'}"
        ]
        context['project_info'] = "\n".join(project_info_parts)
        
        # 2. 最近10章的完整大纲structure（解析JSON转化为文本）
        recent_count = min(10, len(latest_outlines))
        if recent_count > 0:
            recent_outlines = latest_outlines[-recent_count:]
            context['stats']['recent_outlines_count'] = recent_count
            
            outline_texts = []
            outline_texts.append(f"【最近{recent_count}章大纲详情】")
            
            for outline in recent_outlines:
                outline_text = f"\n第{outline.order_index}章《{outline.title}》"
                
                # 尝试解析structure字段
                if outline.structure:
                    try:
                        structure_data = json.loads(outline.structure)
                        
                        # 提取各个字段（使用实际存储的字段名）
                        if structure_data.get('summary'):
                            outline_text += f"\n  概要：{structure_data['summary']}"
                        
                        # key_points 对应 关键事件
                        if structure_data.get('key_points'):
                            events = structure_data['key_points']
                            if isinstance(events, list):
                                outline_text += f"\n  关键事件：{', '.join(events)}"
                            else:
                                outline_text += f"\n  关键事件：{events}"
                        
                        # characters 对应 重点角色/组织（兼容新旧格式）
                        if structure_data.get('characters'):
                            chars = structure_data['characters']
                            if isinstance(chars, list):
                                # 新格式：[{"name": "xxx", "type": "character"/"organization"}]
                                # 旧格式：["角色名1", "角色名2"]
                                char_names = []
                                org_names = []
                                for c in chars:
                                    if isinstance(c, dict):
                                        name = c.get('name', '')
                                        if c.get('type') == 'organization':
                                            org_names.append(name)
                                        else:
                                            char_names.append(name)
                                    elif isinstance(c, str):
                                        char_names.append(c)
                                if char_names:
                                    outline_text += f"\n  重点角色：{', '.join(char_names)}"
                                if org_names:
                                    outline_text += f"\n  涉及组织：{', '.join(org_names)}"
                            else:
                                outline_text += f"\n  重点角色：{chars}"
                        
                        # emotion 对应 情感基调
                        if structure_data.get('emotion'):
                            outline_text += f"\n  情感基调：{structure_data['emotion']}"
                        
                        # goal 对应 叙事目标
                        if structure_data.get('goal'):
                            outline_text += f"\n  叙事目标：{structure_data['goal']}"
                        
                        # scenes 场景信息（可选显示）
                        if structure_data.get('scenes'):
                            scenes = structure_data['scenes']
                            if isinstance(scenes, list) and scenes:
                                outline_text += f"\n  场景：{', '.join(scenes)}"
                            
                    except json.JSONDecodeError:
                        # 如果解析失败，使用content字段
                        outline_text += f"\n  内容：{outline.content}"
                else:
                    # 没有structure，使用content
                    outline_text += f"\n  内容：{outline.content}"
                
                outline_texts.append(outline_text)
            
            context['recent_outlines'] = "\n".join(outline_texts)
            logger.info(f"  ✅ 最近大纲：{recent_count}章")
        
        # 3. 所有角色的全部信息(包括职业信息)
        if characters:
            from app.models.career import Career, CharacterCareer
            
            char_texts = []
            char_texts.append("【角色信息】")
            
            for char in characters:
                char_text = f"\n{char.name}（{'组织' if char.is_organization else '角色'}，{char.role_type}）"
                
                if char.personality:
                    char_text += f"\n  性格特点：{char.personality}"
                
                if char.background:
                    char_text += f"\n  背景故事：{char.background}"
                
                if char.appearance:
                    char_text += f"\n  外貌描述：{char.appearance}"
                
                if char.traits:
                    char_text += f"\n  特征标签：{char.traits}"
                
                # 从 character_relationships 表查询关系
                from sqlalchemy import or_
                rels_result = await db.execute(
                    select(CharacterRelationship).where(
                        CharacterRelationship.project_id == project.id,
                        or_(
                            CharacterRelationship.character_from_id == char.id,
                            CharacterRelationship.character_to_id == char.id
                        )
                    )
                )
                rels = rels_result.scalars().all()
                if rels:
                    # 收集相关角色名称
                    related_ids = set()
                    for r in rels:
                        related_ids.add(r.character_from_id)
                        related_ids.add(r.character_to_id)
                    related_ids.discard(char.id)
                    if related_ids:
                        names_result = await db.execute(
                            select(Character.id, Character.name).where(Character.id.in_(related_ids))
                        )
                        name_map = {row.id: row.name for row in names_result}
                        rel_parts = []
                        for r in rels:
                            if r.character_from_id == char.id:
                                target_name = name_map.get(r.character_to_id, "未知")
                            else:
                                target_name = name_map.get(r.character_from_id, "未知")
                            rel_name = r.relationship_name or "相关"
                            rel_parts.append(f"与{target_name}：{rel_name}")
                        char_text += f"\n  关系网络：{'；'.join(rel_parts)}"
                
                # 组织特有字段
                if char.is_organization:
                    if char.organization_type:
                        char_text += f"\n  组织类型：{char.organization_type}"
                    if char.organization_purpose:
                        char_text += f"\n  组织宗旨：{char.organization_purpose}"
                    # 从 OrganizationMember 表动态查询组织成员
                    org_result = await db.execute(
                        select(Organization).where(Organization.character_id == char.id)
                    )
                    org = org_result.scalar_one_or_none()
                    if org:
                        members_result = await db.execute(
                            select(OrganizationMember, Character.name).join(
                                Character, OrganizationMember.character_id == Character.id
                            ).where(OrganizationMember.organization_id == org.id)
                        )
                        members = members_result.all()
                        if members:
                            member_parts = [f"{name}（{m.position}）" for m, name in members]
                            char_text += f"\n  组织成员：{'、'.join(member_parts)}"
                
                # 查询角色的职业信息
                if not char.is_organization:
                    try:
                        career_result = await db.execute(
                            select(Career, CharacterCareer)
                            .join(CharacterCareer, Career.id == CharacterCareer.career_id)
                            .where(CharacterCareer.character_id == char.id)
                        )
                        career_data = career_result.first()
                        
                        if career_data:
                            career, char_career = career_data
                            char_text += f"\n  职业：{career.name}"
                            if char_career.current_stage:
                                char_text += f"（{char_career.current_stage}阶段）"
                            if char_career.career_type:
                                char_text += f"\n  职业类型：{char_career.career_type}"
                    except Exception as e:
                        logger.warning(f"查询角色 {char.name} 的职业信息失败: {str(e)}")
                
                char_texts.append(char_text)
            
            context['characters_info'] = "\n".join(char_texts)
            logger.info(f"  ✅ 角色信息：{len(characters)}个角色")
        else:
            context['characters_info'] = "【角色信息】\n暂无角色信息"
        
        # 4. 用户输入
        user_input_parts = [
            "【用户输入】",
            f"要生成章节数：{chapter_count}章",
            f"情节阶段：{plot_stage}",
            f"故事发展方向：{story_direction}",
        ]
        if requirements:
            user_input_parts.append(f"其他要求：{requirements}")
        
        context['user_input'] = "\n".join(user_input_parts)
        
        # 计算总长度
        total_length = sum([
            len(context['project_info']),
            len(context['recent_outlines']),
            len(context['characters_info']),
            len(context['user_input'])
        ])
        context['stats']['total_length'] = total_length
        logger.info(f"📊 大纲续写上下文总长度: {total_length} 字符")
        
    except Exception as e:
        logger.error(f"❌ 构建大纲续写上下文失败: {str(e)}", exc_info=True)
    
    return context


async def _check_and_create_missing_characters_from_outlines(
    outline_data: list,
    project_id: str,
    db: AsyncSession,
    user_ai_service: AIService,
    user_id: str = None,
    enable_mcp: bool = True,
    tracker = None
) -> dict:
    """
    大纲生成/续写后，校验structure中的characters是否存在对应角色，
    不存在的自动根据大纲摘要生成角色信息。
    
    Args:
        outline_data: 大纲数据列表（原始JSON解析后的数据，包含characters、summary等字段）
        project_id: 项目ID
        db: 数据库会话
        user_ai_service: AI服务实例
        user_id: 用户ID
        enable_mcp: 是否启用MCP
        tracker: 可选，WizardProgressTracker用于发送进度
        
    Returns:
        {
          "success": bool,
          "created_count": int,
          "created_characters": list,
          "missing_names": list,
          "error": Optional[str]
        }
    """
    try:
        from app.services.auto_character_service import get_auto_character_service
        
        auto_char_service = get_auto_character_service(user_ai_service)
        
        # 定义进度回调
        async def progress_cb(message: str):
            if tracker:
                # 注意：这里不能直接yield，需要通过其他方式处理
                logger.info(f"  📌 {message}")
        
        result = await auto_char_service.check_and_create_missing_characters(
            project_id=project_id,
            outline_data_list=outline_data,
            db=db,
            user_id=user_id,
            enable_mcp=enable_mcp,
            progress_callback=progress_cb
        )

        result.setdefault("created_count", 0)
        result.setdefault("created_characters", [])
        result.setdefault("missing_names", [])
        result["success"] = True
        result["error"] = None
        
        if result["created_count"] > 0:
            logger.info(
                f"🎭 【角色校验完成】自动创建了 {result['created_count']} 个缺失角色: "
                f"{', '.join(c.name for c in result['created_characters'])}"
            )
        
        return result
        
    except Exception as e:
        logger.error(f"⚠️ 【角色校验】校验失败（不影响主流程）: {e}", exc_info=True)
        return {
            "success": False,
            "created_count": 0,
            "created_characters": [],
            "missing_names": [],
            "error": str(e),
        }


async def _check_and_create_missing_organizations_from_outlines(
    outline_data: list,
    project_id: str,
    db: AsyncSession,
    user_ai_service: AIService,
    user_id: str = None,
    enable_mcp: bool = True,
    tracker = None
) -> dict:
    """
    大纲生成/续写后，校验structure中的characters（type=organization）是否存在对应组织，
    不存在的自动根据大纲摘要生成组织信息。
    
    Args:
        outline_data: 大纲数据列表（原始JSON解析后的数据，包含characters、summary等字段）
        project_id: 项目ID
        db: 数据库会话
        user_ai_service: AI服务实例
        user_id: 用户ID
        enable_mcp: 是否启用MCP
        tracker: 可选，WizardProgressTracker用于发送进度
        
    Returns:
        {
          "success": bool,
          "created_count": int,
          "created_organizations": list,
          "missing_names": list,
          "error": Optional[str]
        }
    """
    try:
        from app.services.auto_organization_service import get_auto_organization_service
        
        auto_org_service = get_auto_organization_service(user_ai_service)
        
        # 定义进度回调
        async def progress_cb(message: str):
            if tracker:
                logger.info(f"  📌 {message}")
        
        result = await auto_org_service.check_and_create_missing_organizations(
            project_id=project_id,
            outline_data_list=outline_data,
            db=db,
            user_id=user_id,
            enable_mcp=enable_mcp,
            progress_callback=progress_cb
        )

        result.setdefault("created_count", 0)
        result.setdefault("created_organizations", [])
        result.setdefault("missing_names", [])
        result["success"] = True
        result["error"] = None
        
        if result["created_count"] > 0:
            logger.info(
                f"🏛️ 【组织校验完成】自动创建了 {result['created_count']} 个缺失组织: "
                f"{', '.join(c.name for c in result['created_organizations'])}"
            )
        
        return result
        
    except Exception as e:
        logger.error(f"⚠️ 【组织校验】校验失败（不影响主流程）: {e}", exc_info=True)
        return {
            "success": False,
            "created_count": 0,
            "created_organizations": [],
            "missing_names": [],
            "error": str(e),
        }


class JSONParseError(Exception):
    """JSON解析失败异常，用于触发重试"""
    def __init__(self, message: str, original_content: str = ""):
        super().__init__(message)
        self.original_content = original_content


def _parse_ai_response(ai_response: str, raise_on_error: bool = False) -> list:
    """
    解析AI响应为章节数据列表（使用统一的JSON清洗方法）
    
    Args:
        ai_response: AI返回的原始文本
        raise_on_error: 如果为True，解析失败时抛出异常而不是返回fallback数据
        
    Returns:
        解析后的章节数据列表
        
    Raises:
        JSONParseError: 当raise_on_error=True且解析失败时抛出
    """
    try:
        # 使用统一的JSON清洗方法（从AIService导入）
        from app.services.ai_service import AIService
        ai_service_temp = AIService()
        cleaned_text = ai_service_temp._clean_json_response(ai_response)
        
        outline_data = json.loads(cleaned_text)
        
        # 确保是列表格式
        if not isinstance(outline_data, list):
            # 如果是对象，尝试提取chapters字段
            if isinstance(outline_data, dict):
                outline_data = outline_data.get("chapters", [outline_data])
            else:
                outline_data = [outline_data]
        
        # 验证解析结果是否有效（至少有一个有效章节）
        valid_chapters = [
            ch for ch in outline_data
            if isinstance(ch, dict) and (ch.get("title") or ch.get("summary") or ch.get("content"))
        ]
        
        if not valid_chapters:
            error_msg = "解析结果无效：未找到有效的章节数据"
            logger.error(f"❌ {error_msg}")
            if raise_on_error:
                raise JSONParseError(error_msg, ai_response)
            return [{
                "title": "AI生成的大纲",
                "content": ai_response[:1000],
                "summary": ai_response[:1000]
            }]
        
        logger.info(f"✅ 成功解析 {len(valid_chapters)} 个章节数据")
        return valid_chapters
        
    except json.JSONDecodeError as e:
        error_msg = f"JSON解析失败: {e}"
        logger.error(f"❌ AI响应解析失败: {e}")
        
        if raise_on_error:
            raise JSONParseError(error_msg, ai_response)
        
        # 返回一个包含原始内容的章节
        return [{
            "title": "AI生成的大纲",
            "content": ai_response[:1000],
            "summary": ai_response[:1000]
        }]
    except JSONParseError:
        # 重新抛出JSONParseError
        raise
    except Exception as e:
        error_msg = f"解析异常: {str(e)}"
        logger.error(f"❌ {error_msg}")
        
        if raise_on_error:
            raise JSONParseError(error_msg, ai_response)
        
        return [{
            "title": "解析异常的大纲",
            "content": "系统错误",
            "summary": "系统错误"
        }]


async def _save_outlines(
    project_id: str,
    outline_data: list,
    db: AsyncSession,
    start_index: int = 1
) -> List[Outline]:
    """
    保存大纲到数据库（修复版：从structure中提取title和content保存到数据库）
    
    如果项目为one-to-one模式，同时自动创建对应的章节
    """
    # 获取项目信息以确定outline_mode
    project_result = await db.execute(
        select(Project).where(Project.id == project_id)
    )
    project = project_result.scalar_one_or_none()
    
    outlines = []
    
    for idx, chapter_data in enumerate(outline_data):
        order_idx = chapter_data.get("chapter_number", start_index + idx)
        
        # 🔧 修复：从structure中提取title和summary/content保存到数据库
        chapter_title = chapter_data.get("title", f"第{order_idx}章")
        chapter_content = chapter_data.get("summary") or chapter_data.get("content", "")
        
        outline = Outline(
            project_id=project_id,
            title=chapter_title,  # 从JSON中提取title
            content=chapter_content,  # 从JSON中提取summary或content
            structure=json.dumps(chapter_data, ensure_ascii=False),
            order_index=order_idx
        )
        db.add(outline)
        outlines.append(outline)
    
    # 如果是one-to-one模式，自动创建章节
    if project and project.outline_mode == 'one-to-one':
        await db.flush()  # 确保大纲有ID
        
        for outline in outlines:
            await db.refresh(outline)
            
            # 🔧 从structure中提取title和summary用于创建章节
            try:
                structure_data = json.loads(outline.structure) if outline.structure else {}
                chapter_title = structure_data.get("title", f"第{outline.order_index}章")
                chapter_summary = structure_data.get("summary") or structure_data.get("content", "")
            except json.JSONDecodeError:
                logger.warning(f"解析大纲 {outline.id} 的structure失败，使用默认值")
                chapter_title = f"第{outline.order_index}章"
                chapter_summary = ""
            
            # 为每个大纲创建对应的章节
            chapter = Chapter(
                project_id=project_id,
                title=chapter_title,
                summary=chapter_summary,
                chapter_number=outline.order_index,
                sub_index=1,
                outline_id=None,  # one-to-one模式不关联outline_id
                status='pending',
                content=""
            )
            db.add(chapter)
        
        logger.info(f"一对一模式：为{len(outlines)}个大纲自动创建了对应的章节")
    
    return outlines


async def new_outline_generator(
    data: Dict[str, Any],
    db: AsyncSession,
    user_ai_service: AIService
) -> AsyncGenerator[str, None]:
    """全新生成大纲SSE生成器（MCP增强版）"""
    db_committed = False
    # 初始化标准进度追踪器
    tracker = WizardProgressTracker("大纲")
    
    try:
        yield await tracker.start()
        
        project_id = data.get("project_id")
        # 确保chapter_count是整数（前端可能传字符串）
        chapter_count = int(data.get("chapter_count", 10))
        enable_mcp = data.get("enable_mcp", True)
        
        # 验证项目
        yield await tracker.loading("加载项目信息...", 0.3)
        result = await db.execute(
            select(Project).where(Project.id == project_id)
        )
        project = result.scalar_one_or_none()
        if not project:
            yield await tracker.error("项目不存在", 404)
            return
        
        yield await tracker.loading(f"准备生成{chapter_count}章大纲...", 0.6)
        
        # 获取角色信息
        characters_result = await db.execute(
            select(Character).where(Character.project_id == project_id)
        )
        characters = characters_result.scalars().all()
        characters_info = _build_characters_info(characters)
        
        # 设置用户信息以启用MCP
        user_id_for_mcp = data.get("user_id")
        if user_id_for_mcp:
            user_ai_service.user_id = user_id_for_mcp
            user_ai_service.db_session = db
        
        # 使用提示词模板
        yield await tracker.preparing("准备AI提示词...")
        template = await PromptService.get_template("OUTLINE_CREATE", user_id_for_mcp, db)
        prompt = PromptService.format_prompt(
            template,
            title=project.title,
            theme=data.get("theme") or project.theme or "未设定",
            genre=data.get("genre") or project.genre or "通用",
            chapter_count=chapter_count,
            narrative_perspective=data.get("narrative_perspective") or "第三人称",
            time_period=project.world_time_period or "未设定",
            location=project.world_location or "未设定",
            atmosphere=project.world_atmosphere or "未设定",
            rules=project.world_rules or "未设定",
            characters_info=characters_info or "暂无角色信息",
            requirements=data.get("requirements") or "",
            mcp_references=""
        )
        logger.debug(f"NEW提示词: {prompt}")
        # 添加调试日志
        model_param = data.get("model")
        provider_param = data.get("provider")
        logger.info(f"=== 大纲生成AI调用参数 ===")
        logger.info(f"  provider参数: {provider_param}")
        logger.info(f"  model参数: {model_param}")
        
        # ✅ 流式生成（带字数统计和进度）
        estimated_total = chapter_count * 1000
        accumulated_text = ""
        chunk_count = 0
        
        yield await tracker.generating(current_chars=0, estimated_total=estimated_total)
        
        async for chunk in user_ai_service.generate_text_stream(
            prompt=prompt,
            provider=provider_param,
            model=model_param
        ):
            chunk_count += 1
            accumulated_text += chunk
            
            # 发送内容块
            yield await tracker.generating_chunk(chunk)
            
            # 定期更新进度
            if chunk_count % 10 == 0:
                yield await tracker.generating(
                    current_chars=len(accumulated_text),
                    estimated_total=estimated_total
                )
            
            # 每20个块发送心跳
            if chunk_count % 20 == 0:
                yield await tracker.heartbeat()
        
        yield await tracker.parsing("解析大纲数据...")
        
        ai_content = accumulated_text
        ai_response = {"content": ai_content}
        
        # 解析响应（带重试机制）
        max_retries = 2
        retry_count = 0
        outline_data = None
        
        while retry_count <= max_retries:
            try:
                # 使用 raise_on_error=True，解析失败时抛出异常
                outline_data = _parse_ai_response(ai_content, raise_on_error=True)
                break  # 解析成功，跳出循环
                
            except JSONParseError as e:
                retry_count += 1
                if retry_count > max_retries:
                    # 超过最大重试次数，使用fallback数据
                    logger.error(f"❌ 大纲解析失败，已达最大重试次数({max_retries})，使用fallback数据")
                    yield await tracker.warning("解析失败，使用备用数据")
                    outline_data = _parse_ai_response(ai_content, raise_on_error=False)
                    break
                
                logger.warning(f"⚠️ JSON解析失败（第{retry_count}次），正在重试...")
                yield await tracker.retry(retry_count, max_retries, "JSON解析失败")
                
                # 重试时重置生成进度
                tracker.reset_generating_progress()
                
                # 重新调用AI生成
                accumulated_text = ""
                chunk_count = 0
                
                # 在prompt中添加格式强调
                retry_prompt = prompt + "\n\n【重要提醒】请确保返回完整的JSON数组，不要截断。每个章节对象必须包含完整的title、summary等字段。"
                
                async for chunk in user_ai_service.generate_text_stream(
                    prompt=retry_prompt,
                    provider=provider_param,
                    model=model_param
                ):
                    chunk_count += 1
                    accumulated_text += chunk
                    
                    # 发送内容块
                    yield await tracker.generating_chunk(chunk)
                    
                    # 每20个块发送心跳
                    if chunk_count % 20 == 0:
                        yield await tracker.heartbeat()
                
                ai_content = accumulated_text
                ai_response = {"content": ai_content}
                logger.info(f"🔄 重试生成完成，累计{len(ai_content)}字符")
        
        # 全新生成模式：删除旧大纲和关联的所有章节、伏笔、分析数据
        yield await tracker.saving("清理旧数据（大纲、章节、伏笔、分析）...", 0.2)
        logger.info(f"🧹 全新生成：开始清理项目 {project_id} 的所有旧数据（outline_mode: {project.outline_mode}）")
        
        from sqlalchemy import delete as sql_delete
        
        # 1. 先获取所有旧章节ID（用于后续清理）
        old_chapters_result = await db.execute(
            select(Chapter).where(Chapter.project_id == project_id)
        )
        old_chapters = old_chapters_result.scalars().all()
        old_chapter_ids = [ch.id for ch in old_chapters]
        deleted_word_count = sum(ch.word_count or 0 for ch in old_chapters)
        
        # 2. 清理伏笔数据（删除分析伏笔，重置手动伏笔）
        try:
            foreshadow_result = await foreshadow_service.clear_project_foreshadows_for_reset(db, project_id)
            logger.info(f"✅ 伏笔清理: 删除 {foreshadow_result['deleted_count']} 个分析伏笔, 重置 {foreshadow_result['reset_count']} 个手动伏笔")
        except Exception as e:
            logger.error(f"❌ 清理伏笔数据失败: {str(e)}")
            # 继续流程，但记录错误
        
        # 3. 清理章节分析数据（PlotAnalysis）
        try:
            # 虽然有CASCADE删除，但显式删除更可控
            from app.models.memory import PlotAnalysis
            delete_analysis_result = await db.execute(
                sql_delete(PlotAnalysis).where(PlotAnalysis.project_id == project_id)
            )
            deleted_analysis_count = delete_analysis_result.rowcount
            logger.info(f"✅ 章节分析清理: 删除 {deleted_analysis_count} 个分析记录")
        except Exception as e:
            logger.error(f"❌ 清理章节分析数据失败: {str(e)}")
        
        # 4. 清理向量记忆数据（StoryMemory）
        try:
            from app.models.memory import StoryMemory
            delete_memory_result = await db.execute(
                sql_delete(StoryMemory).where(StoryMemory.project_id == project_id)
            )
            deleted_memory_count = delete_memory_result.rowcount
            if deleted_memory_count > 0:
                logger.info(f"✅ 向量记忆清理: 删除 {deleted_memory_count} 条记忆数据")
        except Exception as e:
            logger.error(f"❌ 清理向量记忆数据失败: {str(e)}")
        
        # 5. 删除向量数据库中的记忆（如果有章节）
        if old_chapter_ids:
            try:
                user_id_for_memory = data.get("user_id")
                if user_id_for_memory:
                    for chapter_id in old_chapter_ids:
                        try:
                            await memory_service.delete_chapter_memories(
                                user_id=user_id_for_memory,
                                project_id=project_id,
                                chapter_id=chapter_id
                            )
                        except Exception as mem_err:
                            logger.debug(f"清理章节 {chapter_id[:8]} 向量记忆失败: {str(mem_err)}")
                    logger.info(f"✅ 向量数据库清理: 已清理 {len(old_chapter_ids)} 个章节的向量记忆")
            except Exception as e:
                logger.warning(f"⚠️ 清理向量数据库失败（不影响主流程）: {str(e)}")
        
        # 6. 删除所有旧章节
        delete_chapters_result = await db.execute(
            sql_delete(Chapter).where(Chapter.project_id == project_id)
        )
        deleted_chapters_count = delete_chapters_result.rowcount
        logger.info(f"✅ 章节清理: 删除 {deleted_chapters_count} 个章节（{deleted_word_count}字）")
        
        # 更新项目字数
        if deleted_word_count > 0:
            project.current_words = max(0, project.current_words - deleted_word_count)
            logger.info(f"更新项目字数：减少 {deleted_word_count} 字")
        
        # 再删除所有旧大纲
        delete_outlines_result = await db.execute(
            sql_delete(Outline).where(Outline.project_id == project_id)
        )
        deleted_outlines_count = delete_outlines_result.rowcount
        logger.info(f"✅ 全新生成：删除了 {deleted_outlines_count} 个旧大纲")
        
        # 保存新大纲
        yield await tracker.saving("保存大纲到数据库...", 0.6)
        outlines = await _save_outlines(
            project_id, outline_data, db, start_index=1
        )

        char_check_result: Dict[str, Any] = {
            "success": True,
            "created_count": 0,
            "created_characters": [],
            "missing_names": [],
            "error": None,
        }
        org_check_result: Dict[str, Any] = {
            "success": True,
            "created_count": 0,
            "created_organizations": [],
            "missing_names": [],
            "error": None,
        }
        
        # 🎭 角色校验：检查大纲structure中的characters是否存在对应角色
        yield await tracker.saving("🎭 校验角色信息...", 0.7)
        try:
            char_check_result = await _check_and_create_missing_characters_from_outlines(
                outline_data=outline_data,
                project_id=project_id,
                db=db,
                user_ai_service=user_ai_service,
                user_id=data.get("user_id"),
                enable_mcp=data.get("enable_mcp", True),
                tracker=tracker
            )
            if not char_check_result.get("success", True):
                error_msg = char_check_result.get("error") or "未知错误"
                yield await tracker.warning(f"角色校验失败（主流程继续）: {error_msg}")
            elif char_check_result["created_count"] > 0:
                created_names = [c.name for c in char_check_result["created_characters"]]
                yield await tracker.saving(
                    f"🎭 自动创建了 {char_check_result['created_count']} 个角色: {', '.join(created_names)}",
                    0.8
                )
        except Exception as e:
            logger.error(f"⚠️ 角色校验失败（不影响主流程）: {e}")
            char_check_result = {
                "success": False,
                "created_count": 0,
                "created_characters": [],
                "missing_names": [],
                "error": str(e),
            }
            yield await tracker.warning(f"角色校验失败（主流程继续）: {e}")
        
        # 🏛️ 组织校验：检查大纲structure中的characters（type=organization）是否存在对应组织
        yield await tracker.saving("🏛️ 校验组织信息...", 0.75)
        try:
            org_check_result = await _check_and_create_missing_organizations_from_outlines(
                outline_data=outline_data,
                project_id=project_id,
                db=db,
                user_ai_service=user_ai_service,
                user_id=data.get("user_id"),
                enable_mcp=data.get("enable_mcp", True),
                tracker=tracker
            )
            if not org_check_result.get("success", True):
                error_msg = org_check_result.get("error") or "未知错误"
                yield await tracker.warning(f"组织校验失败（主流程继续）: {error_msg}")
            elif org_check_result["created_count"] > 0:
                created_names = [c.name for c in org_check_result["created_organizations"]]
                yield await tracker.saving(
                    f"🏛️ 自动创建了 {org_check_result['created_count']} 个组织: {', '.join(created_names)}",
                    0.85
                )
        except Exception as e:
            logger.error(f"⚠️ 组织校验失败（不影响主流程）: {e}")
            org_check_result = {
                "success": False,
                "created_count": 0,
                "created_organizations": [],
                "missing_names": [],
                "error": str(e),
            }
            yield await tracker.warning(f"组织校验失败（主流程继续）: {e}")

        postcheck_summary = _build_postcheck_summary(
            scope="all",
            char_result=char_check_result,
            org_result=org_check_result,
        )
        
        # 记录历史
        history = GenerationHistory(
            project_id=project_id,
            prompt=prompt,
            generated_content=json.dumps(ai_response, ensure_ascii=False) if isinstance(ai_response, dict) else ai_response,
            model=data.get("model") or "default"
        )
        db.add(history)
        
        await db.commit()
        db_committed = True
        
        for outline in outlines:
            await db.refresh(outline)
        
        logger.info(f"全新生成完成 - {len(outlines)} 章")
        
        yield await tracker.complete()
        
        # 发送最终结果
        yield await tracker.result({
            "message": f"成功生成{len(outlines)}章大纲",
            "total_chapters": len(outlines),
            "postcheck": postcheck_summary,
            "outlines": [
                {
                    "id": outline.id,
                    "project_id": outline.project_id,
                    "title": outline.title,
                    "content": outline.content,
                    "order_index": outline.order_index,
                    "structure": outline.structure,
                    "created_at": outline.created_at.isoformat() if outline.created_at else None,
                    "updated_at": outline.updated_at.isoformat() if outline.updated_at else None
                } for outline in outlines
            ]
        })
        
        yield await tracker.done()
        
    except GeneratorExit:
        logger.warning("大纲生成器被提前关闭")
        if not db_committed and db.in_transaction():
            await db.rollback()
            logger.info("大纲生成事务已回滚（GeneratorExit）")
    except Exception as e:
        logger.error(f"大纲生成失败: {str(e)}")
        if not db_committed and db.in_transaction():
            await db.rollback()
            logger.info("大纲生成事务已回滚（异常）")
        yield await tracker.error(f"生成失败: {str(e)}")


async def continue_outline_generator(
    data: Dict[str, Any],
    db: AsyncSession,
    user_ai_service: AIService,
    user_id: str = "system"
) -> AsyncGenerator[str, None]:
    """大纲续写SSE生成器 - 分批生成，推送进度（记忆+MCP增强版）"""
    db_committed = False
    # 初始化标准进度追踪器
    tracker = WizardProgressTracker("大纲续写")
    
    try:
        # === 初始化阶段 ===
        yield await tracker.start("开始续写大纲...")
        
        project_id = data.get("project_id")
        # 确保chapter_count是整数（前端可能传字符串）
        total_chapters_to_generate = int(data.get("chapter_count", 5))
        
        # 验证项目
        yield await tracker.loading("加载项目信息...", 0.2)
        result = await db.execute(
            select(Project).where(Project.id == project_id)
        )
        project = result.scalar_one_or_none()
        if not project:
            yield await tracker.error("项目不存在", 404)
            return
        
        # 获取现有大纲
        yield await tracker.loading("分析已有大纲...", 0.5)
        existing_result = await db.execute(
            select(Outline)
            .where(Outline.project_id == project_id)
            .order_by(Outline.order_index)
        )
        existing_outlines = existing_result.scalars().all()
        
        if not existing_outlines:
            yield await tracker.error("续写模式需要已有大纲，当前项目没有大纲", 400)
            return
        
        current_chapter_count = len(existing_outlines)
        last_chapter_number = existing_outlines[-1].order_index
        
        yield await tracker.loading(
            f"当前已有{str(current_chapter_count)}章，将续写{str(total_chapters_to_generate)}章",
            0.8
        )
        
        # 获取角色信息
        characters_result = await db.execute(
            select(Character).where(Character.project_id == project_id)
        )
        characters = characters_result.scalars().all()
        characters_info = _build_characters_info(characters)

        # 分批配置
        batch_size = 5
        total_batches = (total_chapters_to_generate + batch_size - 1) // batch_size
        
        # 情节阶段指导
        stage_instructions = {
            "development": "继续展开情节，深化角色关系，推进主线冲突",
            "climax": "进入故事高潮，矛盾激化，关键冲突爆发",
            "ending": "解决主要冲突，收束伏笔，给出结局"
        }
        stage_instruction = stage_instructions.get(data.get("plot_stage", "development"), "")
        
        # === 批次生成阶段 ===
        all_new_outlines = []
        current_start_chapter = last_chapter_number + 1
        postcheck_batches: List[Dict[str, Any]] = []
        all_created_character_names: List[str] = []
        all_created_org_names: List[str] = []
        char_errors: List[str] = []
        org_errors: List[str] = []
        
        for batch_num in range(total_batches):
            # 计算当前批次的章节数
            remaining_chapters = int(total_chapters_to_generate) - len(all_new_outlines)
            current_batch_size = min(batch_size, remaining_chapters)
            
            # 每批使用的进度预估
            estimated_chars_per_batch = current_batch_size * 1000
            
            # 重置生成进度以便于每批独立计算
            tracker.reset_generating_progress()
            
            yield await tracker.generating(
                current_chars=0,
                estimated_total=estimated_chars_per_batch,
                message=f"📝 第{str(batch_num + 1)}/{str(total_batches)}批: 生成第{str(current_start_chapter)}-{str(current_start_chapter + current_batch_size - 1)}章"
            )
            
            # 获取最新的大纲列表（包括之前批次生成的）
            latest_result = await db.execute(
                select(Outline)
                .where(Outline.project_id == project_id)
                .order_by(Outline.order_index)
            )
            latest_outlines = latest_result.scalars().all()
            
            # 🚀 使用新的简化上下文构建
            context = await _build_outline_continue_context(
                project=project,
                latest_outlines=latest_outlines,
                characters=characters,
                chapter_count=current_batch_size,
                plot_stage=data.get("plot_stage", "development"),
                story_direction=data.get("story_direction", "自然延续"),
                requirements=data.get("requirements", ""),
                db=db
            )
            
            # 日志统计
            stats = context['stats']
            logger.info(f"📊 批次{batch_num + 1}大纲上下文: 总大纲{stats['total_outlines']}, "
                       f"最近{stats['recent_outlines_count']}章, "
                       f"角色{stats['characters_count']}个, "
                       f"长度{stats['total_length']}字符")
            
            # 设置用户信息以启用MCP
            if user_id:
                user_ai_service.user_id = user_id
                user_ai_service.db_session = db
            
            yield await tracker.generating(
                current_chars=0,
                estimated_total=estimated_chars_per_batch,
                message=f"🤖 调用AI生成第{str(batch_num + 1)}批..."
            )
            
            # 使用标准续写提示词模板（简化版）
            template = await PromptService.get_template("OUTLINE_CONTINUE", user_id, db)
            prompt = PromptService.format_prompt(
                template,
                # 基础信息
                title=project.title,
                theme=project.theme or "未设定",
                genre=project.genre or "通用",
                narrative_perspective=project.narrative_perspective or "第三人称",
                time_period=project.world_time_period or "未设定",
                location=project.world_location or "未设定",
                atmosphere=project.world_atmosphere or "未设定",
                rules=project.world_rules or "未设定",
                # 上下文信息
                recent_outlines=context['recent_outlines'],
                characters_info=context['characters_info'],
                # 续写参数
                chapter_count=current_batch_size,
                start_chapter=current_start_chapter,
                end_chapter=current_start_chapter + current_batch_size - 1,
                current_chapter_count=len(latest_outlines),
                plot_stage_instruction=stage_instruction,
                story_direction=data.get("story_direction", "自然延续"),
                requirements=data.get("requirements", ""),
                mcp_references=""
            )
            logger.debug(f" 续写提示词: {prompt}")
            # 调用AI生成当前批次
            model_param = data.get("model")
            provider_param = data.get("provider")
            logger.info(f"=== 续写批次{batch_num + 1} AI调用参数 ===")
            logger.info(f"  provider参数: {provider_param}")
            logger.info(f"  model参数: {model_param}")
            
            # 流式生成并累积文本
            accumulated_text = ""
            chunk_count = 0
            
            async for chunk in user_ai_service.generate_text_stream(
                prompt=prompt,
                provider=provider_param,
                model=model_param
            ):
                chunk_count += 1
                accumulated_text += chunk
                
                # 发送内容块
                yield await tracker.generating_chunk(chunk)
                
                # 定期更新进度
                if chunk_count % 10 == 0:
                    yield await tracker.generating(
                        current_chars=len(accumulated_text),
                        estimated_total=estimated_chars_per_batch,
                        message=f"📝 第{str(batch_num + 1)}/{str(total_batches)}批生成中"
                    )
                
                # 每20个块发送心跳
                if chunk_count % 20 == 0:
                    yield await tracker.heartbeat()
            
            yield await tracker.parsing(f"✅ 第{str(batch_num + 1)}批AI生成完成，正在解析...")
            
            # 提取内容
            ai_content = accumulated_text
            ai_response = {"content": ai_content}
            
            # 解析响应（带重试机制）
            max_retries = 2
            retry_count = 0
            outline_data = None
            
            while retry_count <= max_retries:
                try:
                    # 使用 raise_on_error=True，解析失败时抛出异常
                    outline_data = _parse_ai_response(ai_content, raise_on_error=True)
                    break  # 解析成功，跳出循环
                    
                except JSONParseError as e:
                    retry_count += 1
                    if retry_count > max_retries:
                        # 超过最大重试次数，使用fallback数据
                        logger.error(f"❌ 第{batch_num + 1}批解析失败，已达最大重试次数({max_retries})，使用fallback数据")
                        yield await tracker.warning(f"第{str(batch_num + 1)}批解析失败，使用备用数据")
                        outline_data = _parse_ai_response(ai_content, raise_on_error=False)
                        break
                    
                    logger.warning(f"⚠️ 第{batch_num + 1}批JSON解析失败（第{retry_count}次），正在重试...")
                    yield await tracker.retry(retry_count, max_retries, f"第{str(batch_num + 1)}批解析失败")
                    
                    # 重试时重置生成进度
                    tracker.reset_generating_progress()
                    
                    # 重新调用AI生成
                    accumulated_text = ""
                    chunk_count = 0
                    
                    # 在prompt中添加格式强调
                    retry_prompt = prompt + "\n\n【重要提醒】请确保返回完整的JSON数组，不要截断。每个章节对象必须包含完整的title、summary等字段。"
                    
                    async for chunk in user_ai_service.generate_text_stream(
                        prompt=retry_prompt,
                        provider=provider_param,
                        model=model_param
                    ):
                        chunk_count += 1
                        accumulated_text += chunk
                        
                        # 发送内容块
                        yield await tracker.generating_chunk(chunk)
                        
                        # 每20个块发送心跳
                        if chunk_count % 20 == 0:
                            yield await tracker.heartbeat()
                    
                    ai_content = accumulated_text
                    ai_response = {"content": ai_content}
                    logger.info(f"🔄 第{batch_num + 1}批重试生成完成，累计{len(ai_content)}字符")
            
            # 保存当前批次的大纲
            batch_outlines = await _save_outlines(
                project_id, outline_data, db, start_index=current_start_chapter
            )
            
            # 🎭 角色校验：检查本批大纲structure中的characters是否存在对应角色
            char_check_result: Dict[str, Any] = {
                "success": True,
                "created_count": 0,
                "created_characters": [],
                "missing_names": [],
                "error": None,
            }
            try:
                char_check_result = await _check_and_create_missing_characters_from_outlines(
                    outline_data=outline_data,
                    project_id=project_id,
                    db=db,
                    user_ai_service=user_ai_service,
                    user_id=user_id,
                    enable_mcp=data.get("enable_mcp", True),
                    tracker=tracker
                )
                if not char_check_result.get("success", True):
                    error_msg = char_check_result.get("error") or "未知错误"
                    char_errors.append(f"第{batch_num + 1}批: {error_msg}")
                    yield await tracker.warning(f"🎭 第{batch_num + 1}批角色校验失败（主流程继续）: {error_msg}")
                elif char_check_result["created_count"] > 0:
                    created_names = _extract_created_names(char_check_result["created_characters"])
                    yield await tracker.saving(
                        f"🎭 第{str(batch_num + 1)}批：自动创建了 {char_check_result['created_count']} 个角色: {', '.join(created_names)}",
                        (batch_num + 1) / total_batches * 0.5
                    )
                    all_created_character_names.extend(created_names)
                    # 更新角色列表（供后续批次使用）
                    characters.extend(char_check_result["created_characters"])
                    characters_info = _build_characters_info(characters)
            except Exception as e:
                logger.error(f"⚠️ 第{batch_num + 1}批角色校验失败（不影响主流程）: {e}")
                char_check_result = {
                    "success": False,
                    "created_count": 0,
                    "created_characters": [],
                    "missing_names": [],
                    "error": str(e),
                }
                char_errors.append(f"第{batch_num + 1}批: {e}")
                yield await tracker.warning(f"🎭 第{batch_num + 1}批角色校验失败（主流程继续）: {e}")
            
            # 🏛️ 组织校验：检查本批大纲structure中的characters（type=organization）是否存在对应组织
            org_check_result: Dict[str, Any] = {
                "success": True,
                "created_count": 0,
                "created_organizations": [],
                "missing_names": [],
                "error": None,
            }
            try:
                org_check_result = await _check_and_create_missing_organizations_from_outlines(
                    outline_data=outline_data,
                    project_id=project_id,
                    db=db,
                    user_ai_service=user_ai_service,
                    user_id=user_id,
                    enable_mcp=data.get("enable_mcp", True),
                    tracker=tracker
                )
                if not org_check_result.get("success", True):
                    error_msg = org_check_result.get("error") or "未知错误"
                    org_errors.append(f"第{batch_num + 1}批: {error_msg}")
                    yield await tracker.warning(f"🏛️ 第{batch_num + 1}批组织校验失败（主流程继续）: {error_msg}")
                elif org_check_result["created_count"] > 0:
                    created_names = _extract_created_names(org_check_result["created_organizations"])
                    yield await tracker.saving(
                        f"🏛️ 第{str(batch_num + 1)}批：自动创建了 {org_check_result['created_count']} 个组织: {', '.join(created_names)}",
                        (batch_num + 1) / total_batches * 0.55
                    )
                    all_created_org_names.extend(created_names)
                    # 更新角色列表（组织也是Character，供后续批次使用）
                    characters.extend(org_check_result["created_organizations"])
                    characters_info = _build_characters_info(characters)
            except Exception as e:
                logger.error(f"⚠️ 第{batch_num + 1}批组织校验失败（不影响主流程）: {e}")
                org_check_result = {
                    "success": False,
                    "created_count": 0,
                    "created_organizations": [],
                    "missing_names": [],
                    "error": str(e),
                }
                org_errors.append(f"第{batch_num + 1}批: {e}")
                yield await tracker.warning(f"🏛️ 第{batch_num + 1}批组织校验失败（主流程继续）: {e}")

            postcheck_batches.append(
                {
                    "batch": batch_num + 1,
                    "characters": _normalize_postcheck_item(
                        section="characters",
                        raw_result=char_check_result,
                    ),
                    "organizations": _normalize_postcheck_item(
                        section="organizations",
                        raw_result=org_check_result,
                    ),
                }
            )
            
            # 记录历史
            history = GenerationHistory(
                project_id=project_id,
                prompt=f"[续写批次{batch_num + 1}/{total_batches}] {str(prompt)[:500]}",
                generated_content=json.dumps(ai_response, ensure_ascii=False) if isinstance(ai_response, dict) else ai_response,
                model=data.get("model") or "default"
            )
            db.add(history)
            
            # 提交当前批次
            await db.commit()
            
            for outline in batch_outlines:
                await db.refresh(outline)
            
            all_new_outlines.extend(batch_outlines)
            current_start_chapter += current_batch_size
            
            yield await tracker.saving(
                f"💾 第{str(batch_num + 1)}批保存成功！本批生成{str(len(batch_outlines))}章，累计新增{str(len(all_new_outlines))}章",
                (batch_num + 1) / total_batches
            )
            
            logger.info(f"第{str(batch_num + 1)}批生成完成，本批生成{str(len(batch_outlines))}章")
        
        db_committed = True
        
        # 返回所有大纲（包括旧的和新的）
        final_result = await db.execute(
            select(Outline)
            .where(Outline.project_id == project_id)
            .order_by(Outline.order_index)
        )
        all_outlines = final_result.scalars().all()

        unique_char_names = list(dict.fromkeys(all_created_character_names))
        unique_org_names = list(dict.fromkeys(all_created_org_names))

        char_agg_result = {
            "success": len(char_errors) == 0,
            "created_count": len(unique_char_names),
            "created_characters": [{"name": n} for n in unique_char_names],
            "missing_names": [],
            "error": "; ".join(char_errors) if char_errors else None,
        }
        org_agg_result = {
            "success": len(org_errors) == 0,
            "created_count": len(unique_org_names),
            "created_organizations": [{"name": n} for n in unique_org_names],
            "missing_names": [],
            "error": "; ".join(org_errors) if org_errors else None,
        }
        postcheck_summary = _build_postcheck_summary(
            scope="all",
            char_result=char_agg_result,
            org_result=org_agg_result,
        )
        postcheck_summary["batches"] = postcheck_batches
        
        yield await tracker.complete()
        
        # 发送最终结果
        yield await tracker.result({
            "message": f"续写完成！共{str(total_batches)}批，新增{str(len(all_new_outlines))}章，总计{str(len(all_outlines))}章",
            "total_batches": total_batches,
            "new_chapters": len(all_new_outlines),
            "total_chapters": len(all_outlines),
            "postcheck": postcheck_summary,
            "outlines": [
                {
                    "id": outline.id,
                    "project_id": outline.project_id,
                    "title": outline.title,
                    "content": outline.content,
                    "order_index": outline.order_index,
                    "structure": outline.structure,
                    "created_at": outline.created_at.isoformat() if outline.created_at else None,
                    "updated_at": outline.updated_at.isoformat() if outline.updated_at else None
                } for outline in all_outlines
            ]
        })
        
        yield await tracker.done()
        
    except GeneratorExit:
        logger.warning("大纲续写生成器被提前关闭")
        if not db_committed and db.in_transaction():
            await db.rollback()
            logger.info("大纲续写事务已回滚（GeneratorExit）")
    except Exception as e:
        logger.error(f"大纲续写失败: {str(e)}")
        if not db_committed and db.in_transaction():
            await db.rollback()
            logger.info("大纲续写事务已回滚（异常）")
        yield await tracker.error(f"续写失败: {str(e)}")


@router.post("/generate-stream", summary="AI生成/续写大纲(SSE流式)")
async def generate_outline_stream(
    data: Dict[str, Any],
    request: Request,
    db: AsyncSession = Depends(get_db),
    user_ai_service: AIService = Depends(get_user_ai_service_for_task("outline_generate"))
):
    """
    使用SSE流式生成或续写小说大纲，实时推送批次进度
    
    支持模式：
    - auto: 自动判断（无大纲→新建，有大纲→续写）
    - new: 全新生成
    - continue: 续写模式
    
    请求体示例：
    {
        "project_id": "项目ID",
        "chapter_count": 5,  // 章节数
        "mode": "auto",  // auto/new/continue
        "theme": "故事主题",  // new模式必需
        "story_direction": "故事发展方向",  // continue模式可选
        "plot_stage": "development",  // continue模式：development/climax/ending
        "narrative_perspective": "第三人称",
        "requirements": "其他要求",
        "provider": "openai",  // 可选
        "model": "gpt-4"  // 可选
    }
    """
    # 验证用户权限
    user_id = getattr(request.state, 'user_id', None)
    project = await verify_project_access(data.get("project_id"), user_id, db)
    
    # 判断模式
    mode = data.get("mode", "auto")
    
    # 获取现有大纲
    existing_result = await db.execute(
        select(Outline)
        .where(Outline.project_id == data.get("project_id"))
        .order_by(Outline.order_index)
    )
    existing_outlines = existing_result.scalars().all()
    
    # 自动判断模式
    if mode == "auto":
        mode = "continue" if existing_outlines else "new"
        logger.info(f"自动判断模式：{'续写' if existing_outlines else '新建'}")
    
    # 获取用户ID
    user_id = getattr(request.state, "user_id", "system")
    
    # 根据模式选择生成器
    if mode == "new":
        return create_sse_response(new_outline_generator(data, db, user_ai_service))
    elif mode == "continue":
        if not existing_outlines:
            raise HTTPException(
                status_code=400,
                detail="续写模式需要已有大纲，当前项目没有大纲"
            )
        return create_sse_response(continue_outline_generator(data, db, user_ai_service, user_id))
    else:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的模式: {mode}"
        )


async def outline_postcheck_generator(
    data: Dict[str, Any],
    db: AsyncSession,
    user_ai_service: AIService,
    user_id: str = "system",
) -> AsyncGenerator[str, None]:
    """
    大纲后处理补全（角色/组织）SSE生成器。

    入参：
    - project_id: 项目ID（必填）
    - scope: all / characters / organizations（默认 all）
    - outline_ids: 可选，仅对指定大纲进行后处理
    - enable_mcp: 是否启用 MCP（默认 true）
    """
    db_committed = False
    tracker = WizardProgressTracker("大纲后处理")

    try:
        yield await tracker.start("开始执行大纲后处理补全...")

        project_id = data.get("project_id")
        if not project_id:
            yield await tracker.error("project_id 是必需参数", 400)
            return

        scope_raw = str(data.get("scope", "all") or "all").strip().lower()
        if scope_raw not in {"all", "characters", "organizations"}:
            yield await tracker.error("scope 仅支持 all / characters / organizations", 400)
            return

        outline_ids = data.get("outline_ids")
        if outline_ids is not None and not isinstance(outline_ids, list):
            yield await tracker.error("outline_ids 必须是数组", 400)
            return
        outline_ids = [str(x) for x in (outline_ids or []) if str(x).strip()]

        # 验证项目存在
        yield await tracker.loading("加载项目信息...", 0.3)
        project_result = await db.execute(select(Project).where(Project.id == project_id))
        project = project_result.scalar_one_or_none()
        if not project:
            yield await tracker.error("项目不存在", 404)
            return

        # 读取目标大纲
        yield await tracker.loading("读取目标大纲...", 0.6)
        q = select(Outline).where(Outline.project_id == project_id)
        if outline_ids:
            q = q.where(Outline.id.in_(outline_ids))
        q = q.order_by(Outline.order_index)
        outlines_result = await db.execute(q)
        outlines = list(outlines_result.scalars().all())

        if not outlines:
            yield await tracker.error("未找到可处理的大纲", 404)
            return

        outline_data = _extract_outline_data_for_postcheck(outlines)
        enable_mcp = data.get("enable_mcp", True)

        if user_id:
            user_ai_service.user_id = user_id
            user_ai_service.db_session = db

        char_result: Optional[Dict[str, Any]] = None
        org_result: Optional[Dict[str, Any]] = None

        if scope_raw in {"all", "characters"}:
            yield await tracker.saving("🎭 校验并补全角色...", 0.72)
            char_result = await _check_and_create_missing_characters_from_outlines(
                outline_data=outline_data,
                project_id=project_id,
                db=db,
                user_ai_service=user_ai_service,
                user_id=user_id,
                enable_mcp=enable_mcp,
                tracker=tracker,
            )
            if not char_result.get("success", True):
                yield await tracker.warning(
                    f"角色补全失败（主流程继续）: {char_result.get('error') or '未知错误'}"
                )
            elif int(char_result.get("created_count", 0) or 0) > 0:
                names = _extract_created_names(char_result.get("created_characters"))
                yield await tracker.saving(
                    f"🎭 自动创建了 {char_result.get('created_count', 0)} 个角色: {', '.join(names)}",
                    0.8,
                )

        if scope_raw in {"all", "organizations"}:
            yield await tracker.saving("🏛️ 校验并补全组织/地点...", 0.82)
            org_result = await _check_and_create_missing_organizations_from_outlines(
                outline_data=outline_data,
                project_id=project_id,
                db=db,
                user_ai_service=user_ai_service,
                user_id=user_id,
                enable_mcp=enable_mcp,
                tracker=tracker,
            )
            if not org_result.get("success", True):
                yield await tracker.warning(
                    f"组织补全失败（主流程继续）: {org_result.get('error') or '未知错误'}"
                )
            elif int(org_result.get("created_count", 0) or 0) > 0:
                names = _extract_created_names(org_result.get("created_organizations"))
                yield await tracker.saving(
                    f"🏛️ 自动创建了 {org_result.get('created_count', 0)} 个组织: {', '.join(names)}",
                    0.88,
                )

        await db.commit()
        db_committed = True

        postcheck_summary = _build_postcheck_summary(
            scope=scope_raw,
            char_result=char_result,
            org_result=org_result,
        )

        # 额外返回项目当前的角色/组织统计
        chars_result = await db.execute(
            select(Character).where(Character.project_id == project_id)
        )
        all_chars = list(chars_result.scalars().all())
        total_characters = len([c for c in all_chars if not c.is_organization])
        total_organizations = len([c for c in all_chars if c.is_organization])

        yield await tracker.complete("大纲后处理补全完成")
        yield await tracker.result(
            {
                "message": "大纲后处理补全完成",
                "project_id": project_id,
                "outline_count": len(outlines),
                "scope": scope_raw,
                "postcheck": postcheck_summary,
                "project_snapshot": {
                    "total_characters": total_characters,
                    "total_organizations": total_organizations,
                },
            }
        )
        yield await tracker.done()

    except GeneratorExit:
        logger.warning("大纲后处理生成器被提前关闭")
        if not db_committed and db.in_transaction():
            await db.rollback()
            logger.info("大纲后处理事务已回滚（GeneratorExit）")
    except Exception as e:
        logger.error(f"大纲后处理失败: {str(e)}", exc_info=True)
        if not db_committed and db.in_transaction():
            await db.rollback()
            logger.info("大纲后处理事务已回滚（异常）")
        yield await tracker.error(f"后处理失败: {str(e)}")


@router.post("/postcheck-stream", summary="大纲后处理补全（角色/组织，SSE流式）")
async def outline_postcheck_stream(
    data: Dict[str, Any],
    request: Request,
    db: AsyncSession = Depends(get_db),
    user_ai_service: AIService = Depends(get_user_ai_service_for_task("outline_generate")),
):
    """手动触发大纲后处理补全，支持按 scope 选择角色/组织。"""
    user_id = getattr(request.state, "user_id", None)
    project_id = data.get("project_id")
    await verify_project_access(project_id, user_id, db)
    return create_sse_response(outline_postcheck_generator(data, db, user_ai_service, user_id or "system"))


async def expand_outline_generator(
    outline_id: str,
    data: Dict[str, Any],
    db: AsyncSession,
    user_ai_service: AIService
) -> AsyncGenerator[str, None]:
    """单个大纲展开SSE生成器 - 实时推送进度（支持分批生成）"""
    db_committed = False
    # 初始化标准进度追踪器
    tracker = WizardProgressTracker("大纲展开")
    
    try:
        yield await tracker.start()
        
        target_chapter_count = int(data.get("target_chapter_count", 3))
        expansion_strategy = data.get("expansion_strategy", "balanced")
        enable_scene_analysis = data.get("enable_scene_analysis", True)
        auto_create_chapters = data.get("auto_create_chapters", False)
        batch_size = int(data.get("batch_size", 5))  # 支持自定义批次大小
        
        # 获取大纲
        yield await tracker.loading("加载大纲信息...", 0.3)
        result = await db.execute(
            select(Outline).where(Outline.id == outline_id)
        )
        outline = result.scalar_one_or_none()
        
        if not outline:
            yield await tracker.error("大纲不存在", 404)
            return
        
        # 获取项目信息
        yield await tracker.loading("加载项目信息...", 0.7)
        project_result = await db.execute(
            select(Project).where(Project.id == outline.project_id)
        )
        project = project_result.scalar_one_or_none()
        if not project:
            yield await tracker.error("项目不存在", 404)
            return
        
        yield await tracker.preparing(
            f"准备展开《{outline.title}》为 {target_chapter_count} 章..."
        )
        
        # 创建展开服务实例
        expansion_service = PlotExpansionService(user_ai_service)
        
        # 分析大纲并生成章节规划（支持分批）
        if target_chapter_count > batch_size:
            yield await tracker.generating(
                current_chars=0,
                estimated_total=target_chapter_count * 500,
                message=f"🤖 AI分批生成章节规划（每批{batch_size}章）..."
            )
        else:
            yield await tracker.generating(
                current_chars=0,
                estimated_total=target_chapter_count * 500,
                message="🤖 AI分析大纲，生成章节规划..."
            )
        
        chapter_plans = await expansion_service.analyze_outline_for_chapters(
            outline=outline,
            project=project,
            db=db,
            target_chapter_count=target_chapter_count,
            expansion_strategy=expansion_strategy,
            enable_scene_analysis=enable_scene_analysis,
            provider=data.get("provider"),
            model=data.get("model"),
            batch_size=batch_size,
            progress_callback=None  # SSE中暂不支持嵌套回调
        )
        
        if not chapter_plans:
            yield await tracker.error("AI分析失败，未能生成章节规划", 500)
            return
        
        yield await tracker.parsing(
            f"✅ 规划生成完成！共 {len(chapter_plans)} 个章节"
        )
        
        # 根据配置决定是否创建章节记录
        created_chapters = None
        if auto_create_chapters:
            yield await tracker.saving("💾 创建章节记录...", 0.3)
            
            created_chapters = await expansion_service.create_chapters_from_plans(
                outline_id=outline_id,
                chapter_plans=chapter_plans,
                project_id=outline.project_id,
                db=db,
                start_chapter_number=None  # 自动计算章节序号
            )
            
            await db.commit()
            db_committed = True
            
            # 刷新章节数据
            for chapter in created_chapters:
                await db.refresh(chapter)
            
            yield await tracker.saving(
                f"✅ 成功创建 {len(created_chapters)} 个章节记录",
                0.8
            )
        
        yield await tracker.complete()
        
        # 构建响应数据
        result_data = {
            "outline_id": outline_id,
            "outline_title": outline.title,
            "target_chapter_count": target_chapter_count,
            "actual_chapter_count": len(chapter_plans),
            "expansion_strategy": expansion_strategy,
            "chapter_plans": chapter_plans,
            "created_chapters": [
                {
                    "id": ch.id,
                    "chapter_number": ch.chapter_number,
                    "title": ch.title,
                    "summary": ch.summary,
                    "outline_id": ch.outline_id,
                    "sub_index": ch.sub_index,
                    "status": ch.status
                }
                for ch in created_chapters
            ] if created_chapters else None
        }
        
        yield await tracker.result(result_data)
        yield await tracker.done()
        
    except GeneratorExit:
        logger.warning("大纲展开生成器被提前关闭")
        if not db_committed and db.in_transaction():
            await db.rollback()
            logger.info("大纲展开事务已回滚（GeneratorExit）")
    except Exception as e:
        logger.error(f"大纲展开失败: {str(e)}")
        if not db_committed and db.in_transaction():
            await db.rollback()
            logger.info("大纲展开事务已回滚（异常）")
        yield await tracker.error(f"展开失败: {str(e)}")


@router.post("/{outline_id}/create-single-chapter", summary="一对一创建章节(传统模式)")
async def create_single_chapter_from_outline(
    outline_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """
    传统模式：一个大纲对应创建一个章节
    
    适用场景：
    - 项目的outline_mode为'one-to-one'
    - 直接将大纲内容作为章节摘要
    - 不调用AI，不展开
    
    流程：
    1. 验证项目模式为one-to-one
    2. 检查该大纲是否已创建章节
    3. 创建章节记录（outline_id=NULL，chapter_number=outline.order_index）
    
    返回：创建的章节信息
    """
    # 验证用户权限
    user_id = getattr(request.state, 'user_id', None)
    
    # 获取大纲
    result = await db.execute(
        select(Outline).where(Outline.id == outline_id)
    )
    outline = result.scalar_one_or_none()
    
    if not outline:
        raise HTTPException(status_code=404, detail="大纲不存在")
    
    # 验证项目权限并获取项目信息
    project = await verify_project_access(outline.project_id, user_id, db)
    
    # 验证项目模式
    if project.outline_mode != 'one-to-one':
        raise HTTPException(
            status_code=400,
            detail=f"当前项目为{project.outline_mode}模式，不支持一对一创建。请使用展开功能。"
        )
    
    # 检查该大纲对应的章节是否已存在
    existing_chapter_result = await db.execute(
        select(Chapter).where(
            Chapter.project_id == outline.project_id,
            Chapter.chapter_number == outline.order_index,
            Chapter.sub_index == 1
        )
    )
    existing_chapter = existing_chapter_result.scalar_one_or_none()
    
    if existing_chapter:
        raise HTTPException(
            status_code=400,
            detail=f"第{outline.order_index}章已存在，不能重复创建"
        )
    
    try:
        # 创建章节（outline_id=NULL表示一对一模式）
        new_chapter = Chapter(
            project_id=outline.project_id,
            title=outline.title,
            summary=outline.content,  # 使用大纲内容作为摘要
            chapter_number=outline.order_index,
            sub_index=1,  # 一对一模式固定为1
            outline_id=None,  # 传统模式不关联outline_id
            status='pending'
        )
        
        db.add(new_chapter)
        await db.commit()
        await db.refresh(new_chapter)
        
        logger.info(f"一对一模式：为大纲 {outline.title} 创建章节 {new_chapter.chapter_number}")
        
        return {
            "message": "章节创建成功",
            "chapter": {
                "id": new_chapter.id,
                "project_id": new_chapter.project_id,
                "title": new_chapter.title,
                "summary": new_chapter.summary,
                "chapter_number": new_chapter.chapter_number,
                "sub_index": new_chapter.sub_index,
                "outline_id": new_chapter.outline_id,
                "status": new_chapter.status,
                "created_at": new_chapter.created_at.isoformat() if new_chapter.created_at else None
            }
        }
        
    except Exception as e:
        logger.error(f"一对一创建章节失败: {str(e)}", exc_info=True)
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"创建章节失败: {str(e)}")


@router.post("/{outline_id}/expand-stream", summary="展开单个大纲为多章(SSE流式)")
async def expand_outline_to_chapters_stream(
    outline_id: str,
    data: Dict[str, Any],
    request: Request,
    db: AsyncSession = Depends(get_db),
    user_ai_service: AIService = Depends(get_user_ai_service_for_task("outline_expand"))
):
    """
    使用SSE流式展开单个大纲，实时推送进度
    
    请求体示例：
    {
        "target_chapter_count": 3,  // 目标章节数
        "expansion_strategy": "balanced",  // balanced/climax/detail
        "auto_create_chapters": false,  // 是否自动创建章节
        "enable_scene_analysis": true,  // 是否启用场景分析
        "provider": "openai",  // 可选
        "model": "gpt-4"  // 可选
    }
    
    进度阶段：
    - 5% - 开始展开
    - 10% - 加载大纲信息
    - 15% - 加载项目信息
    - 20% - 准备展开参数
    - 30% - AI分析大纲（耗时）
    - 70% - 规划生成完成
    - 80% - 创建章节记录（如果auto_create_chapters=True）
    - 90% - 创建完成
    - 95% - 整理结果数据
    - 100% - 全部完成
    """
    # 获取大纲并验证权限
    result = await db.execute(
        select(Outline).where(Outline.id == outline_id)
    )
    outline = result.scalar_one_or_none()
    
    if not outline:
        raise HTTPException(status_code=404, detail="大纲不存在")
    
    # 验证用户权限
    user_id = getattr(request.state, 'user_id', None)
    await verify_project_access(outline.project_id, user_id, db)
    
    return create_sse_response(expand_outline_generator(outline_id, data, db, user_ai_service))


@router.get("/{outline_id}/chapters", summary="获取大纲关联的章节")
async def get_outline_chapters(
    outline_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """
    获取指定大纲已展开的章节列表
    
    用于检查大纲是否已经展开过,如果有则返回章节信息
    """
    # 获取大纲
    result = await db.execute(
        select(Outline).where(Outline.id == outline_id)
    )
    outline = result.scalar_one_or_none()
    
    if not outline:
        raise HTTPException(status_code=404, detail="大纲不存在")
    
    # 验证用户权限
    user_id = getattr(request.state, 'user_id', None)
    await verify_project_access(outline.project_id, user_id, db)
    
    # 查询该大纲关联的章节
    chapters_result = await db.execute(
        select(Chapter)
        .where(Chapter.outline_id == outline_id)
        .order_by(Chapter.sub_index)
    )
    chapters = chapters_result.scalars().all()
    
    # 如果有章节,解析展开规划
    expansion_plans = []
    if chapters:
        for chapter in chapters:
            plan_data = None
            if chapter.expansion_plan:
                try:
                    plan_data = json.loads(chapter.expansion_plan)
                except json.JSONDecodeError:
                    logger.warning(f"章节 {chapter.id} 的expansion_plan解析失败")
                    plan_data = None
            
            expansion_plans.append({
                "sub_index": chapter.sub_index,
                "title": chapter.title,
                "plot_summary": chapter.summary or "",
                "key_events": plan_data.get("key_events", []) if plan_data else [],
                "character_focus": plan_data.get("character_focus", []) if plan_data else [],
                "emotional_tone": plan_data.get("emotional_tone", "") if plan_data else "",
                "narrative_goal": plan_data.get("narrative_goal", "") if plan_data else "",
                "conflict_type": plan_data.get("conflict_type", "") if plan_data else "",
                "estimated_words": plan_data.get("estimated_words", 0) if plan_data else 0,
                "scenes": plan_data.get("scenes") if plan_data else None
            })
    
    return {
        "has_chapters": len(chapters) > 0,
        "outline_id": outline_id,
        "outline_title": outline.title,
        "chapter_count": len(chapters),
        "chapters": [
            {
                "id": ch.id,
                "chapter_number": ch.chapter_number,
                "title": ch.title,
                "summary": ch.summary,
                "sub_index": ch.sub_index,
                "status": ch.status,
                "word_count": ch.word_count
            }
            for ch in chapters
        ],
        "expansion_plans": expansion_plans if expansion_plans else None
    }


async def batch_expand_outlines_generator(
    data: Dict[str, Any],
    db: AsyncSession,
    user_ai_service: AIService
) -> AsyncGenerator[str, None]:
    """批量展开大纲SSE生成器 - 实时推送进度"""
    db_committed = False
    # 初始化标准进度追踪器
    tracker = WizardProgressTracker("批量大纲展开")
    
    try:
        yield await tracker.start()
        
        project_id = data.get("project_id")
        chapters_per_outline = int(data.get("chapters_per_outline", 3))
        expansion_strategy = data.get("expansion_strategy", "balanced")
        auto_create_chapters = data.get("auto_create_chapters", False)
        outline_ids = data.get("outline_ids")
        
        # 获取项目信息
        yield await tracker.loading("加载项目信息...", 0.5)
        project_result = await db.execute(
            select(Project).where(Project.id == project_id)
        )
        project = project_result.scalar_one_or_none()
        if not project:
            yield await tracker.error("项目不存在", 404)
            return
        
        # 获取要展开的大纲列表
        yield await tracker.loading("获取大纲列表...", 0.8)
        if outline_ids:
            outlines_result = await db.execute(
                select(Outline)
                .where(
                    Outline.project_id == project_id,
                    Outline.id.in_(outline_ids)
                )
                .order_by(Outline.order_index)
            )
        else:
            outlines_result = await db.execute(
                select(Outline)
                .where(Outline.project_id == project_id)
                .order_by(Outline.order_index)
            )
        
        outlines = outlines_result.scalars().all()
        
        if not outlines:
            yield await tracker.error("没有找到要展开的大纲", 404)
            return
        
        total_outlines = len(outlines)
        yield await tracker.preparing(
            f"共找到 {total_outlines} 个大纲，开始批量展开..."
        )
        
        # 创建展开服务实例
        expansion_service = PlotExpansionService(user_ai_service)
        
        expansion_results = []
        total_chapters_created = 0
        skipped_outlines = []
        
        for idx, outline in enumerate(outlines):
            try:
                # 计算当前子进度 (0.0-1.0)，用于generating阶段
                sub_progress = idx / max(total_outlines, 1)
                
                yield await tracker.generating(
                    current_chars=idx * chapters_per_outline * 500,
                    estimated_total=total_outlines * chapters_per_outline * 500,
                    message=f"📝 处理第 {idx + 1}/{total_outlines} 个大纲: {outline.title}"
                )
                
                # 检查大纲是否已经展开过
                existing_chapters_result = await db.execute(
                    select(Chapter)
                    .where(Chapter.outline_id == outline.id)
                    .limit(1)
                )
                existing_chapter = existing_chapters_result.scalar_one_or_none()
                
                if existing_chapter:
                    logger.info(f"大纲 {outline.title} (ID: {outline.id}) 已经展开过，跳过")
                    skipped_outlines.append({
                        "outline_id": outline.id,
                        "outline_title": outline.title,
                        "reason": "已展开"
                    })
                    yield await tracker.generating(
                        current_chars=(idx + 1) * chapters_per_outline * 500,
                        estimated_total=total_outlines * chapters_per_outline * 500,
                        message=f"⏭️ {outline.title} 已展开过，跳过"
                    )
                    continue
                
                # 分析大纲生成章节规划
                yield await tracker.generating(
                    current_chars=idx * chapters_per_outline * 500,
                    estimated_total=total_outlines * chapters_per_outline * 500,
                    message=f"🤖 AI分析大纲: {outline.title}"
                )
                
                chapter_plans = await expansion_service.analyze_outline_for_chapters(
                    outline=outline,
                    project=project,
                    db=db,
                    target_chapter_count=chapters_per_outline,
                    expansion_strategy=expansion_strategy,
                    enable_scene_analysis=data.get("enable_scene_analysis", True),
                    provider=data.get("provider"),
                    model=data.get("model")
                )
                
                yield await tracker.generating(
                    current_chars=(idx + 0.5) * chapters_per_outline * 500,
                    estimated_total=total_outlines * chapters_per_outline * 500,
                    message=f"✅ {outline.title} 规划生成完成 ({len(chapter_plans)} 章)"
                )
                
                created_chapters = None
                if auto_create_chapters:
                    # 创建章节记录
                    chapters = await expansion_service.create_chapters_from_plans(
                        outline_id=outline.id,
                        chapter_plans=chapter_plans,
                        project_id=outline.project_id,
                        db=db,
                        start_chapter_number=None  # 自动计算章节序号
                    )
                    created_chapters = [
                        {
                            "id": ch.id,
                            "chapter_number": ch.chapter_number,
                            "title": ch.title,
                            "summary": ch.summary,
                            "outline_id": ch.outline_id,
                            "sub_index": ch.sub_index,
                            "status": ch.status
                        }
                        for ch in chapters
                    ]
                    total_chapters_created += len(chapters)
                    
                    yield await tracker.generating(
                        current_chars=(idx + 1) * chapters_per_outline * 500,
                        estimated_total=total_outlines * chapters_per_outline * 500,
                        message=f"💾 {outline.title} 章节创建完成 ({len(chapters)} 章)"
                    )
                
                expansion_results.append({
                    "outline_id": outline.id,
                    "outline_title": outline.title,
                    "target_chapter_count": chapters_per_outline,
                    "actual_chapter_count": len(chapter_plans),
                    "expansion_strategy": expansion_strategy,
                    "chapter_plans": chapter_plans,
                    "created_chapters": created_chapters
                })
                
                logger.info(f"大纲 {outline.title} 展开完成，生成 {len(chapter_plans)} 个章节规划")
                
            except Exception as e:
                logger.error(f"展开大纲 {outline.id} 失败: {str(e)}", exc_info=True)
                yield await tracker.warning(
                    f"❌ {outline.title} 展开失败: {str(e)}"
                )
                expansion_results.append({
                    "outline_id": outline.id,
                    "outline_title": outline.title,
                    "target_chapter_count": chapters_per_outline,
                    "actual_chapter_count": 0,
                    "expansion_strategy": expansion_strategy,
                    "chapter_plans": [],
                    "created_chapters": None,
                    "error": str(e)
                })
        
        yield await tracker.parsing("整理结果数据...")
        
        db_committed = True
        
        logger.info(f"批量展开完成: {len(expansion_results)} 个大纲，跳过 {len(skipped_outlines)} 个，共生成 {total_chapters_created} 个章节")
        
        yield await tracker.complete()
        
        # 发送最终结果
        result_data = {
            "project_id": project_id,
            "total_outlines_expanded": len(expansion_results),
            "total_chapters_created": total_chapters_created,
            "skipped_count": len(skipped_outlines),
            "skipped_outlines": skipped_outlines,
            "expansion_results": [
                {
                    "outline_id": result["outline_id"],
                    "outline_title": result["outline_title"],
                    "target_chapter_count": result["target_chapter_count"],
                    "actual_chapter_count": result["actual_chapter_count"],
                    "expansion_strategy": result["expansion_strategy"],
                    "chapter_plans": result["chapter_plans"],
                    "created_chapters": result.get("created_chapters")
                }
                for result in expansion_results
            ]
        }
        
        yield await tracker.result(result_data)
        yield await tracker.done()
        
    except GeneratorExit:
        logger.warning("批量展开生成器被提前关闭")
        if not db_committed and db.in_transaction():
            await db.rollback()
            logger.info("批量展开事务已回滚（GeneratorExit）")
    except Exception as e:
        logger.error(f"批量展开失败: {str(e)}")
        if not db_committed and db.in_transaction():
            await db.rollback()
            logger.info("批量展开事务已回滚（异常）")
        yield await SSEResponse.send_error(f"批量展开失败: {str(e)}")


@router.post("/batch-expand-stream", summary="批量展开大纲为多章(SSE流式)")
async def batch_expand_outlines_stream(
    data: Dict[str, Any],
    request: Request,
    db: AsyncSession = Depends(get_db),
    user_ai_service: AIService = Depends(get_user_ai_service_for_task("outline_expand"))
):
    """
    使用SSE流式批量展开大纲，实时推送每个大纲的处理进度
    
    请求体示例：
    {
        "project_id": "项目ID",
        "outline_ids": ["大纲ID1", "大纲ID2"],  // 可选，不传则展开所有大纲
        "chapters_per_outline": 3,  // 每个大纲展开几章
        "expansion_strategy": "balanced",  // balanced/climax/detail
        "auto_create_chapters": false,  // 是否自动创建章节
        "enable_scene_analysis": true,  // 是否启用场景分析
        "provider": "openai",  // 可选
        "model": "gpt-4"  // 可选
    }
    """
    # 验证用户权限
    user_id = getattr(request.state, 'user_id', None)
    await verify_project_access(data.get("project_id"), user_id, db)
    
    return create_sse_response(batch_expand_outlines_generator(data, db, user_ai_service))


@router.post("/{outline_id}/create-chapters-from-plans", response_model=CreateChaptersFromPlansResponse, summary="根据已有规划创建章节")
async def create_chapters_from_existing_plans(
    outline_id: str,
    plans_request: CreateChaptersFromPlansRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    user_ai_service: AIService = Depends(get_user_ai_service_for_task("outline_expand"))
):
    """
    根据前端缓存的章节规划直接创建章节记录，避免重复调用AI
    
    使用场景：
    1. 用户第一次调用 /outlines/{outline_id}/expand?auto_create_chapters=false 获取规划预览
    2. 前端展示规划给用户确认
    3. 用户确认后，前端调用此接口，传递缓存的规划数据，直接创建章节
    
    优势：
    - 避免重复的AI调用，节省Token和时间
    - 确保用户看到的预览和实际创建的章节完全一致
    - 提升用户体验
    
    参数：
    - outline_id: 要展开的大纲ID
    - plans_request: 包含之前AI生成的章节规划列表
    
    返回：
    - 创建的章节列表和统计信息
    """
    # 验证用户权限
    user_id = getattr(request.state, 'user_id', None)
    
    # 获取大纲
    result = await db.execute(
        select(Outline).where(Outline.id == outline_id)
    )
    outline = result.scalar_one_or_none()
    
    if not outline:
        raise HTTPException(status_code=404, detail="大纲不存在")
    
    # 验证项目权限
    await verify_project_access(outline.project_id, user_id, db)
    
    try:
        # 验证规划数据
        if not plans_request.chapter_plans:
            raise HTTPException(status_code=400, detail="章节规划列表不能为空")
        
        logger.info(f"根据已有规划为大纲 {outline_id} 创建 {len(plans_request.chapter_plans)} 个章节")
        
        # 创建展开服务实例
        expansion_service = PlotExpansionService(user_ai_service)
        
        # 将Pydantic模型转换为字典列表
        chapter_plans_dict = [plan.model_dump() for plan in plans_request.chapter_plans]
        
        # 直接使用传入的规划创建章节记录（不调用AI）
        created_chapters = await expansion_service.create_chapters_from_plans(
            outline_id=outline_id,
            chapter_plans=chapter_plans_dict,
            project_id=outline.project_id,
            db=db,
            start_chapter_number=None  # 自动计算章节序号
        )
        
        await db.commit()
        
        # 刷新章节数据
        for chapter in created_chapters:
            await db.refresh(chapter)
        
        logger.info(f"成功根据已有规划创建 {len(created_chapters)} 个章节记录")
        
        # 构建响应
        return CreateChaptersFromPlansResponse(
            outline_id=outline_id,
            outline_title=outline.title,
            chapters_created=len(created_chapters),
            created_chapters=[
                {
                    "id": ch.id,
                    "chapter_number": ch.chapter_number,
                    "title": ch.title,
                    "summary": ch.summary,
                    "outline_id": ch.outline_id,
                    "sub_index": ch.sub_index,
                    "status": ch.status
                }
                for ch in created_chapters
            ]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"根据已有规划创建章节失败: {str(e)}", exc_info=True)
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"创建章节失败: {str(e)}")
