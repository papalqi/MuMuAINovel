"""章节管理API"""
from fastapi import APIRouter, Depends, HTTPException, Request, Query, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload
import json
import asyncio
from typing import Optional, List, Dict, Any
from datetime import datetime
from asyncio import Queue, Lock

from app.database import get_db
from app.api.common import verify_project_access
from app.services.chapter_context_service import (
    OneToManyContextBuilder,
    OneToOneContextBuilder
)
from app.models.chapter import Chapter
from app.models.project import Project
from app.models.outline import Outline
from app.models.character import Character
from app.models.career import Career, CharacterCareer
from app.models.relationship import CharacterRelationship, Organization, OrganizationMember
from app.models.generation_history import GenerationHistory
from app.models.writing_style import WritingStyle
from app.models.analysis_task import AnalysisTask
from app.models.memory import PlotAnalysis, StoryMemory
from app.models.batch_generation_task import BatchGenerationTask
from app.models.regeneration_task import RegenerationTask
from app.schemas.chapter import (
    ChapterCreate,
    ChapterUpdate,
    ChapterResponse,
    ChapterListResponse,
    ChapterGenerateRequest,
    BatchGenerateRequest,
    BatchGenerateResponse,
    BatchGenerateStatusResponse,
    ExpansionPlanUpdate,
    PartialRegenerateRequest
)
from app.schemas.regeneration import (
    ChapterRegenerateRequest,
    RegenerationTaskResponse,
    RegenerationTaskStatus
)
from app.services.ai_service import AIService
from app.services.prompt_service import prompt_service, PromptService, WritingStyleManager
from app.services.plot_analyzer import PlotAnalyzer
from app.services.memory_service import memory_service
from app.services.foreshadow_service import foreshadow_service
from app.services.chapter_regenerator import ChapterRegenerator
from app.logger import get_logger
from app.api.settings import get_user_ai_service_for_task
from app.utils.sse_response import SSEResponse, create_sse_response

router = APIRouter(prefix="/chapters", tags=["章节管理"])
logger = get_logger(__name__)

# 全局数据库写入锁（每个用户一个锁，用于保护SQLite写入操作）
db_write_locks: dict[str, Lock] = {}


async def get_db_write_lock(user_id: str) -> Lock:
    """获取或创建用户的数据库写入锁"""
    if user_id not in db_write_locks:
        db_write_locks[user_id] = Lock()
        logger.debug(f"🔒 为用户 {user_id} 创建数据库写入锁")
    return db_write_locks[user_id]


@router.post("", response_model=ChapterResponse, summary="创建章节")
async def create_chapter(
    chapter: ChapterCreate,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """创建新的章节"""
    # 验证用户权限和项目是否存在
    user_id = getattr(request.state, 'user_id', None)
    project = await verify_project_access(chapter.project_id, user_id, db)
    
    # 计算字数(处理content可能为None的情况)
    word_count = len(chapter.content) if chapter.content else 0
    
    db_chapter = Chapter(
        **chapter.model_dump(),
        word_count=word_count
    )
    db.add(db_chapter)
    
    # 更新项目的当前字数
    project.current_words = project.current_words + word_count
    
    await db.commit()
    await db.refresh(db_chapter)
    return db_chapter


@router.get("/project/{project_id}", response_model=ChapterListResponse, summary="获取项目的所有章节")
async def get_project_chapters(
    project_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """获取指定项目的所有章节（带大纲信息）"""
    # 验证用户权限
    user_id = getattr(request.state, 'user_id', None)
    await verify_project_access(project_id, user_id, db)
    
    # 获取总数
    count_result = await db.execute(
        select(func.count(Chapter.id)).where(Chapter.project_id == project_id)
    )
    total = count_result.scalar_one()
    
    # 获取章节列表，同时加载关联的大纲信息
    result = await db.execute(
        select(Chapter)
        .where(Chapter.project_id == project_id)
        .order_by(Chapter.chapter_number)
    )
    chapters = result.scalars().all()
    
    # 获取所有大纲信息（用于填充outline_title）
    outline_ids = [ch.outline_id for ch in chapters if ch.outline_id]
    outlines_map = {}
    if outline_ids:
        outlines_result = await db.execute(
            select(Outline).where(Outline.id.in_(outline_ids))
        )
        outlines_map = {o.id: o for o in outlines_result.scalars().all()}
    
    # 为所有章节添加大纲信息（统一处理）
    chapters_with_outline = []
    for chapter in chapters:
        chapter_dict = {
            "id": chapter.id,
            "project_id": chapter.project_id,
            "chapter_number": chapter.chapter_number,
            "title": chapter.title,
            "content": chapter.content,
            "summary": chapter.summary,
            "word_count": chapter.word_count,
            "status": chapter.status,
            "outline_id": chapter.outline_id,
            "sub_index": chapter.sub_index,
            "expansion_plan": chapter.expansion_plan,
            "created_at": chapter.created_at,
            "updated_at": chapter.updated_at,
        }
        
        # 添加大纲信息
        if chapter.outline_id and chapter.outline_id in outlines_map:
            outline = outlines_map[chapter.outline_id]
            chapter_dict["outline_title"] = outline.title
            chapter_dict["outline_order"] = outline.order_index
        else:
            chapter_dict["outline_title"] = None
            chapter_dict["outline_order"] = None
        
        chapters_with_outline.append(chapter_dict)
    
    return ChapterListResponse(total=total, items=chapters_with_outline)


@router.get("/{chapter_id}", response_model=ChapterResponse, summary="获取章节详情")
async def get_chapter(
    chapter_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """根据ID获取章节详情"""
    result = await db.execute(
        select(Chapter).where(Chapter.id == chapter_id)
    )
    chapter = result.scalar_one_or_none()
    
    if not chapter:
        raise HTTPException(status_code=404, detail="章节不存在")
    
    # 验证用户权限
    user_id = getattr(request.state, 'user_id', None)
    await verify_project_access(chapter.project_id, user_id, db)
    
    return chapter


@router.get("/{chapter_id}/navigation", summary="获取章节导航信息")
async def get_chapter_navigation(
    chapter_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """
    获取章节的导航信息（上一章/下一章）
    用于章节阅读器的翻页功能
    """
    # 获取当前章节
    result = await db.execute(
        select(Chapter).where(Chapter.id == chapter_id)
    )
    current_chapter = result.scalar_one_or_none()
    
    if not current_chapter:
        raise HTTPException(status_code=404, detail="章节不存在")
    
    # 验证用户权限
    user_id = getattr(request.state, 'user_id', None)
    await verify_project_access(current_chapter.project_id, user_id, db)
    
    # 获取上一章
    prev_result = await db.execute(
        select(Chapter)
        .where(Chapter.project_id == current_chapter.project_id)
        .where(Chapter.chapter_number < current_chapter.chapter_number)
        .order_by(Chapter.chapter_number.desc())
        .limit(1)
    )
    prev_chapter = prev_result.scalar_one_or_none()
    
    # 获取下一章
    next_result = await db.execute(
        select(Chapter)
        .where(Chapter.project_id == current_chapter.project_id)
        .where(Chapter.chapter_number > current_chapter.chapter_number)
        .order_by(Chapter.chapter_number.asc())
        .limit(1)
    )
    next_chapter = next_result.scalar_one_or_none()
    
    return {
        "current": {
            "id": current_chapter.id,
            "chapter_number": current_chapter.chapter_number,
            "title": current_chapter.title
        },
        "previous": {
            "id": prev_chapter.id,
            "chapter_number": prev_chapter.chapter_number,
            "title": prev_chapter.title
        } if prev_chapter else None,
        "next": {
            "id": next_chapter.id,
            "chapter_number": next_chapter.chapter_number,
            "title": next_chapter.title
        } if next_chapter else None
    }


@router.put("/{chapter_id}", response_model=ChapterResponse, summary="更新章节")
async def update_chapter(
    chapter_id: str,
    chapter_update: ChapterUpdate,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """更新章节信息"""
    result = await db.execute(
        select(Chapter).where(Chapter.id == chapter_id)
    )
    chapter = result.scalar_one_or_none()
    
    if not chapter:
        raise HTTPException(status_code=404, detail="章节不存在")
    
    # 验证用户权限
    user_id = getattr(request.state, 'user_id', None)
    await verify_project_access(chapter.project_id, user_id, db)
    
    # 记录旧字数
    old_word_count = chapter.word_count or 0
    
    # 更新字段
    update_data = chapter_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(chapter, field, value)
    
    # 如果内容更新了，重新计算字数（包括清空内容的情况）
    if "content" in update_data:
        new_word_count = len(chapter.content) if chapter.content else 0
        chapter.word_count = new_word_count
        
        # 更新项目字数
        result = await db.execute(
            select(Project).where(Project.id == chapter.project_id)
        )
        project = result.scalar_one_or_none()
        if project:
            project.current_words = project.current_words - old_word_count + new_word_count
        
        # 如果内容被清空，清理相关数据
            if not chapter.content or chapter.content.strip() == "":
                chapter.status = "draft"
                
                # 清理分析任务
                analysis_tasks_result = await db.execute(
                    select(AnalysisTask).where(AnalysisTask.chapter_id == chapter_id)
                )
                analysis_tasks = analysis_tasks_result.scalars().all()
                for task in analysis_tasks:
                    await db.delete(task)
                
                # 清理分析结果
                plot_analysis_result = await db.execute(
                    select(PlotAnalysis).where(PlotAnalysis.chapter_id == chapter_id)
                )
                plot_analyses = plot_analysis_result.scalars().all()
                for analysis in plot_analyses:
                    await db.delete(analysis)
                
                # 清理故事记忆（关系数据库）
                story_memories_result = await db.execute(
                    select(StoryMemory).where(StoryMemory.chapter_id == chapter_id)
                )
                story_memories = story_memories_result.scalars().all()
                for memory in story_memories:
                    await db.delete(memory)
                
                # 清理向量数据库中的记忆数据
                try:
                    await memory_service.delete_chapter_memories(
                        user_id=user_id,
                        project_id=chapter.project_id,
                        chapter_id=chapter_id
                    )
                    logger.info(f"✅ 已清理章节 {chapter_id[:8]} 的向量记忆数据")
                except Exception as e:
                    logger.warning(f"⚠️ 清理向量记忆数据失败: {str(e)}")
                
                # 🔮 清理章节相关的分析伏笔数据
                try:
                    foreshadow_result = await foreshadow_service.delete_chapter_foreshadows(
                        db=db,
                        project_id=chapter.project_id,
                        chapter_id=chapter_id,
                        only_analysis_source=True  # 只删除分析来源的伏笔，保留手动创建的
                    )
                    if foreshadow_result['deleted_count'] > 0:
                        logger.info(f"🔮 已清理章节 {chapter_id[:8]} 的 {foreshadow_result['deleted_count']} 个伏笔数据")
                except Exception as e:
                    logger.warning(f"⚠️ 清理伏笔数据失败: {str(e)}")
                
                logger.info(f"🗑️ 章节 {chapter_id[:8]} 内容已清空，已清理分析、记忆和伏笔数据")
    
    await db.commit()
    await db.refresh(chapter)
    
    chapter_dict = {
        "id": chapter.id,
        "project_id": chapter.project_id,
        "chapter_number": chapter.chapter_number,
        "title": chapter.title,
        "content": chapter.content,
        "summary": chapter.summary,
        "word_count": chapter.word_count,
        "status": chapter.status,
        "outline_id": chapter.outline_id,
        "sub_index": chapter.sub_index,
        "expansion_plan": chapter.expansion_plan,
        "created_at": chapter.created_at,
        "updated_at": chapter.updated_at,
        "outline_title": None,
        "outline_order": None
    }
    
    # 如果章节关联了大纲，查询大纲信息
    if chapter.outline_id:
        outline_result = await db.execute(
            select(Outline).where(Outline.id == chapter.outline_id)
        )
        outline = outline_result.scalar_one_or_none()
        if outline:
            chapter_dict["outline_title"] = outline.title
            chapter_dict["outline_order"] = outline.order_index
    
    return chapter_dict


@router.delete("/{chapter_id}", summary="删除章节")
async def delete_chapter(
    chapter_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """删除章节"""
    result = await db.execute(
        select(Chapter).where(Chapter.id == chapter_id)
    )
    chapter = result.scalar_one_or_none()
    
    if not chapter:
        raise HTTPException(status_code=404, detail="章节不存在")
    
    # 验证用户权限
    user_id = getattr(request.state, 'user_id', None)
    await verify_project_access(chapter.project_id, user_id, db)
    
    # 更新项目字数
    result = await db.execute(
        select(Project).where(Project.id == chapter.project_id)
    )
    project = result.scalar_one_or_none()
    if project:
        # 处理 word_count 和 current_words 可能为 None 的情况
        chapter_word_count = chapter.word_count or 0
        project.current_words = max(0, (project.current_words or 0) - chapter_word_count)
    
    # 🗑️ 清理向量数据库中的记忆数据
    try:
        await memory_service.delete_chapter_memories(
            user_id=user_id,
            project_id=chapter.project_id,
            chapter_id=chapter_id
        )
        logger.info(f"✅ 已清理章节 {chapter_id[:8]} 的向量记忆数据")
    except Exception as e:
        logger.warning(f"⚠️ 清理向量记忆数据失败: {str(e)}")
        # 不阻断删除流程，继续执行
    
    # 🔮 清理与该章节相关的伏笔数据（仅分析来源的伏笔）
    try:
        foreshadow_result = await foreshadow_service.delete_chapter_foreshadows(
            db=db,
            project_id=chapter.project_id,
            chapter_id=chapter_id,
            only_analysis_source=True  # 只删除分析来源的伏笔，保留手动创建的
        )
        if foreshadow_result['deleted_count'] > 0:
            logger.info(f"🔮 已清理章节 {chapter_id[:8]} 的 {foreshadow_result['deleted_count']} 个伏笔数据")
    except Exception as e:
        logger.warning(f"⚠️ 清理伏笔数据失败: {str(e)}")
        # 不阻断删除流程，继续执行
    
    # 删除章节（关系数据库中的记忆会被级联删除）
    await db.delete(chapter)
    await db.commit()
    
    return {"message": "章节删除成功"}


async def check_prerequisites(db: AsyncSession, chapter: Chapter) -> tuple[bool, str, list[Chapter]]:
    """
    检查章节前置条件
    
    Args:
        db: 数据库会话
        chapter: 当前章节
        
    Returns:
        (可否生成, 错误信息, 前置章节列表)
    """
    # 如果是第一章，无需检查前置
    if chapter.chapter_number == 1:
        return True, "", []
    
    # 查询所有前置章节（序号小于当前章节的）
    result = await db.execute(
        select(Chapter)
        .where(Chapter.project_id == chapter.project_id)
        .where(Chapter.chapter_number < chapter.chapter_number)
        .order_by(Chapter.chapter_number)
    )
    previous_chapters = result.scalars().all()
    
    # 检查是否所有前置章节都有内容
    incomplete_chapters = [
        ch for ch in previous_chapters
        if not ch.content or ch.content.strip() == ""
    ]
    
    if incomplete_chapters:
        missing_numbers = [str(ch.chapter_number) for ch in incomplete_chapters]
        error_msg = f"需要先完成前置章节：第 {', '.join(missing_numbers)} 章"
        return False, error_msg, previous_chapters
    
    return True, "", previous_chapters


async def build_characters_info_with_careers(
    db: AsyncSession,
    project_id: str,
    characters: list[Character],
    filter_character_names: Optional[list[str]] = None
) -> str:
    """
    构建包含职业信息的角色上下文
    
    Args:
        db: 数据库会话
        project_id: 项目ID
        characters: 角色列表
        filter_character_names: 可选，筛选特定角色名称列表（用于1-1模式的structure.characters或1-n模式的expansion_plan.character_focus）
        
    Returns:
        格式化的角色信息字符串，包含职业信息
    """
    if not characters:
        return '暂无角色信息'
    
    # 如果提供了筛选名单，只保留匹配的角色
    if filter_character_names:
        filtered_characters = [c for c in characters if c.name in filter_character_names]
        if not filtered_characters:
            logger.warning(f"筛选后无匹配角色，使用全部角色。筛选名单: {filter_character_names}")
            filtered_characters = characters
        else:
            logger.info(f"根据筛选名单保留 {len(filtered_characters)}/{len(characters)} 个角色: {[c.name for c in filtered_characters]}")
        characters = filtered_characters
    
    # 获取所有职业信息（一次性查询，提高效率）
    careers_result = await db.execute(
        select(Career).where(Career.project_id == project_id)
    )
    careers_map = {c.id: c for c in careers_result.scalars().all()}
    
    # 获取所有角色的职业关联（一次性查询）
    character_ids = [c.id for c in characters]
    if not character_ids:
        return '暂无角色信息'
    
    # 构建全局角色名称映射（用于关系显示）
    all_chars_result = await db.execute(
        select(Character.id, Character.name).where(Character.project_id == project_id)
    )
    all_char_name_map = {row.id: row.name for row in all_chars_result.all()}
        
    character_careers_result = await db.execute(
        select(CharacterCareer).where(CharacterCareer.character_id.in_(character_ids))
    )
    character_careers = character_careers_result.scalars().all()
    
    # 获取所有角色的关系（一次性查询）
    from sqlalchemy import or_
    rels_result = await db.execute(
        select(CharacterRelationship).where(
            CharacterRelationship.project_id == project_id,
            or_(
                CharacterRelationship.character_from_id.in_(character_ids),
                CharacterRelationship.character_to_id.in_(character_ids)
            )
        )
    )
    all_relationships = rels_result.scalars().all()
    
    # 按角色ID分组关系
    char_rels_map: dict[str, list] = {cid: [] for cid in character_ids}
    for r in all_relationships:
        if r.character_from_id in char_rels_map:
            char_rels_map[r.character_from_id].append(r)
        if r.character_to_id in char_rels_map:
            char_rels_map[r.character_to_id].append(r)
    
    # 获取所有组织及其成员关系（一次性查询）
    orgs_result = await db.execute(
        select(Organization).where(Organization.project_id == project_id)
    )
    all_orgs = orgs_result.scalars().all()
    
    # 构建组织ID到组织名称的映射（通过关联的Character记录）
    org_name_map = {}  # org_id -> org_name
    char_id_to_org = {}  # character_id -> Organization（用于组织实体补充详情）
    for org in all_orgs:
        org_name_map[org.id] = all_char_name_map.get(org.character_id, '未知组织')
        char_id_to_org[org.character_id] = org
    
    # 获取所有组织的成员关系（一次性查询）
    org_ids = [org.id for org in all_orgs]
    all_org_members = []
    if org_ids:
        all_org_members_result = await db.execute(
            select(OrganizationMember).where(
                OrganizationMember.organization_id.in_(org_ids)
            )
        )
        all_org_members = all_org_members_result.scalars().all()
    
    # 按组织ID分组成员（用于组织实体显示成员列表）
    org_members_map: dict[str, list] = {oid: [] for oid in org_ids}
    for m in all_org_members:
        if m.organization_id in org_members_map:
            org_members_map[m.organization_id].append(m)
    
    # 获取涉及当前非组织角色的成员关系
    non_org_char_ids = [c.id for c in characters if not c.is_organization]
    char_org_map: dict[str, list] = {cid: [] for cid in non_org_char_ids}
    for m in all_org_members:
        if m.character_id in char_org_map:
            char_org_map[m.character_id].append(m)
    
    # 构建角色ID到职业信息的映射
    char_career_map = {}
    for cc in character_careers:
        if cc.character_id not in char_career_map:
            char_career_map[cc.character_id] = {'main': None, 'sub': []}
        
        career = careers_map.get(cc.career_id)
        if not career:
            continue
            
        career_info = {
            'name': career.name,
            'stage': cc.current_stage,
            'max_stage': career.max_stage,
            'stage_progress': cc.stage_progress
        }
        
        if cc.career_type == 'main':
            char_career_map[cc.character_id]['main'] = career_info
        else:
            char_career_map[cc.character_id]['sub'].append(career_info)
    
    # 构建角色信息字符串
    characters_info_parts = []
    for c in characters:
        # 基本信息（含存活状态标记）
        entity_type = '组织' if c.is_organization else '角色'
        status_marker = ""
        char_status = getattr(c, 'status', None) or 'active'
        if char_status != 'active':
            STATUS_MARKERS = {
                'deceased': '💀已死亡',
                'missing': '❓已失踪',
                'retired': '📤已退场',
                'destroyed': '💀已覆灭'
            }
            status_marker = f" [{STATUS_MARKERS.get(char_status, char_status)}]"
        base_info = f"- {c.name}({entity_type}, {c.role_type}){status_marker}"
        
        # 组织实体：补充组织详情
        org_detail_str = ""
        if c.is_organization and c.id in char_id_to_org:
            org = char_id_to_org[c.id]
            org_detail_parts = []
            if c.organization_type:
                org_detail_parts.append(f"类型:{c.organization_type}")
            if c.organization_purpose:
                purpose_preview = c.organization_purpose[:60] if len(c.organization_purpose) > 60 else c.organization_purpose
                org_detail_parts.append(f"宗旨:{purpose_preview}")
            if org.power_level is not None:
                org_detail_parts.append(f"势力等级:{org.power_level}")
            if org.location:
                org_detail_parts.append(f"据点:{org.location}")
            if org.motto:
                org_detail_parts.append(f"口号:{org.motto}")
            if org.member_count:
                org_detail_parts.append(f"成员数:{org.member_count}")
            if org_detail_parts:
                org_detail_str = f" | {', '.join(org_detail_parts)}"
            
            # 显示组织的核心成员列表（最多5个）
            if org.id in org_members_map and org_members_map[org.id]:
                member_parts = []
                for m in sorted(org_members_map[org.id], key=lambda x: -(x.rank or 0))[:5]:
                    m_name = all_char_name_map.get(m.character_id, '未知')
                    m_desc = f"{m_name}({m.position})"
                    if m.status and m.status != 'active':
                        m_desc += f"[{m.status}]"
                    member_parts.append(m_desc)
                if member_parts:
                    org_detail_str += f" | 成员: {', '.join(member_parts)}"
        
        # 职业信息
        career_info_str = ""
        if c.id in char_career_map:
            career_data = char_career_map[c.id]
            
            # 主职业
            if career_data['main']:
                main = career_data['main']
                stage_desc = f"{main['stage']}/{main['max_stage']}阶"
                career_info_str += f" | 主职业: {main['name']}({stage_desc})"
            
            # 副职业
            if career_data['sub']:
                sub_list = []
                for sub in career_data['sub']:
                    stage_desc = f"{sub['stage']}/{sub['max_stage']}阶"
                    sub_list.append(f"{sub['name']}({stage_desc})")
                career_info_str += f" | 副职业: {', '.join(sub_list)}"
        
        # 心理状态（由章节分析自动更新）
        state_str = ""
        if c.current_state:
            state_preview = c.current_state[:50] if len(c.current_state) > 50 else c.current_state
            state_str = f" | 当前状态: {state_preview}"
            if c.state_updated_chapter:
                state_str += f"(第{c.state_updated_chapter}章)"
        
        # 组织成员信息（非组织角色才显示所属组织）
        org_str = ""
        if not c.is_organization and c.id in char_org_map and char_org_map[c.id]:
            org_parts = []
            for m in char_org_map[c.id][:3]:  # 最多显示3个组织
                o_name = org_name_map.get(m.organization_id, '未知组织')
                o_desc = f"{o_name}({m.position})"
                if m.loyalty is not None and m.loyalty != 50:
                    o_desc += f"[忠诚度:{m.loyalty}]"
                if m.status and m.status != 'active':
                    o_desc += f"[{m.status}]"
                org_parts.append(o_desc)
            if org_parts:
                org_str = f" | 所属组织: {', '.join(org_parts)}"
        
        # 关系信息
        rel_str = ""
        if c.id in char_rels_map and char_rels_map[c.id]:
            rel_parts = []
            seen_pairs = set()  # 避免重复显示同一对关系
            for r in char_rels_map[c.id][:5]:  # 最多显示5个关系
                # 确定对方角色名
                if r.character_from_id == c.id:
                    other_name = all_char_name_map.get(r.character_to_id, '未知')
                else:
                    other_name = all_char_name_map.get(r.character_from_id, '未知')
                
                pair_key = tuple(sorted([c.id, r.character_from_id if r.character_from_id != c.id else r.character_to_id]))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                
                rel_name = r.relationship_name or '关联'
                rel_desc = f"{other_name}({rel_name})"
                if r.intimacy_level is not None and r.intimacy_level != 50:
                    rel_desc += f"[亲密度:{r.intimacy_level}]"
                rel_parts.append(rel_desc)
            
            if rel_parts:
                rel_str = f" | 关系: {', '.join(rel_parts)}"
        
        # 性格描述
        personality_str = ""
        if c.personality:
            personality_preview = c.personality[:100] if len(c.personality) > 100 else c.personality
            personality_str = f": {personality_preview}"
        
        # 组合完整信息
        full_info = base_info + org_detail_str + career_info_str + state_str + org_str + rel_str + personality_str
        characters_info_parts.append(full_info)
    
    return "\n".join(characters_info_parts)


@router.get("/{chapter_id}/can-generate", summary="检查章节是否可以生成")
async def check_can_generate(
    chapter_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """
    检查章节是否满足生成条件
    返回可生成状态和前置章节信息
    """
    # 获取章节
    result = await db.execute(
        select(Chapter).where(Chapter.id == chapter_id)
    )
    chapter = result.scalar_one_or_none()
    if not chapter:
        raise HTTPException(status_code=404, detail="章节不存在")
    
    # 验证用户权限
    user_id = getattr(request.state, 'user_id', None)
    await verify_project_access(chapter.project_id, user_id, db)
    
    # 检查前置条件
    can_generate, error_msg, previous_chapters = await check_prerequisites(db, chapter)
    
    # 构建前置章节信息
    previous_info = [
        {
            "id": ch.id,
            "chapter_number": ch.chapter_number,
            "title": ch.title,
            "has_content": bool(ch.content and ch.content.strip()),
            "word_count": ch.word_count or 0
        }
        for ch in previous_chapters
    ]
    
    return {
        "can_generate": can_generate,
        "reason": error_msg if not can_generate else "",
        "previous_chapters": previous_info,
        "chapter_number": chapter.chapter_number
    }


async def analyze_chapter_background(
    chapter_id: str,
    user_id: str,
    project_id: str,
    task_id: str,
    ai_service: AIService
) -> bool:
    """
    后台异步分析章节（支持并发，使用锁保护数据库写入）
    
    Args:
        chapter_id: 章节ID
        user_id: 用户ID
        project_id: 项目ID
        task_id: 任务ID
        ai_service: AI服务实例
        
    Returns:
        bool: True表示分析成功，False表示分析失败
    """
    db_session = None
    write_lock = await get_db_write_lock(user_id)
    
    try:
        logger.info(f"🔍 开始分析章节: {chapter_id}, 任务ID: {task_id}")
        
        # 创建独立数据库会话
        from app.database import get_engine
        from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession
        
        engine = await get_engine(user_id)
        AsyncSessionLocal = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        db_session = AsyncSessionLocal()

        # ✅ 章节分析可能希望使用与“章节生成”不同的模型/Provider
        # 这里强制按 task_key=chapter_analysis 重新创建 AIService（若失败则回退到调用方传入的 ai_service）。
        try:
            from app.api.settings import create_user_ai_service_for_task

            ai_service = await create_user_ai_service_for_task(
                user_id=user_id,
                db=db_session,
                task_key="chapter_analysis",
            )
            logger.info("✅ 已按 chapter_analysis 路由创建 AIService")
        except Exception as route_error:
            logger.warning(f"⚠️ 创建 chapter_analysis 路由 AIService 失败，回退到调用方传入的 AIService: {route_error}")
        
        # 1. 获取任务（读操作）
        task_result = await db_session.execute(
            select(AnalysisTask).where(AnalysisTask.id == task_id)
        )
        task = task_result.scalar_one_or_none()
        
        if not task:
            logger.error(f"❌ 任务不存在: {task_id}")
            return False
        
        # 更新任务状态（写操作，需要锁）
        async with write_lock:
            task.status = 'running'
            task.started_at = datetime.now()
            task.progress = 10
            await db_session.commit()
        
        # 2. 获取章节信息（读操作）
        chapter_result = await db_session.execute(
            select(Chapter).where(Chapter.id == chapter_id)
        )
        chapter = chapter_result.scalar_one_or_none()
        if not chapter or not chapter.content:
            async with write_lock:
                task.status = 'failed'
                task.error_message = '章节不存在或内容为空'
                task.completed_at = datetime.now()
                await db_session.commit()
            logger.error(f"❌ 章节不存在或内容为空: {chapter_id}")
            return False
        
        async with write_lock:
            task.progress = 20
            await db_session.commit()
        
        # 获取已埋入的伏笔列表（用于回收匹配，传入当前章节号以启用智能标记）
        existing_foreshadows = await foreshadow_service.get_planted_foreshadows_for_analysis(
            db=db_session,
            project_id=project_id,
            current_chapter_number=chapter.chapter_number  # 传入当前章节号以启用智能标记
        )
        logger.info(f"📋 后台分析 - 已获取{len(existing_foreshadows)}个已埋入伏笔用于匹配（含智能回收标记）")
        
        # 获取项目角色信息（根据大纲/展开规划筛选本章相关角色）
        filter_character_names = None
        
        # 1-N模式：从expansion_plan中提取character_focus
        if chapter.expansion_plan:
            try:
                plan = json.loads(chapter.expansion_plan)
                focus_names = plan.get('character_focus', [])
                if focus_names:
                    filter_character_names = focus_names
                    logger.info(f"📋 从expansion_plan提取角色焦点: {filter_character_names}")
            except (json.JSONDecodeError, Exception):
                pass
        
        # 1-1模式：从outline.structure中提取characters
        if not filter_character_names and chapter.outline_id:
            try:
                outline_result = await db_session.execute(
                    select(Outline).where(Outline.id == chapter.outline_id)
                )
                chapter_outline = outline_result.scalar_one_or_none()
                if chapter_outline and chapter_outline.structure:
                    structure = json.loads(chapter_outline.structure)
                    raw_characters = structure.get('characters', [])
                    if raw_characters:
                        filter_character_names = [
                            c['name'] if isinstance(c, dict) else c
                            for c in raw_characters
                        ]
                        logger.info(f"📋 从outline.structure提取角色: {filter_character_names}")
            except (json.JSONDecodeError, Exception):
                pass
        
        # 查询角色（根据筛选名单或全部）
        characters_query = select(Character).where(Character.project_id == project_id)
        if filter_character_names:
            characters_query = characters_query.where(Character.name.in_(filter_character_names))
        characters_result = await db_session.execute(characters_query)
        project_characters = characters_result.scalars().all()
        
        # 如果筛选后无角色，降级为全部角色
        if not project_characters and filter_character_names:
            logger.warning(f"⚠️ 筛选后无匹配角色，降级为全部角色")
            characters_result = await db_session.execute(
                select(Character).where(Character.project_id == project_id)
            )
            project_characters = characters_result.scalars().all()
            filter_character_names = None
        
        characters_info = await build_characters_info_with_careers(
            db=db_session,
            project_id=project_id,
            characters=project_characters,
            filter_character_names=filter_character_names
        )
        logger.info(f"📋 后台分析 - 已获取{len(project_characters)}个角色信息用于分析")
        
        # 定义重试回调函数，用于在重试时更新任务状态
        async def on_retry_callback(attempt: int, max_retries: int, wait_time: int, error_reason: str):
            """重试时更新任务状态，让前端能感知到重试进度"""
            try:
                async with write_lock:
                    # 重新获取任务（确保获取最新状态）
                    task_result_retry = await db_session.execute(
                        select(AnalysisTask).where(AnalysisTask.id == task_id)
                    )
                    task_retry = task_result_retry.scalar_one_or_none()
                    if task_retry:
                        # 更新任务状态，保持 running 但更新 started_at 以重置超时计时器
                        task_retry.status = 'running'
                        task_retry.started_at = datetime.now()  # 重置开始时间，防止超时检测误判
                        task_retry.progress = 25 + attempt * 5  # 根据重试次数更新进度
                        task_retry.error_message = f"正在重试({attempt}/{max_retries})：{error_reason[:100]}"
                        await db_session.commit()
                        logger.info(f"🔄 分析任务重试状态已更新: 尝试 {attempt}/{max_retries}, 等待 {wait_time}s, 原因: {error_reason[:50]}...")
            except Exception as callback_error:
                logger.warning(f"⚠️ 更新重试状态失败: {callback_error}")
        
        # 3. 使用PlotAnalyzer分析章节（传入已有伏笔列表、角色信息和重试回调）
        analyzer = PlotAnalyzer(ai_service)
        analysis_result = await analyzer.analyze_chapter(
            chapter_number=chapter.chapter_number,
            title=chapter.title,
            content=chapter.content,
            word_count=chapter.word_count or len(chapter.content),
            existing_foreshadows=existing_foreshadows,
            on_retry=on_retry_callback,
            characters_info=characters_info
        )
        
        if not analysis_result:
            async with write_lock:
                task.status = 'failed'
                task.error_message = 'AI分析失败，请检查日志'
                task.completed_at = datetime.now()
                await db_session.commit()
            logger.error(f"❌ AI分析失败: {chapter_id}")
            return False
        
        async with write_lock:
            task.progress = 60
            await db_session.commit()
        
        # 4. 保存分析结果到数据库（写操作，需要锁）
        async with write_lock:
            existing_analysis_result = await db_session.execute(
                select(PlotAnalysis).where(PlotAnalysis.chapter_id == chapter_id)
            )
            existing_analysis = existing_analysis_result.scalar_one_or_none()
            
            if existing_analysis:
                # 更新现有记录
                logger.info(f"  更新现有分析记录: {existing_analysis.id}")
                existing_analysis.plot_stage = analysis_result.get('plot_stage', '发展')
                existing_analysis.conflict_level = analysis_result.get('conflict', {}).get('level', 0)
                existing_analysis.conflict_types = analysis_result.get('conflict', {}).get('types', [])
                existing_analysis.emotional_tone = analysis_result.get('emotional_arc', {}).get('primary_emotion', '')
                existing_analysis.emotional_intensity = analysis_result.get('emotional_arc', {}).get('intensity', 0) / 10.0
                existing_analysis.hooks = analysis_result.get('hooks', [])
                existing_analysis.hooks_count = len(analysis_result.get('hooks', []))
                existing_analysis.foreshadows = analysis_result.get('foreshadows', [])
                existing_analysis.foreshadows_planted = sum(1 for f in analysis_result.get('foreshadows', []) if f.get('type') == 'planted')
                existing_analysis.foreshadows_resolved = sum(1 for f in analysis_result.get('foreshadows', []) if f.get('type') == 'resolved')
                existing_analysis.plot_points = analysis_result.get('plot_points', [])
                existing_analysis.plot_points_count = len(analysis_result.get('plot_points', []))
                existing_analysis.character_states = analysis_result.get('character_states', [])
                existing_analysis.scenes = analysis_result.get('scenes', [])
                existing_analysis.pacing = analysis_result.get('pacing', 'moderate')
                existing_analysis.overall_quality_score = analysis_result.get('scores', {}).get('overall', 0)
                existing_analysis.pacing_score = analysis_result.get('scores', {}).get('pacing', 0)
                existing_analysis.engagement_score = analysis_result.get('scores', {}).get('engagement', 0)
                existing_analysis.coherence_score = analysis_result.get('scores', {}).get('coherence', 0)
                existing_analysis.analysis_report = analyzer.generate_analysis_summary(analysis_result)
                existing_analysis.suggestions = analysis_result.get('suggestions', [])
                existing_analysis.dialogue_ratio = analysis_result.get('dialogue_ratio', 0)
                existing_analysis.description_ratio = analysis_result.get('description_ratio', 0)
            else:
                # 创建新记录
                logger.info(f"  创建新的分析记录")
                plot_analysis = PlotAnalysis(
                    chapter_id=chapter_id,
                    project_id=project_id,
                    plot_stage=analysis_result.get('plot_stage', '发展'),
                    conflict_level=analysis_result.get('conflict', {}).get('level', 0),
                    conflict_types=analysis_result.get('conflict', {}).get('types', []),
                    emotional_tone=analysis_result.get('emotional_arc', {}).get('primary_emotion', ''),
                    emotional_intensity=analysis_result.get('emotional_arc', {}).get('intensity', 0) / 10.0,
                    hooks=analysis_result.get('hooks', []),
                    hooks_count=len(analysis_result.get('hooks', [])),
                    foreshadows=analysis_result.get('foreshadows', []),
                    foreshadows_planted=sum(1 for f in analysis_result.get('foreshadows', []) if f.get('type') == 'planted'),
                    foreshadows_resolved=sum(1 for f in analysis_result.get('foreshadows', []) if f.get('type') == 'resolved'),
                    plot_points=analysis_result.get('plot_points', []),
                    plot_points_count=len(analysis_result.get('plot_points', [])),
                    character_states=analysis_result.get('character_states', []),
                    scenes=analysis_result.get('scenes', []),
                    pacing=analysis_result.get('pacing', 'moderate'),
                    overall_quality_score=analysis_result.get('scores', {}).get('overall', 0),
                    pacing_score=analysis_result.get('scores', {}).get('pacing', 0),
                    engagement_score=analysis_result.get('scores', {}).get('engagement', 0),
                    coherence_score=analysis_result.get('scores', {}).get('coherence', 0),
                    analysis_report=analyzer.generate_analysis_summary(analysis_result),
                    suggestions=analysis_result.get('suggestions', []),
                    dialogue_ratio=analysis_result.get('dialogue_ratio', 0),
                    description_ratio=analysis_result.get('description_ratio', 0)
                )
                db_session.add(plot_analysis)
            
            await db_session.commit()
            
            task.progress = 80
            await db_session.commit()
        
        # 5. 清理旧的分析伏笔（重新分析时需要先清理）
        try:
            async with write_lock:
                clean_result = await foreshadow_service.clean_chapter_analysis_foreshadows(
                    db=db_session,
                    project_id=project_id,
                    chapter_id=chapter_id
                )
            if clean_result['cleaned_count'] > 0:
                logger.info(f"🧹 重新分析前清理了 {clean_result['cleaned_count']} 个旧伏笔")
        except Exception as clean_error:
            logger.warning(f"⚠️ 清理旧伏笔失败（继续分析）: {str(clean_error)}")
        
        # 6. 提取记忆并保存到向量数据库（传入章节内容用于计算位置）
        memories = analyzer.extract_memories_from_analysis(
            analysis=analysis_result,
            chapter_id=chapter_id,
            chapter_number=chapter.chapter_number,
            chapter_content=chapter.content or "",
            chapter_title=chapter.title or ""
        )

        # 记录本次分析提取的记忆数量（用于前端标记“提取记忆异常”）
        try:
            async with write_lock:
                task.memory_extracted_count = int(len(memories or []))
                task.vector_expected_count = int(len(memories or []))
                task.vector_added_count = 0
                task.vector_skipped_count = 0
                task.vector_error_message = None
                task.vector_embed_id = None
                task.vector_collection = None
                await db_session.commit()
        except Exception as e:
            logger.warning(f"⚠️ 记录记忆数量到分析任务失败（不影响分析继续）: {e}")
        
        # 先删除该章节的旧记忆（写操作，需要锁）
        async with write_lock:
            old_memories_result = await db_session.execute(
                select(StoryMemory).where(StoryMemory.chapter_id == chapter_id)
            )
            old_memories = old_memories_result.scalars().all()
            for old_mem in old_memories:
                await db_session.delete(old_mem)
            await db_session.commit()
            logger.info(f"  删除旧记忆: {len(old_memories)}条")
        
        # 准备批量添加的记忆数据（不需要锁）
        memory_records = []
        for idx, mem in enumerate(memories or []):
            memory_id = f"{chapter_id}_{mem['type']}_{idx}"
            memory_records.append({
                'id': memory_id,
                'content': mem['content'],
                'type': mem['type'],
                'metadata': mem['metadata']
            })
            
        # 保存到关系数据库（写操作，需要锁）
        async with write_lock:
            for idx, mem in enumerate(memories or []):
                memory_id = memory_records[idx]['id']
                text_position = mem['metadata'].get('text_position', -1)
                text_length = mem['metadata'].get('text_length', 0)
                
                story_memory = StoryMemory(
                    id=memory_id,
                    project_id=project_id,
                    chapter_id=chapter_id,
                    memory_type=mem['type'],
                    content=mem['content'],
                    title=mem['title'],
                    importance_score=mem['metadata'].get('importance_score', 0.5),
                    tags=mem['metadata'].get('tags', []),
                    is_foreshadow=mem['metadata'].get('is_foreshadow', 0),
                    story_timeline=chapter.chapter_number,
                    chapter_position=text_position,
                    text_length=text_length,
                    related_characters=mem['metadata'].get('related_characters', []),
                    related_locations=mem['metadata'].get('related_locations', [])
                )
                db_session.add(story_memory)
                
                if text_position >= 0:
                    logger.debug(f"  保存记忆 {memory_id}: position={text_position}, length={text_length}")
            
            await db_session.commit()
        
        # 批量添加到向量数据库
        if memory_records:
            vector_res = await memory_service.batch_add_memories(
                user_id=user_id,
                project_id=project_id,
                memories=memory_records,
                db=db_session,
            )
            try:
                async with write_lock:
                    task.vector_expected_count = int(vector_res.get("requested") or len(memory_records))
                    task.vector_added_count = int(vector_res.get("added") or 0)
                    task.vector_skipped_count = int(vector_res.get("skipped") or 0)
                    task.vector_error_message = vector_res.get("error_message")
                    task.vector_embed_id = vector_res.get("embed_id")
                    task.vector_collection = vector_res.get("collection")
                    await db_session.commit()
            except Exception as e:
                logger.warning(f"⚠️ 写入向量统计到分析任务失败（不影响分析继续）: {e}")

            logger.info(
                f"✅ 向量写入结果: requested={vector_res.get('requested')}, added={vector_res.get('added')}, skipped={vector_res.get('skipped')}"
            )
            if vector_res.get("error_message"):
                logger.warning(f"⚠️ 向量写入警告: {vector_res.get('error_message')}")
        
        # 💼 更新角色职业（根据分析结果）
        if analysis_result.get('character_states'):
            try:
                from app.services.career_update_service import CareerUpdateService
                
                logger.info(f"💼 开始根据分析结果更新角色职业...")
                career_update_result = await CareerUpdateService.update_careers_from_analysis(
                    db=db_session,
                    project_id=project_id,
                    character_states=analysis_result.get('character_states', []),
                    chapter_id=chapter_id,
                    chapter_number=chapter.chapter_number
                )
                
                if career_update_result['updated_count'] > 0:
                    logger.info(
                        f"✅ 更新了 {career_update_result['updated_count']} 个角色的职业信息"
                    )
                    if career_update_result['changes']:
                        for change in career_update_result['changes']:
                            logger.info(f"  - {change}")
                else:
                    logger.info("ℹ️ 本章节无角色职业变化")
                    
            except Exception as career_error:
                # 职业更新失败不应影响整个分析流程
                logger.error(f"⚠️ 更新角色职业失败: {str(career_error)}", exc_info=True)
        else:
            logger.debug("📋 分析结果中无角色状态信息，跳过职业更新")
        
        # 👤 更新角色心理状态和关系（根据分析结果）
        if analysis_result.get('character_states'):
            try:
                from app.services.character_state_update_service import CharacterStateUpdateService
                
                logger.info(f"👤 开始根据分析结果更新角色状态、关系和组织成员...")
                async with write_lock:
                    state_update_result = await CharacterStateUpdateService.update_from_analysis(
                        db=db_session,
                        project_id=project_id,
                        character_states=analysis_result.get('character_states', []),
                        chapter_id=chapter_id,
                        chapter_number=chapter.chapter_number
                    )
                
                total_state_changes = (
                    state_update_result['state_updated_count'] +
                    state_update_result['relationship_created_count'] +
                    state_update_result['relationship_updated_count'] +
                    state_update_result.get('org_updated_count', 0)
                )
                if total_state_changes > 0:
                    logger.info(
                        f"✅ 角色状态更新: 心理状态{state_update_result['state_updated_count']}个, "
                        f"新建关系{state_update_result['relationship_created_count']}个, "
                        f"更新关系{state_update_result['relationship_updated_count']}个, "
                        f"组织变动{state_update_result.get('org_updated_count', 0)}个"
                    )
                    if state_update_result['changes']:
                        for change in state_update_result['changes'][:8]:
                            logger.info(f"  - {change}")
                else:
                    logger.info("ℹ️ 本章节无角色状态、关系或组织变化")
                    
            except Exception as state_error:
                # 角色状态更新失败不应影响整个分析流程
                logger.error(f"⚠️ 更新角色状态、关系和组织失败: {str(state_error)}", exc_info=True)
        
        # 🏛️ 更新组织自身状态（根据分析结果）
        if analysis_result.get('organization_states'):
            try:
                from app.services.character_state_update_service import CharacterStateUpdateService
                
                logger.info(f"🏛️ 开始根据分析结果更新组织自身状态...")
                async with write_lock:
                    org_state_result = await CharacterStateUpdateService.update_organization_states(
                        db=db_session,
                        project_id=project_id,
                        organization_states=analysis_result.get('organization_states', []),
                        chapter_number=chapter.chapter_number
                    )
                
                if org_state_result['updated_count'] > 0:
                    logger.info(
                        f"✅ 组织状态更新: {org_state_result['updated_count']}个组织"
                    )
                    if org_state_result['changes']:
                        for change in org_state_result['changes'][:5]:
                            logger.info(f"  - {change}")
                else:
                    logger.info("ℹ️ 本章节无组织自身状态变化")
                    
            except Exception as org_state_error:
                # 组织状态更新失败不应影响整个分析流程
                logger.error(f"⚠️ 更新组织自身状态失败: {str(org_state_error)}", exc_info=True)
        
        # 🔮 自动更新伏笔状态（根据分析结果）
        if analysis_result.get('foreshadows'):
            try:
                logger.info(f"🔮 开始根据分析结果自动更新伏笔状态...")
                async with write_lock:
                    foreshadow_stats = await foreshadow_service.auto_update_from_analysis(
                        db=db_session,
                        project_id=project_id,
                        chapter_id=chapter_id,
                        chapter_number=chapter.chapter_number,
                        analysis_foreshadows=analysis_result.get('foreshadows', [])
                    )
                
                if foreshadow_stats['planted_count'] > 0 or foreshadow_stats['resolved_count'] > 0:
                    logger.info(
                        f"✅ 伏笔自动更新: 埋入{foreshadow_stats['planted_count']}个, "
                        f"回收{foreshadow_stats['resolved_count']}个"
                    )
                else:
                    logger.info("ℹ️ 本章节无新的伏笔状态变化")
                    
            except Exception as foreshadow_error:
                # 伏笔更新失败不应影响整个分析流程
                logger.error(f"⚠️ 自动更新伏笔失败: {str(foreshadow_error)}", exc_info=True)
        else:
            logger.debug("📋 分析结果中无伏笔信息，跳过伏笔自动更新")
        
        # 最终更新任务状态（写操作，需要锁）- 增加重试机制
        update_success = False
        for retry in range(3):
            try:
                async with write_lock:
                    task.progress = 100
                    task.status = 'completed'
                    # 若此前被 status 轮询标记过“超时/重试”等错误信息，这里在成功完成时清空，
                    # 避免出现 status=completed 但 error_message 仍显示“任务超时”的误导性状态。
                    task.error_message = None
                    task.completed_at = datetime.now()
                    await db_session.commit()
                    update_success = True
                    logger.info(f"✅ 章节分析完成: {chapter_id}, 提取{len(memories)}条记忆")
                    break
            except Exception as commit_error:
                logger.error(f"❌ 提交任务完成状态失败(重试{retry+1}/3): {str(commit_error)}")
                if retry < 2:
                    await asyncio.sleep(0.1)
                else:
                    logger.error(f"❌ 无法更新任务为completed状态: {task_id}")
                    # 即使失败也不抛出异常，因为分析本身已经完成
        
        if not update_success:
            logger.warning(f"⚠️  章节分析完成但状态更新失败: {chapter_id}")
        
        # 返回成功状态
        return True
        
    except Exception as e:
        logger.error(f"❌ 后台分析异常: {str(e)}", exc_info=True)
        # 确保任务状态被更新为failed（写操作，需要锁）
        if db_session:
            # 多次重试更新任务状态
            for retry in range(3):
                try:
                    async with write_lock:
                        # 重新获取任务（可能是旧会话导致的问题）
                        task_result = await db_session.execute(
                            select(AnalysisTask).where(AnalysisTask.id == task_id)
                        )
                        task = task_result.scalar_one_or_none()
                        if task:
                            task.status = 'failed'
                            task.error_message = str(e)[:500]
                            task.completed_at = datetime.now()
                            task.progress = 0
                            await db_session.commit()
                            logger.info(f"✅ 任务状态已更新为failed: {task_id} (重试{retry+1}次)")
                            break
                        else:
                            logger.error(f"❌ 无法找到任务进行状态更新: {task_id}")
                            break
                except Exception as update_error:
                    logger.error(f"❌ 更新任务状态失败(重试{retry+1}/3): {str(update_error)}")
                    if retry < 2:
                        await asyncio.sleep(0.1)  # 短暂等待后重试
                    else:
                        logger.error(f"❌ 任务状态更新失败，已达到最大重试次数: {task_id}")
        
        # 返回失败状态
        return False
        
    finally:
        if db_session:
            await db_session.close()


@router.post("/{chapter_id}/generate-stream", summary="AI创作章节内容（流式）")
async def generate_chapter_content_stream(
    chapter_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
    generate_request: ChapterGenerateRequest = ChapterGenerateRequest(),
    user_ai_service: AIService = Depends(get_user_ai_service_for_task("chapter_generate"))
):
    """
    根据大纲、前置章节内容和项目信息AI创作章节完整内容（流式返回）
    要求：必须按顺序生成，确保前置章节都已完成
    
    请求体参数：
    - style_id: 可选，指定使用的写作风格ID。不提供则不使用任何风格
    - target_word_count: 可选，目标字数，默认3000字，范围500-10000字
    - enable_mcp: 可选，是否启用MCP工具增强，默认True
    
    注意：此函数不使用依赖注入的db，而是在生成器内部创建独立的数据库会话
    以避免流式响应期间的连接泄漏问题
    """
    style_id = generate_request.style_id
    target_word_count = generate_request.target_word_count or 3000
    custom_model = generate_request.model if hasattr(generate_request, 'model') else None
    temp_narrative_perspective = generate_request.narrative_perspective if hasattr(generate_request, 'narrative_perspective') else None
    # 预先验证章节存在性（使用临时会话）
    async for temp_db in get_db(request):
        try:
            result = await temp_db.execute(
                select(Chapter).where(Chapter.id == chapter_id)
            )
            chapter = result.scalar_one_or_none()
            if not chapter:
                raise HTTPException(status_code=404, detail="章节不存在")
            
            # 检查前置条件
            can_generate, error_msg, previous_chapters = await check_prerequisites(temp_db, chapter)
            if not can_generate:
                raise HTTPException(status_code=400, detail=error_msg)
            
            # 保存前置章节数据供生成器使用
            previous_chapters_data = [
                {
                    'id': ch.id,
                    'chapter_number': ch.chapter_number,
                    'title': ch.title,
                    'content': ch.content
                }
                for ch in previous_chapters
            ]
        finally:
            await temp_db.close()
        break
    
    async def event_generator():
        # 在生成器内部创建独立的数据库会话
        db_session = None
        db_committed = False
        # 获取当前用户ID（在生成器外部就需要）
        current_user_id = getattr(request.state, "user_id", "system")
        
        # 初始化标准进度追踪器
        from app.utils.sse_response import WizardProgressTracker
        tracker = WizardProgressTracker("章节")
        
        try:
            yield await tracker.start()
            
            # 创建新的数据库会话
            async for db_session in get_db(request):
                # === 加载阶段 ===
                yield await tracker.loading("加载章节信息...", 0.2)
                
                # 重新获取章节信息
                chapter_result = await db_session.execute(
                    select(Chapter).where(Chapter.id == chapter_id)
                )
                current_chapter = chapter_result.scalar_one_or_none()
                if not current_chapter:
                    yield await tracker.error("章节不存在", 404)
                    return
            
                yield await tracker.loading("加载项目信息...", 0.4)
                
                # 获取项目信息
                project_result = await db_session.execute(
                    select(Project).where(Project.id == current_chapter.project_id)
                )
                project = project_result.scalar_one_or_none()
                if not project:
                    yield await tracker.error("项目不存在", 404)
                    return
                
                # 获取项目的大纲模式
                outline_mode = project.outline_mode if project else 'one-to-many'
                logger.info(f"📋 项目大纲模式: {outline_mode}")
                
                # 获取对应的大纲（优先使用 chapter.outline_id 直接关联）
                if current_chapter.outline_id:
                    outline_result = await db_session.execute(
                        select(Outline)
                        .where(Outline.id == current_chapter.outline_id)
                        .execution_options(populate_existing=True)
                    )
                else:
                    # 回退到按序号查找
                    outline_result = await db_session.execute(
                        select(Outline)
                        .where(Outline.project_id == current_chapter.project_id)
                        .where(Outline.order_index == current_chapter.chapter_number)
                        .execution_options(populate_existing=True)
                    )
                outline = outline_result.scalar_one_or_none()
                
                # 获取写作风格
                style_content = ""
                if style_id:
                    # 使用指定的风格
                    style_result = await db_session.execute(
                        select(WritingStyle).where(WritingStyle.id == style_id)
                    )
                    style = style_result.scalar_one_or_none()
                    if style:
                        # 验证风格是否可用：全局预设风格（user_id为NULL）或者当前用户的自定义风格
                        if style.user_id is None or style.user_id == current_user_id:
                            style_content = style.prompt_content or ""
                            style_type = "全局预设" if style.user_id is None else "用户自定义"
                            logger.info(f"使用指定风格: {style.name} ({style_type})")
                        else:
                            logger.warning(f"风格 {style_id} 不属于当前项目，无法使用")
                    else:
                        logger.warning(f"未找到风格 {style_id}")
                else:
                    logger.info("未指定写作风格，使用原始提示词")
                
                # 🚀 根据大纲模式选择独立的上下文构建器
                if outline_mode == 'one-to-one':
                    # ========== 1-1模式：使用独立的简化构建器 ==========
                    logger.info(f"🔧 [1-1模式] 使用 OneToOneContextBuilder")
                    context_builder = OneToOneContextBuilder(
                        memory_service=memory_service,
                        foreshadow_service=foreshadow_service
                    )
                    chapter_context = await context_builder.build(
                        chapter=current_chapter,
                        project=project,
                        outline=outline,
                        user_id=current_user_id,
                        db=db_session,
                        target_word_count=target_word_count
                    )
                    
                    # 日志输出统计信息
                    logger.info(f"📊 [1-1模式] 上下文统计:")
                    logger.info(f"  - 章节序号: {current_chapter.chapter_number}")
                    logger.info(f"  - 大纲长度: {chapter_context.context_stats.get('outline_length', 0)} 字符")
                    logger.info(f"  - 上一章内容: {chapter_context.context_stats.get('previous_content_length', 0)} 字符")
                    logger.info(f"  - 角色信息: {chapter_context.context_stats.get('characters_length', 0)} 字符")
                    logger.info(f"  - 伏笔提醒: {chapter_context.context_stats.get('foreshadow_length', 0)} 字符")
                    logger.info(f"  - 相关记忆: {chapter_context.context_stats.get('memories_length', 0)} 字符")
                    logger.info(f"  - 总长度: {chapter_context.context_stats.get('total_length', 0)} 字符")
                else:
                    # ========== 1-N模式：使用独立的完整构建器 ==========
                    logger.info(f"🔧 [1-N模式] 使用 OneToManyContextBuilder")
                    context_builder = OneToManyContextBuilder(
                        memory_service=memory_service,
                        foreshadow_service=foreshadow_service
                    )
                    chapter_context = await context_builder.build(
                        chapter=current_chapter,
                        project=project,
                        outline=outline,
                        user_id=current_user_id,
                        db=db_session,
                        style_content=style_content,
                        target_word_count=target_word_count,
                        temp_narrative_perspective=temp_narrative_perspective
                    )
                    
                    # 日志输出统计信息
                    logger.info(f"📊 [1-N模式] 上下文统计:")
                    logger.info(f"  - 章节序号: {current_chapter.chapter_number}")
                    logger.info(f"  - 衔接锚点: {chapter_context.context_stats.get('continuation_length', 0)} 字符")
                    logger.info(f"  - 角色信息: {chapter_context.context_stats.get('characters_length', 0)} 字符")
                    logger.info(f"  - 相关记忆: {chapter_context.context_stats.get('memories_length', 0)} 字符")
                    logger.info(f"  - 故事骨架: {chapter_context.context_stats.get('skeleton_length', 0)} 字符")
                    logger.info(f"  - 伏笔提醒: {chapter_context.context_stats.get('foreshadow_length', 0)} 字符")
                    logger.info(f"  - 总长度: {chapter_context.context_stats.get('total_length', 0)} 字符")
            
                yield await tracker.loading("上下文构建完成", 0.8)
                
                # 🎭 确定使用的叙事人称（临时指定 > 项目默认 > 系统默认）
                chapter_perspective = (
                    temp_narrative_perspective or
                    project.narrative_perspective or
                    '第三人称'
                )
                logger.info(f"📝 使用叙事人称: {chapter_perspective}")

                # 📌 项目硬设定（不可变事实）
                # 说明：
                # - 用于锁死“人名/封号官职/府邸称谓/亲缘关系/婚约对象/时间地点”等关键事实，防止长文生成写串。
                # - 当前复用 Project.world_rules 字段作为“硬设定承载位”（同时兼容原有世界规则）。
                hard_settings = (project.world_rules or "").strip()
                if not hard_settings:
                    hard_settings = "（暂无硬设定/世界规则。请在项目设置→世界规则中补充：关键人物称谓、婚约对象、势力归属等不可变事实。）"
                logger.info(f"📌 硬设定长度: {len(hard_settings)} 字符")
                
                # 🚀 根据大纲模式选择提示词模板和参数
                if outline_mode == 'one-to-one':
                    # 1-1模式
                    if chapter_context.continuation_point:
                        # 有上一章内容
                        template = await PromptService.get_template("CHAPTER_GENERATION_ONE_TO_ONE_NEXT", current_user_id, db_session)
                        base_prompt = PromptService.format_prompt(
                            template,
                            project_title=project.title,
                            chapter_number=current_chapter.chapter_number,
                            chapter_title=current_chapter.title,
                            chapter_outline=chapter_context.chapter_outline,
                            hard_settings=hard_settings,
                            target_word_count=target_word_count,
                            genre=project.genre or '未设定',
                            narrative_perspective=chapter_perspective,
                            previous_chapter_content=chapter_context.continuation_point,
                            previous_chapter_summary=chapter_context.previous_chapter_summary or '（无上一章摘要）',
                            characters_info=chapter_context.chapter_characters or '暂无角色信息',
                            chapter_careers=chapter_context.chapter_careers or '暂无职业信息',
                            foreshadow_reminders=chapter_context.foreshadow_reminders or '暂无需要关注的伏笔',
                            relevant_memories=chapter_context.relevant_memories or '暂无相关记忆'
                        )
                        logger.debug(f"创建第{current_chapter.chapter_number}章提示词: {base_prompt}")
                    else:
                        # 第一章
                        template = await PromptService.get_template("CHAPTER_GENERATION_ONE_TO_ONE", current_user_id, db_session)
                        base_prompt = PromptService.format_prompt(
                            template,
                            project_title=project.title,
                            chapter_number=current_chapter.chapter_number,
                            chapter_title=current_chapter.title,
                            chapter_outline=chapter_context.chapter_outline,
                            hard_settings=hard_settings,
                            target_word_count=target_word_count,
                            genre=project.genre or '未设定',
                            narrative_perspective=chapter_perspective,
                            characters_info=chapter_context.chapter_characters or '暂无角色信息',
                            chapter_careers=chapter_context.chapter_careers or '暂无职业信息',
                            foreshadow_reminders=chapter_context.foreshadow_reminders or '暂无需要关注的伏笔',
                            relevant_memories=chapter_context.relevant_memories or '暂无相关记忆'
                        )
                        logger.debug(f"创建第一章提示词: {base_prompt}")
                else:
                    # ========== 1-n模式：使用完整模板 ==========
                    if chapter_context.continuation_point:
                        # 有前置内容，使用 WITH_CONTEXT 模板
                        logger.info(f"📝 [1-n模式] 使用带上下文的模板（第{current_chapter.chapter_number}章）")
                        
                        # 提取上一章摘要
                        previous_summary = "（无上一章摘要，请根据锚点续写）"
                        if chapter_context.previous_chapter_summary:
                            previous_summary = chapter_context.previous_chapter_summary
                        
                        template = await PromptService.get_template("CHAPTER_GENERATION_ONE_TO_MANY_NEXT", current_user_id, db_session)
                        base_prompt = PromptService.format_prompt(
                            template,
                            project_title=project.title,
                            chapter_number=current_chapter.chapter_number,
                            chapter_title=current_chapter.title,
                            chapter_outline=chapter_context.chapter_outline,
                            hard_settings=hard_settings,
                            target_word_count=target_word_count,
                            continuation_point=chapter_context.continuation_point,
                            genre=project.genre or '未设定',
                            narrative_perspective=chapter_perspective,
                            characters_info=chapter_context.chapter_characters or '暂无角色信息',
                            chapter_careers=chapter_context.chapter_careers or '暂无职业信息',
                            foreshadow_reminders=chapter_context.foreshadow_reminders or '暂无需要关注的伏笔',
                            previous_chapter_summary=previous_summary,
                            recent_chapters_context=chapter_context.recent_chapters_context or '',
                            relevant_memories=chapter_context.relevant_memories or ''
                        )
                        logger.debug(f"创建第{current_chapter.chapter_number}章提示词: {base_prompt}")
                    else:
                        # 第1章，使用无前置内容模板
                        logger.info(f"📝 [1-n模式] 使用第一章模板")
                        template = await PromptService.get_template("CHAPTER_GENERATION_ONE_TO_MANY", current_user_id, db_session)
                        base_prompt = PromptService.format_prompt(
                            template,
                            project_title=project.title,
                            chapter_number=current_chapter.chapter_number,
                            chapter_title=current_chapter.title,
                            chapter_outline=chapter_context.chapter_outline,
                            hard_settings=hard_settings,
                            target_word_count=target_word_count,
                            genre=project.genre or '未设定',
                            narrative_perspective=chapter_perspective,
                            characters_info=chapter_context.chapter_characters or '暂无角色信息',
                            chapter_careers=chapter_context.chapter_careers or '暂无职业信息',
                            foreshadow_reminders=chapter_context.foreshadow_reminders or '暂无需要关注的伏笔',
                            relevant_memories=chapter_context.relevant_memories or '暂无相关记忆'
                        )
                        logger.debug(f"创建第一章提示词: {base_prompt}")
                
                # 应用写作风格
                if style_content:
                    prompt = WritingStyleManager.apply_style_to_prompt(base_prompt, style_content)
                else:
                    prompt = base_prompt
                
                # === 准备阶段 ===
                yield await tracker.preparing("准备AI提示词...")
                
                logger.info(f"开始AI流式创作章节 {chapter_id}")
                
                # 🎨 系统提示词拼装（最高优先级）
                # - 兼容用户自定义 system_prompt（若有）
                # - 注入“硬设定”以防止写串（最高优先级）
                # - 如有写作风格，则同样注入
                system_prompt_parts: list[str] = []
                base_system_prompt = (getattr(user_ai_service, "default_system_prompt", None) or "").strip()
                if base_system_prompt:
                    system_prompt_parts.append(base_system_prompt)
                
                if hard_settings:
                    system_prompt_parts.append(f"""【📌 项目硬设定（不可变事实） - 最高优先级】

{hard_settings}

⚠️ 必须严格遵守以上设定，严禁写错/混淆：人名、封号官职、府邸称谓、亲缘关系、婚约对象、时间地点。""")
                
                if style_content:
                    system_prompt_parts.append(f"""【🎨 写作风格要求 - 最高优先级】

{style_content}

⚠️ 请严格遵循上述写作风格要求进行创作，这是最重要的指令！
确保在整个章节创作过程中始终保持风格的一致性。""")
                
                system_prompt_with_style = "\n\n".join([p for p in system_prompt_parts if p.strip()]) or None
                if system_prompt_with_style:
                    logger.info(f"✅ 已注入系统提示词（硬设定:{len(hard_settings)}字符，风格:{len(style_content or '')}字符）")
                
                # 🔢 计算 max_tokens 限制
                # NOTE:
                # - 数据库字数统计使用 len(full_content)（近似“中文字符数”）。
                # - 许多模型的 token/字 比例并不稳定（尤其中文可能出现多字合并 token），
                #   之前使用 3 倍系数会让模型严重超写，导致项目总字数失控。
                # - 这里改为更保守的系数，用 max_tokens 作为“硬上限”，让模型必须在
                #   接近 target_word_count 处收束。
                # NOTE: max_tokens 是硬上限，过小会导致正文在句子中间被截断。
                # 这里使用较保守但能显著降低“写到一半被截断”的系数。
                calculated_max_tokens = int(target_word_count * 2.0)
                calculated_max_tokens = max(800, min(calculated_max_tokens, 12000))  # 限制在 800-12000 之间
                logger.info(f"📊 目标字数: {target_word_count}, max_tokens(硬上限): {calculated_max_tokens}")
                
                # 准备生成参数
                generate_kwargs = {
                    "prompt": prompt,
                    "system_prompt": system_prompt_with_style,
                    "tool_choice": "required",
                    "max_tokens": calculated_max_tokens  # 添加 max_tokens 限制
                }
                if custom_model:
                    logger.info(f"  使用自定义模型: {custom_model}")
                    generate_kwargs["model"] = custom_model
                    # 注意：这里使用用户配置的AI服务，模型参数会覆盖默认模型
                    # 如果需要切换provider，需要在前端传递provider参数
                
                # === 生成阶段 ===
                full_content = ""
                chunk_count = 0
                
                yield await tracker.generating(
                    current_chars=0,
                    estimated_total=target_word_count
                )
                
                async for chunk in user_ai_service.generate_text_stream(**generate_kwargs):
                    full_content += chunk
                    chunk_count += 1
                    
                    # 发送内容块
                    yield await tracker.generating_chunk(chunk)
                    
                    # 每5个chunk发送一次进度更新
                    if chunk_count % 5 == 0:
                        yield await tracker.generating(
                            current_chars=len(full_content),
                            estimated_total=target_word_count,
                            message=f'正在创作中... 已生成 {len(full_content)} 字'
                        )
                    
                    # 每20个chunk发送心跳
                    if chunk_count % 20 == 0:
                        yield await tracker.heartbeat()
                    
                    await asyncio.sleep(0)  # 让出控制权
                
                # === 保存阶段 ===
                yield await tracker.saving("正在保存章节...", 0.3)
                
                # 更新章节内容到数据库
                old_word_count = current_chapter.word_count or 0
                current_chapter.content = full_content
                new_word_count = len(full_content)
                current_chapter.word_count = new_word_count
                current_chapter.status = "completed"
                
                # 更新项目字数
                project.current_words = project.current_words - old_word_count + new_word_count
                
                # 记录生成历史
                history = GenerationHistory(
                    project_id=current_chapter.project_id,
                    chapter_id=current_chapter.id,
                    prompt=f"创作章节: 第{current_chapter.chapter_number}章 {current_chapter.title}",
                    generated_content=full_content[:500] if len(full_content) > 500 else full_content,
                    model="default"
                )
                db_session.add(history)
                
                await db_session.commit()
                db_committed = True
                await db_session.refresh(current_chapter)
                
                logger.info(f"成功创作章节 {chapter_id}，共 {new_word_count} 字")
                
                # 🔮 章节生成后自动标记计划在本章埋入的伏笔
                try:
                    plant_result = await foreshadow_service.auto_plant_pending_foreshadows(
                        db=db_session,
                        project_id=project.id,
                        chapter_id=chapter_id,
                        chapter_number=current_chapter.chapter_number,
                        chapter_content=full_content
                    )
                    if plant_result.get('planted_count', 0) > 0:
                        logger.info(f"🔮 自动标记伏笔已埋入: {plant_result['planted_count']}个")
                except Exception as plant_error:
                    logger.warning(f"⚠️ 自动标记伏笔埋入失败: {str(plant_error)}")
                
                # 创建分析任务
                analysis_task = AnalysisTask(
                    chapter_id=chapter_id,
                    user_id=current_user_id,
                    project_id=project.id,
                    status='pending',
                    progress=0
                )
                db_session.add(analysis_task)
                await db_session.commit()
                await db_session.refresh(analysis_task)
                
                task_id = analysis_task.id
                logger.info(f"📋 已创建分析任务: {task_id}")
                
                # 短暂延迟确保SQLite WAL完成写入
                await asyncio.sleep(0.05)
                
                # 直接启动后台分析（并发执行）
                background_tasks.add_task(
                    analyze_chapter_background,
                    chapter_id=chapter_id,
                    user_id=current_user_id,
                    project_id=project.id,
                    task_id=task_id,
                    ai_service=user_ai_service
                )
                
                yield await tracker.saving("章节保存完成", 0.8)
                
                # === 完成阶段 ===
                yield await tracker.complete("创作完成！")
                
                # 发送结果数据
                yield await tracker.result({
                    'word_count': new_word_count,
                    'analysis_task_id': task_id
                })
                
                # 发送分析开始事件（使用自定义事件）
                yield await SSEResponse.send_event(
                    event='analysis_started',
                    data={
                        'task_id': task_id,
                        'message': '章节分析已开始'
                    }
                )
                
                # 发送完成信号
                yield await tracker.done()
                
                break  # 退出async for db_session循环
        
        except GeneratorExit:
            # SSE连接断开
            logger.warning("章节生成器被提前关闭（SSE断开）")
            if db_session and not db_committed:
                try:
                    if db_session.in_transaction():
                        await db_session.rollback()
                        logger.info("章节生成事务已回滚（GeneratorExit）")
                except Exception as e:
                    logger.error(f"GeneratorExit回滚失败: {str(e)}")
        except Exception as e:
            logger.error(f"流式创作章节失败: {str(e)}")
            if db_session and not db_committed:
                try:
                    if db_session.in_transaction():
                        await db_session.rollback()
                        logger.info("章节生成事务已回滚（异常）")
                except Exception as rollback_error:
                    logger.error(f"回滚失败: {str(rollback_error)}")
            yield await tracker.error(str(e))
        finally:
            # 确保数据库会话被正确关闭
            if db_session:
                try:
                    # 最后检查：确保没有未提交的事务
                    if not db_committed and db_session.in_transaction():
                        await db_session.rollback()
                        logger.warning("在finally中发现未提交事务，已回滚")
                    
                    await db_session.close()
                    logger.info("数据库会话已关闭")
                except Exception as close_error:
                    logger.error(f"关闭数据库会话失败: {str(close_error)}")
                    # 强制关闭
                    try:
                        await db_session.close()
                    except:
                        pass
    
    return create_sse_response(event_generator())


@router.get("/{chapter_id}/analysis/status", summary="查询章节分析任务状态")
async def get_analysis_task_status(
    chapter_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """
    查询指定章节的最新分析任务状态
    
    自动恢复机制：
    - 如果任务状态为 running 且超过阈值仍未完成，自动标记为 failed（防止任务卡死导致 UI 永久等待）
    - 如果任务状态为 pending 且超过阈值仍未启动，自动标记为 failed
    
    返回:
    - has_task: 是否存在分析任务
    - task_id: 任务ID（如果存在）
    - status: pending/running/completed/failed/none（如果不存在则为none）
    - progress: 0-100
    - error_message: 错误信息(如果失败)
    - auto_recovered: 是否被自动恢复
    - created_at: 创建时间
    - completed_at: 完成时间
    
    注意：当章节不存在或无权访问时返回404，当没有分析任务时返回has_task=false
    """
    from datetime import timedelta
    
    # 先获取章节以验证存在性和权限
    chapter_result = await db.execute(
        select(Chapter).where(Chapter.id == chapter_id)
    )
    chapter = chapter_result.scalar_one_or_none()
    
    if not chapter:
        raise HTTPException(status_code=404, detail="章节不存在")
    
    # 验证用户权限
    user_id = getattr(request.state, 'user_id', None)
    await verify_project_access(chapter.project_id, user_id, db)
    
    # 获取该章节最新的分析任务
    result = await db.execute(
        select(AnalysisTask)
        .where(AnalysisTask.chapter_id == chapter_id)
        .order_by(AnalysisTask.created_at.desc())
        .limit(1)
    )
    task = result.scalar_one_or_none()
    
    if not task:
        # 返回无任务状态，而不是抛出404错误
        return {
            "has_task": False,
            "chapter_id": chapter_id,
            "status": "none",
            "progress": 0,
            "error_message": None,
            "auto_recovered": False,
            "task_id": None,
            "created_at": None,
            "started_at": None,
            "completed_at": None,

            # 记忆/向量统计（无任务时为空）
            "memory_extracted_count": 0,
            "vector_expected_count": 0,
            "vector_added_count": 0,
            "vector_skipped_count": 0,
            "vector_error_message": None,
            "vector_embed_id": None,
            "vector_collection": None,
            "has_analysis_result": False,
            "memories_db_count": 0,
        }
    
    auto_recovered = False
    current_time = datetime.now()
    
    # 自动恢复卡住的任务
    # ⚠️ 注意：
    # - 章节分析会发起一次或多次大模型调用（含重试），且可能包含远端 embedding 写入；
    #   实际耗时经常超过 3 分钟，因此阈值不能过小，否则会误判“卡住”。
    # - started_at 仅表示任务开始时间，并不代表“最后一次进度更新时间”，因此这里只能使用
    #   一个相对保守的超时时间。
    if task.status == 'running':
        # 检查是否正在重试（error_message 包含"重试"信息）
        is_retrying = task.error_message and '重试' in task.error_message
        # 如果正在重试，给予更长的超时时间（30分钟），否则15分钟
        timeout_minutes = 30 if is_retrying else 15
        
        # 如果任务在running状态超过超时时间，标记为失败
        if task.started_at and (current_time - task.started_at) > timedelta(minutes=timeout_minutes):
            task.status = 'failed'
            task.error_message = f'任务超时（超过{timeout_minutes}分钟未完成，已自动恢复）'
            task.completed_at = current_time
            task.progress = 0
            auto_recovered = True
            await db.commit()
            await db.refresh(task)
            logger.warning(f"🔄 自动恢复卡住的任务: {task.id}, 章节: {chapter_id}")
    
    elif task.status == 'pending':
        # 如果任务在pending状态超过5分钟仍未开始，标记为失败
        if task.created_at and (current_time - task.created_at) > timedelta(minutes=5):
            task.status = 'failed'
            task.error_message = '任务启动超时（超过5分钟未启动，已自动恢复）'
            task.completed_at = current_time
            task.progress = 0
            auto_recovered = True
            await db.commit()
            await db.refresh(task)
            logger.warning(f"🔄 自动恢复未启动的任务: {task.id}, 章节: {chapter_id}")

    # 若任务已完成，但仍残留“超时/重试”错误信息（历史遗留或被误判后又成功完成），自动清理。
    if task.status == "completed" and task.error_message:
        if ("任务超时" in task.error_message) or ("正在重试" in task.error_message):
            try:
                task.error_message = None
                await db.commit()
                await db.refresh(task)
            except Exception as e:
                logger.warning(f"⚠️ 清理已完成任务的 error_message 失败（不影响返回）: {e}")
    
    # 补充：分析结果/记忆数量（用于前端标记异常）
    has_analysis_result = False
    memories_db_count = 0
    if task.status == "completed":
        try:
            analysis_exists = await db.execute(
                select(PlotAnalysis.id).where(PlotAnalysis.chapter_id == chapter_id).limit(1)
            )
            has_analysis_result = bool(analysis_exists.scalar_one_or_none())

            from sqlalchemy import func

            mem_count_result = await db.execute(
                select(func.count()).select_from(StoryMemory).where(StoryMemory.chapter_id == chapter_id)
            )
            memories_db_count = int(mem_count_result.scalar() or 0)
        except Exception as e:
            logger.warning(f"⚠️ 获取章节分析/记忆数量失败（不影响状态返回）: {e}")

    return {
        "has_task": True,
        "task_id": task.id,
        "chapter_id": task.chapter_id,
        "status": task.status,
        "progress": task.progress,
        "error_message": task.error_message,
        "auto_recovered": auto_recovered,
        "created_at": task.created_at.isoformat() if task.created_at else None,
        "started_at": task.started_at.isoformat() if task.started_at else None,
        "completed_at": task.completed_at.isoformat() if task.completed_at else None,

        # 记忆/向量统计
        "memory_extracted_count": int(getattr(task, "memory_extracted_count", 0) or 0),
        "vector_expected_count": int(getattr(task, "vector_expected_count", 0) or 0),
        "vector_added_count": int(getattr(task, "vector_added_count", 0) or 0),
        "vector_skipped_count": int(getattr(task, "vector_skipped_count", 0) or 0),
        "vector_error_message": getattr(task, "vector_error_message", None),
        "vector_embed_id": getattr(task, "vector_embed_id", None),
        "vector_collection": getattr(task, "vector_collection", None),
        "has_analysis_result": has_analysis_result,
        "memories_db_count": memories_db_count,
    }


@router.get("/{chapter_id}/analysis", summary="获取章节分析结果")
async def get_chapter_analysis(
    chapter_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """
    获取章节的完整分析结果
    
    返回:
    - analysis_data: 完整的分析数据(JSON)
    - summary: 分析摘要文本
    - memories: 提取的记忆列表
    - created_at: 分析时间
    """
    # 先获取章节以验证权限
    chapter_result_check = await db.execute(
        select(Chapter).where(Chapter.id == chapter_id)
    )
    chapter_check = chapter_result_check.scalar_one_or_none()
    if chapter_check:
        # 验证用户权限
        user_id = getattr(request.state, 'user_id', None)
        await verify_project_access(chapter_check.project_id, user_id, db)
    
    # 获取分析结果
    analysis_result = await db.execute(
        select(PlotAnalysis)
        .where(PlotAnalysis.chapter_id == chapter_id)
        .order_by(PlotAnalysis.created_at.desc())
        .limit(1)
    )
    analysis = analysis_result.scalar_one_or_none()
    
    if not analysis:
        raise HTTPException(status_code=404, detail="该章节暂无分析结果")
    
    # 获取相关记忆
    memories_result = await db.execute(
        select(StoryMemory)
        .where(StoryMemory.chapter_id == chapter_id)
        .order_by(StoryMemory.importance_score.desc())
    )
    memories = memories_result.scalars().all()
    
    return {
        "chapter_id": chapter_id,
        "analysis": analysis.to_dict(),  # 使用to_dict()方法
        "memories": [
            {
                "id": mem.id,
                "type": mem.memory_type,
                "title": mem.title,
                "content": mem.content,
                "importance": mem.importance_score,
                "tags": mem.tags,
                "is_foreshadow": mem.is_foreshadow,
                "position": mem.chapter_position,
                "related_characters": mem.related_characters
            }
            for mem in memories
        ],
        "created_at": analysis.created_at.isoformat() if analysis.created_at else None
    }


@router.get("/{chapter_id}/annotations", summary="获取章节标注数据")
async def get_chapter_annotations(
    chapter_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """
    获取章节的标注数据（用于前端展示标注）
    
    返回格式化的标注列表，包含精确位置信息
    适用于章节内容的可视化标注展示
    """
    # 验证用户权限
    user_id = getattr(request.state, 'user_id', None)
    
    # 获取章节
    chapter_result = await db.execute(
        select(Chapter).where(Chapter.id == chapter_id)
    )
    chapter = chapter_result.scalar_one_or_none()
    
    if not chapter:
        raise HTTPException(status_code=404, detail="章节不存在")
    
    # 验证项目访问权限
    await verify_project_access(chapter.project_id, user_id, db)
    
    # 获取分析结果
    analysis_result = await db.execute(
        select(PlotAnalysis)
        .where(PlotAnalysis.chapter_id == chapter_id)
        .order_by(PlotAnalysis.created_at.desc())
        .limit(1)
    )
    analysis = analysis_result.scalar_one_or_none()
    
    # 获取记忆
    memories_result = await db.execute(
        select(StoryMemory)
        .where(StoryMemory.chapter_id == chapter_id)
        .order_by(StoryMemory.importance_score.desc())
    )
    memories = memories_result.scalars().all()
    
    # 构建标注数据
    annotations = []
    
    for mem in memories:
        # 优先从数据库读取位置信息
        position = mem.chapter_position if mem.chapter_position is not None else -1
        length = mem.text_length if hasattr(mem, 'text_length') and mem.text_length is not None else 0
        metadata_extra = {}
        
        # 如果数据库中没有位置信息，尝试从分析数据中重新计算
        if position == -1 and analysis and chapter.content:
            # 根据记忆类型从分析数据中查找对应项
            if mem.memory_type == 'hook' and analysis.hooks:
                for hook in analysis.hooks:
                    # 通过标题或内容匹配
                    if mem.title and hook.get('type') in mem.title:
                        keyword = hook.get('keyword', '')
                        if keyword:
                            pos = chapter.content.find(keyword)
                            if pos != -1:
                                position = pos
                                length = len(keyword)
                        metadata_extra["strength"] = hook.get('strength', 5)
                        metadata_extra["position_desc"] = hook.get('position', '')
                        break
            
            elif mem.memory_type == 'foreshadow' and analysis.foreshadows:
                for foreshadow in analysis.foreshadows:
                    if foreshadow.get('content') in mem.content:
                        keyword = foreshadow.get('keyword', '')
                        if keyword:
                            pos = chapter.content.find(keyword)
                            if pos != -1:
                                position = pos
                                length = len(keyword)
                        metadata_extra["foreshadow_type"] = foreshadow.get('type', 'planted')
                        metadata_extra["strength"] = foreshadow.get('strength', 5)
                        break
            
            elif mem.memory_type == 'plot_point' and analysis.plot_points:
                for plot_point in analysis.plot_points:
                    if plot_point.get('content') in mem.content:
                        keyword = plot_point.get('keyword', '')
                        if keyword:
                            pos = chapter.content.find(keyword)
                            if pos != -1:
                                position = pos
                                length = len(keyword)
                        break
        else:
            # 如果数据库有位置，也从分析数据中提取额外的元数据
            if analysis:
                if mem.memory_type == 'hook' and analysis.hooks:
                    for hook in analysis.hooks:
                        if mem.title and hook.get('type') in mem.title:
                            metadata_extra["strength"] = hook.get('strength', 5)
                            metadata_extra["position_desc"] = hook.get('position', '')
                            break
                
                elif mem.memory_type == 'foreshadow' and analysis.foreshadows:
                    for foreshadow in analysis.foreshadows:
                        if foreshadow.get('content') in mem.content:
                            metadata_extra["foreshadow_type"] = foreshadow.get('type', 'planted')
                            metadata_extra["strength"] = foreshadow.get('strength', 5)
                            break
        
        annotation = {
            "id": mem.id,
            "type": mem.memory_type,
            "title": mem.title,
            "content": mem.content,
            "importance": mem.importance_score or 0.5,
            "position": position,
            "length": length,
            "tags": mem.tags or [],
            "metadata": {
                "is_foreshadow": mem.is_foreshadow,
                "related_characters": mem.related_characters or [],
                "related_locations": mem.related_locations or [],
                **metadata_extra
            }
        }
        
        annotations.append(annotation)
    
    return {
        "chapter_id": chapter_id,
        "chapter_number": chapter.chapter_number,
        "title": chapter.title,
        "word_count": chapter.word_count or 0,
        "annotations": annotations,
        "has_analysis": analysis is not None,
        "summary": {
            "total_annotations": len(annotations),
            "hooks": len([a for a in annotations if a["type"] == "hook"]),
            "foreshadows": len([a for a in annotations if a["type"] == "foreshadow"]),
            "plot_points": len([a for a in annotations if a["type"] == "plot_point"]),
            "character_events": len([a for a in annotations if a["type"] == "character_event"])
        }
    }


@router.post("/{chapter_id}/analyze", summary="手动触发章节分析")
async def trigger_chapter_analysis(
    chapter_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    user_ai_service: AIService = Depends(get_user_ai_service_for_task("chapter_analysis"))
):
    """
    手动触发章节分析(用于重新分析或分析旧章节)
    """
    # 从请求中获取用户ID
    user_id = getattr(request.state, "user_id", None)
    if not user_id:
        raise HTTPException(status_code=401, detail="未登录")
    
    # 验证章节存在
    chapter_result = await db.execute(
        select(Chapter).where(Chapter.id == chapter_id)
    )
    chapter = chapter_result.scalar_one_or_none()
    
    if not chapter:
        raise HTTPException(status_code=404, detail="章节不存在")
    
    if not chapter.content or chapter.content.strip() == "":
        raise HTTPException(status_code=400, detail="章节内容为空，无法分析")
    
    # 获取项目信息
    project_result = await db.execute(
        select(Project).where(Project.id == chapter.project_id)
    )
    project = project_result.scalar_one_or_none()
    
    if not project:
        raise HTTPException(status_code=404, detail="项目不存在")
    
    # 创建分析任务
    analysis_task = AnalysisTask(
        chapter_id=chapter_id,
        user_id=user_id,
        project_id=project.id,
        status='pending',
        progress=0
    )
    db.add(analysis_task)
    await db.commit()
    
    task_id = analysis_task.id
    logger.info(f"📋 创建分析任务: {task_id}, 章节: {chapter_id}")
    
    # 刷新数据库会话，确保其他会话可以看到新任务
    await db.refresh(analysis_task)
    
    # 短暂延迟确保SQLite WAL完成写入（让其他会话可见）
    await asyncio.sleep(3)
    
    # 直接启动后台分析（并发执行）
    background_tasks.add_task(
        analyze_chapter_background,
        chapter_id=chapter_id,
        user_id=user_id,
        project_id=project.id,
        task_id=task_id,
        ai_service=user_ai_service
    )
    
    return {
        "task_id": task_id,
        "chapter_id": chapter_id,
        "status": "pending",
        "message": "分析任务已创建并开始执行"
    }


@router.post("/{chapter_id}/memories/reindex", summary="重建章节记忆向量索引")
async def reindex_chapter_memories(
    chapter_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    从关系库 story_memories 重建该章节在 ChromaDB 中的向量索引。

    用途：
    - 远端 embedding 被拦截/限流导致写入失败或部分失败
    - 切换 embedding 配置后需要重新写入向量
    """
    user_id = getattr(request.state, "user_id", None)
    if not user_id:
        raise HTTPException(status_code=401, detail="未登录")

    chapter_result = await db.execute(select(Chapter).where(Chapter.id == chapter_id))
    chapter = chapter_result.scalar_one_or_none()
    if not chapter:
        raise HTTPException(status_code=404, detail="章节不存在")

    await verify_project_access(chapter.project_id, user_id, db)

    memories_result = await db.execute(select(StoryMemory).where(StoryMemory.chapter_id == chapter_id))
    memories = memories_result.scalars().all()

    if not memories:
        return {
            "success": True,
            "chapter_id": chapter_id,
            "requested": 0,
            "added": 0,
            "skipped": 0,
            "errors": [],
            "message": "该章节暂无记忆，无需重建索引",
        }

    # 先清理该章节旧向量（跨 collection）
    try:
        await memory_service.delete_chapter_memories(
            user_id=user_id,
            project_id=chapter.project_id,
            chapter_id=chapter_id,
        )
    except Exception as e:
        logger.warning(f"⚠️ 重建索引前清理旧向量失败（继续重建）: {e}")

    memory_records: List[Dict[str, Any]] = []
    for mem in memories:
        memory_records.append(
            {
                "id": mem.id,
                "content": mem.content,
                "type": mem.memory_type,
                "metadata": {
                    "chapter_id": chapter_id,
                    "chapter_number": chapter.chapter_number,
                    "importance_score": mem.importance_score or 0.5,
                    "tags": mem.tags or [],
                    "title": mem.title or "",
                    "is_foreshadow": mem.is_foreshadow or 0,
                    "related_characters": mem.related_characters or [],
                    "related_locations": mem.related_locations or [],
                },
            }
        )

    vector_res = await memory_service.batch_add_memories(
        user_id=user_id,
        project_id=chapter.project_id,
        memories=memory_records,
        db=db,
    )

    # 同步更新该章节最新分析任务的向量写入状态（若存在）
    try:
        task_result = await db.execute(
            select(AnalysisTask)
            .where(AnalysisTask.chapter_id == chapter_id)
            .order_by(AnalysisTask.created_at.desc())
            .limit(1)
        )
        task = task_result.scalar_one_or_none()
        if task:
            task.vector_expected_count = int(vector_res.get("requested") or len(memory_records))
            task.vector_added_count = int(vector_res.get("added") or 0)
            task.vector_skipped_count = int(vector_res.get("skipped") or 0)
            task.vector_error_message = vector_res.get("error_message")
            task.vector_embed_id = vector_res.get("embed_id")
            task.vector_collection = vector_res.get("collection")
            await db.commit()
    except Exception as e:
        logger.warning(f"⚠️ 重建索引后更新分析任务统计失败（不影响接口返回）: {e}")

    return {
        "success": True,
        "chapter_id": chapter_id,
        **vector_res,
        "message": "重建索引完成",
    }



def calculate_estimated_time(
    chapter_count: int,
    target_word_count: int,
    enable_analysis: bool
) -> int:
    """
    计算预估耗时（分钟）
    
    基准：
    - 生成3000字约需2分钟
    - 分析约需1分钟
    """
    generation_time_per_chapter = (target_word_count / 3000) * 2
    analysis_time_per_chapter = 1 if enable_analysis else 0
    
    total_time = chapter_count * (generation_time_per_chapter + analysis_time_per_chapter)
    
    return max(1, int(total_time))


@router.post("/project/{project_id}/batch-generate", response_model=BatchGenerateResponse, summary="批量顺序生成章节内容")
async def batch_generate_chapters_in_order(
    project_id: str,
    batch_request: BatchGenerateRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    user_ai_service: AIService = Depends(get_user_ai_service_for_task("chapter_generate"))
):
    """
    从指定章节开始，按顺序批量生成指定数量的章节
    
    特性：
    1. 严格按章节序号顺序生成（不可跳过）
    2. 自动检测起始章节是否可生成
    3. 可选同步分析（影响耗时和质量）
    4. 失败后终止，不继续后续章节
    """
    user_id = getattr(request.state, "user_id", None)
    if not user_id:
        raise HTTPException(status_code=401, detail="未登录")
    
    # 验证项目存在和用户权限
    project = await verify_project_access(project_id, user_id, db)
    
    # 获取项目的所有章节，按序号排序
    result = await db.execute(
        select(Chapter)
        .where(Chapter.project_id == project_id)
        .order_by(Chapter.chapter_number)
    )
    all_chapters = result.scalars().all()
    
    if not all_chapters:
        raise HTTPException(status_code=404, detail="项目没有章节")
    
    # 计算要生成的章节范围
    start_number = batch_request.start_chapter_number
    end_number = start_number + batch_request.count - 1
    
    # 筛选出要生成的章节
    chapters_to_generate = [
        ch for ch in all_chapters
        if start_number <= ch.chapter_number <= end_number
    ]
    
    if not chapters_to_generate:
        raise HTTPException(status_code=404, detail="指定范围内没有章节")
    
    # 验证起始章节的前置条件
    first_chapter = chapters_to_generate[0]
    can_generate, error_msg, _ = await check_prerequisites(db, first_chapter)
    if not can_generate:
        raise HTTPException(status_code=400, detail=f"起始章节无法生成：{error_msg}")
    
    # 创建批量生成任务
    batch_task = BatchGenerationTask(
        project_id=project_id,
        user_id=user_id,
        start_chapter_number=start_number,
        chapter_count=len(chapters_to_generate),
        chapter_ids=[ch.id for ch in chapters_to_generate],
        style_id=batch_request.style_id,
        target_word_count=batch_request.target_word_count,
        enable_analysis=batch_request.enable_analysis,
        max_retries=batch_request.max_retries,
        status='pending',
        total_chapters=len(chapters_to_generate),
        completed_chapters=0,
        failed_chapters=[],
        current_retry_count=0
    )
    db.add(batch_task)
    await db.commit()
    await db.refresh(batch_task)
    
    batch_id = batch_task.id
    
    # 计算预估耗时
    estimated_time = calculate_estimated_time(
        chapter_count=len(chapters_to_generate),
        target_word_count=batch_request.target_word_count,
        enable_analysis=batch_request.enable_analysis
    )
    
    logger.info(f"📦 创建批量生成任务: {batch_id}, 章节: 第{start_number}-{end_number}章, 预估耗时: {estimated_time}分钟")
    
    # 启动后台批量生成任务，传递model参数
    background_tasks.add_task(
        execute_batch_generation_in_order,
        batch_id=batch_id,
        user_id=user_id,
        ai_service=user_ai_service,
        custom_model=batch_request.model
    )
    
    return BatchGenerateResponse(
        batch_id=batch_id,
        message=f"批量生成任务已创建，将生成 {len(chapters_to_generate)} 个章节",
        chapters_to_generate=[
            {
                "id": ch.id,
                "chapter_number": ch.chapter_number,
                "title": ch.title
            }
            for ch in chapters_to_generate
        ],
        estimated_time_minutes=estimated_time
    )


@router.get("/batch-generate/{batch_id}/status", response_model=BatchGenerateStatusResponse, summary="查询批量生成任务状态")
async def get_batch_generation_status(
    batch_id: str,
    db: AsyncSession = Depends(get_db)
):
    """查询批量生成任务的状态和进度"""
    result = await db.execute(
        select(BatchGenerationTask).where(BatchGenerationTask.id == batch_id)
    )
    task = result.scalar_one_or_none()
    
    if not task:
        raise HTTPException(status_code=404, detail="批量生成任务不存在")
    
    return BatchGenerateStatusResponse(
        batch_id=task.id,
        status=task.status,
        total=task.total_chapters,
        completed=task.completed_chapters,
        current_chapter_id=task.current_chapter_id,
        current_chapter_number=task.current_chapter_number,
        current_retry_count=task.current_retry_count,
        max_retries=task.max_retries,
        failed_chapters=task.failed_chapters or [],
        created_at=task.created_at.isoformat() if task.created_at else None,
        started_at=task.started_at.isoformat() if task.started_at else None,
        completed_at=task.completed_at.isoformat() if task.completed_at else None,
        error_message=task.error_message
    )


@router.get("/project/{project_id}/batch-generate/active", summary="获取项目当前运行中的批量生成任务")
async def get_active_batch_generation(
    project_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """
    获取项目当前运行中的批量生成任务
    用于页面刷新后恢复任务状态
    """
    # 验证用户权限
    user_id = getattr(request.state, 'user_id', None)
    await verify_project_access(project_id, user_id, db)
    
    result = await db.execute(
        select(BatchGenerationTask)
        .where(BatchGenerationTask.project_id == project_id)
        .where(BatchGenerationTask.status.in_(['pending', 'running']))
        .order_by(BatchGenerationTask.created_at.desc())
        .limit(1)
    )
    task = result.scalar_one_or_none()
    
    if not task:
        return {
            "has_active_task": False,
            "task": None
        }
    
    return {
        "has_active_task": True,
        "task": {
            "batch_id": task.id,
            "status": task.status,
            "total": task.total_chapters,
            "completed": task.completed_chapters,
            "current_chapter_id": task.current_chapter_id,
            "current_chapter_number": task.current_chapter_number,
            "created_at": task.created_at.isoformat() if task.created_at else None,
            "started_at": task.started_at.isoformat() if task.started_at else None
        }
    }


@router.post("/batch-generate/{batch_id}/cancel", summary="取消批量生成任务")
async def cancel_batch_generation(
    batch_id: str,
    db: AsyncSession = Depends(get_db)
):
    """取消正在进行的批量生成任务"""
    result = await db.execute(
        select(BatchGenerationTask).where(BatchGenerationTask.id == batch_id)
    )
    task = result.scalar_one_or_none()
    
    if not task:
        raise HTTPException(status_code=404, detail="批量生成任务不存在")
    
    if task.status in ['completed', 'failed', 'cancelled']:
        raise HTTPException(status_code=400, detail=f"任务已处于 {task.status} 状态，无法取消")
    
    task.status = 'cancelled'
    task.completed_at = datetime.now()
    await db.commit()
    
    logger.info(f"🛑 批量生成任务已取消: {batch_id}")
    
    return {
        "message": "批量生成任务已取消",
        "batch_id": batch_id,
        "completed_chapters": task.completed_chapters,
        "total_chapters": task.total_chapters
    }


async def execute_batch_generation_in_order(
    batch_id: str,
    user_id: str,
    ai_service: AIService,
    custom_model: Optional[str] = None
):
    """
    按顺序执行批量生成任务（后台任务）
    - 严格按章节序号顺序
    - 任一章节失败则终止后续生成
    - 可选同步分析
    """
    db_session = None
    write_lock = await get_db_write_lock(user_id)
    
    try:
        logger.info(f"📦 开始执行顺序批量生成任务: {batch_id}")
        
        # 创建独立数据库会话
        from app.database import get_engine
        from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession
        
        engine = await get_engine(user_id)
        AsyncSessionLocal = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        db_session = AsyncSessionLocal()
        
        # 获取任务
        task_result = await db_session.execute(
            select(BatchGenerationTask).where(BatchGenerationTask.id == batch_id)
        )
        task = task_result.scalar_one_or_none()
        
        if not task:
            logger.error(f"❌ 批量生成任务不存在: {batch_id}")
            return
        
        # 更新任务状态为运行中
        async with write_lock:
            task.status = 'running'
            task.started_at = datetime.now()
            await db_session.commit()
        
        # 维护上一章的摘要，用于传递给下一章（防重复上下文）
        last_generated_summary = None

        # 按顺序生成每个章节
        for idx, chapter_id in enumerate(task.chapter_ids, 1):
            # 检查任务是否被取消
            await db_session.refresh(task)
            if task.status == 'cancelled':
                logger.info(f"🛑 批量生成任务已被取消: {batch_id}")
                return
            
            # 更新当前章节
            async with write_lock:
                task.current_chapter_id = chapter_id
                task.current_retry_count = 0  # 重置重试计数
                await db_session.commit()
            
            # 重试循环
            retry_count = 0
            chapter_success = False
            chapter = None
            last_error = None
            
            while retry_count <= task.max_retries and not chapter_success:
                try:
                    # 获取章节信息
                    chapter_result = await db_session.execute(
                        select(Chapter).where(Chapter.id == chapter_id)
                    )
                    chapter = chapter_result.scalar_one_or_none()
                    
                    if not chapter:
                        raise Exception(f"章节 {chapter_id} 不存在")
                    
                    # 更新当前章节序号和重试次数
                    async with write_lock:
                        task.current_chapter_number = chapter.chapter_number
                        task.current_retry_count = retry_count
                        await db_session.commit()
                    
                    if retry_count > 0:
                        logger.info(f"🔄 [{idx}/{task.total_chapters}] 重试生成章节 (第{retry_count}次): 第{chapter.chapter_number}章 《{chapter.title}》")
                    else:
                        logger.info(f"📝 [{idx}/{task.total_chapters}] 开始生成章节: 第{chapter.chapter_number}章 《{chapter.title}》")
                    
                    # 检查前置条件（每次都检查，确保顺序性）
                    can_generate, error_msg, _ = await check_prerequisites(db_session, chapter)
                    if not can_generate:
                        raise Exception(f"前置条件不满足: {error_msg}")
                    
                    # 生成章节内容（复用现有流式生成逻辑的核心部分），传递model参数
                    # 并获取生成后的摘要（如果生成函数支持返回）
                    generated_summary = await generate_single_chapter_for_batch(
                        db_session=db_session,
                        chapter=chapter,
                        user_id=user_id,
                        style_id=task.style_id,
                        target_word_count=task.target_word_count,
                        ai_service=ai_service,
                        write_lock=write_lock,
                        custom_model=custom_model,
                        previous_summary_context=last_generated_summary
                    )
                    
                    # 更新上一章摘要，供下一章使用
                    if generated_summary:
                        last_generated_summary = f"第{chapter.chapter_number}章《{chapter.title}》：{generated_summary}"
                        logger.info(f"📝 已更新上一章摘要上下文: {last_generated_summary[:50]}...")
                    
                    logger.info(f"✅ 章节生成完成: 第{chapter.chapter_number}章")
                    
                    # 如果启用同步分析
                    if task.enable_analysis:
                        logger.info(f"🔍 开始同步分析章节: 第{chapter.chapter_number}章")
                        
                        # 分析重试机制（最多3次）
                        analysis_retry_count = 0
                        analysis_success = False
                        last_analysis_error = None
                        
                        while analysis_retry_count < 3 and not analysis_success:
                            try:
                                if analysis_retry_count > 0:
                                    logger.info(f"🔄 重试分析章节 (第{analysis_retry_count}次): 第{chapter.chapter_number}章")
                                
                                async with write_lock:
                                    analysis_task = AnalysisTask(
                                        chapter_id=chapter_id,
                                        user_id=user_id,
                                        project_id=task.project_id,
                                        status='pending',
                                        progress=0
                                    )
                                    db_session.add(analysis_task)
                                    await db_session.commit()
                                    await db_session.refresh(analysis_task)
                                
                                # 同步执行分析，直接使用返回值判断成功/失败
                                analysis_result = await analyze_chapter_background(
                                    chapter_id=chapter_id,
                                    user_id=user_id,
                                    project_id=task.project_id,
                                    task_id=analysis_task.id,
                                    ai_service=ai_service
                                )
                                
                                # 直接根据返回值判断
                                if not analysis_result:
                                    last_analysis_error = "分析函数返回失败"
                                    logger.error(f"❌ 章节分析失败: 第{chapter.chapter_number}章")
                                    raise Exception(f"章节分析失败")
                                
                                # 分析成功
                                analysis_success = True
                                logger.info(f"✅ 章节分析成功: 第{chapter.chapter_number}章")
                                
                            except Exception as analysis_error:
                                last_analysis_error = str(analysis_error)
                                analysis_retry_count += 1
                                
                                if analysis_retry_count < 3:
                                    # 还有重试机会，等待后重试
                                    wait_time = min(2 ** analysis_retry_count, 10)
                                    logger.warning(f"⏳ 分析失败，等待 {wait_time} 秒后重试...")
                                    await asyncio.sleep(wait_time)
                                else:
                                    # 达到最大重试次数，必须终止整个批量任务
                                    logger.error(f"❌ 章节分析失败，已达最大重试次数(3次): 第{chapter.chapter_number}章")
                                    
                                    # 记录失败信息
                                    failed_info = {
                                        'chapter_id': chapter_id,
                                        'chapter_number': chapter.chapter_number,
                                        'title': chapter.title,
                                        'error': f"分析失败(重试3次): {last_analysis_error}",
                                        'retry_count': 3
                                    }
                                    
                                    async with write_lock:
                                        if task.failed_chapters is None:
                                            task.failed_chapters = []
                                        task.failed_chapters.append(failed_info)
                                        
                                        # 标记任务失败并终止
                                        task.status = 'failed'
                                        task.error_message = f"第{chapter.chapter_number}章分析失败(重试3次): {last_analysis_error}"[:500]
                                        task.completed_at = datetime.now()
                                        task.current_retry_count = 0
                                        await db_session.commit()
                                    
                                    logger.error(f"🛑 批量生成中断: 第{chapter.chapter_number}章分析失败")
                                    return  # 立即终止整个批量生成任务
                    
                    # 标记成功
                    chapter_success = True
                    
                    # 更新完成数
                    async with write_lock:
                        task.completed_chapters += 1
                        task.current_retry_count = 0  # 重置重试计数
                        await db_session.commit()
                    
                    logger.info(f"✅ 进度: {task.completed_chapters}/{task.total_chapters}")
                    
                except Exception as e:
                    last_error = str(e)
                    error_msg = f"第{chapter.chapter_number if chapter else '?'}章出错: {last_error}"
                    logger.error(f"❌ {error_msg}")
                    
                    retry_count += 1
                    
                    # 如果还有重试机会，等待一小段时间后重试
                    if retry_count <= task.max_retries:
                        wait_time = min(2 ** retry_count, 10)  # 指数退避，最多等待10秒
                        logger.info(f"⏳ 等待 {wait_time} 秒后重试...")
                        await asyncio.sleep(wait_time)
                    else:
                        # 达到最大重试次数，记录失败信息
                        logger.error(f"❌ 章节生成失败，已达最大重试次数({task.max_retries}): 第{chapter.chapter_number if chapter else '?'}章")
                        
                        failed_info = {
                            'chapter_id': chapter_id,
                            'chapter_number': chapter.chapter_number if chapter else -1,
                            'title': chapter.title if chapter else '未知',
                            'error': last_error,
                            'retry_count': retry_count - 1
                        }
                        
                        async with write_lock:
                            if task.failed_chapters is None:
                                task.failed_chapters = []
                            task.failed_chapters.append(failed_info)
                            
                            # 标记任务失败并终止
                            task.status = 'failed'
                            task.error_message = f"第{chapter.chapter_number}章生成失败(重试{retry_count-1}次): {last_error}"[:500]
                            task.completed_at = datetime.now()
                            task.current_retry_count = 0
                            await db_session.commit()
                        
                        # ⚠️ 如果启用了同步分析，任何错误都应该中断任务
                        # 因为章节生成或分析失败会影响后续章节的职业更新和剧情连贯性
                        if task.enable_analysis:
                            logger.error(f"🛑 批量生成中断: 因启用同步分析，任何错误都会中断任务以确保职业信息和剧情连贯性")
                        else:
                            logger.error(f"🛑 批量生成终止于第{chapter.chapter_number}章")
                        
                        return
        
        # 全部完成
        async with write_lock:
            task.status = 'completed'
            task.completed_at = datetime.now()
            task.current_chapter_id = None
            task.current_chapter_number = None
            await db_session.commit()
        
        logger.info(f"✅ 批量生成任务全部完成: {batch_id}, 成功生成 {task.completed_chapters} 章")
        
    except Exception as e:
        logger.error(f"❌ 批量生成任务异常: {str(e)}", exc_info=True)
        if db_session and task:
            try:
                async with write_lock:
                    task.status = 'failed'
                    task.error_message = str(e)[:500]
                    task.completed_at = datetime.now()
                    await db_session.commit()
            except Exception as commit_error:
                logger.error(f"❌ 更新任务失败状态失败: {str(commit_error)}")
    finally:
        if db_session:
            await db_session.close()


async def generate_single_chapter_for_batch(
    db_session: AsyncSession,
    chapter: Chapter,
    user_id: str,
    style_id: Optional[int],
    target_word_count: int,
    ai_service: AIService,
    write_lock: Lock,
    custom_model: Optional[str] = None,
    previous_summary_context: Optional[str] = None
) -> Optional[str]:
    """
    为批量生成执行单个章节的生成（非流式）
    复用现有生成逻辑的核心部分
    
    Returns:
        生成章节的摘要（前200字）
    """
    # 获取项目信息
    project_result = await db_session.execute(
        select(Project).where(Project.id == chapter.project_id)
    )
    project = project_result.scalar_one_or_none()
    if not project:
        raise Exception("项目不存在")
    
    # 获取项目的大纲模式
    outline_mode = project.outline_mode if project else 'one-to-many'
    logger.info(f"📋 批量生成 - 项目大纲模式: {outline_mode}")
    
    # 获取对应的大纲（优先使用 chapter.outline_id 直接关联）
    if chapter.outline_id:
        outline_result = await db_session.execute(
            select(Outline).where(Outline.id == chapter.outline_id)
        )
    else:
        # 回退到按序号查找
        outline_result = await db_session.execute(
            select(Outline)
            .where(Outline.project_id == chapter.project_id)
            .where(Outline.order_index == chapter.chapter_number)
        )
    outline = outline_result.scalar_one_or_none()
    
    # 获取写作风格
    style_content = ""
    if style_id:
        style_result = await db_session.execute(
            select(WritingStyle).where(WritingStyle.id == style_id)
        )
        style = style_result.scalar_one_or_none()
        if style:
            if style.user_id is None or style.user_id == user_id:
                style_content = style.prompt_content or ""
    
    # 🚀 根据大纲模式选择独立的上下文构建器（批量生成）
    if outline_mode == 'one-to-one':
        # 1-1模式
        logger.info(f"🔧 批量生成 - [1-1模式] 使用 OneToOneContextBuilder")
        context_builder = OneToOneContextBuilder(
            memory_service=memory_service,
            foreshadow_service=foreshadow_service
        )
        chapter_context = await context_builder.build(
            chapter=chapter,
            project=project,
            outline=outline,
            user_id=user_id,
            db=db_session,
            target_word_count=target_word_count
        )
    else:
        # 1-N模式：使用独立的完整构建器
        logger.info(f"🔧 批量生成 - [1-N模式] 使用 OneToManyContextBuilder")
        context_builder = OneToManyContextBuilder(
            memory_service=memory_service,
            foreshadow_service=foreshadow_service
        )
        chapter_context = await context_builder.build(
            chapter=chapter,
            project=project,
            outline=outline,
            user_id=user_id,
            db=db_session,
            style_content=style_content,
            target_word_count=target_word_count
        )
    
    # 日志输出统计信息
    logger.info(f"📊 批量生成 - 优化上下文统计:")
    logger.info(f"  - 章节序号: {chapter.chapter_number}")
    logger.info(f"  - 衔接锚点长度: {len(chapter_context.continuation_point or '')} 字符")
    logger.info(f"  - 相关记忆: {chapter_context.context_stats.get('memory_count', 0)} 条")
    logger.info(f"  - 总上下文长度: {chapter_context.context_stats.get('total_length', 0)} 字符")

    # 📌 项目硬设定（不可变事实）- 批量生成同样需要锁死关键事实，避免写串
    hard_settings = (project.world_rules or "").strip()
    if not hard_settings:
        hard_settings = "（暂无硬设定/世界规则。请在项目设置→世界规则中补充：关键人物称谓、婚约对象、势力归属等不可变事实。）"
    
    # 🚀 根据大纲模式选择提示词模板（批量生成）
    # 统一使用 context_builder 构建的 chapter_context 结果，与单章生成保持一致
    if outline_mode == 'one-to-one':
        # 1-1模式
        if chapter_context.continuation_point:
            # 有上一章内容
            template = await PromptService.get_template("CHAPTER_GENERATION_ONE_TO_ONE_NEXT", user_id, db_session)
            base_prompt = PromptService.format_prompt(
                template,
                project_title=project.title,
                chapter_number=chapter.chapter_number,
                chapter_title=chapter.title,
                chapter_outline=chapter_context.chapter_outline,
                hard_settings=hard_settings,
                target_word_count=target_word_count,
                genre=project.genre or '未设定',
                narrative_perspective=project.narrative_perspective or '第三人称',
                previous_chapter_content=chapter_context.continuation_point,
                characters_info=chapter_context.chapter_characters or '暂无角色信息',
                chapter_careers=chapter_context.chapter_careers or '暂无职业信息',
                foreshadow_reminders=chapter_context.foreshadow_reminders or '暂无需要关注的伏笔',
                relevant_memories=chapter_context.relevant_memories or '暂无相关记忆',
                previous_chapter_summary=chapter_context.previous_chapter_summary or ''
            )
        else:
            # 第一章
            template = await PromptService.get_template("CHAPTER_GENERATION_ONE_TO_ONE", user_id, db_session)
            base_prompt = PromptService.format_prompt(
                template,
                project_title=project.title,
                chapter_number=chapter.chapter_number,
                chapter_title=chapter.title,
                chapter_outline=chapter_context.chapter_outline,
                hard_settings=hard_settings,
                target_word_count=target_word_count,
                genre=project.genre or '未设定',
                narrative_perspective=project.narrative_perspective or '第三人称',
                characters_info=chapter_context.chapter_characters or '暂无角色信息',
                chapter_careers=chapter_context.chapter_careers or '暂无职业信息',
                foreshadow_reminders=chapter_context.foreshadow_reminders or '暂无需要关注的伏笔',
                relevant_memories=chapter_context.relevant_memories or '暂无相关记忆'
            )
    else:
        # 1-n模式：使用 context_builder 构建的结果，与单章生成保持一致
        if chapter_context.continuation_point:
            # 有前置内容，使用 WITH_CONTEXT 模板
            # 优先使用 context_builder 的摘要，其次使用传入的 previous_summary_context
            final_prev_summary = "（无上一章摘要，请根据锚点续写）"
            
            if chapter_context.previous_chapter_summary:
                final_prev_summary = chapter_context.previous_chapter_summary
            elif previous_summary_context:
                final_prev_summary = previous_summary_context
                    
            template = await PromptService.get_template("CHAPTER_GENERATION_ONE_TO_MANY_NEXT", user_id, db_session)
            base_prompt = PromptService.format_prompt(
                template,
                project_title=project.title,
                chapter_number=chapter.chapter_number,
                chapter_title=chapter.title,
                chapter_outline=chapter_context.chapter_outline,
                hard_settings=hard_settings,
                target_word_count=target_word_count,
                continuation_point=chapter_context.continuation_point,
                genre=project.genre or '未设定',
                narrative_perspective=project.narrative_perspective or '第三人称',
                characters_info=chapter_context.chapter_characters or '暂无角色信息',
                chapter_careers=chapter_context.chapter_careers or '暂无职业信息',
                foreshadow_reminders=chapter_context.foreshadow_reminders or '暂无需要关注的伏笔',
                previous_chapter_summary=final_prev_summary,
                recent_chapters_context=chapter_context.recent_chapters_context or '',
                relevant_memories=chapter_context.relevant_memories or ''
            )
        else:
            # 第一章，使用无前置内容模板
            template = await PromptService.get_template("CHAPTER_GENERATION_ONE_TO_MANY", user_id, db_session)
            base_prompt = PromptService.format_prompt(
                template,
                project_title=project.title,
                chapter_number=chapter.chapter_number,
                chapter_title=chapter.title,
                chapter_outline=chapter_context.chapter_outline,
                hard_settings=hard_settings,
                target_word_count=target_word_count,
                genre=project.genre or '未设定',
                narrative_perspective=project.narrative_perspective or '第三人称',
                characters_info=chapter_context.chapter_characters or '暂无角色信息',
                chapter_careers=chapter_context.chapter_careers or '暂无职业信息',
                foreshadow_reminders=chapter_context.foreshadow_reminders or '暂无需要关注的伏笔',
                relevant_memories=chapter_context.relevant_memories or '暂无相关记忆'
            )
    
    # 应用写作风格
    if style_content:
        prompt = WritingStyleManager.apply_style_to_prompt(base_prompt, style_content)
    else:
        prompt = base_prompt
    
    # 🎨 系统提示词拼装（批量生成，最高优先级）
    system_prompt_parts: list[str] = []
    base_system_prompt = (getattr(ai_service, "default_system_prompt", None) or "").strip()
    if base_system_prompt:
        system_prompt_parts.append(base_system_prompt)
    
    if hard_settings:
        system_prompt_parts.append(f"""【📌 项目硬设定（不可变事实） - 最高优先级】

{hard_settings}

⚠️ 必须严格遵守以上设定，严禁写错/混淆：人名、封号官职、府邸称谓、亲缘关系、婚约对象、时间地点。""")
    
    if style_content:
        system_prompt_parts.append(f"""【🎨 写作风格要求 - 最高优先级】

{style_content}

⚠️ 请严格遵循上述写作风格要求进行创作，这是最重要的指令！
确保在整个章节创作过程中始终保持风格的一致性。""")
        logger.info(f"✅ 批量生成 - 已注入写作风格（{len(style_content)}字符）")
    
    system_prompt_with_style = "\n\n".join([p for p in system_prompt_parts if p.strip()]) or None
    
    # 🔢 计算 max_tokens 限制（批量生成）
    # 参见流式生成处的说明：使用更保守的系数，避免章节严重超写。
    # NOTE: 批量生成同样需要足够的 max_tokens，否则容易出现“章节末尾被截断”。
    calculated_max_tokens = int(target_word_count * 2.0)
    calculated_max_tokens = max(800, min(calculated_max_tokens, 12000))  # 限制在 800-12000 之间
    logger.info(f"📊 批量生成 - 目标字数: {target_word_count}, max_tokens(硬上限): {calculated_max_tokens}")
    
    # 非流式生成内容
    full_content = ""
    # 准备生成参数
    generate_kwargs = {
        "prompt": prompt,
        "system_prompt": system_prompt_with_style,
        "tool_choice": "required",
        "max_tokens": calculated_max_tokens  # 添加 max_tokens 限制
    }
    # 如果传入了自定义模型，使用指定的模型
    if custom_model:
        generate_kwargs["model"] = custom_model
        logger.info(f"  批量生成使用自定义模型: {custom_model}")
    
    # 批量生成中的流式生成（非SSE，不需要修改进度显示）
    async for chunk in ai_service.generate_text_stream(**generate_kwargs):
        full_content += chunk
    
    # 更新章节内容到数据库（使用锁保护）
    async with write_lock:
        old_word_count = chapter.word_count or 0
        chapter.content = full_content
        new_word_count = len(full_content)
        chapter.word_count = new_word_count
        chapter.status = "completed"
        
        # 更新项目字数
        project.current_words = project.current_words - old_word_count + new_word_count
        
        # 记录生成历史
        history = GenerationHistory(
            project_id=chapter.project_id,
            chapter_id=chapter.id,
            prompt=f"批量生成: 第{chapter.chapter_number}章 {chapter.title}",
            generated_content=full_content[:500] if len(full_content) > 500 else full_content,
            model="default"
        )
        db_session.add(history)
        
        await db_session.commit()
        await db_session.refresh(chapter)
    
    logger.info(f"✅ 单章节生成完成: 第{chapter.chapter_number}章，共 {new_word_count} 字")
    
    # 生成简短摘要返回
    summary_preview = full_content[:300].replace('\n', ' ') if full_content else ""
    
    # 🔮 批量生成后自动标记计划在本章埋入的伏笔
    try:
        async with write_lock:
            plant_result = await foreshadow_service.auto_plant_pending_foreshadows(
                db=db_session,
                project_id=chapter.project_id,
                chapter_id=chapter.id,
                chapter_number=chapter.chapter_number,
                chapter_content=full_content
            )
        if plant_result.get('planted_count', 0) > 0:
            logger.info(f"🔮 批量生成 - 自动标记伏笔已埋入: {plant_result['planted_count']}个")
    except Exception as plant_error:
        logger.warning(f"⚠️ 批量生成 - 自动标记伏笔埋入失败: {str(plant_error)}")
        
    return summary_preview




# ==================== 章节重新生成相关API ====================

@router.post("/{chapter_id}/regenerate-stream", summary="流式重新生成章节内容")
async def regenerate_chapter_stream(
    chapter_id: str,
    request: Request,
    regenerate_request: ChapterRegenerateRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    user_ai_service: AIService = Depends(get_user_ai_service_for_task("chapter_regenerate"))
):
    """
    根据分析建议或自定义指令重新生成章节内容（流式返回）
    
    工作流程：
    1. 验证章节和分析结果
    2. 创建重新生成任务
    3. 构建修改指令
    4. 流式生成新内容
    5. 保存为版本历史
    6. 可选自动应用
    """
    user_id = getattr(request.state, 'user_id', None)
    if not user_id:
        raise HTTPException(status_code=401, detail="未登录")
    
    # 验证章节存在
    chapter_result = await db.execute(
        select(Chapter).where(Chapter.id == chapter_id)
    )
    chapter = chapter_result.scalar_one_or_none()
    
    if not chapter:
        raise HTTPException(status_code=404, detail="章节不存在")
    
    if not chapter.content or chapter.content.strip() == "":
        raise HTTPException(status_code=400, detail="章节内容为空，无法重新生成")
    
    # 验证用户权限
    await verify_project_access(chapter.project_id, user_id, db)
    
    # 获取分析结果（如果使用分析建议）
    analysis = None
    if regenerate_request.modification_source in ['analysis_suggestions', 'mixed']:
        analysis_result = await db.execute(
            select(PlotAnalysis)
            .where(PlotAnalysis.chapter_id == chapter_id)
            .order_by(PlotAnalysis.created_at.desc())
            .limit(1)
        )
        analysis = analysis_result.scalar_one_or_none()
        
        if not analysis:
            raise HTTPException(status_code=404, detail="该章节暂无分析结果")
    
    # 预先获取项目上下文数据和写作风格
    async for temp_db in get_db(request):
        try:
            # 获取项目信息
            project_result = await temp_db.execute(
                select(Project).where(Project.id == chapter.project_id)
            )
            project = project_result.scalar_one_or_none()
            
            # 获取角色信息（包含职业信息）
            characters_result = await temp_db.execute(
                select(Character).where(Character.project_id == chapter.project_id)
            )
            characters = characters_result.scalars().all()
            
            # 📝 根据大纲模式智能筛选相关角色（重新生成）
            outline_mode_result = await temp_db.execute(
                select(Project.outline_mode).where(Project.id == chapter.project_id)
            )
            outline_mode = outline_mode_result.scalar_one_or_none() or 'one-to-many'
            
            filter_character_names = None
            if outline_mode == 'one-to-one':
                # 1-1模式：从outline.structure中提取characters字段（优先使用 outline_id）
                if chapter.outline_id:
                    outline_result_temp = await temp_db.execute(
                        select(Outline.structure)
                        .where(Outline.id == chapter.outline_id)
                    )
                else:
                    outline_result_temp = await temp_db.execute(
                        select(Outline.structure)
                        .where(Outline.project_id == chapter.project_id)
                        .where(Outline.order_index == chapter.chapter_number)
                    )
                outline_structure = outline_result_temp.scalar_one_or_none()
                if outline_structure:
                    try:
                        structure = json.loads(outline_structure)
                        filter_character_names = structure.get('characters', [])
                        if filter_character_names:
                            logger.info(f"📋 重新生成 - 1-1模式：从structure提取角色列表 {filter_character_names}")
                    except json.JSONDecodeError:
                        logger.warning(f"⚠️ 重新生成 - outline.structure解析失败，使用全部角色")
            else:
                # 1-n模式：从chapter.expansion_plan中提取character_focus字段
                if chapter.expansion_plan:
                    try:
                        plan = json.loads(chapter.expansion_plan)
                        filter_character_names = plan.get('character_focus', [])
                        if filter_character_names:
                            logger.info(f"📋 重新生成 - 1-n模式：从expansion_plan提取角色焦点 {filter_character_names}")
                    except json.JSONDecodeError:
                        logger.warning(f"⚠️ 重新生成 - expansion_plan解析失败，使用全部角色")
            
            characters_info_with_careers = await build_characters_info_with_careers(
                db=temp_db,
                project_id=chapter.project_id,
                characters=characters,
                filter_character_names=filter_character_names
            )
            
            # 获取章节大纲（优先使用 chapter.outline_id 直接关联）
            if chapter.outline_id:
                outline_result = await temp_db.execute(
                    select(Outline).where(Outline.id == chapter.outline_id)
                )
            else:
                # 回退到按序号查找
                outline_result = await temp_db.execute(
                    select(Outline)
                    .where(Outline.project_id == chapter.project_id)
                    .where(Outline.order_index == chapter.chapter_number)
                )
            outline = outline_result.scalar_one_or_none()
            
            # 获取写作风格
            style_content = ""
            style_id = regenerate_request.style_id
            
            # 如果没有指定风格，尝试使用项目的默认风格
            if not style_id:
                from app.models.project_default_style import ProjectDefaultStyle
                default_style_result = await temp_db.execute(
                    select(ProjectDefaultStyle.style_id)
                    .where(ProjectDefaultStyle.project_id == chapter.project_id)
                )
                default_style_id = default_style_result.scalar_one_or_none()
                if default_style_id:
                    style_id = default_style_id
                    logger.info(f"📝 使用项目默认写作风格: {style_id}")
            
            # 获取风格内容
            if style_id:
                style_result = await temp_db.execute(
                    select(WritingStyle).where(WritingStyle.id == style_id)
                )
                style = style_result.scalar_one_or_none()
                if style:
                    # 验证风格是否可用：全局预设风格（user_id为NULL）或者当前用户的自定义风格
                    if style.user_id is None or style.user_id == user_id:
                        style_content = style.prompt_content or ""
                        style_type = "全局预设" if style.user_id is None else "用户自定义"
                        logger.info(f"✅ 使用写作风格: {style.name} ({style_type})")
                    else:
                        logger.warning(f"⚠️ 风格 {style_id} 不属于当前项目，跳过")
                else:
                    logger.warning(f"⚠️ 未找到风格 {style_id}")
            else:
                logger.info("ℹ️ 未指定写作风格，使用默认提示词")
            
            # 构建项目上下文
            project_context = {
                'project_title': project.title if project else '未知',
                'genre': project.genre if project else '未设定',
                'theme': project.theme if project else '未设定',
                'narrative_perspective': project.narrative_perspective if project else '第三人称',
                'time_period': project.world_time_period if project else '未设定',
                'location': project.world_location if project else '未设定',
                'atmosphere': project.world_atmosphere if project else '未设定',
                'characters_info': characters_info_with_careers,
                'chapter_outline': outline.content if outline else chapter.summary or '暂无大纲',
                'previous_context': ''  # 可以后续扩展添加前置章节上下文
            }
        finally:
            await temp_db.close()
        break
    
    async def event_generator():
        """流式生成事件生成器"""
        db_session = None
        db_committed = False
        
        # 初始化标准进度追踪器
        from app.utils.sse_response import WizardProgressTracker
        tracker = WizardProgressTracker("章节重新生成")
        
        try:
            yield await tracker.start()
            
            # 创建独立数据库会话
            async for db_session in get_db(request):
                yield await tracker.loading("加载章节信息...", 0.5)
                
                # 创建重新生成任务
                regen_task = RegenerationTask(
                    chapter_id=chapter_id,
                    analysis_id=analysis.id if analysis else None,
                    user_id=user_id,
                    project_id=chapter.project_id,
                    modification_instructions="",  # 稍后填充
                    original_suggestions=analysis.suggestions if analysis else None,
                    selected_suggestion_indices=regenerate_request.selected_suggestion_indices,
                    custom_instructions=regenerate_request.custom_instructions,
                    style_id=regenerate_request.style_id,
                    target_word_count=regenerate_request.target_word_count,
                    focus_areas=regenerate_request.focus_areas,
                    preserve_elements=regenerate_request.preserve_elements.model_dump() if regenerate_request.preserve_elements else None,
                    status='running',
                    original_content=chapter.content,
                    original_word_count=chapter.word_count or len(chapter.content),
                    version_note=regenerate_request.version_note,
                    started_at=datetime.now()
                )
                db_session.add(regen_task)
                await db_session.commit()
                await db_session.refresh(regen_task)
                
                task_id = regen_task.id
                logger.info(f"📝 创建重新生成任务: {task_id}")
                
                yield await tracker.preparing("准备重新生成...")
                
                yield await SSEResponse.send_event(
                    event='task_created',
                    data={'task_id': task_id}
                )
                
                # 初始化重新生成器
                regenerator = ChapterRegenerator(user_ai_service)
                
                # === 生成阶段 ===
                full_content = ""
                estimated_total = regenerate_request.target_word_count or len(chapter.content)
                
                yield await tracker.generating(
                    current_chars=0,
                    estimated_total=estimated_total
                )
                
                async for event in regenerator.regenerate_with_feedback(
                    chapter=chapter,
                    analysis=analysis,
                    regenerate_request=regenerate_request,
                    project_context=project_context,
                    style_content=style_content,
                    user_id=user_id,
                    db=db_session
                ):
                    # 处理不同类型的事件
                    if event['type'] == 'chunk':
                        # 内容块
                        chunk = event['content']
                        full_content += chunk
                        yield await tracker.generating_chunk(chunk)
                        
                        # 定期更新进度
                        if len(full_content) % 500 == 0:
                            yield await tracker.generating(
                                current_chars=len(full_content),
                                estimated_total=estimated_total,
                                message=f'重新生成中... 已生成 {len(full_content)} 字'
                            )
                    elif event['type'] == 'progress':
                        # 进度更新 - 映射到对应阶段
                        progress = event.get('progress', 0)
                        message = event.get('message', '')
                        if progress < 20:
                            yield await tracker.preparing(message)
                        elif progress < 85:
                            yield await tracker.generating(
                                current_chars=len(full_content),
                                estimated_total=estimated_total,
                                message=message
                            )
                        else:
                            yield await tracker.parsing(message)
                    
                    await asyncio.sleep(0)
                
                # === 保存阶段 ===
                yield await tracker.saving("保存重新生成的内容...", 0.5)
                
                # 更新任务状态
                regen_task.status = 'completed'
                regen_task.regenerated_content = full_content
                regen_task.regenerated_word_count = len(full_content)
                regen_task.completed_at = datetime.now()
                
                # 计算差异统计
                diff_stats = regenerator.calculate_content_diff(chapter.content, full_content)
                
                await db_session.commit()
                db_committed = True
                
                yield await tracker.saving("保存完成", 0.9)
                
                # === 完成阶段 ===
                yield await tracker.complete("重新生成完成！")
                
                # 发送结果数据
                yield await tracker.result({
                    'task_id': task_id,
                    'word_count': len(full_content),
                    'version_number': regen_task.version_number,
                    'auto_applied': regenerate_request.auto_apply,
                    'diff_stats': diff_stats
                })
                
                # 发送完成信号
                yield await tracker.done()
                
                logger.info(f"✅ 章节重新生成完成: {chapter_id}, 任务: {task_id}")
                
                break
        
        except Exception as e:
            logger.error(f"❌ 重新生成失败: {str(e)}", exc_info=True)
            
            # 更新任务状态为失败
            if db_session and not db_committed:
                try:
                    task_result = await db_session.execute(
                        select(RegenerationTask).where(RegenerationTask.chapter_id == chapter_id)
                        .order_by(RegenerationTask.created_at.desc()).limit(1)
                    )
                    task = task_result.scalar_one_or_none()
                    if task:
                        task.status = 'failed'
                        task.error_message = str(e)[:500]
                        task.completed_at = datetime.now()
                        await db_session.commit()
                except Exception as update_error:
                    logger.error(f"更新任务失败状态失败: {str(update_error)}")
            
            yield await tracker.error(str(e))
        
        finally:
            if db_session:
                try:
                    if not db_committed and db_session.in_transaction():
                        await db_session.rollback()
                    await db_session.close()
                except Exception as close_error:
                    logger.error(f"关闭数据库会话失败: {str(close_error)}")
    
    return create_sse_response(event_generator())


@router.get("/{chapter_id}/regeneration/tasks", summary="获取章节的重新生成任务列表")
async def get_regeneration_tasks(
    chapter_id: str,
    request: Request,
    limit: int = Query(10, ge=1, le=50),
    db: AsyncSession = Depends(get_db)
):
    """获取指定章节的重新生成任务历史"""
    user_id = getattr(request.state, 'user_id', None)
    
    # 验证章节存在和权限
    chapter_result = await db.execute(
        select(Chapter).where(Chapter.id == chapter_id)
    )
    chapter = chapter_result.scalar_one_or_none()
    if not chapter:
        raise HTTPException(status_code=404, detail="章节不存在")
    
    await verify_project_access(chapter.project_id, user_id, db)
    
    # 获取任务列表
    result = await db.execute(
        select(RegenerationTask)
        .where(RegenerationTask.chapter_id == chapter_id)
        .order_by(RegenerationTask.created_at.desc())
        .limit(limit)
    )
    tasks = result.scalars().all()
    
    return {
        "chapter_id": chapter_id,
        "total": len(tasks),
        "tasks": [
            {
                "task_id": task.id,
                "status": task.status,
                "version_number": task.version_number,
                "version_note": task.version_note,
                "original_word_count": task.original_word_count,
                "regenerated_word_count": task.regenerated_word_count,
                "created_at": task.created_at.isoformat() if task.created_at else None,
                "completed_at": task.completed_at.isoformat() if task.completed_at else None
            }
            for task in tasks
        ]
    }


@router.put("/{chapter_id}/expansion-plan", response_model=dict, summary="更新章节规划信息")
async def update_chapter_expansion_plan(
    chapter_id: str,
    expansion_plan: ExpansionPlanUpdate,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """
    更新章节的展开规划信息和情节概要
    
    Args:
        chapter_id: 章节ID
        expansion_plan: 规划信息更新数据(包含summary和expansion_plan字段)
    
    Returns:
        更新后的章节规划信息
    """
    # 获取章节
    result = await db.execute(
        select(Chapter).where(Chapter.id == chapter_id)
    )
    chapter = result.scalar_one_or_none()
    
    if not chapter:
        raise HTTPException(status_code=404, detail="章节不存在")
    
    # 验证用户权限
    user_id = getattr(request.state, 'user_id', None)
    await verify_project_access(chapter.project_id, user_id, db)
    
    # 准备更新数据(排除None值)
    plan_data = expansion_plan.model_dump(exclude_unset=True, exclude_none=True)
    
    # 分离summary和expansion_plan数据
    summary_value = plan_data.pop('summary', None)
    
    # 更新summary字段(如果提供)
    if summary_value is not None:
        chapter.summary = summary_value
        logger.info(f"更新章节概要: {chapter_id}")
    
    # 更新expansion_plan字段(如果有其他字段)
    if plan_data:
        if chapter.expansion_plan:
            try:
                existing_plan = json.loads(chapter.expansion_plan)
                # 合并更新
                existing_plan.update(plan_data)
                chapter.expansion_plan = json.dumps(existing_plan, ensure_ascii=False)
            except json.JSONDecodeError:
                logger.warning(f"章节 {chapter_id} 的expansion_plan格式错误,将覆盖")
                chapter.expansion_plan = json.dumps(plan_data, ensure_ascii=False)
        else:
            chapter.expansion_plan = json.dumps(plan_data, ensure_ascii=False)
    
    await db.commit()
    await db.refresh(chapter)
    
    logger.info(f"章节规划更新成功: {chapter_id}")
    
    # 返回更新后的规划数据
    updated_plan = json.loads(chapter.expansion_plan) if chapter.expansion_plan else None
    
    return {
        "id": chapter.id,
        "summary": chapter.summary,
        "expansion_plan": updated_plan,
        "message": "规划信息更新成功"
    }


# ==================== 局部重写相关API ====================

@router.post("/{chapter_id}/partial-regenerate-stream", summary="流式局部重写选中内容")
async def partial_regenerate_stream(
    chapter_id: str,
    request: Request,
    partial_request: PartialRegenerateRequest,
    db: AsyncSession = Depends(get_db),
    user_ai_service: AIService = Depends(get_user_ai_service_for_task("chapter_regenerate"))
):
    """
    对章节中选中的部分内容进行流式重写
    
    工作流程：
    1. 验证章节和选中内容的有效性
    2. 截取上下文（前后文）
    3. 根据用户要求构建提示词
    4. 流式生成重写内容
    5. 返回重写结果（不自动保存，由前端决定是否应用）
    """
    user_id = getattr(request.state, 'user_id', None)
    if not user_id:
        raise HTTPException(status_code=401, detail="未登录")
    
    # 验证章节存在
    chapter_result = await db.execute(
        select(Chapter).where(Chapter.id == chapter_id)
    )
    chapter = chapter_result.scalar_one_or_none()
    
    if not chapter:
        raise HTTPException(status_code=404, detail="章节不存在")
    
    if not chapter.content or chapter.content.strip() == "":
        raise HTTPException(status_code=400, detail="章节内容为空")
    
    # 验证用户权限
    await verify_project_access(chapter.project_id, user_id, db)
    
    # 验证位置参数
    content_length = len(chapter.content)
    if partial_request.start_position >= content_length:
        raise HTTPException(status_code=400, detail="起始位置超出内容范围")
    if partial_request.end_position > content_length:
        raise HTTPException(status_code=400, detail="结束位置超出内容范围")
    if partial_request.start_position >= partial_request.end_position:
        raise HTTPException(status_code=400, detail="起始位置必须小于结束位置")
    
    # 验证选中的文本是否匹配
    actual_selected = chapter.content[partial_request.start_position:partial_request.end_position]
    if actual_selected != partial_request.selected_text:
        # 位置可能有偏差，尝试在附近查找
        search_start = max(0, partial_request.start_position - 50)
        search_end = min(content_length, partial_request.end_position + 50)
        search_area = chapter.content[search_start:search_end]
        
        if partial_request.selected_text in search_area:
            # 找到了，更新位置
            offset = search_area.find(partial_request.selected_text)
            partial_request.start_position = search_start + offset
            partial_request.end_position = partial_request.start_position + len(partial_request.selected_text)
            logger.info(f"⚠️ 选中文本位置校正: {partial_request.start_position}-{partial_request.end_position}")
        else:
            raise HTTPException(
                status_code=400,
                detail="选中的文本与章节内容不匹配，请刷新页面后重试"
            )
    
    # 预先获取项目信息和写作风格
    project_result = await db.execute(
        select(Project).where(Project.id == chapter.project_id)
    )
    project = project_result.scalar_one_or_none()
    
    # 获取写作风格
    style_content = ""
    style_id = partial_request.style_id
    
    # 如果没有指定风格，尝试使用项目的默认风格
    if not style_id:
        from app.models.project_default_style import ProjectDefaultStyle
        default_style_result = await db.execute(
            select(ProjectDefaultStyle.style_id)
            .where(ProjectDefaultStyle.project_id == chapter.project_id)
        )
        default_style_id = default_style_result.scalar_one_or_none()
        if default_style_id:
            style_id = default_style_id
            logger.info(f"📝 局部重写 - 使用项目默认写作风格: {style_id}")
    
    # 获取风格内容
    if style_id:
        style_result = await db.execute(
            select(WritingStyle).where(WritingStyle.id == style_id)
        )
        style = style_result.scalar_one_or_none()
        if style:
            if style.user_id is None or style.user_id == user_id:
                style_content = style.prompt_content or ""
                style_type = "全局预设" if style.user_id is None else "用户自定义"
                logger.info(f"✅ 局部重写 - 使用写作风格: {style.name} ({style_type})")
            else:
                logger.warning(f"⚠️ 风格 {style_id} 不属于当前用户，跳过")
    
    async def event_generator():
        """流式生成事件生成器"""
        from app.utils.sse_response import WizardProgressTracker
        tracker = WizardProgressTracker("局部重写")
        
        try:
            yield await tracker.start()
            yield await tracker.loading("准备重写上下文...", 0.3)
            
            # 截取上下文
            context_chars = partial_request.context_chars
            start_pos = partial_request.start_position
            end_pos = partial_request.end_position
            
            # 前文：从start_pos往前截取context_chars个字符
            context_before_start = max(0, start_pos - context_chars)
            context_before = chapter.content[context_before_start:start_pos]
            
            # 后文：从end_pos往后截取context_chars个字符
            context_after_end = min(content_length, end_pos + context_chars)
            context_after = chapter.content[end_pos:context_after_end]
            
            # 原文
            original_text = partial_request.selected_text
            original_word_count = len(original_text)
            
            logger.info(f"📝 局部重写 - 原文: {original_word_count}字, 前文: {len(context_before)}字, 后文: {len(context_after)}字")
            
            yield await tracker.loading("构建提示词...", 0.5)
            
            # 构建字数要求
            length_requirement = ""
            if partial_request.length_mode == "similar":
                min_words = int(original_word_count * 0.8)
                max_words = int(original_word_count * 1.2)
                length_requirement = f"保持与原文相近的字数（约{original_word_count}字，允许{min_words}-{max_words}字浮动）"
            elif partial_request.length_mode == "expand":
                min_words = int(original_word_count * 1.2)
                max_words = int(original_word_count * 2.0)
                length_requirement = f"适当扩展内容（目标{min_words}-{max_words}字）"
            elif partial_request.length_mode == "condense":
                min_words = int(original_word_count * 0.5)
                max_words = int(original_word_count * 0.8)
                length_requirement = f"精简压缩内容（目标{min_words}-{max_words}字）"
            elif partial_request.length_mode == "custom" and partial_request.target_word_count:
                length_requirement = f"目标字数：约{partial_request.target_word_count}字（允许±20%浮动）"
            else:
                length_requirement = f"保持与原文相近的字数（约{original_word_count}字）"
            
            # 获取提示词模板
            template = await PromptService.get_template("PARTIAL_REGENERATE", user_id, db)
            if not template:
                template = PromptService.PARTIAL_REGENERATE
            
            # 构建提示词
            prompt = PromptService.format_prompt(
                template,
                context_before=context_before if context_before else "（这是章节开头）",
                original_word_count=original_word_count,
                selected_text=original_text,
                context_after=context_after if context_after else "（这是章节结尾）",
                user_instructions=partial_request.user_instructions,
                length_requirement=length_requirement,
                style_content=style_content if style_content else "保持与原文一致的叙事风格"
            )
            
            yield await tracker.preparing("开始生成...")
            
            # 计算 max_tokens
            if partial_request.length_mode == "expand":
                target_words = int(original_word_count * 2.0)
            elif partial_request.length_mode == "custom" and partial_request.target_word_count:
                target_words = partial_request.target_word_count
            else:
                target_words = int(original_word_count * 1.5)
            
            calculated_max_tokens = max(500, min(int(target_words * 3), 8000))
            
            # 流式生成
            full_content = ""
            chunk_count = 0
            
            yield await tracker.generating(
                current_chars=0,
                estimated_total=target_words
            )
            
            async for chunk in user_ai_service.generate_text_stream(
                prompt=prompt,
                max_tokens=calculated_max_tokens
            ):
                full_content += chunk
                chunk_count += 1
                
                # 发送内容块
                yield await tracker.generating_chunk(chunk)
                
                # 每5个chunk发送一次进度更新
                if chunk_count % 5 == 0:
                    yield await tracker.generating(
                        current_chars=len(full_content),
                        estimated_total=target_words,
                        message=f'正在重写中... 已生成 {len(full_content)} 字'
                    )
                
                await asyncio.sleep(0)
            
            # 清理输出（移除可能的前后缀）
            full_content = full_content.strip()
            
            # 移除常见的AI输出前缀
            prefixes_to_remove = [
                "重写后：", "重写后:", "改写后：", "改写后:",
                "以下是重写后的内容：", "以下是重写后的内容:",
                "重写内容：", "重写内容:"
            ]
            for prefix in prefixes_to_remove:
                if full_content.startswith(prefix):
                    full_content = full_content[len(prefix):].strip()
                    break
            
            # 移除首尾可能的引号
            if (full_content.startswith('"') and full_content.endswith('"')) or \
               (full_content.startswith("'") and full_content.endswith("'")):
                full_content = full_content[1:-1]
            if (full_content.startswith('「') and full_content.endswith('」')) or \
               (full_content.startswith('『') and full_content.endswith('』')):
                full_content = full_content[1:-1]
            
            new_word_count = len(full_content)
            
            logger.info(f"✅ 局部重写完成: 原文{original_word_count}字 -> 新文{new_word_count}字")
            
            # 完成
            yield await tracker.complete("重写完成！")
            
            # 发送结果数据
            yield await tracker.result({
                'new_text': full_content,
                'word_count': new_word_count,
                'original_word_count': original_word_count,
                'start_position': partial_request.start_position,
                'end_position': partial_request.end_position
            })
            
            yield await tracker.done()
            
        except Exception as e:
            logger.error(f"❌ 局部重写失败: {str(e)}", exc_info=True)
            yield await tracker.error(str(e))
    
    return create_sse_response(event_generator())


@router.post("/{chapter_id}/apply-partial-regenerate", summary="应用局部重写结果")
async def apply_partial_regenerate(
    chapter_id: str,
    request: Request,
    apply_request: dict,
    db: AsyncSession = Depends(get_db)
):
    """
    将局部重写的结果应用到章节内容中
    
    请求体：
    - new_text: 重写后的新内容
    - start_position: 原文起始位置
    - end_position: 原文结束位置
    """
    user_id = getattr(request.state, 'user_id', None)
    if not user_id:
        raise HTTPException(status_code=401, detail="未登录")
    
    # 验证章节存在
    chapter_result = await db.execute(
        select(Chapter).where(Chapter.id == chapter_id)
    )
    chapter = chapter_result.scalar_one_or_none()
    
    if not chapter:
        raise HTTPException(status_code=404, detail="章节不存在")
    
    # 验证用户权限
    await verify_project_access(chapter.project_id, user_id, db)
    
    # 获取参数
    new_text = apply_request.get('new_text', '')
    start_position = apply_request.get('start_position', 0)
    end_position = apply_request.get('end_position', 0)
    
    if not new_text:
        raise HTTPException(status_code=400, detail="新内容不能为空")
    
    # 验证位置有效性
    content_length = len(chapter.content)
    if start_position < 0 or end_position > content_length or start_position >= end_position:
        raise HTTPException(status_code=400, detail="位置参数无效")
    
    # 构建新内容
    old_word_count = chapter.word_count or 0
    new_content = chapter.content[:start_position] + new_text + chapter.content[end_position:]
    new_word_count = len(new_content)
    
    # 更新章节
    chapter.content = new_content
    chapter.word_count = new_word_count
    
    # 更新项目字数
    project_result = await db.execute(
        select(Project).where(Project.id == chapter.project_id)
    )
    project = project_result.scalar_one_or_none()
    if project:
        project.current_words = project.current_words - old_word_count + new_word_count
    
    await db.commit()
    await db.refresh(chapter)
    
    logger.info(f"✅ 局部重写已应用: 章节{chapter_id}, {old_word_count}字 -> {new_word_count}字")
    
    return {
        "success": True,
        "chapter_id": chapter_id,
        "word_count": new_word_count,
        "old_word_count": old_word_count,
        "message": "局部重写已应用"
    }
