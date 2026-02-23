"""章节上下文构建服务 - 实现RTCO框架的智能上下文构建"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import json

from app.models.chapter import Chapter
from app.models.project import Project
from app.models.outline import Outline
from app.models.character import Character
from app.models.career import Career, CharacterCareer
from app.models.memory import StoryMemory
from app.models.foreshadow import Foreshadow
from app.models.relationship import CharacterRelationship, Organization, OrganizationMember
from app.logger import get_logger

logger = get_logger(__name__)


@dataclass
class OneToManyContext:
    """
    1-N模式章节上下文数据结构
    
    采用RTCO框架的分层设计：
    - P0-核心：大纲（含最近10章规划）、衔接锚点（500字+摘要）、字数要求
    - P1-重要：角色（完整版含关系/组织/职业）、职业详情、情感基调
    - P2-参考：记忆（始终启用，相关度>0.6）、伏笔提醒
    """
    
    # === P0-核心信息 ===
    chapter_outline: str = ""           # 本章大纲（从expansion_plan构建）
    recent_chapters_context: Optional[str] = None  # 最近10章expansion_plan摘要
    continuation_point: Optional[str] = None  # 衔接锚点（统一500字）
    previous_chapter_summary: Optional[str] = None  # 上一章剧情摘要
    previous_chapter_events: Optional[List[str]] = None  # 上一章关键事件
    target_word_count: int = 3000
    min_word_count: int = 2500
    max_word_count: int = 4000
    narrative_perspective: str = "第三人称"
    
    # === 本章基本信息 ===
    chapter_number: int = 1
    chapter_title: str = ""
    
    # === 项目基本信息 ===
    title: str = ""
    genre: str = ""
    theme: str = ""
    
    # === P1-重要信息 ===
    chapter_characters: str = ""        # 完整版角色信息（含年龄、外貌、背景、关系、组织）
    chapter_careers: Optional[str] = None  # 独立的职业详情（含完整阶段体系）
    emotional_tone: str = ""
    
    # === P2-参考信息 ===
    relevant_memories: Optional[str] = None  # 始终启用（相关度>0.6）
    foreshadow_reminders: Optional[str] = None
    
    # === 元信息 ===
    context_stats: Dict[str, Any] = field(default_factory=dict)
    
    def get_total_context_length(self) -> int:
        """计算总上下文长度"""
        total = 0
        for field_name in ['chapter_outline', 'recent_chapters_context', 'continuation_point',
                          'chapter_characters', 'chapter_careers',
                          'relevant_memories', 'foreshadow_reminders',
                          'previous_chapter_summary']:
            value = getattr(self, field_name, None)
            if value:
                total += len(value)
        return total


@dataclass
class OneToOneContext:
    """
    1-1模式章节上下文数据结构
    
    采用RTCO框架的分层设计：
    - P0-核心：从outline.structure提取的大纲、字数要求
    - P1-重要：上一章最后500字、从structure.characters获取的角色、本章职业体系
    - P2-参考：伏笔提醒、相关记忆（相关度>0.6）
    """
    
    # === P0-核心信息 ===
    chapter_outline: str = ""           # 从outline.structure提取
    target_word_count: int = 3000
    min_word_count: int = 2500
    max_word_count: int = 4000
    narrative_perspective: str = "第三人称"
    
    # === 本章基本信息 ===
    chapter_number: int = 1
    chapter_title: str = ""
    
    # === 项目基本信息 ===
    title: str = ""
    genre: str = ""
    theme: str = ""
    
    # === P1-重要信息 ===
    continuation_point: Optional[str] = None  # 上一章最后500字
    previous_chapter_summary: Optional[str] = None  # 上一章剧情摘要
    chapter_characters: str = ""        # 从structure.characters获取
    chapter_careers: Optional[str] = None  # 本章涉及的职业完整信息
    
    # === P2-参考信息 ===
    foreshadow_reminders: Optional[str] = None
    relevant_memories: Optional[str] = None  # 相关度>0.6
    
    # === 元信息 ===
    context_stats: Dict[str, Any] = field(default_factory=dict)
    
    def get_total_context_length(self) -> int:
        """计算总上下文长度"""
        total = 0
        for field_name in ['chapter_outline', 'continuation_point', 'previous_chapter_summary',
                          'chapter_characters', 'chapter_careers', 'foreshadow_reminders',
                          'relevant_memories']:
            value = getattr(self, field_name, None)
            if value:
                total += len(value)
        return total


# ==================== 1-N模式上下文构建器 ====================

class OneToManyContextBuilder:
    """
    1-N模式上下文构建器
    
    上下文构建策略：
    - 章节大纲：本章expansion_plan + 最近10章expansion_plan摘要
    - 衔接锚点：统一上一章末尾500字 + 摘要
    - 角色信息：完整版（含年龄、外貌、背景、关系、组织、职业）
    - 职业详情：独立的chapter_careers字段，含完整阶段体系
    - 相关记忆：始终启用（相关度>0.6）
    - 伏笔提醒：始终启用
    """
    
    # 配置常量
    ENDING_LENGTH = 500          # 统一衔接长度500字
    MEMORY_COUNT = 10            # 记忆条数
    MEMORY_SIMILARITY_THRESHOLD = 0.6  # 记忆相关度阈值
    RECENT_CHAPTERS_COUNT = 10   # 最近章节规划数量
    
    def __init__(self, memory_service=None, foreshadow_service=None):
        """
        初始化构建器
        
        Args:
            memory_service: 记忆服务实例（可选，用于检索相关记忆）
            foreshadow_service: 伏笔服务实例（可选，用于获取伏笔提醒）
        """
        self.memory_service = memory_service
        self.foreshadow_service = foreshadow_service
    
    async def build(
        self,
        chapter: Chapter,
        project: Project,
        outline: Optional[Outline],
        user_id: str,
        db: AsyncSession,
        style_content: Optional[str] = None,
        target_word_count: int = 3000,
        temp_narrative_perspective: Optional[str] = None
    ) -> OneToManyContext:
        """
        构建章节生成所需的上下文（1-N模式）
        
        Args:
            chapter: 章节对象
            project: 项目对象
            outline: 大纲对象（可选）
            user_id: 用户ID
            db: 数据库会话
            style_content: 写作风格内容（可选，不再使用，保留参数兼容性）
            target_word_count: 目标字数
            temp_narrative_perspective: 临时叙事视角（可选，覆盖项目默认）
        
        Returns:
            OneToManyContext: 结构化的上下文对象
        """
        chapter_number = chapter.chapter_number
        logger.info(f"📝 [1-N模式] 开始构建章节上下文: 第{chapter_number}章")
        
        # 确定叙事视角
        narrative_perspective = (
            temp_narrative_perspective or
            project.narrative_perspective or
            "第三人称"
        )
        
        # 初始化上下文
        context = OneToManyContext(
            chapter_number=chapter_number,
            chapter_title=chapter.title or "",
            title=project.title or "",
            genre=project.genre or "",
            theme=project.theme or "",
            target_word_count=target_word_count,
            min_word_count=max(500, target_word_count - 500),
            max_word_count=target_word_count + 1000,
            narrative_perspective=narrative_perspective
        )
        
        # === P0-核心信息（始终构建）===
        context.chapter_outline = self._build_chapter_outline_1n(chapter, outline)
        
        # === 最近10章expansion_plan摘要 ===
        if chapter_number > 1:
            context.recent_chapters_context = await self._build_recent_chapters_context(
                chapter, project.id, db
            )
            logger.info(f"  ✅ 最近章节规划: {len(context.recent_chapters_context or '')}字符")
        
        # === 衔接锚点（统一500字 + 摘要）===
        if chapter_number == 1:
            context.continuation_point = None
            context.previous_chapter_summary = None
            context.previous_chapter_events = None
            logger.info("  ✅ 第1章无需衔接锚点")
        else:
            ending_info = await self._get_last_ending_enhanced(
                chapter, db, self.ENDING_LENGTH
            )
            context.continuation_point = ending_info.get('ending_text')
            context.previous_chapter_summary = ending_info.get('summary')
            context.previous_chapter_events = ending_info.get('key_events')
            logger.info(f"  ✅ 衔接锚点: {len(context.continuation_point or '')}字符")
        
        # === P1-重要信息 ===
        # 角色信息（完整版：含年龄、外貌、背景、关系、组织、职业）+ 独立职业详情
        characters_info, careers_info = await self._build_chapter_characters_1n(
            chapter, project, outline, db
        )
        context.chapter_characters = characters_info
        context.chapter_careers = careers_info
        context.emotional_tone = self._extract_emotional_tone(chapter, outline)
        logger.info(f"  ✅ 角色信息: {len(context.chapter_characters)}字符")
        logger.info(f"  ✅ 职业信息: {len(context.chapter_careers or '')}字符")
        
        # === P2-参考信息（始终启用）===
        if self.memory_service:
            context.relevant_memories = await self._get_relevant_memories_enhanced(
                user_id, project.id, chapter_number,
                context.chapter_outline, db
            )
            logger.info(f"  ✅ 相关记忆: {len(context.relevant_memories or '')}字符")
        
        # === P2-伏笔提醒===
        if self.foreshadow_service:
            context.foreshadow_reminders = await self._get_foreshadow_reminders(
                project.id, chapter_number, db
            )
            if context.foreshadow_reminders:
                logger.info(f"  ✅ 伏笔提醒: {len(context.foreshadow_reminders)}字符")
        
        # === 统计信息 ===
        context.context_stats = {
            "mode": "one-to-many",
            "chapter_number": chapter_number,
            "has_continuation": context.continuation_point is not None,
            "continuation_length": len(context.continuation_point or ""),
            "characters_length": len(context.chapter_characters),
            "careers_length": len(context.chapter_careers or ""),
            "recent_context_length": len(context.recent_chapters_context or ""),
            "memories_length": len(context.relevant_memories or ""),
            "foreshadow_length": len(context.foreshadow_reminders or ""),
            "total_length": context.get_total_context_length()
        }
        
        logger.info(f"📊 [1-N模式] 上下文构建完成: 总长度 {context.context_stats['total_length']} 字符")
        
        return context
    
    def _build_chapter_outline_1n(
        self,
        chapter: Chapter,
        outline: Optional[Outline]
    ) -> str:
        """构建1-N模式的章节大纲"""
        # 优先使用 expansion_plan 的详细规划
        if chapter.expansion_plan:
            try:
                plan = json.loads(chapter.expansion_plan)
                outline_content = f"""剧情摘要：{plan.get('plot_summary', '无')}

关键事件：
{chr(10).join(f'- {event}' for event in plan.get('key_events', []))}

角色焦点：{', '.join(plan.get('character_focus', []))}
情感基调：{plan.get('emotional_tone', '未设定')}
叙事目标：{plan.get('narrative_goal', '未设定')}
冲突类型：{plan.get('conflict_type', '未设定')}"""
                return outline_content
            except json.JSONDecodeError:
                pass
        
        # 回退到大纲内容
        return outline.content if outline else chapter.summary or '暂无大纲'
    
    async def _build_chapter_characters_1n(
        self,
        chapter: Chapter,
        project: Project,
        outline: Optional[Outline],
        db: AsyncSession
    ) -> tuple[str, Optional[str]]:
        """构建1-N模式的角色信息（完整版：含年龄、外貌、背景、关系、组织、职业）+ 独立职业详情"""
        from sqlalchemy import or_
        
        # 获取所有角色
        characters_result = await db.execute(
            select(Character).where(Character.project_id == project.id)
        )
        all_characters = characters_result.scalars().all()
        
        if not all_characters:
            return "暂无角色信息", None
        
        # 构建全局角色名称映射（用于关系查询）
        all_char_map = {c.id: c.name for c in all_characters}
        
        # 从expansion_plan中提取角色焦点
        filter_character_names = None
        if chapter.expansion_plan:
            try:
                plan = json.loads(chapter.expansion_plan)
                filter_character_names = plan.get('character_focus', [])
            except json.JSONDecodeError:
                pass
        
        # 筛选角色
        characters = all_characters
        if filter_character_names:
            characters = [c for c in all_characters if c.name in filter_character_names]
        
        if not characters:
            return "暂无相关角色", None
        
        # 限制最多10个角色
        characters = characters[:10]
        character_ids = [c.id for c in characters]
        
        # === 批量查询关系数据 ===
        rels_result = await db.execute(
            select(CharacterRelationship).where(
                CharacterRelationship.project_id == project.id,
                or_(
                    CharacterRelationship.character_from_id.in_(character_ids),
                    CharacterRelationship.character_to_id.in_(character_ids)
                )
            )
        )
        all_rels = rels_result.scalars().all()
        
        # 按角色ID分组关系
        char_rels_map: Dict[str, List] = {cid: [] for cid in character_ids}
        for r in all_rels:
            if r.character_from_id in char_rels_map:
                char_rels_map[r.character_from_id].append(r)
            if r.character_to_id in char_rels_map:
                char_rels_map[r.character_to_id].append(r)
        
        # === 批量查询组织成员数据 ===
        non_org_ids = [c.id for c in characters if not c.is_organization]
        org_memberships_map: Dict[str, List] = {cid: [] for cid in non_org_ids}
        
        if non_org_ids:
            member_result = await db.execute(
                select(OrganizationMember, Character.name).join(
                    Organization, OrganizationMember.organization_id == Organization.id
                ).join(
                    Character, Organization.character_id == Character.id
                ).where(OrganizationMember.character_id.in_(non_org_ids))
            )
            for m, org_name in member_result.all():
                if m.character_id in org_memberships_map:
                    org_memberships_map[m.character_id].append((m, org_name))
        
        # === 批量查询职业关联数据（CharacterCareer）===
        char_career_result = await db.execute(
            select(CharacterCareer).where(CharacterCareer.character_id.in_(character_ids))
        )
        all_char_careers = char_career_result.scalars().all()
        
        # 收集所有职业ID
        career_ids = set()
        for cc in all_char_careers:
            career_ids.add(cc.career_id)
        # 也加入 main_career_id
        for c in characters:
            if not c.is_organization and c.main_career_id:
                career_ids.add(c.main_career_id)
        
        careers_map: Dict[str, Career] = {}
        if career_ids:
            careers_result = await db.execute(
                select(Career).where(Career.id.in_(list(career_ids)))
            )
            careers_map = {c.id: c for c in careers_result.scalars().all()}
        
        # 构建角色ID到职业关联的映射
        char_career_relations: Dict[str, Dict[str, List]] = {}
        for cc in all_char_careers:
            if cc.character_id not in char_career_relations:
                char_career_relations[cc.character_id] = {'main': [], 'sub': []}
            if cc.career_type == 'main':
                char_career_relations[cc.character_id]['main'].append(cc)
            else:
                char_career_relations[cc.character_id]['sub'].append(cc)
        
        # === 查询组织角色的成员列表 ===
        org_chars = [c for c in characters if c.is_organization]
        org_members_map: Dict[str, List] = {}
        
        if org_chars:
            org_char_ids = [c.id for c in org_chars]
            orgs_result = await db.execute(
                select(Organization).where(Organization.character_id.in_(org_char_ids))
            )
            orgs = orgs_result.scalars().all()
            
            if orgs:
                org_id_to_char_id = {o.id: o.character_id for o in orgs}
                org_ids = [o.id for o in orgs]
                
                members_result = await db.execute(
                    select(OrganizationMember, Character.name).join(
                        Character, OrganizationMember.character_id == Character.id
                    ).where(OrganizationMember.organization_id.in_(org_ids))
                )
                for m, member_name in members_result.all():
                    char_id = org_id_to_char_id.get(m.organization_id)
                    if char_id:
                        if char_id not in org_members_map:
                            org_members_map[char_id] = []
                        org_members_map[char_id].append((m, member_name))
        
        # === 构建完整版角色信息 ===
        characters_info_parts = []
        for c in characters:
            entity_type = '组织' if c.is_organization else '角色'
            role_type_map = {
                'protagonist': '主角',
                'antagonist': '反派',
                'supporting': '配角'
            }
            role_type = role_type_map.get(c.role_type, c.role_type or '配角')
            
            info_lines = [f"【{c.name}】({entity_type}, {role_type})"]
            
            # 详细属性
            if c.age:
                info_lines.append(f"  年龄: {c.age}")
            if c.gender:
                info_lines.append(f"  性别: {c.gender}")
            if c.appearance:
                appearance_preview = c.appearance[:100] if len(c.appearance) > 100 else c.appearance
                info_lines.append(f"  外貌: {appearance_preview}")
            if c.personality:
                personality_preview = c.personality[:100] if len(c.personality) > 100 else c.personality
                info_lines.append(f"  性格: {personality_preview}")
            if c.background:
                background_preview = c.background[:150] if len(c.background) > 150 else c.background
                info_lines.append(f"  背景: {background_preview}")
            
            # 职业信息
            if c.id in char_career_relations:
                career_rel = char_career_relations[c.id]
                if career_rel['main']:
                    for cc in career_rel['main']:
                        career = careers_map.get(cc.career_id)
                        if career:
                            try:
                                stages = json.loads(career.stages) if isinstance(career.stages, str) else career.stages
                                stage_name = f'第{cc.current_stage}阶'
                                for stage in (stages or []):
                                    if stage.get('level') == cc.current_stage:
                                        stage_name = stage.get('name', stage_name)
                                        break
                            except (json.JSONDecodeError, AttributeError, TypeError):
                                stage_name = f'第{cc.current_stage}阶'
                            info_lines.append(f"  主职业: {career.name} ({cc.current_stage}/{career.max_stage}阶 - {stage_name})")
                if career_rel['sub']:
                    for cc in career_rel['sub']:
                        career = careers_map.get(cc.career_id)
                        if career:
                            try:
                                stages = json.loads(career.stages) if isinstance(career.stages, str) else career.stages
                                stage_name = f'第{cc.current_stage}阶'
                                for stage in (stages or []):
                                    if stage.get('level') == cc.current_stage:
                                        stage_name = stage.get('name', stage_name)
                                        break
                            except (json.JSONDecodeError, AttributeError, TypeError):
                                stage_name = f'第{cc.current_stage}阶'
                            info_lines.append(f"  副职业: {career.name} ({cc.current_stage}/{career.max_stage}阶 - {stage_name})")
            elif not c.is_organization and c.main_career_id:
                career = careers_map.get(c.main_career_id)
                if career:
                    stage = c.main_career_stage or 1
                    info_lines.append(f"  主职业: {career.name}（第{stage}阶段）")
            
            # 角色关系
            if not c.is_organization and c.id in char_rels_map:
                rels = char_rels_map[c.id]
                if rels:
                    rel_parts = []
                    for r in rels:
                        if r.character_from_id == c.id:
                            target_name = all_char_map.get(r.character_to_id, "未知")
                        else:
                            target_name = all_char_map.get(r.character_from_id, "未知")
                        rel_name = r.relationship_name or "相关"
                        rel_parts.append(f"与{target_name}：{rel_name}")
                    info_lines.append(f"  关系网络: {'；'.join(rel_parts)}")
            
            # 组织归属
            if not c.is_organization and c.id in org_memberships_map:
                memberships = org_memberships_map[c.id]
                if memberships:
                    org_parts = [f"{org_name}（{m.position}）" for m, org_name in memberships[:2]]
                    info_lines.append(f"  组织归属: {'、'.join(org_parts)}")
            
            # 组织特有信息
            if c.is_organization:
                if c.organization_type:
                    info_lines.append(f"  组织类型: {c.organization_type}")
                if c.organization_purpose:
                    info_lines.append(f"  组织目的: {c.organization_purpose[:100]}")
                if c.id in org_members_map:
                    members = org_members_map[c.id]
                    if members:
                        member_parts = [f"{name}（{m.position}）" for m, name in members[:5]]
                        info_lines.append(f"  组织成员: {'、'.join(member_parts)}")
            
            characters_info_parts.append("\n".join(info_lines))
        
        characters_result_str = "\n\n".join(characters_info_parts)
        logger.info(f"  ✅ [1-N完整版] 构建了 {len(characters_info_parts)} 个角色信息，总长度: {len(characters_result_str)} 字符")
        
        # === 构建独立职业详情 ===
        careers_info_parts = []
        if careers_map:
            for career_id, career in careers_map.items():
                career_lines = [f"{career.name} ({career.type}职业)"]
                if career.description:
                    career_lines.append(f"  描述: {career.description}")
                if career.category:
                    career_lines.append(f"  分类: {career.category}")
                try:
                    stages = json.loads(career.stages) if isinstance(career.stages, str) else career.stages
                    if stages:
                        career_lines.append(f"  阶段体系: (共{career.max_stage}阶)")
                        for stage in stages:
                            level = stage.get('level', '?')
                            name = stage.get('name', '未命名')
                            desc = stage.get('description', '')
                            career_lines.append(f"    {level}阶-{name}: {desc}")
                except (json.JSONDecodeError, AttributeError, TypeError):
                    career_lines.append(f"  阶段体系: 共{career.max_stage}阶")
                if career.special_abilities:
                    career_lines.append(f"  特殊能力: {career.special_abilities}")
                careers_info_parts.append("\n".join(career_lines))
        
        careers_result_str = None
        if careers_info_parts:
            careers_result_str = "\n\n".join(careers_info_parts)
            logger.info(f"  ✅ [1-N完整版] 构建了 {len(careers_map)} 个职业详情，总长度: {len(careers_result_str)} 字符")
        
        return characters_result_str, careers_result_str
    
    async def _build_recent_chapters_context(
        self,
        chapter: Chapter,
        project_id: str,
        db: AsyncSession
    ) -> Optional[str]:
        """构建最近10章的expansion_plan摘要"""
        try:
            result = await db.execute(
                select(Chapter.chapter_number, Chapter.title, Chapter.expansion_plan, Chapter.summary)
                .where(Chapter.project_id == project_id)
                .where(Chapter.chapter_number < chapter.chapter_number)
                .order_by(Chapter.chapter_number.desc())
                .limit(self.RECENT_CHAPTERS_COUNT)
            )
            recent_chapters = result.all()
            
            if not recent_chapters:
                return None
            
            # 按章节号正序排列
            recent_chapters = sorted(recent_chapters, key=lambda x: x[0])
            
            lines = ["【最近章节规划】"]
            for ch_num, ch_title, expansion_plan, summary in recent_chapters:
                if expansion_plan:
                    try:
                        plan = json.loads(expansion_plan)
                        plot_summary = plan.get('plot_summary', '')
                        key_events = plan.get('key_events', [])
                        events_str = '；'.join(key_events[:3]) if key_events else ''
                        line = f"第{ch_num}章《{ch_title}》：{plot_summary}"
                        if events_str:
                            line += f"（关键事件：{events_str}）"
                        lines.append(line)
                    except json.JSONDecodeError:
                        if summary:
                            lines.append(f"第{ch_num}章《{ch_title}》：{summary[:100]}")
                elif summary:
                    lines.append(f"第{ch_num}章《{ch_title}》：{summary[:100]}")
            
            if len(lines) <= 1:
                return None
            
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"❌ 构建最近章节上下文失败: {str(e)}")
            return None
    
    async def _get_relevant_memories_enhanced(
        self,
        user_id: str,
        project_id: str,
        chapter_number: int,
        chapter_outline: str,
        db: AsyncSession
    ) -> Optional[str]:
        """获取相关记忆（始终启用，相关度>0.6）"""
        if not self.memory_service:
            return None
        
        try:
            query_text = chapter_outline[:500].replace('\n', ' ')
            
            relevant_memories = await self.memory_service.search_memories(
                user_id=user_id,
                project_id=project_id,
                query=query_text,
                limit=15,
                min_importance=0.0,
                db=db,
            )
            
            # 过滤策略：
            # - 未启用 rerank：沿用原逻辑（相似度阈值）
            # - 启用 rerank：不再用相似度硬过滤，避免把高 rerank 分但低相似度的条目误删
            use_rerank = any(mem.get("rerank_score") is not None for mem in relevant_memories)
            if use_rerank:
                filtered_memories = relevant_memories
            else:
                filtered_memories = [
                    mem for mem in relevant_memories
                    if mem.get('similarity', 0) > self.MEMORY_SIMILARITY_THRESHOLD
                ]
            
            if not filtered_memories:
                return None
            
            memory_lines = ["【相关记忆】"]
            for mem in filtered_memories[:self.MEMORY_COUNT]:
                similarity = mem.get('similarity', 0)
                rerank_score = mem.get("rerank_score")
                content = mem.get('content', '')[:100]
                if rerank_score is not None:
                    try:
                        rs = float(rerank_score)
                        memory_lines.append(f"- (重排分:{rs:.3f}|相似度:{similarity:.2f}) {content}")
                    except Exception:
                        memory_lines.append(f"- (重排分:{rerank_score}|相似度:{similarity:.2f}) {content}")
                else:
                    memory_lines.append(f"- (相关度:{similarity:.2f}) {content}")
            
            return "\n".join(memory_lines) if len(memory_lines) > 1 else None
            
        except Exception as e:
            logger.error(f"❌ 获取相关记忆失败: {str(e)}")
            return None
    
    async def _get_last_ending_enhanced(
        self,
        chapter: Chapter,
        db: AsyncSession,
        max_length: int
    ) -> Dict[str, Any]:
        """获取增强版衔接锚点（含上一章摘要和关键事件）"""
        result_info = {
            'ending_text': None,
            'summary': None,
            'key_events': []
        }
        
        if chapter.chapter_number <= 1:
            return result_info
        
        # 查询上一章：不假设序号连续，取 chapter_number < 当前章 中最大的
        result = await db.execute(
            select(Chapter)
            .where(Chapter.project_id == chapter.project_id)
            .where(Chapter.chapter_number < chapter.chapter_number)
            .order_by(Chapter.chapter_number.desc())
            .limit(1)
        )
        prev_chapter = result.scalar_one_or_none()
        
        if not prev_chapter:
            return result_info
        
        # 1. 提取结尾内容
        if prev_chapter.content:
            content = prev_chapter.content.strip()
            if len(content) <= max_length:
                result_info['ending_text'] = content
            else:
                result_info['ending_text'] = content[-max_length:]
        
        # 2. 获取上一章摘要
        summary_result = await db.execute(
            select(StoryMemory.content)
            .where(StoryMemory.project_id == chapter.project_id)
            .where(StoryMemory.chapter_id == prev_chapter.id)
            .where(StoryMemory.memory_type == 'chapter_summary')
            .limit(1)
        )
        summary_mem = summary_result.scalar_one_or_none()
        
        if summary_mem:
            result_info['summary'] = summary_mem[:300]
        elif prev_chapter.summary:
            result_info['summary'] = prev_chapter.summary[:300]
        elif prev_chapter.expansion_plan:
            try:
                plan = json.loads(prev_chapter.expansion_plan)
                result_info['summary'] = plan.get('plot_summary', '')[:300]
            except json.JSONDecodeError:
                pass
        
        # 3. 提取上一章关键事件
        if prev_chapter.expansion_plan:
            try:
                plan = json.loads(prev_chapter.expansion_plan)
                key_events = plan.get('key_events', [])
                if key_events:
                    result_info['key_events'] = key_events[:5]
            except json.JSONDecodeError:
                pass
        
        return result_info
    
    def _extract_emotional_tone(
        self,
        chapter: Chapter,
        outline: Optional[Outline]
    ) -> str:
        """提取本章情感基调"""
        if chapter.expansion_plan:
            try:
                plan = json.loads(chapter.expansion_plan)
                tone = plan.get('emotional_tone')
                if tone:
                    return tone
            except json.JSONDecodeError:
                pass
        
        if outline and outline.structure:
            try:
                structure = json.loads(outline.structure)
                tone = structure.get('emotion') or structure.get('emotional_tone')
                if tone:
                    return tone
            except json.JSONDecodeError:
                pass
        
        return "未设定"
    
    def _summarize_style(self, style_content: str) -> str:
        """将风格描述压缩为关键要点"""
        if not style_content:
            return ""
        
        if len(style_content) <= self.STYLE_MAX_LENGTH:
            return style_content
        
        return style_content[:self.STYLE_MAX_LENGTH] + "..."
    
    async def _get_relevant_memories(
        self,
        user_id: str,
        project_id: str,
        chapter_number: int,
        chapter_outline: str,
        limit: int = 3,
        db: Optional[AsyncSession] = None,
    ) -> Optional[str]:
        """
        获取与本章最相关的记忆
        
        注意：伏笔相关信息统一由 _get_foreshadow_reminders() 通过 foreshadow_service 提供，
        此方法只负责获取故事记忆，不再从旧的 memory_service 获取伏笔信息。
        """
        if not self.memory_service:
            return None
        
        try:
            relevant = await self.memory_service.search_memories(
                user_id=user_id,
                project_id=project_id,
                query=chapter_outline,
                limit=limit,
                min_importance=self.MEMORY_IMPORTANCE_THRESHOLD,
                db=db,
            )
            
            return self._format_memories(relevant, max_length=500)
            
        except Exception as e:
            logger.error(f"❌ 获取相关记忆失败: {str(e)}")
            return None
    
    def _format_memories(
        self,
        relevant: List[Dict[str, Any]],
        max_length: int = 500
    ) -> str:
        """格式化记忆为简洁文本（纯记忆，不含伏笔）"""
        if not relevant:
            return None
        
        lines = ["【相关记忆】"]
        current_length = 0
        
        for mem in relevant:
            content = mem.get('content', '')[:80]
            text = f"- {content}"
            if current_length + len(text) > max_length:
                break
            lines.append(text)
            current_length += len(text)
        
        return "\n".join(lines) if len(lines) > 1 else None
    
    async def _get_foreshadow_reminders(
        self,
        project_id: str,
        chapter_number: int,
        db: AsyncSession
    ) -> Optional[str]:
        """
        获取伏笔提醒信息（增强版）
        
        策略：
        1. 本章必须回收的伏笔（target_resolve_chapter_number == chapter_number）
        2. 超期未回收的伏笔（target_resolve_chapter_number < chapter_number）
        3. 即将到期的伏笔（target_resolve_chapter_number 在未来3章内）
        """
        if not self.foreshadow_service:
            return None
        
        try:
            lines = []
            
            # 1. 本章必须回收的伏笔
            must_resolve = await self.foreshadow_service.get_must_resolve_foreshadows(
                db=db,
                project_id=project_id,
                chapter_number=chapter_number
            )
            
            if must_resolve:
                lines.append("【🎯 本章必须回收的伏笔】")
                for f in must_resolve:
                    lines.append(f"- {f.title}")
                    lines.append(f"  埋入章节：第{f.plant_chapter_number}章")
                    lines.append(f"  伏笔内容：{f.content[:100]}{'...' if len(f.content) > 100 else ''}")
                    if f.resolution_notes:
                        lines.append(f"  回收提示：{f.resolution_notes}")
                    lines.append("")
            
            # 2. 超期未回收的伏笔
            overdue = await self.foreshadow_service.get_overdue_foreshadows(
                db=db,
                project_id=project_id,
                current_chapter=chapter_number
            )
            
            if overdue:
                lines.append("【⚠️ 超期待回收伏笔】")
                for f in overdue[:3]:  # 最多显示3个
                    overdue_chapters = chapter_number - (f.target_resolve_chapter_number or 0)
                    lines.append(f"- {f.title} [已超期{overdue_chapters}章]")
                    lines.append(f"  埋入章节：第{f.plant_chapter_number}章，原计划第{f.target_resolve_chapter_number}章回收")
                    lines.append(f"  伏笔内容：{f.content[:80]}...")
                    lines.append("")
            
            # 3. 即将到期的伏笔（未来3章内）
            upcoming = await self.foreshadow_service.get_pending_resolve_foreshadows(
                db=db,
                project_id=project_id,
                current_chapter=chapter_number,
                lookahead=3
            )
            
            # 过滤：只保留未来章节的，排除本章和超期的
            upcoming_filtered = [f for f in upcoming
                               if (f.target_resolve_chapter_number or 0) > chapter_number]
            
            if upcoming_filtered:
                lines.append("【📋 即将到期的伏笔（仅供参考）】")
                for f in upcoming_filtered[:3]:  # 最多显示3个
                    remaining = (f.target_resolve_chapter_number or 0) - chapter_number
                    lines.append(f"- {f.title}（计划第{f.target_resolve_chapter_number}章回收，还有{remaining}章）")
                lines.append("")
            
            return "\n".join(lines) if lines else None
            
        except Exception as e:
            logger.error(f"❌ 获取伏笔提醒失败: {str(e)}")
            return None
    
    async def _build_story_skeleton(
        self,
        project_id: str,
        chapter_number: int,
        db: AsyncSession
    ) -> Optional[str]:
        """构建故事骨架（每N章采样）"""
        try:
            result = await db.execute(
                select(Chapter.id, Chapter.chapter_number, Chapter.title)
                .where(Chapter.project_id == project_id)
                .where(Chapter.chapter_number < chapter_number)
                .where(Chapter.content != None)
                .where(Chapter.content != "")
                .order_by(Chapter.chapter_number)
            )
            chapters = result.all()
            
            if not chapters:
                return None
            
            skeleton_lines = ["【故事骨架】"]
            for i, (ch_id, ch_num, ch_title) in enumerate(chapters):
                if i % self.SKELETON_SAMPLE_INTERVAL == 0:
                    summary_result = await db.execute(
                        select(StoryMemory.content)
                        .where(StoryMemory.project_id == project_id)
                        .where(StoryMemory.chapter_id == ch_id)
                        .where(StoryMemory.memory_type == 'chapter_summary')
                        .limit(1)
                    )
                    summary = summary_result.scalar_one_or_none()
                    
                    if summary:
                        skeleton_lines.append(f"第{ch_num}章《{ch_title}》：{summary[:100]}")
                    else:
                        skeleton_lines.append(f"第{ch_num}章《{ch_title}》")
            
            if len(skeleton_lines) <= 1:
                return None
            
            return "\n".join(skeleton_lines)
            
        except Exception as e:
            logger.error(f"❌ 构建故事骨架失败: {str(e)}")
            return None


# ==================== 1-1模式上下文构建器 ====================

class OneToOneContextBuilder:
    """
    1-1模式上下文构建器
    
    上下文构建策略：
    P0核心信息：
    1. 从outline.structure的JSON中提取：summary, scenes, key_points, emotion, goal
    2. target_word_count
    
    P1重要信息：
    1. 上一章完整内容的最后500字作为参考
    2. 根据structure中的characters获取角色信息（含职业）
    
    P2参考信息：
    1. 伏笔提醒
    2. 根据角色名检索相关记忆（相关度>0.6）
    """
    
    def __init__(self, memory_service=None, foreshadow_service=None):
        """
        初始化构建器
        
        Args:
            memory_service: 记忆服务实例（可选）
            foreshadow_service: 伏笔服务实例（可选）
        """
        self.memory_service = memory_service
        self.foreshadow_service = foreshadow_service
    
    async def build(
        self,
        chapter: Chapter,
        project: Project,
        outline: Optional[Outline],
        user_id: str,
        db: AsyncSession,
        target_word_count: int = 3000
    ) -> OneToOneContext:
        """
        构建1-1模式上下文
        
        Args:
            chapter: 章节对象
            project: 项目对象
            outline: 大纲对象
            user_id: 用户ID
            db: 数据库会话
            target_word_count: 目标字数
            
        Returns:
            OneToOneContext: 上下文对象
        """
        chapter_number = chapter.chapter_number
        logger.info(f"📝 [1-1模式] 开始构建上下文: 第{chapter_number}章")
        
        # 初始化上下文
        context = OneToOneContext(
            chapter_number=chapter_number,
            chapter_title=chapter.title or "",
            title=project.title or "",
            genre=project.genre or "",
            theme=project.theme or "",
            target_word_count=target_word_count,
            min_word_count=max(500, target_word_count - 500),
            max_word_count=target_word_count + 1000,
            narrative_perspective=project.narrative_perspective or "第三人称"
        )
        
        # === P0-核心信息 ===
        context.chapter_outline = self._build_outline_from_structure(outline, chapter)
        logger.info(f"  ✅ P0-大纲信息: {len(context.chapter_outline)}字符")
        
        # === P1-重要信息 ===
        # 1. 获取上一章内容的最后500字和上一章摘要
        if chapter_number > 1:
            # 查找前一章：不假设序号连续，取 chapter_number < 当前章 中最大的
            prev_chapter_result = await db.execute(
                select(Chapter)
                .where(Chapter.project_id == chapter.project_id)
                .where(Chapter.chapter_number < chapter_number)
                .order_by(Chapter.chapter_number.desc())
                .limit(1)
            )
            prev_chapter = prev_chapter_result.scalar_one_or_none()
            
            if prev_chapter and prev_chapter.content:
                content = prev_chapter.content.strip()
                if len(content) <= 500:
                    context.continuation_point = content
                else:
                    context.continuation_point = content[-500:]
                logger.info(f"  ✅ P1-上一章内容(最后500字): {len(context.continuation_point)}字符")
                
                # 获取上一章摘要（优先从记忆系统获取，其次使用章节摘要）
                summary_result = await db.execute(
                    select(StoryMemory.content)
                    .where(StoryMemory.project_id == chapter.project_id)
                    .where(StoryMemory.chapter_id == prev_chapter.id)
                    .where(StoryMemory.memory_type == 'chapter_summary')
                    .limit(1)
                )
                summary_mem = summary_result.scalar_one_or_none()
                
                if summary_mem:
                    context.previous_chapter_summary = summary_mem[:300]
                    logger.info(f"  ✅ P1-上一章摘要(记忆): {len(context.previous_chapter_summary)}字符")
                elif prev_chapter.summary:
                    context.previous_chapter_summary = prev_chapter.summary[:300]
                    logger.info(f"  ✅ P1-上一章摘要(章节): {len(context.previous_chapter_summary)}字符")
                else:
                    context.previous_chapter_summary = None
                    logger.info(f"  ⚠️ P1-上一章摘要: 无")
            else:
                context.continuation_point = None
                context.previous_chapter_summary = None
                logger.info(f"  ⚠️ P1-上一章内容: 无")
        else:
            context.continuation_point = None
            context.previous_chapter_summary = None
            logger.info(f"  ✅ P1-第1章无需上一章内容")
        
        # 2. 根据structure中的characters获取角色信息（含职业）
        character_names = []
        if outline and outline.structure:
            try:
                structure = json.loads(outline.structure)
                raw_characters = structure.get('characters', [])
                # characters可能是字符串列表或字典列表，统一提取为名称字符串列表
                character_names = [
                    c['name'] if isinstance(c, dict) else c
                    for c in raw_characters
                ]
                logger.info(f"  📋 从structure提取角色: {character_names}")
            except json.JSONDecodeError:
                pass
        
        if character_names:
            # 获取角色基本信息
            characters_result = await db.execute(
                select(Character)
                .where(Character.project_id == project.id)
                .where(Character.name.in_(character_names))
            )
            characters = characters_result.scalars().all()
            
            if characters:
                # 构建包含职业信息的角色上下文和职业详情
                characters_info, careers_info = await self._build_characters_and_careers(
                    db=db,
                    project_id=project.id,
                    characters=characters,
                    filter_character_names=character_names
                )
                context.chapter_characters = characters_info
                context.chapter_careers = careers_info
                logger.info(f"  ✅ P1-角色信息: {len(context.chapter_characters)}字符")
                logger.info(f"  ✅ P1-职业信息: {len(context.chapter_careers or '')}字符")
            else:
                context.chapter_characters = "暂无角色信息"
                context.chapter_careers = None
                logger.info(f"  ⚠️ P1-角色信息: 筛选后无匹配角色")
        else:
            context.chapter_characters = "暂无角色信息"
            context.chapter_careers = None
            logger.info(f"  ⚠️ P1-角色信息: 无")
        
        # === P2-参考信息 ===
        # 1. 伏笔提醒
        if self.foreshadow_service:
            context.foreshadow_reminders = await self._get_foreshadow_reminders(
                project.id, chapter_number, db
            )
            if context.foreshadow_reminders:
                logger.info(f"  ✅ P2-伏笔提醒: {len(context.foreshadow_reminders)}字符")
            else:
                logger.info(f"  ⚠️ P2-伏笔提醒: 无")
        
        # 2. 根据大纲内容检索相关记忆（相关度>0.4）
        if self.memory_service and context.chapter_outline:
            try:
                # 使用大纲内容作为查询（截取前500字符以避免过长）
                query_text = context.chapter_outline[:500].replace('\n', ' ')
                logger.info(f"  🔍 记忆查询关键词: {query_text[:100]}...")
                
                relevant_memories = await self.memory_service.search_memories(
                    user_id=user_id,
                    project_id=project.id,
                    query=query_text,
                    limit=15,
                    min_importance=0.0,
                    db=db,
                )
                
                # 过滤策略：
                # - 未启用 rerank：沿用原逻辑（相似度阈值）
                # - 启用 rerank：不再用相似度硬过滤
                use_rerank = any(mem.get("rerank_score") is not None for mem in relevant_memories)
                if use_rerank:
                    filtered_memories = relevant_memories
                else:
                    filtered_memories = [
                        mem for mem in relevant_memories
                        if mem.get('similarity', 0) > 0.6
                    ]
                
                if filtered_memories:
                    memory_lines = ["【相关记忆】"]
                    for mem in filtered_memories[:10]:  # 最多显示10条
                        similarity = mem.get('similarity', 0)
                        content = mem.get('content', '')[:100]
                        rerank_score = mem.get("rerank_score")
                        if rerank_score is not None:
                            try:
                                rs = float(rerank_score)
                                memory_lines.append(f"- (重排分:{rs:.3f}|相似度:{similarity:.2f}) {content}")
                            except Exception:
                                memory_lines.append(f"- (重排分:{rerank_score}|相似度:{similarity:.2f}) {content}")
                        else:
                            memory_lines.append(f"- (相关度:{similarity:.2f}) {content}")
                    
                    context.relevant_memories = "\n".join(memory_lines)
                    logger.info(f"  ✅ P2-相关记忆: {len(filtered_memories)}条 (相关度>0.6, 共搜索{len(relevant_memories)}条)")
                else:
                    context.relevant_memories = None
                    logger.info(f"  ⚠️ P2-相关记忆: 无符合条件的记忆 (共搜索到{len(relevant_memories)}条)")
                    
            except Exception as e:
                logger.error(f"  ❌ 检索相关记忆失败: {str(e)}")
                context.relevant_memories = None
        else:
            context.relevant_memories = None
            logger.info(f"  ⚠️ P2-相关记忆: 无大纲内容或记忆服务不可用")
        
        # === 统计信息 ===
        context.context_stats = {
            "mode": "one-to-one",
            "chapter_number": chapter_number,
            "has_previous_content": context.continuation_point is not None,
            "previous_content_length": len(context.continuation_point or ""),
            "previous_summary_length": len(context.previous_chapter_summary or ""),
            "outline_length": len(context.chapter_outline),
            "characters_length": len(context.chapter_characters),
            "careers_length": len(context.chapter_careers or ""),
            "foreshadow_length": len(context.foreshadow_reminders or ""),
            "memories_length": len(context.relevant_memories or ""),
            "total_length": context.get_total_context_length()
        }
        
        logger.info(f"📊 [1-1模式] 上下文构建完成: 总长度 {context.context_stats['total_length']} 字符")
        
        return context
    
    def _build_outline_from_structure(
        self,
        outline: Optional[Outline],
        chapter: Chapter
    ) -> str:
        """从outline.structure提取大纲信息（1-1模式专用）"""
        if outline and outline.structure:
            try:
                structure = json.loads(outline.structure)
                
                outline_parts = []
                
                if structure.get('summary'):
                    outline_parts.append(f"【章节概要】\n{structure['summary']}")
                
                if structure.get('scenes'):
                    scenes_text = "\n".join([f"- {scene}" for scene in structure['scenes']])
                    outline_parts.append(f"【场景设定】\n{scenes_text}")
                
                if structure.get('key_points'):
                    points_text = "\n".join([f"- {point}" for point in structure['key_points']])
                    outline_parts.append(f"【情节要点】\n{points_text}")
                
                if structure.get('emotion'):
                    outline_parts.append(f"【情感基调】\n{structure['emotion']}")
                
                if structure.get('goal'):
                    outline_parts.append(f"【叙事目标】\n{structure['goal']}")
                
                return "\n\n".join(outline_parts)
                
            except json.JSONDecodeError as e:
                logger.error(f"  ❌ 解析outline.structure失败: {e}")
                return outline.content if outline else "暂无大纲"
        else:
            return outline.content if outline else "暂无大纲"
    
    async def _build_characters_and_careers(
        self,
        db: AsyncSession,
        project_id: str,
        characters: list,
        filter_character_names: Optional[list] = None
    ) -> tuple[str, Optional[str]]:
        """
        构建角色信息和职业信息（1-1模式专用）
        获取角色的完整数据，并关联查询每个职业的完整数据
        分别返回角色信息和职业信息
        
        Args:
            db: 数据库会话
            project_id: 项目ID
            characters: 角色列表
            filter_character_names: 筛选的角色名称列表
            
        Returns:
            tuple: (角色信息字符串, 职业信息字符串)
        """
        if not characters:
            return '暂无角色信息', None
        
        # 如果提供了筛选名单，只保留匹配的角色
        if filter_character_names:
            filtered_characters = [c for c in characters if c.name in filter_character_names]
            if not filtered_characters:
                logger.warning(f"筛选后无匹配角色，使用全部角色。筛选名单: {filter_character_names}")
                filtered_characters = characters
            else:
                logger.info(f"根据筛选名单保留 {len(filtered_characters)}/{len(characters)} 个角色: {[c.name for c in filtered_characters]}")
            characters = filtered_characters
        
        # 获取角色ID列表
        character_ids = [c.id for c in characters]
        if not character_ids:
            return '暂无角色信息', None
        
        # 重新查询角色的完整数据（确保获取所有字段）
        full_characters_result = await db.execute(
            select(Character).where(Character.id.in_(character_ids))
        )
        full_characters = {c.id: c for c in full_characters_result.scalars().all()}
        
        # 获取所有角色的职业关联数据
        character_careers_result = await db.execute(
            select(CharacterCareer).where(CharacterCareer.character_id.in_(character_ids))
        )
        character_careers = character_careers_result.scalars().all()
        
        # 收集所有需要查询的职业ID
        career_ids = set()
        for cc in character_careers:
            career_ids.add(cc.career_id)
        
        # 查询所有相关职业的完整数据
        careers_map = {}
        if career_ids:
            careers_result = await db.execute(
                select(Career).where(Career.id.in_(list(career_ids)))
            )
            careers_map = {c.id: c for c in careers_result.scalars().all()}
            logger.info(f"  📋 查询到 {len(careers_map)} 个职业的完整数据")
        
        # 构建角色ID到职业关联数据的映射
        char_career_relations = {}
        for cc in character_careers:
            if cc.character_id not in char_career_relations:
                char_career_relations[cc.character_id] = {'main': [], 'sub': []}
            
            # 保存完整的CharacterCareer对象
            if cc.career_type == 'main':
                char_career_relations[cc.character_id]['main'].append(cc)
            else:
                char_career_relations[cc.character_id]['sub'].append(cc)
        
        # 构建角色信息字符串
        characters_info_parts = []
        for char_id in character_ids[:10]:  # 限制最多10个角色
            c = full_characters.get(char_id)
            if not c:
                continue
            
            # === 角色基本信息 ===
            entity_type = '组织' if c.is_organization else '角色'
            role_type_map = {
                'protagonist': '主角',
                'antagonist': '反派',
                'supporting': '配角'
            }
            role_type = role_type_map.get(c.role_type, c.role_type or '配角')
            
            # 构建基本信息行
            info_lines = [f"【{c.name}】({entity_type}, {role_type})"]
            
            # === 角色详细属性 ===
            if c.age:
                info_lines.append(f"  年龄: {c.age}")
            if c.gender:
                info_lines.append(f"  性别: {c.gender}")
            if c.appearance:
                appearance_preview = c.appearance[:100] if len(c.appearance) > 100 else c.appearance
                info_lines.append(f"  外貌: {appearance_preview}")
            if c.personality:
                personality_preview = c.personality[:100] if len(c.personality) > 100 else c.personality
                info_lines.append(f"  性格: {personality_preview}")
            if c.background:
                background_preview = c.background[:150] if len(c.background) > 150 else c.background
                info_lines.append(f"  背景: {background_preview}")
            
            # === 职业信息（完整数据）===
            if char_id in char_career_relations:
                career_relations = char_career_relations[char_id]
                
                # 主职业
                if career_relations['main']:
                    for cc in career_relations['main']:
                        career = careers_map.get(cc.career_id)
                        if career:
                            # 解析职业的完整阶段信息
                            try:
                                stages = json.loads(career.stages) if isinstance(career.stages, str) else career.stages
                                current_stage_info = None
                                for stage in stages:
                                    if stage.get('level') == cc.current_stage:
                                        current_stage_info = stage
                                        break
                                
                                stage_name = current_stage_info.get('name', f'第{cc.current_stage}阶') if current_stage_info else f'第{cc.current_stage}阶'
                            except (json.JSONDecodeError, AttributeError, TypeError) as e:
                                logger.warning(f"解析职业阶段信息失败: {e}")
                                stage_name = f'第{cc.current_stage}阶'
                                stage_desc = ''
                            
                            # 构建主职业信息（只显示引用，详细信息在下面的"本章职业"部分）
                            info_lines.append(f"  主职业: {career.name} ({cc.current_stage}/{career.max_stage}阶 - {stage_name})")
                
                # 副职业
                if career_relations['sub']:
                    info_lines.append(f"  副职业:")
                    for cc in career_relations['sub']:
                        career = careers_map.get(cc.career_id)
                        if career:
                            # 解析副职业阶段信息
                            try:
                                stages = json.loads(career.stages) if isinstance(career.stages, str) else career.stages
                                current_stage_info = None
                                for stage in stages:
                                    if stage.get('level') == cc.current_stage:
                                        current_stage_info = stage
                                        break
                                stage_name = current_stage_info.get('name', f'第{cc.current_stage}阶') if current_stage_info else f'第{cc.current_stage}阶'
                            except (json.JSONDecodeError, AttributeError, TypeError):
                                stage_name = f'第{cc.current_stage}阶'
                            
                            # 副职业也只显示引用
                            info_lines.append(f"    - {career.name} ({cc.current_stage}/{career.max_stage}阶 - {stage_name})")
            
            # === 角色关系信息 ===
            if not c.is_organization:
                from sqlalchemy import or_
                rels_result = await db.execute(
                    select(CharacterRelationship).where(
                        CharacterRelationship.project_id == project_id,
                        or_(
                            CharacterRelationship.character_from_id == c.id,
                            CharacterRelationship.character_to_id == c.id
                        )
                    )
                )
                rels = rels_result.scalars().all()
                if rels:
                    related_ids = set()
                    for r in rels:
                        related_ids.add(r.character_from_id)
                        related_ids.add(r.character_to_id)
                    related_ids.discard(c.id)
                    if related_ids:
                        names_result = await db.execute(
                            select(Character.id, Character.name).where(Character.id.in_(related_ids))
                        )
                        name_map = {row.id: row.name for row in names_result}
                        rel_parts = []
                        for r in rels:
                            if r.character_from_id == c.id:
                                target_name = name_map.get(r.character_to_id, "未知")
                            else:
                                target_name = name_map.get(r.character_from_id, "未知")
                            rel_name = r.relationship_name or "相关"
                            rel_parts.append(f"与{target_name}：{rel_name}")
                        info_lines.append(f"  关系网络: {'；'.join(rel_parts)}")
            
            # === 组织特有信息 ===
            if c.is_organization:
                if c.organization_type:
                    info_lines.append(f"  组织类型: {c.organization_type}")
                if c.organization_purpose:
                    info_lines.append(f"  组织目的: {c.organization_purpose[:100]}")
                # 从 OrganizationMember 表动态查询组织成员
                org_result = await db.execute(
                    select(Organization).where(Organization.character_id == c.id)
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
                        info_lines.append(f"  组织成员: {'、'.join(member_parts)[:100]}")
            
            # 组合完整信息
            full_info = "\n".join(info_lines)
            characters_info_parts.append(full_info)
        
        characters_result = "\n\n".join(characters_info_parts)
        logger.info(f"  ✅ 构建了 {len(characters_info_parts)} 个角色的完整信息，总长度: {len(characters_result)} 字符")
        
        # === 构建职业信息部分 ===
        careers_info_parts = []
        if careers_map:
            for career_id, career in careers_map.items():
                career_lines = [f"{career.name} ({career.type}职业)"]
                
                # 职业描述
                if career.description:
                    career_lines.append(f"  描述: {career.description}")
                
                # 职业分类
                if career.category:
                    career_lines.append(f"  分类: {career.category}")
                
                # 阶段体系
                try:
                    stages = json.loads(career.stages) if isinstance(career.stages, str) else career.stages
                    if stages:
                        career_lines.append(f"  阶段体系: (共{career.max_stage}阶)")
                        for stage in stages:  # 显示所有阶段
                            level = stage.get('level', '?')
                            name = stage.get('name', '未命名')
                            desc = stage.get('description', '')
                            career_lines.append(f"    {level}阶-{name}: {desc}")
                except (json.JSONDecodeError, AttributeError, TypeError) as e:
                    logger.warning(f"解析职业阶段失败: {e}")
                    career_lines.append(f"  阶段体系: 共{career.max_stage}阶")
                
                # 职业要求
                if career.requirements:
                    career_lines.append(f"  职业要求: {career.requirements}")
                
                # 特殊能力
                if career.special_abilities:
                    career_lines.append(f"  特殊能力: {career.special_abilities}")
                
                # 世界观规则
                if career.worldview_rules:
                    career_lines.append(f"  世界观规则: {career.worldview_rules}")
                
                # 属性加成
                if career.attribute_bonuses:
                    try:
                        bonuses = json.loads(career.attribute_bonuses) if isinstance(career.attribute_bonuses, str) else career.attribute_bonuses
                        if bonuses:
                            bonus_str = ", ".join([f"{k}:{v}" for k, v in bonuses.items()])
                            career_lines.append(f"  属性加成: {bonus_str}")
                    except (json.JSONDecodeError, AttributeError, TypeError):
                        pass
                
                careers_info_parts.append("\n".join(career_lines))
        
        careers_result = None
        if careers_info_parts:  # 有职业数据就返回
            careers_result = "\n\n".join(careers_info_parts)
            logger.info(f"  ✅ 构建了 {len(careers_map)} 个职业的完整信息，总长度: {len(careers_result)} 字符")
        else:
            logger.info(f"  ⚠️ 本章无涉及职业")
        
        return characters_result, careers_result
    
    async def _get_foreshadow_reminders(
        self,
        project_id: str,
        chapter_number: int,
        db: AsyncSession
    ) -> Optional[str]:
        """
        获取伏笔提醒信息（增强版）
        
        策略：
        1. 本章必须回收的伏笔（target_resolve_chapter_number == chapter_number）
        2. 超期未回收的伏笔（target_resolve_chapter_number < chapter_number）
        3. 即将到期的伏笔（target_resolve_chapter_number 在未来3章内）
        """
        if not self.foreshadow_service:
            return None
        
        try:
            lines = []
            
            # 1. 本章必须回收的伏笔
            must_resolve = await self.foreshadow_service.get_must_resolve_foreshadows(
                db=db,
                project_id=project_id,
                chapter_number=chapter_number
            )
            
            if must_resolve:
                lines.append("【🎯 本章必须回收的伏笔】")
                for f in must_resolve:
                    lines.append(f"- {f.title}")
                    lines.append(f"  埋入章节：第{f.plant_chapter_number}章")
                    lines.append(f"  伏笔内容：{f.content[:100]}{'...' if len(f.content) > 100 else ''}")
                    if f.resolution_notes:
                        lines.append(f"  回收提示：{f.resolution_notes}")
                    lines.append("")
            
            # 2. 超期未回收的伏笔
            overdue = await self.foreshadow_service.get_overdue_foreshadows(
                db=db,
                project_id=project_id,
                current_chapter=chapter_number
            )
            
            if overdue:
                lines.append("【⚠️ 超期待回收伏笔】")
                for f in overdue[:3]:  # 最多显示3个
                    overdue_chapters = chapter_number - (f.target_resolve_chapter_number or 0)
                    lines.append(f"- {f.title} [已超期{overdue_chapters}章]")
                    lines.append(f"  埋入章节：第{f.plant_chapter_number}章，原计划第{f.target_resolve_chapter_number}章回收")
                    lines.append(f"  伏笔内容：{f.content[:80]}...")
                    lines.append("")
            
            # 3. 即将到期的伏笔（未来3章内）
            upcoming = await self.foreshadow_service.get_pending_resolve_foreshadows(
                db=db,
                project_id=project_id,
                current_chapter=chapter_number,
                lookahead=3
            )
            
            # 过滤：只保留未来章节的，排除本章和超期的
            upcoming_filtered = [f for f in upcoming
                               if (f.target_resolve_chapter_number or 0) > chapter_number]
            
            if upcoming_filtered:
                lines.append("【📋 即将到期的伏笔（仅供参考）】")
                for f in upcoming_filtered[:3]:  # 最多显示3个
                    remaining = (f.target_resolve_chapter_number or 0) - chapter_number
                    lines.append(f"- {f.title}（计划第{f.target_resolve_chapter_number}章回收，还有{remaining}章）")
                lines.append("")
            
            return "\n".join(lines) if lines else None
            
        except Exception as e:
            logger.error(f"❌ 获取伏笔提醒失败: {str(e)}")
            return None
