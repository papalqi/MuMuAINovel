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
from app.logger import get_logger

logger = get_logger(__name__)


@dataclass
class OneToManyContext:
    """
    1-N模式章节上下文数据结构
    
    采用RTCO框架的分层设计：
    - P0-核心：大纲、衔接锚点、字数要求
    - P1-重要：角色、情感基调、风格
    - P2-参考：记忆、故事骨架、伏笔提醒
    """
    
    # === P0-核心信息 ===
    chapter_outline: str = ""           # 本章大纲（从expansion_plan构建）
    continuation_point: Optional[str] = None  # 衔接锚点
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
    chapter_characters: str = ""        # 从character_focus筛选的角色
    emotional_tone: str = ""
    style_instruction: str = ""
    
    # === P2-参考信息 ===
    relevant_memories: Optional[str] = None
    story_skeleton: Optional[str] = None  # 50章+启用
    foreshadow_reminders: Optional[str] = None
    
    # === 元信息 ===
    context_stats: Dict[str, Any] = field(default_factory=dict)
    
    def get_total_context_length(self) -> int:
        """计算总上下文长度"""
        total = 0
        for field_name in ['chapter_outline', 'continuation_point', 'chapter_characters',
                          'relevant_memories', 'story_skeleton', 'style_instruction',
                          'foreshadow_reminders', 'previous_chapter_summary']:
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
        for field_name in ['chapter_outline', 'continuation_point', 'chapter_characters',
                          'chapter_careers', 'foreshadow_reminders', 'relevant_memories']:
            value = getattr(self, field_name, None)
            if value:
                total += len(value)
        return total


# ==================== 1-N模式上下文构建器 ====================

class OneToManyContextBuilder:
    """
    1-N模式上下文构建器
    
    实现动态裁剪逻辑，根据章节序号自动调整上下文复杂度：
    - 第1章：无前置上下文，仅提供大纲和角色
    - 第2-10章：上一章结尾300字 + 涉及角色
    - 第11-50章：上一章结尾500字 + 相关记忆3条
    - 第51章+：上一章结尾500字 + 故事骨架 + 智能记忆5条
    """
    
    # 配置常量
    ENDING_LENGTH_SHORT = 300    # 1-10章：短衔接
    ENDING_LENGTH_NORMAL = 500   # 11章+：标准衔接
    MEMORY_COUNT_LIGHT = 3       # 11-50章：轻量记忆
    MEMORY_COUNT_FULL = 5        # 51章+：完整记忆
    SKELETON_THRESHOLD = 50      # 启用故事骨架的章节阈值
    SKELETON_SAMPLE_INTERVAL = 10  # 故事骨架采样间隔
    MEMORY_IMPORTANCE_THRESHOLD = 0.7  # 记忆重要性阈值
    STYLE_MAX_LENGTH = 200       # 风格描述最大长度
    MAX_CONTEXT_LENGTH = 3000    # 总上下文最大字符数
    
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
            style_content: 写作风格内容（可选）
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
        
        # === 衔接锚点（根据章节调整长度，增强版含摘要和事件）===
        if chapter_number == 1:
            context.continuation_point = None
            context.previous_chapter_summary = None
            context.previous_chapter_events = None
            logger.info("  ✅ 第1章无需衔接锚点")
        elif chapter_number <= 10:
            ending_info = await self._get_last_ending_enhanced(
                chapter, db, self.ENDING_LENGTH_SHORT
            )
            context.continuation_point = ending_info.get('ending_text')
            context.previous_chapter_summary = ending_info.get('summary')
            context.previous_chapter_events = ending_info.get('key_events')
            logger.info(f"  ✅ 衔接锚点（短）: {len(context.continuation_point or '')}字符")
        else:
            ending_info = await self._get_last_ending_enhanced(
                chapter, db, self.ENDING_LENGTH_NORMAL
            )
            context.continuation_point = ending_info.get('ending_text')
            context.previous_chapter_summary = ending_info.get('summary')
            context.previous_chapter_events = ending_info.get('key_events')
            logger.info(f"  ✅ 衔接锚点（标准）: {len(context.continuation_point or '')}字符")
        
        # === P1-重要信息 ===
        context.chapter_characters = await self._build_chapter_characters_1n(
            chapter, project, outline, db
        )
        context.emotional_tone = self._extract_emotional_tone(chapter, outline)
        
        # 写作风格（摘要化）
        if style_content:
            context.style_instruction = self._summarize_style(style_content)
        
        # === P2-参考信息（条件触发）===
        if chapter_number > 10 and self.memory_service:
            memory_limit = (
                self.MEMORY_COUNT_LIGHT if chapter_number <= 50
                else self.MEMORY_COUNT_FULL
            )
            context.relevant_memories = await self._get_relevant_memories(
                user_id, project.id, chapter_number, 
                context.chapter_outline,
                limit=memory_limit
            )
            logger.info(f"  ✅ 相关记忆: {len(context.relevant_memories or '')}字符")
        
        # 故事骨架（50章+）
        if chapter_number > self.SKELETON_THRESHOLD:
            context.story_skeleton = await self._build_story_skeleton(
                project.id, chapter_number, db
            )
            logger.info(f"  ✅ 故事骨架: {len(context.story_skeleton or '')}字符")
        
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
            "memories_length": len(context.relevant_memories or ""),
            "skeleton_length": len(context.story_skeleton or ""),
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
    ) -> str:
        """构建1-N模式的角色信息（从expansion_plan提取character_focus）"""
        # 获取所有角色
        characters_result = await db.execute(
            select(Character).where(Character.project_id == project.id)
        )
        characters = characters_result.scalars().all()
        
        if not characters:
            return "暂无角色信息"
        
        # 从expansion_plan中提取角色焦点
        filter_character_names = None
        if chapter.expansion_plan:
            try:
                plan = json.loads(chapter.expansion_plan)
                filter_character_names = plan.get('character_focus', [])
            except json.JSONDecodeError:
                pass
        
        # 筛选角色
        if filter_character_names:
            characters = [c for c in characters if c.name in filter_character_names]
        
        if not characters:
            return "暂无相关角色"
        
        # 构建精简的角色信息
        char_lines = []
        for c in characters[:10]:
            role_type = "主角" if c.role_type == "protagonist" else (
                "反派" if c.role_type == "antagonist" else "配角"
            )
            personality_brief = ""
            if c.personality:
                personality_brief = c.personality[:50]
                if len(c.personality) > 50:
                    personality_brief += "..."
            char_lines.append(f"- {c.name}({role_type}): {personality_brief}")
        
        return "\n".join(char_lines)
    
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
        
        # 查询上一章
        result = await db.execute(
            select(Chapter)
            .where(Chapter.project_id == chapter.project_id)
            .where(Chapter.chapter_number == chapter.chapter_number - 1)
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
        limit: int = 3
    ) -> Optional[str]:
        """获取与本章最相关的记忆"""
        if not self.memory_service:
            return None
        
        try:
            relevant = await self.memory_service.search_memories(
                user_id=user_id,
                project_id=project_id,
                query=chapter_outline,
                limit=limit,
                min_importance=self.MEMORY_IMPORTANCE_THRESHOLD
            )
            
            foreshadows = await self._get_due_foreshadows(
                user_id, project_id, chapter_number,
                lookahead=5
            )
            
            return self._format_memories(relevant, foreshadows, max_length=500)
            
        except Exception as e:
            logger.error(f"❌ 获取相关记忆失败: {str(e)}")
            return None
    
    async def _get_due_foreshadows(
        self,
        user_id: str,
        project_id: str,
        chapter_number: int,
        lookahead: int = 5
    ) -> List[Dict[str, Any]]:
        """获取即将需要回收的伏笔"""
        if not self.memory_service:
            return []
        
        try:
            foreshadows = await self.memory_service.find_unresolved_foreshadows(
                user_id, project_id, chapter_number
            )
            
            due_foreshadows = []
            for fs in foreshadows:
                meta = fs.get('metadata', {})
                fs_chapter = meta.get('chapter_number', 0)
                if chapter_number - fs_chapter >= lookahead:
                    due_foreshadows.append({
                        'chapter': fs_chapter,
                        'content': fs.get('content', '')[:60],
                        'importance': meta.get('importance', 0.5)
                    })
            
            return due_foreshadows[:2]
            
        except Exception as e:
            logger.error(f"❌ 获取待回收伏笔失败: {str(e)}")
            return []
    
    def _format_memories(
        self,
        relevant: List[Dict[str, Any]],
        foreshadows: List[Dict[str, Any]],
        max_length: int = 500
    ) -> str:
        """格式化记忆为简洁文本"""
        lines = []
        current_length = 0
        
        if foreshadows:
            lines.append("【待回收伏笔】")
            for fs in foreshadows[:2]:
                text = f"- 第{fs['chapter']}章埋下：{fs['content']}"
                if current_length + len(text) > max_length:
                    break
                lines.append(text)
                current_length += len(text)
        
        if relevant and current_length < max_length:
            lines.append("【相关记忆】")
            for mem in relevant:
                content = mem.get('content', '')[:80]
                text = f"- {content}"
                if current_length + len(text) > max_length:
                    break
                lines.append(text)
                current_length += len(text)
        
        return "\n".join(lines) if lines else None
    
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
        # 1. 获取上一章内容的最后500字
        if chapter_number > 1:
            prev_chapter_result = await db.execute(
                select(Chapter)
                .where(Chapter.project_id == chapter.project_id)
                .where(Chapter.chapter_number == chapter_number - 1)
            )
            prev_chapter = prev_chapter_result.scalar_one_or_none()
            
            if prev_chapter and prev_chapter.content:
                content = prev_chapter.content.strip()
                if len(content) <= 500:
                    context.continuation_point = content
                else:
                    context.continuation_point = content[-500:]
                logger.info(f"  ✅ P1-上一章内容(最后500字): {len(context.continuation_point)}字符")
            else:
                context.continuation_point = None
                logger.info(f"  ⚠️ P1-上一章内容: 无")
        else:
            context.continuation_point = None
            logger.info(f"  ✅ P1-第1章无需上一章内容")
        
        # 2. 根据structure中的characters获取角色信息（含职业）
        character_names = []
        if outline and outline.structure:
            try:
                structure = json.loads(outline.structure)
                character_names = structure.get('characters', [])
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
        
        # 2. 根据角色名检索相关记忆（相关度>0.6）
        if character_names and self.memory_service:
            try:
                query_text = " ".join(character_names)
                
                relevant_memories = await self.memory_service.search_memories(
                    user_id=user_id,
                    project_id=project.id,
                    query=query_text,
                    limit=10,
                    min_importance=0.0
                )
                
                filtered_memories = [
                    mem for mem in relevant_memories
                    if mem.get('similarity', 0) > 0.6
                ]
                
                if filtered_memories:
                    memory_lines = ["【相关记忆】"]
                    for mem in filtered_memories:
                        similarity = mem.get('similarity', 0)
                        content = mem.get('content', '')[:100]
                        memory_lines.append(f"- (相关度:{similarity:.2f}) {content}")
                    
                    context.relevant_memories = "\n".join(memory_lines)
                    logger.info(f"  ✅ P2-相关记忆: {len(filtered_memories)}条 (相关度>0.6)")
                else:
                    context.relevant_memories = None
                    logger.info(f"  ⚠️ P2-相关记忆: 无符合条件的记忆")
                    
            except Exception as e:
                logger.error(f"  ❌ 检索相关记忆失败: {str(e)}")
                context.relevant_memories = None
        else:
            context.relevant_memories = None
            logger.info(f"  ⚠️ P2-相关记忆: 无角色或记忆服务不可用")
        
        # === 统计信息 ===
        context.context_stats = {
            "mode": "one-to-one",
            "chapter_number": chapter_number,
            "has_previous_content": context.continuation_point is not None,
            "previous_content_length": len(context.continuation_point or ""),
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
            
            # === 组织特有信息 ===
            if c.is_organization:
                if c.organization_type:
                    info_lines.append(f"  组织类型: {c.organization_type}")
                if c.organization_purpose:
                    info_lines.append(f"  组织目的: {c.organization_purpose[:100]}")
                if c.organization_members:
                    info_lines.append(f"  组织成员: {c.organization_members[:100]}")
            
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

