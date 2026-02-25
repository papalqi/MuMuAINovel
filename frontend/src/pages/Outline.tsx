import { useState, useEffect } from 'react';
import { Button, List, Modal, Form, Input, message, Empty, Space, Popconfirm, Card, Select, Radio, Tag, InputNumber, Tabs } from 'antd';
import { EditOutlined, DeleteOutlined, ThunderboltOutlined, BranchesOutlined, AppstoreAddOutlined, CheckCircleOutlined, ExclamationCircleOutlined, PlusOutlined, FileTextOutlined } from '@ant-design/icons';
import { useStore } from '../store';
import { useOutlineSync } from '../store/hooks';
import { useTaskCenterStore } from '../store/taskCenter';
import { SSEPostClient } from '../utils/sseClient';
import { SSEProgressModal } from '../components/SSEProgressModal';
import { outlineApi, chapterApi, projectApi, characterApi } from '../services/api';
import type { OutlineExpansionResponse, BatchOutlineExpansionResponse, ChapterPlanItem, ApiError, Character } from '../types';

// 大纲生成请求数据类型
interface OutlineGenerateRequestData {
  project_id: string;
  genre: string;
  theme: string;
  chapter_count: number;
  narrative_perspective: string;
  target_words: number;
  requirements?: string;
  mode: 'auto' | 'new' | 'continue';
  story_direction?: string;
  plot_stage: 'development' | 'climax' | 'ending';
  model?: string;
  provider?: string;
}

// 跳过的大纲信息类型
interface SkippedOutlineInfo {
  outline_id: string;
  outline_title: string;
  reason: string;
}

// 场景类型
interface SceneInfo {
  location: string;
  characters: string[];
  purpose: string;
}

// 角色/组织条目类型（新格式）
interface CharacterEntry {
  name: string;
  type: 'character' | 'organization';
}

interface PostcheckItemStatus {
  status: 'success' | 'warning';
  success: boolean;
  created_count: number;
  created_names: string[];
  error?: string | null;
}

interface OutlinePostcheckSummary {
  scope: 'all' | 'characters' | 'organizations';
  overall_status: 'success' | 'partial_failed' | 'failed';
  has_warning: boolean;
  total_created: number;
  characters?: PostcheckItemStatus | null;
  organizations?: PostcheckItemStatus | null;
}

/**
 * 解析 characters 字段，兼容新旧格式
 * 旧格式: string[] -> 全部当作 character
 * 新格式: {name: string, type: "character"|"organization"}[]
 */
function parseCharacterEntries(characters: unknown): CharacterEntry[] {
  if (!Array.isArray(characters) || characters.length === 0) return [];
  
  return characters.map((entry) => {
    if (typeof entry === 'string') {
      // 旧格式：纯字符串，默认为 character
      return { name: entry, type: 'character' as const };
    }
    if (typeof entry === 'object' && entry !== null && 'name' in entry) {
      // 新格式：带类型标识的对象
      return {
        name: (entry as { name: string }).name,
        type: ((entry as { type?: string }).type === 'organization' ? 'organization' : 'character') as 'character' | 'organization'
      };
    }
    return null;
  }).filter((e): e is CharacterEntry => e !== null);
}

/** 从 entries 中提取角色名称列表 */
function getCharacterNames(entries: CharacterEntry[]): string[] {
  return entries.filter(e => e.type === 'character').map(e => e.name);
}

/** 从 entries 中提取组织名称列表 */
function getOrganizationNames(entries: CharacterEntry[]): string[] {
  return entries.filter(e => e.type === 'organization').map(e => e.name);
}

const { TextArea } = Input;

export default function Outline() {
  const { currentProject, outlines, setCurrentProject } = useStore();
  const [isGenerating, setIsGenerating] = useState(false);
  const [editForm] = Form.useForm();
  const [generateForm] = Form.useForm();
  const [expansionForm] = Form.useForm();
  const [modalApi, contextHolder] = Modal.useModal();
  const [batchExpansionForm] = Form.useForm();
  const [manualCreateForm] = Form.useForm();
  const [isMobile, setIsMobile] = useState(window.innerWidth <= 768);
  const [isExpanding, setIsExpanding] = useState(false);
  const [projectCharacters, setProjectCharacters] = useState<Array<{ label: string; value: string }>>([]);

  // ✅ 新增：记录每个大纲的展开状态
  const [outlineExpandStatus, setOutlineExpandStatus] = useState<Record<string, boolean>>({});
  
  // ✅ 新增：记录场景区域的展开/折叠状态
  const [scenesExpandStatus, setScenesExpandStatus] = useState<Record<string, boolean>>({});

  // 缓存批量展开的规划数据，避免重复AI调用
  const [cachedBatchExpansionResponse, setCachedBatchExpansionResponse] = useState<BatchOutlineExpansionResponse | null>(null);

  // 批量展开预览的状态
  const [batchPreviewVisible, setBatchPreviewVisible] = useState(false);
  const [batchPreviewData, setBatchPreviewData] = useState<BatchOutlineExpansionResponse | null>(null);
  const [selectedOutlineIdx, setSelectedOutlineIdx] = useState(0);
  const [selectedChapterIdx, setSelectedChapterIdx] = useState(0);

  // SSE进度状态
  const [sseProgress, setSSEProgress] = useState(0);
  const [sseMessage, setSSEMessage] = useState('');
  const [sseModalVisible, setSSEModalVisible] = useState(false);
  const [isPostchecking, setIsPostchecking] = useState(false);
  const [lastPostcheckSummary, setLastPostcheckSummary] = useState<OutlinePostcheckSummary | null>(null);

  useEffect(() => {
    const handleResize = () => {
      setIsMobile(window.innerWidth <= 768);
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // 使用同步 hooks
  const {
    refreshOutlines,
    updateOutline,
    deleteOutline
  } = useOutlineSync();
  const createTask = useTaskCenterStore((state) => state.createTask);
  const setTaskProgress = useTaskCenterStore((state) => state.setTaskProgress);
  const markTaskSuccess = useTaskCenterStore((state) => state.markTaskSuccess);
  const markTaskFailed = useTaskCenterStore((state) => state.markTaskFailed);

  // 初始加载大纲列表和角色列表
  useEffect(() => {
    if (currentProject?.id) {
      refreshOutlines();
      // 加载项目角色列表
      loadProjectCharacters();
      setLastPostcheckSummary(null);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentProject?.id]); // 只依赖 ID，不依赖函数

  // 加载项目角色列表
  const loadProjectCharacters = async () => {
    if (!currentProject?.id) return;
    try {
      const characters = await characterApi.getCharacters(currentProject.id);
      setProjectCharacters(
        characters.map((char: Character) => ({
          label: char.name,
          value: char.name
        }))
      );
    } catch (error) {
      console.error('加载角色列表失败:', error);
    }
  };

  // ✅ 新增：加载所有大纲的展开状态
  useEffect(() => {
    const loadExpandStatus = async () => {
      if (outlines.length === 0) return;

      const statusMap: Record<string, boolean> = {};
      for (const outline of outlines) {
        try {
          const chapters = await outlineApi.getOutlineChapters(outline.id);
          statusMap[outline.id] = chapters.has_chapters;
        } catch (error) {
          console.error(`加载大纲 ${outline.id} 状态失败:`, error);
          statusMap[outline.id] = false;
        }
      }
      setOutlineExpandStatus(statusMap);
    };

    loadExpandStatus();
  }, [outlines]);

  // 当角色确认数据变化时，初始化选中状态（默认全选）
  // 当组织确认数据变化时，初始化选中状态（默认全选）
  // 移除事件监听，避免无限循环
  // Hook 内部已经更新了 store，不需要再次刷新

  if (!currentProject) return null;

  // 确保大纲按 order_index 排序
  const sortedOutlines = [...outlines].sort((a, b) => a.order_index - b.order_index);

  const handleOpenEditModal = (id: string) => {
    const outline = outlines.find(o => o.id === id);
    if (outline) {
      // 解析structure数据
      let structureData: {
        characters?: unknown[];  // 兼容新旧格式
        scenes?: string[] | Array<{
          location: string;
          characters: string[];
          purpose: string;
        }>;
        key_points?: string[];
        emotion?: string;
        goal?: string;
      } = {};
      
      if (outline.structure) {
        try {
          structureData = JSON.parse(outline.structure);
        } catch (e) {
          console.error('解析structure失败:', e);
        }
      }
      
      // 解析角色/组织条目（兼容新旧格式）
      const editEntries = parseCharacterEntries(structureData.characters);
      const editCharNames = getCharacterNames(editEntries);
      const editOrgNames = getOrganizationNames(editEntries);
      
      // 处理场景数据 - 可能是字符串数组或对象数组
      let scenesText = '';
      if (structureData.scenes) {
        if (typeof structureData.scenes[0] === 'string') {
          // 字符串数组格式
          scenesText = (structureData.scenes as string[]).join('\n');
        } else {
          // 对象数组格式
          scenesText = (structureData.scenes as Array<{location: string; characters: string[]; purpose: string}>)
            .map(s => `${s.location}|${(s.characters || []).join('、')}|${s.purpose}`)
            .join('\n');
        }
      }
      
      // 处理情节要点数据
      const keyPointsText = structureData.key_points ? structureData.key_points.join('\n') : '';
      
      // 设置表单初始值
      editForm.setFieldsValue({
        title: outline.title,
        content: outline.content,
        characters: editCharNames,
        organizations: editOrgNames,
        scenes: scenesText,
        key_points: keyPointsText,
        emotion: structureData.emotion || '',
        goal: structureData.goal || ''
      });
      
      modalApi.confirm({
        title: '编辑大纲',
        width: 800,
        centered: true,
        styles: {
          body: {
            maxHeight: 'calc(100vh - 200px)',
            overflowY: 'auto'
          }
        },
        content: (
          <Form
            form={editForm}
            layout="vertical"
            style={{ marginTop: 12 }}
          >
            <Form.Item
              label="标题"
              name="title"
              rules={[{ required: true, message: '请输入标题' }]}
              style={{ marginBottom: 12 }}
            >
              <Input placeholder="输入大纲标题" />
            </Form.Item>

            <Form.Item
              label="内容"
              name="content"
              rules={[{ required: true, message: '请输入内容' }]}
              style={{ marginBottom: 12 }}
            >
              <TextArea rows={4} placeholder="输入大纲内容..." />
            </Form.Item>
            
            <Form.Item
              label="涉及角色"
              name="characters"
              tooltip="从项目角色中选择，也可以手动输入新角色名"
              style={{ marginBottom: 12 }}
            >
              <Select
                mode="tags"
                style={{ width: '100%' }}
                placeholder="选择或输入角色名"
                options={projectCharacters}
                tokenSeparators={[',', '，']}
                maxTagCount="responsive"
              />
            </Form.Item>
            
            <Form.Item
              label="涉及组织"
              name="organizations"
              tooltip="从项目组织中选择，也可以手动输入新组织名"
              style={{ marginBottom: 12 }}
            >
              <Select
                mode="tags"
                style={{ width: '100%' }}
                placeholder="选择或输入组织/势力名"
                tokenSeparators={[',', '，']}
                maxTagCount="responsive"
              />
            </Form.Item>
            
            <Form.Item
              label="场景信息"
              name="scenes"
              tooltip="支持两种格式：简单描述（每行一个场景）或详细格式（地点|角色|目的）"
              style={{ marginBottom: 12 }}
            >
              <TextArea
                rows={3}
                placeholder="每行一个场景&#10;详细格式：地点|角色1、角色2|目的"
              />
            </Form.Item>
            
            <Form.Item
              label="情节要点"
              name="key_points"
              tooltip="每行一个情节要点"
              style={{ marginBottom: 12 }}
            >
              <TextArea
                rows={2}
                placeholder="每行一个情节要点"
              />
            </Form.Item>
            
            <Form.Item
              label="情感基调"
              name="emotion"
              tooltip="描述本章的情感氛围"
              style={{ marginBottom: 12 }}
            >
              <Input placeholder="例如：冷冽与躁动并存" />
            </Form.Item>
            
            <Form.Item
              label="叙事目标"
              name="goal"
              tooltip="本章要达成的叙事目的"
              style={{ marginBottom: 0 }}
            >
              <Input placeholder="例如：建立世界观对比并完成主角初遇" />
            </Form.Item>
          </Form>
        ),
        okText: '更新',
        cancelText: '取消',
        onOk: async () => {
          const values = await editForm.validateFields();
          try {
            // 解析并重构structure数据
            const originalStructure = outline.structure ? JSON.parse(outline.structure) : {};
            
            // 处理角色和组织数据 - 合并为带类型标识的新格式
            const charNames = Array.isArray(values.characters)
              ? values.characters.filter((c: string) => c && c.trim())
              : [];
            const orgNames = Array.isArray(values.organizations)
              ? values.organizations.filter((c: string) => c && c.trim())
              : [];
            const characters: CharacterEntry[] = [
              ...charNames.map((name: string) => ({ name: name.trim(), type: 'character' as const })),
              ...orgNames.map((name: string) => ({ name: name.trim(), type: 'organization' as const }))
            ];
            
            // 处理场景数据 - 检测原始格式
            let scenes: string[] | Array<{location: string; characters: string[]; purpose: string}> | undefined;
            if (values.scenes) {
              const lines = values.scenes.split('\n')
                .map((line: string) => line.trim())
                .filter((line: string) => line);
              
              // 检查是否包含管道符，判断格式
              const hasStructuredFormat = lines.some((line: string) => line.includes('|'));
              
              if (hasStructuredFormat) {
                // 尝试解析为对象数组格式
                scenes = lines
                  .map((line: string) => {
                    const parts = line.split('|');
                    if (parts.length >= 3) {
                      return {
                        location: parts[0].trim(),
                        characters: parts[1].split('、').map(c => c.trim()).filter(c => c),
                        purpose: parts[2].trim()
                      };
                    }
                    return null;
                  })
                  .filter((s: { location: string; characters: string[]; purpose: string } | null): s is { location: string; characters: string[]; purpose: string } => s !== null);
              } else {
                // 保持字符串数组格式
                scenes = lines;
              }
            }
            
            // 处理情节要点数据
            const keyPoints = values.key_points
              ? values.key_points.split('\n')
                  .map((line: string) => line.trim())
                  .filter((line: string) => line)
              : undefined;
            
            // 合并structure数据，只包含AI实际生成的字段
            const newStructure = {
              ...originalStructure,
              title: values.title,
              summary: values.content,
              characters: characters.length > 0 ? characters : undefined,
              scenes: scenes && scenes.length > 0 ? scenes : undefined,
              key_points: keyPoints && keyPoints.length > 0 ? keyPoints : undefined,
              emotion: values.emotion || undefined,
              goal: values.goal || undefined
            };
            
            // 更新大纲
            await updateOutline(id, {
              title: values.title,
              content: values.content,
              structure: JSON.stringify(newStructure, null, 2)
            });
            
            message.success('大纲更新成功');
          } catch (error) {
            console.error('更新失败:', error);
            message.error('更新失败');
          }
        },
      });
    }
  };

  const handleDeleteOutline = async (id: string) => {
    try {
      await deleteOutline(id);
      message.success('删除成功');
      // 删除后刷新大纲列表和项目信息，更新字数显示
      await refreshOutlines();
      if (currentProject?.id) {
        const updatedProject = await projectApi.getProject(currentProject.id);
        setCurrentProject(updatedProject);
      }
    } catch {
      message.error('删除失败');
    }
  };

  interface GenerateFormValues {
    theme?: string;
    chapter_count?: number;
    narrative_perspective?: string;
    requirements?: string;
    provider?: string;
    model?: string;
    mode?: 'auto' | 'new' | 'continue';
    story_direction?: string;
    plot_stage?: 'development' | 'climax' | 'ending';
    keep_existing?: boolean;
  }

  const handleGenerate = async (values: GenerateFormValues, existingTaskId?: string) => {
    let taskId = existingTaskId;
    try {
      setIsGenerating(true);

      // 添加详细的调试日志
      console.log('=== 大纲生成调试信息 ===');
      console.log('1. Form values 原始数据:', values);
      console.log('2. values.model:', values.model);
      console.log('3. values.provider:', values.provider);

      // 关闭生成表单Modal
      Modal.destroyAll();

      // 显示进度Modal
      setSSEProgress(0);
      setSSEMessage('正在连接AI服务...');
      setSSEModalVisible(true);

      // 准备请求数据
      const requestData: OutlineGenerateRequestData = {
        project_id: currentProject.id,
        genre: currentProject.genre || '通用',
        theme: values.theme || currentProject.theme || '',
        chapter_count: values.chapter_count || 5,
        narrative_perspective: values.narrative_perspective || currentProject.narrative_perspective || '第三人称',
        target_words: currentProject.target_words || 100000,
        requirements: values.requirements,
        mode: values.mode || 'auto',
        story_direction: values.story_direction,
        plot_stage: values.plot_stage || 'development'
      };

      // 只有在用户选择了模型时才添加model参数
      if (values.model) {
        requestData.model = values.model;
        console.log('4. 添加model到请求:', values.model);
      } else {
        console.log('4. values.model为空，不添加到请求');
      }

      // 添加provider参数（如果有）
      if (values.provider) {
        requestData.provider = values.provider;
        console.log('5. 添加provider到请求:', values.provider);
      }

      if (!taskId) {
        taskId = createTask({
          type: 'outline_generate',
          name: `大纲生成/续写（${values.mode || 'auto'}）`,
          message: '任务初始化中...',
          maxRetries: 5,
          retryAction: () => handleGenerate({ ...values }, taskId),
        });
      }

      setTaskProgress(taskId, 0, '正在连接AI服务...');

      console.log('6. 最终请求数据:', JSON.stringify(requestData, null, 2));
      console.log('=========================');

      // 使用SSE客户端
      const apiUrl = `/api/outlines/generate-stream`;
      const client = new SSEPostClient(apiUrl, requestData, {
        onProgress: (msg: string, progress: number) => {
          setSSEMessage(msg);
          setSSEProgress(progress);
          if (taskId) {
            setTaskProgress(taskId, progress, msg);
          }
        },
        onResult: (data: unknown) => {
          console.log('生成完成，结果:', data);
        },
        onError: (error: string) => {
          // 现在只处理真正的错误
          message.error(`生成失败: ${error}`);
          setSSEModalVisible(false);
          setIsGenerating(false);
          if (taskId) {
            markTaskFailed(taskId, error);
          }
        },
        onComplete: () => {
          message.success('大纲生成完成！');
          setSSEModalVisible(false);
          setIsGenerating(false);
          // 刷新大纲列表
          refreshOutlines();
          if (taskId) {
            markTaskSuccess(taskId, '大纲生成完成');
          }
        }
      });

      // 开始连接
      client.connect();

    } catch (error) {
      console.error('AI生成失败:', error);
      message.error('AI生成失败');
      setSSEModalVisible(false);
      setIsGenerating(false);
      if (taskId) {
        markTaskFailed(
          taskId,
          error instanceof Error ? error.message : 'AI生成失败'
        );
      }
    }
  };

  const handleRunOutlinePostcheck = async (
    scope: 'all' | 'characters' | 'organizations' = 'all',
    existingTaskId?: string
  ) => {
    let taskId = existingTaskId;
    try {
      if (!currentProject?.id) {
        message.warning('请先选择项目');
        return;
      }
      if (outlines.length === 0) {
        message.warning('当前没有大纲可供补全');
        return;
      }

      setIsPostchecking(true);
      setSSEProgress(0);
      setSSEMessage('正在准备后处理补全...');
      setSSEModalVisible(true);

      const scopeLabelMap = {
        all: '人物+组织/地点',
        characters: '人物',
        organizations: '组织/地点'
      } as const;

      if (!taskId) {
        taskId = createTask({
          type: 'outline_postcheck',
          name: `大纲后处理补全（${scopeLabelMap[scope]}）`,
          message: '任务初始化中...',
          maxRetries: 5,
          retryAction: () => handleRunOutlinePostcheck(scope, taskId),
        });
      }

      setTaskProgress(taskId, 0, '正在准备后处理补全...');

      let resultPayload: {
        postcheck?: OutlinePostcheckSummary;
      } | null = null;

      const client = new SSEPostClient('/api/outlines/postcheck-stream', {
        project_id: currentProject.id,
        scope,
      }, {
        onProgress: (msg: string, progress: number) => {
          setSSEMessage(msg);
          setSSEProgress(progress);
          if (taskId) {
            setTaskProgress(taskId, progress, msg);
          }
        },
        onResult: (data: unknown) => {
          const payload = (data || {}) as { postcheck?: OutlinePostcheckSummary };
          resultPayload = payload;
          if (payload.postcheck) {
            setLastPostcheckSummary(payload.postcheck);
          }
        },
        onError: (error: string) => {
          message.error(`后处理补全失败: ${error}`);
          setSSEModalVisible(false);
          setIsPostchecking(false);
          if (taskId) {
            markTaskFailed(taskId, error);
          }
        },
        onComplete: () => {
          setSSEModalVisible(false);
          setIsPostchecking(false);
          refreshOutlines();
          loadProjectCharacters();

          const overall = resultPayload?.postcheck?.overall_status;
          if (overall === 'success') {
            message.success('后处理补全完成');
            if (taskId) {
              markTaskSuccess(taskId, '后处理补全完成');
            }
          } else if (overall === 'partial_failed') {
            message.warning('后处理补全完成（部分失败）');
            if (taskId) {
              markTaskSuccess(taskId, '后处理补全完成（部分失败）');
            }
          } else if (overall === 'failed') {
            message.warning('后处理补全完成（失败，请查看状态详情）');
            if (taskId) {
              markTaskSuccess(taskId, '后处理补全结束（失败）');
            }
          } else {
            message.success('后处理补全完成');
            if (taskId) {
              markTaskSuccess(taskId, '后处理补全完成');
            }
          }
        }
      });

      client.connect();
    } catch (error) {
      console.error('后处理补全失败:', error);
      message.error('后处理补全失败');
      setSSEModalVisible(false);
      setIsPostchecking(false);
      if (taskId) {
        markTaskFailed(
          taskId,
          error instanceof Error ? error.message : '后处理补全失败'
        );
      }
    }
  };

  const showGenerateModal = async () => {
    const hasOutlines = outlines.length > 0;
    const initialMode = hasOutlines ? 'continue' : 'new';

    // 直接加载可用模型列表
    const settingsResponse = await fetch('/api/settings');
    const settings = await settingsResponse.json();
    const { api_key, api_base_url, api_provider } = settings;

    let loadedModels: Array<{ value: string, label: string }> = [];
    let defaultModel: string | undefined = undefined;

    if (api_key && api_base_url) {
      try {
        const modelsResponse = await fetch(
          `/api/settings/models?api_key=${encodeURIComponent(api_key)}&api_base_url=${encodeURIComponent(api_base_url)}&provider=${api_provider}`
        );
        if (modelsResponse.ok) {
          const data = await modelsResponse.json();
          if (data.models && data.models.length > 0) {
            loadedModels = data.models;
            defaultModel = settings.llm_model;
          }
        }
      } catch {
        console.log('获取模型列表失败，将使用默认模型');
      }
    }

    modalApi.confirm({
      title: hasOutlines ? (
        <Space>
          <span>AI生成/续写大纲</span>
          <Tag color="blue">当前已有 {outlines.length} 卷</Tag>
        </Space>
      ) : 'AI生成大纲',
      width: 700,
      centered: true,
      content: (
        <Form
          form={generateForm}
          layout="vertical"
          style={{ marginTop: 16 }}
          initialValues={{
            mode: initialMode,
            chapter_count: 5,
            narrative_perspective: currentProject.narrative_perspective || '第三人称',
            plot_stage: 'development',
            keep_existing: true,
            theme: currentProject.theme || '',
            model: defaultModel,
          }}
        >
          {hasOutlines && (
            <Form.Item
              label="生成模式"
              name="mode"
              tooltip="自动判断：根据是否有大纲自动选择；全新生成：删除旧大纲重新生成；续写模式：基于已有大纲继续创作"
            >
              <Radio.Group buttonStyle="solid">
                <Radio.Button value="auto">自动判断</Radio.Button>
                <Radio.Button value="new">全新生成</Radio.Button>
                <Radio.Button value="continue">续写模式</Radio.Button>
              </Radio.Group>
            </Form.Item>
          )}

          <Form.Item
            noStyle
            shouldUpdate={(prevValues, currentValues) => prevValues.mode !== currentValues.mode}
          >
            {({ getFieldValue }) => {
              const mode = getFieldValue('mode');
              const isContinue = mode === 'continue' || (mode === 'auto' && hasOutlines);

              // 续写模式不显示主题输入，使用项目原有主题
              if (isContinue) {
                return null;
              }

              // 全新生成模式需要输入主题
              return (
                <Form.Item
                  label="故事主题"
                  name="theme"
                  rules={[{ required: true, message: '请输入故事主题' }]}
                >
                  <TextArea rows={3} placeholder="描述你的故事主题、核心设定和主要情节..." />
                </Form.Item>
              );
            }}
          </Form.Item>

          <Form.Item
            noStyle
            shouldUpdate={(prevValues, currentValues) => prevValues.mode !== currentValues.mode}
          >
            {({ getFieldValue }) => {
              const mode = getFieldValue('mode');
              const isContinue = mode === 'continue' || (mode === 'auto' && hasOutlines);

              return (
                <>
                  {isContinue && (
                    <>
                      <Form.Item
                        label="故事发展方向"
                        name="story_direction"
                        tooltip="告诉AI你希望故事接下来如何发展"
                      >
                        <TextArea
                          rows={3}
                          placeholder="例如：主角遇到新的挑战、引入新角色、揭示关键秘密等..."
                        />
                      </Form.Item>

                      <Form.Item
                        label="情节阶段"
                        name="plot_stage"
                        tooltip="帮助AI理解当前故事所处的阶段"
                      >
                        <Select>
                          <Select.Option value="development">发展阶段 - 继续展开情节</Select.Option>
                          <Select.Option value="climax">高潮阶段 - 矛盾激化</Select.Option>
                          <Select.Option value="ending">结局阶段 - 收束伏笔</Select.Option>
                        </Select>
                      </Form.Item>
                    </>
                  )}

                  <Form.Item
                    label={isContinue ? "续写章节数" : "章节数量"}
                    name="chapter_count"
                    rules={[{ required: true, message: '请输入章节数量' }]}
                  >
                    <Input
                      type="number"
                      min={1}
                      max={50}
                      placeholder={isContinue ? "建议5-10章" : "如：30"}
                    />
                  </Form.Item>

                  <Form.Item
                    label="叙事视角"
                    name="narrative_perspective"
                    rules={[{ required: true, message: '请选择叙事视角' }]}
                  >
                    <Select>
                      <Select.Option value="第一人称">第一人称</Select.Option>
                      <Select.Option value="第三人称">第三人称</Select.Option>
                      <Select.Option value="全知视角">全知视角</Select.Option>
                    </Select>
                  </Form.Item>

                  <Form.Item label="其他要求" name="requirements">
                    <TextArea rows={2} placeholder="其他特殊要求（可选）" />
                  </Form.Item>

                </>
              );
            }}
          </Form.Item>

          {/* 自定义模型选择 - 移到外层，所有模式都显示 */}
          {loadedModels.length > 0 && (
            <Form.Item
              label="AI模型"
              name="model"
              tooltip="选择用于生成的AI模型，不选则使用系统默认模型"
            >
              <Select
                placeholder={defaultModel ? `默认: ${loadedModels.find(m => m.value === defaultModel)?.label || defaultModel}` : "使用默认模型"}
                allowClear
                showSearch
                optionFilterProp="label"
                options={loadedModels}
                onChange={(value) => {
                  console.log('用户在下拉框中选择了模型:', value);
                  // 手动同步到Form
                  generateForm.setFieldsValue({ model: value });
                  console.log('已同步到Form，当前Form值:', generateForm.getFieldsValue());
                }}
              />
              <div style={{ color: 'var(--color-text-tertiary)', fontSize: 12, marginTop: 4 }}>
                {defaultModel ? `当前默认模型: ${loadedModels.find(m => m.value === defaultModel)?.label || defaultModel}` : '未配置默认模型'}
              </div>
            </Form.Item>
          )}
        </Form>
      ),
      okText: hasOutlines ? '开始续写' : '开始生成',
      cancelText: '取消',
      onOk: async () => {
        const values = await generateForm.validateFields();
        await handleGenerate(values);
      },
    });
  };

  // 手动创建大纲
  const showManualCreateOutlineModal = () => {
    const nextOrderIndex = outlines.length > 0
      ? Math.max(...outlines.map(o => o.order_index)) + 1
      : 1;

    modalApi.confirm({
      title: '手动创建大纲',
      width: 600,
      centered: true,
      content: (
        <Form
          form={manualCreateForm}
          layout="vertical"
          initialValues={{ order_index: nextOrderIndex }}
          style={{ marginTop: 16 }}
        >
          <Form.Item
            label="大纲序号"
            name="order_index"
            rules={[{ required: true, message: '请输入序号' }]}
            tooltip={currentProject?.outline_mode === 'one-to-one' ? '在传统模式下，序号即章节编号' : '在细化模式下，序号为卷数'}
          >
            <InputNumber min={1} style={{ width: '100%' }} placeholder="自动计算的下一个序号" />
          </Form.Item>

          <Form.Item
            label="大纲标题"
            name="title"
            rules={[{ required: true, message: '请输入标题' }]}
          >
            <Input placeholder={currentProject?.outline_mode === 'one-to-one' ? '例如：第一章 初入江湖' : '例如：第一卷 初入江湖'} />
          </Form.Item>

          <Form.Item
            label="大纲内容"
            name="content"
            rules={[{ required: true, message: '请输入内容' }]}
          >
            <TextArea
              rows={6}
              placeholder="描述本章/卷的主要情节和发展方向..."
            />
          </Form.Item>
        </Form>
      ),
      okText: '创建',
      cancelText: '取消',
      onOk: async () => {
        const values = await manualCreateForm.validateFields();

        // 校验序号是否重复
        const existingOutline = outlines.find(o => o.order_index === values.order_index);
        if (existingOutline) {
          modalApi.warning({
            title: '序号冲突',
            content: (
              <div>
                <p>序号 <strong>{values.order_index}</strong> 已被使用：</p>
                <div style={{
                  padding: 12,
                  background: 'var(--color-warning-bg)',
                  borderRadius: 4,
                  border: '1px solid var(--color-warning-border)',
                  marginTop: 8
                }}>
                  <div style={{ fontWeight: 500, color: 'var(--color-warning)' }}>
                    {currentProject?.outline_mode === 'one-to-one'
                      ? `第${existingOutline.order_index}章`
                      : `第${existingOutline.order_index}卷`
                    }：{existingOutline.title}
                  </div>
                </div>
                <p style={{ marginTop: 12, color: 'var(--color-text-secondary)' }}>
                  💡 建议使用序号 <strong>{nextOrderIndex}</strong>，或选择其他未使用的序号
                </p>
              </div>
            ),
            okText: '我知道了',
            centered: true
          });
          throw new Error('序号重复');
        }

        try {
          await outlineApi.createOutline({
            project_id: currentProject.id,
            ...values
          });
          message.success('大纲创建成功');
          await refreshOutlines();
          manualCreateForm.resetFields();
        } catch (error: unknown) {
          const err = error as Error;
          if (err.message === '序号重复') {
            // 序号重复错误已经显示了Modal，不需要再显示message
            throw error;
          }
          message.error('创建失败：' + (err.message || '未知错误'));
          throw error;
        }
      }
    });
  };

  // 展开单个大纲为多章 - 使用SSE显示进度
  const handleExpandOutline = async (outlineId: string, outlineTitle: string) => {
    try {
      setIsExpanding(true);

      // ✅ 新增：检查是否需要按顺序展开
      const currentOutline = sortedOutlines.find(o => o.id === outlineId);
      if (currentOutline) {
        // 获取所有在当前大纲之前的大纲
        const previousOutlines = sortedOutlines.filter(
          o => o.order_index < currentOutline.order_index
        );

        // 检查前面的大纲是否都已展开
        for (const prevOutline of previousOutlines) {
          try {
            const prevChapters = await outlineApi.getOutlineChapters(prevOutline.id);
            if (!prevChapters.has_chapters) {
              // 如果前面有未展开的大纲，显示提示并阻止操作
              setIsExpanding(false);
              modalApi.warning({
                title: '请按顺序展开大纲',
                width: 600,
                centered: true,
                content: (
                  <div>
                    <p style={{ marginBottom: 12 }}>
                      为了保持章节编号的连续性和内容的连贯性，请先展开前面的大纲。
                    </p>
                    <div style={{
                      padding: 12,
                      background: 'var(--color-warning-bg)',
                      borderRadius: 4,
                      border: '1px solid var(--color-warning-border)'
                    }}>
                      <div style={{ fontWeight: 500, marginBottom: 8, color: 'var(--color-warning)' }}>
                        ⚠️ 需要先展开：
                      </div>
                      <div style={{ color: 'var(--color-text-secondary)' }}>
                        第{prevOutline.order_index}卷：《{prevOutline.title}》
                      </div>
                    </div>
                    <p style={{ marginTop: 12, color: 'var(--color-text-secondary)', fontSize: 13 }}>
                      💡 提示：您也可以使用「批量展开」功能，系统会自动按顺序处理所有大纲。
                    </p>
                  </div>
                ),
                okText: '我知道了'
              });
              return;
            }
          } catch (error) {
            console.error(`检查大纲 ${prevOutline.id} 失败:`, error);
            // 如果检查失败，继续处理（避免因网络问题阻塞）
          }
        }
      }

      // 第一步：检查是否已有展开的章节
      const existingChapters = await outlineApi.getOutlineChapters(outlineId);

      if (existingChapters.has_chapters && existingChapters.expansion_plans && existingChapters.expansion_plans.length > 0) {
        // 如果已有章节，显示已有的展开规划信息
        setIsExpanding(false);
        showExistingExpansionPreview(outlineTitle, existingChapters);
        return;
      }

      // 如果没有章节，显示展开表单
      setIsExpanding(false);
      modalApi.confirm({
        title: (
          <Space>
            <BranchesOutlined />
            <span>展开大纲为多章</span>
          </Space>
        ),
        width: 600,
        centered: true,
        content: (
          <div>
            <div style={{ marginBottom: 16, padding: 12, background: 'var(--color-bg-layout)', borderRadius: 4 }}>
              <div style={{ fontWeight: 500, marginBottom: 4 }}>大纲标题</div>
              <div style={{ color: 'var(--color-text-secondary)' }}>{outlineTitle}</div>
            </div>
            <Form
              form={expansionForm}
              layout="vertical"
              initialValues={{
                target_chapter_count: 3,
                expansion_strategy: 'balanced',
              }}
            >
              <Form.Item
                label="目标章节数"
                name="target_chapter_count"
                rules={[{ required: true, message: '请输入目标章节数' }]}
                tooltip="将这个大纲展开为几章内容"
              >
                <InputNumber
                  min={2}
                  max={10}
                  style={{ width: '100%' }}
                  placeholder="建议2-5章"
                />
              </Form.Item>

              <Form.Item
                label="展开策略"
                name="expansion_strategy"
                tooltip="选择如何分配内容到各章节"
              >
                <Radio.Group>
                  <Radio.Button value="balanced">均衡分配</Radio.Button>
                  <Radio.Button value="climax">高潮重点</Radio.Button>
                  <Radio.Button value="detail">细节丰富</Radio.Button>
                </Radio.Group>
              </Form.Item>
            </Form>
          </div>
        ),
        okText: '生成规划预览',
        cancelText: '取消',
        onOk: async () => {
          try {
            const values = await expansionForm.validateFields();

            // 关闭配置表单
            Modal.destroyAll();

            // 显示SSE进度Modal
            setSSEProgress(0);
            setSSEMessage('正在准备展开大纲...');
            setSSEModalVisible(true);
            setIsExpanding(true);

            // 准备请求数据
            const requestData = {
              ...values,
              auto_create_chapters: false, // 第一步：仅生成规划
              enable_scene_analysis: true
            };

            const taskId = createTask({
              type: 'outline_expand',
              name: `展开大纲《${outlineTitle}》`,
              message: '任务初始化中...',
              maxRetries: 5,
              retryAction: () => {
                setIsExpanding(true);
                setSSEProgress(0);
                setSSEMessage('正在重试展开大纲...');
                setSSEModalVisible(true);

                const retryClient = new SSEPostClient(
                  `/api/outlines/${outlineId}/expand-stream`,
                  requestData,
                  {
                    onProgress: (msg: string, progress: number) => {
                      setSSEMessage(msg);
                      setSSEProgress(progress);
                      setTaskProgress(taskId, progress, msg);
                    },
                    onResult: (data: OutlineExpansionResponse) => {
                      setSSEModalVisible(false);
                      showExpansionPreview(outlineId, data);
                    },
                    onError: (error: string) => {
                      message.error(`展开失败: ${error}`);
                      setSSEModalVisible(false);
                      setIsExpanding(false);
                      markTaskFailed(taskId, error);
                    },
                    onComplete: () => {
                      setSSEModalVisible(false);
                      setIsExpanding(false);
                      markTaskSuccess(taskId, `大纲《${outlineTitle}》展开完成`);
                    },
                  }
                );
                retryClient.connect();
              },
            });
            setTaskProgress(taskId, 0, '正在准备展开大纲...');

            // 使用SSE客户端调用新的流式端点
            const apiUrl = `/api/outlines/${outlineId}/expand-stream`;
            const client = new SSEPostClient(apiUrl, requestData, {
              onProgress: (msg: string, progress: number) => {
                setSSEMessage(msg);
                setSSEProgress(progress);
                setTaskProgress(taskId, progress, msg);
              },
              onResult: (data: OutlineExpansionResponse) => {
                console.log('展开完成，结果:', data);
                // 关闭SSE进度Modal
                setSSEModalVisible(false);
                // 显示规划预览
                showExpansionPreview(outlineId, data);
              },
              onError: (error: string) => {
                message.error(`展开失败: ${error}`);
                setSSEModalVisible(false);
                setIsExpanding(false);
                markTaskFailed(taskId, error);
              },
              onComplete: () => {
                setSSEModalVisible(false);
                setIsExpanding(false);
                markTaskSuccess(taskId, `大纲《${outlineTitle}》展开完成`);
              }
            });

            // 开始连接
            client.connect();

          } catch (error) {
            console.error('展开失败:', error);
            message.error('展开失败');
            setSSEModalVisible(false);
            setIsExpanding(false);
          }
        },
      });
    } catch (error) {
      console.error('检查章节失败:', error);
      message.error('检查章节失败');
      setIsExpanding(false);
    }
  };

  // 删除展开的章节内容（保留大纲）
  const handleDeleteExpandedChapters = async (outlineTitle: string, chapters: Array<{ id: string }>) => {
    try {
      // 使用顺序删除避免并发导致的字数计算竞态条件
      // 并发删除会导致多个请求同时读取项目字数并各自减去章节字数，造成计算错误
      for (const chapter of chapters) {
        await chapterApi.deleteChapter(chapter.id);
      }

      message.success(`已删除《${outlineTitle}》展开的所有 ${chapters.length} 个章节`);
      await refreshOutlines();
      // 刷新项目信息以更新字数显示
      if (currentProject?.id) {
        const updatedProject = await projectApi.getProject(currentProject.id);
        setCurrentProject(updatedProject);
      }
      // 更新展开状态
      setOutlineExpandStatus(prev => {
        const newStatus = { ...prev };
        // 找到被删除章节对应的大纲ID并更新其状态
        const outlineId = Object.keys(newStatus).find(id =>
          outlines.find(o => o.id === id && o.title === outlineTitle)
        );
        if (outlineId) {
          newStatus[outlineId] = false;
        }
        return newStatus;
      });
    } catch (error: unknown) {
      const apiError = error as ApiError;
      message.error(apiError.response?.data?.detail || '删除章节失败');
    }
  };

  // 显示已存在章节的展开规划
  const showExistingExpansionPreview = (
    outlineTitle: string,
    data: {
      chapter_count: number;
      chapters: Array<{ id: string; chapter_number: number; title: string }>;
      expansion_plans: Array<{
        sub_index: number;
        title: string;
        plot_summary: string;
        key_events: string[];
        character_focus: string[];
        emotional_tone: string;
        narrative_goal: string;
        conflict_type: string;
        estimated_words: number;
        scenes?: Array<{
          location: string;
          characters: string[];
          purpose: string;
        }> | null;
      }> | null;
    }
  ) => {
    modalApi.info({
      title: (
        <Space style={{ flexWrap: 'wrap' }}>
          <CheckCircleOutlined style={{ color: 'var(--color-success)' }} />
          <span>《{outlineTitle}》展开信息</span>
        </Space>
      ),
      width: isMobile ? '95%' : 900,
      centered: true,
      style: isMobile ? {
        top: 20,
        maxWidth: 'calc(100vw - 16px)',
        margin: '0 8px'
      } : undefined,
      styles: {
        body: {
          maxHeight: isMobile ? 'calc(100vh - 200px)' : 'calc(80vh - 60px)',
          overflowY: 'auto',
          overflowX: 'hidden'
        }
      },
      footer: (
        <Space wrap style={{ width: '100%', justifyContent: isMobile ? 'center' : 'flex-end' }}>
          <Button
            danger
            icon={<DeleteOutlined />}
            onClick={() => {
              Modal.destroyAll();
              modalApi.confirm({
                title: '确认删除',
                icon: <ExclamationCircleOutlined />,
                centered: true,
                content: (
                  <div>
                    <p>此操作将删除大纲《{outlineTitle}》展开的所有 <strong>{data.chapter_count}</strong> 个章节。</p>
                    <p style={{ color: 'var(--color-primary)', marginTop: 8 }}>
                      📝 注意：大纲本身会保留，您可以重新展开
                    </p>
                    <p style={{ color: '#ff4d4f', marginTop: 8 }}>
                      ⚠️ 警告：章节内容将永久删除且无法恢复！
                    </p>
                  </div>
                ),
                okText: '确认删除',
                okType: 'danger',
                cancelText: '取消',
                onOk: () => handleDeleteExpandedChapters(outlineTitle, data.chapters || []),
              });
            }}
            block={isMobile}
            size={isMobile ? 'middle' : undefined}
          >
            删除所有展开的章节 ({data.chapter_count}章)
          </Button>
          <Button onClick={() => Modal.destroyAll()}>
            关闭
          </Button>
        </Space>
      ),
      content: (
        <div>
          <div style={{ marginBottom: 16 }}>
            <Space wrap style={{ maxWidth: '100%' }}>
              <Tag
                color="blue"
                style={{
                  whiteSpace: 'normal',
                  wordBreak: 'break-word',
                  height: 'auto',
                  lineHeight: '1.5',
                  padding: '4px 8px'
                }}
              >
                大纲: {outlineTitle}
              </Tag>
              <Tag color="green">章节数: {data.chapter_count}</Tag>
              <Tag color="orange">已创建章节</Tag>
            </Space>
          </div>
          <Tabs
            defaultActiveKey="0"
            type="card"
            items={data.expansion_plans?.map((plan, idx) => ({
              key: idx.toString(),
              label: (
                <Space size="small" style={{ maxWidth: isMobile ? '150px' : 'none' }}>
                  <span
                    style={{
                      fontWeight: 500,
                      whiteSpace: isMobile ? 'normal' : 'nowrap',
                      wordBreak: isMobile ? 'break-word' : 'normal',
                      fontSize: isMobile ? 12 : 14
                    }}
                  >
                    {plan.sub_index}. {plan.title}
                  </span>
                </Space>
              ),
              children: (
                <div style={{ maxHeight: '500px', overflowY: 'auto', padding: '8px 0' }}>
                  <Space direction="vertical" size="middle" style={{ width: '100%' }}>
                    <Card size="small" title="基本信息">
                      <Space wrap style={{ maxWidth: '100%' }}>
                        <Tag
                          color="blue"
                          style={{
                            whiteSpace: 'normal',
                            wordBreak: 'break-word',
                            height: 'auto',
                            lineHeight: '1.5',
                            padding: '4px 8px'
                          }}
                        >
                          {plan.emotional_tone}
                        </Tag>
                        <Tag
                          color="orange"
                          style={{
                            whiteSpace: 'normal',
                            wordBreak: 'break-word',
                            height: 'auto',
                            lineHeight: '1.5',
                            padding: '4px 8px'
                          }}
                        >
                          {plan.conflict_type}
                        </Tag>
                        <Tag color="green">约{plan.estimated_words}字</Tag>
                      </Space>
                    </Card>

                    <Card size="small" title="情节概要">
                      <div style={{
                        wordBreak: 'break-word',
                        whiteSpace: 'normal',
                        overflowWrap: 'break-word'
                      }}>
                        {plan.plot_summary}
                      </div>
                    </Card>

                    <Card size="small" title="叙事目标">
                      <div style={{
                        wordBreak: 'break-word',
                        whiteSpace: 'normal',
                        overflowWrap: 'break-word'
                      }}>
                        {plan.narrative_goal}
                      </div>
                    </Card>

                    <Card size="small" title="关键事件">
                      <Space direction="vertical" size="small" style={{ width: '100%' }}>
                        {plan.key_events.map((event, eventIdx) => (
                          <div
                            key={eventIdx}
                            style={{
                              wordBreak: 'break-word',
                              whiteSpace: 'normal',
                              overflowWrap: 'break-word'
                            }}
                          >
                            • {event}
                          </div>
                        ))}
                      </Space>
                    </Card>

                    <Card size="small" title="涉及角色">
                      <Space wrap style={{ maxWidth: '100%' }}>
                        {plan.character_focus.map((char, charIdx) => (
                          <Tag
                            key={charIdx}
                            color="purple"
                            style={{
                              whiteSpace: 'normal',
                              wordBreak: 'break-word',
                              height: 'auto',
                              lineHeight: '1.5'
                            }}
                          >
                            {char}
                          </Tag>
                        ))}
                      </Space>
                    </Card>

                    {plan.scenes && plan.scenes.length > 0 && (
                      <Card size="small" title="场景">
                        <Space direction="vertical" size="small" style={{ width: '100%' }}>
                          {plan.scenes.map((scene, sceneIdx) => (
                            <Card
                              key={sceneIdx}
                              size="small"
                              style={{
                                backgroundColor: '#fafafa',
                                maxWidth: '100%',
                                overflow: 'hidden'
                              }}
                            >
                              <div style={{
                                wordBreak: 'break-word',
                                whiteSpace: 'normal',
                                overflowWrap: 'break-word'
                              }}>
                                <strong>地点：</strong>{scene.location}
                              </div>
                              <div style={{
                                wordBreak: 'break-word',
                                whiteSpace: 'normal',
                                overflowWrap: 'break-word'
                              }}>
                                <strong>角色：</strong>{scene.characters.join('、')}
                              </div>
                              <div style={{
                                wordBreak: 'break-word',
                                whiteSpace: 'normal',
                                overflowWrap: 'break-word'
                              }}>
                                <strong>目的：</strong>{scene.purpose}
                              </div>
                            </Card>
                          ))}
                        </Space>
                      </Card>
                    )
                    }
                  </Space>
                </div >
              )
            }))}
          />
        </div >
      ),
    });
  };

  // 显示展开规划预览，并提供确认创建章节的选项
  const showExpansionPreview = (outlineId: string, response: OutlineExpansionResponse) => {
    // 缓存AI生成的规划数据
    const cachedPlans = response.chapter_plans;

    modalApi.confirm({
      title: (
        <Space>
          <CheckCircleOutlined style={{ color: 'var(--color-success)' }} />
          <span>展开规划预览</span>
        </Space>
      ),
      width: 900,
      centered: true,
      okText: '确认并创建章节',
      cancelText: '暂不创建',
      content: (
        <div>
          <div style={{ marginBottom: 16 }}>
            <Tag color="blue">策略: {response.expansion_strategy}</Tag>
            <Tag color="green">章节数: {response.actual_chapter_count}</Tag>
            <Tag color="orange">预览模式（未创建章节）</Tag>
          </div>
          <Tabs
            defaultActiveKey="0"
            type="card"
            items={response.chapter_plans.map((plan, idx) => ({
              key: idx.toString(),
              label: (
                <Space size="small">
                  <span style={{ fontWeight: 500 }}>{idx + 1}. {plan.title}</span>
                </Space>
              ),
              children: (
                <div style={{ maxHeight: '500px', overflowY: 'auto', padding: '8px 0' }}>
                  <Space direction="vertical" size="middle" style={{ width: '100%' }}>
                    <Card size="small" title="基本信息">
                      <Space wrap>
                        <Tag color="blue">{plan.emotional_tone}</Tag>
                        <Tag color="orange">{plan.conflict_type}</Tag>
                        <Tag color="green">约{plan.estimated_words}字</Tag>
                      </Space>
                    </Card>

                    <Card size="small" title="情节概要">
                      {plan.plot_summary}
                    </Card>

                    <Card size="small" title="叙事目标">
                      {plan.narrative_goal}
                    </Card>

                    <Card size="small" title="关键事件">
                      <Space direction="vertical" size="small" style={{ width: '100%' }}>
                        {plan.key_events.map((event, eventIdx) => (
                          <div key={eventIdx}>• {event}</div>
                        ))}
                      </Space>
                    </Card>

                    <Card size="small" title="涉及角色">
                      <Space wrap>
                        {plan.character_focus.map((char, charIdx) => (
                          <Tag key={charIdx} color="purple">{char}</Tag>
                        ))}
                      </Space>
                    </Card>

                    {plan.scenes && plan.scenes.length > 0 && (
                      <Card size="small" title="场景">
                        <Space direction="vertical" size="small" style={{ width: '100%' }}>
                          {plan.scenes.map((scene, sceneIdx) => (
                            <Card key={sceneIdx} size="small" style={{ backgroundColor: '#fafafa' }}>
                              <div><strong>地点：</strong>{scene.location}</div>
                              <div><strong>角色：</strong>{scene.characters.join('、')}</div>
                              <div><strong>目的：</strong>{scene.purpose}</div>
                            </Card>
                          ))}
                        </Space>
                      </Card>
                    )}
                  </Space>
                </div>
              )
            }))}
          />
        </div>
      ),
      onOk: async () => {
        // 第二步：用户确认后，直接使用缓存的规划创建章节（避免重复调用AI）
        await handleConfirmCreateChapters(outlineId, cachedPlans);
      },
      onCancel: () => {
        message.info('已取消创建章节');
      }
    });
  };

  // 确认创建章节 - 使用缓存的规划数据，避免重复AI调用
  const handleConfirmCreateChapters = async (
    outlineId: string,
    cachedPlans: ChapterPlanItem[]
  ) => {
    try {
      setIsExpanding(true);

      // 使用新的API端点，直接传递缓存的规划数据
      const response = await outlineApi.createChaptersFromPlans(outlineId, cachedPlans);

      message.success(
        `成功创建${response.chapters_created}个章节！`,
        3
      );

      console.log('✅ 使用缓存的规划创建章节，避免了重复的AI调用');

      // 刷新大纲和章节列表
      refreshOutlines();

    } catch (error) {
      console.error('创建章节失败:', error);
      message.error('创建章节失败');
    } finally {
      setIsExpanding(false);
    }
  };

  // 批量展开所有大纲 - 使用SSE流式显示进度
  const handleBatchExpandOutlines = () => {
    if (!currentProject?.id || outlines.length === 0) {
      message.warning('没有可展开的大纲');
      return;
    }

    modalApi.confirm({
      title: (
        <Space>
          <AppstoreAddOutlined />
          <span>批量展开所有大纲</span>
        </Space>
      ),
      width: 600,
      centered: true,
      content: (
        <div>
          <div style={{ marginBottom: 16, padding: 12, background: 'var(--color-warning-bg)', borderRadius: 4 }}>
            <div style={{ color: '#856404' }}>
              ⚠️ 将对当前项目的所有 {outlines.length} 个大纲进行展开
            </div>
          </div>
          <Form
            form={batchExpansionForm}
            layout="vertical"
            initialValues={{
              chapters_per_outline: 3,
              expansion_strategy: 'balanced',
            }}
          >
            <Form.Item
              label="每个大纲展开章节数"
              name="chapters_per_outline"
              rules={[{ required: true, message: '请输入章节数' }]}
              tooltip="每个大纲将被展开为几章"
            >
              <InputNumber
                min={2}
                max={10}
                style={{ width: '100%' }}
                placeholder="建议2-5章"
              />
            </Form.Item>

            <Form.Item
              label="展开策略"
              name="expansion_strategy"
            >
              <Radio.Group>
                <Radio.Button value="balanced">均衡分配</Radio.Button>
                <Radio.Button value="climax">高潮重点</Radio.Button>
                <Radio.Button value="detail">细节丰富</Radio.Button>
              </Radio.Group>
            </Form.Item>
          </Form>
        </div>
      ),
      okText: '开始展开',
      cancelText: '取消',
      okButtonProps: { type: 'primary' },
      onOk: async () => {
        try {
          const values = await batchExpansionForm.validateFields();

          // 关闭配置表单
          Modal.destroyAll();

          // 显示SSE进度Modal
          setSSEProgress(0);
          setSSEMessage('正在准备批量展开...');
          setSSEModalVisible(true);
          setIsExpanding(true);

          // 准备请求数据
          const requestData = {
            project_id: currentProject.id,
            ...values,
            auto_create_chapters: false // 第一步：仅生成规划
          };

          const taskId = createTask({
            type: 'outline_batch_expand',
            name: `批量展开大纲（${outlines.length}个）`,
            message: '任务初始化中...',
            maxRetries: 5,
            retryAction: () => {
              setIsExpanding(true);
              setSSEProgress(0);
              setSSEMessage('正在重试批量展开...');
              setSSEModalVisible(true);

              const retryClient = new SSEPostClient('/api/outlines/batch-expand-stream', requestData, {
                onProgress: (msg: string, progress: number) => {
                  setSSEMessage(msg);
                  setSSEProgress(progress);
                  setTaskProgress(taskId, progress, msg);
                },
                onResult: (data: BatchOutlineExpansionResponse) => {
                  setCachedBatchExpansionResponse(data);
                  setBatchPreviewData(data);
                  setSSEModalVisible(false);
                  setSelectedOutlineIdx(0);
                  setSelectedChapterIdx(0);
                  setBatchPreviewVisible(true);
                },
                onError: (error: string) => {
                  message.error(`批量展开失败: ${error}`);
                  setSSEModalVisible(false);
                  setIsExpanding(false);
                  markTaskFailed(taskId, error);
                },
                onComplete: () => {
                  setSSEModalVisible(false);
                  setIsExpanding(false);
                  markTaskSuccess(taskId, '批量大纲展开完成');
                }
              });

              retryClient.connect();
            }
          });
          setTaskProgress(taskId, 0, '正在准备批量展开...');

          // 使用SSE客户端
          const apiUrl = `/api/outlines/batch-expand-stream`;
          const client = new SSEPostClient(apiUrl, requestData, {
            onProgress: (msg: string, progress: number) => {
              setSSEMessage(msg);
              setSSEProgress(progress);
              setTaskProgress(taskId, progress, msg);
            },
            onResult: (data: BatchOutlineExpansionResponse) => {
              console.log('批量展开完成，结果:', data);
              // 缓存AI生成的规划数据
              setCachedBatchExpansionResponse(data);
              setBatchPreviewData(data);
              // 关闭SSE进度Modal
              setSSEModalVisible(false);
              // 重置选择状态
              setSelectedOutlineIdx(0);
              setSelectedChapterIdx(0);
              // 显示批量预览Modal
              setBatchPreviewVisible(true);
            },
            onError: (error: string) => {
              message.error(`批量展开失败: ${error}`);
              setSSEModalVisible(false);
              setIsExpanding(false);
              markTaskFailed(taskId, error);
            },
            onComplete: () => {
              setSSEModalVisible(false);
              setIsExpanding(false);
              markTaskSuccess(taskId, '批量大纲展开完成');
            }
          });

          // 开始连接
          client.connect();

        } catch (error) {
          console.error('批量展开失败:', error);
          message.error('批量展开失败');
          setSSEModalVisible(false);
          setIsExpanding(false);
        }
      },
    });
  };

  // 渲染批量展开预览 Modal 内容
  const renderBatchPreviewContent = () => {
    if (!batchPreviewData) return null;

    return (
      <div>
        {/* 顶部统计信息 */}
        <div style={{ marginBottom: 16 }}>
          <Tag color="blue">已处理: {batchPreviewData.total_outlines_expanded} 个大纲</Tag>
          <Tag color="green">总章节数: {batchPreviewData.expansion_results.reduce((sum: number, r: OutlineExpansionResponse) => sum + r.actual_chapter_count, 0)}</Tag>
          <Tag color="orange">预览模式（未创建章节）</Tag>
          {batchPreviewData.skipped_outlines && batchPreviewData.skipped_outlines.length > 0 && (
            <Tag color="warning">跳过: {batchPreviewData.skipped_outlines.length} 个大纲</Tag>
          )}
        </div>

        {/* 显示跳过的大纲信息 */}
        {batchPreviewData.skipped_outlines && batchPreviewData.skipped_outlines.length > 0 && (
          <div style={{
            marginBottom: 16,
            padding: 12,
            background: 'var(--color-warning-bg)',
            borderRadius: 4,
            border: '1px solid #ffe58f'
          }}>
            <div style={{ fontWeight: 500, marginBottom: 8, color: 'var(--color-warning)' }}>
              ⚠️ 以下大纲已展开过，已自动跳过：
            </div>
            <Space direction="vertical" size="small" style={{ width: '100%' }}>
              {batchPreviewData.skipped_outlines.map((skipped: SkippedOutlineInfo, idx: number) => (
                <div key={idx} style={{ fontSize: 13, color: '#666' }}>
                  • {skipped.outline_title} <Tag color="default" style={{ fontSize: 11 }}>{skipped.reason}</Tag>
                </div>
              ))}
            </Space>
          </div>
        )}

        {/* 水平三栏布局 */}
        <div style={{ display: 'flex', gap: 16, height: 500 }}>
          {/* 左栏：大纲列表 */}
          <div style={{
            width: 280,
            borderRight: '1px solid #f0f0f0',
            paddingRight: 12,
            overflowY: 'auto'
          }}>
            <div style={{ fontWeight: 500, marginBottom: 8, color: '#666' }}>大纲列表</div>
            <List
              size="small"
              dataSource={batchPreviewData.expansion_results}
              renderItem={(result: OutlineExpansionResponse, idx: number) => (
                <List.Item
                  key={idx}
                  onClick={() => {
                    setSelectedOutlineIdx(idx);
                    setSelectedChapterIdx(0);
                  }}
                  style={{
                    cursor: 'pointer',
                    padding: '8px 12px',
                    background: selectedOutlineIdx === idx ? '#e6f7ff' : 'transparent',
                    borderRadius: 4,
                    marginBottom: 4,
                    border: selectedOutlineIdx === idx ? '1px solid var(--color-primary)' : '1px solid transparent'
                  }}
                >
                  <div style={{ width: '100%' }}>
                    <div style={{ fontWeight: 500, fontSize: 13, marginBottom: 4 }}>
                      {idx + 1}. {result.outline_title}
                    </div>
                    <Space size={4}>
                      <Tag color="blue" style={{ fontSize: 11, margin: 0 }}>{result.expansion_strategy}</Tag>
                      <Tag color="green" style={{ fontSize: 11, margin: 0 }}>{result.actual_chapter_count} 章</Tag>
                    </Space>
                  </div>
                </List.Item>
              )}
            />
          </div>

          {/* 中栏：章节列表 */}
          <div style={{
            width: 320,
            borderRight: '1px solid #f0f0f0',
            paddingRight: 12,
            overflowY: 'auto'
          }}>
            <div style={{ fontWeight: 500, marginBottom: 8, color: '#666' }}>
              章节列表 ({batchPreviewData.expansion_results[selectedOutlineIdx]?.actual_chapter_count || 0} 章)
            </div>
            {batchPreviewData.expansion_results[selectedOutlineIdx] && (
              <List
                size="small"
                dataSource={batchPreviewData.expansion_results[selectedOutlineIdx].chapter_plans}
                renderItem={(plan: ChapterPlanItem, idx: number) => (
                  <List.Item
                    key={idx}
                    onClick={() => setSelectedChapterIdx(idx)}
                    style={{
                      cursor: 'pointer',
                      padding: '8px 12px',
                      background: selectedChapterIdx === idx ? '#e6f7ff' : 'transparent',
                      borderRadius: 4,
                      marginBottom: 4,
                      border: selectedChapterIdx === idx ? '1px solid var(--color-primary)' : '1px solid transparent'
                    }}
                  >
                    <div style={{ width: '100%' }}>
                      <div style={{ fontWeight: 500, fontSize: 13, marginBottom: 4 }}>
                        {idx + 1}. {plan.title}
                      </div>
                      <Space size={4} wrap>
                        <Tag color="blue" style={{ fontSize: 11, margin: 0 }}>{plan.emotional_tone}</Tag>
                        <Tag color="orange" style={{ fontSize: 11, margin: 0 }}>{plan.conflict_type}</Tag>
                        <Tag color="green" style={{ fontSize: 11, margin: 0 }}>约{plan.estimated_words}字</Tag>
                      </Space>
                    </div>
                  </List.Item>
                )}
              />
            )}
          </div>

          {/* 右栏：章节详情 */}
          <div style={{ flex: 1, overflowY: 'auto', paddingLeft: 12 }}>
            <div style={{ fontWeight: 500, marginBottom: 12, color: '#666' }}>章节详情</div>
            {batchPreviewData.expansion_results[selectedOutlineIdx]?.chapter_plans[selectedChapterIdx] ? (
              <Space direction="vertical" size="middle" style={{ width: '100%' }}>
                <Card size="small" title="情节概要" bordered={false}>
                  {batchPreviewData.expansion_results[selectedOutlineIdx].chapter_plans[selectedChapterIdx].plot_summary}
                </Card>

                <Card size="small" title="叙事目标" bordered={false}>
                  {batchPreviewData.expansion_results[selectedOutlineIdx].chapter_plans[selectedChapterIdx].narrative_goal}
                </Card>

                <Card size="small" title="关键事件" bordered={false}>
                  <Space direction="vertical" size="small" style={{ width: '100%' }}>
                    {(batchPreviewData.expansion_results[selectedOutlineIdx].chapter_plans[selectedChapterIdx].key_events as string[]).map((event: string, eventIdx: number) => (
                      <div key={eventIdx}>• {event}</div>
                    ))}
                  </Space>
                </Card>

                <Card size="small" title="涉及角色" bordered={false}>
                  <Space wrap>
                    {(batchPreviewData.expansion_results[selectedOutlineIdx].chapter_plans[selectedChapterIdx].character_focus as string[]).map((char: string, charIdx: number) => (
                      <Tag key={charIdx} color="purple">{char}</Tag>
                    ))}
                  </Space>
                </Card>

                {batchPreviewData.expansion_results[selectedOutlineIdx].chapter_plans[selectedChapterIdx].scenes && batchPreviewData.expansion_results[selectedOutlineIdx].chapter_plans[selectedChapterIdx].scenes!.length > 0 && (
                  <Card size="small" title="场景" bordered={false}>
                    <Space direction="vertical" size="small" style={{ width: '100%' }}>
                      {batchPreviewData.expansion_results[selectedOutlineIdx].chapter_plans[selectedChapterIdx].scenes!.map((scene: SceneInfo, sceneIdx: number) => (
                        <Card key={sceneIdx} size="small" style={{ backgroundColor: '#fafafa' }}>
                          <div><strong>地点：</strong>{scene.location}</div>
                          <div><strong>角色：</strong>{scene.characters.join('、')}</div>
                          <div><strong>目的：</strong>{scene.purpose}</div>
                        </Card>
                      ))}
                    </Space>
                  </Card>
                )}
              </Space>
            ) : (
              <Empty description="请选择章节查看详情" />
            )}
          </div>
        </div>
      </div>
    );
  };

  // 处理批量预览确认
  const handleBatchPreviewOk = async () => {
    setBatchPreviewVisible(false);
    await handleConfirmBatchCreateChapters();
  };

  // 处理批量预览取消
  const handleBatchPreviewCancel = () => {
    setBatchPreviewVisible(false);
    message.info('已取消创建章节，规划已保存');
  };


  // 确认批量创建章节 - 使用缓存的规划数据
  const handleConfirmBatchCreateChapters = async () => {
    try {
      setIsExpanding(true);

      // 使用缓存的规划数据，避免重复调用AI
      if (!cachedBatchExpansionResponse) {
        message.error('规划数据丢失，请重新展开');
        return;
      }

      console.log('✅ 使用缓存的批量规划数据创建章节，避免重复AI调用');

      // 逐个大纲创建章节
      let totalCreated = 0;
      const errors: string[] = [];

      for (const result of cachedBatchExpansionResponse.expansion_results) {
        try {
          // 使用create-chapters-from-plans接口，直接传递缓存的规划
          const response = await outlineApi.createChaptersFromPlans(
            result.outline_id,
            result.chapter_plans
          );
          totalCreated += response.chapters_created;
        } catch (error: unknown) {
          const apiError = error as ApiError;
          const err = error as Error;
          const errorMsg = apiError.response?.data?.detail || err.message || '未知错误';
          errors.push(`${result.outline_title}: ${errorMsg}`);
          console.error(`创建大纲 ${result.outline_title} 的章节失败:`, error);
        }
      }

      // 显示结果
      if (errors.length === 0) {
        message.success(
          `批量创建完成！共创建 ${totalCreated} 个章节`,
          3
        );
      } else {
        message.warning(
          `部分完成：成功创建 ${totalCreated} 个章节，${errors.length} 个失败`,
          5
        );
        console.error('失败详情:', errors);
      }

      // 清除缓存
      setCachedBatchExpansionResponse(null);

      // 刷新列表
      refreshOutlines();

    } catch (error) {
      console.error('批量创建章节失败:', error);
      message.error('批量创建章节失败');
    } finally {
      setIsExpanding(false);
    }
  };


  return (
    <>
      {/* 批量展开预览 Modal */}
      <Modal
        title={
          <Space>
            <CheckCircleOutlined style={{ color: 'var(--color-success)' }} />
            <span>批量展开规划预览</span>
          </Space>
        }
        open={batchPreviewVisible}
        onOk={handleBatchPreviewOk}
        onCancel={handleBatchPreviewCancel}
        width={1200}
        centered
        okText="确认并批量创建章节"
        cancelText="暂不创建"
        okButtonProps={{ danger: true }}
      >
        {renderBatchPreviewContent()}
      </Modal>

      {contextHolder}
      {/* SSE进度Modal - 使用统一组件 */}
      <SSEProgressModal
        visible={sseModalVisible}
        progress={sseProgress}
        message={sseMessage}
        title="AI生成中..."
      />

      <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
        {/* 固定头部 */}
        <div style={{
          position: 'sticky',
          top: 0,
          zIndex: 10,
          backgroundColor: 'var(--color-bg-container)',
          padding: isMobile ? '12px 0' : '16px 0',
          marginBottom: isMobile ? 12 : 16,
          borderBottom: '1px solid #f0f0f0',
          display: 'flex',
          flexDirection: isMobile ? 'column' : 'row',
          gap: isMobile ? 12 : 0,
          justifyContent: 'space-between',
          alignItems: isMobile ? 'stretch' : 'center'
        }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            <h2 style={{ margin: 0, fontSize: isMobile ? 18 : 24 }}>
              <FileTextOutlined style={{ marginRight: 8 }} />
              故事大纲
            </h2>
            {currentProject?.outline_mode && (
              <Tag color={currentProject.outline_mode === 'one-to-one' ? 'blue' : 'green'} style={{ width: 'fit-content' }}>
                {currentProject.outline_mode === 'one-to-one' ? '传统模式 (1→1)' : '细化模式 (1→N)'}
              </Tag>
            )}
          </div>
          <Space size="small" wrap={isMobile}>
            <Button
              icon={<PlusOutlined />}
              onClick={showManualCreateOutlineModal}
              block={isMobile}
            >
              手动创建
            </Button>
            <Button
              type="primary"
              icon={<ThunderboltOutlined />}
              onClick={showGenerateModal}
              loading={isGenerating}
              block={isMobile}
            >
              {isMobile ? 'AI生成/续写' : 'AI生成/续写大纲'}
            </Button>
            <Button
              icon={<CheckCircleOutlined />}
              onClick={() => handleRunOutlinePostcheck('all')}
              loading={isPostchecking}
              disabled={isGenerating || outlines.length === 0}
              block={isMobile}
              title="手动触发大纲后处理：补全新人物与新组织/地点信息"
            >
              {isMobile ? '补全人物/地点' : '补全新人物/地点'}
            </Button>
            {outlines.length > 0 && currentProject?.outline_mode === 'one-to-many' && (
              <Button
                icon={<AppstoreAddOutlined />}
                onClick={handleBatchExpandOutlines}
                loading={isExpanding}
                disabled={isGenerating}
                title="将所有大纲展开为多章，实现从大纲到章节的一对多关系"
              >
                {isMobile ? '批量展开' : '批量展开为多章'}
              </Button>
            )}
          </Space>
        </div>

        {/* 可滚动内容区域 */}
        <div style={{ flex: 1, overflowY: 'auto' }}>
          {lastPostcheckSummary && (
            <Card
              size="small"
              style={{
                marginBottom: 12,
                borderColor:
                  lastPostcheckSummary.overall_status === 'success'
                    ? 'var(--color-success-border)'
                    : 'var(--color-warning-border)',
                background:
                  lastPostcheckSummary.overall_status === 'success'
                    ? 'var(--color-success-bg)'
                    : 'var(--color-warning-bg)'
              }}
            >
              <Space size="small" wrap>
                <Tag color={
                  lastPostcheckSummary.overall_status === 'success'
                    ? 'success'
                    : lastPostcheckSummary.overall_status === 'partial_failed'
                      ? 'warning'
                      : 'error'
                }>
                  后处理状态：{lastPostcheckSummary.overall_status}
                </Tag>
                <Tag color="blue">新增总数：{lastPostcheckSummary.total_created}</Tag>
                {lastPostcheckSummary.characters && (
                  <Tag color={lastPostcheckSummary.characters.success ? 'green' : 'orange'}>
                    角色：{lastPostcheckSummary.characters.created_count}
                  </Tag>
                )}
                {lastPostcheckSummary.organizations && (
                  <Tag color={lastPostcheckSummary.organizations.success ? 'green' : 'orange'}>
                    组织/地点：{lastPostcheckSummary.organizations.created_count}
                  </Tag>
                )}
              </Space>
              {(lastPostcheckSummary.characters?.error || lastPostcheckSummary.organizations?.error) && (
                <div style={{ marginTop: 8, color: 'var(--color-warning)' }}>
                  {lastPostcheckSummary.characters?.error && (
                    <div>角色补全错误：{lastPostcheckSummary.characters.error}</div>
                  )}
                  {lastPostcheckSummary.organizations?.error && (
                    <div>组织/地点补全错误：{lastPostcheckSummary.organizations.error}</div>
                  )}
                </div>
              )}
            </Card>
          )}
          {outlines.length === 0 ? (
            <Empty description="还没有大纲，开始创建吧！" />
          ) : (
            <List
              dataSource={sortedOutlines}
              renderItem={(item) => {
                  // 解析structure字段获取所有信息
                  let structureData: {
                    key_events?: string[];
                    key_points?: string[];  // AI生成的情节要点
                    characters_involved?: string[];
                    characters?: unknown[];  // 兼容新旧格式
                    scenes?: string[] | Array<{
                      location: string;
                      characters: string[];
                      purpose: string;
                    }>;
                    emotion?: string;  // AI生成的情感基调
                    goal?: string;  // AI生成的叙事目标
                  } = {};
                  
                  if (item.structure) {
                    try {
                      structureData = JSON.parse(item.structure);
                    } catch (e) {
                      console.error('解析structure失败:', e);
                    }
                  }
                  
                  // 解析角色/组织条目（兼容新旧格式）
                  const characterEntries = parseCharacterEntries(structureData.characters);
                  const characterNames = getCharacterNames(characterEntries);
                  const organizationNames = getOrganizationNames(characterEntries);
                  
                  return (
                    <List.Item
                      style={{
                        marginBottom: 16,
                        padding: 0,
                        border: 'none'
                      }}
                    >
                      <Card
                        style={{
                          width: '100%',
                          borderRadius: isMobile ? 6 : 8,
                          border: '1px solid #f0f0f0',
                          boxShadow: '0 1px 2px rgba(0, 0, 0, 0.03)',
                          transition: 'all 0.3s ease'
                        }}
                        bodyStyle={{
                          padding: isMobile ? '10px 12px' : 16
                        }}
                        onMouseEnter={(e) => {
                          if (!isMobile) {
                            e.currentTarget.style.boxShadow = '0 4px 12px rgba(0, 0, 0, 0.08)';
                            e.currentTarget.style.borderColor = 'var(--color-primary)';
                          }
                        }}
                        onMouseLeave={(e) => {
                          if (!isMobile) {
                            e.currentTarget.style.boxShadow = '0 1px 2px rgba(0, 0, 0, 0.03)';
                            e.currentTarget.style.borderColor = '#f0f0f0';
                          }
                        }}
                      >
                        <List.Item.Meta
                          style={{ width: '100%' }}
                          title={
                            <Space size="small" style={{ fontSize: isMobile ? 13 : 16, flexWrap: 'wrap', lineHeight: isMobile ? '1.4' : '1.5' }}>
                              <span style={{ color: 'var(--color-primary)', fontWeight: 'bold', fontSize: isMobile ? 13 : 16 }}>
                                {currentProject?.outline_mode === 'one-to-one'
                                  ? `第${item.order_index || '?'}章`
                                  : `第${item.order_index || '?'}卷`
                                }
                              </span>
                              <span style={{ fontSize: isMobile ? 13 : 16 }}>{item.title}</span>
                              {/* ✅ 新增：展开状态标识 - 仅在一对多模式显示 */}
                              {currentProject?.outline_mode === 'one-to-many' && (
                                outlineExpandStatus[item.id] ? (
                                  <Tag color="success" icon={<CheckCircleOutlined />} style={{ fontSize: isMobile ? 11 : 12 }}>已展开</Tag>
                                ) : (
                                  <Tag color="default" style={{ fontSize: isMobile ? 11 : 12 }}>未展开</Tag>
                                )
                              )}
                            </Space>
                          }
                          description={
                            <div style={{ fontSize: isMobile ? 12 : 14, lineHeight: isMobile ? '1.5' : '1.6' }}>
                              {/* 大纲内容 */}
                              <div style={{
                                marginBottom: isMobile ? 10 : 12,
                                padding: isMobile ? '8px 10px' : '10px 12px',
                                background: 'linear-gradient(135deg, #fafafa 0%, #f5f5f5 100%)',
                                borderLeft: '3px solid #8c8c8c',
                                borderRadius: isMobile ? 4 : 6,
                                fontSize: isMobile ? 12 : 13,
                                color: '#262626',
                                lineHeight: '1.6'
                              }}>
                                <div style={{
                                  fontWeight: 600,
                                  color: '#595959',
                                  marginBottom: isMobile ? 4 : 6,
                                  fontSize: isMobile ? 12 : 13
                                }}>
                                  📝 大纲内容
                                </div>
                                <div style={{
                                  padding: isMobile ? '6px 8px' : '6px 10px',
                                  background: '#ffffff',
                                  border: '1px solid #d9d9d9',
                                  borderRadius: 4,
                                  fontSize: isMobile ? 12 : 13,
                                  color: '#262626',
                                  lineHeight: '1.6'
                                }}>
                                  {item.content}
                                </div>
                              </div>
                              
                              {/* ✨ 涉及角色展示 - 优化版（支持角色/组织分类显示） */}
                              {characterNames.length > 0 && (
                                <div style={{
                                  marginTop: isMobile ? 10 : 12,
                                  padding: isMobile ? '8px 10px' : '10px 12px',
                                  background: 'linear-gradient(135deg, #f5f3ff 0%, #faf5ff 100%)',
                                  borderLeft: '3px solid #9333ea',
                                  borderRadius: isMobile ? 4 : 6
                                }}>
                                  <div style={{
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: isMobile ? 6 : 8,
                                    marginBottom: isMobile ? 6 : 8
                                  }}>
                                    <span style={{
                                      fontSize: isMobile ? 12 : 13,
                                      fontWeight: 600,
                                      color: '#7c3aed',
                                      display: 'flex',
                                      alignItems: 'center',
                                      gap: 4
                                    }}>
                                      👥 涉及角色
                                      <Tag
                                        color="purple"
                                        style={{
                                          margin: 0,
                                          fontSize: 10,
                                          borderRadius: 10,
                                          padding: '0 6px'
                                        }}
                                      >
                                        {characterNames.length}
                                      </Tag>
                                    </span>
                                  </div>
                                  <Space wrap size={[4, 4]}>
                                    {characterNames.map((name, idx) => (
                                      <Tag
                                        key={idx}
                                        color="purple"
                                        style={{
                                          margin: 0,
                                          borderRadius: 4,
                                          padding: isMobile ? '2px 8px' : '3px 10px',
                                          fontSize: isMobile ? 11 : 12,
                                          fontWeight: 500,
                                          border: '1px solid #e9d5ff',
                                          background: '#ffffff',
                                          color: '#7c3aed',
                                          whiteSpace: 'normal',
                                          wordBreak: 'break-word',
                                          height: 'auto',
                                          lineHeight: '1.5'
                                        }}
                                      >
                                        {name}
                                      </Tag>
                                    ))}
                                  </Space>
                                </div>
                              )}
                              
                              {/* 🏛️ 涉及组织展示 */}
                              {organizationNames.length > 0 && (
                                <div style={{
                                  marginTop: isMobile ? 10 : 12,
                                  padding: isMobile ? '8px 10px' : '10px 12px',
                                  background: 'linear-gradient(135deg, #fff7ed 0%, #fffbeb 100%)',
                                  borderLeft: '3px solid #ea580c',
                                  borderRadius: isMobile ? 4 : 6
                                }}>
                                  <div style={{
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: isMobile ? 6 : 8,
                                    marginBottom: isMobile ? 6 : 8
                                  }}>
                                    <span style={{
                                      fontSize: isMobile ? 12 : 13,
                                      fontWeight: 600,
                                      color: '#ea580c',
                                      display: 'flex',
                                      alignItems: 'center',
                                      gap: 4
                                    }}>
                                      🏛️ 涉及组织
                                      <Tag
                                        color="orange"
                                        style={{
                                          margin: 0,
                                          fontSize: 10,
                                          borderRadius: 10,
                                          padding: '0 6px'
                                        }}
                                      >
                                        {organizationNames.length}
                                      </Tag>
                                    </span>
                                  </div>
                                  <Space wrap size={[4, 4]}>
                                    {organizationNames.map((name, idx) => (
                                      <Tag
                                        key={idx}
                                        color="orange"
                                        style={{
                                          margin: 0,
                                          borderRadius: 4,
                                          padding: isMobile ? '2px 8px' : '3px 10px',
                                          fontSize: isMobile ? 11 : 12,
                                          fontWeight: 500,
                                          border: '1px solid #fed7aa',
                                          background: '#ffffff',
                                          color: '#ea580c',
                                          whiteSpace: 'normal',
                                          wordBreak: 'break-word',
                                          height: 'auto',
                                          lineHeight: '1.5'
                                        }}
                                      >
                                        {name}
                                      </Tag>
                                    ))}
                                  </Space>
                                </div>
                              )}
                              
                              {/* ✨ 场景信息展示 - 优化版（支持折叠，最多显示3个） */}
                              {structureData.scenes && structureData.scenes.length > 0 ? (() => {
                                const isExpanded = scenesExpandStatus[item.id] || false;
                                const maxVisibleScenes = 4;
                                const hasMoreScenes = structureData.scenes!.length > maxVisibleScenes;
                                const visibleScenes = isExpanded ? structureData.scenes : structureData.scenes!.slice(0, maxVisibleScenes);
                                
                                return (
                                  <div style={{
                                    marginTop: isMobile ? 10 : 12,
                                    padding: isMobile ? '8px 10px' : '10px 12px',
                                    background: 'linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%)',
                                    borderLeft: '3px solid #0ea5e9',
                                    borderRadius: isMobile ? 4 : 6
                                  }}>
                                    <div style={{
                                      display: 'flex',
                                      alignItems: 'center',
                                      justifyContent: 'space-between',
                                      marginBottom: isMobile ? 6 : 8,
                                      flexWrap: isMobile ? 'wrap' : 'nowrap',
                                      gap: isMobile ? 4 : 0
                                    }}>
                                      <span style={{
                                        fontSize: isMobile ? 12 : 13,
                                        fontWeight: 600,
                                        color: '#0284c7',
                                        display: 'flex',
                                        alignItems: 'center',
                                        gap: 4
                                      }}>
                                        🎬 场景设定
                                        <Tag
                                          color="cyan"
                                          style={{
                                            margin: 0,
                                            fontSize: 10,
                                            borderRadius: 10,
                                            padding: '0 6px'
                                          }}
                                        >
                                          {structureData.scenes!.length}
                                        </Tag>
                                      </span>
                                      {hasMoreScenes && (
                                        <Button
                                          type="text"
                                          size="small"
                                          onClick={() => setScenesExpandStatus(prev => ({
                                            ...prev,
                                            [item.id]: !isExpanded
                                          }))}
                                          style={{
                                            fontSize: isMobile ? 10 : 11,
                                            height: isMobile ? 20 : 22,
                                            padding: isMobile ? '0 6px' : '0 8px',
                                            color: '#0284c7'
                                          }}
                                        >
                                          {isExpanded ? '收起 ▲' : `展开 (${structureData.scenes!.length - maxVisibleScenes}+) ▼`}
                                        </Button>
                                      )}
                                    </div>
                                    {/* 使用grid布局，移动端一列，桌面端两列 */}
                                    <div style={{
                                      display: 'grid',
                                      gridTemplateColumns: isMobile ? '1fr' : 'repeat(auto-fill, minmax(280px, 1fr))',
                                      gap: isMobile ? 6 : 8,
                                      width: '100%',
                                      minWidth: 0  // 防止grid子元素溢出
                                    }}>
                                      {visibleScenes!.map((scene, idx) => {
                                      // 判断是字符串还是对象
                                      if (typeof scene === 'string') {
                                        // 字符串格式：简洁卡片
                                        return (
                                          <div
                                            key={idx}
                                            style={{
                                              padding: isMobile ? '6px 8px' : '8px 10px',
                                              background: '#ffffff',
                                              border: '1px solid #bae6fd',
                                              borderRadius: isMobile ? 4 : 6,
                                              fontSize: isMobile ? 11 : 12,
                                              color: '#0c4a6e',
                                              display: 'flex',
                                              alignItems: 'flex-start',
                                              gap: isMobile ? 6 : 8,
                                              transition: 'all 0.2s ease',
                                              cursor: 'default',
                                              width: '100%',
                                              minWidth: 0,
                                              boxSizing: 'border-box'
                                            }}
                                            onMouseEnter={(e) => {
                                              if (!isMobile) {
                                                e.currentTarget.style.borderColor = '#0ea5e9';
                                                e.currentTarget.style.boxShadow = '0 2px 8px rgba(14, 165, 233, 0.15)';
                                              }
                                            }}
                                            onMouseLeave={(e) => {
                                              if (!isMobile) {
                                                e.currentTarget.style.borderColor = '#bae6fd';
                                                e.currentTarget.style.boxShadow = 'none';
                                              }
                                            }}
                                          >
                                            <Tag
                                              color="cyan"
                                              style={{
                                                margin: 0,
                                                fontSize: 10,
                                                borderRadius: 4,
                                                flexShrink: 0
                                              }}
                                            >
                                              {idx + 1}
                                            </Tag>
                                            <span style={{
                                              flex: 1,
                                              lineHeight: '1.6',
                                              overflow: 'hidden',
                                              textOverflow: 'ellipsis',
                                              whiteSpace: 'nowrap'
                                            }}>{scene}</span>
                                          </div>
                                        );
                                      } else {
                                        // 对象格式：详细卡片
                                        return (
                                          <div
                                            key={idx}
                                            style={{
                                              padding: isMobile ? '8px 10px' : '10px 12px',
                                              background: '#ffffff',
                                              border: '1px solid #bae6fd',
                                              borderRadius: isMobile ? 4 : 6,
                                              fontSize: isMobile ? 11 : 12,
                                              transition: 'all 0.2s ease',
                                              cursor: 'default',
                                              width: '100%',
                                              minWidth: 0,
                                              boxSizing: 'border-box'
                                            }}
                                            onMouseEnter={(e) => {
                                              if (!isMobile) {
                                                e.currentTarget.style.borderColor = '#0ea5e9';
                                                e.currentTarget.style.boxShadow = '0 2px 8px rgba(14, 165, 233, 0.15)';
                                              }
                                            }}
                                            onMouseLeave={(e) => {
                                              if (!isMobile) {
                                                e.currentTarget.style.borderColor = '#bae6fd';
                                                e.currentTarget.style.boxShadow = 'none';
                                              }
                                            }}
                                          >
                                            <div style={{
                                              display: 'flex',
                                              alignItems: 'center',
                                              gap: isMobile ? 6 : 8,
                                              marginBottom: isMobile ? 4 : 6,
                                              flexWrap: 'wrap'
                                            }}>
                                              <Tag
                                                color="cyan"
                                                style={{
                                                  margin: 0,
                                                  fontSize: 10,
                                                  borderRadius: 4
                                                }}
                                              >
                                                场景{idx + 1}
                                              </Tag>
                                              <span style={{
                                                fontWeight: 600,
                                                color: '#0c4a6e',
                                                fontSize: isMobile ? 12 : 13,
                                                flex: 1,
                                                overflow: 'hidden',
                                                textOverflow: 'ellipsis',
                                                whiteSpace: 'nowrap'
                                              }}>
                                                📍 {scene.location}
                                              </span>
                                            </div>
                                            {scene.characters && scene.characters.length > 0 && (
                                              <div style={{
                                                fontSize: isMobile ? 10 : 11,
                                                color: '#64748b',
                                                marginBottom: 4,
                                                paddingLeft: isMobile ? 2 : 4,
                                                overflow: 'hidden',
                                                textOverflow: 'ellipsis',
                                                whiteSpace: 'nowrap'
                                              }}>
                                                <span style={{ fontWeight: 500 }}>👤 角色：</span>
                                                {scene.characters.join(' · ')}
                                              </div>
                                            )}
                                            {scene.purpose && (
                                              <div style={{
                                                fontSize: isMobile ? 10 : 11,
                                                color: '#64748b',
                                                paddingLeft: isMobile ? 2 : 4,
                                                lineHeight: '1.5',
                                                overflow: 'hidden',
                                                textOverflow: 'ellipsis',
                                                whiteSpace: 'nowrap'
                                              }}>
                                                <span style={{ fontWeight: 500 }}>🎯 目的：</span>
                                                {scene.purpose}
                                              </div>
                                            )}
                                          </div>
                                        );
                                      }
                                      })}
                                    </div>
                                  </div>
                                );
                              })() : null}
                            
                            {/* ✨ 关键事件展示 */}
                            {structureData.key_events && structureData.key_events.length > 0 && (
                              <div style={{
                                marginTop: 12,
                                padding: '10px 12px',
                                background: 'linear-gradient(135deg, #fff7ed 0%, #ffedd5 100%)',
                                borderLeft: '3px solid #f97316',
                                borderRadius: 6
                              }}>
                                <div style={{
                                  display: 'flex',
                                  alignItems: 'center',
                                  gap: 8,
                                  marginBottom: 8
                                }}>
                                  <span style={{
                                    fontSize: 13,
                                    fontWeight: 600,
                                    color: '#ea580c',
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: 4
                                  }}>
                                    ⚡ 关键事件
                                    <Tag
                                      color="orange"
                                      style={{
                                        margin: 0,
                                        fontSize: 11,
                                        borderRadius: 10,
                                        padding: '0 6px'
                                      }}
                                    >
                                      {structureData.key_events.length}
                                    </Tag>
                                  </span>
                                </div>
                                <Space direction="vertical" size={6} style={{ width: '100%' }}>
                                  {structureData.key_events.map((event, idx) => (
                                    <div
                                      key={idx}
                                      style={{
                                        padding: '6px 10px',
                                        background: '#ffffff',
                                        border: '1px solid #fed7aa',
                                        borderRadius: 4,
                                        fontSize: 12,
                                        color: '#9a3412',
                                        display: 'flex',
                                        alignItems: 'flex-start',
                                        gap: 8
                                      }}
                                    >
                                      <Tag
                                        color="orange"
                                        style={{
                                          margin: 0,
                                          fontSize: 11,
                                          borderRadius: 4,
                                          flexShrink: 0
                                        }}
                                      >
                                        {idx + 1}
                                      </Tag>
                                      <span style={{
                                        flex: 1,
                                        lineHeight: '1.6',
                                        overflow: 'hidden',
                                        textOverflow: 'ellipsis',
                                        whiteSpace: 'nowrap'
                                      }}>{event}</span>
                                    </div>
                                  ))}
                                </Space>
                              </div>
                            )}
                            
                            {/* ✨ 情节要点展示 (key_points) */}
                            {structureData.key_points && structureData.key_points.length > 0 && (
                              <div style={{
                                marginTop: 12,
                                padding: '10px 12px',
                                background: 'linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%)',
                                borderLeft: '3px solid #22c55e',
                                borderRadius: 6
                              }}>
                                <div style={{
                                  display: 'flex',
                                  alignItems: 'center',
                                  gap: 8,
                                  marginBottom: 8
                                }}>
                                  <span style={{
                                    fontSize: 13,
                                    fontWeight: 600,
                                    color: '#15803d',
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: 4
                                  }}>
                                    💡 情节要点
                                    <Tag
                                      color="green"
                                      style={{
                                        margin: 0,
                                        fontSize: 11,
                                        borderRadius: 10,
                                        padding: '0 6px'
                                      }}
                                    >
                                      {structureData.key_points.length}
                                    </Tag>
                                  </span>
                                </div>
                                {/* 使用grid布局，移动端一列，桌面端两列 */}
                                <div style={{
                                  display: 'grid',
                                  gridTemplateColumns: isMobile ? '1fr' : 'repeat(auto-fill, minmax(280px, 1fr))',
                                  gap: isMobile ? 6 : 8,
                                  width: '100%',
                                  minWidth: 0
                                }}>
                                  {structureData.key_points.map((point, idx) => (
                                    <div
                                      key={idx}
                                      style={{
                                        padding: isMobile ? '6px 8px' : '8px 10px',
                                        background: '#ffffff',
                                        border: '1px solid #bbf7d0',
                                        borderRadius: isMobile ? 4 : 6,
                                        fontSize: isMobile ? 11 : 12,
                                        color: '#166534',
                                        display: 'flex',
                                        alignItems: 'flex-start',
                                        gap: isMobile ? 6 : 8,
                                        transition: 'all 0.2s ease',
                                        cursor: 'default',
                                        width: '100%',
                                        minWidth: 0,
                                        boxSizing: 'border-box'
                                      }}
                                      onMouseEnter={(e) => {
                                        if (!isMobile) {
                                          e.currentTarget.style.borderColor = '#22c55e';
                                          e.currentTarget.style.boxShadow = '0 2px 8px rgba(34, 197, 94, 0.15)';
                                        }
                                      }}
                                      onMouseLeave={(e) => {
                                        if (!isMobile) {
                                          e.currentTarget.style.borderColor = '#bbf7d0';
                                          e.currentTarget.style.boxShadow = 'none';
                                        }
                                      }}
                                    >
                                      <Tag
                                        color="green"
                                        style={{
                                          margin: 0,
                                          fontSize: 10,
                                          borderRadius: 4,
                                          flexShrink: 0
                                        }}
                                      >
                                        {idx + 1}
                                      </Tag>
                                      <span style={{
                                        flex: 1,
                                        lineHeight: '1.6',
                                        overflow: 'hidden',
                                        textOverflow: 'ellipsis',
                                        whiteSpace: 'nowrap'
                                      }}>{point}</span>
                                    </div>
                                  ))}
                                </div>
                              </div>
                            )}
                            
                            {/* ✨ 情感基调展示 (emotion) */}
                            {structureData.emotion && (
                              <div style={{
                                marginTop: 12,
                                padding: '10px 12px',
                                background: 'linear-gradient(135deg, #fef3c7 0%, #fde68a 100%)',
                                borderLeft: '3px solid #f59e0b',
                                borderRadius: 6,
                                display: 'flex',
                                alignItems: 'center',
                                gap: 8
                              }}>
                                <span style={{
                                  fontSize: 13,
                                  fontWeight: 600,
                                  color: '#b45309'
                                }}>
                                  💫 情感基调：
                                </span>
                                <Tag
                                  color="gold"
                                  style={{
                                    margin: 0,
                                    fontSize: 12,
                                    padding: '2px 12px',
                                    borderRadius: 12,
                                    background: '#ffffff',
                                    border: '1px solid #fbbf24',
                                    color: '#b45309'
                                  }}
                                >
                                  {structureData.emotion}
                                </Tag>
                              </div>
                            )}
                            
                            {/* ✨ 叙事目标展示 (goal) */}
                            {structureData.goal && (
                              <div style={{
                                marginTop: 12,
                                padding: '10px 12px',
                                background: 'linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%)',
                                borderLeft: '3px solid #3b82f6',
                                borderRadius: 6
                              }}>
                                <div style={{
                                  fontSize: 13,
                                  fontWeight: 600,
                                  color: '#1e40af',
                                  marginBottom: 6
                                }}>
                                  🎯 叙事目标
                                </div>
                                <div style={{
                                  fontSize: 12,
                                  color: '#1e3a8a',
                                  lineHeight: '1.6',
                                  padding: '6px 10px',
                                  background: '#ffffff',
                                  border: '1px solid #93c5fd',
                                  borderRadius: 4,
                                  overflow: 'hidden',
                                  textOverflow: 'ellipsis',
                                  whiteSpace: 'nowrap'
                                }}>
                                  {structureData.goal}
                                </div>
                              </div>
                            )}
                          </div>
                        }
                      />
                        
                        {/* 操作按钮区域 - 在卡片内部 */}
                        <div style={{
                          marginTop: 16,
                          paddingTop: 12,
                          borderTop: '1px solid #f0f0f0',
                          display: 'flex',
                          justifyContent: 'flex-end',
                          gap: 8
                        }}>
                          {currentProject?.outline_mode === 'one-to-many' && (
                            <Button
                              icon={<BranchesOutlined />}
                              onClick={() => handleExpandOutline(item.id, item.title)}
                              loading={isExpanding}
                              size={isMobile ? 'middle' : 'small'}
                            >
                              展开
                            </Button>
                          )}
                          <Button
                            icon={<EditOutlined />}
                            onClick={() => handleOpenEditModal(item.id)}
                            size={isMobile ? 'middle' : 'small'}
                          >
                            编辑
                          </Button>
                          <Popconfirm
                            title="确定删除这条大纲吗？"
                            onConfirm={() => handleDeleteOutline(item.id)}
                            okText="确定"
                            cancelText="取消"
                          >
                            <Button
                              danger
                              icon={<DeleteOutlined />}
                              size={isMobile ? 'middle' : 'small'}
                            >
                              删除
                            </Button>
                          </Popconfirm>
                        </div>
                      </Card>
                    </List.Item>
                  );
                }}
              />
          )}
        </div>
      </div>
    </>
  );
}
