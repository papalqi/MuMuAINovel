import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Card, Button, Space, Typography, message, Progress } from 'antd';
import { CheckCircleOutlined, LoadingOutlined } from '@ant-design/icons';
import { wizardStreamApi } from '../services/api';
import type { ApiError } from '../types';

const { Title, Paragraph, Text } = Typography;

export interface GenerationConfig {
  title: string;
  description: string;
  theme: string;
  genre: string | string[];
  narrative_perspective: string;
  target_words: number;
  chapter_count: number;
  character_count: number;
  outline_mode?: 'one-to-one' | 'one-to-many';  // 大纲章节模式
}

interface AIProjectGeneratorProps {
  config: GenerationConfig;
  storagePrefix: 'wizard' | 'inspiration';
  onComplete: (projectId: string) => void;
  onBack?: () => void;
  isMobile?: boolean;
  resumeProjectId?: string;
}

type GenerationStep = 'pending' | 'processing' | 'completed' | 'error';

interface GenerationSteps {
  worldBuilding: GenerationStep;
  characters: GenerationStep;
  outline: GenerationStep;
}

export const AIProjectGenerator: React.FC<AIProjectGeneratorProps> = ({
  config,
  storagePrefix,
  onComplete,
  isMobile = false,
  resumeProjectId
}) => {
  const navigate = useNavigate();

  // 状态管理
  const [loading, setLoading] = useState(false);
  const [projectId, setProjectId] = useState<string>('');

  // SSE流式进度状态
  const [progress, setProgress] = useState(0);
  const [progressMessage, setProgressMessage] = useState('');
  const [errorDetails, setErrorDetails] = useState<string>('');
  const [generationSteps, setGenerationSteps] = useState<GenerationSteps>({
    worldBuilding: 'pending',
    characters: 'pending',
    outline: 'pending'
  });

  // 保存生成数据，用于重试
  const [generationData, setGenerationData] = useState<GenerationConfig | null>(null);
  // 保存世界观生成结果，用于后续步骤
  const [worldBuildingResult, setWorldBuildingResult] = useState<any>(null);

  // LocalStorage 键名
  const storageKeys = {
    projectId: `${storagePrefix}_project_id`,
    generationData: `${storagePrefix}_generation_data`,
    currentStep: `${storagePrefix}_current_step`
  };

  // 保存进度到localStorage
  const saveProgress = (projectId: string, data: GenerationConfig, step: string) => {
    try {
      localStorage.setItem(storageKeys.projectId, projectId);
      localStorage.setItem(storageKeys.generationData, JSON.stringify(data));
      localStorage.setItem(storageKeys.currentStep, step);
    } catch (error) {
      console.error('保存进度失败:', error);
    }
  };

  // 清理localStorage
  const clearStorage = () => {
    localStorage.removeItem(storageKeys.projectId);
    localStorage.removeItem(storageKeys.generationData);
    localStorage.removeItem(storageKeys.currentStep);
  };

  // 开始自动化生成流程
  useEffect(() => {
    if (config) {
      if (resumeProjectId) {
        // 恢复生成模式
        handleResumeGenerate(config, resumeProjectId);
      } else {
        // 新建项目模式
        handleAutoGenerate(config);
      }
    }
  }, [config, resumeProjectId]);

  // 恢复未完成项目的生成
  const handleResumeGenerate = async (data: GenerationConfig, projectIdParam: string) => {
    try {
      setLoading(true);
      setProgress(0);
      setProgressMessage('检查项目状态...');
      setErrorDetails('');
      setGenerationData(data);
      setProjectId(projectIdParam);

      // 获取项目信息,判断当前完成到哪一步
      const response = await fetch(`/api/projects/${projectIdParam}`, {
        credentials: 'include'
      });
      if (!response.ok) {
        throw new Error('获取项目信息失败');
      }
      const project = await response.json();
      const wizardStep = project.wizard_step || 0;

      // 根据wizard_step判断从哪里继续
      if (wizardStep === 0) {
        // 从世界观开始
        message.info('从世界观步骤开始生成...');
        setGenerationSteps({ worldBuilding: 'processing', characters: 'pending', outline: 'pending' });
        await resumeFromWorldBuilding(data);
      } else if (wizardStep === 1) {
        // 世界观已完成,从角色开始
        message.info('世界观已完成,从角色步骤继续...');
        setGenerationSteps({ worldBuilding: 'completed', characters: 'processing', outline: 'pending' });

        // 获取世界观数据
        const worldResult = {
          project_id: projectIdParam,
          time_period: project.world_time_period || '',
          location: project.world_location || '',
          atmosphere: project.world_atmosphere || '',
          rules: project.world_rules || ''
        };
        setWorldBuildingResult(worldResult);
        setProgress(33);

        await resumeFromCharacters(data, worldResult);
      } else if (wizardStep === 2) {
        // 世界观和角色已完成,从大纲开始
        message.info('世界观和角色已完成,从大纲步骤继续...');
        setGenerationSteps({ worldBuilding: 'completed', characters: 'completed', outline: 'processing' });
        setProgress(66);
        await resumeFromOutline(data, projectIdParam);
      } else {
        // 已全部完成
        message.success('项目已完成,正在跳转...');
        setProgress(100);
        onComplete(projectIdParam);
        setTimeout(() => {
          navigate(`/project/${projectIdParam}`);
        }, 1000);
      }
    } catch (error) {
      const apiError = error as ApiError;
      const errorMsg = apiError.response?.data?.detail || apiError.message || '未知错误';
      console.error('恢复生成失败:', errorMsg);
      setErrorDetails(errorMsg);
      message.error('恢复生成失败：' + errorMsg);
      setLoading(false);
    }
  };

  // 恢复:从世界观步骤开始
  const resumeFromWorldBuilding = async (data: GenerationConfig) => {
    const genreString = Array.isArray(data.genre) ? data.genre.join('、') : data.genre;

    const worldResult = await wizardStreamApi.generateWorldBuildingStream(
      {
        title: data.title,
        description: data.description,
        theme: data.theme,
        genre: genreString,
        narrative_perspective: data.narrative_perspective,
        target_words: data.target_words,
        chapter_count: data.chapter_count,
        character_count: data.character_count,
        outline_mode: data.outline_mode || 'one-to-many',  // 传递大纲模式
      },
      {
        onProgress: (msg, prog) => {
          setProgress(Math.floor(prog / 3));
          setProgressMessage(msg);
        },
        onResult: (result) => {
          setWorldBuildingResult(result);
          setGenerationSteps(prev => ({ ...prev, worldBuilding: 'completed' }));
        },
        onError: (error) => {
          console.error('世界观生成失败:', error);
          setErrorDetails(`世界观生成失败: ${error}`);
          setGenerationSteps(prev => ({ ...prev, worldBuilding: 'error' }));
          setLoading(false);
          throw new Error(error);
        },
        onComplete: () => {
          console.log('世界观生成完成');
        }
      }
    );

    await resumeFromCharacters(data, worldResult);
  };

  // 恢复:从角色步骤继续
  const resumeFromCharacters = async (data: GenerationConfig, worldResult: any) => {
    const genreString = Array.isArray(data.genre) ? data.genre.join('、') : data.genre;
    const pid = projectId || worldResult.project_id;

    setGenerationSteps(prev => ({ ...prev, characters: 'processing' }));
    setProgressMessage('正在生成角色...');

    await wizardStreamApi.generateCharactersStream(
      {
        project_id: pid,
        count: data.character_count,
        world_context: {
          time_period: worldResult.time_period || '',
          location: worldResult.location || '',
          atmosphere: worldResult.atmosphere || '',
          rules: worldResult.rules || '',
        },
        theme: data.theme,
        genre: genreString,
      },
      {
        onProgress: (msg, prog) => {
          setProgress(33 + Math.floor(prog / 3));
          setProgressMessage(msg);
        },
        onResult: (result) => {
          console.log(`成功生成${result.characters?.length || 0}个角色`);
          setGenerationSteps(prev => ({ ...prev, characters: 'completed' }));
        },
        onError: (error) => {
          console.error('角色生成失败:', error);
          setErrorDetails(`角色生成失败: ${error}`);
          setGenerationSteps(prev => ({ ...prev, characters: 'error' }));
          setLoading(false);
          throw new Error(error);
        },
        onComplete: () => {
          console.log('角色生成完成');
        }
      }
    );

    await resumeFromOutline(data, pid);
  };

  // 恢复:从大纲步骤继续
  const resumeFromOutline = async (data: GenerationConfig, pid: string) => {
    setGenerationSteps(prev => ({ ...prev, outline: 'processing' }));
    setProgressMessage('正在生成大纲...');

    await wizardStreamApi.generateCompleteOutlineStream(
      {
        project_id: pid,
        chapter_count: data.chapter_count,
        narrative_perspective: data.narrative_perspective,
        target_words: data.target_words,
      },
      {
        onProgress: (msg, prog) => {
          setProgress(66 + Math.floor(prog / 3));
          setProgressMessage(msg);
        },
        onResult: () => {
          console.log('大纲生成完成');
          setGenerationSteps(prev => ({ ...prev, outline: 'completed' }));
        },
        onError: (error) => {
          console.error('大纲生成失败:', error);
          setErrorDetails(`大纲生成失败: ${error}`);
          setGenerationSteps(prev => ({ ...prev, outline: 'error' }));
          setLoading(false);
          throw new Error(error);
        },
        onComplete: () => {
          console.log('大纲生成完成');
        }
      }
    );

    // 全部完成
    setProgress(100);
    setProgressMessage('项目创建完成！正在跳转...');
    message.success('项目创建成功！正在进入项目...');
    clearStorage();
    setLoading(false);

    onComplete(pid);
    setTimeout(() => {
      navigate(`/project/${pid}`);
    }, 1000);
  };

  // 自动化生成流程
  const handleAutoGenerate = async (data: GenerationConfig) => {
    try {
      setLoading(true);
      setProgress(0);
      setProgressMessage('开始创建项目...');
      setErrorDetails('');
      setGenerationData(data);
      saveProgress('', data, 'generating');

      const genreString = Array.isArray(data.genre) ? data.genre.join('、') : data.genre;

      // 步骤1: 生成世界观并创建项目
      setGenerationSteps(prev => ({ ...prev, worldBuilding: 'processing' }));
      setProgressMessage('正在生成世界观...');

      const worldResult = await wizardStreamApi.generateWorldBuildingStream(
        {
          title: data.title,
          description: data.description,
          theme: data.theme,
          genre: genreString,
          narrative_perspective: data.narrative_perspective,
          target_words: data.target_words,
          chapter_count: data.chapter_count,
          character_count: data.character_count,
          outline_mode: data.outline_mode || 'one-to-many',  // 传递大纲模式
        },
        {
          onProgress: (msg, prog) => {
            setProgress(Math.floor(prog / 3));
            setProgressMessage(msg);
          },
          onResult: (result) => {
            setProjectId(result.project_id);
            setWorldBuildingResult(result);
            setGenerationSteps(prev => ({ ...prev, worldBuilding: 'completed' }));
          },
          onError: (error) => {
            console.error('世界观生成失败:', error);
            setErrorDetails(`世界观生成失败: ${error}`);
            setGenerationSteps(prev => ({ ...prev, worldBuilding: 'error' }));
            setLoading(false);
            throw new Error(error);
          },
          onComplete: () => {
            console.log('世界观生成完成');
          }
        }
      );

      if (!worldResult?.project_id) {
        throw new Error('项目创建失败：未获取到项目ID');
      }

      const createdProjectId = worldResult.project_id;
      setProjectId(createdProjectId);
      setWorldBuildingResult(worldResult);
      saveProgress(createdProjectId, data, 'generating');

      // 步骤2: 生成角色
      setGenerationSteps(prev => ({ ...prev, characters: 'processing' }));
      setProgressMessage('正在生成角色...');

      await wizardStreamApi.generateCharactersStream(
        {
          project_id: createdProjectId,
          count: data.character_count,
          world_context: {
            time_period: worldResult.time_period || '',
            location: worldResult.location || '',
            atmosphere: worldResult.atmosphere || '',
            rules: worldResult.rules || '',
          },
          theme: data.theme,
          genre: genreString,
        },
        {
          onProgress: (msg, prog) => {
            setProgress(33 + Math.floor(prog / 3));
            setProgressMessage(msg);
          },
          onResult: (result) => {
            console.log(`成功生成${result.characters?.length || 0}个角色`);
            setGenerationSteps(prev => ({ ...prev, characters: 'completed' }));
          },
          onError: (error) => {
            console.error('角色生成失败:', error);
            setErrorDetails(`角色生成失败: ${error}`);
            setGenerationSteps(prev => ({ ...prev, characters: 'error' }));
            setLoading(false);
            throw new Error(error);
          },
          onComplete: () => {
            console.log('角色生成完成');
          }
        }
      );

      // 步骤3: 生成大纲
      setGenerationSteps(prev => ({ ...prev, outline: 'processing' }));
      setProgressMessage('正在生成大纲...');

      await wizardStreamApi.generateCompleteOutlineStream(
        {
          project_id: createdProjectId,
          chapter_count: data.chapter_count,
          narrative_perspective: data.narrative_perspective,
          target_words: data.target_words,
        },
        {
          onProgress: (msg, prog) => {
            setProgress(66 + Math.floor(prog / 3));
            setProgressMessage(msg);
          },
          onResult: () => {
            console.log('大纲生成完成');
            setGenerationSteps(prev => ({ ...prev, outline: 'completed' }));
          },
          onError: (error) => {
            console.error('大纲生成失败:', error);
            setErrorDetails(`大纲生成失败: ${error}`);
            setGenerationSteps(prev => ({ ...prev, outline: 'error' }));
            setLoading(false);
            throw new Error(error);
          },
          onComplete: () => {
            console.log('大纲生成完成');
          }
        }
      );

      // 全部完成 - 自动跳转到项目详情页
      setProgress(100);
      setProgressMessage('项目创建完成！正在跳转...');
      message.success('项目创建成功！正在进入项目...');
      clearStorage();

      // 调用完成回调
      onComplete(createdProjectId);

      // 延迟1秒后自动跳转到项目详情页
      setTimeout(() => {
        navigate(`/project/${createdProjectId}`);
      }, 1000);

    } catch (error) {
      const apiError = error as ApiError;
      const errorMsg = apiError.response?.data?.detail || apiError.message || '未知错误';
      console.error('创建项目失败:', errorMsg);
      setErrorDetails(errorMsg);
      message.error('创建项目失败：' + errorMsg);
      setLoading(false);
    }
  };

  // 智能重试：从失败的步骤继续生成
  const handleSmartRetry = async () => {
    if (!generationData) {
      message.warning('缺少生成数据');
      return;
    }

    setLoading(true);
    setErrorDetails('');

    try {
      if (generationSteps.worldBuilding === 'error') {
        message.info('从世界观步骤开始重新生成...');
        await retryFromWorldBuilding();
      } else if (generationSteps.characters === 'error') {
        message.info('从角色步骤继续生成...');
        await retryFromCharacters();
      } else if (generationSteps.outline === 'error') {
        message.info('从大纲步骤继续生成...');
        await retryFromOutline();
      }
    } catch (error: any) {
      console.error('智能重试失败:', error);
      message.error('重试失败：' + (error.message || '未知错误'));
      setLoading(false);
    }
  };

  // 从世界观步骤重新开始
  const retryFromWorldBuilding = async () => {
    if (!generationData) return;

    setGenerationSteps(prev => ({ ...prev, worldBuilding: 'processing' }));
    setProgressMessage('重新生成世界观...');

    const genreString = Array.isArray(generationData.genre) ? generationData.genre.join('、') : generationData.genre;

    const worldResult = await wizardStreamApi.generateWorldBuildingStream(
      {
        title: generationData.title,
        description: generationData.description,
        theme: generationData.theme,
        genre: genreString,
        narrative_perspective: generationData.narrative_perspective,
        target_words: generationData.target_words,
        chapter_count: generationData.chapter_count,
        character_count: generationData.character_count,
        outline_mode: generationData.outline_mode || 'one-to-many',  // 传递大纲模式
      },
      {
        onProgress: (msg, prog) => {
          setProgress(Math.floor(prog / 3));
          setProgressMessage(msg);
        },
        onResult: (result) => {
          setProjectId(result.project_id);
          setWorldBuildingResult(result);
          setGenerationSteps(prev => ({ ...prev, worldBuilding: 'completed' }));
        },
        onError: (error) => {
          console.error('世界观生成失败:', error);
          setErrorDetails(`世界观生成失败: ${error}`);
          setGenerationSteps(prev => ({ ...prev, worldBuilding: 'error' }));
          setLoading(false);
          throw new Error(error);
        },
        onComplete: () => {
          console.log('世界观重新生成完成');
        }
      }
    );

    if (!worldResult?.project_id) {
      throw new Error('项目创建失败：未获取到项目ID');
    }

    await continueFromCharacters(worldResult);
  };

  // 从角色步骤继续
  const retryFromCharacters = async () => {
    if (!generationData || !projectId || !worldBuildingResult) {
      message.warning('缺少必要数据，无法从角色步骤继续');
      setLoading(false);
      return;
    }

    setGenerationSteps(prev => ({ ...prev, characters: 'processing' }));
    setProgressMessage('重新生成角色...');

    const genreString = Array.isArray(generationData.genre) ? generationData.genre.join('、') : generationData.genre;

    await wizardStreamApi.generateCharactersStream(
      {
        project_id: projectId,
        count: generationData.character_count,
        world_context: {
          time_period: worldBuildingResult.time_period || '',
          location: worldBuildingResult.location || '',
          atmosphere: worldBuildingResult.atmosphere || '',
          rules: worldBuildingResult.rules || '',
        },
        theme: generationData.theme,
        genre: genreString,
      },
      {
        onProgress: (msg, prog) => {
          setProgress(33 + Math.floor(prog / 3));
          setProgressMessage(msg);
        },
        onResult: (result) => {
          console.log(`成功生成${result.characters?.length || 0}个角色`);
          setGenerationSteps(prev => ({ ...prev, characters: 'completed' }));
        },
        onError: (error) => {
          console.error('角色生成失败:', error);
          setErrorDetails(`角色生成失败: ${error}`);
          setGenerationSteps(prev => ({ ...prev, characters: 'error' }));
          setLoading(false);
          throw new Error(error);
        },
        onComplete: () => {
          console.log('角色重新生成完成');
        }
      }
    );

    await continueFromOutline();
  };

  // 从大纲步骤继续
  const retryFromOutline = async () => {
    if (!generationData || !projectId) {
      message.warning('缺少必要数据，无法从大纲步骤继续');
      setLoading(false);
      return;
    }

    setGenerationSteps(prev => ({ ...prev, outline: 'processing' }));
    setProgressMessage('重新生成大纲...');

    await wizardStreamApi.generateCompleteOutlineStream(
      {
        project_id: projectId,
        chapter_count: generationData.chapter_count,
        narrative_perspective: generationData.narrative_perspective,
        target_words: generationData.target_words,
      },
      {
        onProgress: (msg, prog) => {
          setProgress(66 + Math.floor(prog / 3));
          setProgressMessage(msg);
        },
        onResult: () => {
          console.log('大纲生成完成');
          setGenerationSteps(prev => ({ ...prev, outline: 'completed' }));
        },
        onError: (error) => {
          console.error('大纲生成失败:', error);
          setErrorDetails(`大纲生成失败: ${error}`);
          setGenerationSteps(prev => ({ ...prev, outline: 'error' }));
          setLoading(false);
          throw new Error(error);
        },
        onComplete: () => {
          console.log('大纲重新生成完成');
        }
      }
    );

    setProgress(100);
    setProgressMessage('项目创建完成！正在跳转...');
    message.success('项目创建成功！正在进入项目...');
    setLoading(false);

    // 调用完成回调
    if (projectId) {
      onComplete(projectId);

      // 延迟1秒后自动跳转到项目详情页
      setTimeout(() => {
        navigate(`/project/${projectId}`);
      }, 1000);
    }
  };

  // 从角色步骤开始的完整流程
  const continueFromCharacters = async (worldResult: any) => {
    if (!generationData || !worldResult?.project_id) return;

    const genreString = Array.isArray(generationData.genre) ? generationData.genre.join('、') : generationData.genre;

    setGenerationSteps(prev => ({ ...prev, characters: 'processing' }));
    setProgressMessage('正在生成角色...');

    await wizardStreamApi.generateCharactersStream(
      {
        project_id: worldResult.project_id,
        count: generationData.character_count,
        world_context: {
          time_period: worldResult.time_period || '',
          location: worldResult.location || '',
          atmosphere: worldResult.atmosphere || '',
          rules: worldResult.rules || '',
        },
        theme: generationData.theme,
        genre: genreString,
      },
      {
        onProgress: (msg, prog) => {
          setProgress(33 + Math.floor(prog / 3));
          setProgressMessage(msg);
        },
        onResult: (result) => {
          console.log(`成功生成${result.characters?.length || 0}个角色`);
          setGenerationSteps(prev => ({ ...prev, characters: 'completed' }));
        },
        onError: (error) => {
          console.error('角色生成失败:', error);
          setErrorDetails(`角色生成失败: ${error}`);
          setGenerationSteps(prev => ({ ...prev, characters: 'error' }));
          setLoading(false);
          throw new Error(error);
        },
        onComplete: () => {
          console.log('角色生成完成');
        }
      }
    );

    await continueFromOutline();
  };

  // 从大纲步骤开始的完整流程
  const continueFromOutline = async () => {
    if (!generationData || !projectId) return;

    setGenerationSteps(prev => ({ ...prev, outline: 'processing' }));
    setProgressMessage('正在生成大纲...');

    await wizardStreamApi.generateCompleteOutlineStream(
      {
        project_id: projectId,
        chapter_count: generationData.chapter_count,
        narrative_perspective: generationData.narrative_perspective,
        target_words: generationData.target_words,
      },
      {
        onProgress: (msg, prog) => {
          setProgress(66 + Math.floor(prog / 3));
          setProgressMessage(msg);
        },
        onResult: () => {
          console.log('大纲生成完成');
          setGenerationSteps(prev => ({ ...prev, outline: 'completed' }));
        },
        onError: (error) => {
          console.error('大纲生成失败:', error);
          setErrorDetails(`大纲生成失败: ${error}`);
          setGenerationSteps(prev => ({ ...prev, outline: 'error' }));
          setLoading(false);
          throw new Error(error);
        },
        onComplete: () => {
          console.log('大纲生成完成');
        }
      }
    );

    setProgress(100);
    setProgressMessage('项目创建完成！正在跳转...');
    message.success('项目创建成功！正在进入项目...');
    setLoading(false);

    // 调用完成回调
    if (projectId) {
      onComplete(projectId);

      // 延迟1秒后自动跳转到项目详情页
      setTimeout(() => {
        navigate(`/project/${projectId}`);
      }, 1000);
    }
  };


  // 获取步骤状态图标和颜色
  const getStepStatus = (step: GenerationStep) => {
    if (step === 'completed') return { icon: <CheckCircleOutlined />, color: 'var(--color-success)' };
    if (step === 'processing') return { icon: <LoadingOutlined />, color: 'var(--color-primary)' };
    if (step === 'error') return { icon: '✗', color: 'var(--color-error)' };
    return { icon: '○', color: 'var(--color-text-quaternary)' };
  };

  const hasError = generationSteps.worldBuilding === 'error' ||
    generationSteps.characters === 'error' ||
    generationSteps.outline === 'error';

  // 渲染生成进度页面
  const renderGenerating = () => (
    <div style={{
      textAlign: 'center',
      padding: isMobile ? '32px 16px' : '40px 20px',
      maxWidth: '100%',
      overflow: 'hidden'
    }}>
      <Title
        level={isMobile ? 4 : 3}
        style={{
          marginBottom: 32,
          color: 'var(--color-text-primary)',
          wordBreak: 'break-word',
          whiteSpace: 'normal',
          overflowWrap: 'break-word'
        }}
      >
        正在为《{config.title}》生成内容
      </Title>

      <Card style={{ marginBottom: 24, maxWidth: '100%' }}>
        <Progress
          percent={progress}
          status={hasError ? 'exception' : (progress === 100 ? 'success' : 'active')}
          strokeColor={{
            '0%': 'var(--color-primary)',
            '100%': 'var(--color-primary-active)',
          }}
          style={{ marginBottom: 24 }}
        />

        <Paragraph
          style={{
            fontSize: isMobile ? 14 : 16,
            marginBottom: 32,
            color: hasError ? 'var(--color-error)' : 'var(--color-text-secondary)',
            wordBreak: 'break-word',
            whiteSpace: 'normal',
            overflowWrap: 'break-word'
          }}
        >
          {progressMessage}
        </Paragraph>

        {errorDetails && (
          <Card
            size="small"
            style={{
              marginBottom: 24,
              background: 'var(--color-error-bg)',
              borderColor: 'var(--color-error-border)',
              textAlign: 'left',
              maxWidth: '100%',
              overflow: 'hidden'
            }}
          >
            <Text strong style={{ color: 'var(--color-error)' }}>错误详情：</Text>
            <br />
            <Text
              style={{
                color: 'var(--color-text-secondary)',
                fontSize: 14,
                wordBreak: 'break-word',
                whiteSpace: 'normal',
                overflowWrap: 'break-word',
                display: 'block'
              }}
            >
              {errorDetails}
            </Text>
          </Card>
        )}

        <Space
          direction="vertical"
          size={16}
          style={{
            width: '100%',
            maxWidth: isMobile ? '100%' : 400,
            margin: '0 auto'
          }}
        >
          {[
            { key: 'worldBuilding', label: '生成世界观', step: generationSteps.worldBuilding },
            { key: 'characters', label: '生成角色', step: generationSteps.characters },
            { key: 'outline', label: '生成大纲', step: generationSteps.outline },
          ].map(({ key, label, step }) => {
            const status = getStepStatus(step);
            return (
              <div
                key={key}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  padding: isMobile ? '10px 12px' : '12px 20px',
                  background: step === 'processing' ? 'var(--color-info-bg)' : (step === 'error' ? 'var(--color-error-bg)' : 'var(--color-bg-layout)'),
                  borderRadius: 8,
                  border: `1px solid ${step === 'processing' ? 'var(--color-info-border)' : (step === 'error' ? 'var(--color-error-border)' : 'var(--color-border-secondary)')}`,
                  gap: '8px',
                  maxWidth: '100%',
                  overflow: 'hidden'
                }}
              >
                <Text
                  style={{
                    fontSize: isMobile ? 14 : 16,
                    fontWeight: step === 'processing' ? 600 : 400,
                    wordBreak: 'break-word',
                    whiteSpace: 'normal',
                    overflowWrap: 'break-word',
                    flex: 1,
                    textAlign: 'left'
                  }}
                >
                  {label}
                </Text>
                <span
                  style={{
                    fontSize: 20,
                    color: status.color,
                    flexShrink: 0
                  }}
                >
                  {status.icon}
                </span>
              </div>
            );
          })}
        </Space>
      </Card>

      <Paragraph
        type="secondary"
        style={{
          color: 'var(--color-text-secondary)',
          opacity: 0.9,
          wordBreak: 'break-word',
          whiteSpace: 'normal',
          overflowWrap: 'break-word',
          fontSize: isMobile ? 14 : 16
        }}
      >
        {hasError ? '生成过程中出现错误，请点击重试按钮重新生成' : '请耐心等待，AI正在为您精心创作...'}
      </Paragraph>

      {hasError && (
        <Space style={{ marginTop: 16 }}>
          <Button
            type="primary"
            size="large"
            onClick={handleSmartRetry}
            loading={loading}
            disabled={loading}
          >
            智能重试
          </Button>
        </Space>
      )}

    </div>
  );

  return renderGenerating();
};