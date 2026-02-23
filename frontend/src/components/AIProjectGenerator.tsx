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
}

interface WorldBuildingResult {
  project_id: string;
  time_period: string;
  location: string;
  atmosphere: string;
  rules: string;
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
  // 仅需要 setter（用于记录当前创建/重试出来的项目ID），不需要读取 state 值
  const [, setProjectId] = useState<string>('');

  // SSE流式进度状态
  const [progress, setProgress] = useState(0);
  const [progressMessage, setProgressMessage] = useState('');
  const [errorDetails, setErrorDetails] = useState<string>('');
  const [generationSteps, setGenerationSteps] = useState<GenerationSteps>({
    worldBuilding: 'pending'
  });

  // 保存生成数据，用于重试
  const [generationData, setGenerationData] = useState<GenerationConfig | null>(null);
  // 仅需 setter（用于调试/未来扩展），当前不读取
  const [, setWorldBuildingResult] = useState<WorldBuildingResult | null>(null);

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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [config, resumeProjectId]);

  // 恢复项目：新策略下只要项目存在即可直接进入（不再强制继续生成职业/角色/大纲）
  const handleResumeGenerate = async (_data: GenerationConfig, projectIdParam: string) => {
    try {
      setLoading(true);
      setProgress(0);
      setProgressMessage('正在进入项目...');
      setErrorDetails('');
      setProjectId(projectIdParam);

      // 校验项目是否存在（避免跳转到不存在的项目）
      const response = await fetch(`/api/projects/${projectIdParam}`, {
        credentials: 'include'
      });
      if (!response.ok) {
        throw new Error('获取项目信息失败');
      }

      setProgress(100);
      setProgressMessage('世界观初始化完成，正在跳转...');
      setLoading(false);

      onComplete(projectIdParam);
      setTimeout(() => {
        navigate(`/project/${projectIdParam}`);
      }, 300);
    } catch (error) {
      const apiError = error as ApiError;
      const errorMsg = apiError.response?.data?.detail || apiError.message || '未知错误';
      console.error('进入项目失败:', errorMsg);
      setErrorDetails(errorMsg);
      message.error('进入项目失败：' + errorMsg);
      setLoading(false);
    }
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
            // 直接使用后端返回的进度值
            setProgress(prog);
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

      // ✅ 新策略：向导只生成世界观（职业/角色/大纲在项目内按需生成）
      setProgress(100);
      setProgressMessage('项目创建完成！正在跳转...');
      message.success('项目创建成功！正在进入项目...');
      clearStorage();
      setLoading(false);

      // 调用完成回调
      onComplete(createdProjectId);

      // 延迟1秒后自动跳转到项目详情页
      setTimeout(() => {
        navigate(`/project/${createdProjectId}`);
      }, 500);

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
      } else {
        // 仅世界观生成：理论上不会进入这里
        setLoading(false);
      }
    } catch (error) {
      console.error('智能重试失败:', error);
      const errorMessage = error instanceof Error ? error.message : '未知错误';
      message.error('重试失败：' + errorMessage);
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
          // 直接使用后端返回的进度值
          setProgress(prog);
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

    // ✅ 新策略：重试只生成世界观，生成完成后直接进入项目
    const createdProjectId = worldResult.project_id;
    clearStorage();
    setProgress(100);
    setProgressMessage('项目创建完成！正在跳转...');
    message.success('项目创建成功！正在进入项目...');
    setLoading(false);

    onComplete(createdProjectId);
    setTimeout(() => {
      navigate(`/project/${createdProjectId}`);
    }, 500);
  };

  // 获取步骤状态图标和颜色
  const getStepStatus = (step: GenerationStep) => {
    if (step === 'completed') return { icon: <CheckCircleOutlined />, color: 'var(--color-success)' };
    if (step === 'processing') return { icon: <LoadingOutlined />, color: 'var(--color-primary)' };
    if (step === 'error') return { icon: '✗', color: 'var(--color-error)' };
    return { icon: '○', color: 'var(--color-text-quaternary)' };
  };

  const hasError = generationSteps.worldBuilding === 'error';

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
        正在为《{config.title}》生成世界观
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
        {hasError ? '生成过程中出现错误，请点击重试按钮重新生成' : '请耐心等待，AI正在为您生成世界观...'}
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
