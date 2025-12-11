import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Card, Form, Input, Button, Select, Slider, InputNumber, message, Space, Typography, Spin, Modal, Tooltip, Alert, Grid } from 'antd';
import { SettingOutlined, SaveOutlined, DeleteOutlined, ReloadOutlined, ArrowLeftOutlined, InfoCircleOutlined, CheckCircleOutlined, CloseCircleOutlined, ThunderboltOutlined } from '@ant-design/icons';
import { settingsApi } from '../services/api';
import type { SettingsUpdate } from '../types';

const { Title, Paragraph } = Typography;
const { Option } = Select;
const { useBreakpoint } = Grid;

export default function SettingsPage() {
  const navigate = useNavigate();
  const screens = useBreakpoint();
  const isMobile = !screens.md; // md断点是768px
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [initialLoading, setInitialLoading] = useState(true);
  const [hasSettings, setHasSettings] = useState(false);
  const [isDefaultSettings, setIsDefaultSettings] = useState(false);
  const [modelOptions, setModelOptions] = useState<Array<{ value: string; label: string; description: string }>>([]);
  const [fetchingModels, setFetchingModels] = useState(false);
  const [modelsFetched, setModelsFetched] = useState(false);
  const [testingApi, setTestingApi] = useState(false);
  const [testResult, setTestResult] = useState<{
    success: boolean;
    message: string;
    response_time_ms?: number;
    response_preview?: string;
    error?: string;
    error_type?: string;
    suggestions?: string[];
  } | null>(null);
  const [showTestResult, setShowTestResult] = useState(false);

  useEffect(() => {
    loadSettings();
  }, []);

  const loadSettings = async () => {
    setInitialLoading(true);
    try {
      const settings = await settingsApi.getSettings();
      form.setFieldsValue(settings);

      // 判断是否为默认设置（id='0'表示来自.env的默认配置）
      if (settings.id === '0' || !settings.id) {
        setIsDefaultSettings(true);
        setHasSettings(false);
      } else {
        setIsDefaultSettings(false);
        setHasSettings(true);
      }
    } catch (error: any) {
      // 如果404表示还没有设置，使用默认值
      if (error?.response?.status === 404) {
        setHasSettings(false);
        setIsDefaultSettings(true);
        form.setFieldsValue({
          api_provider: 'openai',
          api_base_url: 'https://api.openai.com/v1',
          llm_model: 'gpt-4',
          temperature: 0.7,
          max_tokens: 2000,
        });
      } else {
        message.error('加载设置失败');
      }
    } finally {
      setInitialLoading(false);
    }
  };

  const handleSave = async (values: SettingsUpdate) => {
    setLoading(true);
    try {
      await settingsApi.saveSettings(values);
      message.success('设置已保存');
      setHasSettings(true);
      setIsDefaultSettings(false);
    } catch (error) {
      message.error('保存设置失败');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    Modal.confirm({
      title: '重置设置',
      content: '确定要重置为默认值吗？',
      okText: '确定',
      cancelText: '取消',
      onOk: () => {
        form.setFieldsValue({
          api_provider: 'openai',
          api_key: '',
          api_base_url: 'https://api.openai.com/v1',
          llm_model: 'gpt-4',
          temperature: 0.7,
          max_tokens: 2000,
        });
        message.info('已重置为默认值，请点击保存');
      },
    });
  };

  const handleDelete = () => {
    Modal.confirm({
      title: '删除设置',
      content: '确定要删除所有设置吗？此操作不可恢复。',
      okText: '确定',
      cancelText: '取消',
      okType: 'danger',
      onOk: async () => {
        setLoading(true);
        try {
          await settingsApi.deleteSettings();
          message.success('设置已删除');
          setHasSettings(false);
          form.resetFields();
        } catch (error) {
          message.error('删除设置失败');
        } finally {
          setLoading(false);
        }
      },
    });
  };

  const apiProviders = [
    { value: 'openai', label: 'OpenAl Compatible', defaultUrl: 'https://api.openai.com/v1' },
    // { value: 'azure', label: 'Azure OpenAI', defaultUrl: 'https://YOUR-RESOURCE.openai.azure.com' },
    // { value: 'anthropic', label: 'Anthropic', defaultUrl: 'https://api.anthropic.com' },
    // { value: 'custom', label: '自定义', defaultUrl: '' },
  ];

  const handleProviderChange = (value: string) => {
    const provider = apiProviders.find(p => p.value === value);
    if (provider && provider.defaultUrl) {
      form.setFieldValue('api_base_url', provider.defaultUrl);
    }
    // 清空模型列表，需要重新获取
    setModelOptions([]);
    setModelsFetched(false);
  };

  const handleFetchModels = async (silent: boolean = false) => {
    const apiKey = form.getFieldValue('api_key');
    const apiBaseUrl = form.getFieldValue('api_base_url');
    const provider = form.getFieldValue('api_provider');

    if (!apiKey || !apiBaseUrl) {
      if (!silent) {
        message.warning('请先填写 API 密钥和 API 地址');
      }
      return;
    }

    setFetchingModels(true);
    try {
      const response = await settingsApi.getAvailableModels({
        api_key: apiKey,
        api_base_url: apiBaseUrl,
        provider: provider || 'openai'
      });

      setModelOptions(response.models);
      setModelsFetched(true);
      if (!silent) {
        message.success(`成功获取 ${response.count || response.models.length} 个可用模型`);
      }
    } catch (error: any) {
      const errorMsg = error?.response?.data?.detail || '获取模型列表失败';
      if (!silent) {
        message.error(errorMsg);
      }
      setModelOptions([]);
      setModelsFetched(true); // 即使失败也标记为已尝试，避免重复请求
    } finally {
      setFetchingModels(false);
    }
  };

  const handleModelSelectFocus = () => {
    // 如果还没有获取过模型列表，自动获取
    if (!modelsFetched && !fetchingModels) {
      handleFetchModels(true); // silent模式，不显示成功消息
    }
  };

  const handleTestConnection = async () => {
    const apiKey = form.getFieldValue('api_key');
    const apiBaseUrl = form.getFieldValue('api_base_url');
    const provider = form.getFieldValue('api_provider');
    const modelName = form.getFieldValue('llm_model');

    if (!apiKey || !apiBaseUrl || !provider || !modelName) {
      message.warning('请先填写完整的配置信息');
      return;
    }

    setTestingApi(true);
    setTestResult(null);

    try {
      const result = await settingsApi.testApiConnection({
        api_key: apiKey,
        api_base_url: apiBaseUrl,
        provider: provider,
        llm_model: modelName
      });

      setTestResult(result);
      setShowTestResult(true);

      if (result.success) {
        message.success(`测试成功！响应时间: ${result.response_time_ms}ms`);
      } else {
        message.error('API 测试失败，请查看详细信息');
      }
    } catch (error: any) {
      const errorMsg = error?.response?.data?.detail || '测试请求失败';
      message.error(errorMsg);
      setTestResult({
        success: false,
        message: '测试请求失败',
        error: errorMsg,
        error_type: 'RequestError',
        suggestions: ['请检查网络连接', '请确认后端服务是否正常运行']
      });
      setShowTestResult(true);
    } finally {
      setTestingApi(false);
    }
  };

  return (
    <div style={{
      minHeight: '100vh',
      background: 'var(--color-bg-base)',
      padding: isMobile ? '16px 12px' : '40px 24px'
    }}>
      <div style={{
        maxWidth: isMobile ? '100%' : 800,
        margin: '0 auto'
      }}>
        <Card
          variant="borderless"
          style={{
            background: 'var(--color-bg-container)',
            borderRadius: isMobile ? 12 : 16,
            boxShadow: 'var(--shadow-card)',
          }}
          styles={{
            body: {
              padding: isMobile ? '16px' : '24px'
            }
          }}
        >
          <Space direction="vertical" size={isMobile ? 'middle' : 'large'} style={{ width: '100%' }}>
            {/* 标题栏 */}
            <div style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              flexWrap: 'wrap',
              gap: '8px'
            }}>
              <Space size={isMobile ? 'small' : 'middle'}>
                <Button
                  icon={<ArrowLeftOutlined />}
                  onClick={() => navigate('/')}
                  type="text"
                  size={isMobile ? 'middle' : 'large'}
                />
                <Title
                  level={isMobile ? 4 : 2}
                  style={{
                    margin: 0,
                    fontSize: isMobile ? '18px' : undefined
                  }}
                >
                  <SettingOutlined style={{ marginRight: 8, color: 'var(--color-primary)' }} />
                  {isMobile ? 'API 设置' : 'AI API 设置'}
                </Title>
              </Space>
            </div>

            <Paragraph
              type="secondary"
              style={{
                marginBottom: 0,
                fontSize: isMobile ? '13px' : '14px',
                lineHeight: isMobile ? '1.5' : '1.6'
              }}
            >
              配置你的AI API接口参数，这些设置将用于小说生成、角色创建等AI功能。
            </Paragraph>

            {/* 默认配置提示 */}
            {isDefaultSettings && (
              <Alert
                message="使用 .env 文件中的默认配置"
                description={
                  <div style={{ fontSize: isMobile ? '12px' : '14px' }}>
                    <p style={{ margin: '8px 0' }}>
                      当前显示的是从服务器 <code>.env</code> 文件读取的默认配置。
                    </p>
                    <p style={{ margin: '8px 0 0 0' }}>
                      点击"保存设置"后，配置将保存到数据库并同步更新到 <code>.env</code> 文件。
                    </p>
                  </div>
                }
                type="info"
                showIcon
                style={{ marginBottom: isMobile ? 12 : 16 }}
              />
            )}

            {/* 已保存配置提示 */}
            {hasSettings && !isDefaultSettings && (
              <Alert
                message="使用已保存的个人配置"
                type="success"
                showIcon
                style={{ marginBottom: isMobile ? 12 : 16 }}
              />
            )}

            {/* 表单 */}
            <Spin spinning={initialLoading}>
              <Form
                form={form}
                layout="vertical"
                onFinish={handleSave}
                autoComplete="off"
              >
                <Form.Item
                  label={
                    <Space size={4}>
                      <span>API 提供商</span>
                      <Tooltip title="选择你的AI服务提供商">
                        <InfoCircleOutlined style={{ color: 'var(--color-text-secondary)', fontSize: isMobile ? '12px' : '14px' }} />
                      </Tooltip>
                    </Space>
                  }
                  name="api_provider"
                  rules={[{ required: true, message: '请选择API提供商' }]}
                >
                  <Select size={isMobile ? 'middle' : 'large'} onChange={handleProviderChange}>
                    {apiProviders.map(provider => (
                      <Option key={provider.value} value={provider.value}>
                        {provider.label}
                      </Option>
                    ))}
                  </Select>
                </Form.Item>

                <Form.Item
                  label={
                    <Space size={4}>
                      <span>API 密钥</span>
                      <Tooltip title="你的API密钥，将加密存储">
                        <InfoCircleOutlined style={{ color: 'var(--color-text-secondary)', fontSize: isMobile ? '12px' : '14px' }} />
                      </Tooltip>
                    </Space>
                  }
                  name="api_key"
                  rules={[{ required: true, message: '请输入API密钥' }]}
                >
                  <Input.Password
                    size={isMobile ? 'middle' : 'large'}
                    placeholder="sk-..."
                    autoComplete="new-password"
                  />
                </Form.Item>

                <Form.Item
                  label={
                    <Space size={4}>
                      <span>API 地址</span>
                      <Tooltip title="API的基础URL地址">
                        <InfoCircleOutlined style={{ color: 'var(--color-text-secondary)', fontSize: isMobile ? '12px' : '14px' }} />
                      </Tooltip>
                    </Space>
                  }
                  name="api_base_url"
                  rules={[
                    { required: true, message: '请输入API地址' },
                    { type: 'url', message: '请输入有效的URL' }
                  ]}
                >
                  <Input
                    size={isMobile ? 'middle' : 'large'}
                    placeholder="https://api.openai.com/v1"
                  />
                </Form.Item>

                <Form.Item
                  label={
                    <Space size={4}>
                      <span>模型名称</span>
                      <Tooltip title="AI模型的名称，如 gpt-4, gpt-3.5-turbo">
                        <InfoCircleOutlined style={{ color: 'var(--color-text-secondary)', fontSize: isMobile ? '12px' : '14px' }} />
                      </Tooltip>
                    </Space>
                  }
                  name="llm_model"
                  rules={[{ required: true, message: '请输入或选择模型名称' }]}
                >
                  <Select
                    size={isMobile ? 'middle' : 'large'}
                    showSearch
                    placeholder={isMobile ? "选择模型" : "输入模型名称或点击获取"}
                    optionFilterProp="label"
                    loading={fetchingModels}
                    onFocus={handleModelSelectFocus}
                    filterOption={(input, option) =>
                      (option?.label ?? '').toLowerCase().includes(input.toLowerCase()) ||
                      (option?.description ?? '').toLowerCase().includes(input.toLowerCase())
                    }
                    dropdownRender={(menu) => (
                      <>
                        {menu}
                        {fetchingModels && (
                          <div style={{ padding: '8px 12px', color: 'var(--color-text-secondary)', textAlign: 'center', fontSize: isMobile ? '12px' : '14px' }}>
                            <Spin size="small" /> 正在获取模型列表...
                          </div>
                        )}
                        {!fetchingModels && modelOptions.length === 0 && modelsFetched && (
                          <div style={{ padding: '8px 12px', color: '#ff4d4f', textAlign: 'center', fontSize: isMobile ? '12px' : '14px' }}>
                            未能获取到模型列表，请检查 API 配置
                          </div>
                        )}
                        {!fetchingModels && modelOptions.length === 0 && !modelsFetched && (
                          <div style={{ padding: '8px 12px', color: 'var(--color-text-secondary)', textAlign: 'center', fontSize: isMobile ? '12px' : '14px' }}>
                            点击输入框自动获取模型列表
                          </div>
                        )}
                      </>
                    )}
                    notFoundContent={
                      fetchingModels ? (
                        <div style={{ padding: '8px 12px', textAlign: 'center', fontSize: isMobile ? '12px' : '14px' }}>
                          <Spin size="small" /> 加载中...
                        </div>
                      ) : (
                        <div style={{ padding: '8px 12px', color: 'var(--color-text-secondary)', textAlign: 'center', fontSize: isMobile ? '12px' : '14px' }}>
                          未找到匹配的模型
                        </div>
                      )
                    }
                    suffixIcon={
                      !isMobile ? (
                        <div
                          onClick={(e) => {
                            e.stopPropagation();
                            if (!fetchingModels) {
                              setModelsFetched(false);
                              handleFetchModels(false);
                            }
                          }}
                          style={{
                            cursor: fetchingModels ? 'not-allowed' : 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            padding: '0 4px',
                            height: '100%',
                            marginRight: -8
                          }}
                          title="重新获取模型列表"
                        >
                          <Button
                            type="text"
                            size="small"
                            icon={<ReloadOutlined />}
                            loading={fetchingModels}
                            style={{ pointerEvents: 'none' }}
                          >
                            刷新
                          </Button>
                        </div>
                      ) : undefined
                    }
                    options={modelOptions.map(model => ({
                      value: model.value,
                      label: model.label,
                      description: model.description
                    }))}
                    optionRender={(option) => (
                      <div>
                        <div style={{ fontWeight: 500, fontSize: isMobile ? '13px' : '14px' }}>{option.data.label}</div>
                        {option.data.description && (
                          <div style={{ fontSize: isMobile ? '11px' : '12px', color: '#8c8c8c', marginTop: '2px' }}>
                            {option.data.description}
                          </div>
                        )}
                      </div>
                    )}
                  />
                </Form.Item>

                <Form.Item
                  label={
                    <Space size={4}>
                      <span>温度参数</span>
                      <Tooltip title="控制输出的随机性，值越高越随机（0.0-2.0）">
                        <InfoCircleOutlined style={{ color: 'var(--color-text-secondary)', fontSize: isMobile ? '12px' : '14px' }} />
                      </Tooltip>
                    </Space>
                  }
                  name="temperature"
                >
                  <Slider
                    min={0}
                    max={2}
                    step={0.1}
                    marks={{
                      0: { style: { fontSize: isMobile ? '11px' : '12px' }, label: '0.0' },
                      0.7: { style: { fontSize: isMobile ? '11px' : '12px' }, label: '0.7' },
                      1: { style: { fontSize: isMobile ? '11px' : '12px' }, label: '1.0' },
                      2: { style: { fontSize: isMobile ? '11px' : '12px' }, label: '2.0' }
                    }}
                  />
                </Form.Item>

                <Form.Item
                  label={
                    <Space size={4}>
                      <span>最大 Token 数</span>
                      <Tooltip title="单次请求的最大token数量">
                        <InfoCircleOutlined style={{ color: 'var(--color-text-secondary)', fontSize: isMobile ? '12px' : '14px' }} />
                      </Tooltip>
                    </Space>
                  }
                  name="max_tokens"
                  rules={[
                    { required: true, message: '请输入最大token数' },
                    { type: 'number', min: 1, message: '请输入大于0的数字' }
                  ]}
                >
                  <InputNumber
                    size={isMobile ? 'middle' : 'large'}
                    style={{ width: '100%' }}
                    min={1}
                    placeholder="2000"
                  />
                </Form.Item>

                {/* 测试结果展示 */}
                {showTestResult && testResult && (
                  <Alert
                    message={
                      <Space>
                        {testResult.success ? (
                          <CheckCircleOutlined style={{ color: 'var(--color-success)', fontSize: isMobile ? '16px' : '18px' }} />
                        ) : (
                          <CloseCircleOutlined style={{ color: 'var(--color-error)', fontSize: isMobile ? '16px' : '18px' }} />
                        )}
                        <span style={{ fontSize: isMobile ? '14px' : '16px', fontWeight: 500 }}>
                          {testResult.message}
                        </span>
                      </Space>
                    }
                    description={
                      <div style={{ marginTop: 8 }}>
                        {testResult.success ? (
                          <Space direction="vertical" size="small" style={{ width: '100%' }}>
                            {testResult.response_time_ms && (
                              <div style={{ fontSize: isMobile ? '12px' : '14px' }}>
                                ⚡ 响应时间: <strong>{testResult.response_time_ms} ms</strong>
                              </div>
                            )}
                            {testResult.response_preview && (
                              <div style={{
                                fontSize: isMobile ? '12px' : '13px',
                                padding: '8px 12px',
                                background: '#f6ffed',
                                borderRadius: '4px',
                                border: '1px solid #b7eb8f',
                                marginTop: '8px'
                              }}>
                                <div style={{ marginBottom: '4px', fontWeight: 500 }}>AI 响应预览:</div>
                                <div style={{ color: '#595959' }}>{testResult.response_preview}</div>
                              </div>
                            )}
                            <div style={{ color: 'var(--color-success)', fontSize: isMobile ? '12px' : '13px', marginTop: '4px' }}>
                              ✓ API 配置正确，可以正常使用
                            </div>
                          </Space>
                        ) : (
                          <Space direction="vertical" size="small" style={{ width: '100%' }}>
                            {testResult.error && (
                              <div style={{
                                fontSize: isMobile ? '12px' : '13px',
                                padding: '8px 12px',
                                background: '#fff2e8',
                                borderRadius: '4px',
                                border: '1px solid #ffbb96',
                                color: '#d4380d'
                              }}>
                                <strong>错误信息:</strong> {testResult.error}
                              </div>
                            )}
                            {testResult.error_type && (
                              <div style={{ fontSize: isMobile ? '11px' : '12px', color: 'var(--color-text-secondary)' }}>
                                错误类型: {testResult.error_type}
                              </div>
                            )}
                            {testResult.suggestions && testResult.suggestions.length > 0 && (
                              <div style={{ marginTop: '8px' }}>
                                <div style={{ fontSize: isMobile ? '12px' : '13px', fontWeight: 500, marginBottom: '4px' }}>
                                  💡 解决建议:
                                </div>
                                <ul style={{
                                  margin: 0,
                                  paddingLeft: isMobile ? '16px' : '20px',
                                  fontSize: isMobile ? '12px' : '13px',
                                  color: '#595959'
                                }}>
                                  {testResult.suggestions.map((suggestion, index) => (
                                    <li key={index} style={{ marginBottom: '4px' }}>{suggestion}</li>
                                  ))}
                                </ul>
                              </div>
                            )}
                          </Space>
                        )}
                      </div>
                    }
                    type={testResult.success ? 'success' : 'error'}
                    closable
                    onClose={() => setShowTestResult(false)}
                    style={{ marginBottom: isMobile ? 16 : 24 }}
                  />
                )}

                {/* 操作按钮 */}
                <Form.Item style={{ marginBottom: 0, marginTop: isMobile ? 24 : 32 }}>
                  {isMobile ? (
                    // 移动端：垂直堆叠布局
                    <Space direction="vertical" size="middle" style={{ width: '100%' }}>
                      <Button
                        type="primary"
                        size="large"
                        icon={<SaveOutlined />}
                        htmlType="submit"
                        loading={loading}
                        block
                        style={{
                          background: 'var(--color-primary)',
                          border: 'none',
                          height: '44px'
                        }}
                      >
                        保存设置
                      </Button>
                      <Button
                        size="large"
                        icon={<ThunderboltOutlined />}
                        onClick={handleTestConnection}
                        loading={testingApi}
                        block
                        style={{
                          borderColor: 'var(--color-success)',
                          color: 'var(--color-success)',
                          fontWeight: 500,
                          height: '44px'
                        }}
                      >
                        {testingApi ? '测试中...' : '测试连接'}
                      </Button>
                      <Space size="middle" style={{ width: '100%' }}>
                        <Button
                          size="large"
                          icon={<ReloadOutlined />}
                          onClick={handleReset}
                          style={{ flex: 1, height: '44px' }}
                        >
                          重置
                        </Button>
                        {hasSettings && (
                          <Button
                            danger
                            size="large"
                            icon={<DeleteOutlined />}
                            onClick={handleDelete}
                            loading={loading}
                            style={{ flex: 1, height: '44px' }}
                          >
                            删除
                          </Button>
                        )}
                      </Space>
                    </Space>
                  ) : (
                    // 桌面端：删除在左边，测试、重置和保存在右边
                    <div style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      gap: '16px',
                      flexWrap: 'wrap'
                    }}>
                      {/* 左侧：删除按钮 */}
                      {hasSettings ? (
                        <Button
                          danger
                          size="large"
                          icon={<DeleteOutlined />}
                          onClick={handleDelete}
                          loading={loading}
                          style={{
                            minWidth: '100px'
                          }}
                        >
                          删除配置
                        </Button>
                      ) : (
                        <div /> // 占位符，保持右侧按钮位置
                      )}

                      {/* 右侧：测试、重置和保存按钮组 */}
                      <Space size="middle">
                        <Button
                          size="large"
                          icon={<ThunderboltOutlined />}
                          onClick={handleTestConnection}
                          loading={testingApi}
                          style={{
                            borderColor: 'var(--color-success)',
                            color: 'var(--color-success)',
                            fontWeight: 500,
                            minWidth: '100px'
                          }}
                        >
                          {testingApi ? '测试中...' : '测试'}
                        </Button>
                        <Button
                          size="large"
                          icon={<ReloadOutlined />}
                          onClick={handleReset}
                          style={{
                            minWidth: '100px'
                          }}
                        >
                          重置
                        </Button>
                        <Button
                          type="primary"
                          size="large"
                          icon={<SaveOutlined />}
                          htmlType="submit"
                          loading={loading}
                          style={{
                            background: 'var(--color-primary)',
                            border: 'none',
                            minWidth: '120px',
                            fontWeight: 500
                          }}
                        >
                          保存
                        </Button>
                      </Space>
                    </div>
                  )}
                </Form.Item>
              </Form>
            </Spin>
          </Space>
        </Card>
      </div>
    </div>
  );
}