import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Card,
  Button,
  Space,
  Typography,
  Modal,
  Form,
  Input,
  Switch,
  Select,
  message,
  Tag,
  Tooltip,
  Spin,
  Empty,
  Alert,
  Descriptions,
  Row,
  Col,
} from 'antd';
import {
  PlusOutlined,
  EditOutlined,
  DeleteOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  ThunderboltOutlined,
  InfoCircleOutlined,
  ToolOutlined,
  ArrowLeftOutlined,
} from '@ant-design/icons';
import { mcpPluginApi } from '../services/api';
import type { MCPPlugin, MCPTool } from '../types';

const { Paragraph, Text, Title } = Typography;
const { TextArea } = Input;

export default function MCPPluginsPage() {
  const navigate = useNavigate();
  const isMobile = window.innerWidth <= 768;
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [plugins, setPlugins] = useState<MCPPlugin[]>([]);
  const [modalVisible, setModalVisible] = useState(false);
  const [editingPlugin, setEditingPlugin] = useState<MCPPlugin | null>(null);
  const [testingPluginId, setTestingPluginId] = useState<string | null>(null);
  const [viewingTools, setViewingTools] = useState<{ pluginId: string; tools: MCPTool[] } | null>(null);

  useEffect(() => {
    loadPlugins();
  }, []);

  const loadPlugins = async () => {
    setLoading(true);
    try {
      const data = await mcpPluginApi.getPlugins();
      setPlugins(data);
    } catch (error) {
      message.error('加载插件列表失败');
    } finally {
      setLoading(false);
    }
  };

  const handleCreate = () => {
    setEditingPlugin(null);
    form.resetFields();
    form.setFieldsValue({
      enabled: true,
      category: 'search',
      config_json: `{
  "mcpServers": {
    "exa": {
      "type": "http",
      "url": "https://mcp.exa.ai/mcp?exaApiKey=YOUR_API_KEY",
      "headers": {}
    }
  }
}`
    });
    setModalVisible(true);
  };

  const handleEdit = (plugin: MCPPlugin) => {
    setEditingPlugin(plugin);

    // 重构为标准MCP配置格式
    const mcpConfig: any = {
      mcpServers: {
        [plugin.plugin_name]: {
          type: plugin.plugin_type || 'http'
        }
      }
    };

    if (plugin.plugin_type === 'http') {
      mcpConfig.mcpServers[plugin.plugin_name].url = plugin.server_url;
      mcpConfig.mcpServers[plugin.plugin_name].headers = plugin.headers || {};
    } else {
      mcpConfig.mcpServers[plugin.plugin_name].command = plugin.command;
      mcpConfig.mcpServers[plugin.plugin_name].args = plugin.args || [];
      mcpConfig.mcpServers[plugin.plugin_name].env = plugin.env || {};
    }

    form.setFieldsValue({
      config_json: JSON.stringify(mcpConfig, null, 2),
      enabled: plugin.enabled,
      category: plugin.category || 'general',
    });
    setModalVisible(true);
  };

  const handleDelete = (plugin: MCPPlugin) => {
    Modal.confirm({
      title: '删除插件',
      content: `确定要删除插件 "${plugin.display_name || plugin.plugin_name}" 吗？`,
      okText: '确定',
      cancelText: '取消',
      okType: 'danger',
      onOk: async () => {
        try {
          await mcpPluginApi.deletePlugin(plugin.id);
          message.success('插件已删除');
          loadPlugins();
        } catch (error) {
          message.error('删除插件失败');
        }
      },
    });
  };

  const handleToggle = async (plugin: MCPPlugin, enabled: boolean) => {
    try {
      await mcpPluginApi.togglePlugin(plugin.id, enabled);
      message.success(enabled ? '插件已启用' : '插件已禁用');
      loadPlugins();
    } catch (error) {
      message.error('切换插件状态失败');
    }
  };

  const handleTest = async (pluginId: string) => {
    setTestingPluginId(pluginId);
    try {
      const result = await mcpPluginApi.testPlugin(pluginId);

      // 测试完成后，无论成功失败都刷新插件列表以更新状态
      await loadPlugins();

      if (result.success) {
        Modal.success({
          title: '✅ 测试成功',
          width: 700,
          content: (
            <div>
              <p style={{ marginBottom: 16, fontSize: '15px', fontWeight: 500 }}>{result.message}</p>

              {/* 显示详细的测试结果 */}
              {result.suggestions && result.suggestions.length > 0 && (
                <div style={{ marginTop: 16 }}>
                  <Text strong style={{ fontSize: '14px' }}>测试详情:</Text>
                  <div style={{
                    marginTop: 8,
                    padding: 12,
                    background: '#f5f5f5',
                    borderRadius: 4,
                    fontSize: '13px',
                    fontFamily: 'monospace',
                    maxHeight: '400px',
                    overflowY: 'auto'
                  }}>
                    {result.suggestions.map((suggestion: string, index: number) => (
                      <div key={index} style={{
                        marginBottom: index < (result.suggestions?.length || 0) - 1 ? 8 : 0,
                        whiteSpace: 'pre-wrap',
                        wordBreak: 'break-word',
                        lineHeight: 1.6
                      }}>
                        {suggestion}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* 显示工具数量 */}
              {result.tools_count !== undefined && (
                <div style={{ marginTop: 12 }}>
                  <Text type="secondary">🔧 可用工具数: <strong>{result.tools_count}</strong></Text>
                </div>
              )}

              {/* 显示响应时间 */}
              {result.response_time_ms !== undefined && (
                <div style={{ marginTop: 4 }}>
                  <Text type="secondary">⏱️ 响应时间: <strong>{result.response_time_ms}ms</strong></Text>
                </div>
              )}

              <div style={{
                marginTop: 16,
                padding: 12,
                background: '#f6ffed',
                border: '1px solid #b7eb8f',
                borderRadius: 4
              }}>
                <Text type="success" style={{ fontSize: '13px' }}>
                  ✓ 插件状态已自动更新为"运行中"
                </Text>
              </div>
            </div>
          ),
        });
      } else {
        Modal.error({
          title: '❌ 测试失败',
          width: 700,
          content: (
            <div>
              <p style={{ marginBottom: 16, fontSize: '15px', fontWeight: 500 }}>{result.message}</p>

              {/* 显示错误信息 */}
              {result.error && (
                <div style={{ marginTop: 16 }}>
                  <Text strong style={{ fontSize: '14px' }}>错误信息:</Text>
                  <div style={{
                    marginTop: 8,
                    padding: 12,
                    background: 'var(--color-error-bg)',
                    borderRadius: 4,
                    color: 'var(--color-error)',
                    fontSize: '13px',
                    fontFamily: 'monospace',
                    whiteSpace: 'pre-wrap',
                    wordBreak: 'break-word',
                    maxHeight: '300px',
                    overflowY: 'auto'
                  }}>
                    {result.error}
                  </div>
                </div>
              )}

              {/* 显示建议 */}
              {result.suggestions && result.suggestions.length > 0 && (
                <div style={{ marginTop: 16 }}>
                  <Text strong style={{ fontSize: '14px' }}>💡 建议:</Text>
                  <ul style={{ marginTop: 8, marginBottom: 0, paddingLeft: 20 }}>
                    {result.suggestions.map((suggestion: string, index: number) => (
                      <li key={index} style={{ marginBottom: 6, fontSize: '13px' }}>
                        {suggestion}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              <div style={{
                marginTop: 16,
                padding: 12,
                background: 'var(--color-warning-bg)',
                border: '1px solid var(--color-warning-border)',
                borderRadius: 4
              }}>
                <Text style={{ fontSize: '13px', color: '#ad6800' }}>
                  ⚠️ 插件状态已更新，请检查配置后重试
                </Text>
              </div>
            </div>
          ),
        });
      }
    } catch (error: any) {
      message.error('测试插件失败');
    } finally {
      setTestingPluginId(null);
    }
  };

  const handleViewTools = async (pluginId: string) => {
    try {
      const result = await mcpPluginApi.getPluginTools(pluginId);
      setViewingTools({ pluginId, tools: result.tools });
    } catch (error) {
      message.error('获取工具列表失败');
    }
  };

  const handleSubmit = async (values: any) => {
    setLoading(true);
    try {
      // 验证JSON格式
      try {
        JSON.parse(values.config_json);
      } catch (e) {
        message.error('配置JSON格式错误，请检查');
        setLoading(false);
        return;
      }

      const data = {
        config_json: values.config_json,
        enabled: values.enabled,
        category: values.category || 'general',
      };

      // 统一使用简化API，后端会自动判断是创建还是更新
      await mcpPluginApi.createPluginSimple(data);
      message.success(editingPlugin ? '插件已更新' : '插件已创建');

      setModalVisible(false);
      form.resetFields();
      loadPlugins();
    } catch (error: any) {
      const errorMsg = error?.response?.data?.detail || '操作失败';
      message.error(errorMsg);
    } finally {
      setLoading(false);
    }
  };

  const getStatusTag = (plugin: MCPPlugin) => {
    if (!plugin.enabled) {
      return <Tag color="default">已禁用</Tag>;
    }
    switch (plugin.status) {
      case 'active':
        return <Tag color="success" icon={<CheckCircleOutlined />}>运行中</Tag>;
      case 'error':
        return (
          <Tooltip title={plugin.last_error}>
            <Tag color="error" icon={<CloseCircleOutlined />}>错误</Tag>
          </Tooltip>
        );
      default:
        return <Tag color="default">未激活</Tag>;
    }
  };

  return (
    <div style={{
      minHeight: '100vh',
      background: 'linear-gradient(180deg, var(--color-bg-base) 0%, #EEF2F3 100%)',
      padding: isMobile ? '20px 16px' : '40px 24px',
      display: 'flex',
      flexDirection: 'column',
    }}>
      <div style={{
        maxWidth: 1400,
        margin: '0 auto',
        width: '100%',
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
      }}>
        {/* 顶部导航卡片 */}
        <Card
          variant="borderless"
          style={{
            background: 'linear-gradient(135deg, var(--color-primary) 0%, #5A9BA5 50%, var(--color-primary-hover) 100%)',
            borderRadius: isMobile ? 16 : 24,
            boxShadow: '0 12px 40px rgba(77, 128, 136, 0.25), 0 4px 12px rgba(0, 0, 0, 0.06)',
            marginBottom: isMobile ? 20 : 24,
            border: 'none',
            position: 'relative',
            overflow: 'hidden'
          }}
        >
          {/* 装饰性背景元素 */}
          <div style={{ position: 'absolute', top: -60, right: -60, width: 200, height: 200, borderRadius: '50%', background: 'rgba(255, 255, 255, 0.08)', pointerEvents: 'none' }} />
          <div style={{ position: 'absolute', bottom: -40, left: '30%', width: 120, height: 120, borderRadius: '50%', background: 'rgba(255, 255, 255, 0.05)', pointerEvents: 'none' }} />
          <div style={{ position: 'absolute', top: '50%', right: '15%', width: 80, height: 80, borderRadius: '50%', background: 'rgba(255, 255, 255, 0.06)', pointerEvents: 'none' }} />

          <Row align="middle" justify="space-between" gutter={[16, 16]} style={{ position: 'relative', zIndex: 1 }}>
            <Col xs={24} sm={12}>
              <Space direction="vertical" size={4}>
                <Space align="center">
                  <Title level={isMobile ? 3 : 2} style={{ margin: 0, color: '#fff', textShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
                    <ToolOutlined style={{ color: 'rgba(255,255,255,0.9)', marginRight: 8 }} />
                    MCP插件管理
                  </Title>
                </Space>
                <Text style={{ fontSize: isMobile ? 12 : 14, color: 'rgba(255,255,255,0.85)', marginLeft: isMobile ? 40 : 48 }}>
                  扩展AI能力，连接外部工具与服务
                </Text>
              </Space>
            </Col>
            <Col xs={24} sm={12}>
              <Space size={12} style={{ display: 'flex', justifyContent: isMobile ? 'flex-start' : 'flex-end', width: '100%' }}>
                <Button
                  icon={<ArrowLeftOutlined />}
                  onClick={() => navigate('/')}
                  style={{
                    borderRadius: 12,
                    background: 'rgba(255, 255, 255, 0.15)',
                    border: '1px solid rgba(255, 255, 255, 0.3)',
                    boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)',
                    color: '#fff',
                    backdropFilter: 'blur(10px)',
                    transition: 'all 0.3s ease'
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.background = 'rgba(255, 255, 255, 0.25)';
                    e.currentTarget.style.transform = 'translateY(-1px)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.background = 'rgba(255, 255, 255, 0.15)';
                    e.currentTarget.style.transform = 'none';
                  }}
                >
                  返回主页
                </Button>
                <Button
                  type="primary"
                  icon={<PlusOutlined />}
                  onClick={handleCreate}
                  style={{
                    borderRadius: 12,
                    background: 'rgba(255, 193, 7, 0.95)',
                    border: '1px solid rgba(255, 255, 255, 0.3)',
                    boxShadow: '0 4px 16px rgba(255, 193, 7, 0.4)',
                    color: '#fff',
                    fontWeight: 600
                  }}
                >
                  添加插件
                </Button>
              </Space>
            </Col>
          </Row>

          {/* 使用提示 */}
          <Alert
            message={
              <Space align="center">
                <InfoCircleOutlined style={{ fontSize: 16, color: 'var(--color-primary)' }} />
                <Text strong style={{ fontSize: isMobile ? 13 : 14, color: 'var(--color-text-primary)' }}>什么是 MCP 插件？</Text>
              </Space>
            }
            description={
              <div>
                <Text style={{ fontSize: isMobile ? 12 : 13, display: 'block', marginBottom: 8 }}>
                  • <strong>MCP (Model Context Protocol)</strong> 是一个标准化的协议，允许 AI 调用外部工具获取数据。
                </Text>
                <Text style={{ fontSize: isMobile ? 12 : 13, display: 'block' }}>
                  • 通过添加 MCP 插件，AI 可以访问搜索引擎、数据库、API 等外部服务，增强创作能力。
                </Text>
              </div>
            }
            type="info"
            showIcon={false}
            style={{
              marginTop: isMobile ? 16 : 24,
              borderRadius: 12,
              background: 'rgba(230, 247, 255, 0.6)',
              border: '1px solid rgba(145, 213, 255, 0.6)',
              backdropFilter: 'blur(5px)'
            }}
          />
        </Card>

        {/* 主内容区 */}
        <div style={{ flex: 1 }}>

          {/* 插件列表 */}
          <Spin spinning={loading}>
            {plugins.length === 0 ? (
              <Empty
                description="还没有添加任何插件"
                image={Empty.PRESENTED_IMAGE_SIMPLE}
                style={{ padding: isMobile ? '40px 0' : '60px 0' }}
              >
                <Button type="primary" icon={<PlusOutlined />} onClick={handleCreate}>
                  添加第一个插件
                </Button>
              </Empty>
            ) : (
              <Space direction="vertical" size="middle" style={{ width: '100%' }}>
                {plugins.map((plugin) => (
                  <Card
                    key={plugin.id}
                    size="small"
                    style={{
                      borderRadius: 8,
                      border: '1px solid #f0f0f0',
                    }}
                  >
                    <div
                      style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'flex-start',
                        gap: '16px',
                        flexWrap: isMobile ? 'wrap' : 'nowrap',
                      }}
                    >
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <Space direction="vertical" size="small" style={{ width: '100%' }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
                            <Text strong style={{ fontSize: isMobile ? '14px' : '16px' }}>
                              {plugin.display_name || plugin.plugin_name}
                            </Text>
                            {getStatusTag(plugin)}
                            <Tag color={plugin.plugin_type === 'http' ? 'blue' : 'cyan'}>
                              {plugin.plugin_type?.toUpperCase() || 'UNKNOWN'}
                            </Tag>
                            {plugin.category && plugin.category !== 'general' && (
                              <Tag color="purple">{plugin.category}</Tag>
                            )}
                          </div>
                          {plugin.description && (
                            <Paragraph
                              type="secondary"
                              style={{
                                margin: 0,
                                fontSize: isMobile ? '12px' : '13px',
                              }}
                              ellipsis={{ rows: 2 }}
                            >
                              {plugin.description}
                            </Paragraph>
                          )}

                          {/* 只显示有值的URL或命令，脱敏处理敏感信息 */}
                          {plugin.plugin_type === 'http' && plugin.server_url && (
                            <div style={{ fontSize: isMobile ? '11px' : '12px' }}>
                              <Text type="secondary" code>
                                {(() => {
                                  // 脱敏处理：隐藏URL中的API Key
                                  const url = plugin.server_url;
                                  try {
                                    const urlObj = new URL(url);
                                    // 替换查询参数中的敏感信息
                                    const params = new URLSearchParams(urlObj.search);
                                    let maskedUrl = `${urlObj.protocol}//${urlObj.host}${urlObj.pathname}`;

                                    const sensitiveKeys = ['apiKey', 'api_key', 'key', 'token', 'secret', 'password', 'auth'];
                                    let hasParams = false;

                                    params.forEach((value, key) => {
                                      const isSensitive = sensitiveKeys.some(k => key.toLowerCase().includes(k.toLowerCase()));
                                      const maskedValue = isSensitive ? '***' : value;
                                      maskedUrl += (hasParams ? '&' : '?') + `${key}=${maskedValue}`;
                                      hasParams = true;
                                    });

                                    return maskedUrl;
                                  } catch {
                                    // 如果URL解析失败，尝试简单替换
                                    return url.replace(/([?&])(apiKey|api_key|key|token|secret|password|auth)=([^&]+)/gi, '$1$2=***');
                                  }
                                })()}
                              </Text>
                            </div>
                          )}

                          {plugin.plugin_type === 'stdio' && plugin.command && (
                            <div style={{ fontSize: isMobile ? '11px' : '12px' }}>
                              <Text type="secondary" code>
                                {plugin.command} {plugin.args?.join(' ')}
                              </Text>
                            </div>
                          )}

                          {/* 显示最后错误信息 */}
                          {plugin.last_error && (
                            <Text type="danger" style={{ fontSize: isMobile ? '11px' : '12px' }}>
                              错误: {plugin.last_error}
                            </Text>
                          )}
                        </Space>
                      </div>

                      <Space size="small" wrap>
                        <Tooltip title={plugin.enabled ? '禁用插件' : '启用插件'}>
                          <Switch
                            checked={plugin.enabled}
                            onChange={(checked) => handleToggle(plugin, checked)}
                            size={isMobile ? 'small' : 'default'}
                            style={{
                              flexShrink: 0,
                              height: isMobile ? 16 : 22,
                              minHeight: isMobile ? 16 : 22,
                              lineHeight: isMobile ? '16px' : '22px'
                            }}
                          />
                        </Tooltip>
                        <Tooltip title="测试连接">
                          <Button
                            icon={<ThunderboltOutlined />}
                            onClick={() => handleTest(plugin.id)}
                            loading={testingPluginId === plugin.id}
                            size={isMobile ? 'small' : 'middle'}
                          />
                        </Tooltip>
                        <Tooltip title="查看工具">
                          <Button
                            icon={<ToolOutlined />}
                            onClick={() => handleViewTools(plugin.id)}
                            disabled={!plugin.enabled || plugin.status !== 'active'}
                            size={isMobile ? 'small' : 'middle'}
                          />
                        </Tooltip>
                        <Tooltip title="编辑">
                          <Button
                            icon={<EditOutlined />}
                            onClick={() => handleEdit(plugin)}
                            size={isMobile ? 'small' : 'middle'}
                          />
                        </Tooltip>
                        <Tooltip title="删除">
                          <Button
                            danger
                            icon={<DeleteOutlined />}
                            onClick={() => handleDelete(plugin)}
                            size={isMobile ? 'small' : 'middle'}
                          />
                        </Tooltip>
                      </Space>
                    </div>
                  </Card>
                ))}
              </Space>
            )}
          </Spin>
        </div>
      </div>

      {/* 创建/编辑插件模态框 */}
      <Modal
        title={editingPlugin ? '编辑插件' : '添加插件'}
        open={modalVisible}
        centered
        onCancel={() => {
          setModalVisible(false);
          form.resetFields();
        }}
        onOk={() => form.submit()}
        width={isMobile ? '100%' : 600}
        confirmLoading={loading}
        okText="保存"
        cancelText="取消"
      >
        <Form form={form} layout="vertical" onFinish={handleSubmit}>
          <Form.Item
            label="MCP配置JSON"
            name="config_json"
            rules={[{ required: true, message: '请输入配置JSON' }]}
            extra="粘贴标准MCP配置，系统自动提取插件名称。支持HTTP和Stdio类型"
          >
            <TextArea
              rows={16}
              placeholder={`示例：
{
  "mcpServers": {
    "exa": {
      "type": "http",
      "url": "https://mcp.exa.ai/mcp?exaApiKey=YOUR_API_KEY",
      "headers": {}
    }
  }
}`}
              style={{ fontFamily: 'monospace', fontSize: '13px' }}
            />
          </Form.Item>

          <Form.Item
            label="插件分类"
            name="category"
            rules={[{ required: true, message: '请选择插件分类' }]}
            extra="选择插件的功能类别，用于AI智能匹配使用场景"
          >
            <Select placeholder="请选择分类">
              <Select.Option value="search">搜索类 (Search) - 网络搜索、信息查询</Select.Option>
              <Select.Option value="analysis">分析类 (Analysis) - 数据分析、文本处理</Select.Option>
              <Select.Option value="filesystem">文件系统 (FileSystem) - 文件读写操作</Select.Option>
              <Select.Option value="database">数据库 (Database) - 数据库查询</Select.Option>
              <Select.Option value="api">API调用 (API) - 第三方服务接口</Select.Option>
              <Select.Option value="generation">生成类 (Generation) - 内容生成工具</Select.Option>
              <Select.Option value="general">通用 (General) - 其他功能</Select.Option>
            </Select>
          </Form.Item>
        </Form>
      </Modal>

      {/* 查看工具列表模态框 */}
      <Modal
        title="可用工具列表"
        open={!!viewingTools}
        onCancel={() => setViewingTools(null)}
        footer={[
          <Button key="close" onClick={() => setViewingTools(null)}>
            关闭
          </Button>,
        ]}
        width={isMobile ? '100%' : 700}
      >
        {viewingTools && (
          <Space direction="vertical" size="middle" style={{ width: '100%' }}>
            {viewingTools.tools.length === 0 ? (
              <Empty description="该插件没有提供任何工具" />
            ) : (
              viewingTools.tools.map((tool, index) => (
                <Card key={index} size="small" style={{ borderRadius: 8 }}>
                  <Descriptions column={1} size="small">
                    <Descriptions.Item label="工具名称">
                      <Text code strong>
                        {tool.name}
                      </Text>
                    </Descriptions.Item>
                    {tool.description && (
                      <Descriptions.Item label="描述">{tool.description}</Descriptions.Item>
                    )}
                    {tool.inputSchema && (
                      <Descriptions.Item label="输入参数">
                        <pre
                          style={{
                            margin: 0,
                            padding: 8,
                            background: '#f5f5f5',
                            borderRadius: 4,
                            fontSize: isMobile ? '11px' : '12px',
                            overflow: 'auto',
                          }}
                        >
                          {JSON.stringify(tool.inputSchema, null, 2)}
                        </pre>
                      </Descriptions.Item>
                    )}
                  </Descriptions>
                </Card>
              ))
            )}
          </Space>
        )}
      </Modal>
    </div>
  );
}