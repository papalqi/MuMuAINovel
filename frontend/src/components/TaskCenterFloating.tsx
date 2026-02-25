import { useMemo, useState } from 'react';
import {
  Badge,
  Button,
  Drawer,
  Empty,
  Grid,
  List,
  Progress,
  Space,
  Tag,
  Tooltip,
  Typography,
} from 'antd';
import {
  CheckCircleOutlined,
  ClearOutlined,
  ClockCircleOutlined,
  CloseCircleOutlined,
  DeleteOutlined,
  ReloadOutlined,
  SyncOutlined,
  UnorderedListOutlined,
} from '@ant-design/icons';
import { useTaskCenterStore } from '../store/taskCenter';

const { Text } = Typography;

const statusMeta = {
  running: {
    icon: <SyncOutlined spin />,
    color: 'processing',
    text: '进行中',
  },
  success: {
    icon: <CheckCircleOutlined />,
    color: 'success',
    text: '成功',
  },
  failed: {
    icon: <CloseCircleOutlined />,
    color: 'error',
    text: '失败',
  },
  cancelled: {
    icon: <ClockCircleOutlined />,
    color: 'default',
    text: '已取消',
  },
} as const;

const typeNameMap: Record<string, string> = {
  outline_generate: '大纲生成',
  outline_expand: '大纲展开',
  outline_batch_expand: '批量大纲展开',
  chapter_generate: '章节生成',
  chapter_analysis: '章节分析',
  chapter_batch_generate: '批量章节生成',
};

const getTypeLabel = (type: string) => typeNameMap[type] || type;

export default function TaskCenterFloating() {
  const [visible, setVisible] = useState(false);
  const screens = Grid.useBreakpoint();
  const isMobile = !screens.md;
  const tasks = useTaskCenterStore((state) => state.tasks);
  const retryTask = useTaskCenterStore((state) => state.retryTask);
  const removeTask = useTaskCenterStore((state) => state.removeTask);
  const clearFinishedTasks = useTaskCenterStore((state) => state.clearFinishedTasks);
  const clearAllTasks = useTaskCenterStore((state) => state.clearAllTasks);

  const failedCount = useMemo(
    () => tasks.filter((task) => task.status === 'failed').length,
    [tasks]
  );

  const runningCount = useMemo(
    () => tasks.filter((task) => task.status === 'running').length,
    [tasks]
  );

  const hasFinishedTasks = useMemo(
    () => tasks.some((task) => task.status !== 'running'),
    [tasks]
  );

  return (
    <>
      <div
        style={{
          position: 'fixed',
          right: isMobile ? 16 : 24,
          bottom: `calc(env(safe-area-inset-bottom, 0px) + ${isMobile ? 16 : 24}px)`,
          zIndex: 1200,
        }}
      >
        <Badge count={failedCount} size="small">
          <Tooltip title="任务中心">
            <Button
              type="primary"
              shape="circle"
              size="large"
              icon={<UnorderedListOutlined />}
              onClick={() => setVisible(true)}
            />
          </Tooltip>
        </Badge>
      </div>

      <Drawer
        title={
          <Space wrap size={[8, 8]}>
            <span>任务中心</span>
            <Tag color="processing">进行中 {runningCount}</Tag>
            <Tag color={failedCount > 0 ? 'error' : 'default'}>
              失败 {failedCount}
            </Tag>
          </Space>
        }
        open={visible}
        onClose={() => setVisible(false)}
        placement={isMobile ? 'bottom' : 'right'}
        width={isMobile ? undefined : 460}
        height={isMobile ? '72vh' : undefined}
        styles={{
          body: {
            paddingBottom: 'calc(env(safe-area-inset-bottom, 0px) + 16px)',
          },
        }}
        extra={
          <Space>
            <Tooltip title="清理已结束任务">
              <Button
                icon={<ClearOutlined />}
                onClick={clearFinishedTasks}
                disabled={!hasFinishedTasks}
              />
            </Tooltip>
            <Tooltip title="清空任务">
              <Button danger icon={<DeleteOutlined />} onClick={clearAllTasks} />
            </Tooltip>
          </Space>
        }
      >
        {tasks.length === 0 ? (
          <Empty description="暂无任务" />
        ) : (
          <List
            dataSource={tasks}
            renderItem={(task) => {
              const meta = statusMeta[task.status];
              const canRetry =
                task.status === 'failed' &&
                !!task.retryAction &&
                task.retryCount < task.maxRetries;

              const desktopActions = [
                canRetry ? (
                  <Button
                    key="retry"
                    type="link"
                    icon={<ReloadOutlined />}
                    loading={task.retrying}
                    onClick={() => retryTask(task.id)}
                  >
                    重试
                  </Button>
                ) : null,
                <Button
                  key="delete"
                  type="text"
                  danger
                  icon={<DeleteOutlined />}
                  onClick={() => removeTask(task.id)}
                />,
              ].filter(Boolean);

              return (
                <List.Item
                  actions={isMobile ? undefined : desktopActions}
                >
                  <Space direction="vertical" style={{ width: '100%' }} size={4}>
                    <Space wrap>
                      <Tag icon={meta.icon} color={meta.color}>
                        {meta.text}
                      </Tag>
                      <Tag>{getTypeLabel(task.type)}</Tag>
                    </Space>

                    <Text strong>{task.name}</Text>

                    {task.status === 'running' && (
                      <Progress
                        percent={Math.max(0, Math.min(100, task.progress || 0))}
                        size="small"
                        status="active"
                      />
                    )}

                    {task.message && (
                      <Text type="secondary" style={{ fontSize: 12 }}>
                        {task.message}
                      </Text>
                    )}

                    {task.error && (
                      <Text type="danger" style={{ fontSize: 12 }}>
                        {task.error}
                      </Text>
                    )}

                    <Text type="secondary" style={{ fontSize: 11 }}>
                      重试次数：{task.retryCount}/{task.maxRetries}
                    </Text>

                    {isMobile && (
                      <Space size={8} style={{ marginTop: 4 }}>
                        {canRetry && (
                          <Button
                            type="default"
                            size="small"
                            icon={<ReloadOutlined />}
                            loading={task.retrying}
                            onClick={() => retryTask(task.id)}
                          >
                            重试
                          </Button>
                        )}
                        <Button
                          danger
                          size="small"
                          icon={<DeleteOutlined />}
                          onClick={() => removeTask(task.id)}
                        >
                          删除
                        </Button>
                      </Space>
                    )}
                  </Space>
                </List.Item>
              );
            }}
          />
        )}
      </Drawer>
    </>
  );
}
