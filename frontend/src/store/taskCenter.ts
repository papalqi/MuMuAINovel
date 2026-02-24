import { create } from 'zustand';

export type UITaskStatus = 'running' | 'success' | 'failed' | 'cancelled';

export type UITaskType =
  | 'outline_generate'
  | 'outline_expand'
  | 'outline_batch_expand'
  | 'chapter_generate'
  | 'chapter_analysis'
  | 'chapter_batch_generate'
  | string;

export interface UITask {
  id: string;
  type: UITaskType;
  name: string;
  status: UITaskStatus;
  progress: number;
  message?: string;
  error?: string;
  createdAt: number;
  updatedAt: number;
  retryCount: number;
  maxRetries: number;
  retrying: boolean;
  retryAction?: () => Promise<void> | void;
}

interface CreateTaskParams {
  type: UITaskType;
  name: string;
  message?: string;
  maxRetries?: number;
  retryAction?: () => Promise<void> | void;
}

interface TaskCenterState {
  tasks: UITask[];
  createTask: (params: CreateTaskParams) => string;
  setTaskProgress: (taskId: string, progress: number, message?: string) => void;
  markTaskSuccess: (taskId: string, message?: string) => void;
  markTaskFailed: (taskId: string, error: string) => void;
  markTaskCancelled: (taskId: string, message?: string) => void;
  updateTaskRetryAction: (
    taskId: string,
    retryAction?: () => Promise<void> | void
  ) => void;
  retryTask: (taskId: string) => Promise<boolean>;
  removeTask: (taskId: string) => void;
  clearFinishedTasks: () => void;
  clearAllTasks: () => void;
}

const MAX_TASKS = 200;

const buildTaskId = () =>
  `task_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;

const normalizeError = (error: unknown): string => {
  if (typeof error === 'string') return error;
  if (error instanceof Error) return error.message;
  try {
    return JSON.stringify(error);
  } catch {
    return '未知错误';
  }
};

export const useTaskCenterStore = create<TaskCenterState>((set, get) => ({
  tasks: [],

  createTask: ({ type, name, message, maxRetries = 5, retryAction }) => {
    const taskId = buildTaskId();
    const now = Date.now();

    const newTask: UITask = {
      id: taskId,
      type,
      name,
      status: 'running',
      progress: 0,
      message: message || '任务已创建',
      createdAt: now,
      updatedAt: now,
      retryCount: 0,
      maxRetries,
      retrying: false,
      retryAction,
    };

    set((state) => ({
      tasks: [newTask, ...state.tasks].slice(0, MAX_TASKS),
    }));

    return taskId;
  },

  setTaskProgress: (taskId, progress, message) => {
    const now = Date.now();
    set((state) => ({
      tasks: state.tasks.map((task) =>
        task.id === taskId
          ? {
              ...task,
              status: 'running',
              progress: Math.max(0, Math.min(100, progress)),
              message: message ?? task.message,
              error: undefined,
              updatedAt: now,
            }
          : task
      ),
    }));
  },

  markTaskSuccess: (taskId, message) => {
    const now = Date.now();
    set((state) => ({
      tasks: state.tasks.map((task) =>
        task.id === taskId
          ? {
              ...task,
              status: 'success',
              progress: 100,
              message: message ?? task.message ?? '任务完成',
              error: undefined,
              retrying: false,
              updatedAt: now,
            }
          : task
      ),
    }));
  },

  markTaskFailed: (taskId, error) => {
    const now = Date.now();
    set((state) => ({
      tasks: state.tasks.map((task) =>
        task.id === taskId
          ? {
              ...task,
              status: 'failed',
              error,
              message: task.message || '任务失败',
              retrying: false,
              updatedAt: now,
            }
          : task
      ),
    }));
  },

  markTaskCancelled: (taskId, message) => {
    const now = Date.now();
    set((state) => ({
      tasks: state.tasks.map((task) =>
        task.id === taskId
          ? {
              ...task,
              status: 'cancelled',
              retrying: false,
              message: message ?? task.message ?? '任务已取消',
              updatedAt: now,
            }
          : task
      ),
    }));
  },

  updateTaskRetryAction: (taskId, retryAction) => {
    const now = Date.now();
    set((state) => ({
      tasks: state.tasks.map((task) =>
        task.id === taskId
          ? {
              ...task,
              retryAction,
              updatedAt: now,
            }
          : task
      ),
    }));
  },

  retryTask: async (taskId) => {
    const task = get().tasks.find((item) => item.id === taskId);
    if (!task || !task.retryAction) return false;

    if (task.retryCount >= task.maxRetries) {
      get().markTaskFailed(taskId, `已超过最大重试次数（${task.maxRetries}）`);
      return false;
    }

    const now = Date.now();
    set((state) => ({
      tasks: state.tasks.map((item) =>
        item.id === taskId
          ? {
              ...item,
              status: 'running',
              progress: 0,
              error: undefined,
              retrying: true,
              retryCount: item.retryCount + 1,
              message: `正在重试（${item.retryCount + 1}/${item.maxRetries}）...`,
              updatedAt: now,
            }
          : item
      ),
    }));

    try {
      await Promise.resolve(task.retryAction());
      return true;
    } catch (error) {
      get().markTaskFailed(taskId, normalizeError(error));
      return false;
    } finally {
      const finallyNow = Date.now();
      set((state) => ({
        tasks: state.tasks.map((item) =>
          item.id === taskId
            ? {
                ...item,
                retrying: false,
                updatedAt: finallyNow,
              }
            : item
        ),
      }));
    }
  },

  removeTask: (taskId) => {
    set((state) => ({
      tasks: state.tasks.filter((task) => task.id !== taskId),
    }));
  },

  clearFinishedTasks: () => {
    set((state) => ({
      tasks: state.tasks.filter((task) => task.status === 'running'),
    }));
  },

  clearAllTasks: () => {
    set({ tasks: [] });
  },
}));

