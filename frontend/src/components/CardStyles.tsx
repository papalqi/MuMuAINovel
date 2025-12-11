import type { CSSProperties } from 'react';

// 统一的卡片样式配置
export const cardStyles = {
  // 基础卡片样式
  base: {
    borderRadius: 12,
    transition: 'all 0.3s ease',
  } as CSSProperties,

  // 悬浮效果
  hoverable: {
    cursor: 'pointer',
  } as CSSProperties,

  // 角色卡片样式
  character: {
    // height: 320,
    display: 'flex',
    flexDirection: 'column',
    borderColor: 'var(--color-info)',
    borderRadius: 12,
  } as CSSProperties,

  // 组织卡片样式
  organization: {
    // height: 320,
    display: 'flex',
    flexDirection: 'column',
    borderColor: 'var(--color-success)',
    backgroundColor: 'var(--color-bg-base)', // 使用柔和的背景色
    borderRadius: 12,
  } as CSSProperties,

  // 项目卡片样式 - 现代化设计
  project: {
    height: '100%',
    borderRadius: 20,
    overflow: 'hidden',
    background: 'var(--color-bg-container)',
    boxShadow: '0 4px 20px rgba(0, 0, 0, 0.08), 0 1px 3px rgba(0, 0, 0, 0.04)',
    transition: 'all 0.35s cubic-bezier(0.4, 0, 0.2, 1)',
    border: '1px solid rgba(0, 0, 0, 0.04)',
  } as CSSProperties,

  // 卡片内容区域样式
  body: {
    padding: 20,
    display: 'flex',
    flexDirection: 'column' as const,
  } as CSSProperties,

  // 卡片描述区域样式（固定高度，内容截断）
  description: {
    marginTop: 12,
    maxHeight: 200,
    overflow: 'hidden' as const,
  } as CSSProperties,

  // 文本截断样式
  ellipsis: {
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap' as const,
  } as CSSProperties,

  // 多行文本截断
  ellipsisMultiline: (lines: number = 2) => ({
    display: '-webkit-box',
    WebkitLineClamp: lines,
    WebkitBoxOrient: 'vertical' as const,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
  } as CSSProperties),
};

// 卡片悬浮动画 - 增强版
export const cardHoverHandlers = {
  onMouseEnter: (e: React.MouseEvent<HTMLDivElement>) => {
    const target = e.currentTarget;
    target.style.transform = 'translateY(-10px) scale(1.01)';
    target.style.boxShadow = '0 20px 40px rgba(77, 128, 136, 0.2), 0 8px 16px rgba(0, 0, 0, 0.08)';
  },
  onMouseLeave: (e: React.MouseEvent<HTMLDivElement>) => {
    const target = e.currentTarget;
    target.style.transform = 'translateY(0) scale(1)';
    target.style.boxShadow = '0 4px 20px rgba(0, 0, 0, 0.08), 0 1px 3px rgba(0, 0, 0, 0.04)';
  },
};

// 响应式网格配置
export const gridConfig = {
  gutter: [16, 16] as [number, number],
  xs: 24,
  sm: 12,
  lg: 8,
  xl: 6,
};

// 角色卡片网格配置
export const characterGridConfig = {
  gutter: 0,  // 移除 gutter，避免负边距
  xs: 24,  // 手机：1列
  sm: 12,  // 平板：2列
  md: 12,   // 中等屏幕：3列
  lg: 6,   // 大屏：4列
  xl: 6,   // 超大屏：4列
  xxl: 5,  // 超超大屏：6列
};

// 文本样式
export const textStyles = {
  label: {
    fontSize: 12,
    color: 'rgba(0, 0, 0, 0.45)',
  } as CSSProperties,

  value: {
    fontSize: 14,
    color: 'rgba(0, 0, 0, 0.85)',
  } as CSSProperties,

  description: {
    fontSize: 12,
    color: 'rgba(0, 0, 0, 0.45)',
    lineHeight: 1.6,
  } as CSSProperties,
};