import { Modal, Button, Space } from 'antd';
import { useEffect, useState } from 'react';

interface AnnouncementModalProps {
  visible: boolean;
  onClose: () => void;
  onDoNotShowToday: () => void;
  onNeverShow: () => void;
}

export default function AnnouncementModal({ visible, onClose, onDoNotShowToday, onNeverShow }: AnnouncementModalProps) {
  const [qqImageError, setQqImageError] = useState(false);
  const [wxImageError, setWxImageError] = useState(false);

  useEffect(() => {
    if (visible) {
      setQqImageError(false);
      setWxImageError(false);
    }
  }, [visible]);

  const handleDoNotShowToday = () => {
    onDoNotShowToday();
    onClose();
  };

  const handleNeverShow = () => {
    onNeverShow();
    onClose();
  };

  return (
    <Modal
      title="🎉 欢迎使用 AI小说创作助手"
      open={visible}
      onCancel={onClose}
      footer={
        <Space style={{ width: '100%', justifyContent: 'center' }}>
          <Button onClick={handleDoNotShowToday} size="large">
            今日内不再展示
          </Button>
          <Button type="primary" onClick={handleNeverShow} size="large">
            永不再展示
          </Button>
        </Space>
      }
      width={800}
      centered
      styles={{
        body: {
          padding: '24px',
        },
      }}
    >
      <div style={{ textAlign: 'center' }}>
        <div style={{
          marginBottom: '16px',
          fontSize: '16px',
          color: 'var(--color-text-secondary)',
          lineHeight: '1.6',
        }}>
          <p>👋 欢迎加入我们的交流群！</p>
          <p>在这里你可以：</p>
          <ul style={{
            textAlign: 'left',
            marginLeft: '40px',
            marginTop: '12px',
            marginBottom: '20px',
          }}>
            <li>💬 与其他创作者交流心得</li>
            <li>💡 获取最新功能更新和使用技巧</li>
            <li>🐛 反馈问题和建议</li>
            <li>📚 分享创作经验和灵感</li>
          </ul>
          <p style={{ fontWeight: 600, color: 'var(--color-text-primary)', marginBottom: '16px' }}>
            扫描下方二维码加入交流群：
          </p>
        </div>

        <div style={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'flex-start',
          gap: '24px',
          padding: '20px',
          background: 'var(--color-bg-layout)',
          borderRadius: '8px',
          flexWrap: 'wrap',
        }}>
          {/* QQ 二维码 */}
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            minWidth: '280px',
          }}>
            <p style={{ fontWeight: 600, color: 'var(--color-text-primary)', marginBottom: '12px', fontSize: '15px' }}>
              QQ交流群
            </p>
            {!qqImageError ? (
              <div style={{
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                background: 'var(--color-bg-container)',
                borderRadius: '8px',
                padding: '8px',
                boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)',
              }}>
                <img
                  src="/qq.jpg"
                  alt="QQ交流群二维码"
                  style={{
                    maxWidth: '280px',
                    maxHeight: '280px',
                    width: 'auto',
                    height: 'auto',
                    display: 'block',
                    objectFit: 'contain',
                  }}
                  onError={() => setQqImageError(true)}
                />
              </div>
            ) : (
              <div style={{
                width: '280px',
                height: '280px',
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                background: 'var(--color-bg-container)',
                borderRadius: '8px',
                color: '#999',
              }}>
                <p>二维码加载失败</p>
              </div>
            )}
          </div>

          {/* 微信二维码 */}
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            minWidth: '280px',
          }}>
            <p style={{ fontWeight: 600, color: 'var(--color-text-primary)', marginBottom: '12px', fontSize: '15px' }}>
              微信交流群
            </p>
            {!wxImageError ? (
              <div style={{
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                background: 'var(--color-bg-container)',
                borderRadius: '8px',
                padding: '8px',
                boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)',
              }}>
                <img
                  src="/WX.png"
                  alt="微信交流群二维码"
                  style={{
                    maxWidth: '280px',
                    maxHeight: '280px',
                    width: 'auto',
                    height: 'auto',
                    display: 'block',
                    objectFit: 'contain',
                  }}
                  onError={() => setWxImageError(true)}
                />
              </div>
            ) : (
              <div style={{
                width: '280px',
                height: '280px',
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                background: 'var(--color-bg-container)',
                borderRadius: '8px',
                color: '#999',
              }}>
                <p>二维码加载失败</p>
              </div>
            )}
          </div>
        </div>

        <div style={{
          marginTop: '20px',
          padding: '12px',
          background: 'var(--color-warning-bg)',
          borderRadius: '8px',
          border: '1px solid var(--color-warning-border)',
          fontSize: '14px',
          color: 'var(--color-warning)',
        }}>
          💡 提示：选择"今日内不再展示"当天不再显示，选择"永不再展示"将永久隐藏此公告
        </div>
      </div>
    </Modal>
  );
}