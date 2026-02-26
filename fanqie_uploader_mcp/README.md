# Fanqie Uploader MCP (Playwright)

> 目标：先把 **MCP 服务跑通**（不改 MuMuAINovel 后端/前端代码），后续再把 MuMu 章节内容接入到这个 MCP 服务里实现“番茄作家后台：创建作品 + 保存草稿 + 覆盖章节”。

## 1. 能力范围（当前阶段）

本服务提供一组 MCP Tools（streamable_http）：

- `fanqie_health`：健康检查
- `fanqie_login_start`：打开番茄作家后台登录页并截图（用于手机扫码）
- `fanqie_login_poll`：轮询登录状态，登录成功后保存 `storage_state.json`
- `fanqie_login_cancel`：取消/关闭登录会话
- `fanqie_debug_open`：用已保存的登录态打开任意 URL 并截图（用于调试选择器/页面变化）
- `fanqie_work_create_draft`：创建作品（**尽力而为**的 UI 自动化；失败会返回截图，方便迭代）
- `fanqie_chapter_upsert_draft`：新增/覆盖章节并“保存草稿”（**尽力而为**的 UI 自动化；失败会返回截图，方便迭代）

> 说明：番茄作家后台属于强风控站点，UI/选择器/流程随时会变化，因此本项目把失败截图/trace 作为一等公民，便于快速修复。

## 2. Docker 运行（推荐）

### 2.1 构建镜像

在仓库根目录执行：

```bash
docker build -t fanqie-uploader-mcp:dev -f fanqie_uploader_mcp/Dockerfile .
```

### 2.2 启动容器

```bash
docker run -d --name fanqie-uploader-mcp \
  -p 9001:9001 \
  -v $(pwd)/.fanqie_uploader_data:/data \
  -e FANQIE_DATA_DIR=/data \
  fanqie-uploader-mcp:dev
```

健康检查：

```bash
curl http://127.0.0.1:9001/health
```

MCP 连接地址（streamable_http）：

```
http://127.0.0.1:9001/mcp/mcp
```

## 3. 在 MuMu 里注册为 MCP 插件（无需改代码）

MuMu UI -> MCP 插件管理 -> 新增插件：

- type：`streamable_http`
- url：`http://<fanqie-uploader-mcp所在主机>:9001/mcp/mcp`

然后在插件测试里调用 `fanqie_health` / `fanqie_login_start` 等工具。

## 4. 数据目录

`FANQIE_DATA_DIR`（默认 `/data`）下会生成：

- `state/`：登录态文件（storage_state），**相当于账号凭证**，请妥善保护
- `shots/`：失败/调试截图
- `traces/`：可选的 Playwright tracing 文件

