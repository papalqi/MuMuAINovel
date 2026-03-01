---
name: mumuainovel-novel-writing
description: "使用 MuMuAINovel 的 UI API / MCP 服务（/mcp/mcp）按“项目向导（仅生成世界观）→（项目内按需）职业/角色/大纲生成与展开→逐章生成→章节分析与记忆索引→抽查质检→修订提示词/再生成→直到目标字数并完结”的流水线创作长篇小说；包含 Codex MCP 配置示例、模型选择（优先 Gemini/Claude/DeepSeek，避免 GLM-5 等 FC 不支持模型）、断章/Token 上限续写策略、以及大纲/人物/地点/时间线一致性审查。适用于：需要通过 mumu_* MCP 工具或直接调用 /api/** 自动化写作、补写、批量生成与审稿的场景。"
---

# MuMuAINovel 长篇写作流水线

## 0) 约束与原则（先立规矩）

- 把 `project.world_rules` 当作“圣经”；任何生成内容必须服从。
- 不要泄露任何 key/密码/Token；不要读取或打印 `.env` 内容；需要凭据时只提示用户**如何配置**。
- 默认不要用 **GLM-5**（用户明确要求）；需要启用 MCP 工具时，必须选 **支持 Function Calling** 的模型/渠道（见 `references/model-selection.md`）。

## 1) MCP 连接与登录（让 Agent 能“点 UI”）

### 1.1 配置 MCP server（Codex）

- MCP 挂载点规则：MuMu base URL + `/mcp/mcp`。
- 在 `~/.codex/config.toml` 添加/确认（示例）：

```toml
[mcp_servers.mumu]
url = "http://127.0.0.1:8000/mcp/mcp"  # 本机访问
# 或
# url = "https://mumu.papalqi.top/mcp/mcp"  # 公网/反代后的访问
```

> 你可以直接查看 `~/.codex/config.toml` 的 `mcp_servers.mumu.url` 字段确认当前 Codex 使用的 MCP URL。

### 1.2 登录拿 `access_token`

- 调 `mumu_login_local(username, password)` → 得到 `access_token`。
- 立刻 `mumu_verify_token(access_token)` 校验过期时间与 scopes（admin 才能开 debug_proxy 等）。

### 1.3 发现可用能力（先 alias，后 action）

- 优先 `mumu_list_aliases`（高频动作别名，最适合写流程）。
- 不够用再 `mumu_list_actions`（全量动作，便于覆盖未别名化的接口）。
- **仅调试**才考虑 `mumu_api_call*`（通常需要服务端显式开启 raw 透传）。

## 2) 建项目（先把“大工程”信息写进系统）

### 2.1 跑向导（仅生成世界观，且会创建项目）

- 调用（SSE）：
  - `wizard_world_building`

> 注意：`wizard_world_building` 会 **创建 Project** 并返回 `project_id`。  
> 不要先 `project_create` 再跑向导，否则会生成两个项目。

向导 body 关键字段（以 world-building 接口为准；`chapter_count/character_count/outline_mode` 仅作为项目配置字段落库，不会触发职业/角色/大纲生成）：

```json
{
  "title": "书名",
  "description": "梗概",
  "theme": "主题",
  "genre": "题材",
  "chapter_count": 500,
  "character_count": 12,
  "narrative_perspective": "第三人称",
  "target_words": 2000000,
  "outline_mode": "one-to-many",
  "enable_mcp": true,
  "model": "gemini-3-flash-preview"
}
```

- 向导完成后，用 `project_get` 读取落库结果（不要只看 SSE 文本）。
- 职业/角色/大纲：改为在项目内按需调用对应接口生成（不再属于向导阶段）。

## 3) 大纲审查（必须逐条读、逐条校验）

### 3.1 拉全量大纲（如果还没有大纲，先生成）

- 若 `outlines_list(project_id)` 为空：先调用 `outline_generate_stream` 生成一版“可审查”的大纲节点。
- `outlines_list(project_id)` → 按 `order_index` 从小到大遍历。

### 3.2 审查清单（高优先级）

按以下顺序检查（更细版本见 `references/outline-audit-checklist.md`）：

1. **基调/身份**：重生/穿越/穿书/原住民定位明确；爽点围绕身份优势落点（信息差/现代知识/改命/贵女压制）。
2. **因果链**：目标→障碍→行动→意外→结果，关键节点有“因果闭环”。
3. **节奏**：打脸/报应不要拖；单个小剧情不要过长；地图切换不要过频。
4. **人物同步**：姓名/称谓一致；成长不要无过渡；避免 OOC（前高智后恋爱脑）。
5. **地点同步**：地名统一；转场理由充分；旧角色/主线牵引新地图。
6. **时间线同步**：年龄/年份/季节/行程自洽；避免“正文与大纲硬设定冲突”。

### 3.3 修正方式（优先小修，避免推倒重来）

- 轻微逻辑问题：用 `outline_generate_stream` 做“按现有结构补洞/改错/补动机”。
- 角色/组织缺失：跑 `outline_postcheck_stream` 自动补齐卡片。
- 需要扩容章节数（长篇 200W 目标）：用 `outline_batch_expand_stream`（one-to-many）生成章节规划，并设置 `auto_create_chapters=true` 落库。

## 4) 章节生成闭环（生成→分析→索引→质检→继续）

### 4.1 生成前置：确认章节已存在

向导只生成世界观，不再自动创建章节。你需要先把“大纲 → 章节记录”落库：

- `outline_mode=one-to-one`：
  1) 先 `outline_generate_stream` 生成大纲  
  2) 再对每个 outline 调 `POST /api/outlines/{outline_id}/create-single-chapter` 创建对应章节（如无 alias，用 `mumu_list_actions` 找 action 后 `mumu_call_action` 调用）
- `outline_mode=one-to-many`：用 `outline_batch_expand_stream`（建议 `auto_create_chapters=true`）创建章节（否则批量生成会找不到章节）。

### 4.2 单章流式生成（适合精控）

1. `chapter_generate_stream(chapter_id)`，body 示例：

```json
{
  "target_word_count": 5000,
  "enable_mcp": true,
  "model": "gemini-3-flash-preview"
}
```

2. `chapter_generate_stream` **会自动创建分析任务并在后台启动分析**（SSE result 里会返回 `analysis_task_id`，并会发 `analysis_started` 事件）。
   - 因此**不要立刻再手动调用** `chapter_analyze`，避免重复分析（浪费成本、且可能重复触发职业/关系/伏笔自动更新）。
3. 用 `chapter_analysis_status(chapter_id)` 轮询等待分析完成（必要时在分析完成后再写下一章，连贯性最好）。
4. 若分析失败/向量写入异常（`vector_added_count=0` 或 `vector_error_message` 非空），再考虑：
   - 重新触发分析：`chapter_analyze`
   - 或仅重建向量：`POST /api/chapters/{chapter_id}/memories/reindex`

### 4.3 批量顺序生成（适合跑量，但要控风险）

- `chapter_batch_generate_start(project_id)`（一次最多 20 章）：
  - `enable_analysis=true`（强烈建议开：否则后续章节缺乏检索上下文，漂移概率上升）
  - 选稳定模型（见 `references/model-selection.md`）
- 轮询：`chapter_batch_generate_status(batch_id)`。
- 失败策略（先保连续性，再保速度）：
  1. **优先切换模型**（尤其是 429/限流/空响应场景），保持原 `target_word_count` 再试一次
  2. 仍失败：对失败章用单章 `chapter_generate_stream` 重跑（更容易控超时/重试），并确保分析已完成后再继续下一批
  3. 最后兜底（确实卡死且你允许调整策略时）：把一章拆成“续写/补写多段”完成（见第 5 节），而不是简单粗暴降字数

> 注意：批量生成如果 `enable_analysis=true`，后端会在每章生成后**自动触发分析**；不要再额外手动 `chapter_analyze`，避免重复跑分析（浪费成本且可能引入不一致的职业/关系自动更新）。

### 4.4 分析与检索一致性（必须过关再写下一章）

用 `chapter_analysis_status(chapter_id)` 检查：
- `has_analysis_result`
- `memories_db_count`
- `vector_expected_count / vector_added_count / vector_skipped_count / vector_error_message`

若出现“分析完成但向量不全/为空”，调用：
- `POST /api/chapters/{chapter_id}/memories/reindex`

该接口当前**未必有 alias**：用 `mumu_list_actions` 找到对应 action，再 `mumu_call_action` 调用。

## 5) 断章/“戒断”处理（Token 上限与续写对齐）

更细策略见 `references/token-continuation.md`。这里给最短流程：

1. 判断原因：命中输出上限 / `max_tokens` 太低 / SSE 断连 / 超时重试截断。
2. 默认选择：**补写**（除非风格/逻辑崩坏才整章重写）。
3. 补写提示词要点：
   - 取最后 200~500 字作“续写锚点”（包含未完句子）
   - 明确要求：“从锚点后一两句开始继续；先对齐语气/人称；不要复述已写内容；不要改变既定设定”
4. 生成后做一次“重复段落”清理，再更新章节内容。

## 6) 抽查质检与提示词调优（长篇必做）

- 每 5~10 章抽查一次：
  - 人设是否 OOC
  - 爽点是否兑现，报应是否及时
  - 称谓/地名/时间线是否一致
  - 单剧情是否拖沓
- 发现系统性问题时：
  - 修改“系统提示词/写作风格模板”
  - 只重写受影响章节范围，避免全线返工
- 需要深度一致性修复/索引异常时：直接参考 `mumuainovel-maintenance` skill。

## 7) 外网调用本机 MCP（只给正确做法）

- 不要试图“定死本机 IP”来长期使用；优先用 **域名 + 反向代理/隧道**（Nginx / Cloudflare Tunnel / frp）。
- 暴露路径：`https://<your-domain>/mcp/mcp`。
- 生产安全：
  - 设置 `MCP_SERVER_SECRET`
  - 限流 + IP 白名单
  - 不要开启 raw 透传（除非短期调试）

## 8) 完结与导出

- 以 `project.current_words >= project.target_words` 为量化完成条件。
- 最后 10% 进入“收束模式”：集中回收伏笔、解决主线冲突、给角色结局。
- 导出：调用 `/api/projects/{project_id}/export`（通常需要用 actions 调用，因为未必有 alias）。
