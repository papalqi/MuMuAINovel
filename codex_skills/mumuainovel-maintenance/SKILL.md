---
name: mumuainovel-maintenance
description: "维护与调试 MuMuAINovel（FastAPI + Postgres + ChromaDB）：定位并修复数据库层的设定/时间线不一致（projects/outlines/chapters/characters/story_memories/plot_analysis）；校验/重建 Chroma 向量索引（本地/远端 embedding）并处理远端 embedding 异常（400 内容拦截、429 限流、200 空返回/数量不匹配等）；确保 Chroma 通过 docker-compose volume 持久化；在 analysis/status 与前端标记并暴露‘部分成功/索引失败’，提供手动重试入口。用于排障/修复，不用于从零创作写作流程（创作请用 mumuainovel-novel-writing）。"
---

# MuMuAINovel 维护技能

## 快速检查清单（精简）

1. 确认/获取 `user_id`、`project_id`、受影响的章节号（尤其已发布章节）。
2. 先确认问题是否真实存在于数据库（不是仅前端界面 UI 显示问题）。
3. 在数据库中修复“权威事实”（projects/outlines/characters/chapters/memories/plot_analysis）。
4. 只要改动了 `story_memories.content`（或任何会参与 embedding 的文本），就要考虑重建向量索引。
5. 校验：`story_memories` 数量对比 Chroma collection 条数；并检查 `/analysis/status` 的子状态字段。
6. 若远端 embedding 拦截文本（400）或批量异常，使用“部分成功”重建索引，并记录被跳过的记忆ID（memory id）。
7. 最后再提交并推送（commit/push，如果用户要求）。

## 流程决策树

- 如果是**已发布正文与大纲/角色设定矛盾** → 走 **流程 A（一致性修复）**
- 如果是**分析显示 completed（已完成），但检索为空/记忆缺失** → 走 **流程 B（索引 + 状态诊断）**
- 如果是**远端 embedding 报错（400/429/200但空/数量不匹配）** → 走 **流程 C（远端 embedding 故障处理）**
- 如果是**前端无法暴露“部分失败”** → 走 **流程 D（UI 异常标记 + 重试入口）**

## 流程 A —— 一致性修复（时间线/设定）

### A1）推导正确值

1. 将“出生年 / 重生年 / 年龄”当作硬约束（不能随意改）。
2. 先算出一个“权威值”（示例推导）：
   - 若 `2005年` 时为 `9岁` → `出生≈1996`
   - 若“前世去世时 44 岁” → 前世年份应为 `1996+44=2040`
3. 选定一个权威时间线，并保证：大纲 + 正文 + 记忆 + 分析文本 全部一致。

### A2）定位所有受影响位置（以数据库为准）

对该项目在 Postgres 做定向搜索：

```bash
# 在 chapters.content 中搜索
docker compose exec -T postgres bash -lc "PGPASSWORD=123456 psql -U mumuai -d mumuai_novel -c \
\"select chapter_number,id from chapters where project_id='PROJECT_ID' and content like '%2035%' order by chapter_number;\""

# 在 outlines.content/structure 中搜索
docker compose exec -T postgres bash -lc "PGPASSWORD=123456 psql -U mumuai -d mumuai_novel -c \
\"select id from outlines where project_id='PROJECT_ID' and (content like '%2035%' or structure::text like '%2035%');\""

# 在 characters.background/personality 中搜索
docker compose exec -T postgres bash -lc "PGPASSWORD=123456 psql -U mumuai -d mumuai_novel -c \
\"select id,name from characters where project_id='PROJECT_ID' and (background like '%2035%' or personality like '%2035%');\""

# 在 story_memories.content 中搜索
docker compose exec -T postgres bash -lc "PGPASSWORD=123456 psql -U mumuai -d mumuai_novel -c \
\"select id from story_memories where project_id='PROJECT_ID' and content like '%2035%';\""
```

如果你改了年龄，不要只搜年份，也要搜：`34岁/三十四岁` 等文本。

### A3）修补数据库（小步、可审计）

1. 优先按 project_id（必要时加 chapter_number 范围）做**小范围 UPDATE**。
2. 必须记录“到底改了哪几章 chapter_number”（因为可能已经发布，需要作者对外同步修订）。
3. UPDATE 完立刻再跑一次搜索，确保相关关键词 **0 行残留**。

### A4）判断是否需要重建向量库

满足任意一条就建议重建向量：
- `story_memories.content`
- 任何后续会参与 embedding 的文本（记忆文档 documents）

如果只改了 `chapters.content`，但没有改记忆文本，则向量重建是“可选”；但若你准备重新分析章节，仍建议重建。

## 流程 B —— “分析显示完成，但记忆/索引不对”

### B1）理解故障模式

把“章节分析”拆成多个子步骤看待：
1. 大模型分析 → `plot_analysis` 存在记录
2. 记忆提取 → `story_memories` 有记录
3. 向量写入 → Chroma collection（向量集合）里能查到对应向量

不要把 `analysis_tasks.status=completed` 当作“2 和 3 也必然成功”。

### B2）通过 API 检查任务状态

调用：
- `GET /api/chapters/{chapter_id}/analysis/status`

重点关注这些字段：
- `has_analysis_result`（PlotAnalysis 是否存在）
- `memories_db_count`（该章节 StoryMemory 条数）
- `vector_expected_count / vector_added_count / vector_skipped_count`（向量预期/写入/跳过数量）
- `vector_error_message`（向量写入异常信息）

### B3）不重跑大模型的恢复方式

若 `memories_db_count > 0` 但向量缺失/部分缺失，直接调用：
- `POST /api/chapters/{chapter_id}/memories/reindex`

该接口应做到：
1. 删除该章节已存在的向量（跨所有 collection）
2. 重新 embedding 并 upsert 写入向量
3. 返回 `{requested, added, skipped, error_message, collection}`

## 流程 C —— 远端 Embedding 故障处理（ModelScope/Jina 等）

更深入细节参见：
- `references/remote-embedding-pitfalls.md`
- `references/retrieval-config.md`（embedding 配置如何映射到 Chroma collections）

### C1）识别常见远端失败

- `HTTP 400` 且包含 “inappropriate content” → 被内容安全拦截
- `HTTP 429` → 限流
- `HTTP 200` 但 `data=[]` 或数量不匹配 → 服务端 bug / 隐藏限制 / 请求过大 等

### C2）采用“部分成功”Embedding 策略

实现/期望这些行为：
1. 降低 batch_size（默认从 32 开始，不行就继续降）
2. 429/超时/网络异常：指数退避 + 抖动重试
3. 400/413/414：二分拆批直到定位到单条问题文本，只跳过该条
4. 始终返回写入统计：requested/added/skipped + 简短错误摘要

### C3）解释“为什么远端会拦截”

假设服务商执行内容安全策略。不要试图绕过；应当：
- 跳过被拦截文本，并记录 memory id 供人工改写
- 或者改用本地 embedding（若拦截频率过高）

## 流程 D —— 前端异常标记 + 手动重试

### D1）在章节列表标记“部分失败”

即使任务状态是 completed（已完成），只要满足任意一条也要显示告警标签：
- `has_analysis_result === false`
- `memories_db_count === 0`
- `vector_added_count < vector_expected_count`
- `vector_error_message` 非空

### D2）提供手动重试入口

必须区分两种重试：
- “重新分析” → `POST /api/chapters/{chapter_id}/analyze`（重新跑大模型 LLM 分析 + 重新生成记忆）
- “重试索引” → `POST /api/chapters/{chapter_id}/memories/reindex`（只重建向量索引，不重跑分析）

## 流程 E —— 排障用“最短闭环”验证（生成 → 分析 → 索引）

> 完整“写作流水线（向导→大纲→批量生成→抽查质检→完结）”请使用：`mumuainovel-novel-writing` skill。

用这个闭环在排障时快速验证“正文/分析/记忆/检索”是否一致：

1. 把“硬设定”写进 `project.world_rules`（人名/称谓/关系/婚约对象/时间线等），视为 P0 不可变。
2. 生成正文（单章流式或批量）。
3. 分析章节：
   - `POST /api/chapters/{chapter_id}/analyze`
4. 校验分析子步骤：
   - `GET /api/chapters/{chapter_id}/analysis/status`
   - 确保 `has_analysis_result=true`、`memories_db_count>0`、`vector_added_count==vector_expected_count`
5. 若向量写入不完整：
   - `POST /api/chapters/{chapter_id}/memories/reindex`
6. 满足以上条件后再写下一章（保证检索上下文可用）。

## 验证命令（Postgres + Chroma）

### 校验项目 StoryMemory 数量

```bash
docker compose exec -T postgres bash -lc "PGPASSWORD=123456 psql -U mumuai -d mumuai_novel -c \
\"select count(*) from story_memories where project_id='PROJECT_ID';\""
```

### 列出 Chroma collections 与条数（在 app 容器里）

```bash
docker compose exec -T mumuainovel python - <<'PY'
import chromadb
client = chromadb.PersistentClient(path='data/chroma_db')
for c in client.list_collections():
    col = client.get_collection(c.name)
    print(c.name, col.count())
PY
```

### 确认 Chroma 持久化挂载存在

检查 `docker-compose.yml` 是否包含：
- `./data/chroma_db:/app/data/chroma_db`

## 常见坑（踩过的）

1. 别假设远端 embedding 真“完全 OpenAI 兼容”，可能出现：200 但 `data=[]`、隐藏批量上限、返回数量不匹配等。
2. 别让一条被拦截的记忆导致整批失败；应拆分定位后“仅跳过该条”。
3. 重建索引/重复写入时不要用 `collection.add`（可能因重复 ID 整批失败）；用 `upsert`。
4. 即使向量未写入也可能显示 completed；必须记录向量统计并在前端展示出来。
5. 若未挂载 `/app/data/chroma_db`，容器重建会丢向量库。
6. “看起来向量被清空”很多时候只是：embedding 配置变了 → 写入到不同 collection。
7. 修改已发布章节必须记录具体 chapter_number（否则作者无法对外同步修订范围）。

## 代码触点（改哪里）

- 后端：
  - `backend/app/services/memory_service.py`（embedding + 批量写入 + 部分成功）
  - `backend/app/api/chapters.py`（analysis/status 子状态字段、reindex 接口）
  - `backend/app/models/analysis_task.py` + alembic migrations（向量统计持久化）
- 前端：
  - `frontend/src/pages/Chapters.tsx`（告警标签）
  - `frontend/src/components/ChapterAnalysis.tsx`（告警提示 + 重试按钮）

## 交付/发布

1. 运行 docker build，确认迁移能正常执行：
   - `docker compose up -d --build mumuainovel`
2. 确认 `/health` 正常。
3. 按需提交并推送：
   - `git status`
   - `git commit -m "..." && git push origin main`
