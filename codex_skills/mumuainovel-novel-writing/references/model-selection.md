# 模型选择（MuMuAINovel 写作流水线）

目标：在“长篇稳定输出 + 需要 MCP 工具（Function Calling）”之间做平衡。

## 1) 硬约束

1. **用户要求：不要使用 GLM-5**（尤其是部分渠道的 GLM-5 变体可能不支持 Function Calling）。
2. 只要 `enable_mcp=true`（章节生成/向导/分析可能会用到外部 MCP 插件），就必须选 **支持 Function Calling** 的模型。

> 若模型不支持 FC：要么切换模型，要么把 `enable_mcp=false`（但连续性与资料检索会变弱）。

## 2) 实用分工（推荐）

> 以本仓库评估报告（`docs/模型渠道能力评估_2026-02-25.md`）为参考；实际可用性以你当前 `/api/settings/models` 与连通测试为准。

- **大纲生成/修补（outline_generate / outline_expand）**
  - 优先：Claude Sonnet 系（结构稳定、指令跟随好）
  - 备选：Gemini Flash 系（速度快）

- **章节分析（chapter_analysis）**
  - 优先：Gemini Flash 系（速度快 + FC 支持常见）

- **正文生成（chapter_generate / batch_generate）**
  - 优先：稳定渠道的 Claude / Gemini
  - 备选：DeepSeek v3.x（若你的渠道可用且稳定）

## 3) 选择策略（当你不知道用哪个）

1. 先选 **Gemini Flash** 或 **Claude Sonnet**（稳定、FC 常见支持）。
2. 如果正文“味道不够/太保守”，再尝试 DeepSeek v3.x 做正文主力（但要配健康检查/熔断）。
3. 若出现以下问题，立刻切换模型或降级：
   - 空响应 / choices 为空
   - 超时频繁
   - 429 限流
   - 不支持 Function Calling（开启 MCP 时会直接影响工具链）

## 4) 与 max_tokens 的关系

- 长篇正文不要靠“无限 max_tokens”硬顶；更稳的方式是：
  - 单次目标字数控制在 3000~6000（必要时分场景续写）
  - 通过“补写/续写锚点”消化长段输出（见 `references/token-continuation.md`）

