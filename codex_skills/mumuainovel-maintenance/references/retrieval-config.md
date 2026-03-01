# 检索 / Embedding 配置（MuMuAINovel）

## 存放位置

用户检索配置存放在：

- `settings.preferences`（JSON 文本）
- 路径：`preferences.retrieval`

查看某个用户的检索配置：

```bash
docker compose exec -T postgres bash -lc "PGPASSWORD=123456 psql -U mumuai -d mumuai_novel -c \
\"select user_id, (preferences::json->'retrieval') as retrieval from settings where user_id='USER_ID';\""
```

## collection 隔离规则（为什么会出现“换模型就换库”）

Chroma collection 名称由以下哈希生成：

- `user_id` → `sha256(user_id)[:8]`
- `project_id` → `sha256(project_id)[:8]`
- `embed_id`（可选）→ `sha256(embed_id)[:8]`

规则：

- 本地 embedding：`u_{user_hash}_p_{project_hash}`
- 远端 embedding：`u_{user_hash}_p_{project_hash}_e_{embed_hash}`

`embed_id` 的构造格式：

```
remote:{provider}:{api_base_url}:{model}
```

这样做是为了避免你切换模型/服务商后出现向量维度不一致的问题。

## 为什么切换设置后“向量像消失了”

当你修改了远端 embedding 的 provider/model/api_base_url，系统会写入到**新的 collection**。
旧数据通常并没有丢，而是仍在旧的 collection 名字里。

用 `scripts/collection_name.py` 可以预测对应的 collection 名称，快速定位“数据在哪个库里”。
