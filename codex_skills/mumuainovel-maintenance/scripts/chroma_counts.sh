#!/usr/bin/env bash
set -euo pipefail

# 在 MuMuAINovel 应用容器内列出 Chroma collections 及其条数（count）。
#
# 用法：
#   chroma_counts.sh [service_name]
#
# 示例：
#   ./chroma_counts.sh mumuainovel

SERVICE_NAME="${1:-mumuainovel}"

docker compose exec -T "$SERVICE_NAME" python - <<'PY'
import chromadb

client = chromadb.PersistentClient(path="data/chroma_db")
cols = client.list_collections()
print(f"collections={len(cols)}")
for c in cols:
    name = getattr(c, "name", None) or c.get("name")
    col = client.get_collection(name)
    print(f"{name}\t{col.count()}")
PY
