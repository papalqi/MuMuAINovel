#!/usr/bin/env python3
"""
根据 user_id / project_id / embed_id 计算 MuMuAINovel 的 ChromaDB collection 名称。

用法：
  collection_name.py <user_id> <project_id> [embed_id]

说明：
  - local/default embed_id -> u_{user_hash}_p_{project_hash}
  - remote embed_id -> u_{user_hash}_p_{project_hash}_e_{embed_hash}
"""

from __future__ import annotations

import hashlib
import sys


def sha8(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:8]


def main() -> int:
    if len(sys.argv) < 3:
        print("用法: collection_name.py <user_id> <project_id> [embed_id]")
        return 2

    user_id = sys.argv[1]
    project_id = sys.argv[2]
    embed_id = sys.argv[3] if len(sys.argv) > 3 else "local"

    user_hash = sha8(user_id)
    project_hash = sha8(project_id)
    embed_norm = str(embed_id or "local")

    if embed_norm in ("local", "default"):
        name = f"u_{user_hash}_p_{project_hash}"
    else:
        embed_hash = sha8(embed_norm)
        name = f"u_{user_hash}_p_{project_hash}_e_{embed_hash}"

    print(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
