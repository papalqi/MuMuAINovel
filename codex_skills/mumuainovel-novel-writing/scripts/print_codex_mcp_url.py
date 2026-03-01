#!/usr/bin/env python3
"""
打印 Codex 配置中的 MCP server URL（只读，不包含任何 key/密码）。

用法：
  ./print_codex_mcp_url.py
"""

from __future__ import annotations

from pathlib import Path


def main() -> int:
    cfg = Path.home() / ".codex" / "config.toml"
    if not cfg.exists():
        print(f"[ERR] 未找到: {cfg}")
        return 2

    # Python 3.11+
    import tomllib

    data = tomllib.loads(cfg.read_text(encoding="utf-8"))
    mcp = (data.get("mcp_servers") or {}).get("mumu") or {}
    url = mcp.get("url")
    if not url:
        print("[WARN] config.toml 中未配置 [mcp_servers.mumu].url")
        return 1

    print(url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

