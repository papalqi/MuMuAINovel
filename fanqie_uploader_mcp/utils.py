from __future__ import annotations

import base64
import datetime as dt
import re
import uuid
from pathlib import Path
from typing import Optional, Tuple


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def now_ts() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def b64encode_bytes(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")


def ensure_parent_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def safe_filename(name: str, max_len: int = 80) -> str:
    name = (name or "").strip()
    # 注意：这里要替换“空白字符”而不是字母 s
    # - \\s: whitespace
    # - \\\\: backslash
    name = re.sub(r"[\s/\\:;*?\"<>|]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    if not name:
        name = "file"
    return name[:max_len]


def write_bytes(path: Path, data: bytes) -> str:
    ensure_parent_dir(path)
    path.write_bytes(data)
    return str(path)


def write_text(path: Path, text: str) -> str:
    ensure_parent_dir(path)
    path.write_text(text, encoding="utf-8")
    return str(path)


def parse_work_id_from_url(url: str) -> Optional[str]:
    if not url:
        return None
    m = re.search(r"/writer/zone/article/(\\d+)", url)
    return m.group(1) if m else None


def parse_chapter_id_from_url(url: str) -> Optional[str]:
    """
    兜底：不同站点可能用不同路由；这里只做弱匹配。
    """
    if not url:
        return None
    for pat in [r"/writer/zone/chapter/(\\d+)", r"/writer/zone/article/\\d+/chapter/(\\d+)"]:
        m = re.search(pat, url)
        if m:
            return m.group(1)
    return None
