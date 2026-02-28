"""Organization/Member input sanitizers.

AI 生成的数据经常会把字段写成“active（额外说明）”这类带注释的长字符串，
而数据库字段（例如 organization_members.status）长度较短（VARCHAR(20)），
会导致写库失败并回滚整个流程（例如大纲生成的后处理阶段）。

这里提供统一的清洗函数，供 API / Service 侧在写库前调用。
"""

from __future__ import annotations

import re
from typing import Any, Optional


ALLOWED_MEMBER_STATUS = {"active", "retired", "expelled", "deceased"}


def normalize_member_status(value: Any, default: str = "active") -> str:
    """Normalize OrganizationMember.status to a safe DB value.

    Rules:
    - Accept allowed values directly.
    - If value contains allowed token (e.g. "active（xxx）"), extract it.
    - Map some common Chinese descriptors to allowed values.
    - Fallback to default.
    """

    if value is None:
        return default

    raw = str(value).strip()
    if not raw:
        return default

    lowered = raw.lower().strip()
    if lowered in ALLOWED_MEMBER_STATUS:
        return lowered[:20]

    # Extract allowed token embedded in a longer string.
    m = re.search(r"\b(active|retired|expelled|deceased)\b", lowered)
    if m:
        return m.group(1)[:20]

    # Common Chinese mappings.
    cn_map = {
        "在职": "active",
        "现役": "active",
        "仍在": "active",
        "核心": "active",
        "成员": "active",
        "退役": "retired",
        "退休": "retired",
        "离任": "retired",
        "离开": "retired",
        "出走": "retired",
        "开除": "expelled",
        "逐出": "expelled",
        "除名": "expelled",
        "叛逃": "expelled",
        "叛变": "expelled",
        "死亡": "deceased",
        "已死": "deceased",
        "身亡": "deceased",
    }
    for k, v in cn_map.items():
        if k in raw:
            return v[:20]

    return str(default).strip().lower()[:20] or "active"


def normalize_required_short_text(value: Any, max_len: int, default: str) -> str:
    """Normalize short required text field (non-nullable DB column)."""

    if value is None:
        return default[:max_len]

    s = str(value).strip()
    if not s:
        return default[:max_len]

    s = re.sub(r"\s+", " ", s)
    return s[:max_len]


def normalize_optional_short_text(value: Any, max_len: int) -> Optional[str]:
    """Normalize short optional text field (nullable DB column)."""

    if value is None:
        return None

    s = str(value).strip()
    if not s:
        return None

    s = re.sub(r"\s+", " ", s)
    return s[:max_len]

