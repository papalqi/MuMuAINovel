from __future__ import annotations

import os
from pathlib import Path


def env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


FANQIE_BASE_URL = os.getenv("FANQIE_BASE_URL", "https://fanqienovel.com").rstrip("/")

# 默认直接打开作家专区登录页（含“扫码登录”Tab），避免停留在 /writer/zone 的落地页。
FANQIE_LOGIN_URL = os.getenv("FANQIE_LOGIN_URL", f"{FANQIE_BASE_URL}/main/writer/login?enter_from=author_zone")

DATA_DIR = Path(os.getenv("FANQIE_DATA_DIR", "/data")).resolve()
STATE_DIR = DATA_DIR / "state"
SHOT_DIR = DATA_DIR / "shots"
TRACE_DIR = DATA_DIR / "traces"

STATE_DIR.mkdir(parents=True, exist_ok=True)
SHOT_DIR.mkdir(parents=True, exist_ok=True)
TRACE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_HEADLESS = env_bool("FANQIE_HEADLESS", True)

# 登录会话在内存中保留的最大时长（秒）；超时会被自动回收
LOGIN_SESSION_TTL_SECONDS = int(os.getenv("FANQIE_LOGIN_SESSION_TTL", "900"))  # 15min

# Playwright 行为参数（可按需调优）
PW_NAV_TIMEOUT_MS = int(os.getenv("FANQIE_PW_NAV_TIMEOUT_MS", "45000"))
PW_ACTION_TIMEOUT_MS = int(os.getenv("FANQIE_PW_ACTION_TIMEOUT_MS", "15000"))
PW_DEVICE_SCALE_FACTOR = float(os.getenv("FANQIE_PW_DEVICE_SCALE_FACTOR", "2"))
PW_VIEWPORT_WIDTH = int(os.getenv("FANQIE_PW_VIEWPORT_WIDTH", "1280"))
PW_VIEWPORT_HEIGHT = int(os.getenv("FANQIE_PW_VIEWPORT_HEIGHT", "720"))

# 是否默认开启 trace（会占用磁盘；建议只在排障阶段开启）
DEFAULT_TRACE = env_bool("FANQIE_TRACE", False)
