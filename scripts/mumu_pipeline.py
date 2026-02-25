#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MuMu 全流程自动化脚本（不改业务代码，仅调用HTTP API）

能力：
1) 可选创建 runner 用户 + 可选复制 admin 设置
2) 预设健康检查（连通性 + function-calling）
3) 模型同题材 benchmark
4) 主流程：世界观->职业->角色->大纲->展开->40章批量生成->质量评估
5) 输出 JSON + Markdown 报告

用法：
  python3 scripts/mumu_pipeline.py --config docs/mumu_pipeline.config.example.json --mode full
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


# ========================= 基础工具 =========================

def now_str() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def clean_text(t: str) -> str:
    return re.sub(r"\s+", "", t or "")


def bi_ngrams(t: str) -> List[str]:
    t = clean_text(t)
    return [t[i : i + 2] for i in range(max(0, len(t) - 1))]


def cosine(c1: Counter, c2: Counter) -> float:
    keys = set(c1) | set(c2)
    dot = sum(c1.get(k, 0) * c2.get(k, 0) for k in keys)
    n1 = math.sqrt(sum(v * v for v in c1.values()))
    n2 = math.sqrt(sum(v * v for v in c2.values()))
    return dot / (n1 * n2) if n1 and n2 else 0.0


SAMPLE_OPENINGS = [
    "哥，爸都气住院了，你就把车过户给大舅呗！上辈子他们逼我买房，榨干我积蓄，这辈子我直接断亲，不当提款机。",
    "小子站住！警察堵门，重生回到九十年代，他知道今晚会有命案，也知道谁会翻盘。",
    "今天是你跟我结婚的日子，你的聘礼就这？婚礼现场翻脸，下一秒女方秘密曝光，全场哗然。",
    "正科干部三十岁，办公室电话一响，局势突变，官场风向瞬间翻盘。",
]

URBAN_LEX = [
    "公司",
    "项目",
    "合同",
    "工地",
    "楼盘",
    "老板",
    "领导",
    "单位",
    "银行",
    "直播",
    "热搜",
    "物业",
    "投资",
    "法院",
    "警方",
    "医院",
    "办公室",
    "会议",
    "职场",
    "婚礼",
    "房子",
    "车",
    "同事",
    "客户",
    "平台",
    "账号",
    "粉丝",
    "商场",
]

SAMPLE_COUNTER = Counter()
for s in SAMPLE_OPENINGS:
    SAMPLE_COUNTER.update(bi_ngrams(s))


def score_flavor(text: str) -> Dict[str, float]:
    t = clean_text(text)
    c = Counter(bi_ngrams(t))
    sim = cosine(c, SAMPLE_COUNTER)
    first = t[:220]
    end = t[-120:]

    hook_keywords = [
        "！",
        "？",
        "突然",
        "却",
        "竟",
        "原来",
        "下一秒",
        "当场",
        "反手",
        "住院",
        "报警",
        "离婚",
        "开除",
        "热搜",
    ]
    hook = min(1.0, sum(1 for k in hook_keywords if k in first) / 5)
    urban = min(1.0, sum(t.count(k) for k in URBAN_LEX) / 18)
    cliff = 1.0 if any(k in end for k in ["？", "！", "下一章", "然而", "却", "原来", "没想到", "电话响了"]) else 0.3

    score = (0.45 * sim + 0.25 * hook + 0.2 * urban + 0.1 * cliff) * 100
    return {
        "flavor_score": round(score, 2),
        "sim": round(sim, 4),
        "hook": round(hook, 4),
        "urban": round(urban, 4),
        "cliff": round(cliff, 4),
    }


def detect_marker_leak(text: str) -> Dict[str, Any]:
    tag_pattern = re.compile(r"【[^\n]{0,20}(?:现实压迫|当场反击|代价升级|章末悬念|→)[^\n]{0,20}】")
    arrow_pattern = re.compile(r"现实压迫\s*→\s*当场反击|当场反击\s*→\s*代价升级|代价升级\s*→\s*章末悬念")
    kw_pattern = re.compile(r"现实压迫|当场反击|代价升级|章末悬念")

    tags = tag_pattern.findall(text or "")
    arrows = arrow_pattern.findall(text or "")
    kws = kw_pattern.findall(text or "")
    return {
        "has_leak": bool(tags or arrows),
        "tag_count": len(tags),
        "arrow_count": len(arrows),
        "keyword_count": len(kws),
        "sample_tag": tags[0] if tags else "",
    }


# ========================= API 客户端 =========================

class MuMuClient:
    def __init__(self, base_url: str, username: str, password: str, name: str = "client"):
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.password = password
        self.name = name
        self.s = requests.Session()
        self.logged_in = False

    def log(self, msg: str) -> None:
        print(f"[{self.name} {now_str()}] {msg}", flush=True)

    def login(self) -> None:
        r = self.s.post(
            f"{self.base_url}/api/auth/local/login",
            json={"username": self.username, "password": self.password},
            timeout=20,
        )
        if r.status_code >= 400:
            raise RuntimeError(f"{self.name} 登录失败: {r.status_code} {r.text[:200]}")
        data = r.json()
        if not data.get("success"):
            raise RuntimeError(f"{self.name} 登录失败: {data}")
        self.logged_in = True

    def _request(self, method: str, path: str, retry: int = 1, **kwargs):
        if not self.logged_in:
            self.login()
        url = f"{self.base_url}{path}"
        r = self.s.request(method, url, **kwargs)
        if r.status_code == 401 and retry > 0:
            self.log("会话失效，自动重登")
            self.login()
            return self._request(method, path, retry=retry - 1, **kwargs)
        return r

    def get(self, path: str, timeout: int = 60):
        r = self._request("GET", path, timeout=timeout)
        if r.status_code >= 400:
            raise RuntimeError(f"GET {path} 失败: {r.status_code} {r.text[:300]}")
        return r.json()

    def post(self, path: str, payload: Dict[str, Any], timeout: int = 120):
        r = self._request("POST", path, json=payload, timeout=timeout)
        if r.status_code >= 400:
            raise RuntimeError(f"POST {path} 失败: {r.status_code} {r.text[:300]}")
        return r.json()

    def put(self, path: str, payload: Dict[str, Any], timeout: int = 120):
        r = self._request("PUT", path, json=payload, timeout=timeout)
        if r.status_code >= 400:
            raise RuntimeError(f"PUT {path} 失败: {r.status_code} {r.text[:300]}")
        return r.json()

    def sse_post(self, path: str, payload: Dict[str, Any], timeout: int = 3600, tag: str = "SSE") -> Tuple[Optional[Dict[str, Any]], str]:
        """返回 (result_data, full_chunk_text)"""
        r = self._request(
            "POST",
            path,
            json=payload,
            stream=True,
            timeout=(20, timeout),
            retry=1,
        )
        if r.status_code >= 400:
            raise RuntimeError(f"SSE {path} 失败: {r.status_code} {r.text[:300]}")

        result_data = None
        content = ""
        last_p = None
        for raw in r.iter_lines(decode_unicode=True):
            if not raw:
                continue
            line = raw.strip()
            if line.startswith(":") or line.startswith("event:"):
                continue
            if not line.startswith("data:"):
                continue
            obj = json.loads(line[5:].strip())
            t = obj.get("type")

            if t == "progress":
                p = obj.get("progress")
                if p in [0, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 85, 90, 92, 95, 98, 100] and p != last_p:
                    self.log(f"[{tag}] {p}% {obj.get('message', '')}")
                    last_p = p
            elif t == "chunk":
                content += obj.get("content", "")
            elif t == "result":
                result_data = obj.get("data")
            elif t == "error":
                raise RuntimeError(f"SSE error: {obj.get('error')}")
            elif t == "done":
                break

        return result_data, content


# ========================= 流程函数 =========================

def backup_database(cfg: Dict[str, Any]) -> Optional[str]:
    backup_cfg = cfg.get("backup", {})
    if not backup_cfg.get("enabled", True):
        return None

    container = backup_cfg.get("postgres_container", "mumuainovel-postgres")
    db = backup_cfg.get("db_name", "mumuai_novel")
    user = backup_cfg.get("db_user", "mumuai")
    out_dir = Path(backup_cfg.get("output_dir", "logs/db_backups"))
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"mumuai_novel_{ts}.dump"

    cmd = ["docker", "exec", "-i", container, "pg_dump", "-U", user, "-d", db, "-Fc"]
    with open(out_path, "wb") as f:
        subprocess.run(cmd, stdout=f, check=True)

    return str(out_path)


def ensure_runner_user(admin: MuMuClient, runner_cfg: Dict[str, Any]) -> None:
    username = runner_cfg["username"]
    display_name = runner_cfg.get("display_name", username)
    password = runner_cfg["password"]

    payload = {
        "username": username,
        "display_name": display_name,
        "password": password,
        "is_admin": False,
        "trust_level": 0,
    }
    try:
        admin.post("/api/admin/users", payload)
        admin.log(f"已创建 runner 用户: {username}")
    except Exception as e:
        # 用户已存在通常会409
        admin.log(f"创建runner跳过: {e}")


def clone_admin_settings_to_runner(admin: MuMuClient, runner: MuMuClient) -> None:
    admin_settings = admin.get("/api/settings")
    payload = {
        "api_provider": admin_settings.get("api_provider"),
        "api_key": admin_settings.get("api_key"),
        "api_base_url": admin_settings.get("api_base_url"),
        "llm_model": admin_settings.get("llm_model"),
        "temperature": admin_settings.get("temperature"),
        "max_tokens": admin_settings.get("max_tokens"),
        "system_prompt": admin_settings.get("system_prompt"),
        "preferences": admin_settings.get("preferences"),
    }
    runner.post("/api/settings", payload)
    runner.log("已复制 admin settings 到 runner")


def health_check_presets(runner: MuMuClient) -> List[Dict[str, Any]]:
    presets = runner.get("/api/settings/presets").get("presets", [])
    rows = []
    for p in presets:
        cfg = p.get("config", {})
        row = {
            "name": p.get("name"),
            "preset_id": p.get("id"),
            "model": cfg.get("llm_model"),
        }
        try:
            t = runner.post(f"/api/settings/presets/{p['id']}/test", {})
            row.update(
                {
                    "test_success": t.get("success"),
                    "resp_ms": t.get("response_time_ms"),
                    "test_error": t.get("error", ""),
                }
            )
        except Exception as e:
            row.update({"test_success": False, "resp_ms": None, "test_error": str(e)})

        try:
            fc = runner.post(
                "/api/settings/check-function-calling",
                {
                    "api_key": cfg.get("api_key", ""),
                    "api_base_url": cfg.get("api_base_url", ""),
                    "provider": cfg.get("api_provider", "openai"),
                    "llm_model": cfg.get("llm_model", ""),
                    "temperature": cfg.get("temperature", 0.7),
                    "max_tokens": cfg.get("max_tokens", 2000),
                },
            )
            row.update(
                {
                    "fc_success": fc.get("success"),
                    "fc_supported": fc.get("supported"),
                    "fc_error": fc.get("error", ""),
                }
            )
        except Exception as e:
            row.update({"fc_success": False, "fc_supported": False, "fc_error": str(e)})

        rows.append(row)

    return rows


def ensure_style(runner: MuMuClient, style_cfg: Dict[str, Any]) -> int:
    name = style_cfg["name"]
    styles = runner.get("/api/writing-styles/user").get("styles", [])
    hit = next((x for x in styles if x.get("name") == name), None)
    if hit:
        return int(hit["id"])

    payload = {
        "name": name,
        "style_type": "custom",
        "description": style_cfg.get("description", "番茄都市拆书仿写风格"),
        "prompt_content": style_cfg["prompt_content"],
    }
    created = runner.post("/api/writing-styles", payload)
    return int(created["id"])


def activate_preset(runner: MuMuClient, preset_id: str) -> None:
    runner.post(f"/api/settings/presets/{preset_id}/activate", {})


def clear_ai_routes(runner: MuMuClient) -> None:
    routes = runner.get("/api/settings/ai-routes").get("routes", {})
    runner.put("/api/settings/ai-routes", {"routes": {k: None for k in routes.keys()}})


def create_benchmark_project(runner: MuMuClient, cfg: Dict[str, Any]) -> Tuple[str, str]:
    p = runner.post(
        "/api/projects",
        {
            "title": cfg.get("title", "模型对比沙盘-都市开篇"),
            "description": cfg.get("description", "用于对比模型开篇风味"),
            "theme": cfg.get("theme", "都市逆袭与商战"),
            "genre": cfg.get("genre", "都市"),
            "target_words": 20000,
            "outline_mode": "one-to-many",
        },
    )
    pid = p["id"]

    o = runner.post(
        "/api/outlines",
        {
            "project_id": pid,
            "title": "第一章 裁员当天我接手烂尾楼",
            "content": "主角被裁后接盘烂尾商业体，在债主与同事嘲讽中发现账本异常。",
            "order_index": 1,
            "structure": json.dumps({"phase": "opening", "hook": "神秘电话与账本异常"}, ensure_ascii=False),
        },
    )

    c = runner.post(
        "/api/chapters",
        {
            "project_id": pid,
            "title": "第一章 裁员当天我接手烂尾楼",
            "chapter_number": 1,
            "summary": "主角失业当天被迫接盘烂尾商业体，面对债主与前同事嘲讽，发现账本漏洞。",
            "status": "draft",
            "outline_id": o["id"],
            "sub_index": 1,
        },
    )
    return pid, c["id"]


def benchmark_models(runner: MuMuClient, style_id: int, cfg: Dict[str, Any]) -> Dict[str, Any]:
    bench_cfg = cfg.get("benchmark", {})
    pid, cid = create_benchmark_project(runner, bench_cfg)

    presets = runner.get("/api/settings/presets").get("presets", [])
    whitelist = set(bench_cfg.get("preset_whitelist", []))
    if whitelist:
        presets = [p for p in presets if p.get("name") in whitelist or p.get("id") in whitelist]

    rows = []
    target_word_count = int(bench_cfg.get("target_word_count", 900))

    for p in presets:
        name = p.get("name")
        model = p.get("config", {}).get("llm_model")
        # 明确跳过配置已知503的可选项
        if name in bench_cfg.get("skip_presets", ["豆包"]):
            rows.append({"name": name, "model": model, "status": "skip"})
            continue

        try:
            activate_preset(runner, p["id"])
            runner.sse_post(
                f"/api/chapters/{cid}/generate-stream",
                {"style_id": style_id, "target_word_count": target_word_count},
                timeout=3600,
                tag=f"benchmark:{name}",
            )
            text = runner.get(f"/api/chapters/{cid}").get("content", "")
            rows.append(
                {
                    "name": name,
                    "preset_id": p["id"],
                    "model": model,
                    "status": "ok",
                    "word_count": len(clean_text(text)),
                    "flavor": score_flavor(text),
                    "marker": detect_marker_leak(text),
                    "preview": text[:220],
                }
            )
        except Exception as e:
            rows.append({"name": name, "preset_id": p.get("id"), "model": model, "status": "error", "error": str(e)})

    ranked = [x for x in rows if x.get("status") == "ok"]
    ranked.sort(key=lambda x: x["flavor"]["flavor_score"], reverse=True)

    return {
        "timestamp": now_str(),
        "project_id": pid,
        "chapter_id": cid,
        "results": rows,
        "ranked": ranked,
        "best": ranked[0] if ranked else None,
    }


def run_batch_and_wait(
    runner: MuMuClient,
    project_id: str,
    start: int,
    count: int,
    style_id: int,
    target_word_count: int,
    poll_sec: int = 12,
) -> Dict[str, Any]:
    req = {
        "start_chapter_number": start,
        "count": count,
        "enable_analysis": True,
        "style_id": style_id,
        "target_word_count": target_word_count,
    }
    resp = runner.post(f"/api/chapters/project/{project_id}/batch-generate", req)
    batch_id = resp["batch_id"]

    last = None
    while True:
        st = runner.get(f"/api/chapters/batch-generate/{batch_id}/status")
        key = (st.get("status"), st.get("completed"), st.get("current_chapter_number"), st.get("current_retry_count"))
        if key != last:
            runner.log(
                f"batch {batch_id[:8]} status={st.get('status')} {st.get('completed')}/{st.get('total')} current={st.get('current_chapter_number')} retry={st.get('current_retry_count')}"
            )
            last = key

        if st.get("status") in ("completed", "failed", "cancelled"):
            if st.get("status") != "completed":
                raise RuntimeError(f"批量任务失败: {st}")
            return st

        time.sleep(poll_sec)


def expand_missing_outlines(runner: MuMuClient, project_id: str, chapters_per_outline: int) -> None:
    outlines = runner.get(f"/api/outlines/project/{project_id}").get("items", [])
    for o in outlines:
        ch = runner.get(f"/api/outlines/{o['id']}/chapters")
        if ch.get("chapter_count", 0) < chapters_per_outline:
            runner.log(f"补展开 outline#{o.get('order_index')} {o.get('title')}")
            runner.sse_post(
                f"/api/outlines/{o['id']}/expand-stream",
                {
                    "target_chapter_count": chapters_per_outline,
                    "expansion_strategy": "balanced",
                    "auto_create_chapters": True,
                    "enable_scene_analysis": True,
                },
                timeout=3600,
                tag=f"expand:{o.get('order_index')}",
            )


def run_main_workflow(runner: MuMuClient, style_id: int, best_preset_name: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    main_cfg = cfg.get("main_workflow", {})

    # 激活最佳预设 & 清路由
    presets = runner.get("/api/settings/presets").get("presets", [])
    best = next((p for p in presets if p.get("name") == best_preset_name), None)
    if not best:
        raise RuntimeError(f"找不到预设: {best_preset_name}")
    activate_preset(runner, best["id"])
    clear_ai_routes(runner)

    # 世界观建项
    world_payload = {
        "title": main_cfg.get("title", "裁员后，我接手烂尾CBD"),
        "description": main_cfg.get(
            "description",
            "地产产品总监被裁当天，被迫接管烂尾商业体，在债主与资本方夹击中完成翻盘。",
        ),
        "theme": main_cfg.get("theme", "都市逆袭+商战博弈+情感修复"),
        "genre": main_cfg.get("genre", "都市"),
        "narrative_perspective": main_cfg.get("narrative_perspective", "第三人称"),
        "target_words": int(main_cfg.get("target_words", 120000)),
        "chapter_count": int(main_cfg.get("chapter_target", 40)),
        "character_count": int(main_cfg.get("character_target", 12)),
        "outline_mode": "one-to-many",
    }
    world_result, _ = runner.sse_post("/api/wizard-stream/world-building", world_payload, timeout=3600, tag="world")
    project_id = world_result["project_id"]

    # 设默认风格
    runner.post(f"/api/writing-styles/{style_id}/set-default", {"project_id": project_id})

    # 职业、角色、大纲、展开
    career_result, _ = runner.sse_post("/api/wizard-stream/career-system", {"project_id": project_id}, timeout=3600, tag="career")
    char_result, _ = runner.sse_post(
        "/api/wizard-stream/characters",
        {
            "project_id": project_id,
            # ⚠️ wizard-stream/characters 读取的是 count，不是 character_count
            "count": int(main_cfg.get("character_target", 12)),
            "requirements": main_cfg.get(
                "character_requirements",
                "请以人物角色为主（组织数量不超过总数20%），并保证主角与核心配角完整。",
            ),
            "theme": main_cfg.get("theme", ""),
            "genre": main_cfg.get("genre", ""),
        },
        timeout=3600,
        tag="characters",
    )
    outline_result, _ = runner.sse_post(
        "/api/wizard-stream/outline",
        {
            "project_id": project_id,
            "chapter_count": int(main_cfg.get("outline_count", 10)),
            "narrative_perspective": main_cfg.get("narrative_perspective", "第三人称"),
            "target_words": int(main_cfg.get("target_words", 120000)),
            "requirements": main_cfg.get(
                "outline_requirements",
                "按番茄都市爽文节奏：每个大纲节点需包含现实冲突、反击与危机升级，章末留钩子。",
            ),
        },
        timeout=3600,
        tag="outline",
    )

    chapters_per_outline = int(main_cfg.get("chapters_per_outline", 4))
    expand_result, _ = runner.sse_post(
        "/api/outlines/batch-expand-stream",
        {
            "project_id": project_id,
            "chapters_per_outline": chapters_per_outline,
            "expansion_strategy": "balanced",
            "auto_create_chapters": True,
            "enable_scene_analysis": True,
        },
        timeout=7200,
        tag="expand",
    )

    # 补展开（处理非确定性）
    expand_missing_outlines(runner, project_id, chapters_per_outline)

    target = int(main_cfg.get("chapter_target", 40))
    chapters = runner.get(f"/api/chapters/project/{project_id}").get("items", [])
    chapters = sorted(chapters, key=lambda x: x["chapter_number"])
    if len(chapters) < target:
        raise RuntimeError(f"展开后章节不足: {len(chapters)} < {target}")

    # 找第一个未生成章节
    first_incomplete = None
    for c in chapters[:target]:
        if not (c.get("content") and c["content"].strip()):
            first_incomplete = c["chapter_number"]
            break

    batch_logs = []
    if first_incomplete is not None:
        curr = first_incomplete
        max_per_batch = int(main_cfg.get("batch_size", 20))
        word_target = int(main_cfg.get("chapter_word_target", 900))

        while curr <= target:
            cnt = min(max_per_batch, target - curr + 1)
            st = run_batch_and_wait(runner, project_id, curr, cnt, style_id, word_target)
            batch_logs.append(
                {
                    "start": curr,
                    "count": cnt,
                    "status": st,
                }
            )
            curr += cnt

    # 评估
    chapters = runner.get(f"/api/chapters/project/{project_id}").get("items", [])
    chapters = sorted(chapters, key=lambda x: x["chapter_number"])[:target]

    quality_rows = []
    analysis_rows = []
    for c in chapters:
        full = runner.get(f"/api/chapters/{c['id']}")
        text = full.get("content", "")
        quality_rows.append(
            {
                "chapter": c["chapter_number"],
                "id": c["id"],
                "title": c.get("title"),
                "word_count": len(clean_text(text)),
                "flavor": score_flavor(text),
                "marker": detect_marker_leak(text),
                "preview": text[:180],
            }
        )
        st = runner.get(f"/api/chapters/{c['id']}/analysis/status")
        analysis_rows.append(
            {
                "chapter": c["chapter_number"],
                "status": st.get("status"),
                "has_task": st.get("has_task"),
                "error": st.get("error_message"),
            }
        )

    avg_flavor = round(sum(x["flavor"]["flavor_score"] for x in quality_rows) / len(quality_rows), 2) if quality_rows else 0
    avg_words = round(sum(x["word_count"] for x in quality_rows) / len(quality_rows), 2) if quality_rows else 0

    completed = sum(1 for x in analysis_rows if x["status"] == "completed")
    failed = [x for x in analysis_rows if x["status"] == "failed"]
    pending = [x for x in analysis_rows if x["status"] in ("pending", "running")]

    foreshadow_stats = None
    memory_stats = None
    try:
        foreshadow_stats = runner.get(f"/api/foreshadows/projects/{project_id}/stats?current_chapter={target}")
    except Exception as e:
        foreshadow_stats = {"error": str(e)}

    try:
        memory_stats = runner.get(f"/api/memories/projects/{project_id}/stats")
    except Exception as e:
        memory_stats = {"error": str(e)}

    proj = runner.get(f"/api/projects/{project_id}")

    return {
        "timestamp": now_str(),
        "project": proj,
        "preset": {"id": best["id"], "name": best["name"], "model": best.get("config", {}).get("llm_model")},
        "style_id": style_id,
        "stage_results": {
            "world": world_result,
            "career": career_result,
            "characters": char_result,
            "outline": outline_result,
            "expand": expand_result,
            "batches": batch_logs,
        },
        "quality_summary": {
            "chapter_count": len(quality_rows),
            "avg_flavor_score": avg_flavor,
            "avg_word_count": avg_words,
            "top5": sorted(quality_rows, key=lambda x: x["flavor"]["flavor_score"], reverse=True)[:5],
            "bottom5": sorted(quality_rows, key=lambda x: x["flavor"]["flavor_score"])[:5],
            "first3": quality_rows[:3],
        },
        "analysis_summary": {
            "completed": completed,
            "failed": failed,
            "pending_or_running": pending,
        },
        "foreshadow_stats": foreshadow_stats,
        "memory_stats": memory_stats,
    }


# ========================= 报告输出 =========================

def to_md_table(headers: List[str], rows: List[List[Any]]) -> str:
    s = "| " + " | ".join(headers) + " |\n"
    s += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    for r in rows:
        s += "| " + " | ".join(str(x).replace("\n", "<br>") for x in r) + " |\n"
    return s


def build_markdown(
    cfg: Dict[str, Any],
    backup_path: Optional[str],
    health: Optional[List[Dict[str, Any]]],
    benchmark: Optional[Dict[str, Any]],
    main_report: Optional[Dict[str, Any]],
) -> str:
    lines = []
    lines.append("# MuMu 固化流程自动化执行报告\n")
    lines.append(f"- 生成时间：{now_str()}")
    lines.append(f"- 模式：{cfg.get('meta', {}).get('name', 'default')}\n")

    if backup_path:
        lines.append("## 数据安全")
        lines.append(f"- 已备份数据库：`{backup_path}`\n")

    if health is not None:
        rows = []
        for x in health:
            rows.append(
                [
                    x.get("name"),
                    x.get("model"),
                    "✅" if x.get("test_success") else "❌",
                    f"{x.get('resp_ms'):.0f}ms" if isinstance(x.get("resp_ms"), (int, float)) else "-",
                    "✅" if x.get("fc_success") else "❌",
                    "✅" if x.get("fc_supported") else "❌",
                ]
            )
        lines.append("## 预设健康检查")
        lines.append(to_md_table(["预设", "模型", "连通", "时延", "FC测试", "FC支持"], rows))
        lines.append("")

    if benchmark is not None:
        ranked = benchmark.get("ranked", [])
        rows = []
        for i, x in enumerate(ranked, 1):
            rows.append([i, x.get("name"), x.get("model"), x.get("flavor", {}).get("flavor_score"), x.get("word_count"), "是" if x.get("marker", {}).get("has_leak") else "否"])
        lines.append("## 模型Benchmark")
        lines.append(to_md_table(["排名", "预设", "模型", "风味分", "字数", "结构标签泄漏"], rows))
        lines.append("")

    if main_report is not None:
        q = main_report.get("quality_summary", {})
        a = main_report.get("analysis_summary", {})
        p = main_report.get("project", {})
        lines.append("## 主流程结果")
        lines.append(f"- 项目：`{p.get('title')}` (`{p.get('id')}`)")
        lines.append(f"- 章节数：**{q.get('chapter_count')}**")
        lines.append(f"- 平均风味分：**{q.get('avg_flavor_score')}**")
        lines.append(f"- 平均字数：**{q.get('avg_word_count')}**")
        lines.append(f"- 分析完成：**{a.get('completed')}**")
        lines.append("")

        top_rows = []
        for x in q.get("top5", []):
            top_rows.append([x.get("chapter"), x.get("title"), x.get("flavor", {}).get("flavor_score"), x.get("word_count")])
        if top_rows:
            lines.append("### Top5")
            lines.append(to_md_table(["章节", "标题", "风味分", "字数"], top_rows))
            lines.append("")

        low_rows = []
        for x in q.get("bottom5", []):
            low_rows.append([x.get("chapter"), x.get("title"), x.get("flavor", {}).get("flavor_score"), x.get("word_count")])
        if low_rows:
            lines.append("### Bottom5")
            lines.append(to_md_table(["章节", "标题", "风味分", "字数"], low_rows))
            lines.append("")

    lines.append("---")
    lines.append("脚本：`scripts/mumu_pipeline.py`")
    return "\n".join(lines)


# ========================= 主入口 =========================

def main():
    parser = argparse.ArgumentParser(description="MuMu 全流程自动化")
    parser.add_argument("--config", required=True, help="配置JSON路径")
    parser.add_argument(
        "--mode",
        default="full",
        choices=["bootstrap", "health", "benchmark", "main", "full"],
        help="执行模式",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"配置文件不存在: {cfg_path}")
        sys.exit(1)

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    base_url = cfg["base_url"]
    runner_cfg = cfg["runner_user"]

    runner = MuMuClient(base_url, runner_cfg["username"], runner_cfg["password"], name="runner")

    admin = None
    admin_cfg = cfg.get("admin_user")
    if admin_cfg and admin_cfg.get("username") and admin_cfg.get("password"):
        admin = MuMuClient(base_url, admin_cfg["username"], admin_cfg["password"], name="admin")

    backup_path = None
    health = None
    benchmark = None
    main_report = None

    if args.mode in ("bootstrap", "full"):
        backup_path = backup_database(cfg)
        if backup_path:
            print(f"[system] DB备份完成: {backup_path}", flush=True)

        # runner 登录检查
        try:
            runner.login()
            runner.log("runner 登录成功")
        except Exception:
            if not admin:
                raise
            admin.login()
            if cfg.get("bootstrap", {}).get("create_runner_if_missing", True):
                ensure_runner_user(admin, runner_cfg)
            runner.login()
            runner.log("runner 登录成功（创建后）")

        if admin and cfg.get("bootstrap", {}).get("clone_admin_settings_to_runner", False):
            admin.login()
            clone_admin_settings_to_runner(admin, runner)

    else:
        runner.login()

    style_id = ensure_style(runner, cfg["style"])
    runner.log(f"风格已就绪 style_id={style_id}")

    if args.mode in ("health", "full"):
        health = health_check_presets(runner)

    if args.mode in ("benchmark", "full"):
        benchmark = benchmark_models(runner, style_id, cfg)

    if args.mode in ("main", "full"):
        if benchmark and benchmark.get("best"):
            best_preset_name = benchmark["best"]["name"]
        else:
            best_preset_name = cfg.get("main_workflow", {}).get("best_preset_name", "随时跑路")
        main_report = run_main_workflow(runner, style_id, best_preset_name, cfg)

    # 输出结果
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg.get("output", {}).get("dir", "docs/reports"))
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = {
        "timestamp": now_str(),
        "mode": args.mode,
        "backup_path": backup_path,
        "style_id": style_id,
        "health": health,
        "benchmark": benchmark,
        "main_report": main_report,
    }
    raw_path = out_dir / f"mumu_pipeline_{ts}.json"
    raw_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")

    md = build_markdown(cfg, backup_path, health, benchmark, main_report)
    md_path = out_dir / f"mumu_pipeline_{ts}.md"
    md_path.write_text(md, encoding="utf-8")

    print(f"\n[done] JSON: {raw_path}")
    print(f"[done] MD  : {md_path}")


if __name__ == "__main__":
    main()
