#!/usr/bin/env python3
"""
根据目标字数与单章目标字数，估算需要多少章；并给出 one-to-many 大纲展开建议。

用法：
  ./wordcount_plan.py --target 2000000 --per-chapter 5000 --outlines 120
"""

from __future__ import annotations

import argparse
import math


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=2_000_000, help="目标字数（默认 2000000）")
    ap.add_argument("--per-chapter", type=int, default=5000, help="单章目标字数（默认 5000）")
    ap.add_argument("--outlines", type=int, default=0, help="大纲节点数量（可选，用于估算每大纲展开几章）")
    args = ap.parse_args()

    if args.target <= 0 or args.per_chapter <= 0:
        raise SystemExit("target/per-chapter 必须为正整数")

    chapters = math.ceil(args.target / args.per_chapter)
    print(f"target_words={args.target}")
    print(f"per_chapter_words={args.per_chapter}")
    print(f"estimated_chapters={chapters}")

    if args.outlines and args.outlines > 0:
        per_outline = math.ceil(chapters / args.outlines)
        print(f"outlines={args.outlines}")
        print(f"suggest_chapters_per_outline={per_outline}  # 用于 outline_batch_expand_stream.chapters_per_outline")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

