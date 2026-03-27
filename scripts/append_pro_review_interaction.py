#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path


DEFAULT_LOG = "md/PRO_REVIEW_INTERACTION_LOG.md"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Append a structured Pro interaction entry to the project log.")
    ap.add_argument("--log-path", default=DEFAULT_LOG)
    ap.add_argument("--date", default=str(date.today()))
    ap.add_argument("--topic", required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--question-summary", action="append", default=[])
    ap.add_argument("--answer-summary", action="append", default=[])
    ap.add_argument("--core-answer", action="append", default=[])
    ap.add_argument("--adopted-action", action="append", default=[])
    ap.add_argument("--evidence-update", action="append", default=[])
    ap.add_argument("--raw-answer-file", default="")
    ap.add_argument("--run-root", default="")
    return ap.parse_args()


def _bullet_lines(items: list[str], default_text: str) -> list[str]:
    values = [item.strip() for item in items if item and item.strip()]
    if not values:
        values = [default_text]
    return [f"- {item}" for item in values]


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    log_path = (repo_root / args.log_path).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        raise FileNotFoundError(f"log file does not exist: {log_path}")

    section_lines: list[str] = []
    section_lines.append("")
    section_lines.append(f"## {args.date}")
    section_lines.append("")
    section_lines.append("### 主题")
    section_lines.append("")
    section_lines.append(args.topic.strip())
    section_lines.append("")
    section_lines.append("### 对应提示词")
    section_lines.append("")
    section_lines.append(f"- `{args.prompt_file.strip()}`")
    section_lines.append("")
    section_lines.append("### 提问摘要")
    section_lines.append("")
    section_lines.extend(_bullet_lines(args.question_summary, "未填写"))
    section_lines.append("")
    section_lines.append("### Pro 回答摘要")
    section_lines.append("")
    section_lines.extend(_bullet_lines(args.answer_summary, "未填写"))
    section_lines.append("")
    section_lines.append("### Pro 核心回答")
    section_lines.append("")
    section_lines.extend(_bullet_lines(args.core_answer, "未填写"))
    section_lines.append("")
    section_lines.append("### 我们最终采纳的动作")
    section_lines.append("")
    section_lines.extend(_bullet_lines(args.adopted_action, "未填写"))
    section_lines.append("")
    section_lines.append("### 后续证据修正")
    section_lines.append("")
    section_lines.extend(_bullet_lines(args.evidence_update, "暂无"))

    if args.raw_answer_file.strip() or args.run_root.strip():
        section_lines.append("")
        section_lines.append("### 相关文件")
        section_lines.append("")
        section_lines.append(f"- prompt: `{args.prompt_file.strip()}`")
        if args.raw_answer_file.strip():
            section_lines.append(f"- raw_answer: `{args.raw_answer_file.strip()}`")
        if args.run_root.strip():
            section_lines.append(f"- run_root: `{args.run_root.strip()}`")

    section_text = "\n".join(section_lines) + "\n"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(section_text)

    print(log_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
