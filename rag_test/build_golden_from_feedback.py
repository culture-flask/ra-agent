"""从用户反馈生成检索金标准候选（P3-19 反馈闭环的最后一环）。

用法：
    python3 rag_test/build_golden_from_feedback.py [--min-rating 1] [--out golden_from_feedback.json]

逻辑：取 feedbacks 表中 rating >= min-rating 的记录，把引用面板冗余的
hits.source 聚合为文档级"期望命中列表"，按 question 去重后输出
与 eval3_offline.py 的 CASES 同构的条目：

    {"query": ..., "expected": ["HDND.pdf", ...], "session_id", "created_at"}

半自动约定：输出是**候选**而非直接金标准——人工过一遍剔除
"点赞但检索其实没帮上忙"的噪声（比如纯闲聊），再合并进下一轮
eval 脚本的 CASES。这一步刻意不自动化：评测集的质量红线必须有人守。
"""

import argparse
import json
from pathlib import Path

from sqlalchemy import select

from app.core.db import SessionLocal
from app.models import Feedback


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-rating", type=int, default=1,
                        help="纳入候选的最低 rating（默认 1=只看点赞）")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).parent / "golden_from_feedback.json")
    args = parser.parse_args()

    with SessionLocal() as db:
        rows = db.scalars(select(Feedback).where(
            Feedback.rating >= args.min_rating)
            .order_by(Feedback.created_at.desc())).all()

    seen: set[str] = set()
    cases: list[dict] = []
    skipped_no_source = 0
    for r in rows:
        question = r.question.strip()
        if not question or question in seen:
            continue
        expected = sorted({h.get("source") for h in (r.hits or [])
                           if h.get("source")})
        if not expected:
            skipped_no_source += 1          # 没有知识库来源：无法作为检索金标准
            continue
        seen.add(question)
        cases.append({
            "query": question,
            "expected": expected,
            "session_id": r.session_id,
            "created_at": r.created_at.isoformat(),
        })

    args.out.write_text(json.dumps(cases, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"反馈 {len(rows)} 条 -> 候选 {len(cases)} 条 "
          f"(无来源跳过 {skipped_no_source}) -> {args.out}")
    for c in cases[:10]:
        print(f"  - {c['query'][:50]}  =>  {', '.join(c['expected'])}")


if __name__ == "__main__":
    main()
