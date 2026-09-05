"""深度调研子图状态（第三支多 agent 团队「溯源社」）。

## 与争鸣社/会讲的 reducer 纪律差异（本团队的关键设计）

前两支团队大量使用 Send fan-out 并行，因此**并行节点只能返回增量**
（列表用 operator.add，且绝不写单值 channel）。本团队不同：

- 本团队**全串行、无 fan-out**。章节必须一章一章走完（后章要引用前章
  已确认的来源与结论，这是「研究参数卡」的核心价值），因此单值 channel
  （current_draft / review_round / chapter_index）由**单写者**安全覆盖。
- `chapters` / `warnings` / `token_budget_used` 用 operator.add（只增累积）。
- `source_pool` 是**单写者覆盖**而非 add：merge_sources 是纯代码节点，
  从当前草稿里抽取引用链接 → 按 URL 去重 → 整池写回。若用 add，同一来源
  会随章节反复累积重复条目，注入上下文时白白吃掉窗口。

## 双层循环

章循环（chapter_index）套着审稿轮次循环（review_round）。两个计数器都是
单写者：chapter_index 由 archive 递增，review_round 由 research 置 0、
由 revise 递增。**不要**让模型自己维护轮次——长上下文中它会记错。
"""

import operator
from typing import Annotated, TypedDict


class DeepResearchState(TypedDict, total=False):
    """深度调研图状态。"""
    # ---- 会话身份 ----
    user_id: str
    session_id: str                       # 建议 dr_ 前缀，与 bs_/sem_/主对话隔离
    topic: str
    roles: list[dict]                     # 含 config_id / model_id 可选（同前两支）
    mode: str                             # full（含审稿闭环）| quick（跳过审稿）
    token_budget: int
    budget_exhausted: bool

    # ---- Phase 1：初调（scout 写）----
    scouting_brief: str                   # 500-1000 字调研摘要
    source_pool: list[dict]               # merge_sources 单写者覆盖（去重后全量）

    # ---- Phase 2：大纲（plan 写）----
    outline: dict                         # {"title":..., "sections":[{index,title,key_points}]}

    # ---- Phase 3：章节循环（章 × 审稿轮次）----
    chapter_index: int                    # archive 单写者递增
    current_draft: str                    # research / revise 单写者覆盖
    review_round: int                     # research 置 0；revise +1
    review_verdict: str                   # PASS / REVISE（review 写）
    review_feedback: str                  # REVISE 意见全文（review 写）
    # 审稿记录（只增）：每章每轮一条 {chapter, round, verdict, content}。
    # 单值字段会被下一章覆盖——没有这条流水，详情回放里审稿人就"消失"了。
    review_log: Annotated[list[dict], operator.add]
    chapters: Annotated[list[dict], operator.add]   # 已通过章节（只增）

    # ---- Phase 4：报告框架（write_frame 写）----
    toc: str
    introduction: str
    conclusion: str
    references: list[dict]

    # ---- Phase 5：成稿（publish 写）----
    final_report: str

    # ---- 控制 ----
    warnings: Annotated[list[dict], operator.add]   # 审稿警告/降级记录
    stopped: bool
    token_budget_used: Annotated[int, operator.add]
