"""格致会讲子图状态。

reducer 纪律与争鸣社实测一致：
- 并行节点（read/propose/improve/score 四处 fan-out）只返回增量，
  列表用 operator.add、token 累计用 operator.add；
- evidence_pool 用 merge_evidence（add + 按 kb/source 去重）：四路 read
  并行直检同一批库，纯 add 会把同一来源存 4 份（同争鸣社）；
- ⚠️ 并行节点绝不写单值 channel（stopped 等）——停止语义由
  is_stopped(sid) 注册表承载，rapporteur/路由自查。
- agenda_index 由 present 单写者递增（覆盖语义）。
"""

import operator
from typing import Annotated, TypedDict


def merge_evidence(left: list | None, right: list | None) -> list:
    """evidence_pool 的合并 reducer：operator.add 语义 + 按 (kb, source) 去重。

    四路 read 并行直检同一批可见库（同查询、同 k），同一来源会被四个学者
    各存一条——纯 add 合并后池里 4 份重复。保留首次出现（found_by 即首个发现者），
    并行写入顺序不定、谁先到谁署名，对证据内容无影响。
    """
    out: list = list(left or [])
    seen = {(e.get("kb"), e.get("source")) for e in out}
    for e in (right or []):
        key = (e.get("kb"), e.get("source"))
        if key not in seen:
            seen.add(key)
            out.append(e)
    return out


class SeminarState(TypedDict, total=False):
    """会讲图状态。"""
    # 会话身份
    user_id: str
    session_id: str                       # 建议 sem_ 前缀，与 bs_/主对话空间隔离
    topic: str
    roles: list[dict]                     # 同争鸣社：含 config_id/model_id 可选
    token_budget: int                     # 本场预算（前端可调，缺省全局配置）
    budget_exhausted: bool                # 后续阶段被熔断截断（rapporteur 写）

    # Phase 0：curate 写
    field_positioning: str
    adjacent_field: str                   # 访问学者的源领域（含理由）
    reading_assignments: dict             # role_id -> 研读任务单
    ideation_directives: dict             # role_id -> 构想策略

    # Phase 1：read fan-out 并行写
    reading_notes: Annotated[list[dict], operator.add]   # 研读笔记
    evidence_pool: Annotated[list[dict], merge_evidence]  # 共享证据（合并时按 kb/source 去重）

    # Phase 2：议程循环
    agenda_index: int                     # 当前汇报到第几位学者（present 递增）
    qa_transcript: Annotated[list[dict], operator.add]   # 汇报+问答记录
    insight_board: Annotated[list[dict], operator.add]   # 洞见池（只增不减）
    open_questions: Annotated[list[dict], operator.add]

    # Phase 3/4：构想与评审
    idea_cards: Annotated[list[dict], operator.add]      # 原始+改进卡（含 builds_on）
    card_registry: list[dict]              # merge/panel 定稿（单写者覆盖）
    review_scores: Annotated[list[dict], operator.add]    # 每学者对每卡一评
    idea_ranking: list[dict]               # aggregate 写（纯代码）

    # Phase 5
    final_proposal: str

    # 控制
    stopped: bool
    token_budget_used: Annotated[int, operator.add]
