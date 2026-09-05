"""头脑风暴子图状态。

reducer 选择（LangGraph 并发写合并的关键）：
- positions / transcript / token_budget_used 用 operator.add：
  ① Phase 1 四个 research 节点并行返回立场书——列表追加、整数累加，
    无需任何锁，LangGraph 按 reducer 合并；
  ② 之后串行阶段每次返回增量，同样走 add 语义。
- evidence_pool 用 merge_evidence（add + 按 kb/source 去重）：四路并行
  直检同一批库（同查询同 k），add 会把同一来源存 4 份、挤占摘要名额。
- 其余字段都是单写者（prepare 写 directives、moderator 写调度），默认覆盖。
"""

import operator
from typing import Annotated, TypedDict


def merge_evidence(left: list | None, right: list | None) -> list:
    """evidence_pool 的合并 reducer：operator.add 语义 + 按 (kb, source) 去重。

    四路 research 并行直检同一批可见库（同查询、同 k），同一来源会被四个
    角色各存一条——纯 add 合并后池里 4 份重复。
    本 reducer 在合并时保留首次出现（found_by即首个发现者），
    池语义变为"去重后的共享证据"。并行写入顺序不定，
    谁先到谁署名，对证据内容无影响。
    """
    out: list = list(left or [])
    seen = {(e.get("kb"), e.get("source")) for e in out}
    for e in (right or []):
        key = (e.get("kb"), e.get("source"))
        if key not in seen:
            seen.add(key)
            out.append(e)
    return out


class BrainstormState(TypedDict, total=False):
    """头脑风暴图状态。"""
    # 会话身份
    user_id: str
    session_id: str                       # 建议前端用 bs_ 前缀，与主对话 thread 空间隔离
    topic: str                            # 用户议题（含附件展开文本）
    roles: list[dict]                     # [{"id","name","temperature","config_id"?,"model_id"?}, ...]
    max_rounds: int                       # 辩论轮数上限（×角色数 = 发言次数上限）
    token_budget: int                     # 本场 token 预算（硬熔断；前端按场可调，缺省用全局配置）
    budget_exhausted: bool                # 辩论因预算熔断被截断（synthesis 写，stats 标注用）

    # Phase 0：prepare 写
    topic_restatement: str
    research_directives: dict             # role_id -> 差异化调研指令

    # Phase 1：research fan-out 并行写（operator.add 合并）
    positions: Annotated[list[dict], operator.add]       # 立场书
    evidence_pool: Annotated[list[dict], merge_evidence]  # 共享证据（合并时按 kb/source 去重）

    # Phase 2：moderator / agent 交替写
    transcript: Annotated[list[dict], operator.add]      # {"turn","agent_id","agent_name","content"}
    turn_count: int                       # 已完成的发言次数
    moderator_notes: dict                 # 最新一轮主持人判定（consensus/divergence/level/focus）
    next_speaker: str                     # moderator 写、路由函数读（""/finish = 结束）

    # Phase 4：synthesis 写
    final_proposal: str

    # 控制
    stopped: bool
    token_budget_used: Annotated[int, operator.add]       # 全程累计 token（预算熔断依据）
