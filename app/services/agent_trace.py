"""多 Agent 场次的执行轨迹留档：思考过程 + 工具调用。

这两类内容此前只经 SSE 实时事件透传——刷新页面后回放就没有了（checkpoint
只存成稿/笔记/洞见等结果数据，不存事件流）。它们体量大（思考实测可达数万
字符）、回放时整体读一次、且与图状态无关：塞进 checkpoint 会让每次节点转移
都重写整个状态，因此落成**场次级 JSON 文件**——流结束时写一次、回放时读出。

数据形状（entries 按发言先后顺序）：
    {"entries": [{"agent": "historian", "phase": "read", "reasoning": "…",
                  "tools": [{"name": "arxiv_search", "args": {...},
                             "ok": true, "error": ""}]}, …]}
"""

import json
import re
from pathlib import Path

from app.core.logging import get_logger
from app.settings import Settings

logger = get_logger("agent_trace")

REASONING_MAX_CHARS = 12_000     # 单次发言思考留档上限（超出截断并标注）
_PREFIX = re.compile(r"^(?:bs_|sem_|dr_)")
_SAFE_ID = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")     # 会话 id 白名单：防路径穿越


def _path(session_id: str) -> Path | None:
    if not _SAFE_ID.match(session_id or ""):
        return None
    return Path(Settings.load().data_dir) / "agent_traces" / f"{session_id}.json"


def new_recorder() -> dict:
    """新建录制器：随 SSE 事件累积每位 agent 的发言轨迹。"""
    return {"entries": [], "cur": None}


def record(rec: dict, ev: dict) -> None:
    """并入一条 SSE 事件；只认 agent_start / reasoning / tool_start / tool_end。"""
    kind = _PREFIX.sub("", str(ev.get("type") or ""))
    cur = rec.get("cur")
    if kind == "agent_start":
        cur = {"agent": str(ev.get("agent") or ""),
               "phase": str(ev.get("phase") or ""),
               "reasoning": "", "tools": []}
        rec["entries"].append(cur)
        rec["cur"] = cur
    elif kind == "reasoning" and cur is not None:
        if len(cur["reasoning"]) < REASONING_MAX_CHARS:
            cur["reasoning"] += str(ev.get("content") or "")
    elif kind == "tool_start" and cur is not None:
        cur["tools"].append({"name": str(ev.get("name") or ""),
                             "args": ev.get("args") or {}, "ok": True, "error": ""})
    elif kind == "tool_end" and cur is not None:
        err = str(ev.get("error") or "")
        if err:                                   # 同名工具取最近一次未收尾的
            name = str(ev.get("name") or "")
            for t in reversed(cur["tools"]):
                if t["name"] == name and t["ok"]:
                    t["ok"], t["error"] = False, err[:500]
                    break
    elif kind in ("agent_end", "agent_fail"):
        rec["cur"] = None


def _finalize(rec: dict) -> dict:
    """收尾：超长思考截断并标注；无思考也无工具的发言不留档。"""
    out = []
    for e in rec.get("entries") or []:
        text = e.get("reasoning") or ""
        if len(text) > REASONING_MAX_CHARS:
            text = text[:REASONING_MAX_CHARS] + f"…（思考过长已截断，共 {len(text)} 字）"
        if not text and not e.get("tools"):
            continue
        out.append({"agent": e.get("agent") or "", "phase": e.get("phase") or "",
                    "reasoning": text, "tools": e.get("tools") or []})
    return {"entries": out}


def save(session_id: str, rec: dict) -> None:
    """流结束/客户端断开时落盘（best-effort：失败只记日志，不影响场次结果）。

    同步写：单个小文件，且在 SSE 生成器的 finally 里调用——那里 await 有
    "生成器被提前关闭"的时序风险，直接写更稳。
    """
    path = _path(session_id)
    if path is None:
        return
    try:
        data = _finalize(rec)
        if not data["entries"]:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except Exception as e:                        # noqa: BLE001
        logger.warning("agent trace save failed %s: %s", session_id, e)


def load(session_id: str) -> list[dict]:
    """回放：读该场次的轨迹（无档案返回空列表）。"""
    path = _path(session_id)
    if path is None or not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("entries") or []
    except Exception as e:                        # noqa: BLE001
        logger.warning("agent trace load failed %s: %s", session_id, e)
        return []


def delete(session_id: str) -> None:
    """重跑同场次时清掉旧档案（best-effort）。"""
    path = _path(session_id)
    try:
        if path is not None and path.exists():
            path.unlink()
    except Exception as e:                        # noqa: BLE001
        logger.warning("agent trace delete failed %s: %s", session_id, e)