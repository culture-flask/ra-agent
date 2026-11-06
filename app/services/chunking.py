"""层次化文本切分（新方案，与 kb_service.split_chunks 的定长窗口方案并存）。

切分规则：
1. 先按段落（空行）切；段落仍超长则按「换行 → 句号 → 叹号 → 问号 →
   分号 → 逗号 → 空格」逐级退让，直到片段长度不超过 max_size。
2. 片段贪心装箱到 300~600 字；尾部不足 300 字的孤块，从上一块回拨片段，
   仍不足则与上一块合并，避免出现十几个字的碎块。
3. 相邻块重叠 50 字（取上一块正文末尾）。
4. 每块开头拼接上下文头：【文档】标题｜【章节】章节路径。
   Markdown 标题（# ~ ######）用于推导章节路径；纯文本/PDF 无章节路径，
   只带文档标题（由调用方传入，通常为文件名）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# 逐级退让的分隔符优先级：段落 → 换行 → 句号 → 感叹号 → 问号 → 分号 → 逗号 → 空格
DEFAULT_SEPARATORS = ("\n\n", "\n", "。", "！", "？", "；", "，", " ")

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_CONTEXT_HEADER_RE = re.compile(r"^(?:【文档】|【章节】)[^\n]*\n")


@dataclass
class SectionChunk:
    """切分结果：text 已包含上下文头，heading_path 供元数据落库/检索过滤。"""

    text: str
    heading_path: str = ""


def split_chunks_hierarchical(text: str, title: str = "",
                              min_size: int = 300, max_size: int = 600,
                              overlap: int = 50,
                              separators: tuple[str, ...] = DEFAULT_SEPARATORS,
                              ) -> list[SectionChunk]:
    """按段落/标点逐级递归切分，返回带上下文头的块列表。"""
    text = (text or "").strip()
    if not text:
        return []
    doc_title = title.strip() or _first_title(text)

    pieces: list[tuple[str, str]] = []
    for para, path in _paragraphs_with_headings(text):
        # 纯标题段落不单独成块：其内容已由上下文头承载，避免 5~10 字的孤块
        if _is_heading_only(para):
            continue
        for part in _recursive_split(para, separators, max_size):
            pieces.append((part, path))

    groups = _pack(pieces, min_size, max_size)
    bodies = [("".join(p for p, _ in g).strip(), g[0][1]) for g in groups]
    bodies = [(b, p) for b, p in bodies if b]
    if not bodies:
        return []

    out: list[SectionChunk] = []
    prev = ""
    for raw, path in bodies:
        path = _drop_title_from_path(path, doc_title)
        body = prev[-overlap:] + raw if prev and overlap > 0 else raw
        header = _context_header(doc_title, path)
        out.append(SectionChunk(text=f"{header}\n{body}" if header else body,
                                heading_path=path))
        prev = raw
    return out


def strip_context_header(text: str) -> str:
    """去掉块首的上下文头（用于全文还原等需要纯正文的场景）。"""
    return _CONTEXT_HEADER_RE.sub("", text, count=1)


def _context_header(title: str, heading_path: str) -> str:
    parts = []
    if title:
        parts.append(f"【文档】{title}")
    if heading_path:
        parts.append(f"【章节】{heading_path}")
    return "｜".join(parts)


def _first_title(text: str) -> str:
    """取第一个一级标题作为文档标题（无 Markdown 标题时为空）。"""
    for line in text.splitlines():
        m = _HEADING_RE.match(line.strip())
        if m and len(m.group(1)) == 1:
            return m.group(2).strip()
    return ""


def _drop_title_from_path(path: str, title: str) -> str:
    """章节路径首段与文档标题重复时去掉，避免「标题 > 标题 > 小节」。"""
    if not path or not title:
        return path
    parts = [p.strip() for p in path.split(" > ")]
    if parts and parts[0] == title:
        parts = parts[1:]
    return " > ".join(parts)


def _is_heading_only(para: str) -> bool:
    lines = [ln.strip() for ln in para.splitlines() if ln.strip()]
    return bool(lines) and all(_HEADING_RE.match(ln) for ln in lines)


def _paragraphs_with_headings(text: str) -> list[tuple[str, str]]:
    """按空行切段落，并计算每个段落所属的章节路径。"""
    stack: dict[int, str] = {}          # 标题层级 -> 标题文本
    out: list[tuple[str, str]] = []
    # 保留段落之间的空行，拼回正文时不会粘连
    parts = re.split(r"(\n{2,})", text)
    for i in range(0, len(parts), 2):
        body = parts[i]
        if not body.strip():
            continue
        for line in body.splitlines():
            m = _HEADING_RE.match(line.strip())
            if not m:
                continue
            level = len(m.group(1))
            for lv in [lv for lv in stack if lv >= level]:
                del stack[lv]
            stack[level] = m.group(2).strip()
        path = " > ".join(stack[lv] for lv in sorted(stack))
        sep = parts[i + 1] if i + 1 < len(parts) else ""
        out.append((body + sep, path))
    return out


def _split_keep_sep(text: str, sep: str) -> list[str]:
    """按分隔符切分并保留分隔符（拼回时与原文一致）。"""
    out, start = [], 0
    while True:
        idx = text.find(sep, start)
        if idx < 0:
            out.append(text[start:])
            break
        out.append(text[start:idx + len(sep)])
        start = idx + len(sep)
    return [p for p in out if p]


def _recursive_split(text: str, separators: tuple[str, ...],
                     max_size: int) -> list[str]:
    """超长文本按分隔符优先级逐级退让，产出不超过 max_size 的片段。"""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_size:
        return [text]
    for i, sep in enumerate(separators):
        if not sep or sep not in text:
            continue
        out: list[str] = []
        for part in _split_keep_sep(text, sep):
            if len(part) <= max_size:
                out.append(part)
            else:
                out.extend(_recursive_split(part, separators[i + 1:], max_size))
        return out
    # 没有任何分隔符可用：硬切
    return [text[i:i + max_size] for i in range(0, len(text), max_size)]


def _group_len(group: list[tuple[str, str]]) -> int:
    return sum(len(p) for p, _ in group)


def _pack(pieces: list[tuple[str, str]], min_size: int,
          max_size: int) -> list[list[tuple[str, str]]]:
    """把片段装箱成 300~600 字的块；章节路径变化处强制断块。"""
    groups: list[list[tuple[str, str]]] = []
    cur: list[tuple[str, str]] = []
    for piece, path in pieces:
        if cur and (path != cur[0][1]
                    or _group_len(cur) + len(piece) > max_size):
            groups.append(cur)
            cur = []
        cur.append((piece, path))
    if cur:
        groups.append(cur)

    if len(groups) >= 2:
        _rebalance_tail(groups, min_size, max_size)
    return groups


def _rebalance_tail(groups: list[list[tuple[str, str]]],
                    min_size: int, max_size: int) -> None:
    """尾块不足 min_size 时从上一块回拨片段；仍不足则合并（不超上限时）。"""
    tail, prev = groups[-1], groups[-2]
    if tail[0][1] != prev[0][1]:
        return
    while _group_len(tail) < min_size and len(prev) > 1:
        moved = prev[-1]
        if _group_len(prev) - len(moved[0]) < min_size:
            break
        prev.pop()
        tail.insert(0, moved)
    if _group_len(tail) < min_size and _group_len(prev) + _group_len(tail) <= max_size:
        prev.extend(tail)
        groups.pop()
