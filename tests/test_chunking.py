"""层级递归切分测试：长度约束、分隔符退让、上下文头与重叠。"""

from app.services.chunking import (split_chunks_hierarchical,
                                   strip_context_header)


def _bodies(chunks) -> list[str]:
    return [strip_context_header(c.text) for c in chunks]


def test_short_text_single_chunk_with_title_header():
    chunks = split_chunks_hierarchical("量子比特可以处于叠加态。", title="量子计算导论")
    assert len(chunks) == 1
    assert chunks[0].text.startswith("【文档】量子计算导论\n")
    assert "量子比特" in chunks[0].text


def test_markdown_heading_becomes_section_path():
    body = "量子比特可以处于叠加态。" * 40          # 480 字，单块
    text = f"# 量子计算导论\n\n## 方法\n\n{body}"
    chunks = split_chunks_hierarchical(text, title="量子计算导论")
    assert len(chunks) == 1
    assert chunks[0].heading_path == "方法"          # 与文档标题重复的 H1 被去掉
    assert chunks[0].text.startswith("【文档】量子计算导论｜【章节】方法\n")


def test_period_fallback_keeps_chunks_in_range():
    """无空行/换行时按句号切，块长落在 300~600，且不在句中截断。"""
    text = ("量" * 98 + "。") * 20                   # 每句 99 字，共 1980 字
    chunks = split_chunks_hierarchical(text, title="长文", overlap=0)
    bodies = _bodies(chunks)
    assert len(bodies) >= 3
    assert all(300 <= len(b) <= 600 for b in bodies)
    assert all(b.endswith("。") for b in bodies)


def test_semicolon_fallback_when_no_period():
    """没有句号时退让到分号。"""
    text = ("甲" * 98 + "；") * 10
    chunks = split_chunks_hierarchical(text, title="长文", overlap=0)
    bodies = _bodies(chunks)
    assert all(300 <= len(b) <= 600 for b in bodies)
    assert all(b.endswith("；") for b in bodies)


def test_paragraph_boundary_preferred():
    """段落可容纳时优先按段落组合，不切进段落内部。"""
    text = "\n\n".join(["段" * 199 + "。"] * 5)      # 5 段 × 200 字
    chunks = split_chunks_hierarchical(text, title="长文", overlap=0)
    bodies = _bodies(chunks)
    assert all(300 <= len(b) <= 600 for b in bodies)
    assert all(len(b) % 200 == 0 for b in bodies)    # 块边界与段落边界对齐


def test_adjacent_chunks_overlap_50_chars():
    text = "\n\n".join(["段" * 399 + "。"] * 3)      # 3 段 × 400 字 → 3 块
    chunks = split_chunks_hierarchical(text, title="长文", overlap=50)
    bodies = _bodies(chunks)
    assert len(bodies) == 3
    assert len(bodies[0]) == 400                     # 首块不加重叠
    assert bodies[1].startswith(bodies[0][-50:])     # 后续块前缀为上一块末尾 50 字
    assert bodies[2].startswith(bodies[1][-50:])


def test_tail_orphan_is_absorbed():
    """尾部十几字的孤块并入上一块（或回拨片段），不单独成块。"""
    text = ("短" * 98 + "。") * 7                     # 693 字
    chunks = split_chunks_hierarchical(text, title="长文", overlap=0)
    bodies = _bodies(chunks)
    assert all(len(b) >= 100 for b in bodies)
    assert len(bodies) == 2                          # 693 → 396/297，尾部孤块已通过回拨消除
