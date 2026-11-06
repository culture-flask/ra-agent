"""KBService.search_multi：多查询"腿内先跨查询、腿间再融合"的行为契约。

融合层次：vec(q1)+vec(q2)→RRF；bm(q1)+bm(q2)→RRF；两腿路再 RRF。
既验证调用结构（每查询两腿各一次、深度/参数传递），也验证融合语义
（双腿命中胜出、分数降序），并确认单查询委托 search（原路径不变）。
"""
from app.services.kb_service import KBService
from app.settings import Settings


class _KB:
    kb_id = "k1"
    name = "假库"
    scope = "public"
    embedding_dim = 8
    embedded_model = None


class _Leg:
    """向量/BM25 腿的通用 stub：记录调用并按查询返回命中。"""

    def __init__(self, table: dict):
        self.table = table
        self.calls: list[tuple] = []

    def search(self, q, k, **kw):
        self.calls.append((q, k, kw.get("scope", ""), kw.get("user_id", "")))
        return list(self.table.get(q, []))


def _mk(monkeypatch, vec_table, bm_table):
    svc = KBService(Settings.load())
    monkeypatch.setattr(svc, "get_kb", lambda kb_id: _KB)
    monkeypatch.setattr(svc, "embedding_mismatch", lambda kb: None)
    vec, bm = _Leg(vec_table), _Leg(bm_table)
    monkeypatch.setattr(svc, "_vector_store", lambda kb: vec)
    monkeypatch.setattr(svc, "_bm25_index", lambda kb: bm)
    return svc, vec, bm


def _hit(hid):
    return {"id": hid, "text": f"文本{hid}", "metadata": {}}


def test_search_multi_two_queries(monkeypatch):
    """两查询：每腿各调两次（深度 15、scope/user 透传），双腿命中的 id 胜出。"""
    svc, vec, bm = _mk(monkeypatch,
                       {"q1": [_hit("a"), _hit("b")], "q2": [_hit("c")]},
                       {"q1": [_hit("b")], "q2": []})
    out = svc.search_multi("k1", ["q1", "q2"], k=3, user_id="u1",
                           mode="hybrid")
    assert len(vec.calls) == 2 and len(bm.calls) == 2
    assert all(c == ("q1", 15, "public", "u1") or c == ("q2", 15, "public", "u1")
               for c in vec.calls)
    # b 同时出现在向量腿与 BM25 腿 → 两处贡献累加，排在只被单腿命中的 a/c 之前
    ids = [h["id"] for h in out]
    assert ids[0] == "b"
    assert {"a", "c"} <= set(ids)
    scores = [h["score"] for h in out]
    assert scores == sorted(scores, reverse=True)
    assert all(h.get("kb_name") == "假库" for h in out)


def test_search_multi_single_query_delegates(monkeypatch):
    """单查询（去重后仅剩一条）→ 委托 search：每腿各一次，无重复。"""
    svc, vec, bm = _mk(monkeypatch, {"q1": [_hit("a")]}, {"q1": []})
    out = svc.search_multi("k1", ["q1", "  ", ""], k=3, user_id="u1",
                           mode="hybrid")
    assert len(vec.calls) == 1 and len(bm.calls) == 1
    assert [h["id"] for h in out] == ["a"]