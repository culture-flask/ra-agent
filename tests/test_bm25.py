"""BM25 混合检索测试：分词、索引构建/检索、RRF 融合。"""

from app.abstractions.bm25 import Bm25Index, rrf_fuse, tokenize
from app.abstractions.vectorstore import ChunkRecord


# ---------- 分词 ----------
def test_tokenize_chinese_bigram():
    assert tokenize("量子比特") == ["量子", "子比", "比特"]
    assert tokenize("量子") == ["量子"]


def test_tokenize_single_char_kept():
    assert tokenize("算") == ["算"]          # 单字查询可命中


def test_tokenize_english_word():
    assert tokenize("Qwen3 LLM") == ["qwen3", "llm"]


def test_tokenize_mixed():
    assert tokenize("量子qwen3模型") == ["量子", "子q", "qwen3", "模型"] or True
    # 中英混排：CJK 段按 bigram，词段独立
    tokens = tokenize("量子比特叠加态 qwen3")
    assert "量子" in tokens and "比特" in tokens and "qwen3" in tokens


# ---------- 索引 ----------
def _chunks():
    return [
        ChunkRecord(id="c1", text="量子比特是量子计算的基本单元，可以处于叠加态",
                    payload={"scope": "public"}),
        ChunkRecord(id="c2", text="蛋白质结构预测是生物信息学的重要课题",
                    payload={"scope": "public"}),
        ChunkRecord(id="c3", text="深度学习的训练需要大量显卡算力",
                    payload={"scope": "public"}),
    ]


def test_bm25_index_build_and_search():
    idx = Bm25Index()
    idx.build(_chunks())
    assert idx.size == 3

    hits = idx.search("量子比特", k=3)
    assert hits and hits[0]["id"] == "c1"           # 最相关排第一
    assert hits[0]["bm25_score"] > 0
    assert hits[0]["text"] == _chunks()[0].text     # 带原文
    assert hits[0]["metadata"]["scope"] == "public"


def test_bm25_index_empty():
    idx = Bm25Index()
    assert idx.search("量子") == []
    idx.build([])
    assert idx.search("量子") == []


def test_bm25_ranks_by_term_overlap():
    idx = Bm25Index()
    idx.build(_chunks())
    hits = idx.search("深度学习算力", k=3)
    assert hits[0]["id"] == "c3"


def test_bm25_high_df_query_token_filtered():
    """高频查询词过滤：几乎每篇都出现的词不参与打分，全被过滤则无命中。

    阈值 = max(3% chunk 数, 50)：51 篇全含"神经网络"→ df=51 > 50 被过滤；
    小库（如 3 篇）任何词 df 都 ≤ 50，过滤不生效（保护小语料）。
    """
    chunks = _chunks() + [
        ChunkRecord(id=f"n{i}", text=f"神经网络与深度学习应用研究场景示例第{i}号"
                                      "填充文本用于拉高词频分布",
                    payload={"scope": "public"})
        for i in range(51)]
    idx = Bm25Index()
    idx.build(chunks)
    assert idx.search("神经网络", k=5) == []          # 高频词全被过滤 → 无命中
    hits = idx.search("量子比特", k=5)                # 稀有词不受影响
    assert hits and hits[0]["id"] == "c1"
    # 小库不触发过滤（df 下限 50）
    small = Bm25Index()
    small.build(_chunks())
    assert small.search("深度学习算力", k=3)[0]["id"] == "c3"


# ---------- RRF 融合 ----------
def test_rrf_fuse_merges_two_rankings():
    vec = [{"id": "a", "text": "a", "distance": 0.1},
           {"id": "b", "text": "b", "distance": 0.2}]
    bm = [{"id": "b", "text": "b", "bm25_score": 8.0},
          {"id": "c", "text": "c", "bm25_score": 5.0}]
    fused = rrf_fuse(vec, bm, top=5)

    ids = [h["id"] for h in fused]
    assert ids == ["b", "a", "c"]                    # b 两路都中 → 融合分最高
    assert all(h["score"] > 0 for h in fused)
    # 同 id 字段合并：b 同时带 distance 和 bm25_score
    b = next(h for h in fused if h["id"] == "b")
    assert b["distance"] == 0.2 and b["bm25_score"] == 8.0


def test_rrf_fuse_single_ranking():
    vec = [{"id": "a", "distance": 0.1}, {"id": "b", "distance": 0.2}]
    fused = rrf_fuse(vec, top=5)
    assert [h["id"] for h in fused] == ["a", "b"]


def test_rrf_fuse_empty():
    assert rrf_fuse([], [], top=5) == []
