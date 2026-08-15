"""BM25 全文检索抽象：与向量检索互补的稀疏检索。

- 中文按 2-gram（bigram）切分（零依赖；效果不够可换 jieba 自定义 tokenizer）
- 英文/数字按词切分，小写化
- 索引与嵌入模型完全解耦：直接基于 chunk 文本构建
- rrf_fuse：Reciprocal Rank Fusion 把向量 + BM25 两路结果按排名融合
"""

import heapq
import re

from rank_bm25 import BM25Okapi

# 连续 CJK 块（含扩展区）
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]+")
# 英文/数字词块
_WORD_RE = re.compile(r"[a-zA-Z0-9]+")


def tokenize(text: str) -> list[str]:
    """混合分词：连续中文按 2-gram，英文/数字按词；统一小写。

    例：tokenize("量子比特叠加态 qwen3") -> ["量子","子比","比特","特叠","叠加",
    "加态","qwen3"]
    """
    tokens: list[str] = []
    for seg in re.findall(r"[\u4e00-\u9fff\u3400-\u4dbf]+|[a-zA-Z0-9]+",
                          text.lower()):
        if not seg:
            continue
        if _CJK_RE.fullmatch(seg):
            if len(seg) == 1:
                tokens.append(seg)                 # 单字也保留（短查询可命中）
            else:
                tokens.extend(seg[i:i + 2] for i in range(len(seg) - 1))
        else:
            tokens.append(seg)
    return tokens


class Bm25Index:
    """单库 BM25 索引：chunk 级稀疏检索（BM25Okapi，k1=1.5 / b=0.75）。"""

    def __init__(self):
        self._bm25: BM25Okapi | None = None
        self._items: list[dict] = []               # 与 corpus 对齐：id/text/metadata
        self._token_docs: list[set] = []           # 每篇文档的 token 集合（判断是否命中）

    @property
    def size(self) -> int:
        return len(self._items)

    def build(self, chunks: list) -> None:
        """用 ChunkRecord（id/text/payload）构建索引。"""
        self._items = [{"id": c.id, "text": c.text, "metadata": c.payload}
                       for c in chunks]
        tokenized = [tokenize(c.text) for c in chunks]
        self._token_docs = [set(t) for t in tokenized]
        self._bm25 = BM25Okapi(tokenized) if chunks else None

    def search(self, query: str, k: int = 5) -> list[dict]:
        """按 BM25 分数降序返回 top-k 命中：[{id, text, metadata, bm25_score}]。

        命中判定按「查询词是否出现在文档里」，不按分数正负——单文档库
        （词出现在唯一文档）时 rank-bm25 的 IDF 地板也为负，分数可能为负，
        但该文档确实是唯一命中。
        """
        if self._bm25 is None or not self._items:
            return []
        q_tokens = set(tokenize(query))
        if not q_tokens:
            return []
        scores = self._bm25.get_scores(list(q_tokens))
        ranked = heapq.nlargest(min(k, len(scores)), range(len(scores)),
                                key=lambda i: scores[i])
        out = []
        for i in ranked:
            if not (q_tokens & self._token_docs[i]):   # 无查询词命中 → 跳过
                continue
            out.append({**self._items[i], "bm25_score": round(float(scores[i]), 6)})
        return out


# RRF 融合常数（标准值 60）：rank 越小贡献越大
RRF_K = 60


def rrf_fuse(*hit_lists: list[dict], top: int = 5) -> list[dict]:
    """Reciprocal Rank Fusion：多路结果按排名融合，同 id 的字段合并。

    - 每路结果按传入顺序视为排名（rank 从 1 起）
    - 同 id 出现在多路时：保留第一路的全部字段，并用后一路的非空字段补全
      （向量命中带 distance、BM25 命中带 bm25_score → 融合后两路分数并存）
    - 输出按融合分降序，附 score（RRF 融合分）
    """
    scores: dict[str, float] = {}
    merged: dict[str, dict] = {}
    for hits in hit_lists:
        for rank, h in enumerate(hits, start=1):
            hid = h["id"]
            scores[hid] = scores.get(hid, 0.0) + 1.0 / (RRF_K + rank)
            if hid in merged:
                # 补全缺失字段（vector 命中没有 bm25_score，反之亦然）
                merged[hid] = {**merged[hid],
                               **{kk: vv for kk, vv in h.items()
                                  if vv is not None and kk not in merged[hid]}}
            else:
                merged[hid] = dict(h)
    ordered = sorted(merged.values(), key=lambda h: scores[h["id"]], reverse=True)
    for h in ordered:
        h["score"] = round(scores[h["id"]], 6)
    return ordered[:top]
