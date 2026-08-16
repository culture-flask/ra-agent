"""第四轮：混合检索超参数优化实验台（2026-08-16）。

思路：
- 向量腿无法离线复现（需要嵌入服务），但它的结果与超参数无关——
  经服务 API mode=vector 取 top-50 缓存，实验时按腿深 D 截断。
- BM25 腿与 RRF 融合完全离线复现（rank_bm25 的 k1/b 是打分期属性，
  索引建一次即可切换参数做网格实验）。
- 用第三轮已验证的 26 个查询（18 长 + 8 短）与同一套金标准。
- 先跑「基线复现校验」：离线融合结果必须与第三轮线上 hybrid 结果一致，
  证明实验台可信，然后才做参数实验。

运行：.venv/bin/python rag_test/eval4_hparam.py
输出：rag_test/eval4_hparam_results.json
"""
import glob
import heapq
import importlib.util
import json
import re
import sys
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

REPO = Path("/home/pyp/interview_preparation/ra-agent")
sys.path.insert(0, str(REPO))                     # 复用项目分词器
from app.abstractions.bm25 import tokenize as tok_bigram   # noqa: E402
from rank_bm25 import BM25Okapi                            # noqa: E402

BASE = "http://127.0.0.1:8000"
KB_ID = "ad4a296fda7c"
USER_ID = "241550d8be134b62b47895dbc59aaa88"
VEC_FETCH = 50                     # 向量腿缓存深度（实验截断用）
TOPK = 10                          # 融合输出深度（与第三轮一致）
CACHE_VEC = REPO / "rag_test/eval4_vector_legs.json"
OUT = REPO / "rag_test/eval4_hparam_results.json"

# ---------------- 查询与金标准（复用第三轮，单一事实来源） ----------------

def _load(path, attr="CASES"):
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, attr)

LONG_CASES = [  # (category, query, gold)
    (c[0], c[1], c[2])
    for c in _load(REPO / "rag_test/eval3_offline.py")]
SHORT_CASES = [(c[0], c[1], c[2]) for c in _load(REPO / "rag_test/eval3_short.py")]
ALL_CASES = ([{"split": "long", "cat": c[0], "query": c[1], "gold": c[2]}
             for c in LONG_CASES]
            + [{"split": "short", "cat": "S-短查询", "query": c[0], "gold": c[2]}
               for c in SHORT_CASES])

# ---------------- 向量腿缓存 ----------------

def fetch_vector_legs():
    if CACHE_VEC.exists():
        return json.loads(CACHE_VEC.read_text(encoding="utf-8"))
    cache = {}
    for case in ALL_CASES:
        q = case["query"]
        if q in cache:
            continue
        qs = urllib.parse.urlencode(
            {"query": q, "k": VEC_FETCH, "mode": "vector", "user_id": USER_ID})
        with urllib.request.urlopen(
                f"{BASE}/api/v1/kbs/{KB_ID}/search?{qs}", timeout=120) as r:
            hits = json.load(r)
        cache[q] = [{"id": h["id"],
                     "source": (h.get("metadata") or {}).get("source")}
                    for h in hits]
    CACHE_VEC.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    return cache

# ---------------- chunk 读取与 BM25 变体 ----------------

def load_chunks():
    chunks = []
    for f in sorted(glob.glob(str(REPO / f"data/chunks/{KB_ID}/*.txt"))):
        stem = Path(f).stem
        meta = json.loads(
            Path(f[:-4] + ".meta.json").read_text(encoding="utf-8"))
        chunks.append({"id": stem,
                       "text": Path(f).read_text(encoding="utf-8"),
                       "source": meta.get("source")})
    return chunks

_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]+")

def tok_uni_bi(text):
    """unigram+bigram 混合分词：中文段同时产出单字与双字，英文按词。"""
    toks = []
    for seg in re.findall(r"[\u4e00-\u9fff\u3400-\u4dbf]+|[a-zA-Z0-9]+",
                          text.lower()):
        if _CJK_RE.fullmatch(seg):
            toks.extend(seg)                                  # 全部单字
            if len(seg) > 1:
                toks.extend(seg[i:i + 2] for i in range(len(seg) - 1))
        else:
            toks.append(seg)
    return toks

TOKENIZERS = {"bigram": tok_bigram, "uni_bi": tok_uni_bi}

class Bm25Var:
    """项目 Bm25Index 的参数化复刻：k1/b 打分期可变，查询词支持 DF 过滤。"""

    def __init__(self, chunks, tokenizer):
        self.tokenizer = tokenizer
        self.items = [{"id": c["id"], "source": c["source"]} for c in chunks]
        corpus = [tokenizer(c["text"]) for c in chunks]
        self.token_docs = [set(t) for t in corpus]
        self.doc_freq = defaultdict(int)
        for s in self.token_docs:
            for t in s:
                self.doc_freq[t] += 1
        self.bm = BM25Okapi(corpus)

    def search(self, query, k, k1=1.5, b=0.75, df_drop=1.0):
        self.bm.k1, self.bm.b = k1, b
        q_tokens = set(self.tokenizer(query))
        if df_drop < 1.0:   # 丢弃出现在超过 df_drop 比例 chunk 里的查询词
            n = len(self.items)
            q_tokens = {t for t in q_tokens
                        if self.doc_freq[t] <= df_drop * n}
        if not q_tokens:
            return []
        scores = self.bm.get_scores(list(q_tokens))
        ranked = heapq.nlargest(min(k, len(scores)), range(len(scores)),
                                key=lambda i: scores[i])
        return [self.items[i]["id"] for i in ranked
                if q_tokens & self.token_docs[i]]

# ---------------- 融合与指标 ----------------

def fuse(vec_ids, bm_ids, K=60, w_vec=1.0, w_bm=1.0, top=TOPK):
    """加权 RRF：score += w / (K + rank)。与项目 rrf_fuse 同插入序（向量路先）。"""
    scores, order = {}, []
    for rank, cid in enumerate(vec_ids, 1):
        if cid not in scores:
            order.append(cid)
        scores[cid] = scores.get(cid, 0.0) + w_vec / (K + rank)
    for rank, cid in enumerate(bm_ids, 1):
        if cid not in scores:
            order.append(cid)
        scores[cid] = scores.get(cid, 0.0) + w_bm / (K + rank)
    ranked = sorted(order, key=lambda cid: -scores[cid])
    return ranked[:top]

def metrics(doc_seq, gold):
    gold_set = set(gold)
    docs, seen = [], set()
    for d in doc_seq:
        if d not in seen:
            seen.add(d)
            docs.append(d)
    h5 = sum(d in gold_set for d in docs[:5])
    h10 = sum(d in gold_set for d in docs[:10])
    mrr = next((1.0 / r for r, d in enumerate(docs, 1) if d in gold_set), 0.0)
    return {"p@5": h5 / 5, "r@5": h5 / len(gold_set),
            "p@10": h10 / 10, "r@10": h10 / len(gold_set), "mrr": mrr}

# ---------------- 实验台 ----------------

class Bench:
    def __init__(self):
        self.vec_legs = fetch_vector_legs()
        chunks = load_chunks()
        self.id2src = {c["id"]: c["source"] for c in chunks}
        self.bm25 = {name: Bm25Var(chunks, tok)
                     for name, tok in TOKENIZERS.items()}

    def run(self, case, cfg):
        """cfg: tokenizer/k1/b/K/w_vec/w_bm/depth/df_drop → 文档序列。"""
        vec_ids = [h["id"] for h in self.vec_legs[case["query"]]]
        bm_ids = self.bm25[cfg["tokenizer"]].search(
            case["query"], cfg["depth"], k1=cfg["k1"], b=cfg["b"],
            df_drop=cfg.get("df_drop", 1.0))
        fused = fuse(vec_ids[:cfg["depth"]], bm_ids,
                     K=cfg["K"], w_vec=cfg["w_vec"], w_bm=cfg["w_bm"])
        return [self.id2src[cid] for cid in fused]

    def evaluate(self, cfg, cases=None):
        cases = cases or ALL_CASES
        rows = [metrics(self.run(c, cfg), c["gold"]) for c in cases]
        agg = {k: sum(r[k] for r in rows) / len(rows)
               for k in ("p@5", "r@5", "p@10", "r@10", "mrr")}
        return agg

BASELINE = {"tokenizer": "bigram", "k1": 1.5, "b": 0.75, "K": 60,
            "w_vec": 1.0, "w_bm": 1.0, "depth": 10, "df_drop": 1.0}


def main():
    bench = Bench()
    print(f"查询数: {len(ALL_CASES)}（长 {len(LONG_CASES)} + 短 {len(SHORT_CASES)}）")

    # ---- 第 0 步：基线复现校验 ----
    print("\n[0] 基线复现校验（离线融合 vs 第三轮线上 hybrid）")
    r3 = {}
    for f, key in [("eval3_offline_results.json", "long"),
                   ("eval3_short_results.json", "short")]:
        d = json.loads((REPO / "rag_test" / f).read_text(encoding="utf-8"))
        for c in d["cases"]:
            r3[c["query"]] = c["hybrid"]
    mismatches = 0
    for case in ALL_CASES:
        docs = bench.run(case, BASELINE)
        m = metrics(docs, case["gold"])
        online = r3[case["query"]]
        if (abs(round(m["mrr"], 3) - online["mrr"]) > 1e-9
                or abs(round(m["r@10"], 3) - online["r@10"]) > 1e-9):   # 线上值存 3 位小数
            mismatches += 1
            print(f"  不一致: {case['query'][:30]} "
                  f"离线 mrr={m['mrr']:.2f} r@10={m['r@10']:.2f} vs "
                  f"线上 mrr={online['mrr']:.2f} r@10={online['r@10']:.2f}")
    base_agg = bench.evaluate(BASELINE)
    print(f"  一致性: {len(ALL_CASES) - mismatches}/{len(ALL_CASES)} 用例完全一致"
          if mismatches == 0 else f"  ⚠ {mismatches} 个用例不一致（见上）")
    print(f"  基线总体: MRR={base_agg['mrr']:.3f} R@10={base_agg['r@10']:.3f} "
          f"P@5={base_agg['p@5']:.3f} R@5={base_agg['r@5']:.3f}")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"baseline": {**BASELINE, "agg": base_agg},
                   "validated": mismatches == 0}, f, ensure_ascii=False, indent=2)
    print(f"\n校验结果已写入 {OUT.name}（敏感性/组合实验由 eval4_run.py 继续）")


if __name__ == "__main__":
    main()
