"""第五轮：分词正则改动（英文词块保留 - 和 _）后的超参数重测（2026-08-20）。

背景：
- 用户把 _WORD_RE 改为保留 -/_，但 tokenize 用的是内联正则，原改动不生效；
  已修复为 _SEG_RE 从 _WORD_RE 派生（本轮确认生效后重测）。
- 期间知识库被重建：新 kb_id=408cbfa4b504（同 81 篇文档，HDND.pdf 更名为
  HDND JCST.pdf），向量腿缓存需对新库重取。

实验设计（全部基于新库语料，内部自洽）：
1. 向量腿：新库 mode=vector 取 top-50 缓存（与分词/超参数无关）。
2. 分词器：tok_old（旧内联正则复刻） vs tok_new（当前 app.tokenize，保留 -/_）。
3. 四配置对比：{旧,新}分词 × {旧,新}超参 —— 回答"分词改动是否影响最优超参数"。
4. 新分词下重跑敏感性 + 576 配置网格 + 5 折稳健性，验证第四轮最优
   （b=0.45 / depth=15 / df=0.03）是否仍最优。

运行：.venv/bin/python rag_test/eval5_reeval.py
输出：rag_test/eval5_vector_legs.json + rag_test/eval5_reeval_results.json
"""
import glob
import heapq
import importlib.util
import itertools
import json
import re
import statistics as st
import sys
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

REPO = Path("/home/pyp/interview_preparation/ra-agent")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "rag_test"))
from app.abstractions.bm25 import tokenize as tok_new          # noqa: E402
from rank_bm25 import BM25Okapi                               # noqa: E402

BASE = "http://127.0.0.1:8000"
KB_ID = "408cbfa4b504"                # 重建后的密码学-神经区分器库
USER_ID = "241550d8be134b62b47895dbc59aaa88"
VEC_FETCH, TOPK = 50, 10
CACHE_VEC = REPO / "rag_test/eval5_vector_legs.json"
OUT = REPO / "rag_test/eval5_reeval_results.json"

# 旧库 → 新库文件名映射（重传时更名的文件）
RENAME = {"HDND.pdf": "HDND JCST.pdf"}

# 旧分词器复刻：原内联正则（英文词块不含 - / _）
_CJK_RE_OLD = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]+")

def tok_old(text):
    toks = []
    for seg in re.findall(r"[\u4e00-\u9fff\u3400-\u4dbf]+|[a-zA-Z0-9]+",
                          text.lower()):
        if _CJK_RE_OLD.fullmatch(seg):
            if len(seg) == 1:
                toks.append(seg)
            else:
                toks.extend(seg[i:i + 2] for i in range(len(seg) - 1))
        else:
            toks.append(seg)
    return toks


def _cases(path):
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CASES

LONG = [{"split": "long", "query": q, "gold": [RENAME.get(g, g) for g in gold]}
        for _, q, gold in _cases(REPO / "rag_test/eval3_offline.py")]
SHORT = [{"split": "short", "query": q,
          "gold": [RENAME.get(g, g) for g in gold]}
         for q, _, gold in _cases(REPO / "rag_test/eval3_short.py")]
CASES = LONG + SHORT


def fetch_vector_legs():
    if CACHE_VEC.exists():
        return json.loads(CACHE_VEC.read_text(encoding="utf-8"))
    cache = {}
    for case in CASES:
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


def load_chunks():
    chunks = []
    for f in sorted(glob.glob(str(REPO / f"data/chunks/{KB_ID}/*.txt"))):
        stem = Path(f).stem
        meta = json.loads(Path(f[:-4] + ".meta.json").read_text(encoding="utf-8"))
        chunks.append({"id": stem,
                       "text": Path(f).read_text(encoding="utf-8"),
                       "source": meta.get("source")})
    return chunks


class Bm25Var:
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
        if df_drop < 1.0:
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


def fuse(vec_ids, bm_ids, K=60, w_vec=1.0, w_bm=1.0, top=TOPK):
    scores, order = {}, []
    for rank, cid in enumerate(vec_ids, 1):
        if cid not in scores:
            order.append(cid)
        scores[cid] = scores.get(cid, 0.0) + w_vec / (K + rank)
    for rank, cid in enumerate(bm_ids, 1):
        if cid not in scores:
            order.append(cid)
        scores[cid] = scores.get(cid, 0.0) + w_bm / (K + rank)
    return sorted(order, key=lambda cid: -scores[cid])[:top]


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


OLD_HP = {"k1": 1.5, "b": 0.75, "K": 60, "w_vec": 1.0, "w_bm": 1.0,
          "depth": 10, "df_drop": 1.0}      # 第三轮前的原始参数
NEW_HP = {"k1": 1.5, "b": 0.45, "K": 60, "w_vec": 1.0, "w_bm": 1.0,
          "depth": 15, "df_drop": 0.03}     # 第四轮最优


class Bench:
    def __init__(self):
        self.vec_legs = fetch_vector_legs()
        chunks = load_chunks()
        self.id2src = {c["id"]: c["source"] for c in chunks}
        self.bm25 = {"old": Bm25Var(chunks, tok_old),
                     "new": Bm25Var(chunks, tok_new)}

    def run(self, case, tok, hp):
        vec_ids = [h["id"] for h in self.vec_legs[case["query"]]]
        bm_ids = self.bm25[tok].search(case["query"], hp["depth"],
                                       k1=hp["k1"], b=hp["b"],
                                       df_drop=hp["df_drop"])
        fused = fuse(vec_ids[:hp["depth"]], bm_ids, K=hp["K"],
                     w_vec=hp["w_vec"], w_bm=hp["w_bm"])
        return [self.id2src[cid] for cid in fused]

    def evaluate(self, tok, hp, cases=None):
        cases = cases or CASES
        rows = [metrics(self.run(c, tok, hp), c["gold"]) for c in cases]
        return {k: sum(r[k] for r in rows) / len(rows)
                for k in ("p@5", "r@5", "p@10", "r@10", "mrr")}


def fmt(a):
    return (f"MRR={a['mrr']:.3f} R@10={a['r@10']:.3f} "
            f"P@5={a['p@5']:.3f} R@5={a['r@5']:.3f}")


def main():
    bench = Bench()
    results = {"meta": {"kb_id": KB_ID, "date": "2026-08-20",
                        "n_cases": len(CASES), "rename": RENAME}}

    # ---- 0. 分词改动审计 ----
    print("== [0] 分词改动审计 ==")
    changed = [c["query"] for c in CASES
               if set(tok_old(c["query"])) != set(tok_new(c["query"]))]
    print(f"  26 个查询中 token 集合发生变化: {len(changed)} 个")
    for q in changed:
        print(f"    {q[:40]:<42} 旧:{sorted(set(tok_old(q)))[:8]}")
        print(f"    {'':<42} 新:{sorted(set(tok_new(q)))[:8]}")
    hyb = bench.bm25["new"]
    comp_tokens = [t for t in hyb.doc_freq if "-" in t or "_" in t]
    dfsum = sum(hyb.doc_freq[t] for t in comp_tokens)
    tot = sum(hyb.doc_freq.values())
    print(f"  新分词语料: 含 -/_ 的复合 token {len(comp_tokens)} 种，"
          f"占总 token 出现次数 {dfsum / tot * 100:.2f}%")
    for t in ("cpdi-nd", "related-key", "rotational-xor"):
        print(f"    df({t}) = {hyb.doc_freq.get(t, 0)}")
    results["tokenizer_audit"] = {
        "queries_changed": len(changed), "changed": changed,
        "compound_token_kinds": len(comp_tokens),
        "compound_token_share": round(dfsum / tot, 6),
        "df": {t: hyb.doc_freq.get(t, 0)
               for t in ("cpdi-nd", "related-key", "rotational-xor")}}

    # ---- 1. 四配置对比 ----
    print("\n== [1] 四配置对比（同一新库语料）==")
    quad = {}
    for tok_name, hp_name in (("old", "OLD_HP"), ("old", "NEW_HP"),
                              ("new", "OLD_HP"), ("new", "NEW_HP")):
        hp = OLD_HP if hp_name == "OLD_HP" else NEW_HP
        a = bench.evaluate(tok_name, hp)
        key = f"tok_{tok_name}+{hp_name}"
        quad[key] = a
        print(f"  {key:<22} {fmt(a)}")
    results["quad"] = quad

    # ---- 2. 新分词下的敏感性 ----
    print("\n== [2] 新分词下敏感性（其余=NEW_HP）==")
    sens = {}
    for label, key, values in [
            ("b", "b", [0.0, 0.15, 0.3, 0.45, 0.6, 0.75]),
            ("depth", "depth", [10, 12, 15, 20, 30]),
            ("df_drop", "df_drop", [1.0, 0.05, 0.03, 0.02, 0.01]),
            ("k1", "k1", [0.9, 1.5, 2.0]),
            ("K", "K", [5, 20, 60, 150]),
            ("w_vec:w_bm", None, [(1, 1), (1.5, 1), (1, 1.5)])]:
        sens[label] = {}
        for v in values:
            hp = dict(NEW_HP)
            if key:
                hp[key] = v
                tag = f"{label}={v}"
            else:
                hp["w_vec"], hp["w_bm"] = v
                tag = f"{label}={v[0]}:{v[1]}"
            a = bench.evaluate("new", hp)
            sens[label][str(v)] = a
            print(f"  {tag:<24} {fmt(a)}")
    results["sensitivity_new_tok"] = sens

    # ---- 3. 新分词下组合网格 + 5 折 ----
    print("\n== [3] 新分词下组合网格 ==")
    GRID = {"k1": [1.2, 1.5], "b": [0.0, 0.15, 0.3, 0.45, 0.6, 0.75],
            "K": [5, 20, 60], "depth": [10, 12, 15, 20],
            "df_drop": [1.0, 0.05, 0.03, 0.02],
            "w_vec": [1.0], "w_bm": [1.0]}
    keys = list(GRID)
    rows = []
    for values in itertools.product(*(GRID[k] for k in keys)):
        hp = dict(zip(keys, values))
        a = bench.evaluate("new", hp)
        rows.append({"hp": hp, "agg": a,
                     "composite": a["mrr"] + a["r@10"]})
    rows.sort(key=lambda r: -r["composite"])
    base_new_tok = bench.evaluate("new", NEW_HP)
    print(f"  现行最优(NEW_HP) 综合={base_new_tok['mrr'] + base_new_tok['r@10']:.3f}")
    print(f"  {'rank':<5}{'b':<6}{'k1':<5}{'K':<5}{'depth':<7}{'df':<7}"
          f"{'MRR':<7}{'R@10':<7}综合")
    for i, r in enumerate(rows[:10], 1):
        h, a = r["hp"], r["agg"]
        print(f"  {i:<5}{h['b']:<6}{h['k1']:<5}{h['K']:<5}{h['depth']:<7}"
              f"{h['df_drop']:<7}{a['mrr']:<7.3f}{a['r@10']:<7.3f}"
              f"{r['composite']:.3f}")
    results["grid_top10"] = rows[:10]
    results["grid_n"] = len(rows)

    # 5 折：NEW_HP vs 网格冠军
    print("\n== [4] 5 折稳健性 ==")
    folds = [[] for _ in range(5)]
    for i, case in enumerate(CASES):
        folds[i % 5].append(case)

    def fold_stats(tok, hp):
        per = []
        for fold in folds:
            ms = [metrics(bench.run(c, tok, hp), c["gold"]) for c in fold]
            per.append({"mrr": sum(x["mrr"] for x in ms) / len(ms),
                        "r@10": sum(x["r@10"] for x in ms) / len(ms)})
        return {"mrr_mean": st.mean(f["mrr"] for f in per),
                "mrr_sd": st.stdev(f["mrr"] for f in per),
                "r10_mean": st.mean(f["r@10"] for f in per),
                "r10_sd": st.stdev(f["r@10"] for f in per), "per_fold": per}

    rob = {"NEW_HP_new_tok": fold_stats("new", NEW_HP)}
    print(f"  NEW_HP(现行)   MRR={rob['NEW_HP_new_tok']['mrr_mean']:.3f}"
          f"±{rob['NEW_HP_new_tok']['mrr_sd']:.3f} "
          f"R@10={rob['NEW_HP_new_tok']['r10_mean']:.3f}"
          f"±{rob['NEW_HP_new_tok']['r10_sd']:.3f}")
    champ = rows[0]["hp"]
    if champ != NEW_HP:
        rob["grid_champion"] = {"hp": champ, **fold_stats("new", champ)}
        print(f"  网格冠军       MRR={rob['grid_champion']['mrr_mean']:.3f}"
              f"±{rob['grid_champion']['mrr_sd']:.3f} "
              f"R@10={rob['grid_champion']['r10_mean']:.3f}"
              f"±{rob['grid_champion']['r10_sd']:.3f}")
    rob["OLD_HP_new_tok"] = fold_stats("new", OLD_HP)
    print(f"  OLD_HP(旧参)   MRR={rob['OLD_HP_new_tok']['mrr_mean']:.3f}"
          f"±{rob['OLD_HP_new_tok']['mrr_sd']:.3f} "
          f"R@10={rob['OLD_HP_new_tok']['r10_mean']:.3f}"
          f"±{rob['OLD_HP_new_tok']['r10_sd']:.3f}")
    results["robustness"] = rob

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n已写入 {OUT.name}")


if __name__ == "__main__":
    main()
