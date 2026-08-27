"""第五轮-线上终验（2026-08-21）：服务实际运行目录为 /home/pyp/research-assistant/ra-agent，
知识库 3a388c1c26ec（密码学+深度学习，81 篇 / 7308 chunks，语料与 408cbfa4b504 相同）。

本脚本可重复运行，用于部署目录分词修复【前/后】的线上行为对照：
- 线上 hybrid 26 查询端到端指标（文档级 MRR/R@10 等）
- chunk 级探针对比：判定线上运行的是 旧分词 还是 新分词（保留 -/_）
- 向量腿与 BM25 语料均来自部署目录实际数据

运行：.venv/bin/python rag_test/eval6_live_final.py [标签]
输出：rag_test/eval6_live_<标签>_results.json
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

sys.path.insert(0, "/home/pyp/interview_preparation/ra-agent")          # rank_bm25 可用
from rank_bm25 import BM25Okapi                                         # noqa: E402

DEPLOY = Path("/home/pyp/research-assistant/ra-agent")   # 服务实际运行目录
KB_ID = "3a388c1c26ec"
BASE = "http://127.0.0.1:8000"
USER_ID = "241550d8be134b62b47895dbc59aaa88"
TAG = sys.argv[1] if len(sys.argv) > 1 else "run"
CACHE_VEC = Path(f"/home/pyp/interview_preparation/ra-agent/rag_test/eval6_vector_legs.json")
OUT = Path(f"/home/pyp/interview_preparation/ra-agent/rag_test/eval6_live_{TAG}_results.json")

RENAME = {"HDND.pdf": "HDND JCST.pdf"}

# ---- 新旧分词器（均在脚本内固定复刻，不受任何一份代码后续修改影响）----
_CJK = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]+")

def _tok(text, word_cls):
    toks = []
    for seg in re.findall(rf"[\u4e00-\u9fff\u3400-\u4dbf]+|[{word_cls}]+", text.lower()):
        if _CJK.fullmatch(seg):
            if len(seg) == 1:
                toks.append(seg)
            else:
                toks.extend(seg[i:i + 2] for i in range(len(seg) - 1))
        else:
            toks.append(seg)
    return toks

def tok_old(t):   # 旧分词：英文词块 [a-zA-Z0-9]+
    return _tok(t, r"a-zA-Z0-9")

def tok_new(t):   # 新分词：英文词块保留 - 和 _
    return _tok(t, r"a-zA-Z0-9_\-")

NEW_HP = {"k1": 1.5, "b": 0.45, "K": 60, "w_vec": 1.0, "w_bm": 1.0,
          "depth": 15, "df_drop": 0.03}
OLD_HP = {"k1": 1.5, "b": 0.75, "K": 60, "w_vec": 1.0, "w_bm": 1.0,
          "depth": 10, "df_drop": 1.0}

def _cases(path):
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CASES

IP = Path("/home/pyp/interview_preparation/ra-agent/rag_test")
LONG = [{"split": "long", "query": q, "gold": [RENAME.get(g, g) for g in gold]}
        for _, q, gold in _cases(IP / "eval3_offline.py")]
SHORT = [{"split": "short", "query": q, "gold": [RENAME.get(g, g) for g in gold]}
         for q, _, gold in _cases(IP / "eval3_short.py")]
CASES = LONG + SHORT


def api(query, mode, k=10):
    qs = urllib.parse.urlencode(
        {"query": query, "k": k, "mode": mode, "user_id": USER_ID})
    with urllib.request.urlopen(
            f"{BASE}/api/v1/kbs/{KB_ID}/search?{qs}", timeout=120) as r:
        return json.load(r)


def fetch_vector_legs():
    if CACHE_VEC.exists():
        return json.loads(CACHE_VEC.read_text(encoding="utf-8"))
    cache = {}
    for case in CASES:
        q = case["query"]
        if q in cache:
            continue
        cache[q] = [{"id": h["id"]} for h in api(q, "vector", k=50)]
    CACHE_VEC.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    return cache


def load_chunks():
    chunks = []
    for f in sorted(glob.glob(str(DEPLOY / f"data/chunks/{KB_ID}/*.txt"))):
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

    def search(self, query, k, k1, b, df_drop):
        self.bm.k1, self.bm.b = k1, b
        q = set(self.tokenizer(query))
        if df_drop < 1.0:
            cap = df_drop * len(self.items)
            q = {t for t in q if self.doc_freq[t] <= cap}
        if not q:
            return []
        scores = self.bm.get_scores(list(q))
        ranked = heapq.nlargest(min(k, len(scores)), range(len(scores)),
                                key=lambda i: scores[i])
        return [self.items[i]["id"] for i in ranked if q & self.token_docs[i]]


def fuse(vec_ids, bm_ids, K, w_vec, w_bm, top=10):
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


def main():
    legs = fetch_vector_legs()
    chunks = load_chunks()
    id2src = {c["id"]: c["source"] for c in chunks}
    bm = {"old": Bm25Var(chunks, tok_old), "new": Bm25Var(chunks, tok_new)}
    print(f"[{TAG}] 语料 {len(chunks)} chunks（部署目录 {DEPLOY.name}）")

    # 线上 hybrid 26 查询
    rows, live_ids_map = [], {}
    for case in CASES:
        hits = api(case["query"], "hybrid")
        live_ids_map[case["query"]] = [h["id"] for h in hits]
        docs, seen = [], set()
        for h in hits:
            src = (h.get("metadata") or {}).get("source")
            if src and src not in seen:
                seen.add(src)
                docs.append(src)
        rows.append(metrics(docs, case["gold"]))
    live_agg = {k: sum(r[k] for r in rows) / len(rows)
                for k in ("p@5", "r@5", "p@10", "r@10", "mrr")}
    print(f"  线上端到端: MRR={live_agg['mrr']:.3f} R@10={live_agg['r@10']:.3f} "
          f"P@5={live_agg['p@5']:.3f} R@5={live_agg['r@5']:.3f}")

    # 离线四配置（本语料）
    def offline_docs(case, tok, hp):
        vec = [h["id"] for h in legs[case["query"]]][:hp["depth"]]
        bml = bm[tok].search(case["query"], hp["depth"], hp["k1"], hp["b"], hp["df_drop"])
        return [id2src[c] for c in fuse(vec, bml, hp["K"], hp["w_vec"], hp["w_bm"])]

    offline = {}
    for name, tok, hp in [("旧分词+旧超参", "old", OLD_HP),
                          ("旧分词+新超参", "old", NEW_HP),
                          ("新分词+新超参", "new", NEW_HP)]:
        rs = [metrics(offline_docs(c, tok, hp), c["gold"]) for c in CASES]
        offline[name] = {k: sum(r[k] for r in rs) / len(rs) for k in rs[0]}
        a = offline[name]
        print(f"  离线{name}: MRR={a['mrr']:.3f} R@10={a['r@10']:.3f}")

    # 探针判定线上分词版本
    verdicts = {}
    for q in ["HDND 高阶差分神经区分器是什么", "CPDI-ND",
              "SIMECK related-key neural distinguisher",
              "Gimli differential distinguishers"]:
        live = live_ids_map[q]
        case = next(c for c in CASES if c["query"] == q)

        def off_ids(tok, hp):
            vec = [h["id"] for h in legs[q]][:hp["depth"]]
            bml = bm[tok].search(q, hp["depth"], hp["k1"], hp["b"], hp["df_drop"])
            return fuse(vec, bml, hp["K"], hp["w_vec"], hp["w_bm"])

        match_old = live == off_ids("old", NEW_HP)
        match_new = live == off_ids("new", NEW_HP)
        match_oldhp = live == off_ids("old", OLD_HP)
        if match_new and not match_old:
            v = "新分词+新超参"
        elif match_old and not match_new:
            v = "旧分词+新超参"
        elif match_new and match_old:
            v = "新旧分词结果相同（无法区分），超参=新"
        elif match_oldhp:
            v = "旧分词+旧超参"
        else:
            v = "均不匹配（需排查）"
        verdicts[q] = v
        print(f"  探针[{q[:24]}] -> {v}")

    json.dump({"tag": TAG, "live_agg": live_agg, "offline": offline,
               "probe_verdicts": verdicts,
               "meta": {"kb": KB_ID, "deploy": str(DEPLOY),
                        "chunks": len(chunks), "date": "2026-08-21"}},
              open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"已写入 {OUT.name}")


if __name__ == "__main__":
    main()
