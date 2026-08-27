"""第三轮补充：短关键词查询电池（与长查询配对对照）。

动机：长自然语言查询中，常见中文 bigram 在 BM25 加性打分下叠加，
会稀释/挤出精确 token 的命中（已在 HDND 用例观测到）。
本电池用"关键词式短查询"测同样的金标准，对照两种查询风格下
vector 与 hybrid 的表现差异。

输出：rag_test/eval3_short_results.json
"""
import json
import urllib.parse
import urllib.request

BASE = "http://127.0.0.1:8000"
KB_ID = "ad4a296fda7c"
USER_ID = "241550d8be134b62b47895dbc59aaa88"
K = 10

# (短查询, 对应的长查询编号说明, 金标准)
CASES = [
    ("HDND", "对应长查询: HDND 高阶差分神经区分器是什么", ["HDND.pdf"]),
    ("CPDI-ND", "对应长查询: CPDI-ND 自注意力跨对差分交互神经区分器", ["CPDI-ND.docx"]),
    ("SKNet", "对应长查询: SKNet 构建差分神经区分器 Speck Simon", ["2026Xu-SKNet.pdf"]),
    ("Fourier analysis neural distinguishers", "对应长查询: 神经区分器的傅里叶分析",
     ["2026ToSC.pdf"]),
    ("partial decryption feature engineering", "对应长查询: 如何用部分解密作为特征工程改进神经区分器",
     ["2025Bellini-GPD.pdf"]),
    ("truncated differential HIGHT key recovery", "对应长查询: HIGHT 上的截断差分神经密钥恢复攻击",
     ["2024Seo.pdf"]),
    ("Gimli differential distinguishers", "对应长查询: Gimli 轻量级密码的机器学习差分区分器",
     ["2020Baksi.pdf", "2023Bellini.pdf", "2024Bellini-SALSA.pdf"]),
    ("SIMECK related-key neural distinguisher", "对应长查询: SIMECK 相关密钥神经区分器改进",
     ["2022Lu.pdf", "2020Yuan.pdf", "2025RelatedKey.pdf", "2024Yuan-Simeck.pdf"]),
]


def search(query, mode):
    qs = urllib.parse.urlencode(
        {"query": query, "k": K, "mode": mode, "user_id": USER_ID})
    with urllib.request.urlopen(f"{BASE}/api/v1/kbs/{KB_ID}/search?{qs}",
                                timeout=120) as r:
        return json.load(r)


def dedup_docs(hits):
    docs, seen = [], set()
    for h in hits:
        src = (h.get("metadata") or {}).get("source")
        if src and src not in seen:
            seen.add(src)
            docs.append(src)
    return docs


def metrics(docs, gold):
    gold_set = set(gold)
    h5 = sum(d in gold_set for d in docs[:5])
    h10 = sum(d in gold_set for d in docs[:10])
    mrr = 0.0
    for rank, d in enumerate(docs, 1):
        if d in gold_set:
            mrr = 1.0 / rank
            break
    return {"p@5": round(h5 / 5, 3), "r@5": round(h5 / len(gold_set), 3),
            "p@10": round(h10 / 10, 3), "r@10": round(h10 / len(gold_set), 3),
            "mrr": round(mrr, 3),
            "gold_hit": sorted(gold_set & set(docs)),
            "gold_miss": sorted(gold_set - set(docs))}


def main():
    results = []
    for query, note, gold in CASES:
        row = {"query": query, "note": note, "gold": gold}
        for mode in ("vector", "hybrid"):
            hits = search(query, mode)
            docs = dedup_docs(hits)
            row[mode] = {**metrics(docs, gold), "docs_ranked": docs}
            m = row[mode]
            print(f"{query[:40]:<42} {mode:<7} P@5={m['p@5']:.2f} "
                  f"R@10={m['r@10']:.2f} MRR={m['mrr']:.2f}")
        results.append(row)

    def agg(rows, mode):
        n = len(rows)
        return {k: round(sum(r[mode][k] for r in rows) / n, 3)
                for k in ("p@5", "r@5", "p@10", "r@10", "mrr")} | {"n": n}

    summary = {"vector": agg(results, "vector"), "hybrid": agg(results, "hybrid")}
    json.dump({"summary": summary, "cases": results,
               "meta": {"kb_id": KB_ID, "k": K, "date": "2026-08-16"}},
              open("/home/pyp/interview_preparation/ra-agent/rag_test/eval3_short_results.json",
                   "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n短查询总体: vector =", summary["vector"])
    print("           hybrid  =", summary["hybrid"])


if __name__ == "__main__":
    main()
