"""第三轮离线检索评测：vector vs hybrid（2026-08-16）。

与上一轮（DeepSeek）的差异：
- 金标准全部经磁盘 chunk 关键词验证（grep 证实 gold 文档确实含对应概念，
  且统计了关键词在全库的分布，剔除了不可命中的用例如 "SALSA"——
  该词在 2024Bellini-SALSA.pdf 提取文本中根本不出现）。
- 用例按查询类型分四类，便于回答"什么场景该用哪种检索"。
- 指标按文档级去重计算（同一文档多个 chunk 只按首次出现计排名）。

运行：python3 rag_test/eval3_offline.py
输出：rag_test/eval3_offline_results.json
"""
import json
import time
import urllib.parse
import urllib.request

BASE = "http://127.0.0.1:8000"
KB_ID = "ad4a296fda7c"
USER_ID = "241550d8be134b62b47895dbc59aaa88"   # pyp
K = 10

# (类别, 查询, 金标准文档列表)
CASES = [
    # A. 精确缩写/专名查询 —— 词法命中是关键，BM25 预期占优
    ("A-精确专名", "HDND 高阶差分神经区分器是什么", ["HDND.pdf"]),
    ("A-精确专名", "CPDI-ND 自注意力跨对差分交互神经区分器", ["CPDI-ND.docx"]),
    ("A-精确专名", "SKNet 构建差分神经区分器 Speck Simon", ["2026Xu-SKNet.pdf"]),
    ("A-精确专名", "SoK: 6 Years of Neural Differential Cryptanalysis", ["2025Bellini-SoK.pdf"]),
    ("A-精确专名", "Rotational-XOR distinguishers for AND-RX block ciphers",
     ["2024Ebrahimi.pdf", "2024Ebrahimi-cn.pdf"]),
    ("A-精确专名", "Cautious Optimizers: Improving Training with One Line of Code",
     ["2024Liang-Cautious.pdf"]),
    # B. 算法/主题查询 —— 语义 + 词法都要发力
    ("B-算法主题", "Gimli 轻量级密码的机器学习差分区分器",
     ["2020Baksi.pdf", "2023Bellini.pdf", "2024Bellini-SALSA.pdf"]),
    ("B-算法主题", "ASCON 和 GIFT-128 的神经差分区分器", ["2024Shen.pdf", "2026Ahmad.pdf"]),
    ("B-算法主题", "PRESENT 和 SKINNY 的改进差分神经区分器", ["2025Guo.pdf"]),
    ("B-算法主题", "HIGHT 上的截断差分神经密钥恢复攻击", ["2024Seo.pdf", "2023Tcydenova.pdf"]),
    ("B-算法主题", "SIMECK 相关密钥神经区分器改进",
     ["2022Lu.pdf", "2020Yuan.pdf", "2025RelatedKey.pdf", "2024Yuan-Simeck.pdf"]),
    # C. 中文语义查询、答案在英文论文 —— 词法无法命中，纯考向量
    ("C-中文问英文", "如何用部分解密作为特征工程改进神经区分器", ["2025Bellini-GPD.pdf"]),
    ("C-中文问英文", "神经区分器的傅里叶分析", ["2026ToSC.pdf"]),
    ("C-中文问英文", "用神经网络架构搜索最大化信息熵构建区分器", ["2026Ren.pdf"]),
    ("C-中文问英文", "剪枝和量化加速深度神经网络推理", ["2021Liang-PruningQuantization.pdf"]),
    ("C-中文问英文", "侧信道攻击中的数据去噪与特征融合", ["2025Huang.pdf"]),
    # D. 跨语言陷阱：中文概念在中文论文和英文论文各有一篇
    ("D-跨语言", "基于深度学习的多面体差分攻击", ["2021Polytope.pdf", "2026Mirzaali.pdf"]),
    # E. 英文标题原题精确查询（上一轮失败用例的修正版）
    ("E-标题精确", "Improving Attacks on Round-Reduced Speck32/64 Using Deep Learning",
     ["2019Gohr.pdf"]),
]


def search(query: str, mode: str) -> list[dict]:
    qs = urllib.parse.urlencode({
        "query": query, "k": K, "mode": mode, "user_id": USER_ID})
    with urllib.request.urlopen(f"{BASE}/api/v1/kbs/{KB_ID}/search?{qs}",
                                timeout=120) as r:
        return json.load(r)


def dedup_docs(hits: list[dict]) -> list[str]:
    """命中 chunk 序列 → 文档序列（保序去重，排名取首次出现位置）。"""
    docs, seen = [], set()
    for h in hits:
        src = (h.get("metadata") or {}).get("source")
        if src and src not in seen:
            seen.add(src)
            docs.append(src)
    return docs


def metrics(docs: list[str], gold: list[str]) -> dict:
    gold_set = set(gold)
    n_gold = len(gold_set)
    hits5 = sum(d in gold_set for d in docs[:5])
    hits10 = sum(d in gold_set for d in docs[:10])
    mrr = 0.0
    for rank, d in enumerate(docs, start=1):
        if d in gold_set:
            mrr = 1.0 / rank
            break
    return {
        "p@5": round(hits5 / 5, 3),
        "r@5": round(hits5 / n_gold, 3),
        "p@10": round(hits10 / 10, 3),
        "r@10": round(hits10 / n_gold, 3),
        "mrr": round(mrr, 3),
        "gold_hit": sorted(gold_set & set(docs)),
        "gold_miss": sorted(gold_set - set(docs)),
    }


def main():
    results = []
    for cat, query, gold in CASES:
        row = {"category": cat, "query": query, "gold": gold}
        for mode in ("vector", "hybrid"):
            t0 = time.time()
            hits = search(query, mode)
            dt = time.time() - t0
            docs = dedup_docs(hits)
            m = metrics(docs, gold)
            methods = {}
            for h in hits:
                methods[h.get("method", "?")] = methods.get(h.get("method", "?"), 0) + 1
            row[mode] = {
                **m,
                "docs_ranked": docs,
                "chunk_methods": methods,
                "latency_s": round(dt, 2),
                "n_chunks": len(hits),
            }
            print(f"[{cat}] {query[:36]:<38} {mode:<7} "
                  f"P@5={m['p@5']:.2f} R@5={m['r@5']:.2f} "
                  f"P@10={m['p@10']:.2f} R@10={m['r@10']:.2f} MRR={m['mrr']:.2f}")
        results.append(row)

    # 汇总：总体 + 分类别
    def agg(rows, mode):
        keys = ("p@5", "r@5", "p@10", "r@10", "mrr")
        n = len(rows)
        return {k: round(sum(r[mode][k] for r in rows) / n, 3) for k in keys} | {"n": n}

    summary = {"vector": agg(results, "vector"), "hybrid": agg(results, "hybrid")}
    by_cat = {}
    for cat in dict.fromkeys(r["category"] for r in results):   # 保序去重
        rows = [r for r in results if r["category"] == cat]
        by_cat[cat] = {"vector": agg(rows, "vector"), "hybrid": agg(rows, "hybrid")}

    out = {"summary": summary, "by_category": by_cat, "cases": results,
           "meta": {"kb_id": KB_ID, "k": K, "user_id": USER_ID,
                    "date": "2026-08-16", "n_cases": len(CASES)}}
    with open("/home/pyp/interview_preparation/ra-agent/rag_test/eval3_offline_results.json",
              "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("\n===== 总体 =====")
    print(f"vector: {summary['vector']}")
    print(f"hybrid: {summary['hybrid']}")
    print("\n===== 分类别 =====")
    for cat, v in by_cat.items():
        print(f"{cat}: vector={v['vector']}")
        print(f"{'':>{len(cat)+2}}hybrid={v['hybrid']}")


if __name__ == "__main__":
    main()
