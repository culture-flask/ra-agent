"""第四轮-阶段3：组合网格搜索 + 5折稳健性验证。

网格取敏感性实验中的关键参数（b / depth / df_drop / K / k1），
其余固定为基线或敏感性最优（tokenizer=bigram，权重 1:1）。

运行：.venv/bin/python rag_test/eval4_combo.py
输出：rag_test/eval4_combo_results.json
"""
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from eval4_hparam import ALL_CASES, BASELINE, Bench

GRID = {
    "k1": [1.2, 1.5],
    "b": [0.0, 0.15, 0.3, 0.45, 0.6, 0.75],
    "K": [5, 20, 60],
    "depth": [10, 12, 15, 20],
    "df_drop": [1.0, 0.05, 0.03, 0.02],
    "tokenizer": ["bigram"],
    "w_vec": [1.0], "w_bm": [1.0],
}


def composite(agg):
    """排序用综合分：MRR 与 R@10 等权（两者对 RAG 都关键）。"""
    return agg["mrr"] + agg["r@10"]


def main():
    bench = Bench()
    keys = list(GRID)
    combos = list(itertools.product(*(GRID[k] for k in keys)))
    print(f"网格规模: {len(combos)} 个配置")

    rows = []
    for i, values in enumerate(combos, 1):
        cfg = dict(zip(keys, values))
        agg = bench.evaluate(cfg)
        rows.append({"cfg": cfg, "agg": agg, "composite": composite(agg)})
        if i % 50 == 0:
            print(f"  ... {i}/{len(combos)}")

    rows.sort(key=lambda r: -r["composite"])
    base = bench.evaluate(BASELINE)

    print(f"\n基线: MRR={base['mrr']:.3f} R@10={base['r@10']:.3f} "
          f"综合={composite(base):.3f}")
    print("\nTop 15 配置：")
    print(f"{'rank':<5}{'b':<6}{'k1':<5}{'K':<5}{'depth':<7}{'df_drop':<9}"
          f"{'MRR':<7}{'R@10':<7}{'P@5':<7}{'R@5':<7}综合")
    for i, r in enumerate(rows[:15], 1):
        c, a = r["cfg"], r["agg"]
        print(f"{i:<5}{c['b']:<6}{c['k1']:<5}{c['K']:<5}{c['depth']:<7}"
              f"{c['df_drop']:<9}{a['mrr']:<7.3f}{a['r@10']:<7.3f}"
              f"{a['p@5']:<7.3f}{a['r@5']:<7.3f}{r['composite']:.3f}")

    # ---------- 5 折稳健性：top5 配置 vs 基线 ----------
    print("\n== 5 折稳健性（26 查询轮转分 5 折，报告各折指标）==")
    folds = [[] for _ in range(5)]
    for i, case in enumerate(ALL_CASES):     # 轮转分折，长短混合均衡
        folds[i % 5].append(case)

    def fold_stats(cfg):
        per_fold = []
        for fold in folds:
            m = []
            for case in fold:
                docs = bench.run(case, cfg)
                from eval4_hparam import metrics
                m.append(metrics(docs, case["gold"]))
            per_fold.append({"mrr": sum(x["mrr"] for x in m) / len(m),
                             "r@10": sum(x["r@10"] for x in m) / len(m)})
        import statistics as st
        return {"mrr_mean": st.mean(f["mrr"] for f in per_fold),
                "mrr_stdev": st.stdev(f["mrr"] for f in per_fold),
                "r10_mean": st.mean(f["r@10"] for f in per_fold),
                "r10_stdev": st.stdev(f["r@10"] for f in per_fold),
                "per_fold": per_fold}

    robust = {}
    robust["baseline"] = fold_stats(BASELINE)
    print(f"  基线           MRR={robust['baseline']['mrr_mean']:.3f}"
          f"±{robust['baseline']['mrr_stdev']:.3f}  "
          f"R@10={robust['baseline']['r10_mean']:.3f}"
          f"±{robust['baseline']['r10_stdev']:.3f}")
    for i, r in enumerate(rows[:5], 1):
        name = f"top{i}"
        robust[name] = {"cfg": r["cfg"], **fold_stats(r["cfg"])}
        print(f"  {name} b={r['cfg']['b']} depth={r['cfg']['depth']} "
              f"df={r['cfg']['df_drop']} K={r['cfg']['K']}  "
              f"MRR={robust[name]['mrr_mean']:.3f}±{robust[name]['mrr_stdev']:.3f}  "
              f"R@10={robust[name]['r10_mean']:.3f}±{robust[name]['r10_stdev']:.3f}")

    with open("rag_test/eval4_combo_results.json", "w", encoding="utf-8") as f:
        json.dump({"grid": GRID, "top20": rows[:20],
                   "baseline_agg": base, "robustness": robust},
                  f, ensure_ascii=False, indent=2)
    print("\n已写入 eval4_combo_results.json")


if __name__ == "__main__":
    main()
