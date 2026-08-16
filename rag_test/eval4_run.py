"""第四轮-阶段2：超参数敏感性实验（每次只动一个参数，其余保持基线）。

运行：.venv/bin/python rag_test/eval4_run.py
输出：打印表格 + 追加写入 rag_test/eval4_sensitivity.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from eval4_hparam import (ALL_CASES, BASELINE, Bench, OUT, REPO, metrics)

def show(tag, agg):
    print(f"  {tag:<38} MRR={agg['mrr']:.3f}  R@10={agg['r@10']:.3f}  "
          f"P@5={agg['p@5']:.3f}  R@5={agg['r@5']:.3f}")


def split_eval(bench, cfg):
    long_cases = [c for c in ALL_CASES if c["split"] == "long"]
    short_cases = [c for c in ALL_CASES if c["split"] == "short"]
    return {"all": bench.evaluate(cfg),
            "long": bench.evaluate(cfg, long_cases),
            "short": bench.evaluate(cfg, short_cases)}


def main():
    bench = Bench()
    results = {}

    print("== 参照线 ==")
    vec_only = {**BASELINE, "w_bm": 0.0}          # 等价纯向量（BM25 权重 0）
    show("纯向量（参照）", bench.evaluate(vec_only))
    show("基线 hybrid（现行参数）", bench.evaluate(BASELINE))
    results["vector_only"] = split_eval(bench, vec_only)
    results["baseline"] = split_eval(bench, BASELINE)

    grids = {
        "k1（BM25 词频饱和）": {"k1": [0.6, 0.9, 1.2, 1.5, 2.0]},
        "b（BM25 长度归一化）": {"b": [0.0, 0.3, 0.6, 0.75, 0.9, 1.0]},
        "K（RRF 融合常数）": {"K": [5, 10, 20, 30, 60, 100, 150]},
        "权重（向量:BM25）": {"w_vec/w_bm": [(1, 1), (1.5, 1), (2, 1),
                                            (1, 1.5), (1, 2)]},
        "depth（每腿候选深度）": {"depth": [8, 10, 15, 20, 30, 50]},
        "tokenizer（分词粒度）": {"tokenizer": ["bigram", "uni_bi"]},
        "df_drop（查询词DF过滤）": {"df_drop": [1.0, 0.2, 0.1, 0.05, 0.03, 0.02]},
    }
    for name, grid in grids.items():
        print(f"\n== {name} ==")
        results[name] = {}
        for key, values in grid.items():
            for v in values:
                if key == "w_vec/w_bm":
                    cfg = {**BASELINE, "w_vec": v[0], "w_bm": v[1]}
                    tag = f"{key}={v[0]}:{v[1]}"
                else:
                    cfg = {**BASELINE, key: v}
                    tag = f"{key}={v}"
                r = split_eval(bench, cfg)
                results[name][tag] = r
                show(tag + f"  [长MRR {r['long']['mrr']:.2f}/短MRR {r['short']['mrr']:.2f}]",
                     r["all"])

    with open(OUT.with_name("eval4_sensitivity.json"), "w",
              encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n已写入 eval4_sensitivity.json")


if __name__ == "__main__":
    main()
