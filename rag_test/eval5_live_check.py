"""第五轮-重启后线上实测（2026-08-20）：验证服务重启后分词修复+第四轮超参数生效。

1. 26 查询直接打线上 hybrid 接口（k=10），文档级指标应与离线/生产端到端
   预期一致（MRR=0.607 / R@10=0.747）。
2. chunk 级探针对比：线上应与"新分词+新超参"复现 10/10 一致（含 Gimli
   ——新旧分词唯一有结果差异的查询）。
3. 向量腿抽查：重启前后应不变（嵌入端点相同）。

运行：.venv/bin/python rag_test/eval5_live_check.py
输出：rag_test/eval5_live_check_results.json
"""
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path("/home/pyp/interview_preparation/ra-agent")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "rag_test"))
from eval5_reeval import (BASE, CASES, KB_ID, NEW_HP, OLD_HP, USER_ID,  # noqa: E402
                          Bench, fuse, metrics)

def api(query, mode, k=10):
    qs = urllib.parse.urlencode(
        {"query": query, "k": k, "mode": mode, "user_id": USER_ID})
    with urllib.request.urlopen(
            f"{BASE}/api/v1/kbs/{KB_ID}/search?{qs}", timeout=120) as r:
        return json.load(r)


def main():
    bench = Bench()
    out = {}

    # ---- 1. 26 查询线上 hybrid 端到端 ----
    print("== [1] 线上 hybrid 端到端（26 查询，k=10）==")
    rows, live_docs = [], {}
    for case in CASES:
        hits = api(case["query"], "hybrid")
        ids = [h["id"] for h in hits]
        docs, seen = [], set()
        for h in hits:
            src = (h.get("metadata") or {}).get("source")
            if src and src not in seen:
                seen.add(src)
                docs.append(src)
        live_docs[case["query"]] = (ids, docs)
        rows.append(metrics(docs, case["gold"]))
    live_agg = {k: sum(r[k] for r in rows) / len(rows)
                for k in ("p@5", "r@5", "p@10", "r@10", "mrr")}
    print(f"  线上实测: MRR={live_agg['mrr']:.3f} R@10={live_agg['r@10']:.3f} "
          f"P@5={live_agg['p@5']:.3f} R@5={live_agg['r@5']:.3f}")
    print("  预期(新分词+新超参): MRR=0.607 R@10=0.747 P@5=0.169 R@5=0.696")
    ok = (abs(live_agg["mrr"] - 0.607) < 0.002
          and abs(live_agg["r@10"] - 0.747) < 0.002)
    print(f"  {'✓ 与预期一致' if ok else '✗ 与预期不一致，需排查'}")
    out["live_agg"] = live_agg
    out["live_matches_expected"] = ok

    # ---- 2. chunk 级探针：确认新分词生效 ----
    print("\n== [2] chunk 级探针（线上 vs 离线复现）==")

    def offline_ids(query, tok, hp):
        vec_ids = [h["id"] for h in bench.vec_legs[query]]
        bm_ids = bench.bm25[tok].search(query, hp["depth"], k1=hp["k1"],
                                        b=hp["b"], df_drop=hp["df_drop"])
        return fuse(vec_ids[:hp["depth"]], bm_ids, K=hp["K"],
                    w_vec=hp["w_vec"], w_bm=hp["w_bm"])

    probes = ["HDND 高阶差分神经区分器是什么", "CPDI-ND",
              "SIMECK related-key neural distinguisher",
              "Gimli differential distinguishers"]
    probe_out = {}
    for q in probes:
        live_ids = live_docs[q][0]
        row = {}
        for name, tok, hp in [("旧分词+新超参", "old", NEW_HP),
                              ("新分词+新超参", "new", NEW_HP),
                              ("旧分词+旧超参", "old", OLD_HP)]:
            off = offline_ids(q, tok, hp)
            row[name] = (live_ids == off)
        probe_out[q] = row
        verdict = ("新分词已生效" if row["新分词+新超参"] and not row["旧分词+新超参"]
                   else ("两分词结果相同，无法区分" if row["新分词+新超参"] and row["旧分词+新超参"]
                         else "✗ 均不匹配"))
        print(f"  {q[:36]:<38} 新超参匹配:{row['新分词+新超参']} "
              f"旧超参匹配:{row['旧分词+旧超参']} -> {verdict}")
    out["probes"] = probe_out

    # ---- 3. 向量腿抽查（重启前后一致性）----
    print("\n== [3] 向量腿抽查（重启前缓存 vs 重启后实取）==")
    spot = {}
    for q in ["HDND 高阶差分神经区分器是什么", "Gimli differential distinguishers"]:
        fresh = [h["id"] for h in api(q, "vector", k=50)]
        cached = [h["id"] for h in bench.vec_legs[q]]
        same = fresh[:20] == cached[:20]
        spot[q] = same
        print(f"  {q[:36]:<38} top20 一致: {same}")
    out["vector_leg_stable"] = spot

    with open(REPO / "rag_test/eval5_live_check_results.json", "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n已写入 eval5_live_check_results.json")


if __name__ == "__main__":
    main()
