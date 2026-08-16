"""第三轮对话级评测：vector vs hybrid 两个持久会话（2026-08-16）。

- 两个会话问完全相同的 8 个问题（顺序一致，模拟真实使用）。
- user_id 传 pyp 的内部 ID → LLM 自动使用 pyp 的用户级默认配置
  （custom / deepseek-v4-flash @ opencode.ai），避开系统默认模型限流。
- 检索参数统一：per_kb_k=5, total_k=5, parent_groups=0（chunk 级检索，
  便于对比来源），temperature=0（降低随机性，回答差异主要来自检索）。
- 会话保留在服务端，测试后不删除。

输出：rag_test/eval3_chat_results.json
"""
import json
import time
import urllib.request

BASE = "http://127.0.0.1:8000"
USER_ID = "241550d8be134b62b47895dbc59aaa88"   # pyp
SESSIONS = {
    "vector": "eval3-vector-20260816",
    "hybrid": "eval3-hybrid-20260816",
}
COMMON = {
    "user_id": USER_ID,
    "per_kb_k": 5,
    "total_k": 5,
    "parent_groups": 0,
    "temperature": 0,
}
QUESTIONS = [
    "请检索知识库回答：Gohr 在 CRYPTO 2019 提出的神经区分器方法核心思想是什么？",
    "请检索知识库回答：HDND 是什么？",
    "请检索知识库回答：哪些论文研究了 ASCON 或 GIFT 的神经区分器？",
    "请检索知识库回答：SIMECK 的相关密钥神经区分器有哪些改进工作？",
    "请检索知识库回答：CPDI-ND 是什么？",
    "请检索知识库回答：神经区分器领域有哪些综述性论文？",
    "请检索知识库回答：如何用部分解密作为特征工程来改进神经区分器？",
    "请检索知识库回答：Gimli 的机器学习差分区分器研究有哪些？",
]


def chat(session_id: str, message: str, retrieval_mode: str) -> dict:
    body = json.dumps({
        "session_id": session_id,
        "message": message,
        "retrieval_mode": retrieval_mode,
        **COMMON,
    }, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE}/api/v1/chat", data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def slim_retrievals(retrievals: list) -> list:
    out = []
    for h in retrievals:
        meta = h.get("metadata") or {}
        out.append({
            "source": h.get("source") or meta.get("source"),
            "page": h.get("page") or meta.get("page"),
            "method": h.get("method"),
            "type": h.get("type"),
            "distance": h.get("distance"),
            "bm25_score": h.get("bm25_score"),
            "score": h.get("score"),
            "text_head": str(h.get("text", ""))[:120],
        })
    return out


def main():
    results = {"meta": {"user_id": USER_ID, "params": COMMON,
                        "sessions": SESSIONS, "date": "2026-08-16"},
               "runs": {}}
    for mode, session_id in SESSIONS.items():
        print(f"\n########## 会话 mode={mode} session={session_id} ##########")
        runs = []
        for i, q in enumerate(QUESTIONS, 1):
            t0 = time.time()
            try:
                resp = chat(session_id, q, mode)
                answer = resp.get("answer", "")
                rets = slim_retrievals(resp.get("retrievals", []))
                err = None
            except Exception as e:
                answer, rets, err = "", [], f"{type(e).__name__}: {e}"
            dt = time.time() - t0
            runs.append({"q": q, "answer": answer, "retrievals": rets,
                         "error": err, "latency_s": round(dt, 1)})
            srcs = [r["source"] for r in rets if r["source"]]
            print(f"  Q{i} [{dt:5.1f}s] 检索来源: {srcs}")
            if err:
                print(f"     ERROR: {err}")
            else:
                print(f"     答: {answer[:120].replace(chr(10), ' ')}")
        results["runs"][mode] = runs
        # 每个会话完成即落盘，避免中途失败丢全部结果
        with open("/home/pyp/interview_preparation/ra-agent/rag_test/eval3_chat_results.json",
                  "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n完成。会话已保留：", SESSIONS)


if __name__ == "__main__":
    main()
