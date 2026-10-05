"""重排（rerank）抽象：用交叉编码器对召回的候选做精排，提升最终排名质量。

- 协议：Cohere 兼容的 POST {base_url}/rerank
  请求 {"model", "query", "documents": [...], "top_n"}；
  响应 {"results": [{"index": <documents 下标>, "relevance_score": <float>}, ...]}
  （bge-reranker-v2-m3 等部署在 vLLM / Xinference / TEI 的端点均为此协议）
- 与嵌入/BM25 解耦：只吃 (query, 候选文本) 出相关性分，不感知库结构与向量空间
- 输入裁剪：多数部署对单条输入有 token 上限（本端点 512），超限直接返回 500
  而非自动截断——故按预算裁剪候选正文（query 保留完整），并留余量 + 超限重试
- 失败可降级：端点不可达 / 超时 / 协议不符时抛异常，由调用方回退原融合排序
"""

import re
from dataclasses import dataclass

import httpx


@dataclass
class RerankMeta:
    base_url: str                 # API 根地址（通常以 /v1 结尾）
    model_id: str                 # 如 bge-reranker-v2-m3
    api_key: str | None = None    # 自建端点免鉴权时为空
    timeout: float = 30.0         # 单次重排请求超时（秒）
    max_input_tokens: int = 512   # 部署的单条输入（query+候选）token 上限
    max_retries: int = 3          # 超限时自动收缩预算的重试次数


# 非 ASCII（CJK/全角标点等）≈1 token/字，ASCII ≈1 token/4 字符。仅用于限长估算，
# 不做精确计量——实际密度随内容浮动（数字/符号密集的文本更高），靠下面两处兜底：
# 裁剪预算留 _BUDGET_RATIO 余量；仍超限则预算减半重试（必须减到文档长度以下，
# 裁剪才会真正发生——预算大于文档时截断是空操作，这也是要减半而非微调的原因）。
_ASCII_RE = re.compile(r"[\x00-\x7f]")
_BUDGET_RATIO = 0.8
_MIN_BUDGET = 32


def _est_tokens(text: str) -> int:
    """粗略 token 估算（偏保守），仅用于裁剪长度判断。"""
    ascii_n = len(_ASCII_RE.findall(text))
    return (len(text) - ascii_n) + ascii_n // 4


def _truncate(text: str, budget: int) -> str:
    """按估算 token 预算裁剪文本尾部；预算 <= 0 时返回空串。"""
    if budget <= 0:
        return ""
    if _est_tokens(text) <= budget:
        return text
    out, used = [], 0
    for ch in text:
        cost = 0.25 if _ASCII_RE.match(ch) else 1
        if used + cost > budget:
            break
        out.append(ch)
        used += cost
    return "".join(out)


class Reranker:
    """Cohere 兼容重排客户端：单次请求对全部候选打分并降序返回。"""

    def __init__(self, meta: RerankMeta):
        self.meta = meta

    def rerank(self, query: str, documents: list[str],
               top_n: int | None = None) -> list[tuple[int, float]]:
        """对候选文档重排，返回 [(原下标, 相关性分)]（按分数降序）。

        query 原样保留，候选按剩余预算裁剪；端点报"输入过大"时收缩预算重试。
        其余错误（连接失败/超时/其他非 2xx/响应缺字段）一律抛异常，
        由上层决定是否回退原排序。
        """
        budget = int(self.meta.max_input_tokens * _BUDGET_RATIO) - _est_tokens(query)
        url = f"{self.meta.base_url.rstrip('/')}/rerank"
        headers = {"Content-Type": "application/json"}
        if self.meta.api_key:
            headers["Authorization"] = f"Bearer {self.meta.api_key}"
        for attempt in range(self.meta.max_retries + 1):
            payload = {"model": self.meta.model_id, "query": query,
                       "documents": [_truncate(d, budget) for d in documents]}
            if top_n:
                payload["top_n"] = top_n
            resp = httpx.post(url, json=payload, headers=headers,
                              timeout=self.meta.timeout)
            if (resp.status_code == 500 and "too large" in resp.text.lower()
                    and attempt < self.meta.max_retries):
                # 服务端不自动截断：估算 token 与实际有偏差时预算减半重试
                budget = budget // 2
                if budget >= _MIN_BUDGET:
                    continue
            if resp.status_code >= 400:
                # 带出响应体：服务端把真实原因（如"input (N tokens) is too large"）
                # 放在 body 里，只报状态码会让排查多绕弯路
                raise RuntimeError(
                    f"rerank HTTP {resp.status_code}: {resp.text[:200]}")
            data = resp.json()
            out: list[tuple[int, float]] = []
            for item in data.get("results") or []:
                idx = item.get("index")
                if isinstance(idx, int):
                    out.append((idx, float(item.get("relevance_score", 0.0))))
            out.sort(key=lambda p: p[1], reverse=True)
            return out
        raise RuntimeError("unreachable")
