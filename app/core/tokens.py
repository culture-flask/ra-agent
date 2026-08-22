"""Token 粗估工具：跨层共用的纯函数（P2-17 分层修正）。

此前 `_estimate_tokens` 定义在编排层（graph/nodes.py），而 API 层
（api/chat.py 的 /chat/context）也需要同一估算——反向 import 编排层
私有符号是分层倒置。现挪到 core 作为公共工具，nodes 与 api 同源引用。

启发式：中英文混合约 2 字符/token，**宁可高估早压缩也不爆窗**。
精确用量另有来源——generate 节点从 LLM 响应提取真实 usage
（stream_usage=True 时流式末块携带），估算只在新会话/假模型时兜底。
"""


def estimate_tokens(messages: list) -> int:
    """按字符数折算消息列表的粗估 token 数。"""
    chars = sum(len(str(getattr(m, "content", ""))) for m in messages)
    return chars // 2
