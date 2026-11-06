"""Token 粗估工具：跨层共用的纯函数。

此前 `_estimate_tokens` 定义在编排层（graph/nodes.py），而 API 层
（api/chat.py 的 /chat/context）也需要同一估算——反向 import 编排层
私有符号是分层倒置。现挪到 core 作为公共工具，nodes 与 api 同源引用。

启发式：中英文混合约 2 字符/token，**宁可高估早压缩也不爆窗**。
精确用量另有来源——generate 节点从 LLM 响应提取真实 usage
（stream_usage=True 时流式末块携带），估算只在新会话/假模型时兜底。
媒体引用采用固定预算估计，避免把 base64 字符误算成文字 token；真实视觉/
音频用量由模型、分辨率、时长决定，固定预算不能作为精确计费数据。
"""

from app.core.message_content import media_kind, text_content


def estimate_tokens(messages: list) -> int:
    """文字约 2 字符/token，媒体用固定保守预算，不计 base64 长度。"""
    chars = 0
    media_tokens = 0
    media_budget = {"image": 1024, "audio": 4096, "video": 8192}
    for message in messages:
        content = (message.get("content", "") if isinstance(message, dict)
                   else getattr(message, "content", ""))
        chars += len(text_content(content))
        if isinstance(content, list):
            media_tokens += sum(media_budget.get(media_kind(block), 0) for block in content)
    return chars // 2 + media_tokens
