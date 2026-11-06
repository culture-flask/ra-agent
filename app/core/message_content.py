"""多模态消息的文本视图与模型出站视图。

checkpoint 保存 media_ref，原始二进制只在调用模型前展开。所有文本用途
只读取 text blocks，避免把 base64 或引用字典意外拼进标题、记忆和检索词。
"""

from copy import deepcopy


MEDIA_KINDS = ("image", "audio", "video")
_TEXT_TYPES = ("text", "output_text", "input_text")
_KIND_LABELS = {"image": "图片", "audio": "音频", "video": "视频"}


def media_kind(block) -> str:
    if not isinstance(block, dict):
        return ""
    kind = block.get("type")
    if kind == "media_ref":
        ref = block.get("media_ref")
        return ref.get("kind", "") if isinstance(ref, dict) else ""
    if kind in ("image", "image_url", "input_image"):
        return "image"
    if kind in ("audio", "input_audio"):
        return "audio"
    if kind in ("video", "video_url"):
        return "video"
    return ""


def media_placeholder(block, *, disabled: bool = False) -> str:
    """只暴露类型和受限文件名，绝不序列化媒体的 data/url 内容。"""
    kind = media_kind(block)
    ref = block.get("media_ref") if isinstance(block, dict) else None
    name = ref.get("filename", "") if isinstance(ref, dict) else ""
    name = name if isinstance(name, str) else ""
    name = "".join(char for char in name if char.isprintable())[:160]
    label = _KIND_LABELS.get(kind, "媒体")
    suffix = f"：{name}" if name else ""
    status = "（本轮已关闭）" if disabled else ""
    return f"[{label}附件{suffix}{status}]"


def text_content(content, *, include_media: bool = False) -> str:
    """提取文本；显式 include_media 时以安全占位表示媒体。"""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            if block.get("type") in _TEXT_TYPES:
                value = block.get("text")
                if isinstance(value, str):
                    parts.append(value)
            elif include_media and media_kind(block):
                parts.append(media_placeholder(block))
    return "".join(parts)


def with_text_content(content, text: str):
    """替换消息的合并文本，保留媒体及其他非文本 blocks。"""
    if not isinstance(content, list):
        return text
    non_text = [deepcopy(block) for block in content
                if isinstance(block, dict) and block.get("type") not in _TEXT_TYPES]
    return [{"type": "text", "text": text}, *non_text]


def filter_content(content, enabled: dict | None):
    """生成本轮出站视图：关闭模态后保留占位，不读附件文件。"""
    if not isinstance(content, list):
        return content
    allowed = enabled if isinstance(enabled, dict) else {}
    out = []
    for block in content:
        kind = media_kind(block)
        if kind and isinstance(block, dict) and block.get("type") != "media_ref":
            # 只允许服务端上传建立的引用进入模型，历史不能注入 URL/base64。
            out.append({"type": "text", "text": "[未授权的内联媒体内容已省略]"})
        elif kind and allowed.get(kind) is not True:
            out.append({"type": "text", "text": media_placeholder(block, disabled=True)})
        else:
            out.append(deepcopy(block))
    return out


def filter_media_content(messages, enabled: dict | None):
    """复制消息并按本轮开关过滤全部历史，原 checkpoint 保持可再次开启。"""
    out = []
    for message in messages:
        if isinstance(message, dict):
            result = deepcopy(message)
            result["content"] = filter_content(message.get("content", ""), enabled)
        else:
            content = filter_content(getattr(message, "content", ""), enabled)
            result = message.model_copy(deep=True, update={"content": content})
        out.append(result)
    return out


def hydrate_messages(messages, service, user_id: str, enabled: dict | None):
    """每次模型调用前展开获准的引用；服务缺省时明确报错而不静默丢资料。"""
    out = filter_media_content(messages, enabled)
    for message in out:
        content = (message.get("content", "") if isinstance(message, dict)
                   else getattr(message, "content", ""))
        refs = (isinstance(content, list)
                and any(isinstance(block, dict) and block.get("type") == "media_ref"
                        for block in content))
        if not refs:
            continue
        if service is None:
            raise ValueError("多模态附件服务未启用，无法读取附件")
        hydrated = service.hydrate_content(content, user_id, enabled=enabled or {})
        if isinstance(message, dict):
            message["content"] = hydrated
        else:
            message.content = hydrated
    return out


def media_files(content) -> list[dict]:
    """历史接口的附件 chips：只含文件元数据，不返回模型媒体内容。"""
    out = []
    if not isinstance(content, list):
        return out
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "media_ref":
            continue
        ref = block.get("media_ref")
        if not isinstance(ref, dict) or ref.get("kind") not in MEDIA_KINDS:
            continue
        name = ref.get("filename")
        if isinstance(name, str) and name:
            out.append({"name": name, "kind": ref["kind"]})
    return out
