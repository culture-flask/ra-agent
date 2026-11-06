"""可单独运行：pytest --confcutdir=tests/unit tests/unit/test_message_content.py。"""

from copy import deepcopy

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.core.message_content import (
    filter_media_content, hydrate_messages, media_files, text_content,
    with_text_content,
)
from app.core.tokens import estimate_tokens


def ref(kind, name=None):
    return {"type": "media_ref", "media_ref": {
        "file_id": "a" * 32, "filename": name or f"asset.{kind}", "kind": kind,
    }}


def test_text_views_never_stringify_binary_or_unknown_data():
    content = [{"type": "text", "text": "问题"},
               {"type": "image_url", "image_url": {"url": "data:image/png;base64,SECRET"}},
               {"type": "input_audio", "input_audio": {"data": "SECRET", "format": "wav"}},
               {"type": "unknown", "text": "SECRET"}, ref("video", "demo.mp4")]
    assert text_content(content) == "问题"
    safe_view = text_content(content, include_media=True)
    assert "demo.mp4" in safe_view
    assert "SECRET" not in safe_view
    assert "data:" not in safe_view
    assert text_content({"content": "SECRET"}) == ""


def test_replacing_text_preserves_independent_media_blocks():
    original = [{"type": "text", "text": "question"}, ref("image"),
                {"type": "text", "text": "old retrieval"}]
    updated = with_text_content(original, "question plus new retrieval")
    assert text_content(updated) == "question plus new retrieval"
    assert updated[1] == original[1]
    updated[1]["media_ref"]["filename"] = "changed"
    assert original[1]["media_ref"]["filename"] == "asset.image"
    assert with_text_content("old text", "new text") == "new text"


def test_switches_filter_history_without_mutating_checkpoint():
    content = [{"type": "text", "text": "check"}, ref("image"), ref("video"), ref("audio")]
    message = HumanMessage(content=content, id="kept-id", additional_kwargs={"source": "upload"})
    original = deepcopy(message)
    filtered = filter_media_content([message], {"image": True, "video": False, "audio": False})
    remaining = [block for block in filtered[0].content if block.get("type") == "media_ref"]
    assert [block["media_ref"]["kind"] for block in remaining] == ["image"]
    assert filtered[0].id == "kept-id"
    assert filtered[0].additional_kwargs == {"source": "upload"}
    assert message == original
    restored = filter_media_content([message], {"image": True, "video": True, "audio": True})
    assert len([b for b in restored[0].content if b["type"] == "media_ref"]) == 3


def test_inline_media_cannot_be_injected_through_history_even_when_enabled():
    message = HumanMessage(content=[
        {"type": "image_url", "image_url": {"url": "http://internal/private"}},
        {"type": "input_audio", "input_audio": {"data": "SECRET", "format": "wav"}},
    ])
    output = hydrate_messages([message], None, "owner", {"image": True, "audio": True})
    assert all(b["type"] == "text" for b in output[0].content)
    assert "SECRET" not in text_content(output[0].content)
    assert "internal" not in text_content(output[0].content)


def test_disabled_media_does_not_read_service_or_require_file():
    class Service:
        def hydrate_content(self, *args, **kwargs):
            raise AssertionError("disabled media must not be loaded")

    output = hydrate_messages([HumanMessage(content=[ref("image")])], Service(), "owner", {})
    assert output[0].content[0]["type"] == "text"
    assert "关闭" in text_content(output[0].content)


def test_enabled_reference_requires_service():
    with pytest.raises(ValueError, match="多模态附件服务"):
        hydrate_messages([HumanMessage(content=[ref("image")])], None, "owner", {"image": True})


def test_hydration_passes_owner_and_preserves_openai_wire_blocks():
    class Service:
        def hydrate_content(self, content, user_id, enabled):
            assert user_id == "owner-from-auth"
            assert enabled == {"image": True, "audio": True}
            return [
                {"type": "text", "text": text_content(content)},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}},
                {"type": "input_audio", "input_audio": {"data": "aGVsbG8=", "format": "wav"}},
            ]

    source = HumanMessage(content=[{"type": "text", "text": "analyze"}, ref("image"), ref("audio")])
    output = hydrate_messages([SystemMessage(content="system"), source], Service(),
                              "owner-from-auth", {"image": True, "audio": True})
    model = ChatOpenAI(api_key="offline-placeholder", model="offline", use_responses_api=False)
    payload = model._get_request_payload(output)
    wire = payload["messages"][1]["content"]
    assert wire[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert wire[2] == {"type": "input_audio", "input_audio": {"data": "aGVsbG8=", "format": "wav"}}
    assert source.content[1]["type"] == "media_ref"


def test_history_exposes_only_media_chip_metadata():
    assert media_files([ref("image", "a.png"), ref("video", "demo.mp4")]) == [
        {"name": "a.png", "kind": "image"}, {"name": "demo.mp4", "kind": "video"},
    ]


def test_media_budget_is_independent_of_base64_length():
    def message(data):
        return HumanMessage(content=[{"type": "text", "text": "中文abcd"},
                                    {"type": "image_url", "image_url": {"url": data}}])

    assert estimate_tokens([message("short")]) == estimate_tokens([message("A" * 1_000_000)])
    assert estimate_tokens([AIMessage(content="abc"), AIMessage(content="d")]) == 2
    assert estimate_tokens([HumanMessage(content=[ref("video")])]) > 0
