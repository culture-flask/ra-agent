"""共享多 agent 输入回归；不运行 tests/conftest.py，不执行数据库操作。"""

import asyncio
from copy import deepcopy
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.graph import agent_runtime


def media_ref(kind):
    return {"type": "media_ref", "media_ref": {
        "file_id": "f" * 32, "filename": f"asset.{kind}", "kind": kind,
    }}


class RecordingMedia:
    def __init__(self):
        self.calls = []

    def hydrate_content(self, content, user_id, enabled):
        self.calls.append((deepcopy(content), user_id, deepcopy(enabled)))
        output = []
        for block in content:
            if block.get("type") != "media_ref":
                output.append(deepcopy(block))
                continue
            kind = block["media_ref"]["kind"]
            if kind == "audio":
                output.append({"type": "input_audio", "input_audio": {"data": "aGVsbG8=", "format": "wav"}})
            else:
                output.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{kind}"}})
        return output


@pytest.mark.parametrize("kind", ["image", "audio", "video"])
@pytest.mark.parametrize("enabled", [False, True])
def test_input_messages_targets_last_human_and_preserves_state_and_metadata(kind, enabled):
    service = RecordingMedia()
    switches = {"image": False, "audio": False, "video": False, kind: enabled}
    state = {"user_id": "authenticated-owner", "input_media": [media_ref(kind)], "multimodal": switches}
    messages = [
        SystemMessage(content="system instruction", id="system-id"),
        HumanMessage(content="previous question", id="previous-human"),
        AIMessage(content="previous answer", id="previous-ai"),
        HumanMessage(content="current question", id="current-human", name="author",
                     additional_kwargs={"source": "upload"}, response_metadata={"request": "kept"}),
        AIMessage(content="trailing assistant", id="trailing-ai"),
    ]
    before_messages, before_state = deepcopy(messages), deepcopy(state)
    output = asyncio.run(agent_runtime.input_messages(SimpleNamespace(multimodal_service=service), state, messages))
    assert state == before_state and messages == before_messages
    for index in (0, 1, 2, 4):
        assert output[index] == messages[index]
    current = output[3]
    assert current.id == "current-human" and current.name == "author"
    assert current.additional_kwargs == {"source": "upload"}
    assert current.response_metadata == {"request": "kept"}
    assert current.content[0] == {"type": "text", "text": "current question"}
    if enabled:
        assert len(service.calls) == 1
        attached, user_id, actual_switches = service.calls[0]
        assert attached == [{"type": "text", "text": "current question"}, media_ref(kind)]
        assert user_id == "authenticated-owner" and actual_switches == switches
        assert current.content[-1]["type"] == ("input_audio" if kind == "audio" else "image_url")
    else:
        assert service.calls == []
        assert all(block["type"] == "text" for block in current.content)
        assert "关闭" in current.content[-1]["text"]


def test_input_messages_preserves_existing_blocks_and_independent_switches():
    service = RecordingMedia()
    message = HumanMessage(content=[{"type": "text", "text": "question"}, media_ref("audio")], id="kept-id")
    state = {"user_id": "owner", "input_media": [media_ref("video"), media_ref("image")],
             "multimodal": {"video": True, "audio": True, "image": False}}
    before = deepcopy((state, message))
    output = asyncio.run(agent_runtime.input_messages(SimpleNamespace(multimodal_service=service), state, [message]))
    assert (state, message) == before and output[0].id == "kept-id"
    wire = output[0].content
    assert sum(block["type"] == "input_audio" for block in wire) == 1
    assert sum(block["type"] == "image_url" for block in wire) == 1
    assert next(block for block in wire if block["type"] == "image_url")["image_url"]["url"].endswith("video")
    assert not any(block["type"] == "media_ref" for block in wire)
    refs = [b["media_ref"]["kind"] for b in service.calls[0][0] if b["type"] == "media_ref"]
    assert refs == ["audio", "video"]


def test_input_messages_without_human_does_not_attach_to_system_or_assistant():
    service = RecordingMedia()
    state = {"user_id": "owner", "input_media": [media_ref("image")], "multimodal": {"image": True}}
    messages = [SystemMessage(content="system"), AIMessage(content="assistant")]
    assert asyncio.run(agent_runtime.input_messages(SimpleNamespace(multimodal_service=service), state, messages)) == messages
    assert service.calls == []


@pytest.mark.parametrize("entrypoint", ["llm_text", "agent_speak"])
def test_control_and_speaking_calls_use_shared_media_input(monkeypatch, entrypoint):
    service, captured = RecordingMedia(), []

    async def model_round(cfg, messages, *args, **kwargs):
        captured.append(deepcopy(messages))
        return AIMessage(content="answer"), None, False

    monkeypatch.setattr(agent_runtime, "native_round", model_round)
    state = {"user_id": "owner", "session_id": "runtime-media", "input_media": [media_ref("video")],
             "multimodal": {"video": True, "image": False, "audio": False}}
    original = deepcopy(state)
    ctx = SimpleNamespace(multimodal_service=service)
    cfg = SimpleNamespace(model_id="offline")
    if entrypoint == "llm_text":
        result = asyncio.run(agent_runtime.llm_text(ctx, state, "system", "question", cfg=cfg))
    else:
        result = asyncio.run(agent_runtime.agent_speak(ctx, state, "scholar", "system", "question",
                                                     0.3, None, 1, cfg=cfg))
    assert result[0] == "answer" and state == original
    assert len(captured) == 1 and len(service.calls) == 1
    assert isinstance(captured[0][0], SystemMessage) and captured[0][0].content == "system"
    assert isinstance(captured[0][1], HumanMessage)
    assert captured[0][1].content[0] == {"type": "text", "text": "question"}
    assert captured[0][1].content[1]["type"] == "image_url"
    assert service.calls[0][1] == "owner"
