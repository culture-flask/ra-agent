"""对话附件：/chat/files 上传解析 + 附件文本拼进用户消息。"""

from fastapi.testclient import TestClient

from app.api.chat import _CHAT_FILES, ChatRequest, _initial_state
from app.main import app


def test_upload_files_mixed():
    """批量上传：合法 txt 解析成功，空文件/不支持的格式逐个报错，互不影响。"""
    with TestClient(app) as c:
        r = c.post("/api/v1/chat/files", files=[
            ("files", ("note.txt", "附件内容：神经区分器训练技巧".encode(), "text/plain")),
            ("files", ("bad.xyz", b"\x00\x01", "application/octet-stream")),
            ("files", ("empty.txt", b"", "text/plain")),
        ])
        assert r.status_code == 200
        files = r.json()["files"]
        assert files[0]["file_id"] and files[0]["filename"] == "note.txt"
        assert files[0]["chars"] > 0
        assert "error" in files[1]                     # 不支持的格式
        assert "error" in files[2]                     # 空文件
        assert files[0]["file_id"] in _CHAT_FILES      # 暂存成功


def test_initial_state_merges_attachments():
    """附件全文拼进本轮用户消息（进 checkpoint），query 保持原始问题。"""
    _CHAT_FILES["f1"] = {"filename": "note.txt", "text": "文件正文ABC"}
    req = ChatRequest(user_id="u1", session_id="s", message="总结一下",
                      attachments=["f1", "ghost"])     # ghost：未知 id 忽略
    state = _initial_state(req)
    msg = state["messages"][-1]
    assert msg["content"].startswith("总结一下")        # 问题在前（标题提取用）
    assert "[附件：note.txt]" in msg["content"] and "文件正文ABC" in msg["content"]
    assert state["query"] == "总结一下"                 # 检索嵌入用原始问题


def test_rewind_does_not_append_or_merge():
    """rewind 场景：不追加消息也不拼附件（checkpoint 里已含上次的完整文本）。"""
    _CHAT_FILES["f2"] = {"filename": "a.md", "text": "X"}
    req = ChatRequest(user_id="u1", session_id="s", message="重答", attachments=["f2"])
    state = _initial_state(req, append_message=False)
    assert state["messages"] == []
