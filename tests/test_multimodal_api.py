"""真实 API/图/检查点与媒体转换，模型使用离线替身。"""
import base64
import copy
import io
import json
import shutil
import subprocess
import wave

import pytest
from PIL import Image
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, AIMessageChunk

from app.api import brainstorm, seminar, deep_research
from app.main import app


def picture():
    out = io.BytesIO()
    Image.new("RGB", (12, 8), "green").save(out, "PNG")
    return out.getvalue()


class Model:
    model_name = "offline-vision"
    def __init__(self, calls):
        self.calls = calls
    def bind(self, **kwargs):
        return self
    def bind_tools(self, tools):
        return self
    async def astream(self, messages, config=None):
        self.calls.append(copy.deepcopy(messages))
        yield AIMessageChunk(content="看到绿色画面", response_metadata={"finish_reason": "stop"})
    async def ainvoke(self, messages, config=None):
        return AIMessage(content='{"memories":[]}')


@pytest.fixture
def client(monkeypatch):
    calls = []
    with TestClient(app) as c:
        service = app.state.llm_service
        monkeypatch.setattr(service, "get_chat_model", lambda *a, **kw: Model(calls))
        monkeypatch.setattr(service, "effective_config", lambda *a, **kw: None)
        monkeypatch.setattr(service, "context_window_for", lambda *a: 128000)
        yield c, calls


def upload(c, headers):
    r = c.post('/api/v1/chat/files', headers=headers,
               files={'files': ('sample.png', picture(), 'image/png')})
    assert r.status_code == 200
    meta = r.json()['files'][0]
    assert meta['kind'] == 'image' and 'assets' not in meta
    return meta['file_id']


def contents(messages):
    return [block for m in messages if isinstance(m.content, list) for block in m.content]


def test_image_reaches_model_only_when_enabled_and_checkpoint_stays_small(client, auth_factory):
    c, calls = client
    headers = auth_factory('mm-owner')
    fid = upload(c, headers)
    body = {'session_id': 'mm-chat', 'message': '描述画面', 'attachments': [fid]}
    disabled = c.post('/api/v1/chat', headers=headers, json=body)
    assert disabled.status_code == 400 and not calls
    body['multimodal'] = {'image': True}
    r = c.post('/api/v1/chat/stream', headers=headers, json=body)
    assert r.status_code == 200 and '"type": "done"' in r.text
    assert any(b.get('type') == 'image_url' for b in contents(calls[-1]))
    snap = c.portal.call(app.state.graph.aget_state, {'configurable': {'thread_id': 'mm-chat'}})
    assert 'base64' not in str(snap.values['messages'])
    assert 'media_ref' in str(snap.values['messages'])
    history = c.get('/api/v1/conversations/mm-chat/messages', headers=headers).json()
    assert history['messages'][0]['files'] == [{'name': 'sample.png', 'kind': 'image'}]
    assert 'base64' not in json.dumps(history)
    for enabled in (False, True):
        r = c.post('/api/v1/chat', headers=headers, json={
            'session_id': 'mm-chat', 'message': '继续分析', 'multimodal': {'image': enabled}})
        assert r.status_code == 200
        assert any(b.get('type') == 'image_url' for b in contents(calls[-1])) == enabled


@pytest.fixture
def real_media(tmp_path):
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('真实音视频接口测试需要 ffmpeg 和 ffprobe')
    audio = io.BytesIO()
    with wave.open(audio, 'wb') as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(b'\x00' * 8000 * 2 * 2)
    video = tmp_path / 'sample.mp4'
    subprocess.run([
        shutil.which('ffmpeg'), '-nostdin', '-v', 'error', '-filter_threads', '1',
        '-f', 'lavfi', '-i', 'color=c=blue:s=160x120:r=10:d=2', '-an',
        '-c:v', 'mpeg4', '-threads', '1', '-y', str(video),
    ], check=True, timeout=15, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return {
        'audio': ('sample.wav', audio.getvalue(), 'audio/wav'),
        'video': ('sample.mp4', video.read_bytes(), 'video/mp4'),
    }


@pytest.mark.parametrize('kind', ['audio', 'video'])
@pytest.mark.parametrize('endpoint', ['chat', 'chat/stream'])
def test_real_audio_and_video_reach_model_with_images_disabled(
        client, auth_factory, real_media, kind, endpoint):
    c, calls = client
    headers = auth_factory('mm-av-owner')
    uploaded = c.post('/api/v1/chat/files', headers=headers,
                      files={'files': real_media[kind]})
    assert uploaded.status_code == 200
    meta = uploaded.json()['files'][0]
    assert meta.get('kind') == kind, meta
    assert meta['duration_seconds'] > 0
    if kind == 'video':
        assert meta['video_visual_only'] is True and 1 <= meta['frame_count'] <= 8
    body = {
        'session_id': f"mm-{kind}-{endpoint.replace('/', '-')}",
        'message': '分析该媒体附件', 'attachments': [meta['file_id']],
        'multimodal': {'image': False, 'audio': False, 'video': False},
    }
    url = '/api/v1/' + endpoint
    disabled = c.post(url, headers=headers, json=body)
    assert disabled.status_code == 400 and not calls
    body['multimodal'][kind] = True
    result = c.post(url, headers=headers, json=body)
    assert result.status_code == 200
    if endpoint.endswith('/stream'):
        assert '"type": "done"' in result.text
        assert '"type": "error"' not in result.text
    sent = contents(calls[-1])
    if kind == 'audio':
        audio_blocks = [block['input_audio'] for block in sent if block.get('type') == 'input_audio']
        assert len(audio_blocks) == 1 and audio_blocks[0]['format'] == 'wav'
        with wave.open(io.BytesIO(base64.b64decode(audio_blocks[0]['data']))) as wav:
            assert wav.getnchannels() == 1 and wav.getframerate() == 16000
        assert not any(block.get('type') == 'image_url' for block in sent)
    else:
        frames = [block for block in sent if block.get('type') == 'image_url']
        assert len(frames) == meta['frame_count']
        assert all(frame['image_url']['url'].startswith('data:image/jpeg;base64,') for frame in frames)
        assert not any(block.get('type') == 'input_audio' for block in sent)
    config = {'configurable': {'thread_id': body['session_id']}}
    snapshot = c.portal.call(app.state.graph.aget_state, config)
    stored = contents(snapshot.values['messages'])
    refs = [block['media_ref'] for block in stored if block.get('type') == 'media_ref']
    assert refs == [{'file_id': meta['file_id'], 'filename': real_media[kind][0], 'kind': kind}]
    assert not any(block.get('type') in ('image_url', 'input_audio') for block in stored)
    assert 'base64' not in str(snapshot.values['messages'])
    history = c.get(f"/api/v1/conversations/{body['session_id']}/messages", headers=headers).json()
    assert history['messages'][0]['files'] == [{'name': real_media[kind][0], 'kind': kind}]
    assert 'base64' not in json.dumps(history)
    # 同一附件关闭后必须拒绝；不携带新附件时，关闭形式也不应重发历史媒体。
    before = len(calls)
    body['multimodal'][kind] = False
    assert c.post(url, headers=headers, json=body).status_code == 400
    assert len(calls) == before
    body.pop('attachments')
    body['message'] = '只使用文本继续回答'
    assert c.post(url, headers=headers, json=body).status_code == 200
    assert not any(block.get('type') in ('image_url', 'input_audio') for block in contents(calls[-1]))
    body['multimodal'][kind] = True
    assert c.post(url, headers=headers, json=body).status_code == 200
    assert any(block.get('type') == ('input_audio' if kind == 'audio' else 'image_url')
               for block in contents(calls[-1]))


def test_owner_boundary_and_inline_history_cannot_bypass_upload(client, auth_factory):
    c, calls = client
    alice, bob = auth_factory('mm-alice'), auth_factory('mm-bob')
    fid = upload(c, alice)
    denied = c.post('/api/v1/chat', headers=bob, json={
        'session_id': 'mm-bob-chat', 'message': 'read', 'attachments': [fid], 'multimodal': {'image': True}})
    assert denied.status_code == 404 and not calls
    assert c.post('/api/v1/chat', headers=alice, json={
        'session_id': 'mm-owned', 'message': 'hello'}).status_code == 200
    assert c.post('/api/v1/chat/stream', headers=bob, json={
        'session_id': 'mm-owned', 'message': 'steal'}).status_code == 403
    assert c.post('/api/v1/chat', headers=alice, json={'session_id': 'inline', 'message': 'hi',
        'history': [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': 'http://private'}}]}]
    }).status_code == 422
    assert c.post('/api/v1/chat', headers=alice, json={'session_id': 'inline', 'message': 'hi',
        'multimodal': {'image': 'true'}}).status_code == 422


def test_rewind_and_fork_preserve_authorized_media(client, auth_factory):
    c, calls = client
    headers = auth_factory('mm-owner')
    fid = upload(c, headers)
    assert c.post('/api/v1/chat', headers=headers, json={
        'session_id': 'mm-origin', 'message': 'describe', 'attachments': [fid],
        'multimodal': {'image': True}}).status_code == 200
    assert c.post('/api/v1/chat', headers=headers, json={
        'session_id': 'mm-origin', 'message': 'describe', 'rewind': True,
        'multimodal': {'image': True}}).status_code == 200
    assert any(b.get('type') == 'image_url' for b in contents(calls[-1]))
    assert c.post('/api/v1/chat', headers=headers, json={
        'session_id': 'mm-fork', 'message': 'branch', 'multimodal': {'image': True},
        'fork_from': {'session_id': 'mm-origin', 'user_count': 1}}).status_code == 200
    assert any(b.get('type') == 'image_url' for b in contents(calls[-1]))


@pytest.mark.parametrize('prefix,graph_name,module,archive', [
    ('brainstorm', 'brainstorm_graph', brainstorm, '_archive_proposal_to_kb'),
    ('seminar', 'seminar_graph', seminar, '_archive_report_to_kb'),
    ('deep-research', 'deep_research_graph', deep_research, '_archive_report_to_kb'),
])
def test_each_team_receives_media_and_rejects_foreign_session(client, auth_factory, monkeypatch,
                                                             prefix, graph_name, module, archive):
    c, _ = client
    headers = auth_factory('mm-team-owner')
    fid = upload(c, headers)
    captured = []
    async def invoke(state, config=None):
        captured.append(copy.deepcopy(state))
        return {**state, 'final_proposal': 'ok', 'final_report': 'ok'}
    monkeypatch.setattr(getattr(app.state, graph_name), 'ainvoke', invoke)
    monkeypatch.setattr(module, archive, lambda *args: None)
    body = {'session_id': 'mm-team', 'topic': '分析图片', 'attachments': [fid], 'multimodal': {'image': True}}
    r = c.post('/api/v1/' + prefix + '/stream', headers=headers, json=body)
    assert r.status_code == 200 and '"type": "done"' in r.text
    assert captured[0]['input_media'][0]['media_ref']['file_id'] == fid
    assert captured[0]['multimodal'] == {'image': True, 'audio': False, 'video': False}
    r = c.post('/api/v1/' + prefix + '/stream', headers=auth_factory('mm-other'),
               json={'session_id': 'mm-team', 'topic': 'steal'})
    assert r.status_code == 403 and len(captured) == 1
    assert c.post('/api/v1/chat', headers=headers,
                  json={'session_id': 'mm-team', 'message': 'wrong workflow'}).status_code == 409
