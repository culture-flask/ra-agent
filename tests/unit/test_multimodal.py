"""离线媒体测试：使用 pytest --noconftest，不加载项目数据库 fixture。"""

import base64
import io
import json
import shutil
import subprocess
import wave
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from app.services.multimodal import (
    MediaLimits,
    MultimodalService,
    detect_media_kind,
    filter_content,
)


def image_bytes(fmt="PNG", size=(32, 24), mode="RGB"):
    output = io.BytesIO()
    Image.new(mode, size, "red").save(output, format=fmt)
    return output.getvalue()


@pytest.fixture
def service(tmp_path):
    return MultimodalService(tmp_path / "media")


def test_real_signature_wins_over_filename_and_mime(service):
    data = image_bytes()
    assert detect_media_kind("fake.txt", "text/plain", data) == "image"
    meta = service.register("alice", "../cat.txt", data, "text/plain")
    assert meta["filename"] == "cat.txt"
    assert meta["kind"] == "image"
    assert (meta["width"], meta["height"]) == (32, 24)
    assert "assets" not in meta
    stored, mime = service.read("alice", meta["file_id"])
    assert mime == "image/jpeg"
    with Image.open(io.BytesIO(stored)) as decoded:
        assert decoded.format == "JPEG"


@pytest.mark.parametrize("name,data,mime", [
    ("fake.png", b"not an image", "image/png"),
    ("x.svg", b'<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>', "image/svg+xml"),
    ("x.bmp", image_bytes("BMP"), "image/bmp"),
    ("x.jpg", b"\xff\xd8\xffgarbage", "image/jpeg"),
])
def test_invalid_and_executable_images_rejected(service, name, data, mime):
    with pytest.raises(ValueError):
        service.register("alice", name, data, mime)
    assert not list(service.root_dir.rglob(".pending-*"))
    assert not list(service.root_dir.rglob("manifest.json"))


def test_resize_and_pixel_limit(tmp_path):
    service = MultimodalService(tmp_path, replace(MediaLimits(), max_image_edge=16))
    meta = service.register("a", "x.png", image_bytes(size=(80, 40)))
    assert (meta["width"], meta["height"]) == (16, 8)
    service = MultimodalService(tmp_path / "small", replace(MediaLimits(), max_image_pixels=100))
    with pytest.raises(ValueError, match="像素"):
        service.register("a", "x.png", image_bytes(size=(11, 10)))


def test_transparency_preserved_and_metadata_stripped(service):
    output = io.BytesIO()
    exif = Image.Exif()
    exif[0x010E] = "private original EXIF data"
    Image.new("RGBA", (32, 24), "red").save(output, format="PNG", exif=exif)
    data = output.getvalue() + b"<script>malicious trailing bytes</script>"
    meta = service.register("a", "transparent.png", data)
    stored, mime = service.read("a", meta["file_id"])
    assert mime == "image/png"
    assert b"script" not in stored
    assert b"private original EXIF" not in stored
    with Image.open(io.BytesIO(stored)) as image:
        assert image.mode == "RGBA"
        assert not image.getexif()


def test_animated_image_uses_first_frame(service):
    output = io.BytesIO()
    first, second = Image.new("RGB", (16, 16), "red"), Image.new("RGB", (16, 16), "blue")
    first.save(output, format="GIF", save_all=True, append_images=[second], duration=100, loop=0)
    meta = service.register("a", "animated.gif", output.getvalue())
    assert meta["first_frame_only"] is True
    stored, _ = service.read("a", meta["file_id"])
    with Image.open(io.BytesIO(stored)) as image:
        assert getattr(image, "n_frames", 1) == 1


def test_persistent_references_and_owner_isolation(service):
    meta = service.register("alice", "cat.png", image_bytes())
    blocks = service.reference_blocks("alice", meta["file_id"])
    assert blocks == [{"type": "media_ref", "media_ref": {
        "file_id": meta["file_id"], "filename": "cat.png", "kind": "image",
    }}]
    assert "base64" not in json.dumps(blocks)
    restarted = MultimodalService(service.root_dir)
    assert restarted.metadata("alice", meta["file_id"]) == meta
    with pytest.raises(FileNotFoundError, match="不属于"):
        restarted.reference_blocks("bob", meta["file_id"])
    with pytest.raises(FileNotFoundError):
        restarted.hydrate_content(blocks, "bob", {"image": True})
    with pytest.raises(ValueError, match="file_id"):
        service.metadata("alice", "../../etc/passwd")


def test_hydrate_only_enabled_and_does_not_mutate_checkpoint(service):
    meta = service.register("a", "cat.png", image_bytes())
    original = [{"type": "text", "text": "describe"}, *service.reference_blocks("a", meta["file_id"])]
    snapshot = json.dumps(original)
    disabled = service.hydrate_content(original, "a")
    assert all(b["type"] == "text" for b in disabled)
    assert "已关闭图片" in disabled[-1]["text"]
    enabled = service.hydrate_content(original, "a", enabled={"image": True})
    assert enabled[-1]["type"] == "image_url"
    assert enabled[-1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert json.dumps(original) == snapshot
    assert service.hydrate_content("plain text", "a") == "plain text"


def test_disabled_reference_never_reads_disk(service, monkeypatch):
    ref = [{"type": "media_ref", "media_ref": {"file_id": "f" * 32, "kind": "audio", "filename": "gone.wav"}}]
    monkeypatch.setattr(service, "_manifest", lambda *args: pytest.fail("disabled media must not be read"))
    assert service.hydrate_content(ref, "a", {"audio": False})[0]["type"] == "text"
    assert service.hydrate_content(ref, "a", {"audio": "true"})[0]["type"] == "text"


def test_raw_inline_media_does_not_bypass_authorized_references():
    raw = [{"type": "image_url", "image_url": {"url": "https://intranet/secret"}},
           {"type": "input_audio", "input_audio": {"data": "AAA", "format": "wav"}}]
    assert all(b["type"] == "text" for b in filter_content(raw, {"image": True, "audio": True}))


def test_mismatched_reference_kind_rejected(service):
    meta = service.register("a", "cat.png", image_bytes())
    blocks = service.reference_blocks("a", meta["file_id"])
    blocks[0]["media_ref"]["kind"] = "video"
    with pytest.raises(ValueError, match="类型"):
        service.hydrate_content(blocks, "a", {"video": True})


def test_count_quota_and_size_limit_leave_no_pending_directory(tmp_path):
    limits = replace(MediaLimits(), max_user_files=1, max_bytes=2000)
    service = MultimodalService(tmp_path, limits)
    service.register("a", "one.png", image_bytes())
    with pytest.raises(ValueError, match="份上限"):
        service.register("a", "two.png", image_bytes())
    with pytest.raises(ValueError, match="MiB"):
        service.register("a", "big.png", b"x" * 2001)
    with pytest.raises(ValueError, match="空文件"):
        service.register("a", "empty.png", b"")
    # 每用户独立计数。
    service.register("b", "one.png", image_bytes())
    assert not list(tmp_path.rglob(".pending-*"))


def test_missing_ffmpeg_is_clear_and_images_still_work(service, monkeypatch):
    monkeypatch.setattr("app.services.multimodal.shutil.which", lambda _: None)
    with pytest.raises(ValueError, match="ffmpeg 和 ffprobe"):
        service.register("a", "x.wav", b"RIFF0000WAVE")
    assert service.register("a", "x.png", image_bytes())["kind"] == "image"
    assert service.availability() == {"image": True, "audio": False, "video": False}


def mock_audio_video(service, monkeypatch, probe):
    calls = []
    monkeypatch.setattr("app.services.multimodal.shutil.which", lambda tool: f"/usr/bin/{tool}")

    def run(command):
        calls.append(command)
        if "ffprobe" in command[0]:
            return json.dumps(probe).encode()
        target = Path(command[-1])
        if "%02d" in target.name:
            frame_count = int(command[command.index("-frames:v") + 1])
            for index in range(1, frame_count + 1):
                Path(str(target).replace("%02d", f"{index:02d}")).write_bytes(image_bytes("JPEG"))
        else:
            target.write_bytes(b"RIFF" + b"0" * 100)
        return b""

    monkeypatch.setattr(service, "_run", run)
    return calls


def test_audio_is_probed_normalized_and_hydrated(service, monkeypatch):
    calls = mock_audio_video(service, monkeypatch, {
        "format": {"duration": "3.2", "format_name": "mp3"},
        "streams": [{"codec_type": "audio"}],
    })
    meta = service.register("a", "x.mp3", b"ID3realdata")
    assert meta["duration_seconds"] == 3.2 and meta["kind"] == "audio"
    assert "-protocol_whitelist" in calls[0]
    assert calls[0][calls[0].index("-protocol_whitelist") + 1] == "file"
    assert "-format_whitelist" in calls[0]
    assert calls[1][calls[1].index("-ar") + 1] == "16000"
    blocks = service.hydrate_content(service.reference_blocks("a", meta["file_id"]), "a", {"audio": True})
    audio = blocks[-1]["input_audio"]
    assert audio["format"] == "wav"
    assert base64.b64decode(audio["data"]).startswith(b"RIFF")
    assert not list(service.root_dir.rglob("source.bin"))


def test_audio_with_cover_is_not_misclassified_as_video(service, monkeypatch):
    mock_audio_video(service, monkeypatch, {
        "format": {"duration": "3.2", "format_name": "mp3"},
        "streams": [
            {"index": 0, "codec_type": "audio"},
            {"index": 1, "codec_type": "video", "width": 100, "height": 100,
             "disposition": {"attached_pic": 1}},
        ],
    })
    meta = service.register("a", "cover.mp3", b"ID3realdata")
    assert meta["kind"] == "audio"


def test_audio_container_candidate_respects_audio_mime():
    assert detect_media_kind("song.m4a", "audio/mp4", b"0000ftypM4A ") == "audio"
    assert detect_media_kind("song.webm", "audio/webm", b"\x1aE\xdf\xa3") == "audio"


@pytest.mark.parametrize("duration", ["121", "nan", "inf", "0", "-1", "broken"])
def test_audio_duration_rejected_before_conversion(service, monkeypatch, duration):
    calls = mock_audio_video(service, monkeypatch, {
        "format": {"duration": duration}, "streams": [{"codec_type": "audio"}],
    })
    with pytest.raises(ValueError):
        service.register("a", "x.wav", b"RIFF0000WAVE")
    assert len(calls) == 1
    assert not list(service.root_dir.rglob("source.bin"))
    assert not list(service.root_dir.rglob(".pending-*"))


def test_video_bounded_frames_and_independent_video_switch(service, monkeypatch):
    calls = mock_audio_video(service, monkeypatch, {
        "format": {"duration": "100", "format_name": "mov,mp4,m4a,3gp,3g2,mj2"},
        "streams": [{"codec_type": "video", "width": 1920, "height": 1080}],
    })
    meta = service.register("a", "x.mp4", b"0000ftypisom")
    assert meta["frame_count"] == 8 and meta["video_visual_only"] is True
    assert calls[1][calls[1].index("-frames:v") + 1] == "8"
    blocks = service.hydrate_content(service.reference_blocks("a", meta["file_id"]), "a",
                                    {"video": True, "image": False})
    assert sum(b["type"] == "image_url" for b in blocks) == 8
    assert "不含音轨" in blocks[1]["text"]
    assert not list(service.root_dir.rglob("source.bin"))


def test_video_maps_actual_stream_after_attached_cover(service, monkeypatch):
    calls = mock_audio_video(service, monkeypatch, {
        "format": {"duration": "2"},
        "streams": [
            {"index": 0, "codec_type": "video", "width": 100, "height": 100,
             "disposition": {"attached_pic": 1}},
            {"index": 2, "codec_type": "video", "width": 320, "height": 240,
             "disposition": {"attached_pic": 0}},
        ],
    })
    service.register("a", "x.mp4", b"0000ftypisom")
    assert calls[1][calls[1].index("-map") + 1] == "0:2"


def test_process_failure_does_not_leak_paths(service, monkeypatch):
    def failed(*args, **kwargs):
        raise subprocess.CalledProcessError(1, ["ffmpeg"], stderr=b"private/path API_SECRET")
    monkeypatch.setattr("app.services.multimodal.subprocess.run", failed)
    with pytest.raises(ValueError, match="无效、损坏") as caught:
        service._run(["ffmpeg"])
    assert "private" not in str(caught.value)


def test_process_timeout_is_bounded(service, monkeypatch):
    def timed_out(*args, **kwargs):
        assert kwargs["timeout"] == 60
        assert "shell" not in kwargs
        raise subprocess.TimeoutExpired("ffmpeg", 60)
    monkeypatch.setattr("app.services.multimodal.subprocess.run", timed_out)
    with pytest.raises(ValueError, match="超时"):
        service._run(["ffmpeg"])


_HAS_AV_TOOLS = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


@pytest.mark.skipif(not _HAS_AV_TOOLS, reason="需 ffmpeg/ffprobe 的真实转换检查")
def test_real_wav_roundtrip(service):
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(2)
        audio.setsampwidth(2)
        audio.setframerate(8000)
        audio.writeframes(b"\x00" * 8000 * 2 * 2)
    meta = service.register("a", "stereo.wav", output.getvalue(), "audio/wav")
    stored, mime = service.read("a", meta["file_id"])
    assert mime == "audio/wav" and meta["duration_seconds"] == 1
    with wave.open(io.BytesIO(stored)) as audio:
        assert audio.getnchannels() == 1
        assert audio.getframerate() == 16000
        assert audio.getnframes() == 16000


@pytest.mark.skipif(not _HAS_AV_TOOLS, reason="需 ffmpeg/ffprobe 的真实转换检查")
def test_real_video_extracts_bounded_frames(service, tmp_path):
    source = tmp_path / "synthetic.mp4"
    subprocess.run([shutil.which("ffmpeg"), "-nostdin", "-v", "error", "-filter_threads", "1",
                    "-f", "lavfi", "-i", "color=c=blue:s=160x120:r=10:d=2", "-an",
                    "-c:v", "mpeg4", "-threads", "1", "-y", str(source)],
                   check=True, timeout=15, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    meta = service.register("a", "synthetic.mp4", source.read_bytes(), "video/mp4")
    assert meta["kind"] == "video" and 1 <= meta["frame_count"] <= 8
    blocks = service.hydrate_content(service.reference_blocks("a", meta["file_id"]), "a", {"video": True})
    assert sum(b["type"] == "image_url" for b in blocks) == meta["frame_count"]
    assert meta["video_visual_only"] is True


@pytest.mark.skipif(not _HAS_AV_TOOLS, reason="需 ffmpeg/ffprobe 的真实转换检查")
def test_real_mp3_cover_remains_audio(service, tmp_path):
    cover = tmp_path / "cover.png"
    cover.write_bytes(image_bytes())
    source = tmp_path / "covered.mp3"
    subprocess.run([shutil.which("ffmpeg"), "-nostdin", "-v", "error", "-filter_threads", "1",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-i", str(cover),
                    "-map", "0:a:0", "-map", "1:v:0", "-c:a", "libmp3lame", "-c:v", "copy",
                    "-threads", "1", "-id3v2_version", "3", "-disposition:v:0", "attached_pic",
                    "-y", str(source)], check=True, timeout=15,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    meta = service.register("a", "covered.mp3", source.read_bytes(), "audio/mpeg")
    assert meta["kind"] == "audio"
    blocks = service.hydrate_content(service.reference_blocks("a", meta["file_id"]), "a", {"audio": True})
    assert blocks[-1]["type"] == "input_audio"
    assert not any(b["type"] == "image_url" for b in blocks)
