"""受控、按用户隔离的对话媒体存储与模型输入转换。

checkpoint 只保存 media_ref，二进制留在 data_dir 下；调用模型时才读取并
编码为 OpenAI Chat Completions 的 image_url / input_audio 内容块。
本模块不导入数据库或运行配置，便于离线测试和复用。
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import re
import shutil
import subprocess
import tempfile
import threading
import uuid
import warnings
from dataclasses import dataclass
from pathlib import Path


MEDIA_KINDS = ("image", "audio", "video")
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".svg", ".avif", ".bmp", ".tiff", ".tif"}
_AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac", ".opus"}
_VIDEO_SUFFIXES = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v"}
_FILE_ID = re.compile(r"[0-9a-f]{32}\Z")
_DEMUXERS = "mov,matroska,webm,wav,mp3,flac,ogg,aac,avi"


@dataclass(frozen=True)
class MediaLimits:
    max_bytes: int = 20 * 1024 * 1024
    max_image_pixels: int = 20_000_000
    max_image_edge: int = 2048
    max_audio_seconds: float = 120.0
    max_video_seconds: float = 120.0
    max_video_frames: int = 8
    max_video_edge: int = 1280
    max_user_files: int = 100
    max_user_bytes: int = 500 * 1024 * 1024
    process_timeout_seconds: float = 60.0


def _safe_filename(filename: str) -> str:
    name = str(filename or "upload.bin").replace("\\", "/").rsplit("/", 1)[-1]
    return "".join(c for c in name if c.isprintable())[:180] or "upload.bin"


def detect_media_kind(filename: str, content_type: str | None = None,
                      content: bytes = b"") -> str | None:
    """仅决定是否交给媒体解析器；MIME/后缀从不作为验证结果。

    即使扩展名为 txt，真实 JPEG 仍走图片验证；SVG 等候选会被解析器明确拒绝。
    音视频共用容器的最终类型由 ffprobe 的实际 stream 决定。
    """
    if (content.startswith((b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff", b"GIF87a", b"GIF89a"))
            or (content[:4] == b"RIFF" and content[8:12] == b"WEBP")):
        return "image"
    if content[:4] == b"RIFF" and content[8:12] == b"WAVE":
        return "audio"
    if content.startswith((b"ID3", b"fLaC", b"OggS")):
        return "audio"
    suffix = Path(_safe_filename(filename)).suffix.lower()
    mime = (content_type or "").split(";", 1)[0].strip().lower()
    if content.startswith(b"\x1aE\xdf\xa3") or content[4:8] == b"ftyp":
        return "audio" if suffix in _AUDIO_SUFFIXES or mime.startswith("audio/") else "video"
    if suffix in _IMAGE_SUFFIXES or mime.startswith("image/"):
        return "image"
    if suffix in _AUDIO_SUFFIXES or mime.startswith("audio/"):
        return "audio"
    if suffix in _VIDEO_SUFFIXES or mime.startswith("video/"):
        return "video"
    return None


def _reference(block: dict) -> dict:
    ref = block.get("media_ref")
    if not isinstance(ref, dict) or ref.get("kind") not in MEDIA_KINDS:
        raise ValueError("无效的媒体引用")
    return ref


def filter_content(content, enabled: dict | None = None):
    """禁用形式转可读占位文本；不读取磁盘，也不把历史媒体偷偷传给模型。"""
    if not isinstance(content, list):
        return content
    enabled = enabled if isinstance(enabled, dict) else {}
    out = []
    for block in content:
        if not isinstance(block, dict):
            out.append(block)
            continue
        if block.get("type") == "media_ref":
            ref = _reference(block)
            if enabled.get(ref["kind"]) is True:
                out.append(block)
            else:
                out.append({"type": "text", "text": f"[已关闭{_kind_label(ref['kind'])}输入：{_safe_filename(ref.get('filename', '媒体'))}]"})
        elif block.get("type") in {"image_url", "image", "input_image", "input_audio", "audio", "video_url", "video"}:
            # 只接受本服务产生的受控引用；不接收历史/客户端植入的外部 URL 或 base64。
            out.append({"type": "text", "text": "[未授权的内联媒体内容已省略]"})
        else:
            out.append(block)
    return out


def _kind_label(kind: str) -> str:
    return {"image": "图片", "audio": "音频", "video": "视频"}[kind]


class MultimodalService:
    def __init__(self, root_dir: Path | str, limits: MediaLimits | None = None):
        self.root_dir = Path(root_dir)
        self.limits = limits or MediaLimits()
        self._lock = threading.RLock()

    @staticmethod
    def availability() -> dict:
        try:
            from PIL import Image  # noqa: F401
            images = True
        except ImportError:
            images = False
        av = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
        return {"image": images, "audio": av, "video": av and images}

    def _user_dir(self, user_id: str) -> Path:
        if not isinstance(user_id, str) or not user_id:
            raise PermissionError("读取媒体需要已登录用户")
        return self.root_dir / hashlib.sha256(user_id.encode()).hexdigest()

    def _file_dir(self, user_id: str, file_id: str) -> Path:
        if not isinstance(file_id, str) or not _FILE_ID.fullmatch(file_id):
            raise ValueError("无效的媒体 file_id")
        return self._user_dir(user_id) / file_id

    def _manifest(self, user_id: str, file_id: str) -> tuple[Path, dict]:
        directory = self._file_dir(user_id, file_id)
        try:
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        except FileNotFoundError:
            # 不透露某个 ID 是不存在，还是属于其他用户。
            raise FileNotFoundError("媒体不存在、已清理或不属于当前用户") from None
        except (OSError, ValueError):
            raise ValueError("媒体元数据损坏，请重新上传") from None
        if manifest.get("file_id") != file_id or manifest.get("kind") not in MEDIA_KINDS:
            raise ValueError("媒体元数据损坏，请重新上传")
        return directory, manifest

    @staticmethod
    def _public_metadata(manifest: dict) -> dict:
        return {k: v for k, v in manifest.items() if k not in {"assets", "stored_bytes", "version"}}

    def metadata(self, user_id: str, file_id: str) -> dict:
        return self._public_metadata(self._manifest(user_id, file_id)[1])

    def read(self, user_id: str, file_id: str, asset_index: int = 0) -> tuple[bytes, str]:
        """读取规范化后的资产；视频默认返回首帧，绝不暴露本地路径。"""
        directory, manifest = self._manifest(user_id, file_id)
        assets = manifest.get("assets", [])
        if not isinstance(asset_index, int) or not 0 <= asset_index < len(assets):
            raise ValueError("无效的媒体资产序号")
        asset = assets[asset_index]
        name = asset.get("name", "")
        if not name or Path(name).name != name or name.startswith("."):
            raise ValueError("媒体资产路径无效")
        try:
            data = (directory / name).read_bytes()
        except OSError:
            raise FileNotFoundError("媒体文件已丢失，请重新上传") from None
        if len(data) > self.limits.max_user_bytes:
            raise ValueError("媒体资产超过读取限制")
        return data, asset["mime_type"]

    def reference_blocks(self, user_id: str, file_id: str) -> list[dict]:
        meta = self.metadata(user_id, file_id)
        return [{"type": "media_ref", "media_ref": {
            "file_id": meta["file_id"], "filename": meta["filename"], "kind": meta["kind"],
        }}]

    def hydrate_content(self, content, user_id: str, enabled: dict | None = None):
        """模型调用前编码；返回新列表，checkpoint 中的轻量引用保持不变。"""
        filtered = filter_content(content, enabled)
        if not isinstance(filtered, list):
            return filtered
        out = []
        for block in filtered:
            if not isinstance(block, dict) or block.get("type") != "media_ref":
                out.append(block)
                continue
            ref = _reference(block)
            _, manifest = self._manifest(user_id, ref.get("file_id"))
            # kind 来自服务端 manifest，不能通过篡改引用绕过关闭开关。
            if manifest["kind"] != ref["kind"]:
                raise ValueError("媒体引用类型与上传文件不一致")
            out.append({"type": "text", "text": f"[{_kind_label(manifest['kind'])}附件：{manifest['filename']}]"})
            for index, asset in enumerate(manifest["assets"]):
                data, mime = self.read(user_id, manifest["file_id"], index)
                encoded = base64.b64encode(data).decode("ascii")
                if manifest["kind"] == "audio":
                    out.append({"type": "input_audio", "input_audio": {"data": encoded, "format": "wav"}})
                else:
                    if manifest["kind"] == "video":
                        out.append({"type": "text", "text": f"[视频抽帧 {index + 1}/{len(manifest['assets'])}，约 {asset['timestamp_seconds']:.1f} 秒；仅画面，不含音轨]"})
                    out.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}})
        return out

    def register(self, user_id: str, filename: str, content: bytes,
                 content_type: str | None = None) -> dict:
        if not content:
            raise ValueError("空文件")
        if len(content) > self.limits.max_bytes:
            raise ValueError(f"媒体超过 {self.limits.max_bytes // (1024 * 1024)}MiB 上限")
        kind = detect_media_kind(filename, content_type, content)
        if kind is None:
            raise ValueError("不支持的媒体类型")
        user_dir = self._user_dir(user_id)
        user_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        file_id = uuid.uuid4().hex
        pending = Path(tempfile.mkdtemp(prefix=".pending-", dir=user_dir))
        try:
            if kind == "image":
                result = self._image(content, pending)
            else:
                result = self._audio_video(content, pending)
            manifest = {
                "version": 1, "file_id": file_id, "filename": _safe_filename(filename),
                "size_bytes": len(content), **result,
            }
            manifest["stored_bytes"] = sum((pending / a["name"]).stat().st_size for a in manifest["assets"])
            if manifest["stored_bytes"] > self.limits.max_user_bytes:
                raise ValueError("规范化后的媒体超过存储配额")
            with self._lock:
                existing = [p for p in user_dir.iterdir() if p.is_dir() and _FILE_ID.fullmatch(p.name)]
                if len(existing) >= self.limits.max_user_files:
                    raise ValueError(f"当前用户媒体缓存已达 {self.limits.max_user_files} 份上限，请联系管理员清理")
                used = 0
                for directory in existing:
                    for path in directory.iterdir():
                        if path.is_file():
                            used += path.stat().st_size
                if used + manifest["stored_bytes"] > self.limits.max_user_bytes:
                    raise ValueError("当前用户媒体缓存已达存储配额，请联系管理员清理")
                meta_path = pending / "manifest.json"
                meta_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
                for path in pending.iterdir():
                    path.chmod(0o600)
                pending.rename(user_dir / file_id)
            return self._public_metadata(manifest)
        finally:
            if pending.exists():
                shutil.rmtree(pending)

    def _image(self, content: bytes, directory: Path) -> dict:
        try:
            from PIL import Image, ImageOps, UnidentifiedImageError
        except ImportError:
            raise ValueError("图片输入需要 Pillow，请安装项目最新依赖") from None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(content)) as probe:
                    if probe.format not in {"JPEG", "PNG", "WEBP", "GIF"}:
                        raise ValueError("图片仅支持 JPEG、PNG、WebP、GIF；不支持 SVG")
                    if probe.width * probe.height > self.limits.max_image_pixels:
                        raise ValueError(f"图片超过 {self.limits.max_image_pixels:,} 像素上限")
                    animated = bool(getattr(probe, "is_animated", False))
                    probe.verify()
                with Image.open(io.BytesIO(content)) as image:
                    image.seek(0)
                    image = ImageOps.exif_transpose(image)
                    image.thumbnail((self.limits.max_image_edge, self.limits.max_image_edge))
                    # 总是重编码，清除文件尾/EXIF/嵌入数据；保留透明通道。
                    alpha = image.mode in {"RGBA", "LA"} or "transparency" in image.info
                    image = image.convert("RGBA" if alpha else "RGB")
                    image.info.clear()
                    name, mime = ("image.png", "image/png") if alpha else ("image.jpg", "image/jpeg")
                    image.save(directory / name, format="PNG" if alpha else "JPEG", **({} if alpha else {"quality": 88}))
                    width, height = image.size
        except ValueError:
            raise
        except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombWarning,
                Image.DecompressionBombError):
            raise ValueError("图片无效、已损坏或不受支持（仅 JPEG、PNG、WebP、GIF）") from None
        return {"kind": "image", "mime_type": mime, "width": width, "height": height,
                "first_frame_only": animated, "assets": [{"name": name, "mime_type": mime}]}

    def _run(self, command: list[str]) -> bytes:
        try:
            # 不在内存积累损坏文件反复产生的 stderr；ffprobe 元数据输出也有上限。
            with tempfile.TemporaryFile() as output:
                subprocess.run(command, stdin=subprocess.DEVNULL, stdout=output,
                               stderr=subprocess.DEVNULL, check=True,
                               timeout=self.limits.process_timeout_seconds)
                if output.tell() > 1024 * 1024:
                    raise ValueError("媒体元数据过大，请转换为标准文件后重试")
                output.seek(0)
                return output.read()
        except subprocess.TimeoutExpired:
            raise ValueError("音视频处理超时，请缩短或压缩文件") from None
        except (subprocess.CalledProcessError, OSError):
            # 不把 ffmpeg 的本地路径和不可信文件 metadata 回显给客户端。
            raise ValueError("音视频文件无效、损坏或编码不受支持") from None

    def _audio_video(self, content: bytes, directory: Path) -> dict:
        ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
        if not ffmpeg or not ffprobe:
            raise ValueError("音频/视频输入需要服务端安装 ffmpeg 和 ffprobe；图片输入可单独使用")
        source = directory / "source.bin"
        source.write_bytes(content)
        common = ["-protocol_whitelist", "file", "-format_whitelist", _DEMUXERS]
        try:
            raw = self._run([ffprobe, "-v", "error", *common, "-show_entries",
                             "format=duration,format_name:stream=index,codec_type,width,height,duration:stream_disposition=attached_pic",
                             "-of", "json", "-i", str(source)])
            try:
                probe = json.loads(raw)
            except (ValueError, TypeError):
                raise ValueError("无法识别音视频文件") from None
            streams = probe.get("streams", [])
            video = next((s for s in streams if s.get("codec_type") == "video"
                          and not s.get("disposition", {}).get("attached_pic")), None)
            audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
            if not video and not audio:
                raise ValueError("文件没有可用的音频或视频轨道")
            kind = "video" if video else "audio"
            selected = video if video else audio
            try:
                duration = float(selected.get("duration") or probe.get("format", {}).get("duration"))
            except (ValueError, TypeError):
                raise ValueError("无法确定媒体时长，请转换为标准 WAV、MP3 或 MP4 后重试") from None
            maximum = self.limits.max_video_seconds if video else self.limits.max_audio_seconds
            if not math.isfinite(duration) or duration <= 0 or duration > maximum:
                raise ValueError(f"{_kind_label(kind)}必须在 0～{maximum:g} 秒内")
            if video:
                return self._video(ffmpeg, source, directory, common, video, duration)
            name = "audio.wav"
            self._run([ffmpeg, "-nostdin", "-v", "error", "-threads", "1", *common, "-i", str(source),
                       "-map", "0:a:0", "-vn", "-t", str(maximum), "-ac", "1", "-ar", "16000",
                       "-c:a", "pcm_s16le", "-threads", "1", "-map_metadata", "-1", "-y", str(directory / name)])
            if not (directory / name).is_file() or (directory / name).stat().st_size < 44:
                raise ValueError("音频转换没有生成有效 WAV")
            return {"kind": "audio", "mime_type": "audio/wav", "duration_seconds": round(duration, 3),
                    "assets": [{"name": name, "mime_type": "audio/wav"}]}
        finally:
            source.unlink(missing_ok=True)

    def _video(self, ffmpeg: str, source: Path, directory: Path, common: list[str],
               stream: dict, duration: float) -> dict:
        width, height = int(stream.get("width") or 0), int(stream.get("height") or 0)
        if width <= 0 or height <= 0 or width * height > self.limits.max_image_pixels:
            raise ValueError("视频画面尺寸无效或超过像素上限")
        count = min(self.limits.max_video_frames, max(1, math.ceil(duration)))
        interval = duration / count
        video_mapping = f"0:{int(stream['index'])}" if "index" in stream else "0:v:0"
        scale = f"scale={self.limits.max_video_edge}:{self.limits.max_video_edge}:force_original_aspect_ratio=decrease"
        self._run([ffmpeg, "-nostdin", "-v", "error", "-threads", "1", "-filter_threads", "1", *common, "-i", str(source),
                   "-map", video_mapping, "-an", "-t", str(self.limits.max_video_seconds),
                   "-vf", f"fps={count / duration:.9f}:start_time=0:round=down,{scale}",
                   "-frames:v", str(count), "-q:v", "3", "-threads", "1", "-map_metadata", "-1", "-y",
                   str(directory / "frame-%02d.jpg")])
        frames = sorted(directory.glob("frame-*.jpg"))[:count]
        if not frames:
            raise ValueError("无法从视频提取画面，请转换为标准 MP4 后重试")
        assets = []
        for index, frame in enumerate(frames):
            # 对 ffmpeg 产出再做图片解码与尺寸检查。
            with tempfile.TemporaryDirectory(prefix=".frame-", dir=directory) as temp:
                normalized = self._image(frame.read_bytes(), Path(temp))
                shutil.copyfile(Path(temp) / normalized["assets"][0]["name"], frame)
            assets.append({"name": frame.name, "mime_type": "image/jpeg",
                           "timestamp_seconds": round(index * interval, 3)})
        return {"kind": "video", "mime_type": "video/mp4", "width": width, "height": height,
                "duration_seconds": round(duration, 3), "frame_count": len(assets),
                "video_visual_only": True, "assets": assets}
