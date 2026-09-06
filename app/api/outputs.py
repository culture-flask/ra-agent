"""产出文件下载 API：多 agent 执笔人经 save_document / export_bibtex 落盘的
完整文档存在 <output_dir>/<用户名>/，此前只躺在服务器磁盘上——前端成稿
卡片展示的是执笔人发言（摘要），导出的 md 也只是摘要。这里补两个通道：
- GET /saved?session_id=   按会话列出成功落盘的产出文件（查 tool_call_log，
  系统回填的最终文件名在工具调用记录的 output JSON 里）
- GET /download?name=      下载本人产出目录里的文件（白名单 + 目录钳制）
"""

import json
import re
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from sqlalchemy import select

from app.core.db import SessionLocal
from app.core.deps import get_current_user
from app.models.entities import ToolCallLog, User

router = APIRouter(prefix="/api/v1/outputs", tags=["outputs"],
                   dependencies=[Depends(get_current_user)])

# 与 _write_user_file 同一白名单：中英文/数字/下划线/连字符/点（天然无路径分隔符）
_FILENAME_RE = re.compile(r"[\w\-\u4e00-\u9fff.]+")


async def _user_dir(request: Request, user_id: str) -> Path:
    """该用户的产出目录：<output_dir>/<用户名>（与 _write_user_file 同源）。"""
    from app.mcp.adapter import _user_dir_name
    name = await run_in_threadpool(_user_dir_name, user_id)
    return Path(request.app.state.settings.output_dir) / name


@router.get("/saved")
async def saved_files(request: Request, session_id: str = Query(min_length=1),
                      user: User = Depends(get_current_user)):
    """列出该会话成功落盘的产出文件（按用户过滤，跨用户查询返回空）。"""
    def _query() -> list[dict]:
        with SessionLocal() as db:
            rows = db.scalars(select(ToolCallLog).where(
                ToolCallLog.session_id == session_id,
                ToolCallLog.user_id == user.id,
                ToolCallLog.kind == "tool",
                ToolCallLog.name.in_(("save_document", "export_bibtex")))
                .order_by(ToolCallLog.started_at)).all()
        # 注意：文件名校验失败等结构化错误不走 tracer.error（error 列为空），
        # 成功与否以 output JSON 里的 saved=True + filename 为准
        files, seen = [], set()
        for r in rows:
            try:
                out = json.loads(r.output or "{}")
            except json.JSONDecodeError:
                continue
            fn = out.get("filename")
            if not out.get("saved") or not fn or fn in seen:
                continue
            seen.add(fn)
            files.append({"filename": fn, "chars": out.get("chars"),
                          "saved_at": str(r.started_at or "")})
        return files
    return await run_in_threadpool(_query)


@router.get("/all")
async def list_all(request: Request, user: User = Depends(get_current_user)):
    """列出本人产出目录的全部文件：name / size / mtime（最近修改优先）。

    前端「笔记和文件」视图用；下载仍走 /download（白名单 + 目录钳制）。"""
    user_dir = await _user_dir(request, user.id)

    def _list() -> list[dict]:
        if not user_dir.is_dir():
            return []
        files = []
        for p in user_dir.iterdir():
            if not p.is_file():
                continue
            st = p.stat()
            files.append({"name": p.name, "size": st.st_size, "mtime": st.st_mtime})
        return sorted(files, key=lambda f: f["mtime"], reverse=True)

    return await run_in_threadpool(_list)


@router.get("/download")
async def download(request: Request, name: str = Query(min_length=1),
                   user: User = Depends(get_current_user)):
    """下载本人产出目录里的文件：只能拿自己目录下的，防路径穿越。"""
    if not _FILENAME_RE.fullmatch(name) or ".." in name:
        raise HTTPException(status_code=400, detail="非法文件名")
    user_dir = await _user_dir(request, user.id)
    path = user_dir / name
    if path.resolve().parent != user_dir.resolve() or not path.is_file():
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(path, filename=name)
