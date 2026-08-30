"""MCP 适配层：统一调用 + 追踪 + 错误回灌，上层无感协议细节。

除转发外部 MCP Server 的工具外，还支持「原生工具」——直接跑在本进程里、
可访问应用内部服务（如知识库）的工具。对 LLM 而言两者无差别：都出现在
schemas_for_llm 的工具目录里，都经 call() 统一执行/追踪。
原生工具按 user_id 隔离（每次调用注入当前用户，天然私有）。
"""

import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.utils.function_calling import convert_to_openai_function

from app.core.tracing import Tracer
from app.mcp.host import MCPHost
from app.services.kb_service import join_document_text


# 完整原文上限：与文件上传读取的截断上限一致（防超长文献撑爆 LLM 上下文）
FULL_ARTICLE_MAX_CHARS = 100_000

# 工具观测回灌上限：任何工具的单次输出超过该长度即截断，
# 只保留头尾并附省略标记。背景：read_webpage 单次可注入 3 万字符，
# 多工具轮次叠加后下一轮 generate 的 prompt 可能突破模型真实窗口——
# 部分供应商对超限/上游失败的降级语义是返回**空完成**而非报错，
# 表现为前端"没有输出就结束"。截断不影响
# 追踪层（那里另有 4000 字符存储上限），也不影响需要全文的场景：
# 模型可带更大 max_chars 分次调用，或改走 get_local_document 读知识库。
TOOL_OUTPUT_MAX_CHARS = 20_000
_TOOL_TRUNCATED_NOTE = "\n…[输出过长已截断：原始 {total} 字符，保留头尾各 {keep}；" \
    "如需更多内容请分次调用或缩小范围]"


def cap_observation(text: str, limit: int = TOOL_OUTPUT_MAX_CHARS) -> str:
    """工具观测统一限长：保头保尾（尾部常含结论/统计），中段以标记省略。

    预算制：头 80% + 尾 20%（扣除省略标记后恰好 ≤ limit），
    保证调用方拿到的长度可预期。"""
    if len(text) <= limit:
        return text
    budget = max(limit - len(_TOOL_TRUNCATED_NOTE.format(total=len(text), keep=0)) - 8,
                 limit // 2)
    head = budget * 4 // 5
    tail = budget - head
    note = _TOOL_TRUNCATED_NOTE.format(total=len(text), keep=head)
    return text[:head] + note + (text[-tail:] if tail else "")


# ---------- 原生工具定义（本进程内执行，可访问 KBService） ----------
def _native_tool_specs() -> list[dict]:
    """原生工具目录：name/description/OpenAI parameters schema + 实现函数。

    实现签名 async def(args: dict, user_id: str) -> dict——user_id 由
    call() 注入（LLM 看不到也伪造不了），保证按用户隔离。
    """
    return [
        {
            "name": "list_kb_files",
            "description": (
                "列出当前用户可检索的每个知识库中的源文件名列表。"
                "用于回答「我的知识库里有哪些文件/资料」这类问题。"
                "无需参数，自动按当前用户过滤（只含本人可见且未被禁用检索的库）。"),
            "parameters": {"type": "object", "properties": {}, "required": []},
            "func": _list_kb_files,
        },
        {
            "name": "get_local_document",
            "description": (
                "取回知识库中检索片段的完整原文。"
                "当检索只返回了文章片段、不足以回答关于该文章的问题时，"
                "应调用本工具取完整原文。"
                "file_name 必须是知识库中真实存在的文件名"
                "（不确定时先调 list_kb_files 获取文件名与所属 kb_id）。"),
            "parameters": {
                "type": "object",
                "properties": {
                    "kb_id": {"type": "string",
                              "description": "文章所在知识库的 kb_id"},
                    "file_name": {"type": "string",
                                  "description": "文章的准确文件名"},
                },
                "required": ["kb_id", "file_name"],
            },
            "func": _get_local_document,
        },
        {
            "name": "get_utc_time",
            "description": (
                "获取当前的 UTC 时间（ISO 8601 格式，附星期与 Unix 时间戳）。"
                "用于回答「现在几点/今天几号」这类问题，或需要给结论标注当前时间的场景。"
                "无需参数。"),
            "parameters": {"type": "object", "properties": {}, "required": []},
            "func": _get_utc_time,
        },
        {
            "name": "search_knowledge_base",
            "description": (
                "在当前用户的知识库中按任意检索词做混合检索（向量+BM25）。"
                "与被动等系统注入的检索结果不同，本工具让你主动用自己关心的角度"
                "查证知识库内容——头脑风暴调研/辩论时应优先用它查知识库，"
                "知识库没有的再用联网/学术检索工具。"
                "可选 kb_name 限定单个知识库；缺省搜索全部可见库。"),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string",
                              "description": "检索词（中英文均可）"},
                    "kb_name": {"type": "string",
                                "description": "可选：限定知识库名称"},
                    "k": {"type": "integer",
                          "description": "每个库返回的条数上限，默认 3，最大 10"},
                },
                "required": ["query"],
            },
            "func": _search_knowledge_base,
        },
        {
            "name": "save_document",
            "description": (
                "把一段 Markdown 内容保存为用户文件区的文档文件。"
                "适用于头脑风暴撰稿人保存：最终科研方案、分节草稿、"
                "辩论纪要、参考文献清单。文件名必须以 .md 结尾。"
                "同名文件会被覆盖——重写前请确认。"),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string",
                                 "description": "文件名，如 research-proposal.md"},
                    "content": {"type": "string",
                                "description": "Markdown 全文"},
                },
                "required": ["filename", "content"],
            },
            "func": _save_document,
        },
        {
            "name": "export_bibtex",
            "description": (
                "把参考文献列表导出为 BibTeX 文件（保存到用户文件区）。"
                "entries 每条含 title/authors/year/venue 字段；"
                "撰稿人成稿后应为本方案引用的所有工作导出引文。"),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string",
                                 "description": "输出文件名，如 refs.bib"},
                    "entries": {"type": "array", "items": {"type": "object"},
                                "description": "引文列表，每条含 "
                                "title/authors(list)/year/venue"},
                },
                "required": ["filename", "entries"],
            },
            "func": _export_bibtex,
        },
    ]


async def _list_kb_files(args: dict, user_id: str, kb_service) -> dict:
    """实现：可检索库 → 每库的源文件名列表（同步盘读放线程池）。

    带显式 file_count。
    """
    kbs = await asyncio.to_thread(kb_service.list_queryable_kbs, user_id)
    out = []
    for kb in kbs:
        docs = await asyncio.to_thread(kb_service.list_documents, kb.kb_id)
        files = [d.get("filename") or d.get("doc_id") for d in docs]
        out.append({
            "kb_id": kb.kb_id,
            "kb_name": kb.name,
            "file_count": len(files),
            "files": files,
        })
    return {"user_id": user_id, "kb_count": len(out), "kbs": out}


async def _get_local_document(args: dict, user_id: str, kb_service) -> dict:
    """实现：权限校验 → 定位文件 → 全部 chunk 去重拼接完整原文。

    超长文献截断到 FULL_ARTICLE_MAX_CHARS（带 truncated 标记），
    避免 100+ chunk 的 PDF 一次性撑爆上下文。
    """
    # LLM 偶发把 id/文件名传成数字：强制转 str（否则 .strip 直接 AttributeError）
    kb_id = str(args.get("kb_id") or "").strip()
    file_name = str(args.get("file_name") or "").strip()
    if not kb_id or not file_name:
        return {"error": "缺少必要参数 kb_id / file_name"
                        "（不确定文件名时先调 list_kb_files）"}

    # 权限：必须是当前用户可检索的库（禁检索的库取不了原文）
    kbs = await asyncio.to_thread(kb_service.list_queryable_kbs, user_id)
    kb = next((k for k in kbs if k.kb_id == kb_id), None)
    if kb is None:
        return {"error": f"知识库 {kb_id} 不存在或当前用户不可检索"}

    # 定位文件：精确匹配 → 大小写容错
    docs = await asyncio.to_thread(kb_service.list_documents, kb_id)
    doc = next((d for d in docs if d.get("filename") == file_name), None)
    if doc is None:
        doc = next((d for d in docs
                    if (d.get("filename") or "").lower() == file_name.lower()),
                   None)
    if doc is None:
        return {"error": f"知识库「{kb.name}」中不存在文件 {file_name}，"
                         f"请先调 list_kb_files 确认文件名",
                "kb_files": [d.get("filename") for d in docs]}

    chunks = await asyncio.to_thread(
        kb_service.get_document_chunks, kb_id, doc["doc_id"])
    if not chunks:
        return {"error": f"文件 {file_name} 没有可用的文本片段"}
    # 拼接时去掉相邻 chunk 的 overlap 重复段（split_chunks 固定 150 字符窗口重叠）
    full_text = await asyncio.to_thread(join_document_text, chunks)
    return {
        "kb_id": kb_id,
        "kb_name": kb.name,
        "file_name": doc.get("filename") or file_name,
        "doc_id": doc["doc_id"],
        "chunk_count": len(chunks),
        "pages": doc.get("pages") or [],
        "total_chars": len(full_text),
        "truncated": len(full_text) > FULL_ARTICLE_MAX_CHARS,
        "full_text": full_text[:FULL_ARTICLE_MAX_CHARS],
    }


async def _get_utc_time(args: dict, user_id: str, kb_service) -> dict:
    """实现：当前 UTC 时间（ISO 8601 + 星期 + Unix 时间戳）。"""
    now = datetime.now(timezone.utc)
    return {
        "utc_iso": now.isoformat(timespec="seconds"),
        "weekday_en": now.strftime("%A"),
        "unix_ts": int(now.timestamp()),
    }


async def _search_knowledge_base(args: dict, user_id: str, kb_service) -> dict:
    """实现：LLM 主动检索知识库——头脑风暴角色差异化调研的关键工具。

    与 research 节点的预取（用议题当 query）互补：这里 query 由 LLM 按
    自己的调研角度提出。逐库检索，单库失败跳过（单库隔离纪律）。
    """
    query = str(args.get("query") or "").strip()
    if not query:
        return {"error": "query 不能为空"}
    try:
        k = max(1, min(int(args.get("k") or 3), 10))
    except (TypeError, ValueError):
        k = 3
    kb_name = (args.get("kb_name") or "").strip()

    kbs = await asyncio.to_thread(kb_service.list_queryable_kbs, user_id)
    if kb_name:                              # 限定单库：精确匹配 → 大小写容错
        kbs = [kb for kb in kbs if kb.name == kb_name] \
            or [kb for kb in kbs if kb.name.lower() == kb_name.lower()]
        if not kbs:
            return {"error": f"知识库「{kb_name}」不存在或当前用户不可检索"}

    results, skipped = [], []
    for kb in kbs:
        try:
            hits = await asyncio.to_thread(
                kb_service.search, kb.kb_id, query, k=k,
                user_id=user_id, mode="hybrid")
        except Exception:                    # 单库隔离：坏库跳过
            skipped.append(kb.name)
            continue
        for h in hits:
            # source 与主链路同款回退：Chroma 命中的 source 在 metadata 里
            meta = h.get("metadata") or {}
            src = h.get("source") or meta.get("source") or "未知来源"
            if meta.get("page"):
                src += f" 第{meta['page']}页"
            results.append({
                "kb_name": kb.name,
                "source": src,
                "text": str(h.get("text", ""))[:600],   # 限长，防撑爆上下文
            })
    out: dict = {"query": query, "kb_count": len(kbs),
                 "hit_count": len(results), "results": results}
    if skipped:
        out["skipped_kbs"] = skipped         # 提示但不阻断（结构化降级）
    return out


_BS_DOCS_SUBDIR = "brainstorm"      # settings.data_dir 下的子目录


async def _save_document(args: dict, user_id: str, kb_service) -> dict:
    """实现：把 Markdown 内容写入 <data_dir>/brainstorm/<user_id>/<filename>。

    防路径穿越（文件名白名单字符集）；user_id 隔离目录——
    各用户只能写自己的目录，天然防越权。
    """
    filename = str(args.get("filename") or "").strip()
    content = args.get("content") or ""
    if not filename:
        return {"error": "filename 不能为空"}
    if not filename.endswith(".md"):
        filename += ".md"
    # 文件名白名单：字母数字-_中点；防 ../ 与系统保留名
    if not re.fullmatch(r"[\w\-\u4e00-\u9fff.]+", filename) or ".." in filename:
        return {"error": "文件名只能包含中英文/数字/下划线/连字符/点"}

    from app.settings import Settings
    data_dir = Path(Settings.load().data_dir) / _BS_DOCS_SUBDIR / user_id
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / filename
    await asyncio.to_thread(path.write_text, content, encoding="utf-8")
    return {"saved": True, "filename": filename,
            "path": str(path), "chars": len(content)}


def _bibtex_key(title: str, year) -> str:
    """生成 BibTeX 引用键：标题首词+年份（同键加序号消歧）。"""
    word = next((w for w in re.split(r"\W+", title or "cite") if w), "cite")
    return f"{word.lower()}{year or ''}"


def _render_bibtex(entries: list[dict]) -> str:
    """entries → BibTeX 字符串。字段缺失容错（authors/year/venue 可选）。"""
    seen: dict[str, int] = {}
    out = []
    for e in entries:
        title = str(e.get("title") or "untitled").strip()
        year = e.get("year") or "n.d."
        key = _bibtex_key(title, year)
        if key in seen:                      # 同键消歧：key-2 / key-3
            seen[key] += 1
            key = f"{key}-{seen[key]}"
        else:
            seen[key] = 0
        authors = " and ".join(e.get("authors") or ["unknown"]) \
            if isinstance(e.get("authors"), list) else str(e.get("authors") or "unknown")
        lines = [f"@article{{{key},",
                 f"  title = {{{title}}},",
                 f"  author = {{{authors}}},",
                 f"  year = {{{year}}},"]
        if e.get("venue"):
            lines.append(f"  journal = {{{e['venue']}}},")
        if e.get("url"):
            lines.append(f"  url = {{{e['url']}}},")
        lines.append("}")
        out.append("\n".join(lines))
    return "\n\n".join(out)


async def _export_bibtex(args: dict, user_id: str, kb_service) -> dict:
    """实现：entries → BibTeX → 写入用户文件区。"""
    filename = str(args.get("filename") or "").strip() or "references.bib"
    if not filename.endswith(".bib"):
        filename += ".bib"
    entries = args.get("entries") or []
    if not isinstance(entries, list) or not entries:
        return {"error": "entries 不能为空"}
    bib = _render_bibtex(entries[:100])       # 上限 100 条：防超长输出
    result = await _save_document({"filename": filename, "content": bib},
                                  user_id, kb_service)
    return {**result, "entry_count": len(entries[:100])}


class MCPToolAdapter:
    """把 MCP 工具包装为统一接口：schema 生成、执行、追踪、错误处理一次搞定。"""

    def __init__(self, host: MCPHost, tracer: Tracer,
                 kb_service=None):
        self.host = host
        self.tracer = tracer
        self.kb_service = kb_service
        self._native = _native_tool_specs()

    async def ensure_catalog(self) -> None:
        """启动时/首次使用前发现工具目录。"""
        if self.host.tools is None:
            await self.host.discover()

    def _native_by_name(self, name: str):
        return next((t for t in self._native if t["name"] == name), None)

    async def schemas_for_llm(self) -> list[dict]:
        """工具目录（外部 MCP + 原生）转成 LLM 能理解的 OpenAI function schema。"""
        await self.ensure_catalog()
        schemas = [convert_to_openai_function(t) for t in self.host.tools]
        schemas += [{"type": "function",
                     "function": {"name": t["name"],
                                  "description": t["description"],
                                  "parameters": t["parameters"]}}
                    for t in self._native]
        return schemas

    async def call(self, name: str, args: dict, session_id: str,
                   user_id: str, parent_id: str | None = None) -> dict:
        """执行一个工具：全链路写 ToolCallLog，错误结构化返回（促 LLM 重试）。"""
        log_id = await asyncio.to_thread(
            self.tracer.start, "tool", name, session_id, user_id, args, parent_id)
        try:
            native = self._native_by_name(name)
            if native is not None:                     # 原生工具：本进程执行
                result = await native["func"](args, user_id, self.kb_service)
            else:                                      # 外部 MCP：tools/call
                tool = next(t for t in self.host.tools if t.name == name)
                result = await tool.ainvoke(args)
            output = json.dumps(result, ensure_ascii=False, default=str) \
                if not isinstance(result, str) else result
            output = cap_observation(output)       # 回灌前统一限长
            await asyncio.to_thread(self.tracer.success, log_id, output)
            return {"output": output}
        except Exception as e:
            await asyncio.to_thread(self.tracer.error, log_id, str(e))
            return {"error": str(e), "name": name}     # 结构化错误回灌 LLM
