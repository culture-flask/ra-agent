"""MCP 适配层：统一调用 + 追踪 + 错误回灌，上层无感协议细节。

除转发外部 MCP Server 的工具外，还支持「原生工具」——直接跑在本进程里、
可访问应用内部服务（如知识库）的工具。对 LLM 而言两者无差别：都出现在
schemas_for_llm 的工具目录里，都经 call() 统一执行/追踪。
原生工具按 user_id 隔离（每次调用注入当前用户，天然私有）。
"""

import asyncio
import json

from langchain_core.utils.function_calling import convert_to_openai_function

from app.core.tracing import Tracer
from app.mcp.host import MCPHost
from app.services.kb_service import join_document_text


# 完整原文上限：与文件上传读取的截断上限一致（防超长文献撑爆 LLM 上下文）
FULL_ARTICLE_MAX_CHARS = 100_000

# 工具观测回灌上限（P3-32）：任何工具的单次输出超过该长度即截断，
# 只保留头尾并附省略标记。背景：read_webpage 单次可注入 3 万字符，
# 多工具轮次叠加后下一轮 generate 的 prompt 可能突破模型真实窗口——
# 部分供应商对超限/上游失败的降级语义是返回**空完成**而非报错，
# 表现为前端"没有输出就结束"（详见清单 P3-31/P3-32）。截断不影响
# 追踪层（那里另有 4000 字符存储上限），也不影响需要全文的场景：
# 模型可带更大 max_chars 分次调用，或改走 get_local_document 读知识库。
TOOL_OUTPUT_MAX_CHARS = 6_000
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
                "取回知识库中某篇文件的完整原文（全部片段按原顺序拼接，"
                "并自动去掉片段间的重叠重复）。"
                "当检索只返回了文章片段、不足以回答关于该文章的问题时，"
                "应调用本工具取完整原文，而不是去网络搜索。"
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
    ]


async def _list_kb_files(args: dict, user_id: str, kb_service) -> dict:
    """实现：可检索库 → 每库的源文件名列表（同步盘读放线程池）。

    带显式 file_count——LLM 数 80+ 项长列表会数错，读数字不会。
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
    kb_id = (args.get("kb_id") or "").strip()
    file_name = (args.get("file_name") or "").strip()
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
            output = cap_observation(output)       # P3-32：回灌前统一限长
            await asyncio.to_thread(self.tracer.success, log_id, output)
            return {"output": output}
        except Exception as e:
            await asyncio.to_thread(self.tracer.error, log_id, str(e))
            return {"error": str(e), "name": name}     # 结构化错误回灌 LLM
