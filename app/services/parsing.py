"""多格式文档解析：txt/md 直接读文本，pdf/docx 用轻量库提取。

选型说明：轻量起步用 pypdf / python-docx（纯 Python、零模型下载、可靠）。
生产级解析（版面分析、表格、公式）可换 Marker / Docling——只需在本模块
增加/替换解析函数，上层（入库流水线）零改动。
"""

import io
from pathlib import Path


def parse_txt(content: bytes) -> str:
    return content.decode("utf-8", errors="replace")


def parse_pdf(content: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(content))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def parse_docx(content: bytes) -> str:
    from docx import Document

    doc = Document(io.BytesIO(content))
    return "\n".join(p.text for p in doc.paragraphs)


# 扩展点：新增格式 = 在此登记一个解析函数
PARSERS = {
    ".txt": parse_txt,
    ".md": parse_txt,          # markdown 本身就是文本，直接读
    ".pdf": parse_pdf,
    ".docx": parse_docx,
}


def parse_file(filename: str, content: bytes) -> str:
    """按扩展名解析文件，返回提取出的纯文本。不支持的格式抛 ValueError。"""
    ext = Path(filename).suffix.lower()
    parser = PARSERS.get(ext)
    if parser is None:
        raise ValueError(f"unsupported file type: {ext} (支持: {', '.join(PARSERS)})")
    text = parser(content)
    if not text.strip():
        raise ValueError(f"无法从 {filename} 中提取到文本（可能是扫描件/加密 PDF）")
    return text