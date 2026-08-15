"""多格式解析测试：在测试里现场生成 pdf/docx 样例文件，自包含、离线。"""

import io

import pytest
from docx import Document

from app.services.kb_service import split_chunks
from app.services.parsing import parse_file
from conftest import make_pdf_pages as _make_pdf_pages


def _make_pdf(text: str) -> bytes:
    """手工构造最小合法 PDF（Helvetica Type1，仅支持 ASCII 文本）。"""
    obj1 = b"<< /Type /Catalog /Pages 2 0 R >>"
    obj2 = b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>"
    obj3 = (b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>")
    stream = b"BT /F1 24 Tf 72 720 Td (" + text.encode("ascii") + b") Tj ET"
    obj4 = b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"
    obj5 = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"

    out = bytearray(b"%PDF-1.4\n")
    offsets = {}
    for n, body in [(1, obj1), (2, obj2), (3, obj3), (4, obj4), (5, obj5)]:
        offsets[n] = len(out)
        out += f"{n} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_pos = len(out)
    out += b"xref\n0 6\n0000000000 65535 f \n"
    for n in range(1, 6):
        out += f"{offsets[n]:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF".encode()
    return bytes(out)


def _make_docx(text: str) -> bytes:
    buf = io.BytesIO()
    doc = Document()
    doc.add_paragraph(text)
    doc.save(buf)
    return buf.getvalue()


def test_parse_txt():
    assert parse_file("readme.txt", "你好 world".encode()) == "你好 world"
    assert parse_file("doc.md", "# 标题\n正文".encode()).startswith("# 标题")


def test_parse_pdf():
    content = _make_pdf("Hello Quantum Computing")
    assert "Hello Quantum Computing" in parse_file("paper.pdf", content)


def test_parse_file_pages_pdf():
    """PDF 逐页解析：页码从 1 起，每页文本独立。"""
    from app.services.parsing import parse_file_pages
    content = _make_pdf_pages(["Hello Page One", "World Page Two"])
    pages = parse_file_pages("paper.pdf", content)
    assert pages == [(1, "Hello Page One"), (2, "World Page Two")]


def test_parse_file_pages_pdf_blank_page_keeps_number():
    """空白页被过滤但保留原页码：第 1 页空 → 第 2 页内容页码为 2。"""
    from app.services.parsing import parse_file_pages
    # 第 1 页无文本（Contents 留空），第 2 页有文本
    content = _make_pdf_pages(["", "Only Second Page Text"])
    pages = parse_file_pages("paper.pdf", content)
    assert pages == [(2, "Only Second Page Text")]


def test_parse_file_pages_txt_and_docx():
    """txt/md/docx 无分页概念：单条 (None, 全文)。"""
    from app.services.parsing import parse_file_pages
    assert parse_file_pages("a.txt", "你好".encode()) == [(None, "你好")]
    docx = _make_docx("量子比特")
    assert parse_file_pages("note.docx", docx) == [(None, "量子比特")]


def test_parse_docx():
    content = _make_docx("量子比特是量子计算的基本单元")
    assert "量子比特" in parse_file("note.docx", content)


def test_parse_pdf_empty_raises():
    """空页 PDF 提取不到文本 → 明确报错（扫描件/加密 PDF 同理）。"""
    import io as _io
    from pypdf import PdfWriter
    buf = _io.BytesIO()
    w = PdfWriter()
    w.add_blank_page(width=200, height=200)
    w.write(buf)
    with pytest.raises(ValueError, match="无法从"):
        parse_file("scan.pdf", buf.getvalue())


def test_unsupported_extension():
    with pytest.raises(ValueError, match="unsupported"):
        parse_file("data.xlsx", b"xx")


def test_split_chunks_basic():
    text = "a" * 1200
    chunks = split_chunks(text, size=500, overlap=100)
    assert chunks[0] == "a" * 500
    assert len(chunks) == 3            # 1200 → 500/400/400 三块
    assert all(len(c) <= 500 for c in chunks)


def test_split_chunks_empty_and_short():
    assert split_chunks("") == []
    assert split_chunks("   ") == []
    assert split_chunks("短文本") == ["短文本"]
