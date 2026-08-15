"""测试公共设施：保证每个测试互不影响、可重复运行。

问题：注册类测试（test_auth.py）共用同一个数据库，第一次跑注册了 alice，
第二次再跑就 409（用户已存在）。解法：每个测试前清空相关表。
"""

import os
import tempfile


def make_pdf_pages(texts: list[str]) -> bytes:
    """手工构造多页最小 PDF（Helvetica Type1，仅支持 ASCII）——测试共用。"""
    objs: dict[int, bytes] = {}
    page_objs, n = [], 6
    for text in texts:
        page_no, content_no = n, n + 1
        # ASCII 用字面量字符串；含中文等非 ASCII 用 UTF-16BE hex（带 BOM，
        # pypdf 可解码提取；无需嵌入字体——测试只关心提取出的文本）
        if text.isascii():
            stream = b"BT /F1 24 Tf 72 720 Td (" + text.encode("ascii") + b") Tj ET"
        else:
            stream = (b"BT /F1 24 Tf 72 720 Td <FEFF"
                      + text.encode("utf-16-be").hex().encode() + b"> Tj ET")
        objs[page_no] = (b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                         b"/Contents " + str(content_no).encode() + b" 0 R "
                         b"/Resources << /Font << /F1 5 0 R >> >> >>")
        objs[content_no] = (b"<< /Length " + str(len(stream)).encode()
                            + b" >>\nstream\n" + stream + b"\nendstream")
        page_objs.append(f"{page_no} 0 R".encode())
        n += 2
    objs[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objs[2] = (b"<< /Type /Pages /Kids [" + b" ".join(page_objs)
               + b"] /Count " + str(len(texts)).encode() + b" >>")
    objs[5] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    out = bytearray(b"%PDF-1.4\n")
    offsets = {}
    for no in sorted(objs):
        offsets[no] = len(out)
        out += f"{no} 0 obj\n".encode() + objs[no] + b"\nendobj\n"
    xref_pos = len(out)
    out += b"xref\n0 " + str(max(offsets) + 1).encode() + b"\n0000000000 65535 f \n"
    for no in range(1, max(offsets) + 1):
        off = offsets.get(no)
        if off is not None:
            out += f"{off:010d} 00000 n \n".encode()
        else:
            out += b"0000000000 65535 f \n"
    out += (f"trailer\n<< /Size {max(offsets) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_pos}\n%%EOF").encode()
    return bytes(out)

# 关键：测试必须离线、确定性——强制本地嵌入 + 临时数据目录。
# 必须在 import app.main 之前设置（Settings.load() 在 lifespan 里读环境变量）。
# 数据库也必须指向专用测试库，绝不能清空开发库（否则跑一次测试就删一次真实知识库）：
#   docker exec ra-postgres psql -U ra -d ra_agent -c "CREATE DATABASE ra_agent_test OWNER ra"
#   DATABASE_URL=postgresql+psycopg://ra:ra@localhost:5432/ra_agent_test .venv/bin/alembic upgrade head
os.environ["DATABASE_URL"] = "postgresql+psycopg://ra:ra@localhost:5432/ra_agent_test"
os.environ["EMBEDDING_DEFAULT_PROVIDER"] = "local"
os.environ["DATA_DIR"] = tempfile.mkdtemp(prefix="ra-test-data-")
os.environ["CHROMA_DIR"] = tempfile.mkdtemp(prefix="ra-test-chroma-")

import pytest
from sqlalchemy import text

from app.core.db import engine


@pytest.fixture(autouse=True)
def clean_db():
    """每个测试前清空 users/kbs 表，保证注册/建库类测试可重复运行。

    autouse=True：所有测试自动使用，无需在测试函数里显式声明。
    这是"测试可重复"的最简单方案；更完整的"每个测试事务回滚"技术在第 7 天讲。
    """
    with engine.begin() as conn:          # 事务：DELETE 执行后自动提交
        conn.execute(text("DELETE FROM user_llm_config"))
        conn.execute(text("DELETE FROM memories"))  # 子表链按外键方向删
        conn.execute(text("DELETE FROM kbs"))       # 再删子表(有外键指向 users)
        conn.execute(text("DELETE FROM users"))
        # LangGraph 检查点：不清理会让上一轮测试的 retrievals 等状态
        # 按 thread_id 泄漏进下一轮测试（会话状态必须每轮重置）
        conn.execute(text("TRUNCATE checkpoint_writes, checkpoint_blobs, checkpoints CASCADE"))
    yield
