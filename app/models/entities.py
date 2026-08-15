from email.policy import default
import uuid
from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_collection, mapped_column, session

from app.models.base import Base, utcnow

def _uuid() -> str:
    return uuid.uuid4().hex    # 32 位十六进制字符串，作为主键

class User(Base):
    """用户： 所有资源（KB/会话/记忆/LLM配置）都归属于某个用户。"""
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(256))   # 第 2 天存哈希而非明文
    role: Mapped[str] = mapped_column(String(16), default="user")   # user | admin
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

class UserSession(Base):
    """会话：LangGraph 的 thread_id 落库，便于追踪与续聊。"""
    __tablename__ = "sessions"

    session_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_active_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

class KnowledgeBase(Base):
    """知识库：嵌入配置（provider/model/端点）创建后可随时修改；向量库实际写入模型记在 embedded_model。"""
    __tablename__ = "kbs"

    kb_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(128))
    description: Mapped[str] = mapped_column(String(512), default="")  # 知识库介绍（新建必填，LLM 选库参考）
    scope: Mapped[str] = mapped_column(String(16), default="public")   # public | private
    owner_user_id: Mapped[str | None] = mapped_column(
        ForeignKey("users.id"), nullable=True, index=True)              # 私人库属主
    category_id: Mapped[str] = mapped_column(String(64), default="default")
    embedding_provider: Mapped[str] = mapped_column(String(32))        # local | doubao | ...
    embedding_model_id: Mapped[str] = mapped_column(String(64))
    embedding_dim: Mapped[int] = mapped_column(Integer)
    embedding_base_url: Mapped[str | None] = mapped_column(
        String(256), nullable=True)          # 该库自定义嵌入端点，空则用 provider 默认
    embedding_api_key: Mapped[str | None] = mapped_column(
        String(512), nullable=True)          # 该库专用嵌入密钥（AES 加密），空则用系统默认
    embedded_model: Mapped[dict | None] = mapped_column(
        JSON, nullable=True)                 # 最近一次成功入库使用的嵌入模型标注（见 embedding_mismatch）
    retrieval_enabled: Mapped[bool] = mapped_column(Boolean, default=True)  # 允许被对话检索
    status: Mapped[str] = mapped_column(String(16), default="indexing")
    #                       ↑ ready | indexing | reembedding | failed（状态机）
    source_doc_ids: Mapped[list] = mapped_column(JSON, default=list)   # 原文档引用
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

class UserLLMConfig(Base):
    """用户自定义 LLM 配置。"""
    __tablename__ = "user_llm_config"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    provider: Mapped[str] = mapped_column(String(32))                  # openai/qwen/...
    api_key: Mapped[str] = mapped_column(String(512))                  # AES-256 加密后
    base_url: Mapped[str] = mapped_column(String(256))
    model_id: Mapped[str] = mapped_column(String(64))
    is_default: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

class ToolCallLog(Base):
    """调用追踪：记录 LLM/工具/检索每次调用，parent 成树。"""
    __tablename__ = "tool_call_log"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    kind: Mapped[str] = mapped_column(String(16))       # llm | tool | retrieve | kb
    name: Mapped[str] = mapped_column(String(64))
    session_id: Mapped[str] = mapped_column(String(36), index=True)
    user_id: Mapped[str] = mapped_column(String(36), index=True)
    args: Mapped[dict] = mapped_column(JSON, default=dict)
    output: Mapped[str] = mapped_column(Text, default="")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    parent_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

class Memory(Base):
    """长记忆：用户级命名空间，跨会话读取。"""
    __tablename__ = "memories"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    key: Mapped[str] = mapped_column(String(64))         # 记忆条目名，如 research_topic
    value: Mapped[dict] = mapped_column(JSON, default=dict)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow,
                                                 onupdate=utcnow)