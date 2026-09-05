# 将yaml/环境变量统一映射成带类型的Python对象

import yaml
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    # 从 ./env读配置，未知字段忽略
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "ra-agent"
    host: str = "0.0.0.0"
    port: int = 8000
    data_dir: Path = Path("./data")
    output_dir: Path = Path("./output")   # 多 agent 产出文件（save_document/export_bibtex）
    database_url: str = "postgresql+psycopg://ra:ra@localhost:5432/ra_agent"
    redis_url: str = "redis://localhost:6379/0"
    vector_store_provider: str = "chroma"
    chroma_persist_dir: Path = Path("./data/chroma")
    max_logs: int = 1000
    jwt_secret: str = "dev-secret-change-me-to-32+bytes!!"
    llm_system_default: dict = Field(default_factory=dict)
    llm_api_key: str = ""
    llm_retry_max_retries: int = 10          # LLM 调用失败重试次数（指数退避）
    llm_retry_base_delay: float = 1.0       # 首次重试基础等待秒数，之后 2 倍递增
    llm_context_window: int = 256000        # 上下文窗口兜底值（token）：探测与用户设置均缺失时使用，80% 触发自动压缩
    retrieval_mode: str = "hybrid"          # 检索模式：vector（纯向量）| hybrid（向量+BM25）
    retrieval_per_kb_k: int = 3             # 对话 RAG：每个知识库检索几条
    retrieval_total_k: int = 5              # 对话 RAG：所有库合并后总共取几条
    retrieval_parent_groups: int = 3        # 聚合返回：展开的父块数（0=关闭，全返回小 chunk）
    retrieval_parent_group_size: int = 3    # 每个父块聚合的相邻 chunk 数
    retrieval_parent_max_chars: int = 4000  # 单个父块拼接文本上限（字符）
    embedding_default_provider: str = "doubao"
    embedding_default_model: str = "doubao-embedding-vision"
    embedding_cloud: dict = Field(default_factory=dict)
    embedding_local_default_dim: int = 384
    embedding_api_key: str = ""
    http_proxy: str = ""
    https_proxy: str = ""
    no_proxy: str = "localhost,127.0.0.1"
    mcp_servers: dict = Field(default_factory=dict)
    llm_providers: dict = Field(default_factory=dict)
    memory_max: int = 50                      # 每用户长期记忆条数上限（超限触发压缩/LRU）
    memory_short_ttl_days: int = 14           # short 层记忆过期天数（未更新自动清除）

    # ---------- 头脑风暴（多 agent 辩论） ----------
    brainstorm_max_rounds: int = 3
    brainstorm_research_tool_loop_max: int = 10
    brainstorm_debate_tool_rounds: int = 1
    brainstorm_token_budget: int = 20000000   # 前端发起时按场可调
    brainstorm_position_max_chars: int = 500
    brainstorm_transcript_window: int = 6
    brainstorm_roles: list = Field(default_factory=lambda: [
        {"id": "innovator", "name": "创新者", "temperature": 0.9},
        {"id": "critic", "name": "批评者", "temperature": 0.4},
        {"id": "methodologist", "name": "方法论专家", "temperature": 0.3},
        {"id": "practitioner", "name": "实践者", "temperature": 0.5},
    ])

    # ---------- 格致会讲（学术研讨式多 agent） ----------
    seminar_reading_tool_loop_max: int = 6
    seminar_notes_max_chars: int = 600
    seminar_insight_window: int = 20
    seminar_cards_per_scholar: int = 2
    seminar_token_budget: int = 20000000
    seminar_roles: list = Field(default_factory=lambda: [
        {"id": "historian", "name": "文献学家", "temperature": 0.3},
        {"id": "theorist", "name": "理论家", "temperature": 0.6},
        {"id": "experimentalist", "name": "实验家", "temperature": 0.4},
        {"id": "visitor", "name": "访问学者", "temperature": 0.9},
    ])

    # ---------- 溯源社（深度调研式多 agent） ----------
    deep_research_scout_tool_loop_max: int = 8      # 初调阶段工具子循环上限
    deep_research_min_sections: int = 3             # 章节数下限（快速模式固定 3）
    deep_research_max_sections: int = 5             # 章节数上限
    deep_research_chapter_min_words: int = 800      # 单章草稿字数下限
    deep_research_chapter_max_words: int = 1500     # 单章草稿字数上限
    deep_research_tool_loop_max: int = 10           # 分章深研工具子循环上限
    deep_research_revise_tool_loop_max: int = 5     # 修订阶段工具子循环上限（低于深研：只补缺口）
    deep_research_token_budget: int = 20000000      # 单场总 token 预算（硬熔断；publish 不受阻断）
    deep_research_roles: list = Field(default_factory=lambda: [
        {"id": "researcher",  "name": "课题研究员", "temperature": 0.4},
        {"id": "planner",     "name": "研究编辑",   "temperature": 0.3},
        {"id": "reviewer",    "name": "审稿人",     "temperature": 0.2},
        {"id": "reviser",     "name": "修订员",     "temperature": 0.3},
        {"id": "writer",      "name": "报告撰写人", "temperature": 0.4},
    ])

    @classmethod
    def load(cls) -> "Settings":
        """加载配置。优先级： 环境变量/.env > settings.ymal > 类默认值

        命名约定：yaml按主题嵌套、代码字段扁平化，靠下面这个映射表桥接。
        规则——新增配置项必须三处同步：yaml键、Settings字段、映射表。
        且字段名与yaml的”叶子键”要完全一致。
        """
        merged = cls()                   # 先让 pydantic-settings 应用环境变量与 .env
        yaml_path = BASE_DIR / "config" / "settings.yaml"
        if yaml_path.exists():
            raw = yaml.safe_load(yaml_path.read_text())
            yaml_values = {
                "app_name" : raw["app"]["name"],
                "host" : raw["app"]["host"],
                "port" : raw["app"]["port"],
                "data_dir" : raw["app"]["data_dir"],
                "output_dir" : raw.get("app", {}).get("output_dir", "./output"),
                "database_url" : raw["database"]["url"],
                "redis_url": raw["redis"]["url"],
                "vector_store_provider": raw["vector_store"]["provider"],
                "chroma_persist_dir": raw["vector_store"]["chroma"]["persist_dir"],
                "max_logs": raw["tracing"]["max_logs"],
                "llm_system_default": raw["llm"]["system_default"],
                "llm_retry_max_retries": raw.get("llm", {}).get("retry", {}).get("max_retries", 3),
                "llm_retry_base_delay": raw.get("llm", {}).get("retry", {}).get("base_delay", 1.0),
                "llm_context_window": raw.get("llm", {}).get("context_window", 256000),
                "retrieval_mode": raw.get("retrieval", {}).get("mode", "hybrid"),
                "retrieval_per_kb_k": raw.get("retrieval", {}).get("per_kb_k", 3),
                "retrieval_total_k": raw.get("retrieval", {}).get("total_k", 5),
                "retrieval_parent_groups": raw.get("retrieval", {}).get("parent_groups", 3),
                "retrieval_parent_group_size": raw.get("retrieval", {}).get("parent_group_size", 3),
                "retrieval_parent_max_chars": raw.get("retrieval", {}).get("parent_max_chars", 4000),
                "embedding_default_provider": raw["embedding"]["default_provider"],
                "embedding_default_model": raw["embedding"]["default_model"],
                "embedding_cloud": raw["embedding"]["cloud"],
                "embedding_local_default_dim": raw["embedding"]["local"]["default_dim"],
                "mcp_servers": raw.get("mcp_servers", {}),
                "llm_providers": raw.get("llm", {}).get("providers", {}),   # 容错:缺段用空目录
                "memory_max": raw.get("memory", {}).get("max", 50),
                "memory_short_ttl_days": raw.get("memory", {}).get("short_ttl_days", 14),
                "brainstorm_max_rounds": raw.get("brainstorm", {}).get("max_rounds", 3),
                "brainstorm_research_tool_loop_max": raw.get("brainstorm", {}).get("research_tool_loop_max", 6),
                "brainstorm_debate_tool_rounds": raw.get("brainstorm", {}).get("debate_tool_rounds", 1),
                "brainstorm_token_budget": raw.get("brainstorm", {}).get("token_budget", 800000),
                "brainstorm_position_max_chars": raw.get("brainstorm", {}).get("position_max_chars", 500),
                "brainstorm_transcript_window": raw.get("brainstorm", {}).get("transcript_window", 6),
                "brainstorm_roles": raw.get("brainstorm", {}).get("roles", []),
                "seminar_reading_tool_loop_max": raw.get("seminar", {}).get("reading_tool_loop_max", 6),
                "seminar_notes_max_chars": raw.get("seminar", {}).get("notes_max_chars", 600),
                "seminar_insight_window": raw.get("seminar", {}).get("insight_window", 20),
                "seminar_cards_per_scholar": raw.get("seminar", {}).get("cards_per_scholar", 2),
                "seminar_token_budget": raw.get("seminar", {}).get("token_budget", 20000000),
                "seminar_roles": raw.get("seminar", {}).get("roles", []),
                "deep_research_scout_tool_loop_max": raw.get("deep_research", {}).get("scout_tool_loop_max", 8),
                "deep_research_min_sections": raw.get("deep_research", {}).get("min_sections", 3),
                "deep_research_max_sections": raw.get("deep_research", {}).get("max_sections", 5),
                "deep_research_chapter_min_words": raw.get("deep_research", {}).get("chapter_min_words", 800),
                "deep_research_chapter_max_words": raw.get("deep_research", {}).get("chapter_max_words", 1500),
                "deep_research_tool_loop_max": raw.get("deep_research", {}).get("tool_loop_max", 10),
                "deep_research_revise_tool_loop_max": raw.get("deep_research", {}).get("revise_tool_loop_max", 5),
                "deep_research_token_budget": raw.get("deep_research", {}).get("token_budget", 20000000),
                "deep_research_roles": raw.get("deep_research", {}).get("roles", []),
            }
            for key, value in yaml_values.items():
                if key not in merged.model_fields_set:
                    setattr(merged, key, value)        # 环境已设置的字段，yaml不再覆盖
        return merged

settings = Settings.load()