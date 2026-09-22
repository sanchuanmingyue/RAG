"""项目配置：集中读取 .env 和常用路径。"""

from pathlib import Path
import os
import re
import sys

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parent.parent
PAPER_DIR = ROOT_DIR / "data" / "papers"
VECTOR_DB_DIR = ROOT_DIR / "storage" / "vector_db"
RAG_ANYTHING_DIR = ROOT_DIR / "storage" / "rag_anything"
RAG_ANYTHING_OUTPUT_DIR = ROOT_DIR / "storage" / "rag_anything_output"
CONVERSATION_DB_PATH = ROOT_DIR / "storage" / "conversations.db"

load_dotenv(ROOT_DIR / ".env")


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_list(name: str, fallback: str = "") -> list[str]:
    """Read a comma-separated environment variable, preserving order."""

    raw_value = os.getenv(name)
    value = fallback if raw_value is None else raw_value
    items: list[str] = []
    for item in re.split(r"[,，;；\r\n]+", value):
        normalized = item.strip()
        if normalized and normalized not in items:
            items.append(normalized)
    return items


class Settings:
    """从环境变量读取模型和检索配置。"""

    def __init__(self) -> None:
        self.llm_api_key = os.getenv("LLM_API_KEY", "").strip()
        self.llm_base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").strip()
        legacy_llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini").strip()
        self.llm_models = _env_list("LLM_MODELS", legacy_llm_model)
        self.llm_model = self.llm_models[0] if self.llm_models else legacy_llm_model
        # Qwen 3.x models on DashScope may otherwise enter thinking mode, which
        # materially increases latency for straightforward RAG answers.
        self.llm_enable_thinking = _env_bool("LLM_ENABLE_THINKING", False)
        self.llm_failover_cooldown_seconds = float(os.getenv("LLM_FAILOVER_COOLDOWN_SECONDS", "60"))
        self.llm_quota_cooldown_seconds = float(os.getenv("LLM_QUOTA_COOLDOWN_SECONDS", "86400"))
        self.embedding_base_url = (
            os.getenv("EMBEDDING_BASE_URL") or self.llm_base_url
        ).strip()
        configured_embedding_key = (
            os.getenv("EMBEDDING_API_KEY") or os.getenv("SILICONFLOW_API_KEY") or ""
        ).strip()
        self.embedding_api_key = configured_embedding_key or (
            self.llm_api_key if self.embedding_base_url.rstrip("/") == self.llm_base_url.rstrip("/") else ""
        )
        if (
            not self.llm_api_key
            and self.embedding_api_key
            and self.embedding_base_url.rstrip("/") == self.llm_base_url.rstrip("/")
        ):
            self.llm_api_key = self.embedding_api_key
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small").strip()
        self.embedding_batch_size = max(int(os.getenv("EMBEDDING_BATCH_SIZE", "10")), 1)
        self.embedding_max_retries = max(int(os.getenv("EMBEDDING_MAX_RETRIES", "3")), 0)
        self.embedding_retry_base_seconds = max(
            float(os.getenv("EMBEDDING_RETRY_BASE_SECONDS", "1")), 0.0
        )
        self.vision_base_url = (os.getenv("VISION_BASE_URL") or self.llm_base_url).strip()
        configured_vision_key = (os.getenv("VISION_API_KEY") or "").strip()
        if self.vision_base_url.rstrip("/") == self.embedding_base_url.rstrip("/"):
            self.vision_api_key = self.embedding_api_key
        elif configured_vision_key:
            self.vision_api_key = configured_vision_key
        elif self.vision_base_url.rstrip("/") == self.llm_base_url.rstrip("/"):
            self.vision_api_key = self.llm_api_key
        else:
            self.vision_api_key = ""
        self.chroma_collection = os.getenv("CHROMA_COLLECTION", "paper_reader_chunks").strip()
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "800"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "120"))
        self.retrieval_top_k = int(os.getenv("RETRIEVAL_TOP_K", "5"))
        self.multi_paper_top_k = int(os.getenv("MULTI_PAPER_TOP_K", "10"))
        self.retrieval_max_distance = float(os.getenv("RETRIEVAL_MAX_DISTANCE", "0"))
        self.retrieval_mode = os.getenv("RETRIEVAL_MODE", "hybrid").strip().lower()
        self.hybrid_candidate_k = int(os.getenv("HYBRID_CANDIDATE_K", "24"))
        self.enable_two_stage_retrieval = _env_bool("ENABLE_TWO_STAGE_RETRIEVAL", True)
        self.document_candidate_k = int(os.getenv("DOCUMENT_CANDIDATE_K", "50"))
        self.document_top_k = int(os.getenv("DOCUMENT_TOP_K", "5"))
        self.section_candidate_k = int(os.getenv("SECTION_CANDIDATE_K", "50"))
        self.section_support_chunks = int(os.getenv("SECTION_SUPPORT_CHUNKS", "3"))
        self.qa_source_max_chars = int(os.getenv("QA_SOURCE_MAX_CHARS", "2400"))
        self.qa_context_max_chars = int(os.getenv("QA_CONTEXT_MAX_CHARS", "12000"))
        self.qa_max_tokens = int(os.getenv("QA_MAX_TOKENS", "1200"))
        self.qa_long_source_max_chars = int(os.getenv("QA_LONG_SOURCE_MAX_CHARS", "4200"))
        self.qa_long_context_max_chars = int(os.getenv("QA_LONG_CONTEXT_MAX_CHARS", "40000"))
        self.qa_long_max_tokens = int(os.getenv("QA_LONG_MAX_TOKENS", "3600"))
        self.qa_long_models = _env_list("QA_LONG_MODELS", "") or self.llm_models
        self.qa_long_enable_thinking = _env_bool("QA_LONG_ENABLE_THINKING", False)
        self.qa_detailed_top_k = int(os.getenv("QA_DETAILED_TOP_K", "12"))
        self.qa_planning_enabled = _env_bool("QA_PLANNING_ENABLED", True)
        self.qa_plan_max_tokens = int(os.getenv("QA_PLAN_MAX_TOKENS", "900"))
        self.qa_auto_continue = _env_bool("QA_AUTO_CONTINUE", True)
        self.summary_max_tokens = int(os.getenv("SUMMARY_MAX_TOKENS", "1800"))
        self.compare_max_tokens = int(os.getenv("COMPARE_MAX_TOKENS", "3600"))
        self.reference_section_penalty = float(os.getenv("REFERENCE_SECTION_PENALTY", "0.35"))
        self.enable_section_index = _env_bool("ENABLE_SECTION_INDEX", True)
        self.section_document_max_chars = int(os.getenv("SECTION_DOCUMENT_MAX_CHARS", "12000"))
        self.keyword_query_cache_size = int(os.getenv("KEYWORD_QUERY_CACHE_SIZE", "64"))
        self.enable_section_reranker = _env_bool(
            "ENABLE_SECTION_RERANKER", _env_bool("ENABLE_CROSS_ENCODER_RERANKER", True)
        )
        self.reranker_provider = os.getenv("RERANKER_PROVIDER", "siliconflow").strip().lower()
        self.reranker_api_key = (os.getenv("RERANKER_API_KEY") or self.embedding_api_key).strip()
        self.reranker_base_url = (os.getenv("RERANKER_BASE_URL") or self.embedding_base_url).strip()
        self.reranker_model = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3").strip()
        self.reranker_candidate_k = int(os.getenv("RERANKER_CANDIDATE_K", "8"))
        self.reranker_weight = float(os.getenv("RERANKER_WEIGHT", "0.65"))
        self.reranker_document_max_chars = int(os.getenv("RERANKER_DOCUMENT_MAX_CHARS", "2000"))
        self.reranker_timeout_seconds = float(os.getenv("RERANKER_TIMEOUT_SECONDS", "45"))
        self.reranker_max_chunks_per_doc = int(os.getenv("RERANKER_MAX_CHUNKS_PER_DOC", "8"))
        self.reranker_overlap_tokens = int(os.getenv("RERANKER_OVERLAP_TOKENS", "64"))
        self.enable_cross_encoder_reranker = self.enable_section_reranker
        self.cross_encoder_model = os.getenv("CROSS_ENCODER_MODEL", "BAAI/bge-reranker-base").strip()
        self.cross_encoder_candidate_k = int(os.getenv("CROSS_ENCODER_CANDIDATE_K", str(self.reranker_candidate_k)))
        self.cross_encoder_weight = float(os.getenv("CROSS_ENCODER_WEIGHT", str(self.reranker_weight)))
        self.cross_encoder_batch_size = int(os.getenv("CROSS_ENCODER_BATCH_SIZE", "8"))
        self.cross_encoder_local_files_only = _env_bool("CROSS_ENCODER_LOCAL_FILES_ONLY", True)
        self.preload_cross_encoder = _env_bool("PRELOAD_CROSS_ENCODER", False)
        self.enable_startup_warmup = _env_bool("ENABLE_STARTUP_WARMUP", True)
        self.rrf_k = int(os.getenv("RRF_K", "60"))
        self.enable_reranker = _env_bool("ENABLE_RERANKER", True)
        self.enable_parent_context = _env_bool("ENABLE_PARENT_CONTEXT", True)
        self.embedding_section_context = _env_bool("EMBEDDING_SECTION_CONTEXT", True)
        self.enable_section_diversity = _env_bool("ENABLE_SECTION_DIVERSITY", True)
        self.section_heading_weight = float(os.getenv("SECTION_HEADING_WEIGHT", "0.25"))
        self.section_max_chunks = int(os.getenv("SECTION_MAX_CHUNKS", "2"))
        self.query_embedding_cache_size = int(os.getenv("QUERY_EMBEDDING_CACHE_SIZE", "256"))
        self.strict_citation = _env_bool("STRICT_CITATION", True)
        self.answer_correctness_threshold = float(os.getenv("ANSWER_CORRECTNESS_THRESHOLD", "0.70"))
        self.answer_correctness_borderline_low = float(
            os.getenv("ANSWER_CORRECTNESS_BORDERLINE_LOW", "0.60")
        )
        self.answer_correctness_borderline_high = float(
            os.getenv("ANSWER_CORRECTNESS_BORDERLINE_HIGH", "0.75")
        )
        self.answer_judge_enabled = _env_bool("ANSWER_JUDGE_ENABLED", True)
        self.answer_judge_models = _env_list("ANSWER_JUDGE_MODELS", "")
        self.claim_citation_eval_enabled = _env_bool("CLAIM_CITATION_EVAL_ENABLED", True)
        self.claim_citation_similarity_threshold = float(
            os.getenv("CLAIM_CITATION_SIMILARITY_THRESHOLD", "0.55")
        )
        self.binary_consistency_retry = _env_bool("BINARY_CONSISTENCY_RETRY", True)
        self.ieee_api_key = os.getenv("IEEE_API_KEY", "").strip()
        self.ieee_api_base_url = os.getenv(
            "IEEE_API_BASE_URL", "https://ieeexploreapi.ieee.org/api/v1"
        ).strip().rstrip("/")
        self.ieee_search_timeout_seconds = float(os.getenv("IEEE_SEARCH_TIMEOUT_SECONDS", "30"))
        self.ieee_search_default_limit = int(os.getenv("IEEE_SEARCH_DEFAULT_LIMIT", "10"))
        self.arxiv_mcp_enabled = _env_bool("ARXIV_MCP_ENABLED", True)
        self.arxiv_mcp_command = (os.getenv("ARXIV_MCP_COMMAND") or sys.executable).strip()
        self.arxiv_mcp_args = _env_list("ARXIV_MCP_ARGS", "-m,arxiv_mcp_server")
        configured_arxiv_storage = os.getenv("ARXIV_MCP_STORAGE_PATH", "").strip()
        self.arxiv_mcp_storage_path = (
            Path(configured_arxiv_storage)
            if configured_arxiv_storage
            else ROOT_DIR / "storage" / "arxiv_mcp"
        )
        self.arxiv_mcp_timeout_seconds = float(os.getenv("ARXIV_MCP_TIMEOUT_SECONDS", "90"))
        self.arxiv_search_default_limit = int(os.getenv("ARXIV_SEARCH_DEFAULT_LIMIT", "5"))
        self.conversation_user_id = os.getenv("CONVERSATION_USER_ID", "local_user").strip() or "local_user"
        configured_conversation_db = os.getenv("CONVERSATION_DB_PATH", "").strip()
        self.conversation_db_path = Path(configured_conversation_db) if configured_conversation_db else CONVERSATION_DB_PATH
        self.conversation_message_limit = int(os.getenv("CONVERSATION_MESSAGE_LIMIT", "100"))
        self.api_max_upload_mb = int(os.getenv("API_MAX_UPLOAD_MB", "50"))
        self.api_background_workers = int(os.getenv("API_BACKGROUND_WORKERS", "2"))
        self.rag_anything_parser = os.getenv("RAG_ANYTHING_PARSER", "mineru").strip()
        self.rag_anything_parse_method = os.getenv("RAG_ANYTHING_PARSE_METHOD", "auto").strip()
        self.rag_anything_device = os.getenv("RAG_ANYTHING_DEVICE", "auto").strip().lower()
        self.rag_anything_backend = os.getenv("RAG_ANYTHING_BACKEND", "pipeline").strip()
        self.rag_anything_source = os.getenv("RAG_ANYTHING_SOURCE", "").strip()
        self.rag_anything_vision_model = os.getenv("RAG_ANYTHING_VISION_MODEL", self.llm_model).strip()
        self.rag_anything_embedding_dim = int(
            os.getenv("RAG_ANYTHING_EMBEDDING_DIM", self._default_embedding_dim())
        )
        self.rag_anything_max_token_size = int(os.getenv("RAG_ANYTHING_MAX_TOKEN_SIZE", "8192"))
        self.rag_anything_image = _env_bool("RAG_ANYTHING_IMAGE", True)
        self.rag_anything_table = _env_bool("RAG_ANYTHING_TABLE", True)
        self.rag_anything_formula = _env_bool("RAG_ANYTHING_FORMULA", True)
        self.mineru_local_api_startup_timeout_seconds = int(
            os.getenv("MINERU_LOCAL_API_STARTUP_TIMEOUT_SECONDS", "600")
        )
        self.mineru_task_result_timeout_seconds = int(
            os.getenv("MINERU_TASK_RESULT_TIMEOUT_SECONDS", "7200")
        )
        self.mineru_task_result_download_timeout_seconds = int(
            os.getenv("MINERU_TASK_RESULT_DOWNLOAD_TIMEOUT_SECONDS", "1200")
        )

    def _default_embedding_dim(self) -> str:
        model_name = self.embedding_model.lower()
        if model_name in {
            "text-embedding-v3",
            "text-embedding-v4",
            "qwen3.7-text-embedding",
            "qwen3.7-text-embedding-flash",
            "baai/bge-m3",
            "pro/baai/bge-m3",
        }:
            return "1024"
        if model_name == "text-embedding-3-large":
            return "3072"
        return "1536"

    @property
    def is_ready(self) -> bool:
        return self.llm_is_ready and self.embedding_is_ready

    @property
    def llm_is_ready(self) -> bool:
        return bool(self.llm_api_key and self.llm_base_url and self.llm_models)

    @property
    def embedding_is_ready(self) -> bool:
        return bool(self.embedding_api_key and self.embedding_base_url and self.embedding_model)

    @property
    def vision_is_ready(self) -> bool:
        return bool(self.vision_api_key and self.vision_base_url and self.rag_anything_vision_model)

    @property
    def ieee_is_ready(self) -> bool:
        return bool(self.ieee_api_key and self.ieee_api_base_url)

    @property
    def arxiv_mcp_is_ready(self) -> bool:
        return bool(self.arxiv_mcp_enabled and self.arxiv_mcp_command and self.arxiv_mcp_args)


settings = Settings()
