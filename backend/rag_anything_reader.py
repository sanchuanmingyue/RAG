"""Optional RAG-Anything integration for multimodal paper reading."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache, partial
import importlib.util
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import os
import subprocess
import sys
import urllib.request
from typing import Any

from backend.config import RAG_ANYTHING_DIR, RAG_ANYTHING_OUTPUT_DIR, settings


@dataclass
class RagAnythingDependencyStatus:
    available: bool
    missing: list[str]


@dataclass(frozen=True)
class ComputeRuntimeStatus:
    cuda_available: bool
    selected_device: str
    torch_version: str
    cuda_version: str | None
    gpu_name: str | None
    vram_gb: float | None


def check_rag_anything_dependencies() -> RagAnythingDependencyStatus:
    """Check packages without importing the heavy RAG-Anything stack."""

    missing: list[str] = []
    if importlib.util.find_spec("raganything") is None:
        missing.append("raganything")
    if importlib.util.find_spec("lightrag") is None:
        missing.append("lightrag")

    return RagAnythingDependencyStatus(available=not missing, missing=missing)


@lru_cache(maxsize=1)
def detect_compute_runtime() -> ComputeRuntimeStatus:
    """Detect the CUDA-capable runtime without importing heavyweight PyTorch."""

    try:
        torch_version = version("torch")
    except PackageNotFoundError:
        torch_version = "unavailable"

    gpu_name: str | None = None
    vram_gb: float | None = None
    try:
        subprocess_kwargs: dict[str, Any] = {
            "capture_output": True,
            "text": True,
            "encoding": "utf-8",
            "errors": "ignore",
            "timeout": 5,
        }
        if os.name == "nt":
            subprocess_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        detected = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            **subprocess_kwargs,
        ).stdout.strip().splitlines()[0]
        gpu_name, memory_mb = (part.strip() for part in detected.split(",", 1))
        vram_gb = round(float(memory_mb) / 1024, 1)
    except Exception:
        pass

    cuda_build = None
    if "+cu" in torch_version:
        cuda_build = torch_version.split("+cu", 1)[1]
        cuda_build = f"{cuda_build[:2]}.{cuda_build[2:]}" if len(cuda_build) >= 3 else cuda_build
    cuda_available = bool(gpu_name and cuda_build)
    try:
        selected = settings.rag_anything_device
        if selected in {"", "auto"}:
            selected = "cuda" if cuda_available else "cpu"
        return ComputeRuntimeStatus(
            cuda_available=cuda_available,
            selected_device=selected,
            torch_version=torch_version,
            cuda_version=cuda_build,
            gpu_name=gpu_name,
            vram_gb=vram_gb,
        )
    except Exception:
        return ComputeRuntimeStatus(False, "cpu", torch_version, None, gpu_name, vram_gb)


def _ensure_runtime_scripts_on_path() -> None:
    scripts_dir = Path(sys.executable).resolve().parent / "Scripts"
    if not scripts_dir.exists():
        return

    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    scripts_path = str(scripts_dir)
    if scripts_path not in path_parts:
        os.environ["PATH"] = f"{scripts_path}{os.pathsep}{os.environ.get('PATH', '')}"


def _patch_mineru_installation_check() -> None:
    """Make RAG-Anything's MinerU check work in Windows conda envs."""

    try:
        from raganything.parser import MineruParser
    except Exception:
        return

    if getattr(MineruParser, "_paper_reader_patch_applied", False):
        return

    original_check = MineruParser.check_installation
    mineru_exe = Path(sys.executable).resolve().parent / "Scripts" / "mineru.exe"

    def check_installation(self) -> bool:
        if mineru_exe.exists():
            subprocess_kwargs = {
                "capture_output": True,
                "text": True,
                "encoding": "utf-8",
                "errors": "ignore",
            }
            if os.name == "nt":
                subprocess_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
            try:
                subprocess.run([str(mineru_exe), "--version"], check=True, **subprocess_kwargs)
                return True
            except Exception:
                pass
        return original_check(self)

    MineruParser.check_installation = check_installation
    MineruParser._paper_reader_patch_applied = True


def _apply_mineru_timeout_env() -> None:
    _ensure_loopback_proxy_bypass()
    os.environ.setdefault(
        "MINERU_LOCAL_API_STARTUP_TIMEOUT_SECONDS",
        str(settings.mineru_local_api_startup_timeout_seconds),
    )
    os.environ.setdefault(
        "MINERU_TASK_RESULT_TIMEOUT_SECONDS",
        str(settings.mineru_task_result_timeout_seconds),
    )
    os.environ.setdefault(
        "MINERU_TASK_RESULT_DOWNLOAD_TIMEOUT_SECONDS",
        str(settings.mineru_task_result_download_timeout_seconds),
    )
    runtime = detect_compute_runtime()
    os.environ["MINERU_DEVICE_MODE"] = runtime.selected_device
    if runtime.vram_gb is not None and runtime.vram_gb <= 4.5:
        os.environ.setdefault("MINERU_HYBRID_BATCH_RATIO", "1")


def _ensure_loopback_proxy_bypass() -> None:
    """Prevent Windows system proxies from intercepting MinerU localhost health checks."""

    # On Windows urllib/httpx can discover proxies from the registry. Once
    # NO_PROXY is set, that discovery may return only the bypass list, so copy
    # the existing system proxy into explicit environment variables first.
    discovered = urllib.request.getproxies()
    proxy_names = {
        "http": ("HTTP_PROXY", "http_proxy"),
        "https": ("HTTPS_PROXY", "https_proxy"),
    }
    for scheme, names in proxy_names.items():
        proxy_url = discovered.get(scheme)
        if not proxy_url or any(os.environ.get(name) for name in names):
            continue
        for name in names:
            os.environ[name] = proxy_url

    required = ("127.0.0.1", "localhost", "::1")
    existing_values = [os.environ.get("NO_PROXY", ""), os.environ.get("no_proxy", "")]
    entries: list[str] = []
    for value in existing_values:
        entries.extend(item.strip() for item in value.split(",") if item.strip())
    normalized = {item.lower() for item in entries}
    entries.extend(item for item in required if item.lower() not in normalized)
    merged = ",".join(dict.fromkeys(entries))
    os.environ["NO_PROXY"] = merged
    os.environ["no_proxy"] = merged


def _mineru_major_version() -> int:
    try:
        return int(version("mineru").split(".", 1)[0])
    except (PackageNotFoundError, ValueError):
        return 0


class RagAnythingPaperReader:
    """Build and run a RAG-Anything pipeline with this project's model config."""

    def __init__(self) -> None:
        _apply_mineru_timeout_env()
        _ensure_runtime_scripts_on_path()
        _patch_mineru_installation_check()

        status = check_rag_anything_dependencies()
        if not status.available:
            missing = ", ".join(status.missing)
            raise RuntimeError(f"RAG-Anything dependencies are not installed: {missing}")

        if not settings.is_ready:
            raise RuntimeError(
                "Please configure both the LLM and EMBEDDING API key, base URL, and model first."
            )
        if not settings.vision_is_ready:
            raise RuntimeError("Please configure VISION_API_KEY, VISION_BASE_URL, and RAG_ANYTHING_VISION_MODEL first.")

        self.rag = self._build_rag()

    def _build_rag(self) -> Any:
        from lightrag.llm.openai import openai_complete_if_cache, openai_embed
        from lightrag.utils import EmbeddingFunc
        from raganything import RAGAnything, RAGAnythingConfig

        RAG_ANYTHING_DIR.mkdir(parents=True, exist_ok=True)
        RAG_ANYTHING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        config = RAGAnythingConfig(
            working_dir=str(RAG_ANYTHING_DIR),
            parser=settings.rag_anything_parser,
            parse_method=settings.rag_anything_parse_method,
            enable_image_processing=settings.rag_anything_image,
            enable_table_processing=settings.rag_anything_table,
            enable_equation_processing=settings.rag_anything_formula,
        )

        def llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs):
            return openai_complete_if_cache(
                settings.llm_model,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages or [],
                api_key=settings.llm_api_key,
                base_url=settings.llm_base_url,
                **kwargs,
            )

        def vision_model_func(
            prompt,
            system_prompt=None,
            history_messages=None,
            image_data=None,
            messages=None,
            **kwargs,
        ):
            if messages:
                return openai_complete_if_cache(
                    settings.rag_anything_vision_model,
                    "",
                    system_prompt=None,
                    history_messages=[],
                    messages=messages,
                    api_key=settings.vision_api_key,
                    base_url=settings.vision_base_url,
                    **kwargs,
                )

            if image_data:
                message_items = []
                if system_prompt:
                    message_items.append({"role": "system", "content": system_prompt})
                message_items.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                            },
                        ],
                    }
                )
                return openai_complete_if_cache(
                    settings.rag_anything_vision_model,
                    "",
                    system_prompt=None,
                    history_messages=[],
                    messages=message_items,
                    api_key=settings.vision_api_key,
                    base_url=settings.vision_base_url,
                    **kwargs,
                )

            return llm_model_func(prompt, system_prompt, history_messages, **kwargs)

        embed_callable = getattr(openai_embed, "func", openai_embed)
        embedding_func = EmbeddingFunc(
            embedding_dim=settings.rag_anything_embedding_dim,
            max_token_size=settings.rag_anything_max_token_size,
            func=partial(
                embed_callable,
                model=settings.embedding_model,
                api_key=settings.embedding_api_key,
                base_url=settings.embedding_base_url,
            ),
        )

        return RAGAnything(
            config=config,
            llm_model_func=llm_model_func,
            vision_model_func=vision_model_func,
            embedding_func=embedding_func,
        )

    async def process_document(
        self,
        file_path: Path,
        *,
        start_page: int | None = None,
        end_page: int | None = None,
        lang: str | None = None,
        device: str | None = None,
        formula: bool | None = None,
        table: bool | None = None,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "file_path": str(file_path),
            "output_dir": str(RAG_ANYTHING_OUTPUT_DIR),
            "parse_method": settings.rag_anything_parse_method,
            "formula": settings.rag_anything_formula if formula is None else formula,
            "table": settings.rag_anything_table if table is None else table,
            "display_stats": True,
        }
        if settings.rag_anything_backend:
            kwargs["backend"] = settings.rag_anything_backend
        if settings.rag_anything_source:
            if _mineru_major_version() >= 3:
                os.environ["MINERU_MODEL_SOURCE"] = settings.rag_anything_source
            else:
                kwargs["source"] = settings.rag_anything_source
        if start_page is not None:
            kwargs["start_page"] = start_page
        if end_page is not None:
            kwargs["end_page"] = end_page
        if lang:
            kwargs["lang"] = lang
        resolved_device = device or detect_compute_runtime().selected_device
        # MinerU 3 uses MINERU_DEVICE_MODE and removed the legacy CLI -d flag.
        if _mineru_major_version() < 3:
            kwargs["device"] = resolved_device

        scripts_dir = Path(sys.executable).resolve().parent / "Scripts"
        if scripts_dir.exists():
            kwargs["env"] = {
                "PATH": f"{scripts_dir}{os.pathsep}{os.environ.get('PATH', '')}",
                "MINERU_DEVICE_MODE": resolved_device,
                "NO_PROXY": os.environ.get("NO_PROXY", "127.0.0.1,localhost,::1"),
                "no_proxy": os.environ.get("no_proxy", "127.0.0.1,localhost,::1"),
            }

        return await self.rag.process_document_complete(**kwargs)

    async def ask(self, question: str, *, mode: str = "hybrid") -> Any:
        return await self.rag.aquery(question, mode=mode)

    async def ask_with_content(
        self,
        question: str,
        multimodal_content: list[dict[str, Any]],
        *,
        mode: str = "hybrid",
    ) -> Any:
        return await self.rag.aquery_with_multimodal(
            question,
            multimodal_content=multimodal_content,
            mode=mode,
        )

    def parser_ready(self) -> bool:
        if hasattr(self.rag, "check_parser_installation"):
            return bool(self.rag.check_parser_installation())
        return True
