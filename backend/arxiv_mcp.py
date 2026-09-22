"""arXiv literature discovery through the local arxiv-mcp-server process."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path
import re
from typing import Any, Coroutine, TypeVar

from backend.config import settings
from backend.ieee_search import IEEEQueryPlanner


_T = TypeVar("_T")
_ARXIV_HINTS = ("arxiv", "arXiv", "预印本", "预印版", "cs.", "stat.", "quant-ph")
_ALLOWED_TOOLS = {
    "search_papers",
    "get_abstract",
    "download_paper",
    "get_paper_outline",
    "read_paper_section",
    "citation_graph",
}


def choose_literature_source(request: str) -> str:
    """Route explicit arXiv requests to MCP and preserve IEEE as the default."""

    return "arxiv" if any(hint.lower() in request.lower() for hint in _ARXIV_HINTS) else "ieee"


def _run_async(coroutine: Coroutine[Any, Any, _T]) -> _T:
    """Run one MCP transaction from synchronous Streamlit/worker code."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    # FastAPI normally calls this service in a worker thread, but this fallback
    # also keeps it safe when invoked from an already-running async environment.
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coroutine).result()


class ArxivMCPClient:
    """Minimal, allow-listed MCP client using the official stdio transport."""

    def __init__(self, *, config=settings) -> None:
        self.config = config

    @property
    def server_args(self) -> list[str]:
        args = list(self.config.arxiv_mcp_args)
        storage_path = Path(self.config.arxiv_mcp_storage_path).resolve()
        storage_path.mkdir(parents=True, exist_ok=True)
        if "--storage-path" not in args:
            args.extend(["--storage-path", str(storage_path)])
        return args

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if not self.config.arxiv_mcp_is_ready:
            raise RuntimeError("arXiv MCP 未启用，请检查 ARXIV_MCP_* 配置。")
        if name not in _ALLOWED_TOOLS:
            raise ValueError(f"不允许调用 arXiv MCP 工具：{name}")
        return _run_async(self._call_tool(name, arguments))

    async def _call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError as exc:
            raise RuntimeError(
                "缺少 MCP 依赖，请运行 pip install -r requirements.txt。"
            ) from exc

        params = StdioServerParameters(
            command=self.config.arxiv_mcp_command,
            args=self.server_args,
            env=os.environ.copy(),
        )
        timeout = max(float(self.config.arxiv_mcp_timeout_seconds), 1.0)
        try:
            async with asyncio.timeout(timeout):
                async with stdio_client(params) as (read_stream, write_stream):
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        tools = await session.list_tools()
                        available = {tool.name for tool in tools.tools}
                        if name not in available:
                            raise RuntimeError(f"arXiv MCP Server 未提供工具：{name}")
                        result = await session.call_tool(name, arguments=arguments)
        except TimeoutError as exc:
            raise RuntimeError(f"arXiv MCP 调用超时（{timeout:g} 秒）。") from exc
        except (OSError, RuntimeError) as exc:
            raise RuntimeError(f"arXiv MCP 调用失败：{exc}") from exc

        texts = [
            str(getattr(item, "text", ""))
            for item in result.content
            if getattr(item, "type", "") == "text"
        ]
        raw_text = "\n".join(text for text in texts if text).strip()
        try:
            payload = json.loads(raw_text) if raw_text else {}
        except json.JSONDecodeError:
            payload = {"text": raw_text}
        if not isinstance(payload, dict):
            payload = {"data": payload}
        if result.isError or payload.get("status") in {"error", "rate_limited"}:
            message = payload.get("message") or raw_text or "未知错误"
            raise RuntimeError(f"arXiv MCP 工具执行失败：{message}")
        return payload


class ArxivSearchService:
    """Plan a natural-language request and normalize MCP search output."""

    def __init__(self, llm_client: Any | None = None, mcp_client: ArxivMCPClient | None = None) -> None:
        self.planner = IEEEQueryPlanner(llm_client, provider="arXiv")
        self.mcp_client = mcp_client or ArxivMCPClient()

    def search(self, request: str, *, limit: int | None = None) -> dict[str, Any]:
        plan = self.planner.plan(request)
        resolved_limit = min(
            max(int(limit or settings.arxiv_search_default_limit), 1),
            50,
        )
        arguments: dict[str, Any] = {
            "query": plan.querytext,
            "max_results": resolved_limit,
            "abstract_mode": "snippet",
            "sort_by": "date" if self._asks_for_latest(request) else "relevance",
        }
        if plan.start_year:
            arguments["date_from"] = f"{plan.start_year}-01-01"
        if plan.end_year:
            arguments["date_to"] = f"{plan.end_year}-12-31"
        categories = list(dict.fromkeys(re.findall(r"\b(?:cs|stat|math|eess)\.[A-Z]{2}\b", request)))
        if categories:
            arguments["categories"] = categories

        payload = self.mcp_client.call_tool("search_papers", arguments)
        raw_papers = payload.get("papers") or []
        results = [
            self._normalize_paper(item, rank)
            for rank, item in enumerate(raw_papers, start=1)
            if isinstance(item, dict)
        ]
        results = [item for item in results if item]
        return {
            "source": "arXiv MCP",
            "request": request,
            "query_plan": {**asdict(plan), "provider": "arxiv"},
            "total_records": int(payload.get("total_results") or len(results)),
            "returned": len(results),
            "results": results,
            "search_log": {
                "querytext": plan.querytext,
                "source_order": ["arXiv MCP", "arXiv API"],
                "searched_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "mcp_tool": "search_papers",
                "transport": "stdio",
                "deduplication": "arXiv ID",
            },
            "mcp": {
                "server": "arxiv-mcp-server",
                "tool": "search_papers",
                "arguments": arguments,
            },
        }

    @staticmethod
    def _asks_for_latest(request: str) -> bool:
        lowered = request.lower()
        return any(term in lowered for term in ("最新", "近期", "recent", "latest", "newest"))

    @staticmethod
    def _normalize_paper(item: dict[str, Any], rank: int) -> dict[str, Any]:
        arxiv_id = str(item.get("id") or item.get("paper_id") or "").strip()
        title = " ".join(str(item.get("title") or "").split())
        published = str(item.get("published") or "")
        year_match = re.match(r"(19\d{2}|20\d{2})", published)
        abstract = " ".join(str(item.get("abstract") or "").split())
        return {
            "rank": rank,
            "arxiv_id": arxiv_id,
            "title": title or "未命名论文",
            "authors": [str(value) for value in item.get("authors") or []],
            "year": int(year_match.group(1)) if year_match else None,
            "published": published,
            "venue": "arXiv",
            "categories": item.get("categories") or [],
            "abstract_snippet": abstract,
            "arxiv_url": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else "",
            "pdf_url": str(item.get("url") or ""),
            "resource_uri": str(item.get("resource_uri") or ""),
        }


def format_arxiv_search_answer(payload: dict[str, Any]) -> str:
    plan = payload.get("query_plan") or {}
    return (
        f"已通过 **arXiv MCP** 调用 `search_papers`，使用检索式 "
        f"`{plan.get('querytext') or '-'}`，找到 {payload.get('total_records', 0)} 条结果，"
        f"当前展示 {payload.get('returned', 0)} 条。"
    )
