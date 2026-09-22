"""Agent Tools：把问答、总结、对比、来源解释和导出封装成统一工具。

Tool 的作用是“把一个具体能力封装成稳定接口”。Agent 不需要知道问答内部
如何检索、总结内部如何构造 Prompt、导出内部如何写文件，只需要调用工具的
run 方法即可。这样后续新增 Word 导出、Rerank、章节解释等能力时，不会破坏
Agent 主流程。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.embeddings import OpenAICompatibleClient
from backend.arxiv_mcp import ArxivSearchService, format_arxiv_search_answer
from backend.corpus_analysis import (
    CorpusAnalysisService,
    format_corpus_classification,
    requested_category_count,
)
from backend.conversation import FollowUpResolution
from backend.exporter import Exporter
from backend.ieee_search import LiteratureSearchService, format_search_answer
from backend.paper_compare import compare_paper_cards
from backend.rag_chain import PaperRAG, format_sources
from backend.schemas import AgentResult
from backend.summarizer import generate_paper_note
from backend.vector_store import ChromaVectorStore


class PaperQATool:
    """论文问答工具。

    这是对原有 PaperRAG 的薄封装。保留原来的 RAG 实现，避免为了 Agent
    改动底层检索链路；Tool 只负责把返回值转换成统一的 AgentResult。
    """

    def __init__(self, vector_store: ChromaVectorStore, llm_client: OpenAICompatibleClient) -> None:
        self.rag = PaperRAG(vector_store, llm_client)

    def run(
        self,
        question: str,
        paper_id: str | None = None,
        *,
        retrieval_question: str | None = None,
    ) -> AgentResult:
        # paper_id 为空时表示在全部论文范围检索；不为空时只检索指定论文。
        result = self.rag.answer(
            question,
            paper_id=paper_id,
            retrieval_question=retrieval_question,
        )
        return AgentResult(answer=result["answer"], intent="qa", sources=result["sources"])


class PaperSummaryTool:
    """论文阅读卡片工具。

    阅读卡片是多论文对比的中间层。相比直接把多篇论文 chunk 混在一起给模型，
    先生成单篇卡片能让每篇论文的信息边界更清楚。
    """

    def __init__(self, vector_store: ChromaVectorStore, llm_client: OpenAICompatibleClient) -> None:
        self.vector_store = vector_store
        self.llm_client = llm_client

    def run(self, paper_id: str | None) -> AgentResult:
        # 总结必须绑定具体论文；否则“这篇论文”没有明确指向。
        if not paper_id:
            return AgentResult(answer="请先选择一篇具体论文，再生成阅读卡片。", intent="summary")
        result = generate_paper_note(paper_id, self.vector_store, self.llm_client)
        return AgentResult(answer=result["note"], intent="summary", sources=result["sources"])


class PaperCompareTool:
    """多论文对比工具。

    输入是若干篇论文的阅读卡片，而不是原始 chunk。这样能把“抽取信息”和
    “比较信息”拆成两个步骤，降低长上下文和跨论文混淆风险。
    """

    def __init__(self, llm_client: OpenAICompatibleClient) -> None:
        self.llm_client = llm_client

    def run(self, cards: dict[str, str]) -> AgentResult:
        if len(cards) < 2:
            return AgentResult(answer="请至少选择两篇论文，并先生成或允许自动生成阅读卡片。", intent="compare")

        # 给每张卡片加上论文 ID，方便模型在对比表中区分不同论文。
        labeled_cards = [f"论文 ID：{paper_id}\n\n{card}" for paper_id, card in cards.items()]
        answer = compare_paper_cards(labeled_cards, self.llm_client)
        return AgentResult(answer=answer, intent="compare", artifacts={"card_count": len(cards)})


class SourceExplainTool:
    """来源解释工具。

    这个工具不调用大模型，也不重新检索，只把最近一次结果的 sources 展示得更清楚。
    它用于回答“刚才这个结论来自哪里”“页码依据是什么”这类追问。
    """

    def run(self, sources: list[dict[str, Any]]) -> AgentResult:
        if not sources:
            return AgentResult(answer="当前没有可解释的来源片段。请先完成一次问答或总结。", intent="source_explain")

        lines = ["下面是最近一次结果使用的来源片段："]
        for index, source in enumerate(format_sources(sources), start=1):
            # distance 是向量距离，通常越小越相关。这里展示出来便于调试检索质量。
            distance = sources[index - 1].get("distance")
            distance_text = f"，distance={distance:.4f}" if isinstance(distance, float) else ""
            lines.append(f"{index}. {source}{distance_text}")
        return AgentResult(answer="\n\n".join(lines), intent="source_explain", sources=sources)


class ExportTool:
    """导出工具。

    导出只依赖 Memory，不需要再次调用模型。这样用户已经生成了结果后，即使 API
    临时不可用，也可以把当前会话、来源和阅读卡片保存下来。
    """

    def __init__(self) -> None:
        self.exporter = Exporter()

    def run(self, memory: Any, file_format: str) -> AgentResult:
        # 真正的文件写入逻辑放在 Exporter 中，Tool 只负责统一返回 AgentResult。
        path = self.exporter.export_memory(memory, file_format=file_format)
        return AgentResult(
            answer=f"已导出到：{Path(path)}",
            intent="export",
            artifacts={"path": str(path), "format": file_format},
        )


class LibraryStatusTool:
    """Read deterministic knowledge-base metadata without retrieval or an LLM call."""

    def __init__(self, vector_store: ChromaVectorStore) -> None:
        self.vector_store = vector_store

    def run(self, question: str = "") -> AgentResult:
        papers = self.vector_store.list_papers()
        total_chunks = self.vector_store.count_chunks()
        if not papers:
            return AgentResult(
                answer="当前知识库中没有已索引论文。",
                intent="library_status",
                artifacts={"paper_count": 0, "chunk_count": 0, "papers": []},
            )

        normalized = question.lower()
        wants_details = any(
            keyword in normalized
            for keyword in ("列出", "列表", "哪些", "有什么", "每篇", "文件", "chunk", "状态", "是否")
        )
        answer = f"当前知识库共有 **{len(papers)} 篇**已索引论文，共 **{total_chunks} 个 chunks**。"
        if wants_details:
            rows = []
            for index, paper in enumerate(papers, start=1):
                rows.append(
                    f"{index}. `{paper.get('file_name') or paper.get('paper_id') or '未命名论文'}`"
                    f" — {int(paper.get('chunk_count') or 0)} chunks"
                )
            answer += "\n\n" + "\n".join(rows)
        return AgentResult(
            answer=answer,
            intent="library_status",
            artifacts={
                "paper_count": len(papers),
                "chunk_count": total_chunks,
                "papers": papers,
            },
        )


class LiteratureSearchTool:
    """Expose verified IEEE Xplore discovery through the Agent tool interface."""

    def __init__(self, search_service: LiteratureSearchService) -> None:
        self.search_service = search_service

    def run(self, request: str) -> AgentResult:
        payload = self.search_service.search(request)
        lines = [format_search_answer(payload)]
        for item in payload.get("results") or []:
            url = item.get("ieee_url") or (f"https://doi.org/{item['doi']}" if item.get("doi") else "")
            title = item.get("title") or "未命名论文"
            title_text = f"[{title}]({url})" if url else title
            authors = ", ".join((item.get("authors") or [])[:4]) or "作者信息未返回"
            score = (item.get("score") or {}).get("total", 0)
            doi = item.get("doi") or "未返回"
            lines.append(
                f"{item.get('rank')}. **{title_text}**\n"
                f"   - {authors} · {item.get('venue') or 'IEEE'} · {item.get('year') or '年份未知'}\n"
                f"   - DOI：{doi} · 综合评分：{score}/12"
            )
        return AgentResult(
            answer="\n\n".join(lines),
            intent="literature_search",
            artifacts={"search": payload},
        )


class ArxivSearchTool:
    """Expose arXiv discovery through a real MCP stdio tool call."""

    def __init__(self, search_service: ArxivSearchService) -> None:
        self.search_service = search_service

    def run(self, request: str) -> AgentResult:
        payload = self.search_service.search(request)
        lines = [format_arxiv_search_answer(payload)]
        for item in payload.get("results") or []:
            title = item.get("title") or "未命名论文"
            url = item.get("arxiv_url") or item.get("pdf_url") or ""
            title_text = f"[{title}]({url})" if url else title
            authors = ", ".join((item.get("authors") or [])[:4]) or "作者信息未返回"
            categories = ", ".join((item.get("categories") or [])[:4]) or "未分类"
            lines.append(
                f"{item.get('rank')}. **{title_text}**\n"
                f"   - {authors} · {item.get('year') or '年份未知'} · {categories}\n"
                f"   - arXiv：`{item.get('arxiv_id') or '未知'}`"
            )
        return AgentResult(
            answer="\n\n".join(lines),
            intent="literature_search",
            artifacts={"search": payload, "provider": "arxiv_mcp"},
        )


class CorpusClassificationTool:
    """Classify the complete paper library by background-section semantics."""

    def __init__(self, vector_store: ChromaVectorStore, llm_client: OpenAICompatibleClient) -> None:
        self.service = CorpusAnalysisService(vector_store, llm_client)

    def run(
        self,
        request: str,
        *,
        previous_artifact: dict[str, Any] | None = None,
        resolution: FollowUpResolution | None = None,
    ) -> AgentResult:
        previous = previous_artifact or {}
        operation = resolution.operation if resolution else None

        if operation == "list_category":
            category = self._category(previous, resolution.target_category)
            if category is None:
                return AgentResult(
                    answer="上一轮分类结果中没有找到这个类别，请先确认类别编号。",
                    intent="corpus_analysis",
                    artifacts={"corpus_analysis": previous} if previous else {},
                )
            papers = category.get("papers") or []
            answer = (
                f"### 第 {category.get('rank')} 类：{category.get('label')}（{len(papers)} 篇）\n\n"
                + "\n".join(f"- `{paper.get('file_name')}`" for paper in papers)
            )
            return AgentResult(
                answer=answer,
                intent="corpus_analysis",
                artifacts={"corpus_analysis": previous},
            )

        if operation == "explain_classification" and previous:
            lines = [
                f"本轮采用 `{previous.get('method')}`，共处理 {previous.get('paper_count')} 篇论文。"
            ]
            if previous.get("silhouette_score") is not None:
                lines.append(f"聚类轮廓系数为 `{previous['silhouette_score']}`。")
            for category in previous.get("categories") or []:
                keywords = "、".join(category.get("keywords") or []) or "未提取到稳定关键词"
                representatives = "、".join(category.get("representative_papers") or [])
                lines.append(
                    f"- **第 {category.get('rank')} 类 {category.get('label')}**："
                    f"关键词为 {keywords}；代表论文为 {representatives}。"
                )
            return AgentResult(
                answer="\n\n".join(lines),
                intent="corpus_analysis",
                artifacts={"corpus_analysis": previous},
            )

        if operation == "subdivide_category":
            category = self._category(previous, resolution.target_category)
            if category is None:
                raise ValueError("上一轮分类结果中没有找到需要细分的类别。")
            paper_ids = {str(item.get("paper_id")) for item in category.get("papers") or [] if item.get("paper_id")}
            if len(paper_ids) < 3:
                raise ValueError("该类别少于 3 篇论文，无法继续进行有意义的语义细分。")
            forced_count = requested_category_count(request) or min(3, len(paper_ids) - 1)
            payload = self.service.classify_by_background(
                request,
                paper_ids=paper_ids,
                category_count=forced_count,
            )
            payload["parent_category"] = {
                "rank": category.get("rank"),
                "label": category.get("label"),
                "source_artifact_id": previous.get("artifact_id"),
            }
        elif operation == "refine_classification" and previous:
            old_count = int(previous.get("category_count") or 3)
            paper_count = int(previous.get("paper_count") or 0)
            forced_count = requested_category_count(request) or min(max(old_count + 2, old_count * 2), 12)
            forced_count = min(forced_count, max(paper_count - 1, 2))
            payload = self.service.classify_by_background(request, category_count=forced_count)
            payload["refined_from_artifact_id"] = previous.get("artifact_id")
        else:
            payload = self.service.classify_by_background(request)
        return AgentResult(
            answer=format_corpus_classification(payload),
            intent="corpus_analysis",
            artifacts={"corpus_analysis": payload},
        )

    @staticmethod
    def _category(payload: dict[str, Any], rank: int | None) -> dict[str, Any] | None:
        if rank is None:
            return None
        return next(
            (item for item in payload.get("categories") or [] if int(item.get("rank") or 0) == rank),
            None,
        )
