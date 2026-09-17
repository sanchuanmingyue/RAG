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
from backend.exporter import Exporter
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

    def run(self, question: str, paper_id: str | None = None) -> AgentResult:
        # paper_id 为空时表示在全部论文范围检索；不为空时只检索指定论文。
        result = self.rag.answer(question, paper_id=paper_id)
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
