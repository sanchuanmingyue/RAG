"""科研论文阅读 Agent：Router + Tools + Memory + Evaluator。

这个模块是方案中“Agent 调度层”的主入口。它不直接实现 PDF 解析、
向量检索或大模型调用，而是把这些能力包装成工具后统一调度。

一次完整执行流程是：
1. 接收用户输入；
2. Router 判断用户意图；
3. 调用对应 Tool；
4. Evaluator 检查结果是否有依据；
5. Memory 记录本轮输入、输出和来源，方便后续继续追问、解释来源或导出。
"""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from backend.arxiv_mcp import ArxivSearchService, choose_literature_source
from backend.embeddings import OpenAICompatibleClient
from backend.conversation import ConversationResolver
from backend.evaluator import ResultEvaluator
from backend.exporter import Exporter
from backend.ieee_search import LiteratureSearchService
from backend.memory import AgentMemory
from backend.rag_chain import analyze_question, is_refusal_answer
from backend.router import AgentRouter
from backend.schemas import AgentIntent, AgentResult
from backend.tools import (
    ArxivSearchTool,
    ExportTool,
    CorpusClassificationTool,
    LibraryStatusTool,
    LiteratureSearchTool,
    PaperCompareTool,
    PaperQATool,
    PaperSummaryTool,
    SourceExplainTool,
)
from backend.vector_store import ChromaVectorStore


DEEP_CARD_PROFILE_MARKER = "<!-- paper-card-profile:deep-v2 -->"


class AgentGraphState(TypedDict, total=False):
    """Serializable execution state passed between LangGraph nodes."""

    user_query: str
    intent: AgentIntent
    resolution: Any
    previous_artifact: dict[str, Any]
    result: AgentResult
    retrieval_question: str | None
    retry_count: int
    needs_retry: bool
    trace: list[dict[str, Any]]


class ResearchAgent:
    """统一调度论文阅读任务。

    这里采用“可控 Agent”而不是完全让大模型自主规划。原因是论文阅读场景
    对可追溯性要求高：系统应该清楚地知道自己调用了哪个工具、用了哪些来源、
    为什么拒答。规则 Router + 固定工具链更稳定，也更适合课程项目和面试讲解。
    """

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        llm_client: OpenAICompatibleClient,
        memory: AgentMemory,
    ) -> None:
        # vector_store 和 memory 保存在 Agent 上，方便多个工具共享同一套论文库和会话状态。
        self.vector_store = vector_store
        self.memory = memory
        self.llm_client = llm_client

        # Router 负责“判断做什么”，Evaluator 负责“检查结果是否可信”。
        self.router = AgentRouter()
        self.evaluator = ResultEvaluator()

        # 每个工具只关心自己的任务：问答、总结、对比、解释来源或导出。
        # Agent 只负责编排，不把所有业务逻辑堆在一个函数里。
        self.qa_tool = PaperQATool(vector_store, llm_client)
        self.summary_tool = PaperSummaryTool(vector_store, llm_client)
        self.compare_tool = PaperCompareTool(llm_client)
        self.source_tool = SourceExplainTool()
        self.export_tool = ExportTool()
        self.library_status_tool = LibraryStatusTool(vector_store)
        self.literature_search_tool = LiteratureSearchTool(
            LiteratureSearchService(llm_client=llm_client)
        )
        self.arxiv_search_tool = ArxivSearchTool(
            ArxivSearchService(llm_client=llm_client)
        )
        self.corpus_classification_tool = CorpusClassificationTool(vector_store, llm_client)
        self.graph = self._build_graph()

    def run(self, user_query: str) -> AgentResult:
        """执行一次 Agent 调度。

        参数 user_query 是用户在 Agent 工作台输入的自然语言指令，例如：
        “总结这篇论文”“比较选中的论文”“解释刚才的来源”“导出 JSON”。
        """

        final_state = self.graph.invoke(
            {
                "user_query": user_query,
                "retry_count": 0,
                "trace": [],
            }
        )
        return final_state["result"]

    def _build_graph(self):
        workflow = StateGraph(AgentGraphState)
        workflow.add_node("route", self._route_node)
        workflow.add_node("local_qa", self._local_qa_node)
        workflow.add_node("external_search", self._external_search_node)
        workflow.add_node("corpus_analysis", self._corpus_analysis_node)
        workflow.add_node("document_action", self._document_action_node)
        workflow.add_node("evidence_gate", self._evidence_gate_node)
        workflow.add_node("rewrite_query", self._rewrite_query_node)
        workflow.add_node("evaluate", self._evaluate_node)
        workflow.add_node("persist", self._persist_node)

        workflow.add_edge(START, "route")
        workflow.add_conditional_edges(
            "route",
            self._route_branch,
            {
                "local_qa": "local_qa",
                "external_search": "external_search",
                "corpus_analysis": "corpus_analysis",
                "document_action": "document_action",
            },
        )
        for node in ("local_qa", "external_search", "corpus_analysis", "document_action"):
            workflow.add_edge(node, "evidence_gate")
        workflow.add_conditional_edges(
            "evidence_gate",
            self._evidence_branch,
            {"retry": "rewrite_query", "accept": "evaluate"},
        )
        workflow.add_edge("rewrite_query", "local_qa")
        workflow.add_edge("evaluate", "persist")
        workflow.add_edge("persist", END)
        return workflow.compile()

    def _route_node(self, state: AgentGraphState) -> dict[str, Any]:
        user_query = state["user_query"]
        resolution = ConversationResolver(self.llm_client).resolve(
            user_query, self.memory.state
        )
        self.memory.add_turn("user", user_query)
        return {
            "intent": resolution.intent,
            "resolution": resolution,
            "previous_artifact": dict(self.memory.state.active_artifact),
            "trace": self._trace(state, "route", intent=resolution.intent),
        }

    @staticmethod
    def _route_branch(state: AgentGraphState) -> str:
        intent = state["intent"]
        if intent == "qa":
            return "local_qa"
        if intent == "literature_search":
            return "external_search"
        if intent == "corpus_analysis":
            return "corpus_analysis"
        return "document_action"

    def _local_qa_node(self, state: AgentGraphState) -> dict[str, Any]:
        resolution = state["resolution"]
        retrieval_question = state.get("retrieval_question")
        if not retrieval_question and resolution.is_followup:
            retrieval_question = resolution.rewritten_query
        result = self.qa_tool.run(
            state["user_query"],
            paper_id=self.memory.current_paper_id,
            retrieval_question=retrieval_question,
        )
        return {
            "result": result,
            "retrieval_question": retrieval_question,
            "trace": self._trace(
                state,
                "local_qa",
                source_count=len(result.sources),
                retry_count=state.get("retry_count", 0),
            ),
        }

    def _external_search_node(self, state: AgentGraphState) -> dict[str, Any]:
        provider = choose_literature_source(state["user_query"])
        tool = self.arxiv_search_tool if provider == "arxiv" else self.literature_search_tool
        result = tool.run(state["user_query"])
        return {
            "result": result,
            "trace": self._trace(state, "external_search", provider=provider),
        }

    def _corpus_analysis_node(self, state: AgentGraphState) -> dict[str, Any]:
        result = self.corpus_classification_tool.run(
            state["user_query"],
            previous_artifact=state.get("previous_artifact") or {},
            resolution=state["resolution"],
        )
        return {
            "result": result,
            "trace": self._trace(state, "corpus_analysis"),
        }

    def _document_action_node(self, state: AgentGraphState) -> dict[str, Any]:
        intent = state["intent"]
        user_query = state["user_query"]
        if intent == "summary":
            result = self.summary_tool.run(self.memory.current_paper_id)
            if self.memory.current_paper_id and result.sources:
                self.memory.generated_cards[self.memory.current_paper_id] = result.answer
        elif intent == "compare":
            result = self._run_compare()
        elif intent == "source_explain":
            result = self.source_tool.run(self.memory.last_sources)
        elif intent == "export":
            file_format = Exporter.pick_format(user_query)
            result = self.export_tool.run(self.memory, file_format=file_format)
            self.memory.last_export_path = result.artifacts.get("path", "")
        elif intent == "library_status":
            result = self.library_status_tool.run(user_query)
        else:
            raise ValueError(f"未实现的 Agent 意图：{intent}")
        return {
            "result": result,
            "trace": self._trace(state, "document_action", action=intent),
        }

    def _evidence_gate_node(self, state: AgentGraphState) -> dict[str, Any]:
        result = state["result"]
        retry_count = state.get("retry_count", 0)
        needs_retry = (
            state["intent"] == "qa"
            and retry_count < 1
            and (not result.sources or is_refusal_answer(result.answer))
        )
        return {
            "needs_retry": needs_retry,
            "trace": self._trace(
                state,
                "evidence_gate",
                decision="retry" if needs_retry else "accept",
                source_count=len(result.sources),
            ),
        }

    @staticmethod
    def _evidence_branch(state: AgentGraphState) -> str:
        return "retry" if state.get("needs_retry") else "accept"

    def _rewrite_query_node(self, state: AgentGraphState) -> dict[str, Any]:
        original = state["user_query"]
        fallback = analyze_question(original).rewritten_query
        prompt = f"""把下面的问题改写成一条可独立用于科研论文语义检索的查询。
保留原有实体、方法名、数据集和数值约束；补充必要的英文学术同义词，但不要回答问题。
只输出改写后的查询，不要解释。
问题：{original}"""
        try:
            rewritten = self.llm_client.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=180,
            ).strip()
        except Exception:
            rewritten = fallback
        rewritten = " ".join((rewritten or fallback).split())[:1200]
        return {
            "retrieval_question": rewritten,
            "retry_count": state.get("retry_count", 0) + 1,
            "trace": self._trace(state, "rewrite_query", query=rewritten),
        }

    def _evaluate_node(self, state: AgentGraphState) -> dict[str, Any]:
        checked = self.evaluator.check(state["result"])
        return {
            "result": checked,
            "trace": self._trace(state, "evaluate", warning_count=len(checked.warnings)),
        }

    def _persist_node(self, state: AgentGraphState) -> dict[str, Any]:
        checked = state["result"]
        resolution = state["resolution"]
        trace = self._trace(state, "persist")
        checked.artifacts["workflow"] = {
            "engine": "langgraph",
            "retry_count": state.get("retry_count", 0),
            "trace": trace,
        }
        self.memory.remember_result(
            checked.answer,
            checked.sources,
            intent=checked.intent,
            user_query=state["user_query"],
            rewritten_query=state.get("retrieval_question")
            or (resolution.rewritten_query if resolution.is_followup else None),
            artifacts=checked.artifacts,
        )
        self.memory.state.selected_category = resolution.target_category
        self.memory.add_turn("assistant", checked.answer)
        return {"result": checked, "trace": trace}

    @staticmethod
    def _trace(state: AgentGraphState, node: str, **details: Any) -> list[dict[str, Any]]:
        return [*(state.get("trace") or []), {"node": node, **details}]

    def _run_compare(self) -> AgentResult:
        """执行多论文对比。

        方案里建议“先生成单篇阅读卡片，再基于卡片对比”。这样做可以减少多文档
        混淆：每篇论文先形成独立结构化表示，再进入对比工具。
        """

        selected_ids = self.memory.selected_paper_ids
        if len(selected_ids) < 2:
            return AgentResult(answer="请在左侧至少选择两篇论文后再进行对比。", intent="compare")

        cards: dict[str, str] = {}
        for paper_id in selected_ids:
            card = self.memory.generated_cards.get(paper_id)
            if not card or not card.startswith(DEEP_CARD_PROFILE_MARKER):
                # 如果某篇论文还没有卡片，就自动补生成。用户只需要点击“对比”，
                # Agent 会完成“补卡片 -> 汇总对比”的多步编排。旧版短卡片没有
                # profile 标记，会在第一次深度对比时自动刷新。
                summary = self.summary_tool.run(paper_id)
                checked_summary = self.evaluator.check(summary)
                if checked_summary.sources:
                    card = f"{DEEP_CARD_PROFILE_MARKER}\n{checked_summary.answer}"
                    self.memory.generated_cards[paper_id] = card
            if card:
                cards[paper_id] = card

        # 对比工具只接收结构化卡片，不直接混检多篇论文原文。
        return self.compare_tool.run(cards)
