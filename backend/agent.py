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

from backend.embeddings import OpenAICompatibleClient
from backend.evaluator import ResultEvaluator
from backend.exporter import Exporter
from backend.memory import AgentMemory
from backend.router import AgentRouter
from backend.schemas import AgentResult
from backend.tools import ExportTool, PaperCompareTool, PaperQATool, PaperSummaryTool, SourceExplainTool
from backend.vector_store import ChromaVectorStore


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

    def run(self, user_query: str) -> AgentResult:
        """执行一次 Agent 调度。

        参数 user_query 是用户在 Agent 工作台输入的自然语言指令，例如：
        “总结这篇论文”“比较选中的论文”“解释刚才的来源”“导出 JSON”。
        """

        # 先记录用户输入，这样导出时可以看到完整会话过程。
        self.memory.add_turn("user", user_query)

        # Router 把自然语言输入映射为固定意图，后续根据意图选择工具。
        intent = self.router.route(user_query)

        if intent == "summary":
            # 阅读卡片必须针对单篇论文，因此使用 memory.current_paper_id。
            result = self.summary_tool.run(self.memory.current_paper_id)
            if self.memory.current_paper_id and result.sources:
                # 生成成功后把卡片缓存到 Memory。多论文对比会复用这些卡片，
                # 避免同一篇论文被重复总结，减少模型调用成本。
                self.memory.generated_cards[self.memory.current_paper_id] = result.answer
        elif intent == "compare":
            # 多论文对比是一个复合任务，单独拆成私有方法，保持 run 主流程清楚。
            result = self._run_compare()
        elif intent == "source_explain":
            # 来源解释不重新检索，只解释上一次问答/总结留下的 sources。
            result = self.source_tool.run(self.memory.last_sources)
        elif intent == "export":
            # 导出格式从用户指令中解析；未写 JSON 时默认导出 Markdown。
            file_format = Exporter.pick_format(user_query)
            result = self.export_tool.run(self.memory, file_format=file_format)
            self.memory.last_export_path = result.artifacts.get("path", "")
        else:
            # 默认意图是论文问答。paper_id 为 None 时表示在全部论文中检索。
            result = self.qa_tool.run(user_query, paper_id=self.memory.current_paper_id)

        # Evaluator 是可信生成的最后一道门：检查是否没有来源、是否超过距离阈值等。
        checked = self.evaluator.check(result)

        # 把最终结果写回 Memory，后续“解释来源”“导出结果”都会读取这里。
        self.memory.remember_result(checked.answer, checked.sources)
        self.memory.add_turn("assistant", checked.answer)
        return checked

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
            if not card:
                # 如果某篇论文还没有卡片，就自动补生成。用户只需要点击“对比”，
                # Agent 会完成“补卡片 -> 汇总对比”的多步编排。
                summary = self.summary_tool.run(paper_id)
                checked_summary = self.evaluator.check(summary)
                if checked_summary.sources:
                    card = checked_summary.answer
                    self.memory.generated_cards[paper_id] = card
            if card:
                cards[paper_id] = card

        # 对比工具只接收结构化卡片，不直接混检多篇论文原文。
        return self.compare_tool.run(cards)
