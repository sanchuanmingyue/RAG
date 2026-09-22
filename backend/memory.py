"""Agent 会话记忆。

Memory 对应方案里的“会话状态”。它让系统知道当前正在看哪篇论文、
用户选择了哪些论文、上一次回答用了哪些来源、哪些论文已经生成过阅读卡片。
这些信息使 Agent 能支持“继续”“解释刚才来源”“导出刚才结果”等多轮任务。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from backend.schemas import ChatTurn


@dataclass
class ConversationState:
    """Compact state used for routing follow-ups without replaying full chat history."""

    last_intent: str | None = None
    last_user_query: str = ""
    rewritten_query: str | None = None
    active_document_id: str | None = None
    active_section: str | None = None
    active_artifact_id: str | None = None
    active_artifact_type: str | None = None
    active_artifact: dict[str, Any] = field(default_factory=dict)
    selected_category: int | None = None
    filters: dict[str, Any] = field(default_factory=dict)
    last_answer_summary: str = ""


@dataclass
class AgentMemory:
    """保存当前论文、已选论文、历史问题和中间结果。

    这个对象被放在 Streamlit 的 session_state 中，因此同一次浏览器会话里，
    用户切换页面或继续追问时，状态不会立刻丢失。
    """

    # 当前单篇论文范围。为 None 时表示“全部论文”。
    current_paper_id: str | None = None

    # 多论文对比范围，由侧边栏 multiselect 维护。
    selected_paper_ids: list[str] = field(default_factory=list)

    # 已生成的阅读卡片缓存：paper_id -> card markdown。
    # 对比时优先复用缓存，避免重复调用总结工具。
    generated_cards: dict[str, str] = field(default_factory=dict)

    # 最近一次 Agent 最终回答。导出和来源解释都依赖这些字段。
    last_answer: str = ""
    last_sources: list[dict[str, Any]] = field(default_factory=list)
    last_export_path: str = ""

    # Router/Resolver 使用短状态判断追问；完整历史不直接塞进路由 Prompt。
    state: ConversationState = field(default_factory=ConversationState)

    # 完整聊天历史，导出 Markdown/JSON 时会写入。
    history: list[ChatTurn] = field(default_factory=list)

    def add_turn(self, role: str, content: str) -> None:
        """追加一条会话消息。role 通常是 user 或 assistant。"""

        self.history.append(ChatTurn(role=role, content=content))

    def set_scope(self, current_paper_id: str | None, selected_paper_ids: list[str] | None = None) -> None:
        """更新当前论文范围和多论文选择范围。

        Streamlit 每次交互都会重新执行脚本，所以侧边栏选择变化后需要把新的
        范围同步进 Memory。
        """

        self.current_paper_id = current_paper_id
        self.selected_paper_ids = selected_paper_ids or ([] if current_paper_id is None else [current_paper_id])
        self.state.active_document_id = current_paper_id

    def remember_result(
        self,
        answer: str,
        sources: list[dict[str, Any]],
        *,
        intent: str | None = None,
        user_query: str | None = None,
        rewritten_query: str | None = None,
        artifacts: dict[str, Any] | None = None,
    ) -> None:
        """保存最近一次最终结果，供来源解释和导出使用。"""

        self.last_answer = answer
        self.last_sources = sources
        if intent:
            self.state.last_intent = intent
        if user_query is not None:
            self.state.last_user_query = user_query
        self.state.rewritten_query = rewritten_query
        self.state.last_answer_summary = self._summarize_answer(answer, artifacts or {})

        artifact_type, artifact = self._active_artifact(artifacts or {})
        if artifact:
            self.state.active_artifact_type = artifact_type
            self.state.active_artifact = artifact
            self.state.active_artifact_id = str(
                artifact.get("artifact_id") or f"{artifact_type}_{len(self.history)}"
            )
            if artifact_type == "search":
                plan = artifact.get("query_plan") or {}
                self.state.filters = {
                    key: plan.get(key)
                    for key in ("start_year", "end_year")
                    if plan.get(key) is not None
                }

    @staticmethod
    def _active_artifact(artifacts: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
        for key in ("corpus_analysis", "search"):
            value = artifacts.get(key)
            if isinstance(value, dict):
                return key, value
        return None, {}

    @staticmethod
    def _summarize_answer(answer: str, artifacts: dict[str, Any]) -> str:
        corpus = artifacts.get("corpus_analysis")
        if isinstance(corpus, dict):
            labels = [
                f"{item.get('rank')}. {item.get('label')}（{item.get('count')}篇）"
                for item in corpus.get("categories") or []
            ]
            return (
                f"上一轮对{corpus.get('paper_count', 0)}篇论文进行了研究背景分类，"
                f"共{corpus.get('category_count', 0)}类：" + "；".join(labels)
            )[:1200]
        search = artifacts.get("search")
        if isinstance(search, dict):
            query = (search.get("query_plan") or {}).get("querytext") or ""
            return f"上一轮使用检索式“{query}”找到{search.get('returned', 0)}篇 IEEE 文献。"
        compact = " ".join(answer.split())
        return compact[:800]
