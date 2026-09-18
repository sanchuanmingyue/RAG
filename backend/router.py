"""Agent Router：根据用户输入选择论文工具。

Router 对应方案里的“任务路由”。它把用户自然语言指令映射成固定意图，
例如 qa、summary、compare、source_explain、export。后面的 Agent 根据这个
意图决定调用哪个 Tool。
"""

from __future__ import annotations

import re

from backend.ieee_search import is_literature_search_query
from backend.schemas import AgentIntent


_LIBRARY_SCOPE_KEYWORDS = (
    "知识库",
    "向量库",
    "当前库",
    "库中",
    "库里",
    "已索引",
    "索引状态",
    "collection",
)
_LIBRARY_ACTION_KEYWORDS = (
    "多少",
    "几篇",
    "数量",
    "统计",
    "列出",
    "列表",
    "哪些",
    "有什么",
    "是否",
    "状态",
    "chunk",
    "文件",
)


def is_library_status_query(text: str) -> bool:
    """Detect metadata/status questions that should not enter semantic RAG."""

    normalized = " ".join(text.lower().strip().split())
    has_scope = any(keyword in normalized for keyword in _LIBRARY_SCOPE_KEYWORDS)
    has_action = any(keyword in normalized for keyword in _LIBRARY_ACTION_KEYWORDS)
    uploaded_count = (
        any(keyword in normalized for keyword in ("上传了", "导入了", "收录了"))
        and any(keyword in normalized for keyword in ("多少篇", "几篇", "哪些论文", "论文列表"))
    )
    return (has_scope and has_action) or uploaded_count


def is_corpus_analysis_query(text: str) -> bool:
    """Detect requests that require analyzing every paper rather than Top-K retrieval."""

    normalized = " ".join(text.lower().strip().split())
    has_action = any(
        keyword in normalized
        for keyword in ("分类", "归类", "聚类", "分组", "分成", "分为", "划分")
    )
    has_subject = any(keyword in normalized for keyword in ("论文", "文章", "文献"))
    has_batch_scope = any(
        keyword in normalized
        for keyword in ("全部", "所有", "全库", "当前库", "知识库", "库中", "库里", "这些", "这批", "多篇")
    ) or bool(re.search(r"\d+\s*篇", normalized))
    return has_action and has_subject and has_batch_scope


class AgentRouter:
    """规则优先的轻量 Router。

    科研论文阅读任务的意图边界比较清楚，用规则可以降低额外 LLM 调用成本，
    也方便面试时解释“为什么系统会调用某个工具”。
    """

    # 关键词顺序很重要：更具体的意图放前面。
    # 例如“导出对比结果”同时包含“导出”和“对比”，实际应该优先执行导出。
    EXPORT_KEYWORDS = ("导出", "保存", "下载", "生成文件", "markdown", "json", "export")
    COMPARE_KEYWORDS = ("对比", "比较", "差异", "共同点", "多篇", "文献综述", "综述", "compare")
    SUMMARY_KEYWORDS = ("总结", "概括", "阅读笔记", "阅读卡片", "论文卡片", "创新点", "实验设置")
    SOURCE_KEYWORDS = ("来源", "引用", "证据", "原文", "片段", "页码", "依据", "source")

    def route(self, user_query: str) -> AgentIntent:
        """返回用户输入对应的任务意图。"""

        # lower 主要服务英文关键词；中文不受影响。strip 去掉首尾空白。
        normalized = user_query.lower().strip()

        if self._has_any(normalized, self.EXPORT_KEYWORDS):
            return "export"
        if is_corpus_analysis_query(normalized):
            return "corpus_analysis"
        if is_literature_search_query(normalized):
            return "literature_search"
        if is_library_status_query(normalized):
            return "library_status"
        if self._has_any(normalized, self.COMPARE_KEYWORDS):
            return "compare"
        if self._has_any(normalized, self.SOURCE_KEYWORDS):
            return "source_explain"
        if self._has_any(normalized, self.SUMMARY_KEYWORDS):
            return "summary"
        # 如果没有命中特殊任务，就按最常见的论文问答处理。
        return "qa"

    @staticmethod
    def _has_any(text: str, keywords: tuple[str, ...]) -> bool:
        """判断文本中是否包含任一关键词。"""

        return any(keyword in text for keyword in keywords)
