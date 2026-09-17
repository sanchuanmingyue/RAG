"""Agent Router：根据用户输入选择论文工具。

Router 对应方案里的“任务路由”。它把用户自然语言指令映射成固定意图，
例如 qa、summary、compare、source_explain、export。后面的 Agent 根据这个
意图决定调用哪个 Tool。
"""

from __future__ import annotations

from backend.schemas import AgentIntent


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
