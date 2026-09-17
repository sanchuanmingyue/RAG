"""Agent 结果质量检查。

Evaluator 对应方案里的“可信生成机制”。RAG 并不能天然保证没有幻觉，
因此生成后需要再检查：是否有来源、来源是否可信、是否应该触发拒答。
"""

from __future__ import annotations

from backend.config import settings
from backend.schemas import AgentResult


class ResultEvaluator:
    """检查 sources、距离阈值和拒答边界。

    当前版本做轻量检查：
    - 问答和总结没有 sources 时拒答；
    - 如果配置了 RETRIEVAL_MAX_DISTANCE，则过滤距离过大的来源；
    - 如果过滤后没有可用来源，也拒答。
    """

    REFUSAL = "论文中没有找到明确依据。"

    def check(self, result: AgentResult) -> AgentResult:
        """返回经过可信性检查后的 AgentResult。"""

        # qa/summary 都应该基于论文片段；如果没有来源，就不能让模型自由发挥。
        if result.intent in {"qa", "summary"} and not result.sources:
            result.answer = self.REFUSAL
            result.warnings.append("未检索到可用来源。")
            return result

        # retrieval_max_distance 默认为 0，表示不启用阈值过滤。
        # 不同 embedding 模型和向量距离算法数值范围不同，所以阈值需要实测后再配置。
        if settings.retrieval_max_distance <= 0 or not result.sources:
            return result

        # Chroma 返回 distance，通常越小越相似。没有 distance 的来源先保留，
        # 避免非向量工具结果被误删。
        usable_sources = [
            source
            for source in result.sources
            if not isinstance(source.get("distance"), (int, float))
            or source["distance"] <= settings.retrieval_max_distance
        ]
        if not usable_sources and result.intent in {"qa", "summary"}:
            # 如果所有来源都太远，说明检索结果可能不可靠，触发无依据拒答。
            result.answer = self.REFUSAL
            result.sources = []
            result.warnings.append("检索结果距离超过阈值，已触发无依据拒答。")
        else:
            result.sources = usable_sources

        return result
