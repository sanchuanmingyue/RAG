"""Context-aware follow-up detection, intent inheritance, and query rewriting."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

from backend.memory import ConversationState
from backend.router import AgentRouter


_CONTINUATION_MARKERS = (
    "继续",
    "再详细",
    "再找",
    "再搜",
    "详细一点",
    "具体一点",
    "展开",
    "具体呢",
    "为什么",
    "怎么实现",
    "如何实现",
    "重新",
    "分得更细",
    "再分类",
    "那",
    "它",
    "这些",
    "上述",
    "刚才",
    "上一",
)
_CHINESE_NUMBERS = {
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6,
    "七": 7, "八": 8, "九": 9, "十": 10,
}


@dataclass(frozen=True)
class FollowUpResolution:
    is_followup: bool
    intent: str
    followup_type: str = "new_query"
    operation: str | None = None
    target_category: int | None = None
    rewritten_query: str | None = None
    confidence: float = 1.0


def _category_number(text: str) -> int | None:
    match = re.search(r"第\s*(\d{1,2}|[一二三四五六七八九十])\s*类", text)
    if not match:
        return None
    value = match.group(1)
    return int(value) if value.isdigit() else _CHINESE_NUMBERS.get(value)


def _json_object(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except (json.JSONDecodeError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None


class ConversationResolver:
    """Resolve new queries and follow-ups using rules first and a small LLM fallback."""

    def __init__(self, llm_client: Any | None = None) -> None:
        self.llm_client = llm_client
        self.router = AgentRouter()

    def resolve(self, query: str, state: ConversationState) -> FollowUpResolution:
        clean_query = " ".join(query.strip().split())
        explicit_intent = self.router.route(clean_query)
        previous_intent = state.last_intent
        if not previous_intent:
            return FollowUpResolution(False, explicit_intent, rewritten_query=clean_query)

        corpus_rule = self._resolve_corpus_operation(clean_query, state)
        if corpus_rule is not None:
            return corpus_rule

        # An explicit non-QA command starts or continues its own workflow and
        # must win over generic words such as "继续".
        if explicit_intent != "qa":
            is_followup = explicit_intent == previous_intent and self._looks_like_followup(clean_query)
            rewritten = self._rewrite(clean_query, state, explicit_intent) if is_followup else clean_query
            return FollowUpResolution(
                is_followup,
                explicit_intent,
                "modify_conditions" if is_followup else "new_query",
                rewritten_query=rewritten,
            )

        if self._looks_like_followup(clean_query):
            return FollowUpResolution(
                True,
                previous_intent,
                "content_deepen",
                rewritten_query=self._rewrite(clean_query, state, previous_intent),
                confidence=0.95,
            )

        if self._potentially_contextual(clean_query):
            llm_resolution = self._resolve_with_llm(clean_query, state)
            if llm_resolution is not None:
                return llm_resolution
        return FollowUpResolution(False, explicit_intent, rewritten_query=clean_query)

    def _resolve_corpus_operation(
        self, query: str, state: ConversationState
    ) -> FollowUpResolution | None:
        has_classification = (
            state.last_intent == "corpus_analysis"
            or state.active_artifact_type == "corpus_analysis"
        )
        if not has_classification:
            return None
        target = _category_number(query)
        if target and any(word in query for word in ("列出", "有哪些", "包括", "论文", "文章")):
            if not any(word in query for word in ("细分", "再分", "分成", "分为")):
                return FollowUpResolution(
                    True, "corpus_analysis", "artifact_operation", "list_category", target, query
                )
        if target and any(word in query for word in ("细分", "再分", "分成", "分为", "分类")):
            return FollowUpResolution(
                True, "corpus_analysis", "artifact_operation", "subdivide_category", target, query
            )
        if any(word in query for word in ("为什么这样分类", "分类依据", "怎么分类", "如何分类")):
            return FollowUpResolution(
                True, "corpus_analysis", "artifact_operation", "explain_classification", None, query
            )
        if "每类" in query and any(word in query for word in ("介绍", "说明", "展开", "详情")):
            return FollowUpResolution(
                True, "corpus_analysis", "artifact_operation", "explain_classification", None, query
            )
        if any(word in query for word in ("再分类", "分得更细", "分类详细", "更细", "细一点")):
            return FollowUpResolution(
                True, "corpus_analysis", "artifact_operation", "refine_classification", None, query
            )
        return None

    @staticmethod
    def _looks_like_followup(query: str) -> bool:
        normalized = query.lower()
        return len(normalized) <= 80 and any(marker in normalized for marker in _CONTINUATION_MARKERS)

    @staticmethod
    def _potentially_contextual(query: str) -> bool:
        if len(query) > 60:
            return False
        return any(
            marker in query
            for marker in ("这个", "该", "其", "其中", "每类", "哪类", "前者", "后者", "呢", "上述")
        )

    def _rewrite(self, query: str, state: ConversationState, intent: str) -> str:
        if intent == "corpus_analysis":
            return query
        if intent == "literature_search":
            return f"原检索需求：{state.last_user_query}；追加条件：{query}"
        if intent == "library_status":
            return f"上一轮知识库问题：{state.last_user_query}；当前追问：{query}"

        fallback = f"上一轮问题：{state.last_user_query}\n当前追问：{query}"
        if self.llm_client is None:
            return fallback
        prompt = f"""把多轮论文问答中的当前追问改写成一个可独立检索的问题。
只补足被省略的主题和指代，不增加事实，不回答问题，只输出改写后的问题。
上一轮问题：{state.last_user_query}
上一轮答案摘要：{state.last_answer_summary}
当前追问：{query}"""
        try:
            rewritten = self.llm_client.chat(
                [{"role": "user", "content": prompt}], temperature=0.0, max_tokens=220
            ).strip()
        except Exception:
            return fallback
        return rewritten[:1200] if rewritten else fallback

    def _resolve_with_llm(
        self, query: str, state: ConversationState
    ) -> FollowUpResolution | None:
        if self.llm_client is None or len(query) > 120:
            return None
        prompt = f"""判断当前输入是否是上一轮科研论文任务的追问。只输出 JSON：
{{"is_followup":true,"intent":"qa","followup_type":"content_deepen","rewritten_query":"完整独立问题","confidence":0.9}}
合法 intent：qa, summary, compare, source_explain, export, library_status, literature_search, corpus_analysis。
上一轮意图：{state.last_intent}
上一轮问题：{state.last_user_query}
上一轮答案摘要：{state.last_answer_summary}
当前输入：{query}"""
        try:
            raw = self.llm_client.chat(
                [{"role": "user", "content": prompt}], temperature=0.0, max_tokens=350
            )
            payload = _json_object(raw)
        except Exception:
            return None
        if not payload or not payload.get("is_followup"):
            return None
        intent = str(payload.get("intent") or state.last_intent or "qa")
        if intent not in {
            "qa", "summary", "compare", "source_explain", "export",
            "library_status", "literature_search", "corpus_analysis",
        }:
            return None
        confidence = float(payload.get("confidence") or 0.0)
        if confidence < 0.8:
            return None
        rewritten = str(payload.get("rewritten_query") or "").strip() or self._rewrite(
            query, state, intent
        )
        return FollowUpResolution(
            True,
            intent,
            str(payload.get("followup_type") or "content_deepen"),
            rewritten_query=rewritten,
            confidence=confidence,
        )
