"""Agent 层的轻量数据结构。

把数据结构单独放在 schemas.py，可以避免各模块互相依赖具体实现。
Router、Tool、Evaluator、Agent 都围绕这些结构通信，后续增加新工具时也更容易维护。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


# AgentIntent 限定 Router 只能返回这些合法意图。
# 这样编辑器和类型检查工具可以更早发现拼写错误，例如把 summary 写成 sumarize。
AgentIntent = Literal["qa", "summary", "compare", "source_explain", "export"]


@dataclass
class AgentResult:
    """工具和 Agent 统一返回结构。

    不同工具的原始返回值可能不同：问答有 answer/sources，总结有 note/sources，
    导出有文件路径。统一成 AgentResult 后，Agent 和前端就可以用同一种方式处理。
    """

    # 最终要展示给用户的文本答案。
    answer: str

    # 本次结果来自哪个意图，方便 Evaluator 做不同检查，也方便前端调试。
    intent: AgentIntent

    # RAG 检索到的来源片段。问答、总结通常需要 sources；导出、对比可以为空。
    sources: list[dict[str, Any]] = field(default_factory=list)

    # 非文本产物，例如导出路径、对比卡片数量等。
    artifacts: dict[str, Any] = field(default_factory=dict)

    # 质量检查或工具执行中的提示，不一定是错误，但需要展示给用户。
    warnings: list[str] = field(default_factory=list)


@dataclass
class ChatTurn:
    """保存一条会话消息。"""

    # role 与 Streamlit chat_message 保持一致，通常为 user 或 assistant。
    role: str
    content: str
