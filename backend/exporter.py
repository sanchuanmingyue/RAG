"""导出 Agent 结果。

导出功能把 Agent Memory 中的“最近回答、来源片段、阅读卡片、会话记录”
写成文件。它让系统不只是一个在线 Demo，而是能融入实际科研工作流。
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import re
from typing import Any

from backend.config import ROOT_DIR
from backend.memory import AgentMemory
from backend.rag_chain import format_sources


EXPORT_DIR = ROOT_DIR / "storage" / "exports"


class Exporter:
    """支持把最近一次 Agent 结果导出为 Markdown 或 JSON。

    Markdown 适合直接阅读和整理笔记；JSON 适合后续程序继续处理，例如生成
    Word、Excel 或构建评估集。
    """

    def export_memory(self, memory: AgentMemory, file_format: str = "markdown") -> Path:
        """根据 file_format 选择导出格式，并返回生成的文件路径。"""

        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        normalized_format = file_format.lower().strip()
        if normalized_format in {"md", "markdown"}:
            return self._export_markdown(memory)
        if normalized_format == "json":
            return self._export_json(memory)
        raise ValueError("当前仅支持 Markdown 和 JSON 导出。")

    def _export_markdown(self, memory: AgentMemory) -> Path:
        """导出为适合人阅读的 Markdown 文档。"""

        path = EXPORT_DIR / f"paper_agent_{self._timestamp()}.md"
        # lines 用列表逐步拼接，最后 join，避免频繁字符串相加。
        lines = [
            "# 科研论文阅读 Agent 结果",
            "",
            "## 最近回答",
            "",
            memory.last_answer or "暂无回答。",
            "",
        ]

        if memory.last_sources:
            # 来源片段保留文件名、页码和原文摘要，方便用户回到论文核查。
            lines.extend(["## 来源片段", ""])
            for index, source in enumerate(format_sources(memory.last_sources), start=1):
                lines.append(f"{index}. {source}")
            lines.append("")

        if memory.generated_cards:
            # 阅读卡片是多论文对比和文献综述的中间产物，导出时一并保留。
            lines.extend(["## 已生成阅读卡片", ""])
            for paper_id, card in memory.generated_cards.items():
                lines.extend([f"### {paper_id}", "", card, ""])

        if memory.history:
            # 会话记录用于复盘用户问了什么、系统如何回答。
            lines.extend(["## 会话记录", ""])
            for turn in memory.history:
                role = "用户" if turn.role == "user" else "助手"
                lines.extend([f"### {role}", "", turn.content, ""])

        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def _export_json(self, memory: AgentMemory) -> Path:
        """导出为结构化 JSON，便于程序二次处理。"""

        path = EXPORT_DIR / f"paper_agent_{self._timestamp()}.json"
        payload: dict[str, Any] = {
            "current_paper_id": memory.current_paper_id,
            "selected_paper_ids": memory.selected_paper_ids,
            "last_answer": memory.last_answer,
            "last_sources": memory.last_sources,
            "generated_cards": memory.generated_cards,
            "history": [turn.__dict__ for turn in memory.history],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    @staticmethod
    def pick_format(user_query: str) -> str:
        """从用户指令中推断导出格式。没有明确写 JSON 时默认 Markdown。"""

        return "json" if re.search(r"\bjson\b", user_query, flags=re.IGNORECASE) else "markdown"

    @staticmethod
    def _timestamp() -> str:
        """生成文件名中的时间戳，避免多次导出互相覆盖。"""

        return datetime.now().strftime("%Y%m%d_%H%M%S")
