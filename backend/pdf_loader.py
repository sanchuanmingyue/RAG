"""PDF 解析模块：按页提取文本，并保留页码信息。"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any


def normalize_paper_id(file_name: str) -> str:
    """把文件名转换成适合作为 paper_id 的稳定字符串。"""

    stem = Path(file_name).stem
    paper_id = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "_", stem).strip("_")
    return paper_id or "paper"


def clean_text(text: str) -> str:
    """做轻量清洗，去掉多余空白。"""

    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf_pages(pdf_path: str | Path, paper_id: str | None = None) -> list[dict[str, Any]]:
    """读取 PDF，每页返回一个包含 paper_id、file_name、page、text 的字典。"""

    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise ImportError("请先安装 PyMuPDF：pip install pymupdf") from exc

    path = Path(pdf_path)
    current_paper_id = paper_id or normalize_paper_id(path.name)
    pages: list[dict[str, Any]] = []

    with fitz.open(path) as doc:
        for index, page in enumerate(doc, start=1):
            text = clean_text(page.get_text("text"))
            if not text:
                continue
            pages.append(
                {
                    "paper_id": current_paper_id,
                    "file_name": path.name,
                    "page": index,
                    "text": text,
                }
            )

    return pages
