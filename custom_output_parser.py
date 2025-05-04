
from langchain.schema import BaseOutputParser
import re
from typing import Union

class CustomOutputParser(BaseOutputParser):
    def parse(self, text: Union[str, dict]) -> dict:
        """
        Parse đầu ra từ LLM, có thể là string hoặc dict.
        Trả về:
        - raw: gốc
        - markdown: có định dạng <br>
        - highlight: tô đậm Điều, Khoản, Mục, Chương...
        """
        if isinstance(text, dict):
            text = text.get("answer", "")

        if not isinstance(text, str):
            raise ValueError("Input to parser must be string or dict with 'answer' key.")

        raw_output = text.strip()
        markdown_output = self.convert_to_markdown(raw_output)
        highlighted_output = self.highlight_legal_terms(markdown_output)

        return {
            "raw": raw_output,
            "markdown": markdown_output,
            "highlight": highlighted_output
        }

    def convert_to_markdown(self, text: str) -> str:
        """
        Chuyển văn bản sang markdown với dòng cách (thay \n bằng <br>).
        """
        return "<br>".join(line.strip() for line in text.splitlines() if line.strip())

    def highlight_legal_terms(self, text: str) -> str:
        """
        Tô đậm các từ khóa pháp lý như: Điều, Khoản, Mục, Chương.
        """
        keywords = [r"(Điều\s+\d+)", r"(Khoản\s+\d+)", r"(Mục\s+\d+)", r"(Chương\s+\w+)"]
        for kw in keywords:
            text = re.sub(kw, r"**\1**", text, flags=re.IGNORECASE)
        return text
