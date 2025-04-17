# custom_output_parser.py

from langchain.schema import BaseOutputParser
import re

class CustomOutputParser(BaseOutputParser):
    def parse(self, text: str) -> dict:
        """
        Hàm này nhận vào đầu ra từ LLM và trả về một dictionary với các phần đã được tùy chỉnh:
        - raw: văn bản ban đầu.
        - markdown: văn bản đã được chuyển đổi thành markdown.
        - highlight: văn bản đã được tô đậm các điều luật (nếu có).
        """
        raw_output = text
        markdown_output = self.convert_to_markdown(text)
        highlighted_output = self.highlight_laws(text)

        return {
            "raw": raw_output,
            "markdown": markdown_output,
            "highlight": highlighted_output
        }

    def convert_to_markdown(self, text: str) -> str:
        """
        Chuyển đổi văn bản thành markdown (ví dụ: thêm tiêu đề, danh sách, các đoạn mã).
        """
        # Đơn giản chỉ là chuyển đổi xuống dòng thành <br> để hiển thị tốt hơn trong Markdown
        return text.replace("\n", "<br>")

    def highlight_laws(self, text: str) -> str:
        """
        Tô đậm các Điều luật trong văn bản. Ví dụ: 'Điều 12' sẽ được tô đậm.
        """
        return re.sub(r"(Điều\s+\d+)", r"**\1**", text)
