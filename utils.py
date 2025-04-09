# utils.py
import os
import regex as re
from tqdm import tqdm # Sử dụng tqdm thường thay vì tqdm.notebook
from langchain_core.documents import Document

def clean_text_optimized(text):
    """Làm sạch văn bản từ file .txt đã trích xuất, loại bỏ header/footer/noise."""
    # 1. Loại bỏ các khối header/footer nhiều dòng trước
    text = re.sub(r"^\s*.*?LỚN NHÁT VIỆT NAM.*?bàn dịch tiếng Anh\s*", "", text, flags=re.MULTILINE | re.DOTALL | re.IGNORECASE)
    text = re.sub(r"^\s*Trung tâm LuafVietnam[\s\S]*?(?:lawdat\s*afl\s*u\s*atvistnarn\.vni|378536589|Email:.*?@luatvietnam\.vn)\s*$", "", text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r"^\s*:?\s*CƠ SỞ DỮ LIỆU VĂN BẢN PHÁP LUẬT\s*$", "", text, flags=re.MULTILINE | re.IGNORECASE)

    # 2. Xử lý từng dòng
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        # Các regex loại bỏ dòng như trong code Colab Cell 2
        if re.match(r"^\s*(?:LuatW?ietnam|LỚN NHÁT VIỆT NAM|Tiện ích văn bản luật|www\.vanbanluat\.vn)\s*$", line, re.IGNORECASE): continue
        if re.match(r"^\s*(?:teeeeokanlbaglueloen|Tee===|Tc=e===|nem|SN Hntlin sa:|HT:|Hntlin sa:).*", line, re.IGNORECASE): continue
        if re.match(r"^\s*\d{1,3}(?:\.\d{3})*.*văn bản pháp luật.*tiếng Anh\s*$", line, re.IGNORECASE): continue
        if re.match(r"^\s*(?:QUỐC HỘI|CỘNG HÒ?A\s+XÃ\s+HỘ?I\s+CHỦ?\s+NGHĨ?A\s+VIỆ?T\s+NAM)\s*$", line, re.IGNORECASE): continue
        if re.match(r"^\s*[-—\s]*Độc lập\s*-\s*Tự do\s*-\s*Hạnh phúc\s*[-—\s]*$", line, re.IGNORECASE): continue
        if re.match(r"^\s*(?:Luật|Nghị định|Bộ luật|Thông tư|Quyết định)\s+số:.*$", line, re.IGNORECASE): continue
        if re.match(r"^\s*Số:.*(?:QH|NĐ-CP|TT-BTC).*", line, re.IGNORECASE): continue
        if re.match(r"^\s*(?:Hà Nội|Tp\. Hồ Chí Minh),\s+ngày\s+\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4}\s*$", line, re.IGNORECASE): continue
        if re.match(r"^\s*(?:Nguyễn Sinh Hùng|Nông Đức Mạnh|Nguyễn Tấn Dũng)\s*(?:\(Đã ký\))?\s*$", line, flags=re.IGNORECASE): continue
        if re.match(r"^\s*CHỦ TỊCH QUỐC HỘI\s*$", line, flags=re.IGNORECASE): continue
        if re.match(r"^\s*TM\. CHÍNH PHỦ\s*$", line, flags=re.IGNORECASE): continue
        if re.match(r"^\s*THỦ TƯỚNG\s*$", line, flags=re.IGNORECASE): continue

        line = re.sub(r"^\s*[-—_\s]*(CHƯƠNG\s+[IVXLCDM\d]+)\s*[-—_\s]*", r"\1", line, flags=re.IGNORECASE)
        line = re.sub(r"^\s*[-—_\s]*(Mục\s+\d+)\s*[-—_\s]*", r"\1", line, flags=re.IGNORECASE)
        line = re.sub(r"^\s*[-—_\s]*(PHẦN\s+(?:THỨ\s+[A-Z]+|CHUNG|CÁC TỘI PHẠM))\s*[-—_\s]*", r"\1", line, flags=re.IGNORECASE)
        line = re.sub(r"[¡¬_`„´ˆ˜]", "", line)
        line = line.replace("aflu atvistnarn.vni", "@luatvietnam.vn")

        if line:
            cleaned_lines.append(line)

    final_text = "\n".join(cleaned_lines)
    final_text = re.sub(r'\n{3,}', '\n\n', final_text)
    final_text = re.sub(r'[ \t]{2,}', ' ', final_text)
    return final_text.strip()

def load_and_clean_documents(folder_path):
    """Đọc các file .txt từ thư mục, làm sạch và trả về list các Document."""
    all_docs = []
    if not os.path.isdir(folder_path):
        print(f"Lỗi: Thư mục '{folder_path}' không tồn tại.")
        return all_docs

    txt_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.txt')]
    if not txt_files:
        print(f"Lỗi: Không tìm thấy file .txt nào trong '{folder_path}'.")
        return all_docs

    print(f"Tìm thấy {len(txt_files)} file .txt. Đang đọc và làm sạch...")
    for filename in tqdm(txt_files, desc="Đọc và Làm sạch file txt"):
        file_path = os.path.join(folder_path, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_content = f.read()
            cleaned_content = clean_text_optimized(raw_content)
            if cleaned_content:
                doc = Document(page_content=cleaned_content, metadata={"source": os.path.basename(file_path)})
                all_docs.append(doc)
            else:
                 print(f"\nCảnh báo: File '{filename}' rỗng sau khi làm sạch.")
        except Exception as e:
            print(f"\nLỗi khi đọc/làm sạch file '{filename}': {e}")

    print(f"Đã xử lý và làm sạch {len(all_docs)} tài liệu.")
    return all_docs