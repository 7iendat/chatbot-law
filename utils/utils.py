# utils.py
import os
import logging
import regex as re
from tqdm import tqdm # Sử dụng tqdm thường thay vì tqdm.notebook
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter # Import lại
import fitz
from PIL import Image
import io
import pytesseract
import json
from typing import List, Optional
from pathlib import Path
from langchain_core.runnables import RunnableLambda, Runnable
from schemas.chat import ChatHistoryItem, ChatHistoryResponse
from redis.client import Redis
import bcrypt
from datetime import datetime, timedelta
from jose import jwt
from config import SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES
from langchain_core.runnables import Runnable
from typing import Dict, Any


def router_as_runnable(
    routes: dict[str, Runnable],
    get_key: Runnable,
    default: Runnable = None
) -> Runnable:
    def dispatch(input):
        # Kiểm tra xem get_key có thể trả về key hợp lệ không
        key = get_key.invoke(input)
        # In debug key để kiểm tra
        print(f"[Router] Route key: {key}")
        # Trả về route từ key, nếu không tìm thấy thì trả về default
        return routes.get(key, default)

    return RunnableLambda(dispatch).bind()


# Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_text_from_scanned_pdf(pdf_path: str, lang='vie') -> str:
    """
    Trích xuất văn bản từ file PDF scan bằng OCR (PyMuPDF + Tesseract OCR).
    Args:
        pdf_path (str): Đường dẫn đến file PDF.
        lang (str): Mã ngôn ngữ OCR (ví dụ: 'vie' cho tiếng Việt).
    Returns:
        str: Văn bản trích xuất từ file PDF.
    """
    from io import StringIO
    text_buffer = StringIO()

    try:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file: {pdf_path}")

        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        print(f"[OCR] Bắt đầu OCR cho {total_pages} trang...")

        # Thiết lập cấu hình OCR

        custom_config = f"--oem 3 --psm 6"

        for page_num, page in enumerate(doc, start=1):
            zoom = 300 / 72  # DPI cao hơn giúp nhận diện tốt hơn
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            img = Image.open(io.BytesIO(pix.tobytes("ppm")))

            try:
                page_text = pytesseract.image_to_string(img, config=custom_config, lang=lang)
                print(f"[OCR] Trang {page_num}/{total_pages}: {len(page_text.strip())} ký tự")
                text_buffer.write(page_text + "\n")

            except Exception as ocr_err:
                print(f"[OCR ERROR] Lỗi OCR trang {page_num}: {ocr_err}")
                continue

        doc.close()
        return text_buffer.getvalue().strip()

    except Exception as e:
        print(f"[ERROR] Không thể xử lý file PDF '{pdf_path}': {e}")
        return ""

def extract_text_from_pdf_auto(pdf_path: str, lang='vie') -> str:
    """Tự động trích xuất text từ PDF (thường hoặc scan)."""
    try:
        print(f"[DEBUG] Đang mở PDF: {pdf_path}")
        doc = fitz.open(pdf_path)
        text = ""
        for page_num, page in enumerate(doc):
            page_text = page.get_text().strip()
            print(f"[DEBUG] Trang {page_num + 1} có độ dài text: {len(page_text)}")
            text += page_text + "\n"
        doc.close()

        # Nếu text rất ít, có thể là file scan => dùng OCR
        if len(text.strip()) < 100:
            print(f"[DEBUG] Text rất ít ({len(text.strip())} ký tự), nghi là PDF scan, chuyển sang OCR...")
            text = extract_text_from_scanned_pdf(pdf_path, lang=lang)

        if not text or len(text.strip()) == 0:
            raise ValueError("Không thể trích xuất text từ PDF (text hoặc OCR đều rỗng).")

        return text

    except Exception as e:
        print(f"[ERROR] Lỗi khi auto-extract PDF: {e}")
        return None

# Khởi tạo text_splitter ở global scope để tránh tạo lại nhiều lần
base_text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500, # Cho các đoạn quá dài, giảm xuống
                chunk_overlap=100,
                length_function=len,
                add_start_index=False,
            )

def clean_text_optimized(text: str) -> str:
    """
    Làm sạch văn bản từ file .txt đã trích xuất, loại bỏ header/footer/noise, giữ lại nội dung quan trọng.
    """

    # --- 1. Loại bỏ khối lớn header/footer đặc biệt ---
    block_patterns = [
        r"^\s*.*?LỚN NHÁT VIỆT NAM.*?bàn dịch tiếng Anh\s*",
        r"^\s*Trung tâm LuafVietnam[\s\S]*?(?:lawdat\s*afl\s*u\s*atvistnarn\.vni|378536589|Email:.*?@luatvietnam\.vn)\s*$",
        r"^\s*:?\s*CƠ SỞ DỮ LIỆU VĂN BẢN PHÁP LUẬT\s*$",
    ]
    for pat in block_patterns:
        text = re.sub(pat, "", text, flags=re.MULTILINE | re.DOTALL | re.IGNORECASE)

    # --- 2. Loại bỏ dòng noise rõ ràng ---
    remove_line_patterns = [
        r"^(.*LuatWielnam.*|.*LuatVietnam\.vn.*|.*Tiện ích văn bản luật.*)$",
        r"^\s*\[\s*Hình\s*ảnh\s*]\s*$",
        r"^[=*_\-]{3,}$",  # Dòng toàn dấu gạch ngang
        r"^\s*(LuatW?ietnam|LỚN NHÁT VIỆT NAM|Tiện ích văn bản luật|www\.vanbanluat\.vn)\s*$",
        r"^\s*(teeeeokanlbaglueloen|Tee===|Tc=e===|nem|SN Hntlin sa:|HT:|Hntlin sa:).*",
        r"^\s*\d{1,3}(?:\.\d{3})*.*văn bản pháp luật.*tiếng Anh\s*$",
        r"^\s*(QUỐC HỘI|CỘNG HÒ?A\s+XÃ\s+HỘ?I\s+CHỦ?\s+NGHĨ?A\s+VIỆ?T\s+NAM)\s*$",
        r"^\s*[-—\s]*Độc lập\s*-\s*Tự do\s*-\s*Hạnh phúc\s*[-—\s]*$",
        r"^\s*(Luật|Nghị định|Bộ luật|Thông tư|Quyết định)\s+số:.*$",
        r"^\s*Số:.*(?:QH|NĐ-CP|TT-BTC).*",
        r"^\s*(Hà Nội|Tp\. Hồ Chí Minh),\s+ngày\s+\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4}\s*$",
        r"^\s*(Nguyễn Sinh Hùng|Nông Đức Mạnh|Nguyễn Tấn Dũng)\s*(\(Đã ký\))?\s*$",
        r"^\s*CHỦ TỊCH QUỐC HỘI\s*$",
        r"^\s*TM\. CHÍNH PHỦ\s*$",
        r"^\s*THỦ TƯỚNG\s*$"
    ]
    remove_line_regex = [re.compile(pat, flags=re.IGNORECASE) for pat in remove_line_patterns]

    # --- 3. Tiền xử lý dòng ---
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        line = line.strip()

        # Bỏ nếu trùng bất kỳ pattern noise nào
        if any(pat.match(line) for pat in remove_line_regex):
            continue

        # --- 4. Làm sạch heading và lỗi OCR ---
        line = re.sub(r"^\s*[-—_\s]*(CHƯƠNG\s+[IVXLCDM\d]+)\s*[-—_\s]*", r"\1", line, flags=re.IGNORECASE)
        line = re.sub(r"^\s*[-—_\s]*(Mục\s+\d+)\s*[-—_\s]*", r"\1", line, flags=re.IGNORECASE)
        line = re.sub(r"^\s*[-—_\s]*(PHẦN\s+(?:THỨ\s+[A-Z]+|CHUNG|CÁC TỘI PHẠM))\s*[-—_\s]*", r"\1", line, flags=re.IGNORECASE)

        # Bỏ ký tự lạ do OCR
        line = re.sub(r"[¡¬_`„´ˆ˜]", "", line)

        # Sửa lỗi email
        line = line.replace("aflu atvistnarn.vni", "@luatvietnam.vn")

        if line:
            cleaned_lines.append(line)

    # --- 5. Ghép lại văn bản ---
    final_text = "\n".join(cleaned_lines)

    # Xử lý xuống dòng thừa, khoảng trắng
    final_text = re.sub(r"\n{3,}", "\n\n", final_text)
    final_text = re.sub(r"[ \t]{2,}", " ", final_text)

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


def split_by_structure(doc: Document, max_chunk_size: int = 2000) -> list[Document]:
    """
    Chia văn bản luật thành các chunk dựa trên cấu trúc Điều, Khoản, Chương, Phần, Mục.
    Nếu chunk quá lớn, dùng RecursiveTextSplitter để chia nhỏ hơn.
    """
    text = doc.page_content
    source = doc.metadata.get("source", "N/A")
    lines = text.splitlines()

    chunks = []
    buffer = []
    current_meta = {"source": source}

    # Regex nhận diện
    patterns = {
        "phan": re.compile(r"^\s*(Phần\s+(?:thứ\s+[a-z]+|chung|các tội phạm))\s*[:.]?\s*(.*)", re.IGNORECASE),
        "chuong": re.compile(r"^\s*(Chương\s+[IVXLCDM\d]+)\s*[:.]?\s*(.*)", re.IGNORECASE),
        "muc": re.compile(r"^\s*(Mục\s+\d+)\s*[:.]?\s*(.*)", re.IGNORECASE),
        "dieu": re.compile(r"^\s*(Điều\s+\d+)\s*[:.]?\s*(.*)", re.IGNORECASE),
        "khoan": re.compile(r"^\s*(\d+)\.\s+(.*)", re.IGNORECASE),  # Chưa dùng, nhưng giữ lại để sau mở rộng
    }

    def update_meta(key: str, value: str):
        """Cập nhật metadata và reset các cấp dưới nếu cần"""
        nonlocal current_meta
        current_meta[key] = value
        if key == "phan":
            current_meta.pop("chuong", None)
            current_meta.pop("muc", None)
            current_meta.pop("dieu", None)
        elif key == "chuong":
            current_meta.pop("muc", None)
            current_meta.pop("dieu", None)
        elif key == "muc":
            current_meta.pop("dieu", None)

    def flush_chunk(force=False):
        """Lưu chunk hiện tại nếu có nội dung, và xử lý chunk dài"""
        nonlocal buffer, current_meta
        content = "\n".join(buffer).strip()
        if content or force:
            if len(content) > max_chunk_size:
                logger.warning(
                    f"Chunk từ '{current_meta.get('dieu', 'N/A')}' trong file '{source}' quá lớn ({len(content)} chars). Đang chia nhỏ..."
                )
                sub_docs = base_text_splitter.create_documents([content])
                for sub_doc in sub_docs:
                    sub_doc.metadata = current_meta.copy()
                chunks.extend(sub_docs)
            else:
                chunks.append(Document(page_content=content, metadata=current_meta.copy()))
        buffer = []

    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue  # Bỏ dòng trắng

        # Ưu tiên theo cấp: Phần > Chương > Mục > Điều
        matched = False
        for key in ["phan", "chuong", "muc", "dieu"]:
            match = patterns[key].match(line_stripped)
            if match:
                if key == "dieu":
                    flush_chunk()
                update_meta(key, match.group(1).strip())
                if key == "dieu":
                    buffer.append(line)  # Bắt đầu nội dung của Điều
                matched = True
                break

        if not matched:
            if buffer:
                buffer.append(line)

    flush_chunk()  # Lưu chunk cuối

    return chunks

def route_logic(input: dict) -> str:
    question = input.get("question", "").lower()
    print(f"[RouteLogic] Câu hỏi: {question}")  # Debug log
    if any(x in question for x in ["bạn là ai", "trợ lý", "chatbot", "ai", "giúp được gì", "có thể làm gì", "tên gì"]):
        return "general"  # Trả về "general" cho câu hỏi tổng quát
    else:
        return "legal"  # Trả về "legal" cho câu hỏi pháp lý


# === Redis helpers ===
def save_chat_to_redis(r:Redis, session_id: str, question: str, answer: str):
    item = json.dumps({"question": question, "answer": answer})
    r.rpush(f"chat:{session_id}", item)

def get_redis_history(r:Redis, session_id: str) -> List[ChatHistoryItem]:
    history_raw = r.lrange(f"chat:{session_id}", 0, -1)
    return [ChatHistoryItem(**json.loads(item)) for item in history_raw]


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.now() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


class WrappedLLMChain(Runnable):
    def __init__(self, chain):
        self.chain = chain

    def invoke(self, input: Dict[str, Any], config: dict = None, **kwargs) -> Dict[str, Any]:
        response = self.chain.invoke(input, config=config, **kwargs)
        return {"answer": response}
