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
from typing import List, Dict, Tuple, Optional, Set, Any
from data.dic.dic import VIETNAMESE_DICTIONARY, THREE_WORD_PHRASES
import unicodedata
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from collections import defaultdict, Counter
# from time_priority_retriever import YearPriorityRetriever
# from langchain.retrievers.multi_query import MultiQueryRetriever



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

def extract_year_from_text(text: str) -> int | None:
    """
    Trích xuất năm từ nội dung văn bản với các mẫu thông dụng.
    """
    patterns = [
        r"\b(?:số[:\s]*)?\d{1,3}/(19|20)\d{2}/[A-Z]+\d*\b",             # Số 10/2012/QH13
        r"\b(?:số[:\s]*)?\d{1,3}[-–]\d{1,2}[-–](19|20)\d{2}\b",          # Số 23 - 15-12-1992
        r"\b(?:năm|ngày\s+\d{1,2}\s+tháng\s+\d{1,2}\s+năm)\s+(19|20)\d{2}\b",  # ngày 12 tháng 6 năm 1999
        r"\b(19|20)\d{2}\b",  # fallback chung
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1))  # nhóm 1 là năm
    return None
def extract_year_from_filename(filename: str) -> int | None:
    match = re.search(r"(19|20)\d{2}", filename)
    return int(match.group()) if match else None
def extract_years_from_vectorstore(vectorstore):
    all_docs = vectorstore.similarity_search(" ", k=1000)  # hoặc dùng cách khác để lấy toàn bộ doc
    years = []
    for doc in all_docs:
        year = doc.metadata.get("year")
        if isinstance(year, int):
            years.append(year)
    year_counts = Counter(years)
    sorted_years = sorted(year_counts, key=lambda y: (-year_counts[y], -y))
    return sorted_years
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
        year = extract_year_from_filename(filename)
        if not year:
            year = extract_year_from_text(raw_content)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_content = f.read()
            cleaned_content = clean_text_optimized(raw_content)
            if cleaned_content:
                doc = Document(page_content=cleaned_content, metadata={"source": os.path.basename(file_path), "year": year})
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
    current_meta = {"source": source, "year": doc.metadata.get("year", None)}

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
                flush_chunk()  # Lưu đoạn trước khi sang phần mới
                update_meta(key, match.group(1).strip())
                buffer.append(line)
                matched = True
                break


        if not matched:
            if buffer:
                buffer.append(line)

    flush_chunk()  # Lưu chunk cuối

    return chunks

def route_logic(input: dict) -> str:
    # Kiểm tra xem input có phải là dict không
    if not isinstance(input, dict):
        raise ValueError("Input phải là một dictionary.")

    question = input.get("question", "").lower()
    print(f"[RouteLogic] Câu hỏi: {question}")  # Debug log

    if any(x in question for x in ["bạn là ai", "trợ lý", "chatbot", "ai", "giúp được gì", "có thể làm gì", "tên gì"]):
        return "general"  # Trả về "general" cho câu hỏi tổng quát
    else:
        return "legal"  # Trả về "legal" cho câu hỏi pháp lý


# === Redis helpers ===
def save_chat_to_redis(r:Redis, chat_id: str, question: str, answer: str):
    item = json.dumps({"question": question, "answer": answer})
    r.rpush(f"chat:{chat_id}:messages", item)

def get_redis_history(r: Redis, chat_id: str) -> List[ChatHistoryItem]:
    try:
        history_raw = r.lrange(f"chat:{chat_id}:messages", 0, -1)
        chat_history = []
        for item in history_raw:
            try:
                parsed = json.loads(item)
                if isinstance(parsed, dict):
                    q = parsed.get("question", "")
                    a = parsed.get("answer", "")
                    if q and a:
                        chat_history.append(ChatHistoryItem(question=q, answer=a))
            except Exception as e:
                logger.error(f"Error parsing chat item: {e}")
        return chat_history
    except Exception as e:
        logger.error(f"Error fetching history from Redis: {e}")
        raise

def delete_chat_from_redis(r: Redis, chat_id: str):
    # Xóa cả metadata và messages
    r.delete(f"chat:{chat_id}:messages")
    r.delete(f"chat:{chat_id}:meta")

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


# === Vietnamese text processing ===
def remove_vietnamese_accents(text: str) -> str:
    """
    Remove all Vietnamese diacritical marks from a text.
    """
    text = unicodedata.normalize('NFD', text)
    text = re.sub(r'[\u0300-\u036f]', '', text)
    # Replace đ/Đ with d/D
    text = text.replace('đ', 'd').replace('Đ', 'D')
    return text

def build_unaccented_dictionary() -> Dict[str, str]:
    """
    Build a dictionary mapping unaccented forms to accented forms.
    """
    unaccented_dict = {}

    # Process main dictionary
    for accented, unaccented in [(v, k) for k, v in VIETNAMESE_DICTIONARY.items()]:
        unaccented_dict[unaccented] = accented

    # Also map unaccented form to itself (for words that are already in unaccented form)
    for word in VIETNAMESE_DICTIONARY.keys():
        unaccented = remove_vietnamese_accents(word)
        if unaccented not in unaccented_dict:
            unaccented_dict[unaccented] = word

    # Also add the accented form as a key to itself
    for word in VIETNAMESE_DICTIONARY.values():
        unaccented_dict[word] = word

    # Process 3-word phrases
    for accented, unaccented in [(v, k) for k, v in THREE_WORD_PHRASES.items()]:
        unaccented_dict[unaccented] = accented

    return unaccented_dict

def is_vietnamese_question(text: str) -> bool:
    """
    Detect if a text is likely a Vietnamese question.
    """
    # Check if already has question mark
    if text.strip().endswith('?'):
        return True

    # Common Vietnamese question words/patterns (both with and without accents)
    question_patterns = [
        r'\b(ai|Ai)\b',                      # who
        r'\b(gi|Gi|gì|Gì)\b',                # what
        r'\b(nao|Nao|nào|Nào)\b',            # which
        r'\b(the nao|The nao|thế nào|Thế nào)\b',  # how
        r'\b(bao (nhieu|giờ|lâu)|Bao (nhieu|giờ|lâu)|bao (nhiêu|giờ|lâu)|Bao (nhiêu|giờ|lâu))\b',  # how much/many/long
        r'\b(o dau|O dau|ở đâu|Ở đâu)\b',    # where
        r'\b(khi nao|Khi nao|khi nào|Khi nào)\b',  # when
        r'\b(tai sao|Tai sao|tại sao|Tại sao)\b',  # why
        r'\b(co|Co|có|Có)\b.+\b(khong|Khong|không|Không)\b',  # yes/no pattern with "khong" at end
        r'\b(co phai|Co phai|có phải|Có phải)\b',  # is it true that
        r'\b(lam sao|Lam sao|làm sao|Làm sao)\b',  # how to
        r'\b(nhu the nao|Nhu the nao|như thế nào|Như thế nào)\b',  # in what way
        r'\b(hoi|Hoi|hỏi|Hỏi)\b',            # ask
        r'\b(cho hoi|Cho hoi|cho hỏi|Cho hỏi)\b',  # may I ask
        r'\b(xin hoi|Xin hoi|xin hỏi|Xin hỏi)\b',  # please ask
        r'\b(vui long|Vui long|vui lòng|Vui lòng).+\b(gi|gì|cho|về|ve)\b',  # please tell me pattern
        r'\b(toi|Toi|tôi|Tôi)\b.+\b(muon hoi|muốn hỏi|muon biet|muốn biết)\b',  # I want to ask/know pattern
    ]

    # Check if the text matches any question pattern
    for pattern in question_patterns:
        if re.search(pattern, text):
            return True

    # Check for common question words at the beginning
    first_word = text.split()[0].lower() if text.split() else ""
    question_starters = ['ai', 'gi', 'gì', 'nao', 'nào', 'sao', 'tại', 'tai', 'khi',
                         'đâu', 'dau', 'hỏi', 'hoi', 'xin', 'làm', 'lam', 'có', 'co']
    if first_word in question_starters:
        return True

    # Check for question words anywhere in the text
    text_lower = text.lower()
    question_indicators = ['?', 'hỏi', 'hoi', 'thắc mắc', 'thac mac', 'được không', 'duoc khong',
                           'có không', 'co khong', 'phải không', 'phai khong',
                           'có phải', 'co phai', 'được chứ', 'duoc chu', 'được không', 'duoc khong']
    for indicator in question_indicators:
        if indicator in text_lower:
            return True

    return False

def complete_question_mark(text: str) -> str:
    """
    Add question mark at the end of Vietnamese questions if missing.
    """
    # Trim whitespace
    text = text.strip()

    # Skip if already has question mark
    if text.endswith('?'):
        return text

    # Skip if ends with other punctuation
    if text and text[-1] in '.!;:,':
        return text

    # Check if it's a question
    if is_vietnamese_question(text):
        return text + '?'

    return text

UNACCENTED_TO_ACCENTED = build_unaccented_dictionary()

def restore_vietnamese_accents(text: str) -> str:
    """
    Restore Vietnamese accents using an improved dictionary-based approach.
    """
    # If text is empty, return as is
    if not text:
        return text

    # Split text into sentences for better context handling
    sentences = re.split(r'([.!?;:])', text)
    processed_sentences = []

    for i in range(0, len(sentences), 2):
        sentence = sentences[i].strip()

        if not sentence:  # Skip empty sentences
            if i + 1 < len(sentences):
                processed_sentences.append(sentences[i + 1])
            continue

        # Process each sentence
        words = sentence.split()
        result_words = []

        # Try to match 3-word phrases first
        i_word = 0
        while i_word < len(words):
            matched = False

            # Try 3-word phrase
            if i_word + 2 < len(words):
                three_word = words[i_word].lower() + " " + words[i_word + 1].lower() + " " + words[i_word + 2].lower()
                if three_word in THREE_WORD_PHRASES:
                    result_words.append(THREE_WORD_PHRASES[three_word])
                    i_word += 3
                    matched = True
                    continue

            # Try 2-word phrase
            if i_word + 1 < len(words):
                two_word = words[i_word].lower() + " " + words[i_word + 1].lower()
                if two_word in VIETNAMESE_DICTIONARY:
                    result_words.append(VIETNAMESE_DICTIONARY[two_word])
                    i_word += 2
                    matched = True
                    continue

            # Try single word
            word_lower = words[i_word].lower()
            if word_lower in VIETNAMESE_DICTIONARY:
                result_words.append(VIETNAMESE_DICTIONARY[word_lower])
            elif word_lower in UNACCENTED_TO_ACCENTED:
                result_words.append(UNACCENTED_TO_ACCENTED[word_lower])
            else:
                # No match found, keep original word
                result_words.append(words[i_word])

            i_word += 1

        # Join words back into a sentence
        processed_sentence = " ".join(result_words)

        # Preserve original capitalization of first word
        if sentence and len(processed_sentence) > 0:
            if sentence[0].isupper():
                processed_sentence = processed_sentence[0].upper() + processed_sentence[1:]

        processed_sentences.append(processed_sentence)

        # Add punctuation back if it exists
        if i + 1 < len(sentences):
            processed_sentences.append(sentences[i + 1])

    # Join sentences back together
    result_text = "".join(processed_sentences)

    return result_text

def preprocess_vietnamese_query(query: str) -> Dict[str, str]:
    """
    Preprocess Vietnamese query:
    1. Add question mark if needed
    2. Restore accents if missing
    3. Create accented/unaccented versions

    Returns:
        Dictionary with various processed forms of the query
    """
    results = {
        "original": query,  # Original query as received
    }

    # Check if original query has accents
    normalized_query = remove_vietnamese_accents(query)
    results["normalized"] = normalized_query  # Normalized (no accents) version
    results["had_accents"] = (query != normalized_query)  # Boolean: did original have accents?

    # Complete with question mark if needed
    processed_query = complete_question_mark(query)
    results["processed"] = processed_query  # Original with question mark if needed

    # If query doesn't have accents, restore them
    if not results["had_accents"]:
        accented_query = restore_vietnamese_accents(processed_query)
        # Apply question mark to accented version too
        if processed_query.endswith('?') and not accented_query.endswith('?'):
            accented_query += '?'
        results["accented"] = accented_query  # Restored accents + question mark
    else:
        results["accented"] = processed_query  # Already has accents + question mark

    # Always provide normalized version of the processed query (with question mark if appropriate)
    results["normalized_processed"] = remove_vietnamese_accents(processed_query)

    # Is it a question?
    results["is_question"] = is_vietnamese_question(query)

    return results


#legal_glossary_lookup


def load_legal_dictionary(path: str = 'legal_terms.json') -> list:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['dictionary']

def is_definition_question(query: str) -> bool:
    definition_keywords = ["là gì", "định nghĩa", "nghĩa là gì", "hiểu thế nào", "khái niệm"]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in definition_keywords)


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def search_term_in_dictionary(query: str, dictionary: list) -> Optional[dict]:
    if not is_definition_question(query):
        return None  # Chỉ tra từ điển nếu là câu hỏi định nghĩa

    query_normalized = normalize_text(query)
    for entry in dictionary:
        term_normalized = normalize_text(entry['term'])

        # So khớp gần như chính xác
        if term_normalized == query_normalized or term_normalized in query_normalized:
            return entry
    return None


# def expand_query(question: str, llm) -> str:
#     """
#     Mở rộng truy vấn ban đầu bằng cách tạo các cách hỏi khác nhau.
#     """
#     prompt = PromptTemplate(
#         input_variables=["question"],
#         template="Hãy viết lại hoặc mở rộng câu hỏi sau thành nhiều cách hỏi khác nhau để dễ dàng tìm tài liệu: {question}"
#     )
#     chain = LLMChain(llm=llm, prompt=prompt)
#     expanded_query = chain.run(question)

#     print(f"Expanded query: {expanded_query}")
#     return expanded_query.strip()

# def rerank(query: str, documents: List[Document], top_k: int = 5, reranker=None) -> List[Document]:
#     """
#     Rerank các documents bằng mô hình CrossEncoder dựa trên mức độ liên quan.
#     """
#     if not documents:
#         return []




#     pairs = [(query, doc.page_content) for doc in documents]
#     scores = reranker.predict(pairs)

#     # Gắn điểm số vào từng document
#     scored_docs = list(zip(documents, scores))

#     # Sắp xếp giảm dần theo score
#     scored_docs.sort(key=lambda x: x[1], reverse=True)

#     # Chọn top_k documents
#     top_documents = [doc for doc, score in scored_docs[:top_k]]

#     print(f"Selected top {len(top_documents)} documents after reranking.")
#     return top_documents


# def create_retreival(vectorstore, llm, search_k: int = 10, year_metadata_field: str = "year", min_docs_per_query: int = 3 ):
#     base_retriever = vectorstore.as_retriever(search_kwargs={"k": search_k})

#     # Ưu tiên năm
#     year_priority_retriever = YearPriorityRetriever(
#         retriever=base_retriever,
#         now_year=2025,
#         min_docs_per_query=min_docs_per_query,
#         year_metadata_field=year_metadata_field,
#         debug=True  # Show debug output
#     )

#     # Mở rộng câu hỏi đa dạng
#     multi_query_retriever = MultiQueryRetriever.from_llm(
#         retriever=year_priority_retriever,
#         llm=llm
#     )



#     return multi_query_retriever
