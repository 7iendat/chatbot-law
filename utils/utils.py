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
from typing import List, Optional, Union
from pathlib import Path
from langchain_core.runnables import RunnableLambda, Runnable, RunnableSequence
from schemas.chat import  Message
from redis.client import Redis
import bcrypt
from datetime import datetime, timedelta
from jose import jwt
from config import SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES
from langchain_core.runnables import Runnable
from typing import List, Dict, Tuple, Optional, Any
from data.dic.dic import VIETNAMESE_DICTIONARY
import unicodedata
from collections import  Counter
from underthesea import word_tokenize as underthesea_tokenize
import cv2
import numpy as np
import time
from functools import lru_cache
from unidecode import unidecode
from db.mongoDB import user_collection
import secrets
from fastapi import HTTPException, status
from redis.exceptions import RedisError, ConnectionError as RedisConnectionError
from langchain_community.chat_message_histories import RedisChatMessageHistory


# Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def router_as_runnable(
        routes: dict[str, RunnableSequence], # Sửa kiểu dữ liệu
        get_key: RunnableLambda, # Sửa kiểu dữ liệu
        default: RunnableSequence = None # Sửa kiểu dữ liệu
    ) -> RunnableLambda: # Sửa kiểu dữ liệu
        def dispatch(input_data_dict): # Đổi tên biến cho rõ ràng
            key = get_key.invoke(input_data_dict)
            logging.info(f"🔸[Router] Route key: {key}")
            selected_runnable = routes.get(key, default)
            if selected_runnable is None:
                raise ValueError(f"No route found for key '{key}' and no default route is set.")
            logging.info(f"🔸[Router] Selected runnable type: {type(selected_runnable)}")
            return selected_runnable # RunnableLambda sẽ tự động invoke runnable này

        # Quan trọng: RunnableLambda sẽ nhận input_data_dict, chạy hàm dispatch,
        # hàm dispatch trả về một Runnable (selected_runnable).
        # Sau đó, Langchain sẽ tự động .invoke(input_data_dict) trên selected_runnable đó.
        return RunnableLambda(dispatch)


def preprocess_image_for_ocr(image: Image.Image) -> Image.Image:
    """
    Tiền xử lý ảnh để tăng chất lượng OCR.
    Args:
        image: Ảnh PIL cần xử lý.
    Returns:
        Ảnh PIL đã được xử lý.
    """
    # Chuyển ảnh sang OpenCV
    img_array = np.array(image)
    if len(img_array.shape) == 3:  # Chuyển sang grayscale nếu là ảnh màu
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Tăng độ tương phản và giảm nhiễu
    img_array = cv2.convertScaleAbs(img_array, alpha=1.5, beta=0)
    img_array = cv2.GaussianBlur(img_array, (3, 3), 0)
    _, img_array = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return Image.fromarray(img_array)

def extract_text_from_scanned_pdf(
    pdf_path: str,
    lang: str = 'vie',
    dpi: int = 300,
    max_retries: int = 2,
    custom_config: str = '--oem 3 --psm 6 -c preserve_interword_spaces=1'
) -> Tuple[str, dict]:
    """
    Trích xuất văn bản từ file PDF scan bằng OCR (PyMuPDF + Tesseract OCR).
    Args:
        pdf_path (str): Đường dẫn đến file PDF.
        lang (str): Mã ngôn ngữ OCR (ví dụ: 'vie' cho tiếng Việt).
        dpi (int): Độ phân giải DPI cho ảnh (mặc định: 300).
        max_retries (int): Số lần thử lại nếu OCR thất bại.
        custom_config (str): Cấu hình tùy chỉnh cho Tesseract.
    Returns:
        Tuple[str, dict]: Văn bản trích xuất và metadata (số trang, lỗi, thời gian xử lý).
    """
    import time
    start_time = time.time()
    text_buffer = io.StringIO()
    metadata = {
        "total_pages": 0,
        "processed_pages": 0,
        "errors": [],
        "processing_time": 0
    }

    try:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file: {pdf_path}")

        doc = fitz.open(pdf_path)
        metadata["total_pages"] = len(doc)
        logger.info(f"Bắt đầu OCR cho {pdf_path}: {metadata['total_pages']} trang")

        for page_num, page in enumerate(doc, start=1):
            attempt = 0
            success = False
            while attempt <= max_retries and not success:
                try:
                    # Tạo ảnh từ trang PDF
                    zoom = dpi / 72
                    mat = fitz.Matrix(zoom, zoom)
                    pix = page.get_pixmap(matrix=mat)

                    # Tiền xử lý ảnh
                    img = Image.open(io.BytesIO(pix.tobytes("ppm")))
                    img = preprocess_image_for_ocr(img)

                    # Thực hiện OCR
                    page_text = pytesseract.image_to_string(
                        img,
                        config=custom_config,
                        lang=lang
                    )

                    page_text = page_text.strip()
                    text_buffer.write(page_text + "\n")
                    metadata["processed_pages"] += 1
                    logger.info(f"🔸Trang {page_num}/{metadata['total_pages']}: {len(page_text)} ký tự")
                    success = True

                except Exception as ocr_err:
                    attempt += 1
                    error_msg = f"Lỗi OCR trang {page_num}, thử {attempt}/{max_retries}: {ocr_err}"
                    metadata["errors"].append(error_msg)
                    logger.warning(error_msg)
                    if attempt > max_retries:
                        logger.error(f"🔸Bỏ qua trang {page_num} sau {max_retries} lần thử")
                    continue

            # Giải phóng bộ nhớ
            pix = None
            img = None

        doc.close()
        extracted_text = text_buffer.getvalue().strip()
        metadata["processing_time"] = time.time() - start_time

        if not extracted_text:
            logger.warning("🔸Không trích xuất được văn bản từ PDF")

        return extracted_text, metadata

    except Exception as e:
        logger.error(f"🔸Không thể xử lý file PDF '{pdf_path}': {e}")
        metadata["errors"].append(str(e))
        metadata["processing_time"] = time.time() - start_time
        return "", metadata


def is_likely_scanned_pdf(doc: fitz.Document, sample_pages: int = 3) -> bool:
    """
    Kiểm tra xem PDF có phải là bản scan (không có văn bản nhúng).
    Args:
        doc: Tài liệu PDF đã mở bằng fitz.
        sample_pages: Số trang mẫu để kiểm tra.
    Returns:
        bool: True nếu PDF có khả năng là bản scan.
    """
    for page_num in range(min(sample_pages, len(doc))):
        page = doc[page_num]
        text = page.get_text().strip()
        if len(text) > 50:  # Có văn bản đáng kể, không phải scan
            return False

        # Kiểm tra xem trang có chứa ảnh
        images = page.get_images(full=True)
        if images:  # Có ảnh, có thể là scan
            return True

    return True  # Không có văn bản, giả định là scan

def extract_text_from_pdf_auto(
    pdf_path: str,
    lang: str = 'vie',
    dpi: int = 300,
    max_retries: int = 2,
    min_text_length: int = 100
) -> Tuple[str, Dict]:
    """
    Tự động trích xuất văn bản từ PDF (thường hoặc scan).
    Args:
        pdf_path (str): Đường dẫn đến file PDF.
        lang (str): Mã ngôn ngữ OCR (mặc định: 'vie' cho tiếng Việt).
        dpi (int): Độ phân giải DPI cho OCR.
        max_retries (int): Số lần thử lại nếu OCR thất bại.
        min_text_length (int): Độ dài văn bản tối thiểu để coi là PDF văn bản (không scan).
    Returns:
        Tuple[str, Dict]: Văn bản trích xuất và metadata (số trang, lỗi, thời gian xử lý).
    """
    start_time = time.time()
    metadata = {
        "total_pages": 0,
        "processed_pages": 0,
        "errors": [],
        "processing_time": 0,
        "method": "text"  # text hoặc ocr
    }

    try:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file: {pdf_path}")

        logger.info(f"Đang mở PDF: {pdf_path}")
        doc = fitz.open(pdf_path)
        metadata["total_pages"] = len(doc)
        text_buffer = io.StringIO()

        # Thử trích xuất văn bản trực tiếp
        for page_num, page in enumerate(doc, start=1):
            page_text = page.get_text().strip()
            text_buffer.write(page_text + "\n")
            metadata["processed_pages"] += 1
            logger.info(f"🔸Trang {page_num}/{metadata['total_pages']}: {len(page_text)} ký tự (text trực tiếp)")

        extracted_text = text_buffer.getvalue().strip()

        # Kiểm tra xem PDF có phải là bản scan
        if len(extracted_text) < min_text_length or is_likely_scanned_pdf(doc):
            logger.info(f"🔸Text ít ({len(extracted_text)} ký tự) hoặc nghi là PDF scan, chuyển sang OCR")
            doc.close()  # Đóng tài liệu trước khi gọi OCR
            metadata["method"] = "ocr"
            ocr_text, ocr_metadata = extract_text_from_scanned_pdf(
                pdf_path=pdf_path,
                lang=lang,
                dpi=dpi,
                max_retries=max_retries
            )
            extracted_text = ocr_text
            metadata["errors"].extend(ocr_metadata["errors"])
            metadata["processed_pages"] = ocr_metadata["processed_pages"]

        else:
            doc.close()

        metadata["processing_time"] = time.time() - start_time

        if not extracted_text:
            logger.warning("🔸Không trích xuất được văn bản từ PDF (text hoặc OCR)")
            metadata["errors"].append("Không trích xuất được văn bản")

        return extracted_text, metadata

    except Exception as e:
        logger.error(f"🔸Lỗi khi auto-extract PDF '{pdf_path}': {e}")
        metadata["errors"].append(str(e))
        metadata["processing_time"] = time.time() - start_time
        return "", metadata

# Initialize text splitter with optimized parameters
base_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,         # Kích thước chunk lớn phù hợp văn bản luật dài
    chunk_overlap=200,       # Tăng overlap để giữ mạch logic và câu dài
    length_function=len,     # Hàm tính độ dài (số ký tự)
    add_start_index=False,   # Không thêm chỉ số bắt đầu (tuỳ nhu cầu)
    separators=[             # Ưu tiên tách theo cấu trúc logic
        "\nĐiều",            # Tách trước các Điều
        "\nChương",          # Tách theo Chương (nếu có)
        "\n",                # Tách theo dòng mới
        " ",                 # Tách theo khoảng trắng
        ""                   # Tách từng ký tự nếu vẫn quá dài
    ],
)


def clean_text_optimized(text: str) -> str:
    """Clean text from extracted .txt files, removing headers/footers/noise."""
    # Remove large header/footer blocks
    block_patterns = [
        r"^\s*.*?LỚN NHÁT VIỆT NAM.*?bàn dịch tiếng Anh\s*",
        r"^\s*Trung tâm LuafVietnam[\s\S]*?(?:lawdat\s*afl\s*u\s*atvistnarn\.vni|378536589|Email:.*?@luatvietnam\.vn)\s*$",
        r"^\s*:?\s*CƠ SỞ DỮ LIỆU VĂN BẢN PHÁP LUẬT\s*$",
        r"^\s*LuatVietnam\.vn[\s\S]*?(?:Đặt mua văn bản gốc|Hotline:.*?)\s*$",
    ]
    for pat in block_patterns:
        text = re.sub(pat, "", text, flags=re.MULTILINE | re.DOTALL | re.IGNORECASE)

    # Remove noise lines
    remove_line_patterns = [
        r"^\s*(.*LuatWielnam.*|.*LuatVietnam\.vn.*|.*Tiện ích văn bản luật.*|.*www\.vanbanluat\.vn.*)$",
        r"^\s*\[\s*Hình\s*ảnh\s*]\s*$",
        r"^[=*_\-]{3,}$",
        r"^\s*(teeeeokanlbaglueloen|Tee===|Tc=e===|nem|SN Hntlin sa:|HT:|Hntlin sa:).*",
        r"^\s*\d{1,3}(?:\.\d{3})*.*văn bản pháp luật.*tiếng Anh\s*$",
        r"^\s*Đặt mua văn bản gốc.*Hotline.*$",
    ]
    remove_line_regex = [re.compile(pat, flags=re.IGNORECASE) for pat in remove_line_patterns]

    # Process lines
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if not line or any(pat.match(line) for pat in remove_line_regex):
            continue

        # Giữ lại các dòng có thể chứa thông tin pháp lý
        if re.match(r"^\s*(Điều\s+\d+|Khoản\s+\d+|Điểm\s+[a-z]|Phần\s+[IVXLCDM]+|Chương\s+[IVXLCDM]+|Mục\s+\d+)", line, re.IGNORECASE):
            cleaned_lines.append(line)
            continue

        # Clean headings and OCR errors
        line = re.sub(r"^\s*[-—_\s]*(CHƯƠNG\s+[IVXLCDM\d]+)\s*[-—_\s]*", r"\1", line, flags=re.IGNORECASE)
        line = re.sub(r"^\s*[-—_\s]*(Mục\s+\d+)\s*[-—_\s]*", r"\1", line, flags=re.IGNORECASE)
        line = re.sub(r"^\s*[-—_\s]*(PHẦN\s+(?:THỨ\s+[A-Z]+|CHUNG|CÁC TỘI PHẠM))\s*[-—_\s]*", r"\1", line, flags=re.IGNORECASE)
        line = re.sub(r"[¡¬_`„´ˆ˜]", "", line)
        line = line.replace("aflu atvistnarn.vni", "@luatvietnam.vn")
        line = re.sub(r"\s{2,}", " ", line)

        if line:
            cleaned_lines.append(line)

    # Join and normalize
    final_text = "\n".join(cleaned_lines)
    final_text = re.sub(r"\n{3,}", "\n\n", final_text)
    final_text = re.sub(r"[ \t]{2,}", " ", final_text)
    return final_text.strip()

def extract_date_from_text(text: str) -> Optional[str]:
    """Trích xuất ngày hiệu lực từ văn bản pháp luật."""
    patterns = [
        r"\b(?:có hiệu lực từ ngày|ngày hiệu lực)\s*(\d{1,2}/\d{1,2}/(19|20)\d{2})\b",  # e.g., có hiệu lực từ ngày 1/1/2025
        r"\b(?:ngày\s+\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+(19|20)\d{2})\b",  # e.g., ngày 1 tháng 1 năm 2025
        r"\b(?:ngày\s+\d{1,2}-\d{1,2}-(19|20)\d{2})\b",  # e.g., ngày 1-1-2025
    ]
    for pattern in patterns:
        match = re.search(pattern, text[:2000], re.IGNORECASE | re.UNICODE)  # Giới hạn 2000 ký tự đầu
        if match:
            date_text = match.group(1)
            try:
                # Chuẩn hóa định dạng ngày thành YYYY-MM-DD
                date = datetime.strptime(date_text, "%d/%m/%Y")
                return date.strftime("%Y-%m-%d")
            except ValueError:
                try:
                    date = datetime.strptime(date_text, "%d-%m-%Y")
                    return date.strftime("%Y-%m-%d")
                except ValueError:
                    continue
    return None

def extract_year_from_text(text: str) -> Optional[int]:
    """Extract year from text using common Vietnamese legal document patterns."""
    patterns = [
        r"\b(?:số[:\s]*)?\d{1,3}/((?:19|20)\d{2})/[A-Z]+\d*\b",  # 168/2024/NĐ-CP
        r"\b(?:số[:\s]*)?\d{1,3}[-–]\d{1,2}[-–]((?:19|20)\d{2})\b",  # 23-15-1992
        r"\b(?:năm|ngày\s+\d{1,2}\s+tháng\s+\d{1,2}\s+năm)\s+((?:19|20)\d{2})\b",  # ngày 12 tháng 6 năm 1999
        r"\bngày\s+\d{1,2}/\d{1,2}/((?:19|20)\d{2})\b",  # ngày 12/6/1999
        r"\b((?:19|20)\d{2})\b",  # Standalone year
    ]

    for pattern in patterns:
        match = re.search(pattern, text[:2000], re.IGNORECASE)
        if match:
            return int(match.group(1))  # Chắc chắn nhóm 1 chứa năm đầy đủ

    return None

def extract_year_from_filename(filename: str) -> Optional[int]:
    """Extract year from filename."""
    match = re.search(r"(19|20)\d{2}", filename)
    return int(match.group()) if match else None

def extract_years_from_vectorstore(vectorstore):
    """Extract years from vectorstore for statistical analysis."""
    all_docs = vectorstore.similarity_search(" ", k=1000)  # Adjust k based on dataset size
    years = [doc.metadata.get("year") for doc in all_docs if isinstance(doc.metadata.get("year"), int)]
    year_counts = Counter(years)
    return sorted(year_counts, key=lambda y: (-year_counts[y], -y))

def detect_document_structure(text: str) -> str:
    """Detect the structure of the document."""
    if re.search(r"^(?:\s*(Điều\s+\d+[a-z]?|Phần\s+(?:thứ\s+[A-Z]+|[IVXLCDM]+)|Chương\s+[IVXLCDM]+|Mục\s+\d+))\s*[:.]?", text, re.MULTILINE | re.IGNORECASE):
        return "legal_structure"
    elif re.search(r"^(?:\s*(Mục lục|Phụ lục|Biểu mẫu|I\.|II\.|1\.|A\.))", text, re.MULTILINE | re.IGNORECASE):
        return "outline_structure"
    return "free_text"

def correct_ocr_date_string(date_str: str) -> str:
    """Sửa lỗi OCR phổ biến trong chuỗi ngày tháng."""
    replacements = {
        'ó': '6', 'Ò': '0', 'ọ': '0', 'O': '0',  # thay O/ó thành 0
        'l': '1', 'I': '1', 'i': '1',            # chữ l -> 1
        'Z': '2', 'z': '2',                      # OCR nhầm Z
        'B': '8',                                # OCR nhầm B
    }

    for wrong, right in replacements.items():
        date_str = date_str.replace(wrong, right)

    return date_str



def infer_field(query: str) -> str:
    """Suy ra lĩnh vực pháp luật từ câu hỏi hoặc nội dung tài liệu."""
    query_lower = query.lower()
    field_keywords = {
        "giao_thong": ["giao thông", "đèn đỏ", "tốc độ", "say rượu", "bằng lái", "phạt", "vi phạm giao thông", "xe cộ"],
        "hinh_su": ["trộm cắp", "ăn cắp", "tội phạm", "giết người", "lừa đảo", "cướp", "ma túy"],
        "dan_su": ["thừa kế", "hợp đồng", "ly hôn", "tài sản", "tranh chấp", "khiếu kiện"],
        "thuong_mai": ["hợp đồng", "doanh nghiệp", "thương mại", "bán hàng", "xuất nhập khẩu"],
        "moi_truong": ["xả thải", "ô nhiễm", "môi trường", "rác thải", "xử lý chất thải"],
        "lao_dong": ["sa thải", "hợp đồng lao động", "lương", "nghỉ việc", "bảo hiểm xã hội"],
        "hanh_chinh": ["thủ tục hành chính", "giấy phép", "đăng ký", "xử phạt hành chính"],
    }
    for field, keywords in field_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            return field
    # Sử dụng mô hình zero-shot nếu có tài nguyên
    try:
        from transformers import pipeline
        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        candidate_labels = list(field_keywords.keys())
        result = classifier(query, candidate_labels, multi_label=False)
        if result["scores"][0] > 0.5:
            return result["labels"][0]
    except Exception as e:
        logger.warning(f"🔸Zero-shot classification failed: {e}")
    return "unknown"

def infer_entity_type(query: str, field: str) -> str:
    """Suy ra loại thực thể từ câu hỏi và lĩnh vực."""
    query_lower = query.lower()
    entity_types = {
        "giao_thong": {
            "xe_oto": ["ô tô", "xe hơi"],
            "xe_moto": ["mô tô", "xe máy"],
            "xe_dap": ["xe đạp"],
            "xe_tho_so": ["xe thô sơ"],
            "tat_ca_phuong_tien": ["phương tiện", "xe", "tất cả"],
            "nguoi_dieu_khien": ["người điều khiển", "tài xế"],
        },
        "hinh_su": {
            "person": ["người", "cá nhân", "tội phạm", "bị can", "bị cáo"],
            "organization": ["tổ chức", "nhóm", "băng đảng"],
        },
        "dan_su": {
            "person": ["người", "cá nhân"],
            "contract": ["hợp đồng"],
            "asset": ["tài sản", "di sản"],
        },
        "thuong_mai": {
            "contract": ["hợp đồng"],
            "business": ["doanh nghiệp", "công ty"],
        },
        "moi_truong": {
            "facility": ["cơ sở", "nhà máy"],
            "organization": ["tổ chức", "công ty"],
        },
        "lao_dong": {
            "employee": ["nhân viên", "người lao động"],
            "employer": ["người sử dụng lao động", "công ty"],
        },
        "hanh_chinh": {
            "person": ["cá nhân", "người"],
            "organization": ["tổ chức", "cơ quan"],
        },
    }
    if field in entity_types:
        for entity, keywords in entity_types[field].items():
            if any(keyword in query_lower for keyword in keywords):
                return entity
        return list(entity_types[field].keys())[0]  # Mặc định lấy entity đầu tiên
    return "unknown"

def extract_penalty(text: str, field: str) -> Optional[Dict]:
    """Extract penalty information based on field."""
    text_lower = text.lower()
    penalty_types = {
        "giao_thong": [
            (r"phạt tiền từ (\d+\.?\d*)\.?\d* đồng đến (\d+\.?\d*)\.?\d* đồng", {"type": "fine", "min": 1, "max": 2}),
            (r"phạt tiền (\d+\.?\d*)\.?\d* đồng", {"type": "fine", "amount": 1}),
            (r"tước bằng lái (\d+) tháng", {"type": "license_suspension", "duration": 1}),
        ],
        "hinh_su": [
            (r"tù từ (\d+) tháng đến (\d+) năm", {"type": "prison", "min": 1, "max": 2}),
            (r"tù (\d+) năm", {"type": "prison", "amount": 1}),
            (r"cải tạo không giam giữ đến (\d+) năm", {"type": "reform", "max": 1}),
            (r"tịch thu tài sản", {"type": "confiscation"}),
        ],
        "dan_su": [
            (r"bồi thường thiệt hại từ (\d+\.?\d*)\.?\d* đồng", {"type": "compensation", "amount": 1}),
        ],
        "thuong_mai": [
            (r"bồi thường thiệt hại từ (\d+\.?\d*)\.?\d* đồng", {"type": "compensation", "amount": 1}),
            (r"phạt tiền từ (\d+\.?\d*)\.?\d* đồng đến (\d+\.?\d*)\.?\d* đồng", {"type": "fine", "min": 1, "max": 2}),
        ],
        "moi_truong": [
            (r"phạt tiền từ (\d+\.?\d*)\.?\d* đồng đến (\d+\.?\d*)\.?\d* đồng", {"type": "fine", "min": 1, "max": 2}),
        ],
        "hanh_chinh": [
            (r"phạt tiền từ (\d+\.?\d*)\.?\d* đồng đến (\d+\.?\d*)\.?\d* đồng", {"type": "fine", "min": 1, "max": 2}),
        ],
    }
    if field in penalty_types:
        for pattern, penalty_dict in penalty_types[field]:
            match = re.search(pattern, text_lower)
            if match:
                result = penalty_dict.copy()
                for i, group in enumerate(match.groups(), 1):
                    result[list(result.keys())[i-1]] = group
                return result
    return None




# def extract_law_structure(text: str) -> Dict[str, str]:
#     """Extract structure of legal document (type, number, title, issuance date)."""
#     structure = {}
#     combined_pattern = re.compile(
#         r"(?:(số|Số)[:\s]*(\d+[/-]\d{2,4}/[\w-]+))"
#         r"(Luật|Nghị định|Thông tư|Quyết định|Bộ luật|Pháp lệnh|Nghị quyết)|"
#         r"(ngày\s+\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4})",
#         re.IGNORECASE
#     )

#     for match in combined_pattern.finditer(text):
#         if match.group(1):  # Số hiệu
#             structure["so_hieu"] = match.group(2)
#         elif match.group(3):  # Loại văn bản
#             structure["loai_van_ban"] = match.group(3).title()
#         elif match.group(4):  # Ngày ban hành
#             structure["ngay_ban_hanh"] = match.group(4)

#     year = extract_year_from_text(text)
#     if year:
#         structure["nam_ban_hanh"] = year

#     return structure

def extract_law_structure(text: str) -> Dict[str, Optional[str]]:
    """Extract structure of legal document (type, number, title, issuance date)."""
    structure = {
        "so_hieu": None,
        "loai_van_ban": None,
        "ngay_ban_hanh": None,
        "nam_ban_hanh": None
    }
    head_text = text[:1000]  # Chỉ lấy đoạn đầu

    # Tìm số hiệu
    so_hieu_pattern = re.compile(r"số[:\s]*(\d+[/-]\d{2,4}/[\w-]+)", re.IGNORECASE)
    so_hieu_match = so_hieu_pattern.search(head_text)
    if so_hieu_match:
        structure["so_hieu"] = so_hieu_match.group(1).strip()

    # Tìm loại văn bản
    loai_vb_pattern = re.compile(r"(luật|nghị định|thông tư|quyết định|bộ luật|pháp lệnh|nghị quyết)", re.IGNORECASE)
    loai_vb_match = loai_vb_pattern.search(head_text)
    if loai_vb_match:
        structure["loai_van_ban"] = loai_vb_match.group(1).strip().title()

    # Tìm ngày ban hành - phương pháp 1: Tìm ngày gần số hiệu
    if structure["so_hieu"]:
        # Lấy vị trí của số hiệu
        so_hieu_pos = head_text.find(structure["so_hieu"])
        if so_hieu_pos != -1:
            # Tìm ngày trong vòng 200 ký tự sau số hiệu
            search_text = head_text[so_hieu_pos:so_hieu_pos + 200]
            date_pattern = re.compile(
                r"ngày\s+(\d{1,2}[^\w]?)\s*tháng\s+(\d{1,2}[^\w]?)\s*năm\s+(\d{4})",
                re.IGNORECASE
            )
            date_match = date_pattern.search(search_text)
            if date_match:
                full_date = date_match.group(0)
                structure["ngay_ban_hanh"] = correct_ocr_date_string(full_date)

    # Nếu không tìm thấy, dùng phương pháp 2: Tìm ngày đầu tiên trong đoạn đầu
    if not structure["ngay_ban_hanh"]:
        date_pattern = re.compile(
            r"ngày\s+(\d{1,2}[^\w]?)\s*tháng\s+(\d{1,2}[^\w]?)\s*năm\s+(\d{4})",
            re.IGNORECASE
        )
        date_matches = date_pattern.findall(head_text)
        if date_matches:
            # Lấy ngày đầu tiên tìm thấy
            first_date = "ngày " + date_matches[0][0] + " tháng " + date_matches[0][1] + " năm " + date_matches[0][2]
            structure["ngay_ban_hanh"] = correct_ocr_date_string(first_date)

    # Tách năm từ ngày ban hành nếu có
    if structure["ngay_ban_hanh"]:
        year_pattern = re.compile(r"năm\s+(\d{4})")
        year_match = year_pattern.search(structure["ngay_ban_hanh"])
        if year_match:
            structure["nam_ban_hanh"] = year_match.group(1)
    else:
        # fallback: lấy năm đầu tiên tìm được
        structure["nam_ban_hanh"] = extract_year_from_text(text)

    return structure

def parse_vietnamese_law_item(line: str) -> Tuple[str, str, str]:
    """Parse a line to extract legal structure (Điều, Khoản, Điểm, Tiết)."""
    line = line.strip()
    dieu_match = re.match(r"^\s*(Điều\s+\d+[a-z]?)\s*[\.:]?\s*(.*?)$", line, re.IGNORECASE)
    if dieu_match:
        return "dieu", dieu_match.group(1), dieu_match.group(2)

    khoan_match = re.match(r"^\s*(\d+)\.\s*(.*?)$", line)
    if khoan_match:
        return "khoan", khoan_match.group(1), khoan_match.group(2)

    diem_match = re.match(r"^\s*([a-z])\)\s*(.*?)$", line)
    if diem_match:
        return "diem", diem_match.group(1), diem_match.group(2)

    tiet_match = re.match(r"^\s*[-–]\s*(.*?)$", line)
    if tiet_match:
        return "tiet", "-", tiet_match.group(1)

    return "text", "", line


def split_by_law_structure(doc: Document, max_chunk_size: int = 1000) -> List[Document]:
    """Split legal document into chunks based on Vietnamese legal structure."""
    text = doc.page_content
    source = doc.metadata.get("source", "N/A")
    lines = text.splitlines()

    chunks = []
    buffer = []
    current_meta = doc.metadata.copy() if doc.metadata else {"source": source}
    if "source" not in current_meta:
        current_meta["source"] = source

    patterns = {
        "phan": re.compile(r"^\s*(Phần\s+(?:thứ\s+[a-zA-Z]+|[IVXLCDM]+|chung|các tội phạm))\s*[:.]?\s*(.*)", re.IGNORECASE),
        "chuong": re.compile(r"^\s*(Chương\s+[IVXLCDM\d]+)\s*[:.]?\s*(.*)", re.IGNORECASE),
        "muc": re.compile(r"^\s*(Mục\s+\d+)\s*[:.]?\s*(.*)", re.IGNORECASE),
        "dieu": re.compile(r"^\s*(Điều\s+\d+[a-z]?)\s*[:.]?\s*(.*)", re.IGNORECASE),
    }

    def update_meta(key: str, value: str):
        nonlocal current_meta
        current_meta[key] = value
        if key == "phan":
            current_meta.pop("chuong", None)
            current_meta.pop("muc", None)
            current_meta.pop("dieu", None)
            current_meta.pop("khoan", None)
            current_meta.pop("diem", None)
            current_meta.pop("tiet", None)
        elif key == "chuong":
            current_meta.pop("muc", None)
            current_meta.pop("dieu", None)
            current_meta.pop("khoan", None)
            current_meta.pop("diem", None)
            current_meta.pop("tiet", None)
        elif key == "muc":
            current_meta.pop("dieu", None)
            current_meta.pop("khoan", None)
            current_meta.pop("diem", None)
            current_meta.pop("tiet", None)
        elif key == "dieu":
            current_meta.pop("khoan", None)
            current_meta.pop("diem", None)
            current_meta.pop("tiet", None)
        elif key == "khoan":
            current_meta.pop("diem", None)
            current_meta.pop("tiet", None)
        elif key == "diem":
            current_meta.pop("tiet", None)

    def flush_chunk(force=False):
        nonlocal buffer, current_meta
        content = "\n".join(buffer).strip()
        if content or force:
            title_parts = []
            for key in ["phan", "chuong", "muc", "dieu", "khoan", "diem"]:
                if key in current_meta:
                    title_parts.append(current_meta[key])
            chunk_title = " - ".join(title_parts) if title_parts else "Phần không có cấu trúc"
            chunk_meta = current_meta.copy()
            chunk_meta["title"] = chunk_title

            if len(content) > max_chunk_size:
                logger.warning(f"🔸Chunk '{chunk_title}' in '{source}' too large ({len(content)} chars). Splitting...")
                sub_docs = base_text_splitter.create_documents([content], metadatas=[chunk_meta])
                for i, sub_doc in enumerate(sub_docs):
                    sub_doc.metadata["sub_chunk"] = i + 1
                chunks.extend(sub_docs)
            else:
                chunks.append(Document(page_content=content, metadata=chunk_meta))
        buffer = []

    current_dieu = None
    current_khoan = None
    current_diem = None
    last_item_type = None

    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue

        matched = False
        for key in ["phan", "chuong", "muc", "dieu"]:
            match = patterns[key].match(line_stripped)
            if match:
                flush_chunk()  # Save previous chunk
                update_meta(key, match.group(1).strip())
                if key == "dieu":
                    current_dieu = match.group(1).strip()
                    current_khoan = None
                    current_diem = None
                buffer.append(line)
                last_item_type = key
                matched = True
                break

        if not matched:
            item_type, item_code, item_content = parse_vietnamese_law_item(line_stripped)
            if item_type == "khoan" and current_dieu:
                if current_khoan and (last_item_type in ["khoan", "diem", "tiet"]):
                    flush_chunk()
                current_khoan = item_code
                current_diem = None
                update_meta("khoan", f"Khoản {item_code}")
                buffer.append(line)
                last_item_type = "khoan"

            elif item_type == "diem" and current_khoan:
                if current_diem and (last_item_type in ["diem", "tiet"]):
                    flush_chunk()
                current_diem = item_code
                update_meta("diem", f"Điểm {item_code}")
                buffer.append(line)
                last_item_type = "diem"

            elif item_type == "tiet" and current_diem:
                flush_chunk()  # Flush trước khi thêm tiết
                update_meta("tiet", "Tiết")
                buffer.append(line)
                last_item_type = "tiet"

            else:
                buffer.append(line)
                last_item_type = "text"

    flush_chunk(force=True)  # Save final chunk
    return chunks

def split_document(doc: Document) -> List[Document]:
    """Split document based on detected structure."""
    structure = detect_document_structure(doc.page_content)
    if structure == "legal_structure":
        return split_by_law_structure(doc)
    else:
        return base_text_splitter.create_documents([doc.page_content], metadatas=[doc.metadata])


def load_and_clean_documents(folder_path: str) -> List[Document]:
    """Load and clean .txt files from folder, return list of Documents."""
    all_docs = []
    if not os.path.isdir(folder_path):
        logger.error(f"🔸Folder '{folder_path}' does not exist.")
        return all_docs

    txt_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.txt')]
    if not txt_files:
        logger.error(f"🔸No .txt files found in '{folder_path}'.")
        return all_docs

    logger.info(f"🔸Found {len(txt_files)} .txt files. Processing...")
    for filename in tqdm(txt_files, desc="Processing txt files"):
        file_path = os.path.join(folder_path, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_content = f.read()

            year = extract_year_from_filename(filename) or extract_year_from_text(raw_content)
            law_structure = extract_law_structure(raw_content)
            cleaned_content = clean_text_optimized(raw_content)

            if cleaned_content:
                field = infer_field(cleaned_content)
                metadata = {
                    "source": os.path.basename(file_path),
                    "year": year,
                    "field": field,
                    "entity_type": infer_entity_type(cleaned_content, field),
                    "penalty": extract_penalty(cleaned_content, field),
                    **law_structure,  # Thêm số hiệu, loại văn bản, ngày ban hành
                }
                # Kiểm tra metadata
                if not metadata["field"] or metadata["field"] == "other":
                    logger.warning(f"🔸Field not inferred for '{filename}', defaulting to 'unknown'")
                    metadata["field"] = "unknown"
                if not metadata["entity_type"]:
                    metadata["entity_type"] = "unknown"
                if not metadata["penalty"]:
                    metadata["penalty"] = None

                doc = Document(page_content=cleaned_content, metadata=metadata)
                all_docs.append(doc)
            else:
                logger.warning(f"🔸File '{filename}' empty after cleaning.")
        except Exception as e:
            logger.error(f"🔸Error processing file '{filename}': {e}")

    logger.info(f"🔸Processed {len(all_docs)} documents.")
    return all_docs


# Hàm hỗ trợ tìm kiếm nâng cao trong tài liệu luật
def search_in_law_documents(chunks, query, n_results=5):
    """
    Tìm kiếm đơn giản trong các chunks văn bản luật dựa trên từ khóa.

    Args:
        chunks: Danh sách các Document đã phân chia
        query: Từ khóa tìm kiếm
        n_results: Số kết quả tối đa trả về

    Returns:
        Danh sách các Document phù hợp nhất
    """
    query_lower = query.lower()
    results = []

    for chunk in chunks:
        content = chunk.page_content.lower()
        if query_lower in content:
            # Tính điểm đơn giản dựa trên số lần xuất hiện
            score = content.count(query_lower)
            results.append((chunk, score))

    # Sắp xếp kết quả theo điểm và lấy n kết quả đầu tiên
    results.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in results[:n_results]]

def route_logic(input: Dict) -> str:
    """
    Phân loại câu hỏi thành 'general' hoặc 'legal' dựa trên từ khóa và ngữ cảnh.

    Args:
        input (Dict): Dictionary chứa câu hỏi với key 'question'.

    Returns:
        str: 'general' hoặc 'legal'.

    Raises:
        ValueError: Nếu input không hợp lệ.
    """
    if not isinstance(input, dict):
        logger.error(f"Input không phải dictionary: {input}")
        raise ValueError("Input phải là một dictionary.")

    question = input.get("input", "").strip()
    if not question:
        logger.error("Thiếu key 'question' hoặc câu hỏi rỗng.")
        raise ValueError("Input phải chứa key 'input' với câu hỏi không rỗng.")

    question_tokenized = word_tokenize(question, format="text")

    # Danh sách từ khóa
    general_keywords = [
        "bạn là ai", "trợ lý", "chatbot", "ai", "robot", "trí tuệ nhân tạo", "giúp được gì",
        "có thể làm gì", "tên gì", "được tạo ra", "hoạt động", "được phát triển", "công ty nào",
        "tính năng", "khả năng", "được huấn luyện", "dữ liệu", "thông tin", "làm việc như thế nào",
        "tạo ra bởi", "phiên bản", "hỗ trợ", "ngôn ngữ", "hiểu được", "cập nhật", "mới nhất"
    ]

    legal_keywords = [
        "pháp luật", "luật", "nghị định", "mức phạt", "điều luật", "quy định",
        "thừa kế", "hôn nhân", "lao động", "giao thông", "hình sự", "dân sự",
        "đất đai", "thuế", "doanh nghiệp", "hợp đồng", "sở hữu trí tuệ",
        "vi phạm", "xử phạt", "quyền", "nghĩa vụ"
    ]

    # Phân loại bằng từ khóa
    if any(keyword in question_tokenized for keyword in general_keywords):
        logger.info(f"[RouteLogic] Phân loại: general (từ khóa: {general_keywords})")
        return "general"
    if any(keyword in question_tokenized for keyword in legal_keywords):
        logger.info(f"[RouteLogic] Phân loại: legal (từ khóa: {legal_keywords})")
        return "legal"

    # Xử lý các câu hỏi tổng quát về pháp luật
    if any(word in question_tokenized for word in ["ai", "người nào", "đối tượng", "có quyền", "được phép"]) and \
       any(word in question_tokenized for word in ["thừa kế", "kết hôn", "lao động", "sở hữu", "đất đai"]):
        logger.info(f"[RouteLogic] Phân loại: legal (câu hỏi tổng quát về pháp luật)")
        return "legal"

    # Mặc định là general
    logger.info(f"[RouteLogic] Phân loại: general (mặc định, không khớp từ khóa)")
    return "legal"


def word_tokenize(text: str, format: str = "list") -> Union[List[str], str]:
    """
    Phân tách văn bản tiếng Việt thành danh sách từ hoặc chuỗi được nối bằng dấu cách.

    Args:
        text (str): Văn bản đầu vào cần phân tách.
        format (str): Định dạng đầu ra, 'list' (danh sách từ) hoặc 'text' (chuỗi nối bằng dấu cách).

    Returns:
        Union[List[str], str]: Danh sách từ hoặc chuỗi tùy theo format.

    Raises:
        ValueError: Nếu input không phải chuỗi hoặc format không hợp lệ.

    Examples:
        >>> word_tokenize("vượt đèn đỏ", format="list")
        ['vượt', 'đèn đỏ']
        >>> word_tokenize("vượt đèn đỏ", format="text")
        'vượt đèn đỏ'
    """
    if not isinstance(text, str):
        raise ValueError("Input phải là chuỗi văn bản (str).")

    if format not in ["list", "text"]:
        raise ValueError("Format phải là 'list' hoặc 'text'.")

    # Phân tách từ bằng underthesea
    try:
        tokens = underthesea_tokenize(text, format="text").split()
    except Exception as e:
        print(f"Lỗi khi phân tách từ: {e}")
        tokens = text.split()  # Fallback: chia theo khoảng trắng nếu underthesea lỗi

    # Trả về theo định dạng yêu cầu
    if format == "list":
        return tokens
    return " ".join(tokens)

# === Redis helpers ===
# def save_chat_to_redis(r:Redis, chat_id: str, question: str, answer: str):
#     item = json.dumps({"question": question, "answer": answer})
#     r.rpush(f"chat:{chat_id}:messages", item)

# def save_chat_to_redis(r: Redis, chat_id: str, question: str, answer: str) -> bool:
#     """
#     Lưu tin nhắn người dùng và trợ lý vào Redis với định dạng chuẩn hóa.

#     Args:
#         r (Redis): Đối tượng Redis client.
#         chat_id (str): ID của hội thoại.
#         question (str): Câu hỏi đã xử lý của người dùng.
#         answer (str): Phản hồi của trợ lý.

#     Returns:
#         bool: True nếu lưu thành công, False nếu thất bại.

#     Raises:
#         ValueError: Nếu chat_id, question hoặc answer rỗng.
#         redis.RedisError: Nếu có lỗi khi tương tác với Redis.
#     """
#     # Kiểm tra đầu vào
#     if not chat_id or not question or not answer:
#         logger.error(f"Đầu vào không hợp lệ: chat_id={chat_id}, question={question}, answer={answer}")
#         raise ValueError("chat_id, question và answer không được rỗng")

#     redis_key = f"conversation:{chat_id}"
#     try:
#         # Kiểm tra kết nối Redis
#         r.ping()

#         # Tạo tin nhắn người dùng
#         user_message = {
#             "role": "user",
#             "content": question,
#             "timestamp": datetime.now().isoformat()
#         }

#         # Tạo tin nhắn trợ lý
#         assistant_message = {
#             "role": "assistant",
#             "content": answer,
#             "timestamp": datetime.now().isoformat()
#         }

#         # Lưu tin nhắn vào Redis
#         r.rpush(redis_key, json.dumps(user_message))
#         r.rpush(redis_key, json.dumps(assistant_message))

#         # Đặt TTL nếu key mới được tạo (kiểm tra số lượng tin nhắn trước khi lưu)
#         if r.llen(redis_key) == 2:  # Chỉ đặt TTL khi key mới (2 tin nhắn vừa thêm)
#             r.expire(redis_key, 86400)  # TTL: 24 giờ

#         logger.info(f"Đã lưu {redis_key}: 2 tin nhắn (user: {question[:50]}..., assistant: {answer[:50]}...)")
#         return True

#     except r.RedisError as e:
#         logger.error(f"Lỗi khi lưu vào Redis cho chat_id {chat_id}: {e}")
#         raise
#     except Exception as e:
#         logger.error(f"Lỗi không mong muốn khi lưu vào Redis cho chat_id {chat_id}: {e}")
#         return False


def save_chat_to_redis(
    r: Redis, # Hoặc redis.asyncio.Redis
    chat_id: str,
    user_question_content: str, # Nội dung câu hỏi gốc hoặc đã xử lý (tùy bạn)
    assistant_answer_content: str,
    user_question_timestamp: datetime, # Cung cấp timestamp
    assistant_answer_timestamp: datetime # Cung cấp timestamp
) -> bool:
    """
    Lưu tin nhắn mới của người dùng và trợ lý vào Redis với định dạng chuẩn hóa.
    Cập nhật 'updated_at' và 'message_count' trong metadata.
    """
    if not all([chat_id, user_question_content, assistant_answer_content]):
        logger.error("chat_id, user_question_content, và assistant_answer_content không được rỗng.")
        # raise ValueError("Đầu vào không hợp lệ.") # Hoặc trả về False
        return False

    messages_key = f"conversation_messages:{chat_id}"
    meta_key = f"conversation_meta:{chat_id}"

    try:
        # Tạo Pydantic models cho tin nhắn
        user_message = Message(role="user", content=user_question_content, timestamp=user_question_timestamp)
        assistant_message = Message(role="assistant", content=assistant_answer_content, timestamp=assistant_answer_timestamp)

        # Sử dụng pipeline cho các thao tác Redis
        pipe = r.pipeline()
        pipe.rpush(messages_key, user_message.model_dump_json())  # Pydantic V2
        pipe.rpush(messages_key, assistant_message.model_dump_json()) # Pydantic V2
        # Hoặc .json() cho Pydantic V1

        # Đặt TTL cho key messages nếu nó mới được tạo (hoặc luôn refresh TTL)
        # Nếu bạn muốn TTL chỉ đặt một lần, bạn cần kiểm tra sự tồn tại của key trước
        # hoặc kiểm tra llen trước khi push (phức tạp hơn với pipeline).
        # Cách đơn giản là luôn đặt lại TTL.
        pipe.expire(messages_key, 86400) # 24 giờ

        # Cập nhật metadata
        pipe.hset(meta_key, "updated_at", assistant_answer_timestamp.isoformat())
        pipe.hincrby(meta_key, "message_count", 2) # Tăng số lượng tin nhắn
        pipe.expire(meta_key, 86400) # Refresh TTL cho meta

        pipe.execute()

        logger.info(f"Đã lưu 2 tin nhắn mới vào {messages_key} và cập nhật {meta_key}.")
        return True

    except RedisError as e:
        logger.error(f"Lỗi Redis khi lưu tin nhắn cho chat_id {chat_id}: {e}", exc_info=True)
        raise # Re-raise để service xử lý HTTPException
    except Exception as e:
        logger.error(f"Lỗi không mong muốn khi lưu vào Redis cho chat_id {chat_id}: {e}", exc_info=True)
        # raise # Hoặc trả về False tùy theo cách bạn muốn xử lý
        return False

# def get_redis_history(r: Redis, chat_id: str) -> List[ChatHistoryItem]:
#     try:
#         history_raw = r.lrange(f"chat:{chat_id}:messages", 0, -1)
#         chat_history = []
#         for item in history_raw:
#             try:
#                 parsed = json.loads(item)
#                 if isinstance(parsed, dict):
#                     q = parsed.get("input", "")
#                     a = parsed.get("answer", "")
#                     if q and a:
#                         chat_history.append(ChatHistoryItem(input=q, answer=a))
#             except Exception as e:
#                 logger.error(f"Error parsing chat item: {e}")
#         return chat_history
#     except Exception as e:
#         logger.error(f"Error fetching history from Redis: {e}")
#         raise

def get_redis_history(r: Redis, chat_id: str, max_messages: int = 100) -> List[Message]:
    """
    Lấy lịch sử hội thoại từ Redis với định dạng chuẩn hóa.

    Args:
        r (Redis): Đối tượng Redis client.
        chat_id (str): ID của hội thoại.
        max_messages (int): Số tin nhắn tối đa trả về (mặc định 100).

    Returns:
        List[Message]: Danh sách tin nhắn (role, content, timestamp).

    Raises:
        ValueError: Nếu chat_id rỗng.
        redis.RedisError: Nếu có lỗi khi tương tác với Redis.
    """
    # Kiểm tra đầu vào
    if not chat_id:
        logger.error("chat_id không được rỗng")
        raise ValueError("chat_id là bắt buộc")

    redis_key = f"conversation:{chat_id}"
    try:
        # Kiểm tra kết nối Redis
        r.ping()

        # Lấy tin nhắn (giới hạn max_messages từ cuối)
        history_raw = r.lrange(redis_key, -max_messages, -1)
        chat_history = []

        for item in history_raw:
            try:
                parsed = json.loads(item)
                if not isinstance(parsed, dict):
                    logger.warning(f"Tin nhắn không phải dict trong {redis_key}: {item}")
                    continue

                # Kiểm tra các trường bắt buộc
                role = parsed.get("role")
                content = parsed.get("content")
                timestamp = parsed.get("timestamp")
                if not all([role, content, timestamp]):
                    logger.warning(f"Tin nhắn thiếu trường trong {redis_key}: {parsed}")
                    continue

                # Đảm bảo role hợp lệ
                if role not in ["user", "assistant"]:
                    logger.warning(f"Role không hợp lệ trong {redis_key}: {role}")
                    continue

                chat_history.append(Message(
                    role=role,
                    content=content,
                    timestamp=timestamp
                ))
            except json.JSONDecodeError as e:
                logger.error(f"Lỗi parse JSON trong {redis_key}: {item}, lỗi: {e}")
                continue
            except Exception as e:
                logger.error(f"Lỗi xử lý tin nhắn trong {redis_key}: {item}, lỗi: {e}")
                continue

        logger.info(f"Lấy {len(chat_history)} tin nhắn từ {redis_key}")
        return chat_history

    except r.RedisError as e:
        logger.error(f"Lỗi khi lấy lịch sử từ Redis cho chat_id {chat_id}: {e}")
        raise
    except Exception as e:
        logger.error(f"Lỗi không mong muốn khi lấy lịch sử từ Redis cho chat_id {chat_id}: {e}")
        return []

# def delete_chat_from_redis(r: Redis, chat_id: str):
#     # Xóa cả metadata và messages
#     r.delete(f"chat:{chat_id}:messages")
#     r.delete(f"chat:{chat_id}:meta")


def delete_chat_from_redis(r: Redis, chat_id: str) -> bool:
    """
    Xóa dữ liệu hội thoại và metadata từ Redis.

    Args:
        r (Redis): Đối tượng Redis client.
        chat_id (str): ID của hội thoại.

    Returns:
        bool: True nếu xóa thành công, False nếu thất bại.

    Raises:
        ValueError: Nếu chat_id rỗng.
        redis.RedisError: Nếu có lỗi khi tương tác với Redis.
    """
    # Kiểm tra đầu vào
    if not chat_id:
        logger.error("chat_id không được rỗng")
        raise ValueError("chat_id là bắt buộc")

    redis_key = f"conversation:{chat_id}"
    meta_key = f"chat:{chat_id}:meta"

    try:
        # Kiểm tra kết nối Redis
        r.ping()

        # Kiểm tra sự tồn tại của các key
        keys_to_delete = []
        if r.exists(redis_key):
            keys_to_delete.append(redis_key)
        if r.exists(meta_key):
            keys_to_delete.append(meta_key)

        if not keys_to_delete:
            logger.info(f"Không tìm thấy dữ liệu cho chat_id {chat_id} trong Redis")
            return True  # Không có gì để xóa, coi như thành công

        # Xóa các key
        deleted_count = r.delete(*keys_to_delete)
        logger.info(f"Đã xóa {deleted_count} key cho chat_id {chat_id}: {keys_to_delete}")
        return True

    except r.RedisError as e:
        logger.error(f"Lỗi khi xóa dữ liệu từ Redis cho chat_id {chat_id}: {e}")
        raise
    except Exception as e:
        logger.error(f"Lỗi không mong muốn khi xóa dữ liệu từ Redis cho chat_id {chat_id}: {e}")
        return False

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.now() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def create_refresh_token(email: str) -> str:
    """
    Tạo và lưu refresh token vào cơ sở dữ liệu.

    Args:
        email (str): Địa chỉ email của người dùng.

    Returns:
        str: Refresh token được tạo.

    Raises:
        HTTPException: Nếu có lỗi khi lưu token.
    """
    try:
        # Generate refresh token
        refresh_token = secrets.token_urlsafe(32)
        expiry = datetime.now() + timedelta(days=7)  # Refresh token valid for 7 days

        # Store refresh token in database
        result =  user_collection.update_one(
            {"email": email.lower()},
            {
                "$set": {
                    "refresh_token": refresh_token,
                    "refresh_token_expiry": expiry,
                    "refresh_token_timestamp": datetime.now(),
                    "revoked": False
                }
            }
        )

        if result.modified_count != 1:
            logger.error(f"Không thể lưu refresh token cho email: {email}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Lỗi khi lưu refresh token."
            )

        logger.info(f"Refresh token created for email: {email}")
        return refresh_token

    except Exception as e:
        logger.error(f"Lỗi khi tạo refresh token: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Lỗi hệ thống khi tạo refresh token."
        )


class WrappedLLMChain(Runnable):
    def __init__(self, chain):
        self.chain = chain

    def invoke(self, input: Dict[str, Any], config: dict = None, **kwargs) -> Dict[str, Any]:
        logger.info(f"WrappedLLMChain input: {input}")
        response = self.chain.invoke(input, config=config, **kwargs)
        logger.info(f"WrappedLLMChain raw response: {response}")

        # Handle ConversationalRetrievalChain response
        if isinstance(response, dict) and "answer" in response:
            # Preserve the answer and any additional keys (e.g., source_documents)
            result = {"answer": response["answer"]}
            if "source_documents" in response:
                result["source_documents"] = response["source_documents"]
        else:
            # Fallback for unexpected response formats
            result = {"answer": str(response)}

        logger.info(f"WrappedLLMChain processed result: {result}")
        return result

    def with_config(self, config):
        self.chain.with_config(config)
        return self


# === Vietnamese text processing ===
def remove_vietnamese_accents(text: str) -> str:
    """
    Remove all Vietnamese diacritical marks from a text.
    """
    text = unicodedata.normalize('NFD', text)
    text = re.sub(r'[\u0300-\u036f]', '', text)
    # Replace đ/Đ with d/D
    return text.replace('đ', 'd').replace('Đ', 'D')

# def build_unaccented_dictionary() -> Dict[str, str]:
#     """
#     Build a dictionary mapping unaccented forms to accented forms.
#     """
#     unaccented_dict = {}

#     # Process main dictionary
#     for accented, unaccented in [(v, k) for k, v in VIETNAMESE_DICTIONARY.items()]:
#         unaccented_dict[unaccented] = accented

#     # Also map unaccented form to itself (for words that are already in unaccented form)
#     for word in VIETNAMESE_DICTIONARY.keys():
#         unaccented = remove_vietnamese_accents(word)
#         if unaccented not in unaccented_dict:
#             unaccented_dict[unaccented] = word

#     # Also add the accented form as a key to itself
#     for word in VIETNAMESE_DICTIONARY.values():
#         unaccented_dict[word] = word

#     # Process 3-word phrases
#     for accented, unaccented in [(v, k) for k, v in THREE_WORD_PHRASES.items()]:
#         unaccented_dict[unaccented] = accented

#     return unaccented_dict

def is_vietnamese_question(text: str) -> bool:
    """Detect if a text is likely a Vietnamese question."""
    text_lower = text.lower()
    if text.strip().endswith('?'):
        return True

    # Kết hợp các pattern để giảm số lần quét
    question_patterns = re.compile(
        r'\b(ai|gì|gi|nào|nao|thế nào|the nao|bao (nhiêu|giờ|lâu)|ở đâu|o dau|khi nào|khi nao|'
        r'tại sao|tai sao|có phải|co phai|làm sao|lam sao|như thế nào|nhu the nao|hỏi|hoi|'
        r'cho hỏi|cho hoi|xin hỏi|xin hoi)\b|'
        r'\b(có|co)\b.+\b(không|khong)\b|'
        r'\b(tôi|toi)\b.+\b(muốn hỏi|muon hoi|muốn biết|muon biet)\b|'
        r'\b(vui lòng|vui long).+\b(gì|gi|cho|về|ve)\b',
        re.IGNORECASE
    )

    if question_patterns.search(text_lower):
        return True

    question_indicators = ['?', 'hỏi', 'thắc mắc', 'được không', 'có không', 'phải không', 'có phải', 'được chứ']
    return any(indicator in text_lower for indicator in question_indicators)

def complete_question_mark(text: str) -> str:
    """Add question mark if the text is a question but missing ?."""
    text = text.strip()
    if text.endswith('?') or text and text[-1] in '.!;:' or not text:
        return text
    return f"{text}?" if  is_vietnamese_question(text) else text

# UNACCENTED_TO_ACCENTED = build_unaccented_dictionary()

# def restore_vietnamese_accents(text: str) -> str:
#     """
#     Restore Vietnamese accents using an improved dictionary-based approach.
#     """
#     # If text is empty, return as is
#     if not text:
#         return text
#     if not isinstance(text, str):
#         raise ValueError(f"Expected string, got {type(text)}: {text}")
#     # Split text into sentences for better context handling
#     sentences = re.split(r'([.!?;:])', text)
#     processed_sentences = []

#     for i in range(0, len(sentences), 2):
#         sentence = sentences[i].strip()

#         if not sentence:  # Skip empty sentences
#             if i + 1 < len(sentences):
#                 processed_sentences.append(sentences[i + 1])
#             continue

#         # Process each sentence
#         words = sentence.split()
#         result_words = []

#         # Try to match 3-word phrases first
#         i_word = 0
#         while i_word < len(words):
#             matched = False

#             # Try 3-word phrase
#             if i_word + 2 < len(words):
#                 three_word = words[i_word].lower() + " " + words[i_word + 1].lower() + " " + words[i_word + 2].lower()
#                 if three_word in THREE_WORD_PHRASES:
#                     result_words.append(THREE_WORD_PHRASES[three_word])
#                     i_word += 3
#                     matched = True
#                     continue

#             # Try 2-word phrase
#             if i_word + 1 < len(words):
#                 two_word = words[i_word].lower() + " " + words[i_word + 1].lower()
#                 if two_word in VIETNAMESE_DICTIONARY:
#                     result_words.append(VIETNAMESE_DICTIONARY[two_word])
#                     i_word += 2
#                     matched = True
#                     continue

#             # Try single word
#             word_lower = words[i_word].lower()
#             if word_lower in VIETNAMESE_DICTIONARY:
#                 result_words.append(VIETNAMESE_DICTIONARY[word_lower])
#             elif word_lower in UNACCENTED_TO_ACCENTED:
#                 result_words.append(UNACCENTED_TO_ACCENTED[word_lower])
#             else:
#                 # No match found, keep original word
#                 result_words.append(words[i_word])

#             i_word += 1

#         # Join words back into a sentence
#         processed_sentence = " ".join(result_words)

#         # Preserve original capitalization of first word
#         if sentence and len(processed_sentence) > 0:
#             if sentence[0].isupper():
#                 processed_sentence = processed_sentence[0].upper() + processed_sentence[1:]

#         processed_sentences.append(processed_sentence)

#         # Add punctuation back if it exists
#         if i + 1 < len(sentences):
#             processed_sentences.append(sentences[i + 1])

#     # Join sentences back together
#     result_text = "".join(processed_sentences)

#     return result_text

# def preprocess_vietnamese_query(query: str) -> Dict[str, str]:
#     """
#     Preprocess Vietnamese query:
#     1. Add question mark if needed
#     2. Restore accents if missing
#     3. Create accented/unaccented versions

#     Returns:
#         Dictionary with various processed forms of the query
#     """
#     results = {
#         "original": query,  # Original query as received
#     }

#     # Check if original query has accents
#     normalized_query = remove_vietnamese_accents(query)
#     results["normalized"] = normalized_query  # Normalized (no accents) version
#     results["had_accents"] = (query != normalized_query)  # Boolean: did original have accents?

#     # Complete with question mark if needed
#     processed_query = complete_question_mark(query)
#     results["processed"] = processed_query  # Original with question mark if needed

#     # If query doesn't have accents, restore them
#     if not results["had_accents"]:
#         accented_query = restore_vietnamese_accents(processed_query)
#         # Apply question mark to accented version too
#         if processed_query.endswith('?') and not accented_query.endswith('?'):
#             accented_query += '?'
#         results["accented"] = accented_query  # Restored accents + question mark
#     else:
#         results["accented"] = processed_query  # Already has accents + question mark

#     # Always provide normalized version of the processed query (with question mark if appropriate)
#     results["normalized_processed"] = remove_vietnamese_accents(processed_query)

#     # Is it a question?
#     results["is_question"] = is_vietnamese_question(query)

#     return results



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

def load_mappings(file_path: str = "mappings.json") -> List[Dict]:
    """Load mappings from JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get("mappings", [])
    except FileNotFoundError:
        logger.error(f"Mapping file {file_path} not found.")
        return []
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in {file_path}.")
        return []

def normalize_vietnamese(text: str) -> str:
    """
    Chuẩn hóa các từ không dấu thành có dấu dựa trên từ điển.
    """
    # Sắp xếp từ điển theo độ dài giảm dần để ưu tiên cụm từ dài
    sorted_dict = sorted(VIETNAMESE_DICTIONARY.items(), key=lambda x: len(x[0]), reverse=True)
    text_no_diacritics = unidecode(text).lower()

    for wrong, correct in sorted_dict:
        wrong_no_diacritics = unidecode(wrong).lower()
        pattern = r'\b' + re.escape(wrong_no_diacritics) + r'\b'
        if re.search(pattern, text_no_diacritics, re.IGNORECASE):
            text = re.sub(pattern, correct, text_no_diacritics, flags=re.IGNORECASE)
            text_no_diacritics = re.sub(pattern, correct, text_no_diacritics, flags=re.IGNORECASE)

    return text

# Danh sách từ dừng tiếng Việt (có thể mở rộng)
STOP_WORDS = {
    # Từ liên kết và quan hệ
    "thì", "mà", "là", "và", "của", "với", "tại", "ở", "cho", "để", "nếu", "khi", "đã", "đang", "sẽ",
    "cùng", "hoặc", "nhưng", "bởi", "vì", "do", "bằng", "trong", "ngoài", "giữa", "trên", "dưới", "từ",
    "đến", "ra", "vào", "lên", "xuống", "qua", "lại",

    # Từ chỉ thời gian hoặc trạng thái
    "hiện", "nay", "bây giờ", "trước", "sau", "luôn", "thường", "vẫn", "được", "chưa", "rồi", "từng",
    "mới", "gần", "xa", "khoảng", "hôm", "ngày", "tháng", "năm",

    # Từ chỉ mức độ hoặc số lượng
    "rất", "quá", "hơi", "khá", "nhiều", "ít", "mấy", "bao nhiêu", "một", "hai", "ba", "các", "mọi",
    "tất cả", "mỗi", "vài", "nào", "ai", "gì", "đâu", "nơi",

    # Từ dừng pháp lý hoặc giao tiếp
    "về", "liên quan", "đối với", "theo", "thuộc", "bên", "này", "kia", "đấy", "đó", "đây", "như",
    "vậy", "thế", "nên", "vì thế", "do đó", "tuy", "dù", "mặc dù", "nếu như", "trừ khi"
}

@lru_cache(maxsize=1000)
def preprocess_input(text: str) -> str:
    """
    Tiền xử lý câu hỏi: sửa lỗi chính tả, loại bỏ ký tự đặc biệt, chuẩn hóa khoảng trắng,
    bảo vệ cụm từ quan trọng, và loại bỏ từ dừng.

    Args:
        text (str): Câu hỏi gốc.

    Returns:
        str: Câu hỏi đã được tiền xử lý.

    Raises:
        ValueError: Nếu input rỗng hoặc chỉ chứa ký tự không hợp lệ.
    """
    if not text or not text.strip():
        logger.error("Input rỗng hoặc không hợp lệ")
        raise ValueError("Input không được rỗng")

    logger.debug(f"Input gốc: {text}")

    # Loại bỏ ký tự đặc biệt
    text = re.sub(r'[^\w\s.,!?]', '', text)

    # Chuẩn hóa khoảng trắng
    text = re.sub(r'\s+', ' ', text).strip()

    # Chuyển về chữ thường
    text = text.lower()

    # Bảo vệ cụm từ quan trọng trước khi xử lý
    protected_phrases = ["đèn đỏ", "vượt đèn đỏ"]  # Có thể mở rộng
    placeholders = {}
    for i, phrase in enumerate(protected_phrases):
        placeholder = f"__PROTECTED_{i}__"
        placeholders[placeholder] = phrase
        text = text.replace(phrase.lower(), placeholder)

    # Chuẩn hóa dấu tiếng Việt
    text = normalize_vietnamese(text)

    # Sửa lỗi chính tả bằng từ điển
    sorted_dict = sorted(VIETNAMESE_DICTIONARY.items(), key=lambda x: len(x[0]), reverse=True)
    changes = []
    for wrong, correct in sorted_dict:
        pattern = r'\b' + re.escape(wrong) + r'\b'
        if re.search(pattern, text, re.IGNORECASE):
            text = re.sub(pattern, correct, text, flags=re.IGNORECASE)
            changes.append(f"{wrong} → {correct}")

    # Khôi phục cụm từ được bảo vệ
    for placeholder, phrase in placeholders.items():
        text = text.replace(placeholder, phrase)

    # Loại bỏ từ dừng
    words = text.split()
    words = [word for word in words if word not in STOP_WORDS]
    text = ' '.join(words)

    # Kiểm tra output
    if not text.strip():
        logger.warning("Output rỗng sau khi tiền xử lý")
        raise ValueError("Không thể tiền xử lý: output rỗng")

    if changes:
        logger.info(f"Sửa lỗi chính tả: {', '.join(changes)}")
    logger.info(f"Đã tiền xử lý: {text}")

    return text

def process_with_groq (groq_client,input: str) -> str:
    try:
        PRE_PROCESS_INPUT_PROMPT = f"""
            Bạn là một trợ lý pháp lý chuyên xử lý và chuẩn hóa câu hỏi liên quan đến pháp luật Việt Nam, nhằm tối ưu việc truy xuất tài liệu pháp lý mới nhất. Nhiệm vụ của bạn là:

            1. Thêm dấu câu (chấm, phẩy, hỏi...) để làm rõ nghĩa câu hỏi nếu còn thiếu.
            2. Loại bỏ từ dư thừa hoặc từ lặp lại không cần thiết.
            3. Thay thế các từ thông dụng hoặc mơ hồ bằng **thuật ngữ pháp lý chính xác và phổ biến trong văn bản pháp luật** (ví dụ: "bị phạt bao nhiêu tiền" → "mức xử phạt hành chính").
            4. Bổ sung các **từ khóa chuyên ngành pháp lý** phù hợp như: "quy định", "nghị định", "pháp luật hiện hành", "mức xử phạt", "trách nhiệm pháp lý", "thẩm quyền", v.v... nếu cần thiết để tăng khả năng truy xuất tài liệu pháp lý đúng.
            5. Ưu tiên bổ sung các cụm từ nhấn mạnh **tính cập nhật, hiệu lực hiện hành của pháp luật**, như: "theo pháp luật hiện hành", "theo nghị định mới nhất", "quy định hiện nay".
            6. Giữ nguyên ý nghĩa gốc của câu hỏi, không được làm thay đổi bản chất hoặc mục đích tra cứu.
            7. Chỉ trả về **câu hỏi đã được xử lý**, không kèm giải thích.

            Ví dụ:
            - Input: "liet ke nhung nguoi co the thua ke"
            - Output: "Liệt kê những người được quyền thừa kế theo quy định pháp luật hiện hành."
            - Input: "thue thu nhap ca nhan o vn"
            - Output: "Quy định về thuế thu nhập cá nhân tại Việt Nam theo pháp luật hiện hành."
            - Input: "xe may vuot den do bi phat bao nhieu tien"
            - Output: "Mức xử phạt hành chính đối với hành vi điều khiển xe máy vượt đèn đỏ theo quy định pháp luật giao thông đường bộ hiện hành."

            Câu hỏi gốc: "{input}"
            """

        response = groq_client.chat.completions.create(
            model="llama3-8b-8192",  # Mô hình nhẹ, miễn phí trên Groq llama3-8b-8192
            messages=[
                {"role": "system", "content": "Bạn là một trợ lý pháp lý."},
                {"role": "user", "content": PRE_PROCESS_INPUT_PROMPT}
            ],
            max_tokens=200,
            temperature=0.7
        )
        processed_text = response.choices[0].message.content.strip()
        return processed_text
    except Exception as e:
        print(f"Lỗi khi gọi Groq API: {e}")
        return input

def save_chat_to_mongo(conversations_collection,chat_id: str, user_email: str,user_question_content: str, # Nội dung câu hỏi
    assistant_answer_content: str, # Nội dung trả lời
    user_question_timestamp: datetime,
    assistant_answer_timestamp: datetime):
    user_message = {
        "role": "user",
        "content": user_question_content,
        "timestamp": user_question_timestamp
    }
    assistant_message = {
        "role": "assistant",
        "content": assistant_answer_content,
        "timestamp": assistant_answer_timestamp
    }
    conversation = conversations_collection.find_one({"conversation_id": chat_id})
    if not conversation:
        conversation = {
            "user_id": user_email,
            "conversation_id": chat_id,
            "messages": [user_message, assistant_message],
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }
        conversations_collection.insert_one(conversation)
    else:
        conversations_collection.update_one(
            {"conversation_id": chat_id},
            {
                "$push": {"messages": {"$each": [user_message, assistant_message]}},
                "$set": {"updated_at": datetime.now()}
            }
        )



async def get_langchain_chat_history(app_state, chat_id: str) -> RedisChatMessageHistory:
    """
    Retrieves and synchronizes chat history for Langchain.
    """
    redis_url = os.environ.get("REDIS_URL_LANGCHAIN", os.environ.get("REDIS_URL"))
    if not redis_url:
        raise ValueError("Redis URL for chat history is required.")

    # Đây là history mà Langchain sẽ sử dụng để đọc/ghi
    langchain_chat_history = RedisChatMessageHistory( # Hoặc RedisChatMessageHistoryAsync
        url=redis_url,
        session_id=chat_id,
        ttl=86400, # 1 day
    )

    # Đồng bộ hóa: Lấy từ key "source of truth" của chúng ta và nạp vào key của Langchain
    messages_key = f"conversation_messages:{chat_id}"
    # Sử dụng await nếu redis client của app_state là async
    raw_messages_from_our_redis = await app_state.redis.lrange(messages_key, 0, -1)

    # Xóa history cũ trong key của Langchain để tránh trùng lặp khi đồng bộ
    # Nếu dùng RedisChatMessageHistoryAsync: await langchain_chat_history.aclear()
    langchain_chat_history.clear() # Cho bản đồng bộ

    for msg_json_bytes in raw_messages_from_our_redis:
        msg_data = json.loads(msg_json_bytes.decode()) # decode bytes to str
        message = Message(**msg_data) # Validate

        if message.role == "user":
            # Nếu dùng RedisChatMessageHistoryAsync: await langchain_chat_history.aadd_user_message(message.content)
            langchain_chat_history.add_user_message(message.content)
        elif message.role == "assistant":
            # await langchain_chat_history.aadd_ai_message(message.content)
            langchain_chat_history.add_ai_message(message.content)

    return langchain_chat_history