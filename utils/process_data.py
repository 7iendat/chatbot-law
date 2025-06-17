import re
import os
from typing import List, Dict, Optional, Tuple, Union
import logging
from tqdm import tqdm
import uuid
import json
from langchain_core.documents import Document


logger = logging.getLogger(__name__)


def filter_and_serialize_complex_metadata(documents: List[Document]) -> List[Document]:
    """Hàm cuối cùng để chuẩn hóa metadata ngay trước khi ingest vào vector store."""
    updated_documents = []
    allowed_types = (str, bool, int, float, type(None))
    serialize_keys = ["penalties", "cross_references"]

    for doc in documents:
        filtered_metadata = {}
        for key, value in doc.metadata.items():
            if key in serialize_keys and value: # Chỉ serialize nếu value không rỗng
                try:
                    filtered_metadata[key] = json.dumps(value, ensure_ascii=False, default=str)
                except TypeError:
                    logger.warning(f"Không thể serialize key '{key}' cho doc ID {doc.id}. Chuyển thành string.")
                    filtered_metadata[key] = str(value)
            elif isinstance(value, allowed_types):
                filtered_metadata[key] = value
            elif isinstance(value, list):
                filtered_metadata[key] = json.dumps(value, ensure_ascii=False)
            else:
                filtered_metadata[key] = str(value)
        doc.metadata = filtered_metadata
        updated_documents.append(doc)
    return updated_documents


# Các hằng số
LEGAL_DOC_TYPES = ["Luật", "Bộ luật", "Nghị định", "Thông tư", "Quyết định", "Pháp lệnh", "Nghị quyết", "Chỉ thị", "Hiến pháp"]
MAX_CHUNK_SIZE = 2500  # Kích thước tối đa cho một chunk trước khi bị chia nhỏ hơn
CHUNK_OVERLAP = 200    # Độ chồng lấn khi chia nhỏ chunk quá lớn

class SimpleTextSplitter:
    """Một text splitter đơn giản để chia nhỏ các chunk quá lớn."""
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        if not text: return []
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunks.append(text[i : i + self.chunk_size])
        return chunks

base_text_splitter = SimpleTextSplitter(chunk_size=MAX_CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

def generate_structured_id(doc_so_hieu: Optional[str], structure_path: List[str], filename: str) -> str:
    """
    CẢI TIẾN: Tạo ra một chuỗi UUID v5 nhất quán từ ID có cấu trúc.
    Thêm filename để đảm bảo tính duy nhất.
    """
    # Ưu tiên so_hieu, nhưng fallback về filename để tránh "unknown-document"
    base_id = doc_so_hieu if doc_so_hieu else filename
    safe_base_id = re.sub(r'[/\s\.]', '-', base_id) # Thay thế các ký tự không an toàn
    path_str = '_'.join(structure_path)

    # Đảm bảo unique_string_id khác nhau cho mỗi file
    unique_string_id = f"{safe_base_id}_{path_str}"

    generated_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, unique_string_id)
    return str(generated_uuid)



def general_ocr_corrections(text: str) -> str:
    """Sửa các lỗi OCR phổ biến."""
    corrections = {
        "LuatWielnam": "LuatVietnam", "LualVietnam": "LuatVietnam",
        "aflu atvistnarn.vni": "@luatvietnam.vn", "Tien ch van bán luat": "Tiện ích văn bản luật",
        "teeeeokanlbaglueloen": "",
        "Tee===": "", "Tc=e===": "", "nem": "",
        "SN Hntlin sa:": "", "HT:": "", "Hntlin sa:": "",
        r"([a-z])([A-Z])": r"\1 \2",
        "Nghịđịnh": "Nghị định", " điểu ": " điều ", "Chưong": "Chương",
        " điềm ": " điểm ", "khoån": "khoản", "Chínhphủ": "Chính phủ",
        " điềukhỏan": " điều khoản", "LuậtTổ": "Luật Tổ", "LuậtXử": "Luật Xử",
        "LuậtTrật": "Luật Trật", " điềuchỉnh": " điều chỉnh", " cá_nhân": " cá nhân",
        " tổ_chức": " tổ chức", " hành_chính": " hành chính", " giấy_phép": " giấy phép",
        " lái_xe": " lái xe", " Giao_thông": " Giao thông",
        # 'ó': '6', 'Ò': '0', 'ọ': '0', 'l': '1', 'I': '1', 'i': '1',
        # 'Z': '2', 'z': '2', 'B': '8',
    }
    for wrong, right in corrections.items():
        if wrong.startswith(r"([a-z])([A-Z])"):
            text = re.sub(wrong, right, text)
        else:
            text = text.replace(wrong, right)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def clean_document_text(text: str) -> str:
    """Làm sạch văn bản, loại bỏ header, footer, nhiễu."""
    # Giữ lại phần đầu có thể chứa metadata quan trọng cho extract_document_metadata
    # Chỉ loại bỏ các phần header/footer rõ ràng và nhiễu chung
    text_lines = text.splitlines()

    # Tìm điểm bắt đầu của nội dung chính (sau metadata đầu văn bản)
    start_content_index = 0
    for i, line in enumerate(text_lines):
        # Các dấu hiệu kết thúc phần metadata đầu văn bản và bắt đầu nội dung chính
        if re.match(r"^\s*(?:PHẦN CHUNG|LỜI NÓI ĐẦU|Chương\s+[IVXLCDM]+|Điều\s+1)", line, re.IGNORECASE):
            start_content_index = i
            break
        # Hoặc sau các dòng "Căn cứ...", "Theo đề nghị..."
        if line.strip().lower().startswith(("căn cứ", "theo đề nghị của")):
             # Tìm dòng trống hoặc dòng cấu trúc tiếp theo làm điểm bắt đầu
            for j in range(i + 1, len(text_lines)):
                if not text_lines[j].strip() or \
                   re.match(r"^\s*(?:PHẦN CHUNG|LỜI NÓI ĐẦU|Chương\s+[IVXLCDM]+|Điều\s+1)", text_lines[j], re.IGNORECASE):
                    start_content_index = j
                    break
            else: # Nếu không tìm thấy, dùng dòng ngay sau "Căn cứ"
                start_content_index = i + 1
            break

    head_section_to_keep = "\n".join(text_lines[:start_content_index])
    main_content_and_footer = "\n".join(text_lines[start_content_index:])

    # Loại bỏ footer (Nơi nhận, chữ ký) từ phần main_content_and_footer
    cleaned_main_content = re.sub(r"Nơi nhận:[\s\S]*?(?:TM\.\s*CHÍNH PHỦ|TM\.\s*BAN BÍ THƯ|CHỦ TỊCH QUỐC HỘI|THỦ TƯỚNG)[\s\S]*$", "", main_content_and_footer, flags=re.MULTILINE | re.IGNORECASE)
    cleaned_main_content = re.sub(r"\s*\(Đã ký\)\s*$", "", cleaned_main_content, flags=re.MULTILINE | re.IGNORECASE)
    cleaned_main_content = re.sub(r"^\s*[A-ZÀ-Ỹ\s]{5,}\s*$", "", cleaned_main_content, flags=re.MULTILINE) # Loại bỏ tên người ký nếu nó đứng một mình, viết hoa

    # Nối lại phần đầu và phần nội dung đã làm sạch footer
    text = head_section_to_keep + "\n" + cleaned_main_content

    # Loại bỏ các dòng nhiễu chung của LuatVietnam và các dòng trống
    lines = text.splitlines()
    cleaned_lines = []
    luatvn_noise_patterns = [
        r"^\s*LuatVietnam(?:\.vn)?.*$",
        r"^\s*Tiện ích văn bản luật\s*$",
        r"^\s*www\.vanbanluat\.vn\s*$",
        r"^\s*\[\s*Hình\s*ảnh\s*]\s*$",
        r"^[=*_\-]{5,}$",
        r"^\s*Trang \d+ / \d+\s*$",
        r"^\s*LuatVietnam\.vn\s+Luật Việt Nam\s+Cơ sở dữ liệu văn bản pháp luật lớn nhất Việt Nam.*$", # Các dòng quảng cáo
        r"^\s*Hotline:\s*\d{4}\.\d{3}\.\d{3}.*Email:.*",
        r"^\s*Đặt mua văn bản gốc.*",
    ]
    luatvn_noise_regex = [re.compile(pat, flags=re.IGNORECASE) for pat in luatvn_noise_patterns]

    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            continue
        if any(regex.match(stripped_line) for regex in luatvn_noise_regex):
            continue
        cleaned_lines.append(stripped_line)

    cleaned_text = "\n".join(cleaned_lines)
    cleaned_text = general_ocr_corrections(cleaned_text)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
    return cleaned_text.strip()

def extract_document_metadata(raw_text: str, filename: str) -> Dict[str, Optional[Union[str, int]]]:
    """Trích xuất metadata từ raw_text (số hiệu, loại, tên, ngày/năm, cơ quan ban hành)."""
    metadata: Dict[str, Optional[Union[str, int]]] = {
        "so_hieu": None, "loai_van_ban": None, "ten_van_ban": None,
        "ngay_ban_hanh_str": None, "nam_ban_hanh": None,
        "co_quan_ban_hanh": None, "ngay_hieu_luc_str": None,
    }
    # Phân tích ~30 dòng đầu hoặc 2500 ký tự đầu của raw_text
    head_text_lines = raw_text.splitlines()[:30]
    head_text = "\n".join(head_text_lines)
    if len(head_text) > 2500:
        head_text = head_text[:2500]

    head_text = general_ocr_corrections(head_text) # Sửa lỗi OCR cho phần đầu trước khi trích xuất

    # 1. Cơ quan ban hành (Thường ở đầu tiên)
    issuing_body_patterns = [
        r"^\s*(CHÍNH PHỦ)", r"^\s*(QUỐC HỘI)", r"^\s*(BỘ TRƯỞNG\s+BỘ\s+[\w\sÀ-Ỹà-ỹ]+)",
        r"^\s*(THỦ TƯỚNG\s+CHÍNH PHỦ)", r"^\s*(CHỦ TỊCH\s+NƯỚC)",
        r"^\s*(HỘI ĐỒNG THẨM PHÁN TOÀ ÁN NHÂN DÂN TỐI CAO)", r"^\s*(ỦY BAN THƯỜNG VỤ QUỐC HỘI)"
    ]
    for pattern in issuing_body_patterns:
        match = re.search(pattern, head_text, re.MULTILINE | re.IGNORECASE)
        if match:
            metadata["co_quan_ban_hanh"] = match.group(1).strip().upper()
            break

    # 2. Số hiệu
    so_hieu_match = re.search(r"Số\s*:\s*([\w\d/.-]+(?:-\w+/\w+-\w+)?(?:/\w+-\w+)?)\s*(?:\n|\r)", head_text, re.IGNORECASE)
    if so_hieu_match:
        metadata["so_hieu"] = so_hieu_match.group(1).strip()

    # 3. Ngày ban hành và Năm ban hành
    date_location_patterns = [
        r"(?:Hà Nội|[\w\s.]+),\s*ngày\s*(\d{1,2})\s*tháng\s*(\d{1,2})\s*năm\s*(\d{4})",
        r"ngày\s*(\d{1,2})\s*tháng\s*(\d{1,2})\s*năm\s*(\d{4})"
    ]
    found_date = False
    if metadata["so_hieu"]:
        idx_so_hieu = head_text.find(str(metadata["so_hieu"]))
        if idx_so_hieu != -1:
            search_area = head_text[idx_so_hieu : min(len(head_text), idx_so_hieu + 250)] # Mở rộng search area
            for pattern in date_location_patterns:
                date_match = re.search(pattern, search_area, re.IGNORECASE)
                if date_match:
                    day, month, year = date_match.groups()[-3:]
                    metadata["ngay_ban_hanh_str"] = f"ngày {day} tháng {month} năm {year}"
                    metadata["nam_ban_hanh"] = int(year)
                    found_date = True
                    break
    if not found_date:
        for pattern in date_location_patterns:
            date_match = re.search(pattern, head_text, re.IGNORECASE)
            if date_match:
                day, month, year = date_match.groups()[-3:]
                metadata["ngay_ban_hanh_str"] = f"ngày {day} tháng {month} năm {year}"
                metadata["nam_ban_hanh"] = int(year)
                break

    if not metadata["nam_ban_hanh"] and metadata["so_hieu"]:
        year_match_in_so_hieu = re.search(r"/(\d{4})/", str(metadata["so_hieu"])) or \
                                re.search(r"-(\d{4})-", str(metadata["so_hieu"]))
        if year_match_in_so_hieu: metadata["nam_ban_hanh"] = int(year_match_in_so_hieu.group(1))
    if not metadata["nam_ban_hanh"]:
        year_filename_match = re.search(r"[-_](\d{4})[-_.]", filename) # Năm trong tên file thường có gạch nối/chấm
        if not year_filename_match: year_filename_match = re.search(r"(\d{4})", filename)
        if year_filename_match: metadata["nam_ban_hanh"] = int(year_filename_match.group(1))

    # 4. Loại văn bản và Tên văn bản
    loai_vb_ten_vb_patterns = [
        # Pattern cho loại VB và tên VB nằm trên các dòng khác nhau hoặc cùng dòng
        # Ưu tiên bắt cụm (LOẠI VĂN BẢN \n TÊN VĂN BẢN) hoặc (LOẠI VĂN BẢN TÊN VĂN BẢN)
        r"^\s*(NGHỊ ĐỊNH|BỘ LUẬT|LUẬT|THÔNG TƯ|QUYẾT ĐỊNH|PHÁP LỆNH|NGHỊ QUYẾT|CHỈ THỊ)\s*\n+\s*((?:[\w\sÀ-Ỹà-ỹ\d()/'.,-]+)(?:\n[\w\sÀ-Ỹà-ỹ\d()/'.,-]+)*)\s*(?=\n(?:Căn cứ|Theo đề nghị|Chương I|PHẦN CHUNG|Điều 1)|$)",
        r"^\s*(NGHỊ ĐỊNH|BỘ LUẬT|LUẬT|THÔNG TƯ|QUYẾT ĐỊNH|PHÁP LỆNH|NGHỊ QUYẾT|CHỈ THỊ)\s+((?:[\w\sÀ-Ỹà-ỹ\d()/'.,-]+)(?:\n[\w\sÀ-Ỹà-ỹ\d()/'.,-]+)*)\s*(?=\n(?:Căn cứ|Theo đề nghị|Chương I|PHẦN CHUNG|Điều 1)|$)"
    ]
    found_type_and_title = False
    for pattern_str in loai_vb_ten_vb_patterns:
        match = re.search(pattern_str, head_text, re.MULTILINE | re.IGNORECASE)
        if match:
            metadata["loai_van_ban"] = match.group(1).strip().upper()
            raw_title = match.group(2).strip()
            # Làm sạch tên VB: loại bỏ các dòng chỉ có gạch ngang, số hiệu (nếu lẫn vào)
            title_lines = [line.strip() for line in raw_title.split('\n') if line.strip() and not re.match(r"^-+$", line.strip())]
            cleaned_title = " ".join(title_lines)
            if metadata["so_hieu"] and metadata["so_hieu"] in cleaned_title: # Loại bỏ số hiệu nếu lẫn
                cleaned_title = cleaned_title.replace(metadata["so_hieu"], "").strip()
            metadata["ten_van_ban"] = re.sub(r"\s+", " ", cleaned_title).strip()
            found_type_and_title = True
            break

    # Fallback nếu không tìm được theo cụm
    if not found_type_and_title:
        loai_vb_simple_patterns = [
            r"\b(NGHỊ ĐỊNH)\b", r"\b(BỘ LUẬT)\b", r"\b(LUẬT)\b",
            r"\b(THÔNG TƯ)\b", r"\b(QUYẾT ĐỊNH)\b", r"\b(PHÁP LỆNH)\b",
            r"\b(NGHỊ QUYẾT)\b", r"\b(CHỈ THỊ)\b"
        ]
        for pattern_str in loai_vb_simple_patterns:
            loai_match = re.search(pattern_str, head_text, re.IGNORECASE)
            if loai_match:
                metadata["loai_van_ban"] = loai_match.group(1).upper()
                # Thử tìm tên văn bản sau loại, trước các keyword kết thúc
                start_search_title = loai_match.end()
                end_title_keywords = [r"Căn cứ", r"Theo đề nghị", r"Chương\s+I", r"PHẦN CHUNG", r"LỜI NÓI ĐẦU", r"Điều\s+1"]
                end_search_pos = len(head_text)
                for keyword in end_title_keywords:
                    kw_match = re.search(keyword, head_text[start_search_title:], re.IGNORECASE)
                    if kw_match:
                        end_search_pos = min(end_search_pos, start_search_title + kw_match.start())

                title_block = head_text[start_search_title:end_search_pos].strip()
                title_lines = [line.strip() for line in title_block.split('\n') if line.strip() and (line.isupper() or (line[0].isupper() and len(line.split()) > 1)) and not line.lower().startswith(("căn cứ", "theo đề nghị"))]
                if title_lines:
                    raw_title = " ".join(title_lines).strip()
                    if metadata["so_hieu"] and metadata["so_hieu"] in raw_title:
                        raw_title = raw_title.replace(metadata["so_hieu"], "").strip()
                    metadata["ten_van_ban"] = re.sub(r"\s+", " ", raw_title).strip()
                break

    # 5. Ngày hiệu lực (tìm trong toàn bộ raw_text vì có thể ở cuối)
    eff_date_text = general_ocr_corrections(raw_text[-1000:]) # Kiểm tra 1000 ký tự cuối
    effective_date_match = re.search(r"(?:Nghị định|Luật|Thông tư)\s+này\s+có\s+hiệu\s+lực\s+(?:thi\s+hành\s+)?(?:kể\s+)?từ\s+ngày\s*(\d{1,2})\s*tháng\s*(\d{1,2})\s*năm\s*(\d{4})", eff_date_text, re.IGNORECASE)
    if effective_date_match:
        day, month, year = effective_date_match.groups()
        metadata["ngay_hieu_luc_str"] = f"ngày {day} tháng {month} năm {year}"

    if metadata["ngay_ban_hanh_str"]:
        metadata["ngay_ban_hanh_str"] = str(metadata["ngay_ban_hanh_str"]) # Đảm bảo là string
    if metadata["ngay_hieu_luc_str"]:
        metadata["ngay_hieu_luc_str"] = str(metadata["ngay_hieu_luc_str"])

    return metadata


def parse_law_item_line(line: str) -> Tuple[Optional[str], Optional[str], str]:
    line = line.strip()
    phan_match = re.match(r"^\s*(PHẦN\s+(?:THỨ\s+[\w\sÀ-Ỹà-ỹ]+|[IVXLCDM]+|CHUNG|CÁC TỘI PHẠM))\s*[:.]?\s*(.*)", line, re.IGNORECASE)
    if phan_match: return "phan", phan_match.group(1).strip(), phan_match.group(2).strip()

    chuong_match = re.match(r"^\s*(Chương\s+[IVXLCDM\d]+)\s*[:.]?\s*(.*)", line, re.IGNORECASE)
    if chuong_match: return "chuong", chuong_match.group(1).strip(), chuong_match.group(2).strip()

    muc_match = re.match(r"^\s*(Mục\s+\d+)\s*[:.]?\s*(.*)", line, re.IGNORECASE)
    if muc_match: return "muc", muc_match.group(1).strip(), muc_match.group(2).strip()

    dieu_match = re.match(r"^\s*(Điều\s+\d+[a-z]?)\s*[\.:\s]\s*(.*?)(?=\n|$)", line, re.IGNORECASE)
    if dieu_match: return "dieu", dieu_match.group(1).strip(), dieu_match.group(2).strip()

    khoan_match = re.match(r"^\s*(\d+)\.\s*(.+)", line)
    if khoan_match:
        content_after_number = khoan_match.group(2).strip()
        # Heuristic: Nếu nội dung sau số không bắt đầu bằng một "điểm" (ví dụ "1. a) ...")
        # VÀ nội dung có vẻ là một câu (bắt đầu bằng chữ hoa và có nhiều hơn 1-2 từ)
        # thì coi là Khoản.
        if not re.match(r"^\s*[a-zđ]\)", content_after_number):
            if content_after_number and content_after_number[0].isupper() and len(content_after_number.split()) > 1 : #  and content_after_number.endswith(('.', ';', ':'))
                # Điều kiện kết thúc bằng dấu câu có thể quá chặt, nhiều khoản không có.
                return "khoan", khoan_match.group(1).strip(), content_after_number

    diem_match = re.match(r"^\s*([a-zđ])\)\s*(.+)", line)
    if diem_match: return "diem", diem_match.group(1).strip(), diem_match.group(2).strip()

    tiet_match = re.match(r"^\s*[-–—]\s*(.+)", line)
    if tiet_match: return "tiet", "-", tiet_match.group(1).strip()

    return None, None, line # Là dòng văn bản thường



def extract_cross_references(text_chunk_content: str, current_doc_full_metadata: Dict) -> List[Dict]:
    references = []
    internal_ref_patterns = [
        re.compile(r"(?:quy định tại|xem|theo|như|tại)\s+(?:(điểm\s+[a-zđ])\s+)?(?:(khoản\s+\d+)\s+)?(Điều\s+\d+[a-z]?)(?:\s+của\s+(?:Nghị định|Luật|Bộ luật|Thông tư|Pháp lệnh|Quyết định|Nghị quyết)\s+này)?", re.IGNORECASE),
        re.compile(r"(?:quy định tại|xem|theo|như|tại)\s+(?:(khoản\s+\d+)\s+)?(Điều\s+\d+[a-z]?)(?:\s+của\s+(?:Nghị định|Luật|Bộ luật|Thông tư|Pháp lệnh|Quyết định|Nghị quyết)\s+này)?", re.IGNORECASE),
        re.compile(r"(?:quy định tại|xem|theo|như|tại)\s+(Điều\s+\d+[a-z]?)(?:\s+của\s+(?:Nghị định|Luật|Bộ luật|Thông tư|Pháp lệnh|Quyết định|Nghị quyết)\s+này|\s+nêu trên|\s+dưới đây)?", re.IGNORECASE),
    ]
    # ... (phần internal_ref_patterns giữ nguyên) ...
    for pattern in internal_ref_patterns:
        for match in pattern.finditer(text_chunk_content):
            original_text_internal = match.group(0) # Lấy toàn bộ chuỗi khớp
            groups = match.groups()
            ref_diem, ref_khoan, ref_dieu = None, None, None

            # Logic xác định điểm, khoản, điều dựa trên số lượng group và nội dung
            # Pattern 1: (điểm)? (khoản)? (Điều)
            if pattern.pattern.count('(') - pattern.pattern.count('(?:') == 3: # Đếm số capturing groups
                ref_diem_text = groups[0] if groups[0] and "điểm" in groups[0].lower() else None
                ref_khoan_text = groups[1] if groups[1] and "khoản" in groups[1].lower() else None
                ref_dieu_text = groups[2] if groups[2] and "điều" in groups[2].lower() else None

                # Nếu group 1 là khoản, group 2 là điều (do điểm optional)
                if not ref_diem_text and ref_khoan_text is None and (groups[0] and "khoản" in groups[0].lower()):
                    ref_khoan_text = groups[0]
                    ref_dieu_text = groups[1]
                # Nếu group 1 là điều (do điểm và khoản optional)
                elif not ref_diem_text and not ref_khoan_text and (groups[0] and "điều" in groups[0].lower()):
                    ref_dieu_text = groups[0]


            # Pattern 2: (khoản)? (Điều)
            elif pattern.pattern.count('(') - pattern.pattern.count('(?:') == 2:
                ref_khoan_text = groups[0] if groups[0] and "khoản" in groups[0].lower() else None
                ref_dieu_text = groups[1] if groups[1] and "điều" in groups[1].lower() else None
                # Nếu group 0 là điều
                if not ref_khoan_text and (groups[0] and "điều" in groups[0].lower()):
                    ref_dieu_text = groups[0]

            # Pattern 3: (Điều)
            elif pattern.pattern.count('(') - pattern.pattern.count('(?:') == 1:
                ref_dieu_text = groups[0] if groups[0] and "điều" in groups[0].lower() else None

            ref_dieu = ref_dieu_text.replace("Điều ", "").strip() if ref_dieu_text else None
            ref_khoan = ref_khoan_text.replace("khoản ", "").strip() if ref_khoan_text else None
            ref_diem = ref_diem_text.replace("điểm ", "").strip() if ref_diem_text else None

            references.append({
                "type": "internal", "original_text": original_text_internal,
                "target_dieu": ref_dieu,
                "target_khoan": ref_khoan,
                "target_diem": ref_diem,
                "target_document_id": current_doc_full_metadata.get("so_hieu"),
                "target_document_title": current_doc_full_metadata.get("ten_van_ban")
            })


    external_ref_pattern = re.compile(
        r"(?:quy định tại|theo|tại|của|trong)\s+"
        # Group 1: Cụm điểm/khoản/điều (optional), ví dụ "điểm a khoản 1 Điều 5 " hoặc "Điều 5 "
        r"((?:điểm\s+[a-zđ]\s*)?(?:khoản\s+\d+\s*)?(?:Điều\s+\d+[a-z]?\s*)?)?"
        r"(?:của\s+)?" # Non-capturing "của "
        # Group 2: Loại VB + Tên VB, ví dụ "Luật Giao thông đường bộ" hoặc "Nghị định 100/2019/NĐ-CP"
        r"((?:Nghị định|Luật|Bộ luật|Thông tư|Pháp lệnh|Quyết định|Nghị quyết|Hiến pháp)"
        r"(?:\s+[\w\sÀ-Ỹà-ỹ\d()/'.,-]+?)?)"
        # Group 3: Số hiệu (optional), ví dụ "100/2019/NĐ-CP"
        r"(?:\s+(?:số|số hiệu)?\s*([\w\d/.-]+(?:-\d{4}-[\w\d.-]+)?))?"
        # Group 4: Năm từ ngày ban hành (optional), ví dụ "2019"
        r"(?:\s*ngày\s*\d{1,2}\s*(?:tháng|-|/)\s*\d{1,2}\s*(?:năm|-|/)\s*(\d{4}))?"
        r"(?:\s*của\s*(?:Chính phủ|Quốc hội|[\w\sÀ-Ỹà-ỹ]+))?", # Non-capturing cơ quan ban hành
        re.IGNORECASE
    )

    for match in external_ref_pattern.finditer(text_chunk_content):
        # match.groups() sẽ trả về 4 phần tử tương ứng với 4 capturing groups ở trên
        matched_groups = match.groups()
        original_text_external = match.group(0) # Toàn bộ chuỗi khớp

        provision_elements_str = matched_groups[0] if matched_groups[0] else ""
        target_doc_full_name_raw = matched_groups[1].strip() if matched_groups[1] else ""
        target_doc_number_explicit = matched_groups[2].strip() if matched_groups[2] else None
        target_doc_year_in_ref_str = matched_groups[3].strip() if matched_groups[3] else None

        # Phân tích provision_elements_str để lấy điểm, khoản, điều
        target_diem = None
        if diem_match_obj := re.search(r"điểm\s+([a-zđ])", provision_elements_str, re.IGNORECASE):
            target_diem = diem_match_obj.group(1)

        target_khoan = None
        if khoan_match_obj := re.search(r"khoản\s+(\d+)", provision_elements_str, re.IGNORECASE):
            target_khoan = khoan_match_obj.group(1)

        target_dieu = None
        if dieu_match_obj := re.search(r"Điều\s+(\d+[a-z]?)", provision_elements_str, re.IGNORECASE):
            target_dieu = dieu_match_obj.group(1)

        # Phân tích loại và tên văn bản từ target_doc_full_name_raw
        target_doc_type = None
        target_doc_title = target_doc_full_name_raw # Gán giá trị mặc định

        for doc_type_keyword in LEGAL_DOC_TYPES:
            # Sử dụng \b để khớp từ chính xác hơn và re.escape để xử lý ký tự đặc biệt nếu có
            # Khớp ở đầu chuỗi và không phân biệt hoa thường
            if re.match(rf"^{re.escape(doc_type_keyword)}\b", target_doc_full_name_raw, re.IGNORECASE):
                target_doc_type = doc_type_keyword.upper() # Chuẩn hóa về chữ hoa
                # Loại bỏ phần loại văn bản khỏi tên, và các khoảng trắng thừa
                temp_title = re.sub(rf"^{re.escape(doc_type_keyword)}\b\s*", "", target_doc_full_name_raw, count=1, flags=re.IGNORECASE)
                target_doc_title = temp_title.strip()
                break

        final_doc_number = target_doc_number_explicit
        final_doc_year = None
        if target_doc_year_in_ref_str:
            final_doc_year = int(target_doc_year_in_ref_str)

        # Nếu không có số hiệu rõ ràng, thử trích từ tên (ví dụ: "Nghị định 100/2019/NĐ-CP")
        if not final_doc_number and target_doc_title:
            # Cố gắng bắt số hiệu dạng X/YYYY/ABC-XYZ hoặc X-YYYY-ABC
            number_in_title_match = re.search(r"(\d+(?:/\d{4})?/[\w.-]+(?:-[\w\d.-]+)?|\d+-\d{4}-[\w.-]+)", target_doc_title)
            if number_in_title_match:
                final_doc_number = number_in_title_match.group(1)
                # Cập nhật lại title nếu đã lấy số hiệu ra
                target_doc_title = target_doc_title.replace(final_doc_number, "").strip()
                target_doc_title = re.sub(r"^\s*(?:số|số hiệu)\s*$", "", target_doc_title, flags=re.IGNORECASE).strip() # Bỏ "số" thừa
                target_doc_title = target_doc_title.replace("của", "").strip() # Bỏ "của" thừa nếu có

        # Nếu không có năm rõ ràng, thử trích từ số hiệu hoặc tên
        if not final_doc_year and final_doc_number:
            year_in_number_match = re.search(r"/(\d{4})/", final_doc_number) or \
                                   re.search(r"-(\d{4})-", final_doc_number)
            if year_in_number_match:
                final_doc_year = int(year_in_number_match.group(1))

        if not final_doc_year and target_doc_title:
            year_in_title_match = re.search(r"(?:năm|khóa)\s+(\d{4})", target_doc_title, re.IGNORECASE) # Thêm "khóa"
            if year_in_title_match:
                final_doc_year = int(year_in_title_match.group(1))

        # Bỏ qua nếu không có loại văn bản HOẶC (cả tên văn bản VÀ số hiệu đều không có)
        if not target_doc_type or (not target_doc_title and not final_doc_number):
            # logger.debug(f"Skipping external ref: {original_text_external} -> Type: {target_doc_type}, Title: {target_doc_title}, Number: {final_doc_number}")
            continue

        references.append({
            "type": "external",
            "original_text": original_text_external,
            "target_document_type": target_doc_type,
            "target_document_title": target_doc_title if target_doc_title else None,
            "target_document_number": final_doc_number,
            "target_document_year": final_doc_year, # Đã là int hoặc None
            "target_dieu": target_dieu,
            "target_khoan": target_khoan,
            "target_diem": target_diem,
            "target_document_year_hint": final_doc_year # Dùng final_doc_year đã được chuẩn hóa
        })
    return references


# def hierarchical_split_law_document(doc_obj: Document, max_chunk_size: int = 2500) -> List[Document]:
#     text = doc_obj.page_content
#     source_metadata = doc_obj.metadata.copy() # Metadata của cả văn bản gốc

#     chunks = []
#     current_chunk_lines = []
#     current_hierarchical_meta = {}

#     document_level_meta_keys = [
#         "so_hieu", "loai_van_ban", "ten_van_ban", "ngay_ban_hanh_str",
#         "nam_ban_hanh", "co_quan_ban_hanh", "ngay_hieu_luc_str", "source",
#         "field", "entity_type", "penalty"
#     ]

#     def create_chunk_base_metadata(dieu_code: Optional[str] = None, dieu_title: Optional[str] = None) -> Dict:
#         # Bắt đầu với metadata của văn bản gốc
#         chunk_meta = {k: v for k, v in source_metadata.items() if k in document_level_meta_keys and v is not None}
#         # Thêm metadata phân cấp hiện tại (Phần, Chương, Mục)
#         chunk_meta.update(current_hierarchical_meta)

#         title_parts = []
#         for key_code, key_title_content in [("phan_code", "phan_title"), ("chuong_code", "chuong_title"), ("muc_code", "muc_title")]:
#             if key_code in chunk_meta:
#                 title_part = chunk_meta[key_code]
#                 if chunk_meta.get(key_title_content):
#                     title_part += f": {chunk_meta[key_title_content]}"
#                 title_parts.append(title_part)

#         if dieu_code:
#             chunk_meta["dieu_code"] = dieu_code
#             chunk_meta["dieu_title"] = dieu_title if dieu_title else dieu_code
#             dieu_part = dieu_code
#             if dieu_title and dieu_title.lower() != dieu_code.lower().replace("điều", "Điều").strip():
#                  dieu_part += f": {dieu_title}"
#             title_parts.append(dieu_part)

#         # Tạo tiêu đề cho chunk
#         if title_parts:
#             chunk_meta["title"] = " - ".join(title_parts)
#         elif source_metadata.get("ten_van_ban"):
#             chunk_meta["title"] = str(source_metadata.get("ten_van_ban"))
#         else:
#             chunk_meta["title"] = str(source_metadata.get("source", "N/A"))

#         return chunk_meta

#     def flush_current_chunk():
#         nonlocal current_chunk_lines
#         if not current_chunk_lines:
#             return

#         content = "\n".join(current_chunk_lines).strip()
#         current_chunk_lines = [] # Reset buffer ngay

#         if not content: return

#         # Metadata cho chunk này (bao gồm thông tin Điều nếu có)
#         chunk_base_meta = create_chunk_base_metadata(
#             current_hierarchical_meta.get("current_dieu_code"),
#             current_hierarchical_meta.get("current_dieu_title")
#         )

#         # Làm giàu thêm cho metadata của chunk
#         penalties = extract_penalties_from_text(content)
#         if penalties: chunk_base_meta["penalties"] = penalties

#         cross_refs = extract_cross_references(content, source_metadata)
#         if cross_refs: chunk_base_meta["cross_references"] = cross_refs

#         if len(content) > max_chunk_size:
#             logger.warning(f"🔸 Chunk '{chunk_base_meta.get('title')}' quá lớn ({len(content)} chars). Splitting.")
#             # Khi chia nhỏ, mỗi sub-chunk sẽ kế thừa metadata và có ID riêng
#             meta_for_sub_chunks = chunk_base_meta.copy()
#             sub_docs_from_splitter = base_text_splitter.create_documents([content], metadatas=[meta_for_sub_chunks])

#             for i, sub_doc in enumerate(sub_docs_from_splitter):
#                 # ID đã được tạo bởi base_text_splitter.create_documents
#                 sub_doc.metadata["sub_chunk_index"] = i + 1
#                 chunks.append(sub_doc)
#         else:
#             chunk_id = str(uuid.uuid4())
#             chunks.append(Document(page_content=content, metadata=chunk_base_meta, id=chunk_id))

#     lines = text.splitlines()
#     is_preamble = True

#     for line_idx, line_text in enumerate(lines):
#         line_stripped = line_text.strip()
#         if not line_stripped:
#             continue

#         item_type, item_code, item_title_content = parse_law_item_line(line_stripped)

#         # Nếu đang ở preamble và gặp dòng cấu trúc đầu tiên (Phần, Chương, Mục, Điều)
#         if is_preamble and item_type in ["phan", "chuong", "muc", "dieu"]:
#             if current_chunk_lines: # Xả preamble chunk nếu có
#                 # Preamble chunk không có dieu_code/title cụ thể
#                 current_hierarchical_meta.pop("current_dieu_code", None)
#                 current_hierarchical_meta.pop("current_dieu_title", None)
#                 flush_current_chunk()
#             is_preamble = False # Kết thúc preamble

#         if item_type == "phan":
#             flush_current_chunk() # Xả chunk trước đó (nếu có)
#             current_hierarchical_meta = {"phan_code": item_code, "phan_title": item_title_content} # Reset, chỉ giữ phần
#         elif item_type == "chuong":
#             flush_current_chunk()
#             phan_info = {k:v for k,v in current_hierarchical_meta.items() if "phan" in k}
#             current_hierarchical_meta = {**phan_info, "chuong_code": item_code, "chuong_title": item_title_content}
#         elif item_type == "muc":
#             flush_current_chunk()
#             phan_chuong_info = {k:v for k,v in current_hierarchical_meta.items() if "phan" in k or "chuong" in k}
#             current_hierarchical_meta = {**phan_chuong_info, "muc_code": item_code, "muc_title": item_title_content}
#         elif item_type == "dieu":
#             flush_current_chunk() # Xả chunk của Điều trước đó (nếu có) hoặc preamble/Mục/Chương
#             # Thiết lập thông tin cho Điều hiện tại, sẽ được dùng khi flush_current_chunk tiếp theo
#             current_hierarchical_meta["current_dieu_code"] = item_code
#             current_hierarchical_meta["current_dieu_title"] = item_title_content

#         current_chunk_lines.append(line_stripped) # Thêm dòng hiện tại vào buffer

#     flush_current_chunk() # Xả chunk cuối cùng

#     # Xử lý trường hợp toàn bộ văn bản không có cấu trúc Điều nào được nhận diện
#     # Hoặc nếu chunks rỗng (văn bản quá ngắn, không có cấu trúc nào ngoài preamble)
#     if (not any(c.metadata.get("dieu_code") for c in chunks) and text) or (not chunks and text):
#         if not chunks: # Nếu chunks hoàn toàn rỗng
#              logger.warning(f"🔸 Document '{source_metadata.get('source', 'N/A')}' không tạo được chunk cấu trúc. Splitting full text.")
#         else: # Có chunk nhưng không có Điều (ví dụ chỉ có Preamble)
#              logger.warning(f"🔸 Document '{source_metadata.get('source', 'N/A')}' không có 'Điều' structure. Treating existing chunks and/or splitting full text.")
#              # Giữ lại các chunk đã có (ví dụ preamble) và chia phần còn lại nếu cần
#              # Hoặc đơn giản là chia lại toàn bộ nếu logic này quá phức tạp

#         # Đơn giản: nếu không có Điều, chia lại toàn bộ bằng base_splitter
#         # (Điều này có thể làm mất các chunk preamble đã tạo nếu có)
#         if not any(c.metadata.get("dieu_code") for c in chunks):
#             chunks.clear() # Xóa các chunk đã có (nếu có)
#             base_doc_meta = source_metadata.copy()
#             base_doc_meta.pop("id", None); base_doc_meta.pop("weaviate_id", None) # ID sẽ do create_documents tạo
#             if "title" not in base_doc_meta: # Đặt title chung
#                  base_doc_meta["title"] = base_doc_meta.get("ten_van_ban", base_doc_meta.get("source", "N/A"))

#             # Thêm penalties và cross_references cho toàn bộ văn bản nếu chia kiểu này
#             penalties = extract_penalties_from_text(text)
#             if penalties: base_doc_meta["penalties"] = penalties
#             cross_refs = extract_cross_references(text, source_metadata)
#             if cross_refs: base_doc_meta["cross_references"] = cross_refs

#             sub_docs = base_text_splitter.create_documents([text], metadatas=[base_doc_meta])
#             chunks.extend(sub_docs) # Thêm các sub_docs này vào

#     return chunks

# New code
def hierarchical_split_law_document(doc_obj: Document) -> List[Document]:
    """
    CẢI TIẾN LỚN: Chia văn bản luật theo cấu trúc Điều -> Khoản.
    Giải quyết vấn đề lặp metadata và chunking không tối ưu.
    """
    text = doc_obj.page_content
    source_metadata = doc_obj.metadata.copy()
    doc_so_hieu = source_metadata.get("so_hieu")
    filename = source_metadata.get("source")

    final_chunks = []
    lines = text.splitlines()

    hierarchy_context = {}
    current_dieu_lines = []
    current_dieu_code = None
    current_dieu_title = ""

    def flush_dieu_buffer():
        nonlocal final_chunks, current_dieu_lines, current_dieu_code, current_dieu_title
        if not current_dieu_lines: return

        # 1. Chia buffer của "Điều" thành các buffer nhỏ hơn cho từng "Khoản"
        khoan_buffers: Dict[str, List[str]] = {}
        current_khoan_code = "khoan-0" # Buffer cho tiêu đề Điều và nội dung không thuộc khoản nào
        khoan_buffers[current_khoan_code] = []

        for line in current_dieu_lines:
            item_type, item_code, _ = parse_law_item_line(line)
            if item_type == "khoan":
                current_khoan_code = item_code
                if current_khoan_code not in khoan_buffers:
                    khoan_buffers[current_khoan_code] = []
            khoan_buffers[current_khoan_code].append(line)

        # 2. Tạo chunk từ buffer của từng "Khoản"
        for khoan_code, khoan_lines in khoan_buffers.items():
            khoan_content = "\n".join(khoan_lines).strip()
            if not khoan_content: continue

            structure_path = [v for k, v in hierarchy_context.items() if k.endswith('_code')]
            if current_dieu_code: structure_path.append(current_dieu_code)
            if khoan_code != "khoan-0": structure_path.append(khoan_code)

            chunk_metadata_base = {**source_metadata, **hierarchy_context}
            if current_dieu_code:
                chunk_metadata_base["dieu_code"] = current_dieu_code
                chunk_metadata_base["dieu_title"] = current_dieu_title
            if khoan_code != "khoan-0":
                chunk_metadata_base["khoan_code"] = khoan_code
            chunk_metadata_base["title"] = f"{source_metadata.get('ten_van_ban', 'Văn bản')} - {current_dieu_code or 'Nội dung chung'}"

            # Nếu một Khoản quá lớn, mới chia nhỏ nó
            if len(khoan_content) > MAX_CHUNK_SIZE:
                sub_texts = base_text_splitter.split_text(khoan_content)
                for i, sub_text in enumerate(sub_texts):
                    sub_chunk_path = structure_path + [f"part-{i}"]
                    sub_chunk_id = generate_structured_id(doc_so_hieu, sub_chunk_path,filename)
                    sub_chunk_meta = chunk_metadata_base.copy()
                    sub_chunk_meta["sub_chunk_index"] = i
                    final_chunks.append(Document(page_content=sub_text, metadata=sub_chunk_meta, id=sub_chunk_id))
            else:
                chunk_id = generate_structured_id(doc_so_hieu, structure_path, filename)
                final_chunks.append(Document(page_content=khoan_content, metadata=chunk_metadata_base.copy(), id=chunk_id))

        current_dieu_lines = []

    # Vòng lặp chính để xác định các khối "Điều"
    for line_text in lines:
        if not line_text.strip(): continue
        item_type, item_code, item_title_content = parse_law_item_line(line_text)

        if item_type == "dieu":
            flush_dieu_buffer()
            current_dieu_code = item_code
            current_dieu_title = item_title_content

        if item_type == "phan": hierarchy_context.update({"phan_code": item_code, "phan_title": item_title_content})
        elif item_type == "chuong": hierarchy_context.update({"chuong_code": item_code, "chuong_title": item_title_content})

        if current_dieu_code: current_dieu_lines.append(line_text)

    flush_dieu_buffer()

    # Fallback cho văn bản không có cấu trúc "Điều"
    if not final_chunks and text:
        logger.warning(f"Văn bản '{doc_so_hieu}' không có cấu trúc 'Điều'. Chia toàn bộ văn bản.")
        sub_texts = base_text_splitter.split_text(text)
        for i, sub_text in enumerate(sub_texts):
            chunk_id = generate_structured_id(doc_so_hieu, [f"fulltext-part-{i}"], filename)
            chunk_meta = source_metadata.copy()
            chunk_meta["title"] = chunk_meta.get("ten_van_ban", doc_so_hieu)
            final_chunks.append(Document(page_content=sub_text, metadata=chunk_meta, id=chunk_id))

    # 3. LÀM GIÀU METADATA SAU KHI CHIA (FIX LỖI LẶP DỮ LIỆU)
    enriched_chunks = []
    for chunk in final_chunks:
        # Trích xuất metadata từ nội dung CỤ THỂ của CHÍNH CHUNK này
        chunk.metadata["penalties"] = extract_penalties_from_text(chunk.page_content)
        chunk.metadata["cross_references"] = extract_cross_references(chunk.page_content, source_metadata)
        enriched_chunks.append(chunk)

    return enriched_chunks


def infer_field(text_content: str, doc_title: Optional[str]) -> str:
    """
    CẢI TIẾN: Suy ra lĩnh vực pháp luật bằng hệ thống tính điểm để tăng độ chính xác.
    """
    if not doc_title and not text_content: return "khac"

    search_text = ((doc_title.lower() if doc_title else "") + " " + text_content[:1000].lower()).strip()
    title_lower = doc_title.lower() if doc_title else ""

        # Cấu trúc từ khóa có trọng số (weight)
    field_keywords = {
        # 1. Giao thông
        "giao_thong": [
            ("trật tự, an toàn giao thông đường bộ", 12),
            ("xử phạt vi phạm hành chính trong lĩnh vực giao thông", 10),
            ("giao thông đường bộ", 10),
            ("giao thông đường sắt", 10),
            ("giấy phép lái xe", 8),
            ("đèn tín hiệu giao thông", 8),
            ("đăng kiểm", 7),
            ("xe ô tô", 5),
            ("xe mô tô", 5),
            ("nồng độ cồn", 5),
            ("tốc độ", 3),
            ("biển báo", 3),
            ("lái xe", 2),
        ],

        # 2. Hình sự
        "hinh_su": [
            ("bộ luật hình sự", 12),
            ("truy cứu trách nhiệm hình sự", 10),
            ("tội phạm", 8),
            ("khởi tố", 8),
            ("điều tra hình sự", 8),
            ("tòa án nhân dân", 7),
            ("viện kiểm sát", 7),
            ("giết người", 5),
            ("cướp giật tài sản", 5),
            ("ma túy", 5),
            ("tử hình", 5),
            ("tù chung thân", 5),
        ],

        # 3. Dân sự
        "dan_su": [
            ("bộ luật dân sự", 12),
            ("bồi thường thiệt hại ngoài hợp đồng", 10),
            ("giao dịch dân sự", 9),
            ("quyền sở hữu", 8),
            ("thừa kế", 8),
            ("di chúc", 8),
            ("hợp đồng dân sự", 7),
            ("tranh chấp dân sự", 5),
            ("ly hôn", 4), # Có thể thuộc cả hôn nhân gia đình
        ],

        # 4. Hôn nhân và Gia đình
        "hon_nhan_gia_dinh": [
            ("luật hôn nhân và gia đình", 12),
            ("kết hôn", 9),
            ("ly hôn", 9),
            ("quan hệ giữa vợ và chồng", 8),
            ("tài sản chung của vợ chồng", 8),
            ("quyền, nghĩa vụ của cha mẹ và con", 8),
            ("cấp dưỡng", 7),
            ("giám hộ", 5),
        ],

        # 5. Lao động
        "lao_dong": [
            ("bộ luật lao động", 12),
            ("hợp đồng lao động", 10),
            ("người sử dụng lao động", 8),
            ("người lao động", 8),
            ("bảo hiểm xã hội", 7),
            ("tiền lương", 5),
            ("thời giờ làm việc", 5),
            ("kỷ luật lao động", 5),
            ("sa thải", 5),
            ("công đoàn", 3),
        ],

        # 6. Đất đai
        "dat_dai": [
            ("luật đất đai", 12),
            ("quyền sử dụng đất", 10),
            ("giấy chứng nhận quyền sử dụng đất", 9),
            ("thu hồi đất", 8),
            ("bồi thường, hỗ trợ, tái định cư", 8),
            ("quy hoạch, kế hoạch sử dụng đất", 7),
            ("sổ đỏ", 5), # Từ thông tục nhưng rất đặc trưng
        ],

        # 7. Doanh nghiệp & Đầu tư
        "doanh_nghiep": [
            ("luật doanh nghiệp", 12),
            ("luật đầu tư", 12),
            ("thành lập doanh nghiệp", 9),
            ("giấy chứng nhận đăng ký doanh nghiệp", 9),
            ("công ty cổ phần", 8),
            ("công ty trách nhiệm hữu hạn", 8),
            ("doanh nghiệp tư nhân", 8),
            ("vốn điều lệ", 7),
            ("cổ đông", 5),
            ("phá sản", 5),
        ],

        # 8. Xây dựng & Nhà ở
        "xay_dung": [
            ("luật xây dựng", 12),
            ("luật nhà ở", 12),
            ("giấy phép xây dựng", 9),
            ("quy hoạch xây dựng", 8),
            ("chủ đầu tư", 7),
            ("dự án đầu tư xây dựng", 7),
            ("công trình xây dựng", 5),
            ("thi công", 3),
        ],

        # 9. Hành chính
        "hanh_chinh": [
            ("luật xử lý vi phạm hành chính", 10), # Cụm từ dài và đặc trưng
            ("khiếu nại, tố cáo", 8),
            ("thủ tục hành chính", 7),
            ("công chức, viên chức", 7),
            ("xử phạt vi phạm hành chính", 4), # Trọng số trung bình, vì nó là một phần của nhiều luật khác
            ("nghị định", 1), # Trọng số cực thấp
            ("thông tư", 1), # Trọng số cực thấp
        ],

        # 10. Thuế & Tài chính & Ngân hàng
        "tai_chinh_thue": [
            ("luật các tổ chức tín dụng", 12),
            ("luật quản lý thuế", 12),
            ("thuế giá trị gia tăng", 9),
            ("thuế thu nhập doanh nghiệp", 9),
            ("thuế thu nhập cá nhân", 9),
            ("ngân sách nhà nước", 8),
            ("trái phiếu", 5),
            ("cổ phiếu", 5),
            ("kế toán, kiểm toán", 4),
        ],

        # 11. Môi trường
        "moi_truong": [
            ("luật bảo vệ môi trường", 12),
            ("đánh giá tác động môi trường", 9),
            ("ô nhiễm môi trường", 8),
            ("chất thải", 5),
            ("khí thải", 3),
        ],

        # 12. Sở hữu trí tuệ
        "so_huu_tri_tue": [
            ("luật sở hữu trí tuệ", 12),
            ("quyền tác giả", 9),
            ("quyền liên quan", 9),
            ("sáng chế", 8),
            ("nhãn hiệu", 8),
            ("bản quyền", 7),
        ],

        # 13. Giáo dục
        "giao_duc": [
            ("luật giáo dục", 12),
            ("học sinh, sinh viên", 7),
            ("cơ sở giáo dục", 7),
            ("học phí", 5),
            ("đào tạo", 3),
            ("giáo viên", 3),
        ],

        # 14. Y tế
        "y_te": [
            ("luật khám bệnh, chữa bệnh", 12),
            ("bảo hiểm y tế", 10),
            ("dược", 8),
            ("trang thiết bị y tế", 7),
            ("bệnh viện", 5),
            ("thuốc", 3),
        ],

        # ... Bạn có thể thêm các lĩnh vực khác như Thương mại, An ninh Quốc phòng ...
    }

    field_scores = {field: 0 for field in field_keywords.keys()}

    for field, weighted_keywords in field_keywords.items():
        score = 0
        for keyword, weight in weighted_keywords:
            if keyword in title_lower:
                score += weight * 3 # Nhân 3 lần điểm nếu ở trong tiêu đề

            occurrences_in_text = search_text.count(keyword)
            if occurrences_in_text > 0:
                score += weight * occurrences_in_text
        field_scores[field] = score

    # Lọc ra các lĩnh vực có điểm > 0
    positive_scores = {f: s for f, s in field_scores.items() if s > 0}

    if not positive_scores:
        return "khac"

    # In ra điểm số để debug
    logger.debug(f"Field scores for title '{doc_title}': {positive_scores}")

    best_field = max(positive_scores, key=positive_scores.get)
    return best_field


# def infer_field(text_content: str, doc_title: Optional[str]) -> str:
#     """Suy ra lĩnh vực pháp luật từ tiêu đề và nội dung."""
#     search_text = ((doc_title.lower() if doc_title else "") + " " + text_content[:500].lower()).strip()
#     field_keywords = {
#         "giao_thong": ["giao thông", "trật tự an toàn giao thông", "lái xe", "tốc độ", "biển báo", "phạt nguội", "đường bộ", "đường sắt", "trừ điểm", "giấy phép lái xe"],
#         "hinh_su": ["tội phạm", "hình sự", "giết người", "trộm cắp", "cướp", "ma túy", "bộ luật hình sự", "tử hình", "tù chung thân", "truy cứu"],
#         "dan_su": ["dân sự", "thừa kế", "hợp đồng dân sự", "ly hôn", "bồi thường thiệt hại ngoài hợp đồng", "quyền sở hữu", "tranh chấp"],
#         "lao_dong": ["lao động", "hợp đồng lao động", "tiền lương", "bảo hiểm xã hội", "thời giờ làm việc", "sa thải", "công đoàn", "người sử dụng lao động"],
#         "doanh_nghiep": ["doanh nghiệp", "công ty", "thành lập doanh nghiệp", "phá sản", "cổ phần", "đầu tư", "kinh doanh"],
#         "thuong_mai": ["thương mại", "hợp đồng mua bán", "xuất nhập khẩu", "hàng hóa", "dịch vụ", "quảng cáo"],
#         "dat_dai": ["đất đai", "sử dụng đất", "thu hồi đất", "giấy chứng nhận quyền sử dụng đất", "quy hoạch đất", "bồi thường đất"],
#         "xay_dung": ["xây dựng", "giấy phép xây dựng", "quy hoạch xây dựng", "công trình", "thi công"],
#         "moi_truong": ["môi trường", "bảo vệ môi trường", "ô nhiễm", "chất thải", "đánh giá tác động môi trường", "khí thải", "nước thải"],
#         "hanh_chinh": ["hành chính", "xử phạt vi phạm hành chính", "thủ tục hành chính", "công chức", "viên chức", "khiếu nại", "tố cáo", "nghị định"],
#         "tai_chinh_ngan_hang": ["ngân hàng", "tín dụng", "thuế", "kế toán", "kiểm toán", "ngân sách", "trái phiếu", "cổ phiếu", "thị trường chứng khoán"],
#         "so_huu_tri_tue": ["sở hữu trí tuệ", "quyền tác giả", "bản quyền", "sáng chế", "nhãn hiệu", "kiểu dáng công nghiệp"],
#         "giao_duc": ["giáo dục", "học sinh", "sinh viên", "đào tạo", "trường học", "giáo viên", "học phí"],
#         "y_te": ["y tế", "khám bệnh", "chữa bệnh", "dược", "bảo hiểm y tế", "thuốc", "bệnh viện"],
#         "an_ninh_quoc_phong": ["an ninh quốc gia", "quốc phòng", "quân sự", "công an", "bí mật nhà nước"],
#     }
#     # Ưu tiên khớp nếu tên văn bản chứa từ khóa loại văn bản
#     if doc_title:
#         for field, keywords in field_keywords.items():
#             # Kiểm tra xem có từ khóa nào trong field đó (thường là từ khóa chính như "luật giao thông", "bộ luật hình sự")
#             # xuất hiện trong doc_title không
#             main_keywords_for_field = keywords[:2] # Lấy vài từ khóa đầu tiên làm đại diện
#             if any(mk.lower() in doc_title.lower() for mk in main_keywords_for_field):
#                  if any(keyword.lower() in doc_title.lower() for keyword in keywords): # Check lại để chắc chắn hơn
#                     return field

#     for field, keywords in field_keywords.items():
#         if any(keyword in search_text for keyword in keywords):
#             return field
#     return "khac"

def infer_entity_type(query_or_text: str, field: str) -> Union[str, List[str], None]:
    text_lower = query_or_text.lower()
    entity_definitions = {
        "giao_thong": {
            "xe_oto": {"keywords": ["ô tô", "xe hơi", "xe con"], "priority": 10},
            "xe_may": {"keywords": ["xe máy", "mô tô", "xe gắn máy"], "priority": 10},
            "nguoi_dieu_khien": {"keywords": ["người điều khiển", "lái xe", "tài xế"], "priority": 9},
            "phuong_tien": {"keywords": ["phương tiện", "xe"], "priority": 5},
        },
        "hinh_su": {
            "nguoi_pham_toi": {"keywords": ["tội phạm", "bị can", "bị cáo", "người phạm tội"], "priority": 10},
            "nan_nhan": {"keywords": ["nạn nhân", "người bị hại"], "priority": 9},
        },
         "lao_dong": {
            "nguoi_lao_dong": {"keywords": ["người lao động", "nhân viên", "công nhân"], "priority": 10},
            "nguoi_su_dung_lao_dong": {"keywords": ["người sử dụng lao động", "công ty", "doanh nghiệp"], "priority": 10},
            "hop_dong_lao_dong": {"keywords": ["hợp đồng lao động"], "priority": 8},
        },
        # ... (Thêm các định nghĩa khác nếu cần) ...
        "khac": {
            "ca_nhan": {"keywords": ["cá nhân", "người", "công dân"], "priority": 7},
            "to_chuc": {"keywords": ["tổ chức", "cơ quan", "đơn vị"], "priority": 7},
        }
    }
    found_entities = []
    current_field_entities = entity_definitions.get(field, entity_definitions["khac"])
    sorted_entities = sorted(current_field_entities.items(), key=lambda item: item[1]["priority"], reverse=True)
    for entity_type, definition in sorted_entities:
        sorted_keywords = sorted(definition["keywords"], key=len, reverse=True)
        if any(re.search(r"\b" + re.escape(keyword) + r"\b", text_lower) for keyword in sorted_keywords):
            if entity_type not in found_entities: found_entities.append(entity_type)
    if not found_entities: return None
    return found_entities[0] if len(found_entities) == 1 else found_entities


def parse_law_item_line(line: str) -> Tuple[Optional[str], Optional[str], str]:
    """CẢI TIẾN: Phân tích dòng và trả về code đã được chuẩn hóa."""
    line = line.strip()

    phan_match = re.match(r"^\s*(PHẦN\s+(?:THỨ\s+[\w\sÀ-Ỹà-ỹ]+|[IVXLCDM]+|CHUNG|CÁC TỘI PHẠM))\s*[:.]?\s*(.*)", line, re.IGNORECASE)
    if phan_match: return "phan", phan_match.group(1).lower().replace(" ", "-"), phan_match.group(2).strip()

    chuong_match = re.match(r"^\s*(Chương\s+[IVXLCDM\d]+)\s*[:.]?\s*(.*)", line, re.IGNORECASE)
    if chuong_match: return "chuong", chuong_match.group(1).lower().replace(" ", "-"), chuong_match.group(2).strip()

    dieu_match = re.match(r"^\s*(Điều\s+\d+[a-z]?)\s*[\.:]?\s*(.*)", line, re.IGNORECASE)
    if dieu_match:
        item_code = dieu_match.group(1).lower().replace(" ", "-")
        item_title = dieu_match.group(2).strip()
        if re.match(r"^\d+\.", item_title):
            return "dieu", item_code, ""
        return "dieu", item_code, item_title

    khoan_match = re.match(r"^\s*(\d+)\.\s*(.+)", line)
    if khoan_match:
        content = khoan_match.group(2).strip()
        if not re.match(r"^\s*[a-zđ]\)", content) and content and content[0].isupper():
            return "khoan", f"khoan-{khoan_match.group(1)}", content

    diem_match = re.match(r"^\s*([a-zđ])\)\s*(.+)", line)
    if diem_match: return "diem", f"diem-{diem_match.group(1)}", diem_match.group(2).strip()

    return None, None, line




def _normalize_money(value_str: str) -> Optional[float]:
    if not value_str: return None
    try:
        return float(value_str.replace(".", "").replace(",", "."))
    except ValueError: return None

def _normalize_duration(value_str: str, unit_str: str) -> Optional[Dict[str, Union[int, str]]]:
    if not value_str or not unit_str: return None
    try:
        value = int(value_str)
        unit = unit_str.lower()
        if unit == "tháng": return {"value": value, "unit": "months"}
        if unit == "năm": return {"value": value, "unit": "years"}
        if unit == "ngày": return {"value": value, "unit": "days"}
        return {"value": value, "unit": unit_str}
    except ValueError: return None

def extract_penalties_from_text(text_content: str) -> List[Dict]:
    penalties = []
    # Chuyển text_content thành chữ thường một lần để tìm kiếm không phân biệt hoa thường
    # Nhưng vẫn giữ text_content gốc để lấy original_text
    text_lower_for_search = text_content.lower()

    # 1. Phạt tiền (KHOẢNG trước)
    fine_range_pattern = r"phạt tiền từ\s*([\d\.,]+)\s*đồng\s*đến\s*([\d\.,]+)\s*đồng"
    # Lưu vị trí các match của fine_range để không xử lý lại phần này cho fine_fixed
    fine_range_spans = []
    for m in re.finditer(fine_range_pattern, text_content, re.IGNORECASE):
        penalties.append({
            "type": "fine",
            "min_amount": _normalize_money(m.group(1)),
            "max_amount": _normalize_money(m.group(2)),
            "currency": "đồng",
            "original_text": m.group(0)
        })
        fine_range_spans.append(m.span()) # Lưu lại span (start, end) của match này

    # 2. Phạt tiền (CỐ ĐỊNH sau, và không nằm trong các khoảng đã tìm thấy)
    fine_fixed_pattern = r"phạt tiền\s*([\d\.,]+)\s*đồng"
    for m in re.finditer(fine_fixed_pattern, text_content, re.IGNORECASE):
        current_span = m.span()
        is_part_of_range = False
        # Kiểm tra xem match này có nằm trong một fine_range_span không
        for r_span_start, r_span_end in fine_range_spans:
            # Nếu match của fine_fixed nằm hoàn toàn trong một match của fine_range
            if r_span_start <= current_span[0] and current_span[1] <= r_span_end:
                # Và nếu nó không phải là chính fine_range_match đó (trường hợp pattern giống hệt)
                # (So sánh original_text để chắc chắn hơn, nhưng span là đủ)
                if text_content[r_span_start:r_span_end] != m.group(0):
                     is_part_of_range = True
                     break
        if is_part_of_range:
            continue # Bỏ qua nếu nó là một phần của "phạt tiền từ...đến..."

        # Heuristic bổ sung: Kiểm tra từ khóa "từ", "đến" xung quanh
        # Ngữ cảnh trước (ví dụ 15 ký tự)
        context_before = text_content[max(0, m.start() - 15):m.start()].lower()
        # Ngữ cảnh sau (ví dụ 10 ký tự)
        context_after = text_content[m.end():min(len(text_content), m.end() + 10)].lower()

        if ("từ" in context_before and "đến" in text_content[m.start():m.end()+30].lower()) or \
           ("đến" in context_after and "từ" in text_content[max(0, m.start()-30):m.end()].lower()):
            # Có khả năng cao đây là một phần của một khoảng phạt mà regex trên chưa bắt hết
            # Ví dụ: "phạt tiền từ năm trăm nghìn đồng, phạt tiền một triệu đồng đến hai triệu đồng"
            # "phạt tiền một triệu đồng" có thể bị bắt nhầm.
            # Logic này cần cẩn thận để không loại bỏ nhầm.
            # Tạm thời có thể bỏ qua heuristic phức tạp này nếu việc kiểm tra span đã đủ tốt.
            pass # Hiện tại, dựa vào fine_range_spans là chính

        penalties.append({
            "type": "fine",
            "amount": _normalize_money(m.group(1)),
            "currency": "đồng",
            "original_text": m.group(0)
        })

    # 3. Hình phạt tù (KHOẢNG trước)
    prison_range_pattern = r"phạt tù từ\s*(\d+)\s*(tháng|năm|ngày)\s*đến\s*(\d+)\s*(tháng|năm|ngày)"
    prison_range_spans = []
    for m in re.finditer(prison_range_pattern, text_lower_for_search): # Dùng text_lower
        penalties.append({
            "type": "prison",
            "min_duration": _normalize_duration(m.group(1), m.group(2)),
            "max_duration": _normalize_duration(m.group(3), m.group(4)),
            "original_text": m.group(0) # Lấy từ text_lower_for_search, nhưng khi hiển thị có thể muốn text gốc
                                      # Để nhất quán, có thể finditer trên text_content và dùng re.IGNORECASE
                                      # Hoặc lưu original_text từ text_content[m.start():m.end()]
        })
        prison_range_spans.append(m.span())

    # 4. Hình phạt tù (CỐ ĐỊNH sau, và không nằm trong các khoảng đã tìm thấy)
    prison_fixed_pattern = r"phạt tù\s*(\d+)\s*(tháng|năm|ngày)"
    for m in re.finditer(prison_fixed_pattern, text_lower_for_search): # Dùng text_lower
        current_span = m.span()
        is_part_of_range = any(r_start <= current_span[0] and current_span[1] <= r_end and text_lower_for_search[r_start:r_end] != m.group(0)
                               for r_start, r_end in prison_range_spans)
        if is_part_of_range:
            continue

        # Heuristic kiểm tra "từ", "đến" xung quanh
        context_before = text_lower_for_search[max(0, m.start() - 10):m.start()]
        context_after = text_lower_for_search[m.end():min(len(text_lower_for_search), m.end() + 15)]
        if ("từ" in context_before and "đến" in context_after) or \
           ("từ" in context_before and re.search(r"đến\s*\d+\s*(tháng|năm|ngày)", text_lower_for_search[m.end():m.end()+30])):
            continue # Có khả năng là một phần của khoảng mà regex chưa bắt hết

        penalties.append({
            "type": "prison",
            "duration": _normalize_duration(m.group(1), m.group(2)),
            "original_text": m.group(0) # Tương tự trên, cân nhắc lấy từ text_content
        })

    # ... (các loại penalty khác giữ nguyên hoặc áp dụng logic tương tự nếu có look-behind phức tạp) ...
    # Tù chung thân
    if match := re.search(r"phạt tù chung thân", text_lower_for_search):
        penalties.append({"type": "prison", "duration_type": "life_imprisonment", "original_text": match.group(0)})
    # Tử hình
    if match := re.search(r"phạt tử hình", text_lower_for_search):
        penalties.append({"type": "prison", "duration_type": "death_penalty", "original_text": match.group(0)})

    # Cải tạo không giam giữ
    for m in re.finditer(r"cải tạo không giam giữ\s*(?:đến\s*)?(\d+)\s*(năm|tháng)", text_lower_for_search):
        penalties.append({"type": "non_custodial_reform", "max_duration": _normalize_duration(m.group(1), m.group(2)), "original_text": m.group(0)})

    # Cảnh cáo
    if match := re.search(r"\bphạt cảnh cáo\b", text_lower_for_search):
        penalties.append({"type": "warning", "original_text": match.group(0)})

    # Tịch thu
    if match := re.search(r"tịch thu tang vật(?:\s*,\s*phương tiện(?: được sử dụng để vi phạm hành chính)?)?", text_lower_for_search):
        penalties.append({"type": "confiscation_object_vehicle", "original_text": match.group(0)})
    if m := re.search(r"tịch thu (một phần hoặc toàn bộ|toàn bộ|một phần) tài sản", text_lower_for_search):
        penalties.append({"type": "confiscation_property", "scope": m.group(1), "original_text": m.group(0)})
    elif match := re.search(r"tịch thu tài sản", text_lower_for_search):
         penalties.append({"type": "confiscation_property", "scope": "không xác định", "original_text": match.group(0)})

    # Tước quyền sử dụng
    for m in re.finditer(r"tước quyền sử dụng\s*(?:giấy phép lái xe|giấy phép|chứng chỉ hành nghề|phù hiệu|tem kiểm định)\s*(?:có thời hạn|từ)?\s*(\d+)\s*(?:đến\s*(\d+)\s*)?(tháng|năm)", text_lower_for_search):
        min_v_str, max_v_str, unit_str = m.groups()
        duration_info = {}
        if max_v_str: # Có khoảng từ...đến
            duration_info["min_duration"] = _normalize_duration(min_v_str, unit_str)
            duration_info["max_duration"] = _normalize_duration(max_v_str, unit_str)
        else: # Chỉ có một giá trị
            duration_info["duration"] = _normalize_duration(min_v_str, unit_str)
        penalties.append({"type": "license_revocation", **duration_info, "original_text": m.group(0)})

    if match := re.search(r"tước quyền sử dụng\s*(?:giấy phép lái xe|giấy phép|chứng chỉ hành nghề)\s*vĩnh viễn", text_lower_for_search):
        penalties.append({"type": "license_revocation", "duration_type": "permanent", "original_text": match.group(0)})

    # Trục xuất
    if match := re.search(r"\btrục xuất\b", text_lower_for_search):
        penalties.append({"type": "deportation", "original_text": match.group(0)})

    # Cấm đảm nhiệm, hành nghề, cư trú
    for m in re.finditer(r"(cấm đảm nhiệm chức vụ|cấm hành nghề(?: hoặc làm công việc nhất định)?|cấm cư trú)\s*(?:từ\s*(\d+)\s*(?:đến\s*(\d+)\s*)?(năm|tháng))?", text_lower_for_search):
        p_text, min_v_str, max_v_str, unit_str = m.groups()
        p_type = "prohibit_holding_office"
        if "hành nghề" in p_text: p_type = "prohibit_profession"
        elif "cư trú" in p_text: p_type = "prohibit_residence"
        d_info = {}
        if min_v_str and unit_str:
            if max_v_str:
                d_info["min_duration"] = _normalize_duration(min_v_str, unit_str)
                d_info["max_duration"] = _normalize_duration(max_v_str, unit_str)
            else:
                d_info["duration"] = _normalize_duration(min_v_str, unit_str)
        penalties.append({"type": p_type, **d_info, "original_text": m.group(0)})

    # Biện pháp khắc phục
    for m in re.finditer(r"buộc\s*((?:khôi phục lại tình trạng ban đầu|phá dỡ|tháo dỡ|nộp lại số lợi bất hợp pháp|thực hiện biện pháp khắc phục|tái xuất|trả lại tài sản|công khai xin lỗi|bồi thường thiệt hại)(?:[\w\sÀ-Ỹà-ỹ,]*)?)", text_lower_for_search):
        action_full = m.group(1)
        action_key = action_full.split(',')[0].split(' và ')[0].strip()
        remedy_type_map = {
            "khôi phục lại tình trạng ban đầu": "remedial_restore", "phá dỡ": "remedial_demolish",
            "tháo dỡ": "remedial_demolish", "nộp lại số lợi bất hợp pháp": "remedial_return_illegal_profit",
            "tái xuất": "remedial_re_export", "trả lại tài sản": "remedial_return_property",
            "công khai xin lỗi": "remedial_public_apology", "bồi thường thiệt hại": "remedial_compensation"
        }
        remedy_type = "remedial_action"
        for key_text, r_type in remedy_type_map.items():
            if key_text in action_key:
                remedy_type = r_type
                break
        penalties.append({"type": remedy_type, "description": action_full, "original_text": m.group(0)})

    # Trừ điểm GPLX
    for m in re.finditer(r"trừ điểm giấy phép lái xe\s*(\d+)\s*điểm", text_lower_for_search):
        penalties.append({"type": "demerit_points_license", "points": int(m.group(1)) if m.group(1).isdigit() else None, "original_text": m.group(0)})

    # Lọc penalties để loại bỏ trùng lặp và ưu tiên match dài hơn
    # Sắp xếp theo vị trí bắt đầu, sau đó theo độ dài giảm dần (để xử lý match dài trước)
    penalties.sort(key=lambda p: (text_content.find(p["original_text"]), -len(p["original_text"])))

    final_penalties = []
    processed_spans = [] # List các (start, end) của các penalty đã được chọn

    for p in penalties:
        current_penalty_span_start = text_content.find(p["original_text"])
        current_penalty_span_end = current_penalty_span_start + len(p["original_text"])
        current_penalty_span = (current_penalty_span_start, current_penalty_span_end)

        is_overlapping_or_substring = False
        for proc_start, proc_end in processed_spans:
            # Nếu penalty hiện tại nằm trong hoặc bị bao phủ bởi một penalty đã xử lý
            if (proc_start <= current_penalty_span[0] and current_penalty_span[1] <= proc_end) or \
               (current_penalty_span[0] <= proc_start and proc_end <= current_penalty_span[1]):
                # Nếu p là substring của một cái đã có, bỏ qua p
                # Nếu một cái đã có là substring của p, thì cần phức tạp hơn, nhưng do đã sort, p dài hơn sẽ được ưu tiên
                if current_penalty_span[0] >= proc_start and current_penalty_span[1] <= proc_end and len(p["original_text"]) < (proc_end - proc_start):
                    is_overlapping_or_substring = True
                    break

        if not is_overlapping_or_substring:
            final_penalties.append(p)
            processed_spans.append(current_penalty_span)

    return final_penalties


def load_process_and_split_documents(folder_path: str) -> List[Document]:
    all_final_chunks = []
    if not os.path.isdir(folder_path):
        logger.error(f"Folder '{folder_path}' does not exist.")
        return all_final_chunks

    txt_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.txt')]
    if not txt_files:
        logger.warning(f"No .txt files found in '{folder_path}'.")
        return all_final_chunks

    logger.info(f"Found {len(txt_files)} .txt files. Processing...")
    for filename in tqdm(txt_files, desc="Processing files"):
        file_path = os.path.join(folder_path, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_content = f.read()

            if not raw_content.strip():
                logger.warning(f"File '{filename}' is empty or contains only whitespace.")
                continue

            # Trích xuất metadata gốc từ raw_content trước khi làm sạch quá nhiều
            doc_metadata_original = extract_document_metadata(raw_content, filename)
            doc_metadata_original["source"] = filename

            # Làm sạch nội dung để xử lý (có thể giữ lại phần đầu nếu clean_document_text được điều chỉnh)
            cleaned_content_for_processing = clean_document_text(raw_content)
            if not cleaned_content_for_processing.strip():
                logger.warning(f"File '{filename}' is empty after cleaning for processing.")
                continue

            # Suy luận lĩnh vực và các thông tin khác
            doc_metadata_original["field"] = infer_field(cleaned_content_for_processing, doc_metadata_original.get("ten_van_ban"))
            doc_metadata_original["entity_type"] = infer_entity_type(cleaned_content_for_processing, doc_metadata_original.get("field"))
            # Penalty sẽ được trích xuất cho từng chunk

            # Tạo đối tượng Document lớn ban đầu để truyền vào hàm chia chunk
            # Nội dung là cleaned_content, metadata là doc_metadata_original
            # Tham số id của Document này không quá quan trọng vì nó sẽ được chia nhỏ
            doc_to_split = Document(page_content=cleaned_content_for_processing, metadata=doc_metadata_original)

            chunks_from_file = hierarchical_split_law_document(doc_to_split)
            all_final_chunks.extend(chunks_from_file)

        except Exception as e:
            logger.error(f"Error processing file '{filename}': {e}", exc_info=True)

    logger.info(f"Processed {len(txt_files)} files, generated {len(all_final_chunks)} final chunks.")
    # Log kiểm tra cuối cùng trước khi trả về
    for i, chk in enumerate(all_final_chunks[:3]): # Log 3 chunk đầu tiên
        logger.debug(f"Final Chunk {i} ID: {chk.id if hasattr(chk, 'id') else 'NO ID ATTR'}, Metadata: {chk.metadata}")
        if not hasattr(chk, 'id') or not chk.id:
             logger.error(f"!!! FINAL CHECK: Chunk {i} from {chk.metadata.get('source')} is missing valid ID attribute before returning from load_process_and_split_documents.")

    return all_final_chunks


# new code
def process_single_file(file_path: str) -> List[Document]:
    """
    Hàm này thực hiện toàn bộ pipeline xử lý cho một file duy nhất.
    Nó sẽ được gọi song song cho nhiều file.

    Returns:
        Một list các đối tượng Document (chunks) đã được xử lý cho file đó.
        Metadata phức tạp (list/dict) vẫn ở dạng Python object.
    """
    filename = os.path.basename(file_path)
    logger.debug(f"Bắt đầu xử lý file: {filename}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_content = f.read()

        if not raw_content.strip():
            logger.warning(f"File '{filename}' rỗng.")
            return []

        # 1. Trích xuất metadata gốc từ raw_content
        doc_metadata = extract_document_metadata(raw_content, filename)
        doc_metadata["source"] = filename

        # 2. Làm sạch nội dung chính
        cleaned_content = clean_document_text(raw_content)
        if not cleaned_content.strip():
            logger.warning(f"File '{filename}' rỗng sau khi làm sạch.")
            return []

        # 3. Suy luận metadata bổ sung cho toàn bộ văn bản
        doc_metadata["field"] = infer_field(cleaned_content, doc_metadata.get("ten_van_ban"))

        # infer_entity_type có thể trả về list, giữ nguyên dạng list
        doc_metadata["entity_type"] = infer_entity_type(cleaned_content, doc_metadata.get("field",""))

        # 4. Tạo đối tượng Document lớn ban đầu để truyền vào hàm chia chunk
        doc_to_split = Document(page_content=cleaned_content, metadata=doc_metadata)

        # 5. Chia văn bản thành các chunk theo cấu trúc, đồng thời trích xuất
        # metadata cho từng chunk (penalties, cross_references)
        # Hàm hierarchical_split_law_document sẽ tự thêm các metadata này vào từng chunk
        chunks_from_file = hierarchical_split_law_document(doc_to_split)

        # Không cần gọi filter_and_serialize_complex_metadata ở đây nữa.
        # Việc này sẽ được thực hiện tập trung trước khi ingest.

        logger.info(f"✅ Xử lý xong file '{filename}', tạo ra {len(chunks_from_file)} chunks.")
        return chunks_from_file

    except Exception as e:
        logger.error(f"❌ Lỗi khi xử lý file '{filename}': {e}", exc_info=True)
        return []
