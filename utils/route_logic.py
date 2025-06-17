import logging
from typing import Dict, List
from underthesea import word_tokenize # Giả sử bạn đang dùng underthesea

logger = logging.getLogger(__name__)

# --- DANH SÁCH TỪ KHÓA (Nên được định nghĩa ở một nơi dễ quản lý) ---
# Chuyển thành set để tìm kiếm nhanh hơn và viết thường tất cả
GENERAL_FULL_PHRASES = {
    "bạn là ai", "bạn tên gì", "mày là ai", "mày tên gì", "chatbot tên gì",
    "bạn có thể làm gì", "bạn giúp được gì", "chức năng của bạn là gì",
    "bạn hoạt động như thế nào", "bạn được tạo ra bởi ai", "bạn được phát triển bởi ai",
    "bạn được huấn luyện như thế nào", "bạn sử dụng dữ liệu gì"
}

GENERAL_SINGLE_KEYWORDS = {
    "ai", "tên", # Từ "ai", "tên" cần cẩn thận vì có thể xuất hiện trong câu hỏi luật
    "chatbot", "robot", "trợ lý", "juri", "juribot", # Tên riêng của bot
    "llm", "mô hình ngôn ngữ", "trí tuệ nhân tạo",
    "tạo", "phát triển", "huấn luyện", "công ty",
    "tính năng", "khả năng", "phiên bản", "cập nhật", "thông tin" # "thông tin" cũng chung chung
}

LEGAL_KEYWORDS = {
    "pháp luật", "luật", "nghị định", "thông tư", "quyết định", "bộ luật",
    "pháp lệnh", "nghị quyết", "công văn", "chỉ thị", "điều khoản", "điều luật",
    "quy định", "mức phạt", "xử phạt", "chế tài", "bồi thường", "trách nhiệm pháp lý",
    "thừa kế", "di chúc", "hôn nhân", "gia đình", "ly hôn",
    "lao động", "việc làm", "hợp đồng lao động", "bảo hiểm xã hội",
    "giao thông", "tai nạn", "bằng lái", "xe",
    "hình sự", "tội phạm", "khởi tố", "điều tra", "xét xử", "tù",
    "dân sự", "tranh chấp", "nghĩa vụ dân sự", "quyền dân sự",
    "đất đai", "sổ đỏ", "quy hoạch", "thu hồi đất",
    "thuế", "kê khai", "miễn giảm", "hoàn thuế",
    "doanh nghiệp", "công ty", "thành lập", "giải thể", "phá sản",
    "hợp đồng", "giao dịch", "vô hiệu",
    "sở hữu trí tuệ", "bản quyền", "nhãn hiệu", "sáng chế",
    "vi phạm", "khiếu nại", "tố cáo", "giải quyết tranh chấp",
    "quyền", "nghĩa vụ", "điều kiện", "thủ tục", "thẩm quyền", "đối tượng", "áp dụng"
}


def find_first_matching_keyword(text_tokens: List[str], keywords_set: set) -> bool:
    """Kiểm tra xem có bất kỳ từ khóa nào trong set xuất hiện trong list token không."""
    for token in text_tokens:
        if token in keywords_set:
            return True
    return False

def find_first_matching_phrase(lower_text: str, phrases_set: set) -> bool:
    """Kiểm tra xem có cụm từ nào trong set xuất hiện trong text không."""
    for phrase in phrases_set:
        if phrase in lower_text:
            return True
    return False

def route_logic(input_data: Dict) -> str:
    """
    Phân loại câu hỏi thành 'general' hoặc 'legal' dựa trên từ khóa và ngữ cảnh.
    Ưu tiên kiểm tra các cụm từ general đầy đủ trước, sau đó đến từ khóa.
    """
    if not isinstance(input_data, dict):
        logger.error(f"Input không phải dictionary: {input_data}")
        return "general"

    question_original = input_data.get("input", "")

    if not isinstance(question_original, str) or not question_original.strip():
        logger.warning("Thiếu key 'input' hoặc câu hỏi rỗng. Mặc định là 'general'.")
        return "general"

    question_lower = question_original.strip().lower()

    # 1. Kiểm tra các cụm từ general đầy đủ (ưu tiên cao nhất)
    if find_first_matching_phrase(question_lower, GENERAL_FULL_PHRASES):
        logger.info(f"[RouteLogic] Phân loại: general (khớp cụm từ general đầy đủ trong: '{question_lower[:50]}...')")
        return "general"

    try:
        question_tokenized_list = word_tokenize(question_lower, format="text").lower().split()
        question_tokenized_set = set(question_tokenized_list)
    except Exception as e:
        logger.error(f"Lỗi khi tokenize câu hỏi '{question_lower[:50]}...': {e}. Sử dụng split() đơn giản.")
        question_tokenized_list = question_lower.split()
        question_tokenized_set = set(question_tokenized_list)

    if find_first_matching_keyword(question_tokenized_set, LEGAL_KEYWORDS):
        logger.info(f"[RouteLogic] Phân loại: legal (khớp từ khóa legal trong: '{question_lower[:50]}...')")
        return "legal"

    if find_first_matching_keyword(question_tokenized_set, GENERAL_SINGLE_KEYWORDS):
        is_likely_general_despite_common_words = True
        if "ai" in question_tokenized_set or "thông tin" in question_tokenized_set:
            pass

        if is_likely_general_despite_common_words:
            logger.info(f"[RouteLogic] Phân loại: general (khớp từ khóa general đơn lẻ trong: '{question_lower[:50]}...')")
            return "general"

    logger.info(f"[RouteLogic] Phân loại: legal (mặc định, không khớp general, ưu tiên xử lý như câu hỏi luật)")
    return "legal"