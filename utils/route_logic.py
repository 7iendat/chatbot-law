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

# --- HÀM HỖ TRỢ (Nên tách ra nếu dùng ở nhiều nơi) ---
def find_first_matching_keyword(text_tokens: List[str], keywords_set: set) -> bool:
    """Kiểm tra xem có bất kỳ từ khóa nào trong set xuất hiện trong list token không."""
    # Chuyển text_tokens thành set để tìm giao集 hiệu quả hơn nếu keywords_set lớn
    # tuy nhiên, với list token ngắn, duyệt qua cũng ổn
    for token in text_tokens:
        if token in keywords_set:
            return True # Trả về True ngay khi tìm thấy từ khóa đầu tiên
    return False

def find_first_matching_phrase(lower_text: str, phrases_set: set) -> bool:
    """Kiểm tra xem có cụm từ nào trong set xuất hiện trong text không."""
    for phrase in phrases_set:
        if phrase in lower_text:
            return True
    return False

# --- HÀM ROUTE_LOGIC ĐÃ CẢI TIẾN ---
def route_logic(input_data: Dict) -> str: # Đổi tên tham số để rõ ràng hơn
    """
    Phân loại câu hỏi thành 'general' hoặc 'legal' dựa trên từ khóa và ngữ cảnh.
    Ưu tiên kiểm tra các cụm từ general đầy đủ trước, sau đó đến từ khóa.
    """
    if not isinstance(input_data, dict):
        logger.error(f"Input không phải dictionary: {input_data}")
        # Không nên raise ValueError ở đây nếu đây là một phần của chain có thể có default
        # Trả về một route mặc định hoặc log và để chain xử lý tiếp
        return "general" # Hoặc "legal" tùy theo hành vi mong muốn khi input lỗi

    question_original = input_data.get("input", "")
    if not isinstance(question_original, str) or not question_original.strip():
        logger.warning("Thiếu key 'input' hoặc câu hỏi rỗng. Mặc định là 'general'.")
        return "general"

    question_lower = question_original.strip().lower()

    # 1. Kiểm tra các cụm từ general đầy đủ (ưu tiên cao nhất)
    if find_first_matching_phrase(question_lower, GENERAL_FULL_PHRASES):
        logger.info(f"[RouteLogic] Phân loại: general (khớp cụm từ general đầy đủ trong: '{question_lower[:50]}...')")
        return "general"

    # 2. Tokenize câu hỏi để kiểm tra từ khóa đơn lẻ
    # Lưu ý: underthesea.word_tokenize có thể chậm với nhiều request.
    # Cân nhắc chỉ tokenize nếu các bước kiểm tra cụm từ không thành công.
    try:
        question_tokenized_list = word_tokenize(question_lower, format="text").lower().split()
        # Chuyển thành set để tìm kiếm từ khóa hiệu quả hơn
        question_tokenized_set = set(question_tokenized_list)
    except Exception as e:
        logger.error(f"Lỗi khi tokenize câu hỏi '{question_lower[:50]}...': {e}. Sử dụng split() đơn giản.")
        question_tokenized_list = question_lower.split()
        question_tokenized_set = set(question_tokenized_list)


    # 3. Kiểm tra từ khóa legal (ưu tiên hơn từ khóa general đơn lẻ nếu có xung đột)
    # Lý do: các từ như "ai", "thông tin" có thể xuất hiện trong cả câu hỏi luật và general
    # nhưng nếu có từ khóa luật rõ ràng, nên ưu tiên luật.
    if find_first_matching_keyword(question_tokenized_set, LEGAL_KEYWORDS):
        logger.info(f"[RouteLogic] Phân loại: legal (khớp từ khóa legal trong: '{question_lower[:50]}...')")
        return "legal"

    # 4. Kiểm tra từ khóa general đơn lẻ
    # Cần cẩn thận với các từ khóa quá chung chung như "ai", "thông tin" nếu không có ngữ cảnh khác.
    # Có thể tạo một danh sách từ khóa general "mạnh" hơn.
    # Hiện tại, vẫn giữ nguyên logic tìm kiếm trong GENERAL_SINGLE_KEYWORDS
    if find_first_matching_keyword(question_tokenized_set, GENERAL_SINGLE_KEYWORDS):
        # Kiểm tra thêm để tránh trường hợp "ai có quyền thừa kế" bị nhầm là general
        # Nếu câu hỏi chứa từ "ai" nhưng cũng có vẻ là câu hỏi luật (ví dụ, chứa "quyền", "nghĩa vụ", tên luật)
        # thì có thể đã được bắt ở bước LEGAL_KEYWORDS.
        # Logic này có thể cần tinh chỉnh thêm dựa trên các trường hợp thực tế.
        is_likely_general_despite_common_words = True
        if "ai" in question_tokenized_set or "thông tin" in question_tokenized_set:
            # Nếu "ai" hoặc "thông tin" xuất hiện, nhưng không có từ khóa luật nào khác,
            # thì khả năng cao là general. Nếu có từ khóa luật, bước 3 đã bắt.
            pass # Logic này ổn vì bước 3 (legal) được ưu tiên

        if is_likely_general_despite_common_words:
            logger.info(f"[RouteLogic] Phân loại: general (khớp từ khóa general đơn lẻ trong: '{question_lower[:50]}...')")
            return "general"


    # 5. Logic cũ của bạn về "câu hỏi tổng quát về pháp luật" có thể được tích hợp hoặc bỏ đi
    # nếu LEGAL_KEYWORDS đã đủ mạnh. Hiện tại, bỏ qua để đơn giản hóa.
    # if any(word in question_tokenized_set for word in {"ai", "người nào", "đối tượng", "có quyền", "được phép"}) and \
    #    any(word in question_tokenized_set for word in {"thừa kế", "kết hôn", "lao động", "sở hữu", "đất đai"}):
    #     logger.info(f"[RouteLogic] Phân loại: legal (câu hỏi tổng quát về pháp luật)")
    #     return "legal"

    # 6. Mặc định là "legal" (NHƯ BẠN YÊU CẦU ĐỂ ƯU TIÊN TRẢ LỜI LUẬT)
    #    Tuy nhiên, nếu mục tiêu là hỏi "bạn là ai" phải ra general, thì mặc định nên là "general".
    #    Hãy xem xét kỹ hành vi mặc định bạn muốn.
    #    Nếu bạn muốn ưu tiên cho các câu hỏi không rõ ràng được xử lý như câu hỏi luật:
    logger.info(f"[RouteLogic] Phân loại: legal (mặc định, không khớp general, ưu tiên xử lý như câu hỏi luật)")
    return "legal"
    # HOẶC, nếu bạn muốn câu không rõ ràng được xử lý như general:
    # logger.info(f"[RouteLogic] Phân loại: general (mặc định, không khớp từ khóa nào)")
    # return "general" # <<<< SỬA THÀNH "general" NẾU MUỐN MẶC ĐỊNH LÀ GENERAL