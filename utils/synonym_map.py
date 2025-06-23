# utils/synonym_expander.py

import re
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ==============================================================================
# TỪ ĐIỂN ĐỒNG NGHĨA (SYNONYM MAP)
# ==============================================================================

# Cấu trúc: "thuật ngữ thông tục": "thuật ngữ pháp lý tương ứng"
# Key là tiếng thường, value là tiếng pháp lý.
# Mỗi lĩnh vực sẽ có một từ điển riêng.
# utils/synonym_expander.py

LEGAL_SYNONYM_MAP: Dict[str, Dict[str, str]] = {
    # ==========================================================================
    # LĨNH VỰC GIAO THÔNG
    # ==========================================================================
    "giao_thong": {
        # Hành vi vi phạm
        "vượt đèn đỏ": "không chấp hành hiệu lệnh của đèn tín hiệu giao thông",
        "uống rượu bia lái xe": "điều khiển xe trên đường mà trong máu hoặc hơi thở có nồng độ cồn",
        "thổi nồng độ cồn": "kiểm tra nồng độ cồn",
        "chạy quá tốc độ": "điều khiển xe chạy quá tốc độ quy định",
        "đi ngược chiều": "đi ngược chiều của đường một chiều, đi ngược chiều trên đường có biển “Cấm đi ngược chiều”",
        "không đội mũ bảo hiểm": "không đội “mũ bảo hiểm cho người đi mô tô, xe máy” hoặc đội “mũ bảo hiểm cho người đi mô tô, xe máy” không cài quai đúng quy cách",
        "sử dụng điện thoại khi lái xe": "dùng tay sử dụng điện thoại di động khi đang điều khiển xe",
        "lấn làn": "đi không đúng phần đường hoặc làn đường quy định",
        "không có bằng lái": "không có Giấy phép lái xe",
        "không có bảo hiểm xe": "không có Giấy chứng nhận bảo hiểm trách nhiệm dân sự của chủ xe cơ giới",

        # Đối tượng
        "xe máy": "xe mô tô, xe gắn máy",
        "ô tô": "xe ô tô",
        "xe hơi": "xe ô tô",

        # Giấy tờ & Thủ tục
        "bằng lái xe": "giấy phép lái xe",
        "gplx": "giấy phép lái xe",
        "đăng kiểm": "kiểm định an toàn kỹ thuật và bảo vệ môi trường",
        "phạt nguội": "xử phạt vi phạm hành chính được phát hiện thông qua phương tiện, thiết bị kỹ thuật nghiệp vụ",
        "sang tên xe": "thủ tục đăng ký sang tên xe",
    },

    # ==========================================================================
    # LĨNH VỰC HÔN NHÂN & GIA ĐÌNH
    # ==========================================================================
    "hon_nhan_gia_dinh": {
        "ly dị": "ly hôn",
        "ly dị đơn phương": "ly hôn theo yêu cầu của một bên",
        "ly dị thuận tình": "thuận tình ly hôn",
        "giành quyền nuôi con": "tranh chấp về quyền trực tiếp nuôi con sau khi ly hôn",
        "tiền cấp dưỡng": "nghĩa vụ cấp dưỡng",
        "độ tuổi kết hôn": "điều kiện về độ tuổi kết hôn",
        "đăng ký kết hôn": "thủ tục đăng ký kết hôn",
        "tài sản chung": "chế độ tài sản của vợ chồng",
        "chia tài sản": "chia tài sản chung trong thời kỳ hôn nhân hoặc khi ly hôn",
    },

    # ==========================================================================
    # LĨNH VỰC ĐẤT ĐAI & NHÀ Ở
    # ==========================================================================
    "dat_dai": {
        "làm sổ đỏ": "thủ tục cấp Giấy chứng nhận quyền sử dụng đất, quyền sở hữu nhà ở và tài sản khác gắn liền với đất",
        "sổ đỏ": "Giấy chứng nhận quyền sử dụng đất, quyền sở hữu nhà ở và tài sản khác gắn liền với đất",
        "sổ hồng": "Giấy chứng nhận quyền sử dụng đất, quyền sở hữu nhà ở và tài sản khác gắn liền với đất",
        "đền bù đất": "bồi thường, hỗ trợ, tái định cư khi Nhà nước thu hồi đất",
        "thu hồi đất": "trường hợp Nhà nước thu hồi đất",
        "tranh chấp đất đai": "giải quyết tranh chấp đất đai",
        "mua bán đất": "chuyển nhượng quyền sử dụng đất",
        "tách thửa": "thủ tục tách thửa đất",
    },

    # ==========================================================================
    # LĨNH VỰC LAO ĐỘNG
    # ==========================================================================
    "lao_dong": {
        "hợp đồng lao động": "hợp đồng lao động",
        "hđlđ": "hợp đồng lao động",
        "thử việc": "hợp đồng thử việc",
        "bị đuổi việc": "xử lý kỷ luật sa thải hoặc đơn phương chấm dứt hợp đồng lao động",
        "đòi lương": "tranh chấp lao động về tiền lương",
        "bảo hiểm xã hội": "chế độ bảo hiểm xã hội",
        "bhxh": "bảo hiểm xã hội",
        "thất nghiệp": "bảo hiểm thất nghiệp",
    },

    # ==========================================================================
    # LĨNH VỰC HÌNH SỰ
    # ==========================================================================
    "hinh_su": {
        "tự thú": "tự thú, đầu thú",
        "bị bắt": "bị bắt, tạm giữ, tạm giam",
        "tại ngoại": "biện pháp ngăn chặn tại ngoại, cấm đi khỏi nơi cư trú",
        "án treo": "hưởng án treo",
        "xóa án tích": "điều kiện xóa án tích",
    },

    # ==========================================================================
    # LĨNH VỰC DOANH NGHIỆP
    # ==========================================================================
    "doanh_nghiep": {
        "mở công ty": "thủ tục thành lập doanh nghiệp",
        "đăng ký kinh doanh": "đăng ký doanh nghiệp",
        "giải thể công ty": "thủ tục giải thể doanh nghiệp",
        "phá sản": "thủ tục phá sản doanh nghiệp",
    },

    # ==========================================================================
    # THUẬT NGỮ CHUNG (GENERAL)
    # ==========================================================================
    "general": {
        # Câu hỏi về mức phạt
        "phạt bao nhiêu tiền": "mức xử phạt hành chính là bao nhiêu",
        "phạt bao nhiêu": "mức xử phạt",
        "bị phạt gì": "hình thức xử phạt",
        "mức phạt": "mức xử phạt",

        # Câu hỏi về thủ tục, giấy tờ
        "cần giấy tờ gì": "hồ sơ bao gồm những giấy tờ gì",
        "thủ tục như thế nào": "trình tự, thủ tục thực hiện như thế nào",
        "làm thế nào": "cách thức thực hiện",
        "phải làm sao": "quy trình giải quyết như thế nào",
        "nộp đơn ở đâu": "thẩm quyền giải quyết của cơ quan nào",

        # Câu hỏi về định nghĩa
        "là gì": "được định nghĩa như thế nào",
        "được hiểu là gì": "được hiểu như thế nào",

        # Câu hỏi về điều kiện, quyền lợi
        "khi nào thì được": "điều kiện để được",
        "có được không": "có được phép hay không",
        "ai được": "đối tượng nào được hưởng",
    }
}


# ==============================================================================
# HÀM LOGIC ĐỂ VIẾT LẠI CÂU HỎI
# ==============================================================================

def rewrite_query_with_legal_synonyms(query: str, field: Optional[str] = None) -> str:
    """
    CẢI TIẾN V2: Đảm bảo thay thế hoạt động.
    """
    rewritten_query = query

    keys_to_check = []
    keys_to_check.extend(LEGAL_SYNONYM_MAP.get("general", {}).items())
    if field and field in LEGAL_SYNONYM_MAP:
        keys_to_check.extend(LEGAL_SYNONYM_MAP[field].items())

    # Sắp xếp theo độ dài key giảm dần
    sorted_items = sorted(keys_to_check, key=lambda item: len(item[0]), reverse=True)

    for colloquial_term, legal_term in sorted_items:
        # Sử dụng re.sub với cờ IGNORECASE để thay thế không phân biệt hoa thường
        # `\b` đảm bảo chỉ khớp toàn bộ từ/cụm từ
        pattern = r'\b' + re.escape(colloquial_term) + r'\b'
        # re.sub sẽ tự động tìm và thay thế, không cần `re.search` trước
        new_query, count = re.subn(pattern, legal_term, rewritten_query, flags=re.IGNORECASE)
        if count > 0:
            logger.info(f"Query Rewriting: Replaced '{colloquial_term}' with '{legal_term}'")
            rewritten_query = new_query

    return rewritten_query