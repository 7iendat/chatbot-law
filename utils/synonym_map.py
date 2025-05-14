"""
Bản đồ từ đồng nghĩa để ánh xạ ngôn ngữ thông dụng với ngôn ngữ chuyên ngành trong lĩnh vực pháp luật Việt Nam.
File này cung cấp các cặp từ/cụm từ để cải thiện truy xuất tài liệu từ vector database, bao phủ các lĩnh vực pháp luật phổ biến.

Cấu trúc:
- Mỗi lĩnh vực pháp luật (như 'giao_thong', 'hanh_chinh', 'thue', v.v.) chứa một dictionary.
- Key: Từ/cụm từ thông dụng (ngôn ngữ người dùng hay dùng).
- Value: Danh sách các từ/cụm từ chuyên ngành tương ứng trong tài liệu pháp luật.

Cách sử dụng:
- Import vào code chính (ví dụ: create_retriever) và sử dụng để mở rộng truy vấn hoặc lọc tài liệu.
- Có thể mở rộng bằng cách thêm các lĩnh vực hoặc cặp từ đồng nghĩa mới.
"""

SYNONYM_MAP = {
    "giao_thong": {
        "vượt đèn đỏ": [
            "không chấp hành hiệu lệnh của đèn tín hiệu giao thông",
            "vi phạm tín hiệu đèn đỏ",
            "đi qua đèn đỏ",
            "không dừng tại tín hiệu đèn đỏ"
        ],
        "phạt bao nhiêu": [
            "mức phạt hành chính",
            "số tiền phạt",
            "mức xử phạt",
            "khoản phạt"
        ],
        "xe máy": [
            "mô tô",
            "xe gắn máy",
            "xe hai bánh có động cơ"
        ],
        "ô tô": [
            "xe ô tô",
            "xe chở người bốn bánh có động cơ",
            "xe hơi"
        ],
        "chạy quá tốc độ": [
            "vượt quá tốc độ cho phép",
            "lái xe quá tốc độ quy định",
            "vi phạm tốc độ tối đa"
        ],
        "đi ngược chiều": [
            "đi ngược chiều trên đường một chiều",
            "vi phạm quy tắc giao thông đường bộ về hướng đi",
            "lái xe ngược chiều"
        ],
        "không đội mũ bảo hiểm": [
            "không sử dụng mũ bảo hiểm khi điều khiển xe mô tô",
            "vi phạm quy định về đội mũ bảo hiểm"
        ],
        "tước bằng lái": [
            "tước giấy phép lái xe",
            "bị thu hồi giấy phép lái xe",
            "tạm giữ giấy phép lái xe"
        ],
        "đèn giao thông": [
            "tín hiệu giao thông",
            "đèn tín hiệu",
            "hệ thống điều khiển giao thông"
        ],
        "đỗ xe sai chỗ": [
            "dừng xe không đúng nơi quy định",
            "đỗ xe trái phép",
            "vi phạm quy định về dừng đỗ xe"
        ],
        "uống rượu bia lái xe": [
            "điều khiển phương tiện có nồng độ cồn vượt mức cho phép",
            "lái xe trong tình trạng có cồn",
            "vi phạm quy định về nồng độ cồn"
        ]
    },
    "hanh_chinh": {
        "đăng ký hộ khẩu": [
            "đăng ký thường trú",
            "đăng ký hộ khẩu thường trú",
            "thủ tục nhập khẩu"
        ],
        "chứng minh nhân dân": [
            "căn cước công dân",
            "thẻ căn cước",
            "giấy chứng minh nhân dân"
        ],
        "phạt hành chính": [
            "xử phạt vi phạm hành chính",
            "mức phạt hành chính",
            "khoản phạt hành chính"
        ],
        "khiếu nại": [
            "khiếu nại hành chính",
            "đơn khiếu nại",
            "yêu cầu giải quyết tranh chấp hành chính"
        ],
        "đăng ký tạm trú": [
            "đăng ký tạm trú tại địa phương",
            "thủ tục khai báo tạm trú",
            "đăng ký nơi ở tạm thời"
        ],
        "cấp giấy phép": [
            "cấp phép hành chính",
            "giấy phép hoạt động",
            "thủ tục cấp phép"
        ]
    },
    "thue": {
        "thuế thu nhập": [
            "thuế thu nhập cá nhân",
            "thuế TNCN",
            "thuế đối với thu nhập cá nhân"
        ],
        "trốn thuế": [
            "gian lận thuế",
            "trốn tránh nghĩa vụ thuế",
            "vi phạm quy định về nộp thuế"
        ],
        "thuế VAT": [
            "thuế giá trị gia tăng",
            "thuế GTGT",
            "thuế đối với hàng hóa và dịch vụ"
        ],
        "miễn thuế": [
            "miễn giảm thuế",
            "ưu đãi thuế",
            "không phải nộp thuế"
        ],
        "khấu trừ thuế": [
            "khấu trừ thuế tại nguồn",
            "trừ thuế trước khi trả",
            "giảm trừ thuế"
        ]
    },
    "lao_dong": {
        "nghỉ thai sản": [
            "chế độ thai sản",
            "nghỉ sinh con",
            "trợ cấp thai sản"
        ],
        "sa thải": [
            "chấm dứt hợp đồng lao động",
            "cho thôi việc",
            "kỷ luật sa thải"
        ],
        "bảo hiểm xã hội": [
            "bảo hiểm xã hội bắt buộc",
            "BHXH",
            "chế độ bảo hiểm xã hội"
        ],
        "lương tối thiểu": [
            "mức lương tối thiểu vùng",
            "lương cơ bản tối thiểu",
            "mức lương thấp nhất"
        ],
        "hợp đồng lao động": [
            "hợp đồng làm việc",
            "thỏa thuận lao động",
            "hợp đồng công việc"
        ],
        "ngày nghỉ phép": [
            "nghỉ phép năm",
            "ngày nghỉ có hưởng lương",
            "chế độ nghỉ phép"
        ]
    },
    "doanh_nghiep": {
        "đăng ký kinh doanh": [
            "đăng ký doanh nghiệp",
            "thành lập công ty",
            "giấy phép kinh doanh"
        ],
        "phá sản": [
            "tuyên bố phá sản",
            "giải thể doanh nghiệp",
            "không có khả năng thanh toán nợ"
        ],
        "cổ phần hóa": [
            "chuyển đổi thành công ty cổ phần",
            "tư nhân hóa doanh nghiệp",
            "phát hành cổ phần"
        ],
        "hợp đồng thương mại": [
            "hợp đồng kinh doanh",
            "thỏa thuận thương mại",
            "giao dịch kinh doanh"
        ],
        "giải thể công ty": [
            "chấm dứt hoạt động doanh nghiệp",
            "đóng cửa công ty",
            "thủ tục giải thể"
        ]
    },
    "dat_dai": {
        "sổ đỏ": [
            "giấy chứng nhận quyền sử dụng đất",
            "sổ hồng",
            "giấy chứng nhận quyền sở hữu nhà ở và đất"
        ],
        "chuyển nhượng đất": [
            "chuyển nhượng quyền sử dụng đất",
            "mua bán đất đai",
            "sang nhượng đất"
        ],
        "tranh chấp đất đai": [
            "tranh chấp quyền sử dụng đất",
            "mâu thuẫn đất đai",
            "khiếu kiện về đất"
        ],
        "thu hồi đất": [
            "nhà nước thu hồi đất",
            "cưỡng chế đất đai",
            "đền bù đất bị thu hồi"
        ],
        "cấp sổ đỏ": [
            "cấp giấy chứng nhận quyền sử dụng đất",
            "thủ tục cấp sổ đỏ",
            "đăng ký quyền sử dụng đất"
        ]
    },
    "hon_nhan": {
        "ly hôn": [
            "chấm dứt hôn nhân",
            "giải quyết ly hôn",
            "ly hôn theo pháp luật"
        ],
        "chia tài sản": [
            "phân chia tài sản chung",
            "giải quyết tài sản sau ly hôn",
            "phân chia tài sản hôn nhân"
        ],
        "nuôi con": [
            "quyền nuôi con",
            "trách nhiệm nuôi dưỡng con",
            "chăm sóc con sau ly hôn"
        ],
        "kết hôn": [
            "đăng ký kết hôn",
            "thủ tục kết hôn",
            "hôn nhân hợp pháp"
        ],
        "bạo lực gia đình": [
            "hành vi bạo lực gia đình",
            "bạo hành trong gia đình",
            "vi phạm pháp luật về bạo lực gia đình"
        ]
    },
    "hinh_su": {
        "trộm cắp": [
            "chiếm đoạt tài sản",
            "trộm cắp tài sản",
            "vi phạm điều luật về chiếm đoạt"
        ],
        "giết người": [
            "tội giết người",
            "cố ý gây tử vong",
            "vi phạm điều luật về tính mạng"
        ],
        "lừa đảo": [
            "lừa đảo chiếm đoạt tài sản",
            "gian lận tài sản",
            "tội lừa đảo"
        ],
        "tham nhũng": [
            "tham ô tài sản",
            "lạm dụng chức vụ quyền hạn",
            "tội tham nhũng"
        ],
        "ma túy": [
            "buôn bán ma túy",
            "tàng trữ chất ma túy",
            "vi phạm pháp luật về ma túy"
        ]
    },
    "dan_su": {
        "hợp đồng": [
            "hợp đồng dân sự",
            "thỏa thuận dân sự",
            "giao dịch dân sự"
        ],
        "bồi thường": [
            "bồi thường thiệt hại",
            "trách nhiệm bồi thường",
            "đền bù thiệt hại"
        ],
        "nợ": [
            "khoản vay",
            "nghĩa vụ trả nợ",
            "nợ dân sự"
        ],
        "tranh chấp hợp đồng": [
            "mâu thuẫn hợp đồng",
            "tranh chấp giao dịch dân sự",
            "khiếu kiện hợp đồng"
        ],
        "thừa kế": [
            "phân chia di sản thừa kế",
            "quyền thừa kế",
            "di chúc hợp pháp"
        ]
    },
    "moi_truong": {
        "ô nhiễm": [
            "gây ô nhiễm môi trường",
            "vi phạm quy định về bảo vệ môi trường",
            "ô nhiễm nguồn nước"
        ],
        "xử lý rác thải": [
            "quản lý chất thải",
            "xử lý rác thải sinh hoạt",
            "vi phạm quy định về chất thải"
        ],
        "phạt môi trường": [
            "xử phạt vi phạm về môi trường",
            "mức phạt môi trường",
            "khoản phạt bảo vệ môi trường"
        ],
        "bảo vệ môi trường": [
            "tuân thủ quy định bảo vệ môi trường",
            "trách nhiệm bảo vệ môi trường",
            "chính sách môi trường"
        ]
    },
    "y_te": {
        "khám bệnh": [
            "khám chữa bệnh",
            "dịch vụ y tế",
            "chăm sóc sức khỏe"
        ],
        "bảo hiểm y tế": [
            "bảo hiểm y tế bắt buộc",
            "BHYT",
            "chế độ bảo hiểm y tế"
        ],
        "sai sót y khoa": [
            "sai phạm trong khám chữa bệnh",
            "lỗi y khoa",
            "vi phạm quy định y tế"
        ],
        "vắc xin": [
            "tiêm chủng",
            "chương trình vắc xin",
            "phòng ngừa bệnh bằng vắc xin"
        ]
    },
    "giao_duc": {
        "học phí": [
            "mức học phí",
            "chi phí học tập",
            "phí giáo dục"
        ],
        "trường công": [
            "trường học công lập",
            "cơ sở giáo dục công lập",
            "trường nhà nước"
        ],
        "miễn học phí": [
            "miễn giảm học phí",
            "hỗ trợ học phí",
            "miễn phí giáo dục"
        ],
        "kỷ luật học sinh": [
            "xử lý vi phạm học sinh",
            "kỷ luật trong trường học",
            "quy định về hành vi học sinh"
        ]
    },
    "xay_dung": {
        "giấy phép xây dựng": [
            "cấp phép xây dựng",
            "giấy phép thi công",
            "thủ tục cấp phép xây dựng"
        ],
        "xây nhà không phép": [
            "xây dựng không có giấy phép",
            "vi phạm quy định về xây dựng",
            "xây dựng trái phép"
        ],
        "quy hoạch": [
            "quy hoạch đô thị",
            "kế hoạch sử dụng đất",
            "quy hoạch xây dựng"
        ],
        "an toàn xây dựng": [
            "quy định an toàn thi công",
            "bảo đảm an toàn xây dựng",
            "vi phạm an toàn công trình"
        ]
    },
    "thuong_mai": {
        "mua bán hàng hóa": [
            "giao dịch thương mại",
            "mua bán hàng hóa",
            "hợp đồng mua bán"
        ],
        "xuất nhập khẩu": [
            "hoạt động xuất nhập khẩu",
            "thương mại quốc tế",
            "giao dịch xuất nhập khẩu"
        ],
        "cạnh tranh không lành mạnh": [
            "vi phạm quy định về cạnh tranh",
            "hành vi cạnh tranh không lành mạnh",
            "lạm dụng vị trí cạnh tranh"
        ],
        "bảo vệ người tiêu dùng": [
            "quyền lợi người tiêu dùng",
            "bảo vệ lợi ích người mua",
            "luật bảo vệ người tiêu dùng"
        ]
    },
    "cong_nghe": {
        "bảo mật dữ liệu": [
            "bảo vệ dữ liệu cá nhân",
            "an ninh dữ liệu",
            "quy định về bảo mật thông tin"
        ],
        "tội phạm mạng": [
            "tội phạm công nghệ cao",
            "hành vi xâm phạm hệ thống mạng",
            "vi phạm pháp luật về công nghệ"
        ],
        "sở hữu trí tuệ": [
            "quyền sở hữu trí tuệ",
            "bản quyền",
            "sáng chế và nhãn hiệu"
        ],
        "phần mềm lậu": [
            "sử dụng phần mềm không có bản quyền",
            "vi phạm quyền sở hữu trí tuệ",
            "phần mềm bất hợp pháp"
        ]
    }
}

def get_synonyms(field, term):
    """
    Lấy danh sách từ đồng nghĩa cho một từ/cụm từ trong lĩnh vực cụ thể.

    Args:
        field (str): Lĩnh vực pháp luật (ví dụ: 'giao_thong', 'hanh_chinh').
        term (str): Từ/cụm từ cần tìm đồng nghĩa.

    Returns:
        list: Danh sách các từ/cụm từ đồng nghĩa, hoặc [] nếu không tìm thấy.
    """
    return SYNONYM_MAP.get(field, {}).get(term.lower(), [])