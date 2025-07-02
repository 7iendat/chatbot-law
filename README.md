---
title: JuriBot API
emoji: ⚖️🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 5000
pinned: false
---

# JuriBot API - Backend cho Trợ lý Pháp luật AI

![JuriBot Banner](URL_TO_YOUR_BANNER_IMAGE) <!-- Tùy chọn: Thêm một ảnh banner đẹp -->

Đây là kho lưu trữ mã nguồn cho phần backend của **JuriBot**, một hệ thống chatbot thông minh được xây dựng để tra cứu và giải đáp các thắc mắc về pháp luật Việt Nam. Hệ thống sử dụng kiến trúc **Retrieval-Augmented Generation (RAG)** tiên tiến để cung cấp các câu trả lời chính xác, đáng tin cậy và luôn được cập nhật từ một cơ sở dữ liệu vector chuyên biệt.

## ✨ Tính năng Nổi bật

*   **Hỏi-đáp Thông minh:** Trả lời các câu hỏi bằng ngôn ngữ tự nhiên về nhiều lĩnh vực pháp luật.
*   **Hiểu Ngữ cảnh Hội thoại:** Theo dõi và duy trì ngữ cảnh qua nhiều lượt hỏi-đáp.
*   **Truy xuất Nâng cao:** Tích hợp các kỹ thuật RAG tiên tiến:
    *   **Query Rewriting & Keyword Extraction:** "Dịch" câu hỏi của người dùng sang ngôn ngữ pháp lý và trích xuất từ khóa chính.
    *   **Hybrid Search:** Kết hợp tìm kiếm ngữ nghĩa (vector) và từ khóa (BM25).
    *   **Intent-based Boosting:** Ưu tiên đúng loại văn bản (Luật, Nghị định) dựa trên ý định câu hỏi.
    *   **Cross-Encoder Re-ranking:** Sắp xếp lại các kết quả để đảm bảo độ chính xác cao nhất.
*   **Hệ thống Xác thực Toàn diện:** Hỗ trợ đăng ký/đăng nhập bằng email và **Google OAuth 2.0** một cách an toàn.
*   **Quản lý Dữ liệu:**
    *   API cho phép quản trị viên upload tài liệu mới (PDF, DOCX, DOC).
    *   API tự động cào (scrape) dữ liệu từ các nguồn tin pháp luật để làm giàu cơ sở tri thức.
*   **Kiến trúc Hiện đại:** Xây dựng trên FastAPI, Docker, và các dịch vụ cloud mạnh mẽ.

## 🚀 Kiến trúc & Công nghệ

Hệ thống được xây dựng trên một kiến trúc microservices hiện đại, phân tán trên các nền tảng cloud hàng đầu.

| Thành phần | Công nghệ / Dịch vụ | Vai trò |
| :--- | :--- | :--- |
| **Backend API** | **FastAPI, Python, Docker, Gunicorn** | Xử lý logic, điều phối RAG chain, quản lý user. |
| **Frontend** | **Next.js, React, Tailwind CSS** | Giao diện người dùng, tương tác với chatbot. |
| **Vector DB** | **Weaviate Cloud Services (WCS)** | Lưu trữ, quản lý và tìm kiếm vector pháp luật. |
| **Cache & Session** | **Upstash (Redis)** | Lưu trữ lịch sử hội thoại và session OAuth. |
| **User Database** | **MongoDB Atlas** | Lưu trữ thông tin người dùng và tài khoản. |
| **LLM** | **Groq (Llama 3)** | Sinh câu trả lời, tiền xử lý và phân loại câu hỏi. |
| **Embedding Model** | **[Tên model embedding của bạn, ví dụ: `bkai-foundation-models/vietnamese-bi-encoder`]** | Chuyển đổi văn bản thành vector. |
| **Re-ranker Model** | **[Tên model cross-encoder, ví dụ: `cross-encoder/ms-marco-MiniLM-L-6-v2`]** | Sắp xếp lại kết quả tìm kiếm. |
| **Email Service** | **Resend** | Gửi email xác thực và các thông báo khác. |

## ⚙️ Hướng dẫn Cài đặt & Chạy Local

Để chạy dự án này trên máy local, vui lòng tham khảo **[PHỤ LỤC 1: HƯỚNG DẪN CÀI ĐẶT](link_den_file_huong_dan_cai_dat.md)** trong tài liệu báo cáo.
*(**Ghi chú:** Bạn có thể tạo một file `INSTALL.md` riêng và dẫn link đến nó ở đây)*

**Các bước chính:**
1.  **Clone repository:** `git clone https://github.com/7iendat/chatbot-law.git`
2.  **Cài đặt Dependencies:** `pip install -r requirements.txt` và `Docker`.
3.  **Cấu hình Biến môi trường:** Tạo file `.env` từ file `.env.example` và điền đầy đủ các API key, URL từ các dịch vụ cloud (WCS, Upstash, MongoDB Atlas, Google OAuth).
4.  **Chạy Docker Compose:** `docker-compose up --build`

## ☁️ Triển khai (Deployment)

Hệ thống được thiết kế để triển khai dễ dàng lên các nền tảng cloud:

*   **Backend (FastAPI - Docker):** Triển khai trên **Hugging Face Spaces**.
*   **Frontend (Next.js):** Triển khai trên **Vercel**.

Chi tiết các bước triển khai có trong **[PHỤ LỤC 1: HƯỚNG DẪN CÀI ĐẶT](link_den_file_huong_dan_cai_dat.md)**.

## Người thực hiện: Entidi Nguyenx



## 📜 Giấy phép

Dự án này được cấp phép dưới Giấy phép MIT. Xem file `LICENSE` để biết thêm chi tiết.
