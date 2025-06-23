# =================================================================
# STAGE 1: BUILDER - Stage để cài đặt các dependencies nặng
# =================================================================
# Sử dụng image đầy đủ để có các công cụ build cần thiết
FROM python:3.10 as builder

# Cập nhật và cài đặt các gói hệ thống cho việc build
# Chỉ cài những gì thực sự cần để `pip install` hoạt động
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Thiết lập thư mục làm việc
WORKDIR /app

# Tạo một môi trường ảo (virtual environment) để quản lý dependencies
# Đây là một thực hành tốt, giúp cô lập thư viện
RUN python -m venv /opt/venv
# Kích hoạt venv cho các lệnh RUN tiếp theo
ENV PATH="/opt/venv/bin:$PATH"

# Sao chép file requirements trước để tận dụng Docker layer caching
COPY requirements.txt .

# Cài đặt tất cả các thư viện Python trong một lệnh RUN duy nhất
# Điều này giúp tối ưu hóa số lượng layer của Docker
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# =================================================================
# STAGE 2: FINAL - Stage cuối cùng, nhỏ gọn để chạy ứng dụng
# =================================================================
# Bắt đầu từ một image slim siêu nhẹ
FROM python:3.10-slim

# Cài đặt chỉ các dependencies hệ thống cần thiết cho RUNTIME
# Không cần `build-essential`, `git`, `curl` ở đây nữa
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Thiết lập thư mục làm việc
WORKDIR /app

# Sao chép môi trường ảo đã được cài đặt sẵn từ stage builder
COPY --from=builder /opt/venv /opt/venv

# Kích hoạt virtual environment cho container
ENV PATH="/opt/venv/bin:$PATH"

# Thiết lập các biến môi trường quan trọng
# Thư mục cache sẽ nằm bên trong container
ENV HF_HOME=/app/cache
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1
# Đảm bảo log Python hiển thị ngay lập tức, rất quan trọng cho việc debug trên Render
ENV PYTHONUNBUFFERED=1

# Sao chép toàn bộ mã nguồn của ứng dụng
COPY . .

# Tải trước (pre-download/bake) các model vào trong image
# Điều này giúp giảm đáng kể thời gian khởi động (cold start) trên Render.
# Các model sẽ được lưu vào thư mục cache đã định nghĩa bởi HF_HOME.
# **QUAN TRỌNG**: Đảm bảo tên model ở đây khớp chính xác với tên trong file config.py của bạn.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder')"
RUN python -c "from langchain_community.cross_encoders import HuggingFaceCrossEncoder; HuggingFaceCrossEncoder(model_name='cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Mở cổng mà ứng dụng sẽ lắng nghe bên trong container
# Port này phải khớp với port trong lệnh CMD
EXPOSE 10000

# Lệnh chạy ứng dụng cho PRODUCTION sử dụng Gunicorn
# Gunicorn ổn định và hiệu quả hơn Uvicorn --reload
# Nó sẽ tự động sử dụng biến $PORT do Render cung cấp
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:10000", "--timeout", "120"]