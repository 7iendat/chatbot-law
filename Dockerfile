FROM python:3.10-slim

# Cài đặt các gói hệ thống cần thiết
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    tesseract-ocr \
    tesseract-ocr-vie \
    poppler-utils \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    build-essential \
    git \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Thiết lập thư mục làm việc
WORKDIR /app

# Sao chép và cài đặt requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Cài đặt thêm thư viện cho Weaviate và nhúng
RUN pip install --no-cache-dir \
    langchain-weaviate \
    weaviate-client \
    sentence-transformers

# Thiết lập biến môi trường
ENV HF_HOME=/app/cache
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1
ENV TESSDATA_PREFIX=/app/data/tessdata/

# Sao chép mã nguồn
COPY . .

# Tải trước mô hình nhúng
RUN python -c "from langchain_huggingface import HuggingFaceEmbeddings; HuggingFaceEmbeddings(model_name='bkai-foundation-models/vietnamese-bi-encoder')"

# Mở cổng cho FastAPI
EXPOSE 5000

# Chạy ứng dụng
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000", "--reload"]