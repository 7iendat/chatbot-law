FROM python:3.10-slim

# Cài đặt các gói cần thiết
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-vie \
    poppler-utils \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Thiết lập thư mục làm việc
WORKDIR /app

# Copy requirements.txt riêng để tận dụng cache khi build
COPY requirements.txt .

# Cài thư viện Python trước
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ code vào sau
COPY . .

# Thiết lập biến môi trường Tesseract cho tiếng Việt
ENV TESSDATA_PREFIX=/app/data/tessdata/

# Expose port cho FastAPI (nếu cần)
EXPOSE 5000

# Khởi động ứng dụng FastAPI
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "5000", "--reload"]
