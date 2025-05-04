FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-vie \
    poppler-utils \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    build-essential \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


ENV TRANSFORMERS_CACHE=/app/cache
ENV HF_HOME=/app/cache
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1


COPY . .


RUN python -c "from langchain_huggingface import HuggingFaceEmbeddings; HuggingFaceEmbeddings(model_name='BAAI/bge-m3')"


ENV TESSDATA_PREFIX=/app/data/tessdata/


EXPOSE 5000


CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000", "--reload"]
