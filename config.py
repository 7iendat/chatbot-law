# config.py
import os
from langchain.output_parsers import RegexParser


API_HOST = "0.0.0.0"
API_PORT = 5000

# --- Cấu hình Đường dẫn ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_TXT_FOLDER = os.path.join(BASE_DIR, "data")
TESSDATA_DIR = os.path.join(BASE_DIR, "..", "data", "tessdata")
# Cấu hình cho ChromaDB
CHROMA_PERSIST_DIR = os.path.join(BASE_DIR, "vector_store")
CHROMA_COLLECTION_NAME = "luat_vn_docs" # Đặt tên cho collection

# --- Cấu hình Model ---
EMBEDDING_MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder"
# EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL_NAME = "llama3-70b-8192"

# --- Cấu hình Chunking và Retrieval ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
SEARCH_K = 5

# --- Cấu hình LLM Endpoint ---
LLM_TEMPERATURE = 0.2
LLM_MAX_NEW_TOKENS = 1024

# REDIS_HOST = "epic-whale-20059.upstash.io"
# REDIS_PORT = 6379
# REDIS_PASSWORD = "AU5bAAIjcDFmODMwYzI5MGE4YWI0OTc4ODQ4NzJmMWY1MTI0MjM5NnAxMA"