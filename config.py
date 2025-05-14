# config.py
import os


API_HOST = "0.0.0.0"
API_PORT = 5000

# --- Cấu hình Đường dẫn ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_TXT_FOLDER = os.path.join(BASE_DIR, "data")
TESSDATA_DIR = os.path.join(BASE_DIR, "..", "data", "tessdata")
# Cấu hình cho DB
# CHROMA_PERSIST_DIR = os.path.join(BASE_DIR, "vector_store")
# CHROMA_COLLECTION_NAME = "luat_vn_docs" # Đặt tên cho collection
WEAVIATE_URL = "http://weaviate:8080"
WEAVIATE_COLLECTION_NAME = "LawDocuments"

# --- Cấu hình Model ---
# EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
EMBEDDING_MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder"
# EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL_NAME = "llama3-70b-8192"
# RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --- Cấu hình Chunking và Retrieval ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 250
SEARCH_K = 10

# --- Cấu hình LLM Endpoint ---
LLM_TEMPERATURE = 0.2
LLM_MAX_NEW_TOKENS = 1024

# REDIS_HOST = "epic-whale-20059.upstash.io"
# REDIS_PORT = 6379
# REDIS_PASSWORD = "AU5bAAIjcDFmODMwYzI5MGE4YWI0OTc4ODQ4NzJmMWY1MTI0MjM5NnAxMA"

SECRET_KEY = "4pqtnocaspevbuquv2vwkwvgf60t5lk1pbf7zsu1eikpc604cvktuuz5zl68hmvh"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60