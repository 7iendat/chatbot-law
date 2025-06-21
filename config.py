# config.py
import os

# config.py
from dotenv import load_dotenv
load_dotenv()


API_HOST = "0.0.0.0"
API_PORT = 5000

# --- Cấu hình Đường dẫn ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CORE_DATA_FOLDER = os.path.join(BASE_DIR, "data", "core")
PENDING_UPLOADS_FOLDER = os.path.join(BASE_DIR, "data", "pending_uploads")
PROCESSED_FILES_FOLDER = os.path.join(BASE_DIR, "data", "processed_files")
FAILED_FILES_FOLDER = os.path.join(BASE_DIR, "data", "failed_files")
PROCESSED_HASH_LOG = os.path.join(BASE_DIR, "data", "processed_hashes.log")
TESSDATA_DIR = os.path.join(BASE_DIR, "data", "tessdata")

# Cấu hình cho DB
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
CHUNK_OVERLAP = 150
SEARCH_K = 10

# --- Cấu hình LLM Endpoint ---
LLM_TEMPERATURE = 0.2
LLM_MAX_NEW_TOKENS = 1024

# REDIS_HOST = "epic-whale-20059.upstash.io"
# REDIS_PORT = 6379
# REDIS_PASSWORD = "AU5bAAIjcDFmODMwYzI5MGE4YWI0OTc4ODQ4NzJmMWY1MTI0MjM5NnAxMA"

SECRET_KEY = "1eb10877ca31c8589ef2e9c7eebd1e68"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

LLAMA_CLOUD_API_KEY="llx-yvHfZ8BOQL6imOd2cudcMKpFcsEnjiAdBEy8uQANsCujCiOW"

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET")

FRONTEND_URL = "http://localhost:3000"