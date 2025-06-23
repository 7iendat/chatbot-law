# config.py
import os

# config.py
from dotenv import load_dotenv
load_dotenv()


API_HOST = os.environ.get("API_HOST")
API_PORT = os.environ.get("API_PORT")

# --- Cấu hình Đường dẫn ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CORE_DATA_FOLDER = os.path.join(BASE_DIR, "data", "core")
PENDING_UPLOADS_FOLDER = os.path.join(BASE_DIR, "data", "pending_uploads")
PROCESSED_FILES_FOLDER = os.path.join(BASE_DIR, "data", "processed_files")
FAILED_FILES_FOLDER = os.path.join(BASE_DIR, "data", "failed_files")
PROCESSED_HASH_LOG = os.path.join(BASE_DIR, "data", "processed_hashes.log")
LEGAL_DIC_FOLDER = os.path.join(BASE_DIR, "data", "dictionary")

# Cấu hình cho DB
WEAVIATE_URL = os.environ.get("WEAVIATE_URL")
WEAVIATE_COLLECTION_NAME = os.environ.get("WEAVIATE_COLLECTION_NAME")
WEAVIATE_API_KEY = os.environ.get("WEAVIATE_API_KEY")

# --- Cấu hình Model ---
EMBEDDING_MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GROQ_MODEL_NAME = "llama-3.3-70b-versatile"



LEGAL_DOC_TYPES = ["Luật", "Bộ luật", "Nghị định", "Thông tư", "Quyết định", "Pháp lệnh", "Nghị quyết", "Chỉ thị", "Hiến pháp"]
MAX_CHUNK_SIZE = 3000  # Kích thước tối đa cho một chunk trước khi bị chia nhỏ hơn
CHUNK_OVERLAP = 300

REDIS_URL = os.environ.get("REDIS_URL")

SECRET_KEY = os.environ.get("SECRET_KEY")
ALGORITHM = os.environ.get("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES", 60)

LLAMA_CLOUD_API_KEY=os.environ.get("LLAMA_CLOUD_API_KEY")

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET")

FRONTEND_URL = os.environ.get("FRONTEND_URL")

CHECKPOINT_FILE = "processed_files.log"