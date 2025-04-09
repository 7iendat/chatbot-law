# config.py
import os

# --- Cấu hình Đường dẫn ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_TXT_FOLDER = os.path.join(BASE_DIR, "data")
# Cấu hình cho ChromaDB
CHROMA_PERSIST_DIR = os.path.join(BASE_DIR, "vector_store")
CHROMA_COLLECTION_NAME = "luat_vn_docs" # Đặt tên cho collection

# --- Cấu hình Model ---
# EMBEDDING_MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_REPO_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"

# --- Cấu hình Chunking và Retrieval ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
SEARCH_K = 5

# --- Cấu hình LLM Endpoint ---
LLM_TEMPERATURE = 0.2
LLM_MAX_NEW_TOKENS = 1024

# --- Cấu hình Prompt ---
PROMPT_TEMPLATE = """Sử dụng thông tin dưới đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời dựa trên thông tin này, hãy nói rằng bạn không biết, đừng cố bịa ra câu trả lời. Hãy trả lời bằng tiếng Việt một cách đầy đủ và rõ ràng.

Ngữ cảnh:
{context}

Câu hỏi: {question}

Câu trả lời chi tiết bằng tiếng Việt:"""