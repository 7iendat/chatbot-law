from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
import config
import os
from dotenv import load_dotenv
from db.mongoDB import user_collection, blacklist_collection
import torch
import rag_components
from db.redis import r
from config import SECRET_KEY, ALGORITHM
from utils.utils import load_legal_dictionary
from time_priority_retriever import create_retriever

# Bearer token security scheme
bearer_scheme = HTTPBearer()

def get_app_state(request: Request):
    return request.app.state.app_state


# Giả sử app_state là dict lưu các component khởi tạo từ main.py
def initialize_api_components(app_state):
    """Khởi tạo các thành phần cần thiết cho API (Tải model và vector store)."""
    print(f"***** Bắt đầu Khởi tạo API Components *****")

    load_dotenv()
    # --- Kiểm tra kết nối tới Redis ---
    app_state['redis'] = r
    app_state['dict'] = load_legal_dictionary('./data/dictionary/legal_terms.json')

    # --- Kiểm tra kết nối tới MongoDB ---
    if user_collection is  None:
        raise HTTPException(status_code=500, detail="Lỗi kết nối tới database.")

    app_state["groq_api_key"] = os.environ.get("GROQ_API_KEY") # Lấy key Groq
    if not app_state["groq_api_key"]:
        print(f"Cảnh báo: GROQ_API_KEY không được cung cấp. API có thể không hoạt động.")
        raise HTTPException(status_code=500, detail="Missing GROQ API Key")

    app_state["device"] = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Sử dụng thiết bị: {app_state['device']}")

    # 1. Tải Embedding Model (giữ nguyên)
    print(f"Đang tải Embedding Model...")
    app_state["embeddings"] = rag_components.get_huggingface_embeddings(
        config.EMBEDDING_MODEL_NAME, app_state["device"]
    )
    if not app_state["embeddings"]:
        raise HTTPException(status_code=500, detail="Failed to load embedding model")

    # 2. Tải Vector Store (ChromaDB) (giữ nguyên)
    print(f"Đang tải Vector Store...")
    app_state["vectorstore"] = rag_components.create_or_load_chroma_vectorstore(
        embeddings=app_state["embeddings"],
        persist_directory=config.CHROMA_PERSIST_DIR,
        collection_name=config.CHROMA_COLLECTION_NAME,
        chunks=None # Không cung cấp chunks khi khởi tạo API
    )



    if not app_state["vectorstore"]:
         raise HTTPException(status_code=500, detail="Failed to load or create Vectorstore")

    # 3. Tải LLM (thay đổi để dùng Groq)
    print(f"Đang tải LLM (Groq)...")
    llm = rag_components.get_groq_llm(
        app_state["groq_api_key"],
        temperature=config.LLM_TEMPERATURE,
        max_new_tokens=config.LLM_MAX_NEW_TOKENS
    )

    app_state["llm"] = llm


    if not app_state["llm"]:
        raise HTTPException(status_code=500, detail="Failed to load LLM")

    # 4. Tạo retriever (giữ nguyên)
    print(f"=> Đang tạo retriever...")
    app_state["retriever"] = create_retriever(
        app_state["vectorstore"],
        app_state["embeddings"],
        llm
    )
    if app_state["retriever"] is None:
        raise HTTPException(status_code=500, detail="Failed to create retriever")
    print(f"=> Đã tạo retriever thành công.")

    # 5. Tạo QA Chain (giữ nguyên)
    print(f"Đang tạo QA Chain...")
    app_state["qa_chain"] = rag_components.create_qa_chain(
        app_state["llm"],
        app_state["vectorstore"],
        app_state["retriever"],
        chat_id=None
    )
    if app_state["qa_chain"] is None:
        raise HTTPException(status_code=500, detail="Failed to create QA Chain")

    print(f"***** Khởi tạo API Components hoàn tất *****")


# Lấy user hiện tại từ token
def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    token = credentials.credentials

    # Kiểm tra token có bị thu hồi (blacklist) không
    if blacklist_collection.find_one({"token": token}):
        raise HTTPException(status_code=401, detail="Token đã bị thu hồi")

    try:
        # Giải mã token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")

        if email is None:
            raise HTTPException(status_code=401, detail="Token không hợp lệ")

        return email

    except JWTError:
        raise HTTPException(status_code=401, detail="Token không hợp lệ")
