from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
import os
from dotenv import load_dotenv
from db.mongoDB import user_collection, blacklist_collection
import torch
import rag_components
from db.redis import r
from config import SECRET_KEY, ALGORITHM,EMBEDDING_MODEL_NAME, LLM_TEMPERATURE, LLM_MAX_NEW_TOKENS, WEAVIATE_COLLECTION_NAME, WEAVIATE_URL, SEARCH_K
from utils.utils import load_legal_dictionary
# from time_priority_retriever import WeaviateHybridRetriever

import logging
from db.weaviateDB import connect_to_weaviate
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Bearer token security scheme
bearer_scheme = HTTPBearer()

def get_app_state(request: Request):
    return request.app.state.app_state


def initialize_api_components(app_state):
    """Khởi tạo các thành phần cần thiết cho API """
    logger.info("🔸Bắt đầu Khởi tạo API Components")

    load_dotenv()
    # --- Kiểm tra kết nối tới Redis ---
    app_state['redis'] = r
    app_state['dict'] = load_legal_dictionary('./data/dictionary/legal_terms.json')
    app_state["weaviateDB"] = connect_to_weaviate()
    # --- Kiểm tra kết nối tới MongoDB ---
    if user_collection is  None or app_state["weaviateDB"] is None:
        logger.error("🔸Lỗi kết nối tới MongoDB hoặc Weaviate.")
        raise HTTPException(status_code=500, detail="Lỗi kết nối tới database.")

    app_state["groq_api_key"] = os.environ.get("GROQ_API_KEY") # Lấy key Groq
    if not app_state["groq_api_key"]:
        logger.error("🔸GROQ API Key không được cung cấp.")
        raise HTTPException(status_code=500, detail="Missing GROQ API Key")

    app_state["device"] = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"🔸Sử dụng thiết bị: {app_state['device']}")

    # 1. Tải Embedding Model (giữ nguyên)
    print(f"Đang tải Embedding Model...")
    app_state["embeddings"] = rag_components.get_huggingface_embeddings(
        EMBEDDING_MODEL_NAME, app_state["device"]
    )
    if not app_state["embeddings"]:
        raise HTTPException(status_code=500, detail="Failed to load embedding model")

    # 2. Tải Vector Store (ChromaDB) (giữ nguyên)
    print(f"Đang tải Vector Store...")
    # app_state["vectorstore"] = rag_components.create_or_load_chroma_vectorstore(
    #     embeddings=app_state["embeddings"],
    #     persist_directory=CHROMA_PERSIST_DIR,
    #     collection_name=CHROMA_COLLECTION_NAME,
    #     chunks=None # Không cung cấp chunks khi khởi tạo API
    # )

    app_state['vectorstore'] = rag_components.create_or_load_vectorstore(
        embeddings=app_state["embeddings"],
        weaviate_url=WEAVIATE_URL,
        collection_name=WEAVIATE_COLLECTION_NAME,
        weaviate_client=app_state["weaviateDB"],
        chunks=None,
    )

    if not app_state["vectorstore"]:
         raise HTTPException(status_code=500, detail="Failed to load or create Vectorstore")

    # 3. Tải LLM (thay đổi để dùng Groq)
    logger.info(f"🔸Đang tải LLM (Groq)...")
    llm = rag_components.get_groq_llm(
        app_state["groq_api_key"],
        temperature=LLM_TEMPERATURE,
        max_new_tokens=LLM_MAX_NEW_TOKENS
    )

    app_state["llm"] = llm
    logger.info(f"🔸Tải LLM (Groq) thanh cong")


    if not app_state["llm"]:
        raise HTTPException(status_code=500, detail="Failed to load LLM")

    # 4. Tạo retriever (giữ nguyên)
    logger.info(f"🔸Đang tạo retriever...")
    # app_state["retriever"] = create_retriever(
    #     app_state["vectorstore"],
    #     app_state["embeddings"],
    #     llm,
    # )
    # app_state["retriever"] = create_retriever(
    #     app_state["vectorstore"],
    #     app_state["embeddings"],
    #     llm,
    #     config={
    #         "recent_years": 5,
    #         "primary_docs": 4,
    #         "total_docs": 6,
    #         "content_min_chars": 1500,
    #         "use_llm_paraphrase": False
    #     }
    # )

    app_state["retriever"] = app_state["vectorstore"].as_retriever(
        search_type="similarity",  # Valid value to bypass Pydantic validation
        search_kwargs={"k": SEARCH_K, "alpha": 0.6}
    )

    if app_state["retriever"] is None:
        raise HTTPException(status_code=500, detail="Failed to create retriever")
    logger.info(f"🔸Đã tạo retriever thành công.")

    # 5. Tạo QA Chain (giữ nguyên)
    logger.info(f"🔸Đang tạo QA Chain...")
    app_state["qa_chain"] = rag_components.create_qa_chain(
        app_state["llm"],
        app_state["vectorstore"],
        app_state["retriever"],
    )
    if app_state["qa_chain"] is None:
        raise HTTPException(status_code=500, detail="Failed to create QA Chain")

    logger.info(f"🔸Khởi tạo API Components hoàn tất ")


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
