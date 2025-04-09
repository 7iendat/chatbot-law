# api.py
import os
import torch
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from dotenv import load_dotenv
from contextlib import asynccontextmanager
import uvicorn
import time
import gc

# Import từ các file khác
import config
import utils # Vẫn cần nếu bạn muốn thêm API endpoint để làm sạch text chẳng hạn
import rag_components

# --- Pydantic Models (Giữ nguyên) ---
class QueryRequest(BaseModel):
    question: str
class SourceDocument(BaseModel):
    source: str | None = None
    page_content_preview: str | None = None
class AnswerResponse(BaseModel):
    answer: str
    sources: list[SourceDocument] | None = None
    processing_time: float

# --- Biến toàn cục (Giữ nguyên) ---
app_state = {
    "embeddings": None,
    "vectorstore": None,
    "llm": None,
    "qa_chain": None,
    "device": "cpu",
    "hf_api_token": None
}

# --- Hàm khởi tạo ĐƠN GIẢN HÓA cho API ---
def initialize_api_components():
    """Khởi tạo các thành phần cần thiết cho API (Tải model và vector store)."""
    print("--- Bắt đầu Khởi tạo API Components ---")
    load_dotenv()
    app_state["hf_api_token"] = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if not app_state["hf_api_token"]:
        print("Cảnh báo: HUGGINGFACEHUB_API_TOKEN không được cung cấp.")

    app_state["device"] = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Sử dụng thiết bị: {app_state['device']}")

    # 1. Tải Embedding Model
    print("Đang tải Embedding Model...")
    app_state["embeddings"] = rag_components.get_huggingface_embeddings(
        config.EMBEDDING_MODEL_NAME, app_state["device"]
    )
    if not app_state["embeddings"]: return # Dừng nếu lỗi

    # 2. Tải Vector Store (ChromaDB) - KHÔNG truyền chunks
    print("Đang tải Vector Store...")
    app_state["vectorstore"] = rag_components.create_or_load_chroma_vectorstore(
        embeddings=app_state["embeddings"],
        persist_directory=config.CHROMA_PERSIST_DIR,
        collection_name=config.CHROMA_COLLECTION_NAME,
        chunks=None # << QUAN TRỌNG: Không truyền chunks ở đây
    )
    if not app_state["vectorstore"]: return # Dừng nếu lỗi

    # 3. Tải LLM
    print("Đang tải LLM...")
    app_state["llm"] = rag_components.get_huggingface_endpoint_llm(
        config.LLM_REPO_ID,
        app_state["hf_api_token"],
        config.LLM_TEMPERATURE,
        config.LLM_MAX_NEW_TOKENS
    )
    if not app_state["llm"]: return # Dừng nếu lỗi

    # 4. Tạo QA Chain
    print("Đang tạo QA Chain...")
    app_state["qa_chain"] = rag_components.create_qa_chain(
        app_state["llm"],
        app_state["vectorstore"],
        config.SEARCH_K,
        config.PROMPT_TEMPLATE
    )
    if not app_state["qa_chain"]: return # Dừng nếu lỗi

    print("--- Khởi tạo API Components hoàn tất ---")

# --- Lifespan và Phần còn lại của API (Giữ nguyên) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_api_components() # Gọi hàm khởi tạo mới
    yield
    print("API shutting down.")
    app_state.clear()

app = FastAPI(
    title="Chatbot Hỏi Đáp Luật Việt Nam API (Large Data Optimized)",
    description="API cho phép hỏi đáp về Luật Việt Nam dựa trên RAG, tối ưu cho dữ liệu lớn.",
    version="1.1.0",
    lifespan=lifespan
)

# Endpoint /ask và /health giữ nguyên như trước
@app.post("/ask", response_model=AnswerResponse, tags=["Chatbot"])
async def ask_question(request: QueryRequest):
    # ... (code endpoint giữ nguyên) ...
    start_time = time.time()
    question = request.question
    if not app_state.get("qa_chain"):
        raise HTTPException(status_code=503, detail="Service Unavailable: QA Chain chưa sẵn sàng.")
    print(f"Received question: {question}")
    try:
        result = app_state["qa_chain"].invoke({"query": question})
        end_time = time.time()
        answer = result.get("result", "Không thể tạo câu trả lời.")
        sources_list = []
        if result.get("source_documents"):
            for doc in result["source_documents"]:
                 sources_list.append(SourceDocument(
                     source=doc.metadata.get('source', 'N/A'),
                     page_content_preview=doc.page_content[:200] + "..."
                 ))
        return AnswerResponse(
            answer=answer,
            sources=sources_list if sources_list else None,
            processing_time=round(end_time - start_time, 2)
        )
    except Exception as e:
        print(f"Error during QA Chain invocation: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


@app.get("/health", tags=["Status"])
async def health_check():
    # ... (code endpoint giữ nguyên) ...
    if app_state.get("qa_chain"):
        return {"status": "OK", "message": "QA Chain is ready."}
    else:
        return {"status": "Initializing or Failed", "message": "QA Chain is not ready."}


# Chạy API (nếu chạy file này trực tiếp)
if __name__ == "__main__":
    print("Chạy FastAPI server với Uvicorn...")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True) # Bỏ reload=True khi deploy