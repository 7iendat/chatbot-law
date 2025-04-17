# api.py
import os
import torch
from fastapi import FastAPI, HTTPException,UploadFile, File,BackgroundTasks
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_core.documents import Document
from contextlib import asynccontextmanager
import uvicorn
import time
import gc
import prompt_templete
from typing import List
import asyncio
# Import từ các file khác
import config
import utils # Vẫn cần nếu bạn muốn thêm API endpoint để làm sạch text chẳng hạn
import rag_components
import redis
import uuid
from models import QueryRequest, SourceDocument, AnswerResponse,ChatHistoryResponse
from utils import save_chat_to_redis, get_redis_history

# Kết nối tới Redis
redis_url = "redis://localhost:6379"
r = redis.Redis.from_url(redis_url)
# --- Biến toàn cục (Giữ nguyên) ---
app_state = {
    "embeddings": None,
    "vectorstore": None,
    "llm": None,
    "qa_chain": None,
    "device": "cpu",
    "groq_api_key": None
}



# --- Hàm khởi tạo ĐƠN GIẢN HÓA cho API ---
def initialize_api_components():
    """Khởi tạo các thành phần cần thiết cho API (Tải model và vector store)."""
    print(f"***** Bắt đầu Khởi tạo API Components *****")

    load_dotenv()
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


    print("DEBUG => type vectorstore:", type(app_state["vectorstore"]))


    if not app_state["vectorstore"]:
         raise HTTPException(status_code=500, detail="Failed to load or create Vectorstore")

    # 3. Tải LLM (thay đổi để dùng Groq)
    print(f"Đang tải LLM (Groq)...")
    app_state["llm"] = rag_components.get_groq_llm(
        app_state["groq_api_key"],
        temperature=config.LLM_TEMPERATURE,
        max_new_tokens=config.LLM_MAX_NEW_TOKENS
    )
    if not app_state["llm"]:
        raise HTTPException(status_code=500, detail="Failed to load LLM")

    # 4. Tạo QA Chain (giữ nguyên)
    print(f"Đang tạo QA Chain...")
    app_state["qa_chain"] = rag_components.create_qa_chain(
        app_state["llm"],
        app_state["vectorstore"],
        config.SEARCH_K,
        redis_instance=redis_url,
        session_id=None # Không cần session_id khi khởi tạo API
    )
    if not app_state["qa_chain"]:
        raise HTTPException(status_code=500, detail="Failed to create QA Chain")

    print(f"***** Khởi tạo API Components hoàn tất *****")


# --- Lifespan và các phần còn lại của API (giữ nguyên, chỉ thay tên biến) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_api_components()
    yield
    print(f"API shutting down.")
    app_state.clear()

app = FastAPI(
    title="Chatbot Hỏi Đáp Luật Việt Nam API (Large Data + GroqCloud)",
    description="API cho phép hỏi đáp về Luật Việt Nam dựa trên RAG, tối ưu cho dữ liệu lớn và tốc độ.",
    version="1.2.0",
    lifespan=lifespan
)

@app.post("/ask", response_model=AnswerResponse, tags=["Chatbot"])
async def ask_question(request: QueryRequest):
    start_time = time.time()
    question = request.question
    # Tự sinh session_id nếu người dùng không truyền hoặc truyền là "string"
    session_id = request.session_id or str(uuid.uuid4())
    if not app_state.get("qa_chain"):
        raise HTTPException(status_code=503, detail="Service Unavailable: QA Chain chưa sẵn sàng.")

    print(f"Received question: {question}")
    try:
        raw_result = app_state["qa_chain"].invoke({
            "question": question,
            "chat_history": get_redis_history(r, session_id)
        })

        end_time = time.time()
        if isinstance(raw_result.get("answer"), dict):
            answer_data = raw_result["answer"]
            answer = answer_data.get("answer", "Không thể tạo câu trả lời.")
            source_documents = answer_data.get("source_documents", None)
        else:
            answer = raw_result.get("answer", "Không thể tạo câu trả lời.")
            source_documents = None


        # Lưu vào Redis
        save_chat_to_redis(r, session_id, question, answer)
        sources_list = []
        if raw_result.get("source_documents"):
            for doc in raw_result["source_documents"]:
                 sources_list.append(SourceDocument(
                     source=doc.metadata.get('source', 'N/A'),
                     page_content_preview=doc.page_content[:200] + "..."
                 ))
        return AnswerResponse(
            session_id=session_id,
            answer=answer,
            sources=sources_list if sources_list else None,
            processing_time=round(end_time - start_time, 2)
        )
    except Exception as e:
        print(f"Error during QA Chain invocation: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.get("/health", tags=["Status"])
async def health_check():
    if app_state.get("qa_chain"):
        return {"status": "OK", "message": "QA Chain is ready."}
    else:
        return {"status": "Initializing or Failed", "message": "QA Chain is not ready."}

@app.get("/chat-history/{session_id}", response_model=ChatHistoryResponse, tags=["Chatbot"])
def get_chat_history( session_id: str):
    history = get_redis_history(r, session_id)
    return ChatHistoryResponse(session_id=session_id, history=history)
# Import các thư viện cần thiết chỉ khi cần xử lý PDF
try:
    import fitz
    from PIL import Image
    import io
    import pytesseract
except ImportError:
    print("PyMuPDF, Pillow hoặc pytesseract không được cài đặt. Chức năng xử lý PDF sẽ không hoạt động.")
    fitz = None
    Image = None
    io = None
    pytesseract = None

async def update_vectorstore_bg(new_file_path: str):
    """[BG Task] Cập nhật Vector Store từ file mới (.txt hoặc .pdf)"""
    print(f"[BG Task] => Bắt đầu cập nhật với file: {new_file_path}", flush=True)

    if not os.path.exists(new_file_path):
        print(f"[BG Task] => File không tồn tại: {new_file_path}", flush=True)
        return

    try:
        # 1. Trích xuất nội dung từ file
        ext = os.path.splitext(new_file_path)[1].lower()
        content = ""

        if ext == '.pdf':
            content = utils.extract_text_from_pdf_auto(new_file_path, lang='vie')
        elif ext in ['.txt', '.md']:
            with open(new_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        else:
            print(f"[BG Task] => Định dạng file '{ext}' chưa được hỗ trợ.", flush=True)
            return

        if not content or not content.strip():
            print(f"[BG Task] => Không thể trích xuất nội dung từ file: {new_file_path}", flush=True)
            return

        print(f"[BG Task] => Trích xuất xong. Độ dài văn bản: {len(content)} ký tự", flush=True)
        # 2. Tạo Document và chunk theo cấu trúc
        print(f"[BG Task] => Bắt đầu chunking...", flush=True)
        doc = Document(
            page_content=content,
            metadata={"source": os.path.basename(new_file_path)}
        )

        doc_chunks = utils.split_by_structure(doc, max_chunk_size=config.CHUNK_SIZE * 2)

        print(f"[BG Task] => Chunking xong. Có {len(doc_chunks)} chunks.", flush=True)


        if not doc_chunks:
            print(f"[BG Task] => Không có chunk nào được tạo.", flush=True)
            return

        # 3. Nhúng và thêm vào Vectorstore
        embeddings = app_state.get("embeddings")
        vectorstore = app_state.get("vectorstore")

        if not embeddings or not vectorstore:
            print(f"=> Không tìm thấy embeddings hoặc vectorstore trong app_state.", flush=True)
            return

        print(f"[BG Task] => Đang thêm {len(doc_chunks)} chunks vào Vector Store...", flush=True)
        vectorstore.add_documents(doc_chunks)

        print(f"[BG Task] => Đã thêm vào vectorstore. Bắt đầu persist...", flush=True)

        # 4. Persist
        start_time = time.time()
        await asyncio.to_thread(vectorstore.persist)
        print(f"[BG Task] => Persist hoàn tất sau {time.time() - start_time:.2f}s.", flush=True)

        # 5. Cleanup
        del doc, doc_chunks, content
        gc.collect()
        print(f"[BG Task] => Đã thêm dữ liệu từ {os.path.basename(new_file_path)} vào Vector Store.", flush=True)

    except Exception as e:
        print(f"=> Lỗi khi cập nhật Vector Store: {e}", flush=True)

    print(f"=> Cập nhật Vector Store hoàn tất.\n", flush=True)



# Hàm này sẽ chạy ở background
@app.post("/upload", tags=["Data Management"])
async def upload_file(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Upload file văn bản luật (.txt hoặc .pdf) và xử lý background để cập nhật vectorstore.
    """
    if not file:
        raise HTTPException(status_code=400, detail="=> Không có file nào được tải lên.")

    filename = file.filename
    file_extension = os.path.splitext(filename)[1].lower()
    base_filename = os.path.splitext(filename)[0]
    output_filename = base_filename + ".txt"
    output_path = os.path.join(config.INPUT_TXT_FOLDER, output_filename)

    # Kiểm tra trùng lặp
    if any([
        os.path.exists(os.path.join(config.INPUT_TXT_FOLDER, filename)),
        os.path.exists(output_path)
    ]):
        return {"message": f"=> File '{filename}' hoặc bản .txt của nó đã tồn tại. Bỏ qua."}

    # Kiểm tra định dạng hợp lệ
    if file_extension not in [".txt", ".pdf"]:
        raise HTTPException(status_code=400, detail="=> Chỉ hỗ trợ file .txt và .pdf.")

    # Đảm bảo thư mục tồn tại
    os.makedirs(config.INPUT_TXT_FOLDER, exist_ok=True)

    try:
        contents = await file.read()
        text = ""

        # Xử lý theo định dạng
        if file_extension == ".txt":
            text = contents.decode("utf-8")

        elif file_extension == ".pdf":
            if not all([utils.fitz, utils.Image, utils.io, utils.pytesseract]):
                raise HTTPException(
                    status_code=500,
                    detail="=> Thiếu thư viện xử lý PDF: fitz, Pillow, pytesseract..."
                )

            # Lưu tạm file PDF
            temp_dir = "/tmp"
            os.makedirs(temp_dir, exist_ok=True)
            temp_pdf_path = os.path.join(temp_dir, filename)

            with open(temp_pdf_path, "wb") as f:
                f.write(contents)

            print(f"=> PDF tạm lưu tại: {temp_pdf_path}")
            text = utils.extract_text_from_scanned_pdf(temp_pdf_path)
            os.remove(temp_pdf_path)

            if not text or len(text.strip()) == 0:
                raise ValueError(f"=> Không thể trích xuất nội dung từ '{filename}'.")

        # Làm sạch văn bản
        cleaned_text = utils.clean_text_optimized(text)

        # Ghi nội dung vào file .txt
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(cleaned_text)

        print(f"=> File sạch đã lưu: {output_path}")

        # Thêm vào background task
        background_tasks.add_task(update_vectorstore_bg, new_file_path=output_path)

        return {"message": f"=> File '{filename}' đã được tải lên. Vector store đang được cập nhật nền."}

    except Exception as e:
        print(f"=> Lỗi khi xử lý file: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý file: {str(e)}")


if __name__ == "__main__":
    print(f"=> Chạy FastAPI server với Uvicorn...")
    uvicorn.run("api:app", host=config.API_HOST, port=config.API_PORT, reload=True)
