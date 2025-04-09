# build_vectorstore.py
import os
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time
import gc

# Import từ các file khác trong project
import config
import utils
import rag_components

def build_store():
    """Hàm chính để xây dựng hoặc cập nhật Vector Store."""

    print("--- Bắt đầu Quá trình Xây dựng Vector Store ---")

    # --- 1. Tải cấu hình ---
    print("--- Bước 1: Tải cấu hình ---")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Sử dụng thiết bị: {device}")

    # --- 2. Tải và Làm sạch Dữ liệu ---
    print("\n--- Bước 2: Tải và Làm sạch Dữ liệu ---")
    docs = utils.load_and_clean_documents(config.INPUT_TXT_FOLDER)
    if not docs:
        print("Không có tài liệu nào được tải. Dừng quá trình.")
        return

    # --- 3. Chunking ---
    print("\n--- Bước 3: Chunking ---")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Đã chia thành {len(chunks)} chunks.")
    if not chunks:
        print("Không có chunks nào được tạo. Dừng quá trình.")
        return
    del docs
    gc.collect()

    # --- 4. Embedding ---
    print("\n--- Bước 4: Embedding ---")
    embeddings = rag_components.get_huggingface_embeddings(config.EMBEDDING_MODEL_NAME, device)
    if not embeddings:
        print("Không thể khởi tạo model embedding. Dừng quá trình.")
        return

    # --- 5. Indexing (Tạo Vector Store) ---
    # Quan trọng: Truyền 'chunks' vào đây để tạo mới (nếu chưa có)
    print("\n--- Bước 5: Tạo/Cập nhật Vector Store (ChromaDB) ---")
    start_time = time.time()
    vectorstore = rag_components.create_or_load_chroma_vectorstore(
        embeddings=embeddings,
        persist_directory=config.CHROMA_PERSIST_DIR,
        collection_name=config.CHROMA_COLLECTION_NAME,
        chunks=chunks # << Truyền chunks vào đây
    )
    end_time = time.time()

    if vectorstore:
        print(f"Quá trình tạo/cập nhật Vector Store hoàn tất sau {end_time - start_time:.2f} giây.")
    else:
        print("Xảy ra lỗi trong quá trình tạo/cập nhật Vector Store.")

    del chunks
    del embeddings
    gc.collect()

if __name__ == "__main__":
    build_store()