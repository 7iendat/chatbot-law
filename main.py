# main.py
import os
import torch
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time
import gc # Garbage collector

# Import từ các file khác trong project
import config
import utils
import rag_components

def main():
    """Hàm chính để chạy chatbot trên dòng lệnh."""

    # --- 1. Tải cấu hình và môi trường ---
    print("--- Bước 1: Tải cấu hình và môi trường ---")
    load_dotenv() # Tải biến môi trường từ file .env (quan trọng cho API key)
    hf_api_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if not hf_api_token:
        print("Cảnh báo: HUGGINGFACEHUB_API_TOKEN không tìm thấy trong môi trường. LLM Endpoint có thể không hoạt động.")
        # Bạn có thể yêu cầu nhập ở đây nếu muốn, nhưng tốt hơn là đặt trong .env

    # Xác định thiết bị
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Sử dụng thiết bị: {device}")

    # --- 2. Tải và Làm sạch Dữ liệu ---
    print("\n--- Bước 2: Tải và Làm sạch Dữ liệu ---")
    # Hàm này đọc từ config.INPUT_TXT_FOLDER và dùng utils.clean_text_optimized
    docs = utils.load_and_clean_documents(config.INPUT_TXT_FOLDER)
    if not docs:
        print("Không có tài liệu nào được tải. Thoát.")
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
        print("Không có chunks nào được tạo. Thoát.")
        return
    del docs # Giải phóng bộ nhớ của docs gốc
    gc.collect()


    # --- 4. Embedding ---
    print("\n--- Bước 4: Embedding ---")
    embeddings = rag_components.get_huggingface_embeddings(config.EMBEDDING_MODEL_NAME, device)
    if not embeddings:
        print("Không thể khởi tạo model embedding. Thoát.")
        return

    # --- 5. Indexing (Vector Store) ---
    print("\n--- Bước 5: Indexing (Vector Store) ---")
    vectorstore = rag_components.create_or_load_faiss_vectorstore(
        chunks,
        embeddings,
        config.FAISS_INDEX_FOLDER,
        config.FAISS_INDEX_NAME
    )
    if not vectorstore:
        print("Không thể tạo hoặc tải Vector Store. Thoát.")
        return
    del chunks # Giải phóng bộ nhớ chunks
    gc.collect()

    # --- 6. Khởi tạo LLM ---
    print("\n--- Bước 6: Khởi tạo LLM ---")
    llm = rag_components.get_huggingface_endpoint_llm(
        config.LLM_REPO_ID,
        hf_api_token,
        config.LLM_TEMPERATURE,
        config.LLM_MAX_NEW_TOKENS
    )
    if not llm:
        print("Không thể khởi tạo LLM. Thoát.")
        return

    # --- 7. Tạo QA Chain ---
    print("\n--- Bước 7: Tạo QA Chain ---")
    qa_chain = rag_components.create_qa_chain(
        llm,
        vectorstore,
        config.SEARCH_K,
        config.PROMPT_TEMPLATE # Sử dụng template từ config
    )
    if not qa_chain:
        print("Không thể tạo QA Chain. Thoát.")
        return

    # --- 8. Vòng lặp Hỏi Đáp ---
    print("\n--- Bước 8: Chatbot đã sẵn sàng! ---")
    print("Nhập câu hỏi của bạn (hoặc gõ 'quit' để thoát)")
    while True:
        try:
            question = input("> ")
            if question.lower() == 'quit':
                break
            if not question.strip():
                continue

            start_time = time.time()
            result = qa_chain.invoke({"query": question})
            end_time = time.time()

            print("\nTrả lời:")
            print(result.get("result", "Xin lỗi, tôi không thể tìm thấy câu trả lời."))
            print("\nNguồn tham khảo:")
            if result.get("source_documents"):
                 unique_sources = set(doc.metadata.get('source', 'N/A') for doc in result["source_documents"])
                 for src in sorted(list(unique_sources)):
                      print(f"- {src}")
            else:
                 print("- Không có nguồn cụ thể.")
            print(f"(Thời gian: {end_time - start_time:.2f} giây)\n")

        except KeyboardInterrupt:
            print("\nĐã nhận tín hiệu thoát. Tạm biệt!")
            break
        except Exception as e:
            print(f"\nĐã xảy ra lỗi: {e}")
            # import traceback # Bỏ comment để debug lỗi chi tiết
            # traceback.print_exc()

if __name__ == "__main__":
    main()