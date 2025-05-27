import torch
import time
import gc
from tqdm import tqdm
import config
import utils.utils as utils
import rag_components
import logging
import weaviate
from db.weaviateDB import connect_to_weaviate

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clear_weaviate_collection( collection_name):
    """Xóa collection cũ trong Weaviate nếu tồn tại."""
    try:
        # Kết nối đến Weaviate
        client = connect_to_weaviate()

        try:
            collection = client.collections.get(collection_name)
            # Nếu không lỗi, tức là collection tồn tại
            logger.info(f"🔸Collection '{collection_name}' đã tồn tại. Đang xóa...")
            client.collections.delete(collection_name)
            logger.info(f"🔸Đã xóa collection '{collection_name}' thành công.")
        except weaviate.exceptions.WeaviateQueryError:
            # Collection không tồn tại
            logger.info(f"🔸Collection '{collection_name}' không tồn tại. Không cần xóa.")

        client.close()
        return True

    except Exception as e:
        logger.error(f"🔸Lỗi khi xóa collection: {str(e)}")
        return False

def build_store():
    """Hàm chính để xây dựng hoặc cập nhật Vector Store."""

    logger.info("🔸Bắt đầu Quá trình Xây dựng Vector Store")

    # --- 1. Tải cấu hình ---
    logger.info("🔸Tải cấu hình")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"🔸Sử dụng thiết bị: {device}")

    # --- 1.5. Xóa dữ liệu cũ ---
    logger.info("🔸Xóa dữ liệu cũ trong Weaviate")
    if not clear_weaviate_collection(config.WEAVIATE_COLLECTION_NAME):
        logger.warning("🔸Không thể xóa collection cũ, nhưng sẽ tiếp tục quá trình.")

    # --- 2. Tải và Làm sạch Dữ liệu ---
    logger.info("🔸Tải và Làm sạch Dữ liệu")
    docs = utils.load_and_clean_documents(config.INPUT_TXT_FOLDER)
    if not docs:
        logger.error("🔸Không có tài liệu nào được tải. Dừng quá trình.")
        return
    logger.info(f"🔸Đã tải {len(docs)} tài liệu từ thư mục {config.INPUT_TXT_FOLDER}")

    # --- 3. Chunking theo Cấu trúc ---
    logger.info("🔸Chunking theo cấu trúc (Điều/Khoản)")
    chunks = []
    if not docs:
        logger.error("🔸Không có tài liệu để chunking.")
    else:

        # Lặp qua từng Document (từng file luật)
        for doc in tqdm(docs, desc="Chunking tài liệu"):
            doc_chunks = utils.split_by_law_structure(doc, max_chunk_size=config.CHUNK_SIZE*1.5) # Cho phép chunk lớn hơn một chút khi chia theo điều
            logger.info(f"🔸Tài liệu: {doc.metadata.get('source', 'Không rõ')} => {len(doc_chunks)} chunks")
            chunks.extend(doc_chunks)

        logger.info(f"🔸Đã chia thành {len(chunks)} chunks theo cấu trúc.")
        if not chunks:
            logger.error("🔸Không có chunks nào được tạo. Dừng quá trình.")
            return

        # Xem thử chunk đầu tiên
        logger.info("🔸Chunk đầu tiên (ví dụ):")
        logger.info(f"🔸Metadata: {chunks[0].metadata}")
        logger.info(chunks[0].page_content[:500] + "...")
    del docs
    gc.collect()

    # --- 4. Embedding ---
    logger.info("🔸Embedding")
    embeddings = rag_components.get_huggingface_embeddings(config.EMBEDDING_MODEL_NAME, device)
    if not embeddings:
        logger.error("🔸Không thể khởi tạo model embedding. Dừng quá trình.")
        return

    # --- 5. Indexing (Tạo Vector Store) ---
    # Quan trọng: Truyền 'chunks' vào đây để tạo mới (nếu chưa có)
    logger.info("🔸Tạo/Cập nhật Vector Store ")
    start_time = time.time()
    # vectorstore = rag_components.create_or_load_chroma_vectorstore(
    #     embeddings=embeddings,
    #     persist_directory=config.CHROMA_PERSIST_DIR,
    #     collection_name=config.CHROMA_COLLECTION_NAME,
    #     chunks=chunks # << Truyền chunks vào đây
    # )

    weaviate_client = connect_to_weaviate()
    vectorstore = rag_components.create_or_load_vectorstore(
        embeddings=embeddings,
        weaviate_url=config.WEAVIATE_URL,
        collection_name=config.WEAVIATE_COLLECTION_NAME,
        weaviate_client=weaviate_client,
        chunks=chunks
    )
    end_time = time.time()
    if vectorstore:
        logger.info("🔸Vector Store đã được tạo/cập nhật thành công.")
    else:
        logger.error("🔸Xảy ra lỗi trong quá trình tạo/cập nhật Vector Store.")

    logger.info(f"🔸Thời gian tạo Vector Store: {end_time - start_time:.2f} giây")

    del chunks
    del embeddings
    gc.collect()

if __name__ == "__main__":
    build_store()