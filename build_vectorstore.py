# build_v5.py
import time
import gc
import os
import logging
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import các hàm cần thiết
import config
from db.weaviateDB import connect_to_weaviate
from rag_components import (
    get_huggingface_embeddings,
    create_weaviate_schema_if_not_exists,
    ingest_chunks_with_native_batching,
    filter_and_serialize_complex_metadata
)
from utils.process_data import process_single_file

logger = logging.getLogger(__name__)

# --- CÁC HÀM HỖ TRỢ CHO CHECKPOINTING ---
def load_processed_files() -> set:
    """Đọc file checkpoint và trả về một set các tên file đã xử lý."""
    if not os.path.exists(config.CHECKPOINT_FILE):
        return set()
    with open(config.CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
        return {line.strip() for line in f if line.strip()}

def log_processed_file(filename: str):
    """Ghi tên file đã xử lý thành công vào file checkpoint."""
    with open(config.CHECKPOINT_FILE, 'a', encoding='utf-8') as f:
        f.write(filename + '\n')

def clear_weaviate_collection(client, collection_name: str):
    """Hàm helper để xóa collection và file checkpoint."""
    try:
        if client.collections.exists(collection_name):
            logger.warning(f"🗑️ Đang xóa collection '{collection_name}'...")
            client.collections.delete(collection_name)
            logger.info(f"✅ Đã xóa collection '{collection_name}' thành công.")

        # Xóa cả file checkpoint khi rebuild
        if os.path.exists(config.CHECKPOINT_FILE):
            os.remove(config.CHECKPOINT_FILE)
            logger.info(f"🗑️ Đã xóa file checkpoint '{config.CHECKPOINT_FILE}'.")

    except Exception as e:
        logger.error(f"❌ Lỗi khi xóa collection: {e}")
        raise e

def build_store_v5(force_rebuild: bool = False, pool_batch_size: int = 50):
    """
    Hàm xây dựng Vector Store với Checkpointing và Pool Restart để xử lý các job dài hơi.

    Args:
        force_rebuild (bool): Xóa dữ liệu cũ và bắt đầu lại từ đầu.
        pool_batch_size (int): Số lượng file xử lý trước khi tái khởi động worker pool.
    """
    logger.info(f"🚀 Bắt đầu Quá trình Xây dựng Vector Store v5 (Checkpointing & Pool Restart) 🚀")
    total_start_time = time.time()
    total_chunks_indexed = 0
    weaviate_client = None

    try:
        # --- 1. SETUP PHASE ---
        logger.info("⚙️  1. Thiết lập môi trường...")
        device = 'cpu' # Giả định không có GPU
        logger.info(f"💻 Sử dụng thiết bị: {device}")

        weaviate_client = connect_to_weaviate(run_diagnostics=False)
        if not weaviate_client: raise ConnectionError("Không thể kết nối đến Weaviate.")

        logger.info("🧠 Tải model embedding (chạy trên CPU)...")
        embeddings_model = get_huggingface_embeddings(config.EMBEDDING_MODEL_NAME, device)
        if not embeddings_model: raise RuntimeError("Không thể khởi tạo model embedding.")

        collection_name = config.WEAVIATE_COLLECTION_NAME

        if force_rebuild:
            clear_weaviate_collection(weaviate_client, collection_name)

        create_weaviate_schema_if_not_exists(weaviate_client, collection_name)

        # --- 2. LỌC FILE VÀ CHUẨN BỊ BATCH ---
        processed_files = load_processed_files()
        all_txt_paths = [os.path.join(config.CORE_DATA_FOLDER, f) for f in os.listdir(config.CORE_DATA_FOLDER) if f.lower().endswith('.txt')]

        # Lọc ra những file chưa được xử lý
        files_to_process = [path for path in all_txt_paths if os.path.basename(path) not in processed_files]

        if not files_to_process:
            logger.info("✅ Tất cả các file đã được xử lý. Không có gì để làm.")
            return

        logger.info(f"🔍 Đã xử lý {len(processed_files)} files. Còn lại {len(files_to_process)} files cần xử lý.")

        # --- 3. XỬ LÝ THEO LÔ ĐỂ CHỐNG RÒ RỈ BỘ NHỚ ---
        max_workers = os.cpu_count() or 1

        # Tạo thanh tiến trình tổng
        main_progress_bar = tqdm(total=len(files_to_process), desc="Tổng tiến trình")

        # Chia danh sách file thành các lô nhỏ
        for i in range(0, len(files_to_process), pool_batch_size):
            file_batch = files_to_process[i:i + pool_batch_size]
            logger.info(f"\n🔄 Đang xử lý lô {i//pool_batch_size + 1}, gồm {len(file_batch)} files. Tái khởi động Worker Pool...")

            # Tạo một Worker Pool MỚI cho mỗi lô
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_file = {executor.submit(process_single_file, path): path for path in file_batch}

                for future in as_completed(future_to_file):
                    path = future_to_file[future]
                    filename = os.path.basename(path)
                    try:
                        chunks_from_file = future.result()
                        if not chunks_from_file:
                            logger.warning(f"File '{filename}' không có chunks.")
                            log_processed_file(filename) # Vẫn ghi nhận là đã xử lý
                            main_progress_bar.update(1)
                            continue

                        processed_chunks = filter_and_serialize_complex_metadata(chunks_from_file)

                        ingest_chunks_with_native_batching(
                            client=weaviate_client,
                            collection_name=collection_name,
                            chunks=processed_chunks,
                            embeddings_model=embeddings_model
                        )

                        total_chunks_indexed += len(chunks_from_file)
                        log_processed_file(filename) # Ghi nhận thành công
                        main_progress_bar.set_description(f"✅ Ingested '{filename}'")
                        main_progress_bar.update(1) # Cập nhật thanh tiến trình tổng

                        del chunks_from_file, processed_chunks
                        gc.collect()

                    except Exception as e:
                        logger.error(f"❌ Lỗi khi xử lý file '{filename}': {e}", exc_info=True)

        main_progress_bar.close()
        logger.info(f"\n✅ Đã ingest thành công thêm {total_chunks_indexed} chunks.")

    except (ConnectionError, RuntimeError, Exception) as e:
        logger.error(f"💥 Đã xảy ra lỗi nghiêm trọng: {e}", exc_info=True)

    finally:
        # --- 4. DỌN DẸP ---
        logger.info("\n⚙️  4. Dọn dẹp tài nguyên...")
        if 'embeddings_model' in locals(): del embeddings_model
        if weaviate_client and weaviate_client.is_connected():
            weaviate_client.close()
            logger.info("🔌 Đã đóng kết nối Weaviate.")
        gc.collect()
        total_end_time = time.time()
        logger.info(f"⏱️  Tổng thời gian chạy lần này: {total_end_time - total_start_time:.2f} giây")
        logger.info("🎉 Chương trình kết thúc! 🎉")

if __name__ == "__main__":
    # Khi chạy lại, chỉ cần chạy lệnh này. Nó sẽ tự động tiếp tục.
    # Đặt force_rebuild=True chỉ khi bạn muốn xóa sạch và làm lại từ đầu.
    build_store_v5(force_rebuild=True)