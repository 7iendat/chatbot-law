# services/reranker_service.py
import logging
from functools import lru_cache
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

logger = logging.getLogger(__name__)

# Đặt tên model vào một hằng số để dễ quản lý
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

@lru_cache(maxsize=1)
def get_reranker_compressor(top_n: int = 4):
    """
    Tải và trả về một đối tượng CrossEncoderReranker.
    Sử dụng lru_cache để đảm bảo model chỉ được tải một lần duy nhất.
    """
    logger.info(f"🧠 Loading Re-ranker model '{RERANKER_MODEL_NAME}'...")
    try:
        # Tải model cross-encoder
        model = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL_NAME)

        # Tạo đối tượng compressor
        compressor = CrossEncoderReranker(model=model, top_n=top_n)

        logger.info("✅ Re-ranker model is ready.")
        return compressor
    except Exception as e:
        logger.error(f"❌ Could not load Re-ranker model: {e}", exc_info=True)
        raise