from langchain.schema import BaseRetriever, Document
from typing import List, Dict, Any, Optional
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun, AsyncCallbackManagerForRetrieverRun
from pydantic import Field
import logging
from datetime import datetime, date
import re # Import re để parse ngày tháng từ "ngày X tháng Y năm Z"

logger = logging.getLogger(__name__)

# Helper function để parse ngày tháng từ format "ngày X tháng Y năm Z"
def parse_vietnamese_date(date_str_vi: Optional[str]) -> Optional[date]:
    if not date_str_vi:
        return None
    # Chuẩn hóa khoảng trắng và chữ thường
    date_str_vi = date_str_vi.lower().strip()
    date_str_vi = re.sub(r'\s+', ' ', date_str_vi) # Thay nhiều khoảng trắng bằng một

    match = re.match(r"ngày (\d{1,2}) tháng (\d{1,2}) năm (\d{4})", date_str_vi)
    if match:
        try:
            day = int(match.group(1))
            month = int(match.group(2))
            year = int(match.group(3))
            return date(year, month, day)
        except ValueError:
            logger.warning(f"Could not parse Vietnamese date string after regex match: {date_str_vi}")
            return None
    else:
        # Thử thêm một số biến thể phổ biến khác nếu cần, ví dụ có dấu "/" hoặc "-"
        # Hoặc nếu ngày tháng năm có thể không đầy đủ
        logger.warning(f"Vietnamese date string did not match expected format 'ngày X tháng Y năm Z': {date_str_vi}")
        return None

class TimeSensitiveRetriever(BaseRetriever):
    """Retriever ưu tiên văn bản pháp luật mới nhất, sử dụng metadata từ extract_law_structure."""

    base_retriever: BaseRetriever = Field(description="The base retriever to fetch initial documents.")
    legal_document_hierarchy: Dict = Field(description="Hierarchy and recency info for legal documents.")
    top_k_after_sort: int = Field(default=5, description="Number of documents to return after sorting.")
    recency_weight: float = Field(default=0.6, description="Weight for recency score.")
    hierarchy_weight: float = Field(default=0.2, description="Weight for document type hierarchy score.")
    superseded_penalty_weight: float = Field(default=0.2, description="Weight for superseded document penalty.")


    def __init__(self, **data: Any):
        super().__init__(**data)
        self.tags: List[str] = ["time-sensitive", "legal-documents", "custom-retriever", "metadata-driven"]
        logger.debug(f"TimeSensitiveRetriever initialized. Base retriever type: {type(self.base_retriever)}")
        # logger.debug(f"Legal hierarchy config: {self.legal_document_hierarchy}") # In ra toàn bộ config
        logger.info(f"Retriever weights: Recency={self.recency_weight}, Hierarchy={self.hierarchy_weight}, SupersededPenalty={self.superseded_penalty_weight}")


    def _get_document_score(self, doc: Document) -> float:
        metadata = doc.metadata if hasattr(doc, 'metadata') else {}
        content_lower = doc.page_content.lower()

        # --- Lấy thông tin từ metadata sử dụng key từ extract_law_structure ---
        # Chuẩn hóa "so_hieu" để so sánh (ví dụ: bỏ dấu cách, viết thường)
        # QUAN TRỌNG: Quyết định một quy ước chuẩn hóa cho "so_hieu" và áp dụng nhất quán
        # giữa metadata và cấu hình latest_documents. Ở đây tôi dùng lowercase và bỏ khoảng trắng.
        meta_so_hieu_raw = str(metadata.get("so_hieu", "")).strip()
        meta_so_hieu_normalized = re.sub(r'\s+', '', meta_so_hieu_raw).lower()

        # "loai_van_ban" từ metadata (ví dụ: "Nghị Định")
        # So sánh với key trong hierarchy (ví dụ: "Nghị định")
        meta_loai_van_ban_raw = str(metadata.get("loai_van_ban", "")).strip()
        # Chuẩn hóa loai_van_ban từ metadata để khớp với key trong hierarchy (ví dụ: .title())
        # Nếu key trong hierarchy là "Nghị định", thì metadata cũng nên là "Nghị định"
        meta_loai_van_ban_for_hierarchy = meta_loai_van_ban_raw.title() # Hoặc cách chuẩn hóa khác bạn dùng

        meta_ngay_ban_hanh_str = str(metadata.get("ngay_ban_hanh", "")).strip()
        meta_issued_date = parse_vietnamese_date(meta_ngay_ban_hanh_str)

        current_date = date.today()

        # 1. ĐIỂM ƯU TIÊN THỨ BẬC (Hierarchy Score)
        hierarchy_score = 999
        if meta_loai_van_ban_for_hierarchy:
            # Key trong self.legal_document_hierarchy["hierarchy"] cần khớp với meta_loai_van_ban_for_hierarchy
            hierarchy_score = self.legal_document_hierarchy.get("hierarchy", {}).get(meta_loai_van_ban_for_hierarchy, 999)
        else:
            for type_key, type_val in self.legal_document_hierarchy.get("hierarchy", {}).items():
                if type_key.lower() in content_lower:
                    hierarchy_score = min(hierarchy_score, type_val)
                    break

        # 2. ĐIỂM TÍNH MỚI (Recency Score)
        recency_days_diff = 365 * 20 # Mặc định rất cũ
        if meta_issued_date:
            recency_days_diff = (current_date - meta_issued_date).days
            if recency_days_diff < 0:
                recency_days_diff = -1 # Văn bản tương lai

        # 3. ĐIỂM PHẠT CHO VĂN BẢN BỊ THAY THẾ (Superseded Penalty)
        superseded_penalty = 0
        is_explicitly_latest = False

        if meta_so_hieu_normalized: # Cần số hiệu để kiểm tra
            # QUAN TRỌNG: Key trong "latest_documents" phải được chuẩn hóa GIỐNG như meta_so_hieu_normalized
            # Ví dụ: nếu meta_so_hieu_normalized là "168/2024/nđ-cp" (lowercase, no space)
            # thì key trong latest_documents cũng phải là "168/2024/nđ-cp"
            latest_doc_config_key = meta_so_hieu_normalized
            latest_doc_info = self.legal_document_hierarchy.get("latest_documents", {}).get(latest_doc_config_key)

            if latest_doc_info:
                is_explicitly_latest = True
                # Kiểm tra lại ngày ban hành từ config nếu có, có thể dùng để fine-tune
                config_issued_date = parse_vietnamese_date(latest_doc_info.get("ngay_ban_hanh_config")) # Giả sử bạn thêm key này vào config
                if config_issued_date and meta_issued_date and config_issued_date == meta_issued_date:
                     recency_days_diff = min(recency_days_diff, -10) # Thưởng thêm
                else: # Nếu chỉ khớp số hiệu mà ngày khác (hoặc thiếu ngày trong config), thưởng ít hơn
                    recency_days_diff = min(recency_days_diff, -5)
                logger.debug(f"Document '{meta_so_hieu_raw}' is explicitly latest by config.")

            if not is_explicitly_latest:
                for latest_id_key_config, info_latest_config in self.legal_document_hierarchy.get("latest_documents", {}).items():
                    # `supersedes` trong config cũng phải chứa các số hiệu đã chuẩn hóa (lowercase, no space)
                    superseded_list_normalized = [re.sub(r'\s+', '', s).lower() for s in info_latest_config.get("supersedes", [])]
                    if meta_so_hieu_normalized in superseded_list_normalized:
                        superseded_penalty = 1000 # Phạt nặng
                        logger.debug(f"Document '{meta_so_hieu_raw}' is superseded by '{info_latest_config.get('title', latest_id_key_config)}'.")
                        break
        else:
            superseded_penalty = 50 # Phạt nhẹ vì thiếu số hiệu

        # --- TỔNG HỢP ĐIỂM ---
        normalized_recency_score = max(0, recency_days_diff / 30)
        if is_explicitly_latest and recency_days_diff <=0 : # Nếu là latest và ngày ban hành không quá xa trong quá khứ
             normalized_recency_score = -10

        total_score = (normalized_recency_score * self.recency_weight +
                       hierarchy_score * self.hierarchy_weight +
                       superseded_penalty * self.superseded_penalty_weight)

        logger.debug(
            f"Doc: {str(metadata.get('source', 'N/A'))[:30]} (Số hiệu: {meta_so_hieu_raw}, Loại: {meta_loai_van_ban_raw}, Ngày BH: {meta_ngay_ban_hanh_str}) | "
            f"Recency(norm): {normalized_recency_score:.2f} (raw_days: {recency_days_diff}), "
            f"Hierarchy: {hierarchy_score}, SupersededPenalty: {superseded_penalty} | "
            f"TOTAL SCORE: {total_score:.2f}"
        )
        return total_score

    # ... (các hàm sort_by_legal_relevance, _get_relevant_documents, _aget_relevant_documents giữ nguyên)
    # Chỉ cần đảm bảo rằng trong _get_relevant_documents và _aget_relevant_documents, khi log top doc, bạn log đúng metadata
    # Ví dụ: metadata.get('so_hieu', 'N/A')
    def sort_by_legal_relevance(self, docs: List[Document]) -> List[Document]:
        if not docs:
            return []
        return sorted(docs, key=self._get_document_score)

    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        logger.info(f"[TimeSensitiveRetriever] Getting relevant documents for query: '{query}'")
        initial_docs = self.base_retriever.get_relevant_documents(
            query, callbacks=run_manager.get_child() if run_manager else None
        )
        logger.info(f"[TimeSensitiveRetriever] Got {len(initial_docs)} docs from base_retriever.")

        if not initial_docs:
            logger.info("[TimeSensitiveRetriever] No documents from base_retriever. Returning empty list.")
            return []

        sorted_docs = self.sort_by_legal_relevance(initial_docs)
        if sorted_docs:
             # Lấy điểm của tài liệu đầu tiên để log (gọi lại _get_document_score)
             top_doc_score = self._get_document_score(sorted_docs[0])
             logger.info(f"[TimeSensitiveRetriever] Docs sorted. Top doc (after sort) metadata[so_hieu]: {sorted_docs[0].metadata.get('so_hieu', 'N/A')}, score: {top_doc_score:.2f}")
        else:
            logger.info("[TimeSensitiveRetriever] No documents after sorting (or initial_docs was empty).")

        final_docs = sorted_docs[:self.top_k_after_sort]
        logger.info(f"[TimeSensitiveRetriever] Returning {len(final_docs)} docs after slicing.")
        return final_docs

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: Optional[AsyncCallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        logger.info(f"[TimeSensitiveRetriever] Async getting relevant documents for query: '{query}'")
        initial_docs = await self.base_retriever.aget_relevant_documents(
            query, callbacks=run_manager.get_child() if run_manager else None
        )
        logger.info(f"[TimeSensitiveRetriever] Async got {len(initial_docs)} docs from base_retriever.")

        if not initial_docs:
            logger.info("[TimeSensitiveRetriever] Async no documents from base_retriever. Returning empty list.")
            return []

        sorted_docs = self.sort_by_legal_relevance(initial_docs)
        if sorted_docs:
            top_doc_score = self._get_document_score(sorted_docs[0])
            logger.info(f"[TimeSensitiveRetriever] Async docs sorted. Top doc (after sort) metadata[so_hieu]: {sorted_docs[0].metadata.get('so_hieu', 'N/A')}, score: {top_doc_score:.2f}")
        else:
            logger.info("[TimeSensitiveRetriever] Async no documents after sorting.")

        final_docs = sorted_docs[:self.top_k_after_sort]
        logger.info(f"[TimeSensitiveRetriever] Async returning {len(final_docs)} docs after slicing.")
        return final_docs

# Hàm tạo retriever (ví dụ tên)
def build_law_retriever(
    vectorstore,
    legal_hierarchy_config=None,
    top_k_val=5,
    base_retriever_k=15,
    base_retriever_fetch_k=30
):
    base_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": base_retriever_k,
            "fetch_k": base_retriever_fetch_k,
            "lambda_mult": 0.6
        }
    )

    # QUAN TRỌNG: Cấu hình này phải khớp với metadata của bạn
    default_legal_hierarchy = {
        "hierarchy": { # Key là giá trị metadata['loai_van_ban'] SAU KHI CHUẨN HÓA (ví dụ: .title())
            "Hiến Pháp": 1, "Bộ Luật": 2, "Luật": 3, "Nghị Quyết": 4, # .title() sẽ ra case này
            "Nghị Định": 5, "Quyết Định": 6, "Thông Tư": 7, "Công Văn": 8
        },
        "latest_documents": {
            # KEY là "so_hieu" đã được CHUẨN HÓA (ví dụ: lowercase, no space)
            "168/2024/nđ-cp": {
                "title": "Nghị định 168/2024/NĐ-CP...",
                "ngay_ban_hanh_config": "ngày 26 tháng 12 năm 2024", # Để tham khảo, có thể dùng để kiểm tra thêm
                "loai_van_ban_config": "Nghị định", # Để tham khảo
                "supersedes": ["100/2019/nđ-cp", "123/2021/nđ-cp"] # "so_hieu" đã chuẩn hóa
            },
            # Thêm các văn bản khác
        }
    }
    final_legal_hierarchy = legal_hierarchy_config if legal_hierarchy_config is not None else default_legal_hierarchy

    return TimeSensitiveRetriever(
        base_retriever=base_retriever,
        legal_document_hierarchy=final_legal_hierarchy,
        top_k_after_sort=top_k_val
    )