# retrievers/advanced_retriever.py

import logging
from typing import List, Dict, Any, Optional
import datetime
import re
import weaviate
import weaviate.classes.query as wvc_query
from concurrent.futures import ThreadPoolExecutor

from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Import các hàm cần thiết
from utils.process_data import infer_field, infer_entity_type
from utils.synonym_map import rewrite_query_with_legal_synonyms # <-- Sử dụng hàm này
import prompt_templete # <-- Import file prompt

logger = logging.getLogger(__name__)

class AdvancedLawRetriever(BaseRetriever):
    client: weaviate.WeaviateClient
    collection_name: str
    llm: Any
    reranker: Any
    embeddings_model: Any

    default_k: int = 5
    initial_k: int = 15 # Lấy nhiều ứng viên ban đầu
    hybrid_search_alpha: float = 0.5
    doc_type_boost: float = 0.4

    class Config:
        arbitrary_types_allowed = True

    # === CÁC HÀM HELPER ===

    def _extract_searchable_keywords_with_llm(self, question: str) -> List[str]:
        """Sử dụng LLM để trích xuất các cụm từ khóa tìm kiếm hiệu quả."""
        keyword_extraction_prompt = ChatPromptTemplate.from_template(prompt_templete.KEYWORD_EXTRACTION_PROMPT)
        keyword_chain = keyword_extraction_prompt | self.llm | StrOutputParser() | (lambda text: [k.strip() for k in text.strip().split("\n") if k.strip()])
        try:
            keywords = keyword_chain.invoke({"question": question})
            # Luôn bao gồm cả câu hỏi gốc đã được viết lại làm một truy vấn để không mất ngữ cảnh
            return [question] + keywords
        except Exception as e:
            logger.error(f"Failed to extract keywords: {e}")
            return [question]

    def _extract_and_build_filters(self, filters_dict: Dict[str, Any]) -> Optional[wvc_query.Filter]:
        """
        CẢI TIẾN: Hàm này CHỈ nhận một dict và xây dựng đối tượng Filter.
        Nó không còn nhiệm vụ suy luận nữa.
        """
        if not filters_dict:
            return None

        filter_conditions = []
        for key, value in filters_dict.items():
            if value is None:
                continue

            # Logic xây dựng Filter
            if key == "entity_type" and isinstance(value, list) and value:
                filter_conditions.append(wvc_query.Filter.by_property(key).contains_any(value))
            elif isinstance(value, str):
                filter_conditions.append(wvc_query.Filter.by_property(key).equal(value))
            # Thêm các điều kiện khác nếu cần

        if not filter_conditions:
            return None

        return wvc_query.Filter.all_of(filter_conditions) if len(filter_conditions) > 1 else filter_conditions[0]


    def _perform_hybrid_search(self, query: str, k: int, where_filter: Optional[wvc_query.Filter]) -> List[Document]:
        # ... (giữ nguyên logic) ...
        try:
            collection = self.client.collections.get(self.collection_name)
            query_vector = self.embeddings_model.embed_query(query)
            response = collection.query.hybrid(query=query, vector=query_vector, limit=k, alpha=self.hybrid_search_alpha, filters=where_filter, return_metadata=wvc_query.MetadataQuery(score=True))
            docs = [Document(page_content=obj.properties.pop('text', ''), metadata={**obj.properties, 'hybrid_score': obj.metadata.score if obj.metadata else 0}) for obj in response.objects]
            return docs
        except Exception: return []



    # === HÀM CHÍNH _get_relevant_documents ===
    def _get_relevant_documents(
    self, query: str, *, run_manager: CallbackManagerForRetrieverRun
) -> List[Document]:

        # =================================================================
        # PHASE 0: PREPARATION - Chuẩn bị và làm giàu truy vấn
        # =================================================================

        # 0.1. Đảm bảo an toàn cho input
        safe_query = str(query)
        logger.info(f"--- Starting Advanced Retrieval (FINAL) for Original Query: '{safe_query}' ---")

        # 0.2. Trích xuất thông tin và ý định từ câu hỏi gốc
        query_info = self._extract_query_info_with_intent(safe_query)
        inferred_field = query_info.get("base_filters", {}).get("field")
        preferred_doc_type = query_info.get("preferred_doc_type")

        # 0.3. "Dịch" câu hỏi sang ngôn ngữ pháp lý bằng từ điển
        rewritten_query = rewrite_query_with_legal_synonyms(safe_query, field=inferred_field)
        if safe_query != rewritten_query:
            logger.info(f"Query after Synonym Rewriting: '{rewritten_query}'")

        # 0.4. Trích xuất từ khóa "vàng" bằng LLM từ câu hỏi đã được viết lại
        search_terms = self._extract_searchable_keywords_with_llm(rewritten_query)
        logger.info(f"Extracted {len(search_terms)} searchable terms: {search_terms}")

        # 0.5. Xây dựng bộ lọc Weaviate từ thông tin đã trích xuất
        base_weaviate_filter = self._extract_and_build_filters(query_info["base_filters"])

        # =================================================================
        # PHASE 1: RETRIEVAL - Truy xuất dữ liệu có fallback
        # =================================================================

        def run_search_tasks(filters: Optional[wvc_query.Filter]) -> List[Document]:
            """Hàm nội bộ để thực hiện tìm kiếm song song."""
            docs = []
            with ThreadPoolExecutor(max_workers=len(search_terms) or 1) as executor:
                futures = [executor.submit(self._perform_hybrid_search, term, self.initial_k, filters) for term in search_terms]
                for future in futures:
                    try: docs.extend(future.result())
                    except Exception as e: logger.error(f"A search task failed: {e}")
            return docs

        logger.info(f"--- Attempt 1: Searching with inferred filters: {base_weaviate_filter} ---")
        retrieved_docs = run_search_tasks(base_weaviate_filter)

        # Lọc trùng lặp
        unique_docs_dict = {doc.page_content: doc for doc in retrieved_docs if isinstance(doc.page_content, str)}

        # Cơ chế Fallback
        if len(unique_docs_dict) < self.default_k and base_weaviate_filter is not None:
            logger.warning("Initial search yielded few results. Retrying without any filters (fallback)...")
            fallback_docs = run_search_tasks(None)
            for doc in fallback_docs:
                if isinstance(doc.page_content, str) and doc.page_content not in unique_docs_dict:
                    unique_docs_dict[doc.page_content] = doc

        candidate_docs_list = list(unique_docs_dict.values())

        # =================================================================
        # PHASE 2: REFINEMENT - Tinh chỉnh, ưu tiên và xếp hạng kết quả
        # =================================================================

        # 2.1. Intent-based Boosting: Tăng điểm dựa trên loại văn bản ưu tiên
        final_candidates_for_rerank = candidate_docs_list
        if preferred_doc_type:
            logger.info(f"Applying INTENT-BASED BOOST for preferred type: '{preferred_doc_type}'")
            docs_with_scores = []
            for doc in candidate_docs_list:
                score = doc.metadata.get('hybrid_score', 0.5)
                if doc.metadata.get("loai_van_ban") == preferred_doc_type:
                    score += self.doc_type_boost
                else:
                    score -= 0.05 # Giảm nhẹ điểm của các loại không ưu tiên
                docs_with_scores.append((doc, score))

            docs_with_scores.sort(key=lambda x: x[1], reverse=True)
            final_candidates_for_rerank = [doc for doc, score in docs_with_scores]

        logger.info(f"Found {len(final_candidates_for_rerank)} candidates for re-ranking.")
        if not final_candidates_for_rerank: return []

        # 2.2. Cross-Encoder Re-ranking với Structured Context
        logger.info("Applying Cross-Encoder re-ranking with STRUCTURED CONTEXT...")

        docs_for_reranking = []
        for doc in final_candidates_for_rerank:
            # Tạo chuỗi context giàu thông tin
            structured_content = (
                f"Loại văn bản: {doc.metadata.get('loai_van_ban', 'N/A')}. "
                f"Lĩnh vực: {doc.metadata.get('field', 'N/A')}. "
                f"Đối tượng: {doc.metadata.get('entity_type', 'N/A')}.\n"
                f"Nội dung trích từ {doc.metadata.get('title', 'N/A')}: {doc.page_content}"
            )
            docs_for_reranking.append({"original_doc": doc, "structured_content": structured_content})

        contents_to_rank = [item["structured_content"] for item in docs_for_reranking]

        try:
            # Sử dụng câu hỏi đã được viết lại để có ngữ cảnh tốt nhất
            ranked_results_info = self.reranker.rank(rewritten_query, contents_to_rank, return_documents=False, top_k=self.default_k * 2) # Lấy nhiều hơn một chút
        except Exception as e:
            logger.error(f"Failed to re-rank with custom structured content: {e}. Falling back to default re-ranking.")
            # Fallback về cách re-rank mặc định nếu có lỗi
            reranked_docs = self.reranker.compress_documents(final_candidates_for_rerank, rewritten_query)
            return reranked_docs[:self.default_k]

        # Lấy lại các Document gốc theo thứ tự đã được re-rank
        final_reranked_docs = []
        for rank_info in ranked_results_info:
            original_doc = docs_for_reranking[rank_info['corpus_id']]["original_doc"]
            original_doc.metadata['rerank_score'] = rank_info['score']
            final_reranked_docs.append(original_doc)

        # 2.3. Log và Trả về kết quả cuối cùng
        logger.info(f"--- Re-ranked down to {len(final_reranked_docs)} documents. Final results: ---")
        for i, doc in enumerate(final_reranked_docs[:self.default_k]):
            score_str = f"{doc.metadata.get('rerank_score', 0.0):.4f}"
            logger.info(f"  - RANK #{i+1} | ReRank Score: {score_str} | Source: {doc.metadata.get('source')}")
            logger.info(f"    CONTENT: {doc.page_content[:400]}...") # Log dài hơn
            logger.info("-" * 25)

        return final_reranked_docs[:self.default_k]

    def _extract_query_info_with_intent(self, query: str) -> Dict[str, Any]:
        """
        Trích xuất filter và xác định ý định của câu hỏi để ưu tiên loại văn bản.
        """
        info = {"base_filters": {}, "preferred_doc_type": None}
        query_lower = query.lower()

        # 1. Suy luận field và entity
        inferred_field = infer_field(query, None)
        if inferred_field and inferred_field != "khac":
            info["base_filters"]["field"] = inferred_field

        inferred_entities = infer_entity_type(query, inferred_field)
        if inferred_entities:
            info["base_filters"]["entity_type"] = inferred_entities

        # 2. XÁC ĐỊNH Ý ĐỊNH -> ƯU TIÊN LOẠI VĂN BẢN
        # Nếu câu hỏi về MỨC PHẠT, ưu tiên tuyệt đối NGHỊ ĐỊNH
        if any(kw in query_lower for kw in ["phạt bao nhiêu", "mức xử phạt", "tiền phạt", "xử phạt"]):
            info["preferred_doc_type"] = "NGHỊ ĐỊNH"
            logger.info("Intent detected: Sanction/Penalty -> Preferring 'NGHỊ ĐỊNH'.")
        # Nếu câu hỏi về NGUYÊN TẮC CHUNG, QUYỀN, NGHĨA VỤ, ưu tiên LUẬT
        elif any(kw in query_lower for kw in ["nguyên tắc", "quyền và nghĩa vụ", "cấm", "được phép", "khái niệm", "định nghĩa"]):
             info["preferred_doc_type"] = "LUẬT"
             logger.info("Intent detected: General Rule/Definition -> Preferring 'LUẬT'.")

        logger.info(f"Extracted query info: {info}")
        return info
