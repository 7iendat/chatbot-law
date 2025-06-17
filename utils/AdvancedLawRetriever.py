from langchain_core.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from typing import List, Dict, Any, Optional, Set, Tuple
import datetime
import re
import logging
from utils.synonym_map import SYNONYM_MAP,get_synonyms_for_term
import random
# import weaviate # Nếu bạn dùng Weaviate client trực tiếp ở đâu đó

logger = logging.getLogger(__name__)

def generate_expanded_queries(original_query: str, field: Optional[str] = None, max_expansions: int = 3) -> List[str]:
    """
    Tạo ra các câu hỏi mở rộng dựa trên từ đồng nghĩa của các cụm từ chính trong câu hỏi gốc.
    """
    original_query_lower = original_query.lower()
    expanded_queries_set: Set[str] = {original_query} # Luôn bao gồm query gốc

    # Xác định các cụm từ khóa tiềm năng trong query gốc để thay thế
    # Đây là một heuristic đơn giản, có thể cần NLP phức tạp hơn (ví dụ: NER, Noun Phrase Chunker)
    # để xác định chính xác các "named entities" hoặc "key phrases".
    # Tạm thời, chúng ta sẽ thử với các key có trong SYNONYM_MAP.

    key_phrases_to_replace: List[Tuple[str, List[str]]] = [] # (phrase_in_query, list_of_synonyms)

    # Tìm các key phrase từ SYNONYM_MAP có trong query
    all_synonym_keys = []
    if field and field in SYNONYM_MAP:
        all_synonym_keys.extend(SYNONYM_MAP[field].keys())
    all_synonym_keys.extend(SYNONYM_MAP.get("general", {}).keys())

    # Sắp xếp các key theo độ dài giảm dần để ưu tiên khớp cụm dài hơn trước
    sorted_phrase_keys = sorted(list(set(all_synonym_keys)), key=len, reverse=True)

    temp_query_for_finding_phrases = original_query_lower
    found_phrases_spans = []

    for phrase_key in sorted_phrase_keys:
        # Tìm tất cả các vị trí của phrase_key trong query
        for match in re.finditer(re.escape(phrase_key), temp_query_for_finding_phrases):
            start, end = match.span()
            # Kiểm tra xem span này có chồng lấn với span đã tìm thấy không
            is_overlapping = any(max(start, s_start) < min(end, s_end) for s_start, s_end in found_phrases_spans)
            if not is_overlapping:
                syns = get_synonyms_for_term(phrase_key, field)
                if syns:
                    key_phrases_to_replace.append((phrase_key, syns))
                    found_phrases_spans.append((start,end))
                    # "Đánh dấu" phần đã tìm thấy để không tìm lại phrase con bên trong
                    # Điều này hơi khó với replace đơn giản, cách tốt hơn là dùng tokenization
                    # Tạm thời, chúng ta chấp nhận có thể có nhiều phrase được tìm thấy

    if not key_phrases_to_replace:
        return list(expanded_queries_set)

     # =========================================================================
    # BƯỚC 2: Mở rộng dựa trên các từ khóa pháp lý chiến lược
    # =========================================================================
    strategic_keywords = [
        "theo quy định mới nhất",
        "hiện hành",
        "sửa đổi bổ sung",
        "được cập nhật"
    ]

    # Chỉ áp dụng các từ khóa này nếu câu hỏi không chứa từ chỉ năm cụ thể
    if not re.search(r"\b(năm\s+)?(19\d{2}|20\d{2})\b", original_query_lower):
        # Tạo phiên bản mới bằng cách thêm từ khóa vào cuối
        for keyword in strategic_keywords:
            new_query = f"{original_query} {keyword}"
            expanded_queries_set.add(new_query)

        # Nếu đã xác định được lĩnh vực, tạo câu hỏi cụ thể hơn
        # Ví dụ: "mức phạt vượt đèn đỏ theo quy định giao thông mới nhất"
        if field and field != "khac":
            # Chuyển field 'giao_thong' -> 'giao thông'
            field_display = field.replace('_', ' ')
            for keyword in strategic_keywords:
                 new_query = f"{original_query} trong lĩnh vực {field_display} {keyword}"
                 expanded_queries_set.add(new_query)

    logger.info(f"Tổng số query sau 2 bước mở rộng: {len(expanded_queries_set)}")

    # =========================================================================
    # BƯỚC 3: Chọn lọc và trả về kết quả cuối cùng
    # =========================================================================
    final_expanded_list = list(expanded_queries_set)

    # Nếu số lượng vượt quá giới hạn, ưu tiên các câu hỏi đã được mở rộng chiến lược
    if len(final_expanded_list) > max_expansions + 1: # +1 cho query gốc
        strategic_queries = [q for q in final_expanded_list if any(kw in q.lower() for kw in strategic_keywords)]
        synonym_queries = [q for q in final_expanded_list if q not in strategic_queries and q != original_query]

        # Ưu tiên giữ lại: query gốc, các query chiến lược, sau đó là các query từ đồng nghĩa (nếu còn chỗ)
        final_list = [original_query]

        # Thêm các query chiến lược, ưu tiên các câu ngắn hơn
        strategic_queries.sort(key=len)
        remaining_slots = max_expansions - len(final_list) + 1
        final_list.extend(strategic_queries[:remaining_slots])

        # Nếu vẫn còn chỗ, thêm các query từ đồng nghĩa
        if len(final_list) < max_expansions + 1:
            random.shuffle(synonym_queries)
            remaining_slots = max_expansions - len(final_list) + 1
            final_list.extend(synonym_queries[:remaining_slots])

        final_expanded_list = list(dict.fromkeys(final_list)) # Loại bỏ trùng lặp và giữ thứ tự

    logger.info(f"Query: '{original_query}' -> Final Expanded to ({len(final_expanded_list)}): {final_expanded_list}")
    return final_expanded_list


class AdvancedLawRetriever(BaseRetriever):
    vectorstore: VectorStore
    default_k: int = 5
    initial_fetch_k_multiplier: int = 3
    recency_bias_weight: float = 0.65
    year_filter_margin: int = 0
    # initial_score_threshold: Optional[float] = 0.35 # Sẽ lọc sau
    final_score_threshold_after_bias: Optional[float] = 0.45 # Ngưỡng cho điểm cuối cùng
    enable_query_expansion: bool = True
    max_expanded_queries: int = 3 # Số query mở rộng (không tính gốc)

    class Config:
        arbitrary_types_allowed = True

    def _parse_year_from_query(self, query: str) -> Optional[int]:
        match = re.search(r"\b(?:năm\s+)?(19\d{2}|20\d{2})\b", query, re.IGNORECASE)
        return int(match.group(1) if len(match.groups())==1 and match.group(1) else match.group(2)) if match else None


    def _extract_subject_from_query(self, query: str) -> Optional[str]:
        """Trích xuất đối tượng áp dụng chính từ câu hỏi."""
        query_lower = query.lower()
        # Ưu tiên các cụm từ dài hơn và chính xác hơn
        if "xe máy" in query_lower or "xe gắn máy" in query_lower or "mô tô" in query_lower:
            return "xe máy"
        if "ô tô" in query_lower or "xe hơi" in query_lower:
            return "ô tô"
        if "người đi bộ" in query_lower:
            return "người đi bộ"
        # Trả về None nếu không tìm thấy đối tượng cụ thể
        return None

    def _build_weaviate_where_filter(self, filters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if "operator" in filters and "operands" in filters:
            return filters
        operands = []
        for key, value in filters.items():
            if value is None: continue # Bỏ qua nếu giá trị filter là None

            if key == "year_range_or" or key == "entity_type_list_or":
                operands.append(value)
            elif key == "nam_ban_hanh_range":
                year_ops = []
                if "gte" in value: year_ops.append({"path": ["nam_ban_hanh"], "operator": "GreaterThanEqual", "valueInt": value["gte"]})
                if "lte" in value: year_ops.append({"path": ["nam_ban_hanh"], "operator": "LessThanEqual", "valueInt": value["lte"]})
                if year_ops: operands.append({"operator": "And", "operands": year_ops} if len(year_ops) > 1 else year_ops[0])
            elif isinstance(value, str):
                operands.append({"path": [key], "operator": "Equal", "valueString": value})
            elif isinstance(value, int):
                operands.append({"path": [key], "operator": "Equal", "valueInt": value})
            elif isinstance(value, list) and all(isinstance(item, str) for item in value) and key == "entity_type":
                 # Nếu metadata 'entity_type' là list of strings và Weaviate schema hỗ trợ
                operands.append({"path": [key], "operator": "ContainsAny", "valueText": value})
            # Bỏ qua các kiểu khác hoặc log warning
            # else:
            #    logger.warning(f"Kiểu dữ liệu không hỗ trợ cho filter Weaviate: key='{key}', value_type='{type(value)}'")


        if not operands: return None
        return {"operator": "And", "operands": operands} if len(operands) > 1 else operands[0]

    def _extract_filters_and_query_year(self, query: str, query_metadata: Optional[Dict] = None) -> Tuple[Dict[str, Any], Optional[int]]:
        filters = {}
        query_lower = query.lower()

        extracted_field = query_metadata.get("field") if query_metadata else None
        if not extracted_field or extracted_field == "khac":
            # (Đây là nơi bạn gọi hàm infer_field đầy đủ của mình)
            if "giao thông" in query_lower: filters["field"] = "giao_thong"
            elif "hình sự" in query_lower: filters["field"] = "hinh_su"
            elif "lao động" in query_lower: filters["field"] = "lao_dong"
        elif extracted_field and extracted_field != "khac":
            filters["field"] = extracted_field

        if "nghị định" in query_lower: filters["loai_van_ban"] = "NGHỊ ĐỊNH"
        elif "bộ luật" in query_lower: filters["loai_van_ban"] = "BỘ LUẬT"
        elif "luật" in query_lower and "luật sư" not in query_lower: filters["loai_van_ban"] = "LUẬT"
        elif "thông tư" in query_lower: filters["loai_van_ban"] = "THÔNG TƯ"

        query_year = self._parse_year_from_query(query)
        if query_year:
            if self.year_filter_margin > 0:
                year_ops = [{"path": ["nam_ban_hanh"], "operator": "Equal", "valueInt": query_year + y_offset}
                            for y_offset in range(-self.year_filter_margin, self.year_filter_margin + 1)]
                if year_ops: filters["year_range_or"] = {"operator": "Or", "operands": year_ops}
            else:
                filters["nam_ban_hanh"] = query_year

        if m := re.search(r"(\d+/\d{4}/[\w.-]+(?:-[\w\d.-]+)?|\d+-\d{4}-[\w.-]+)", query, re.IGNORECASE):
            filters["so_hieu"] = m.group(1)

        if query_metadata and query_metadata.get("entity_type"):
            filters["entity_type"] = query_metadata.get("entity_type")

        logger.debug(f"Filters extracted (pre-build): {filters}")
        return filters, query_year

    def _apply_recency_and_relevance(self, documents_with_scores: List[Tuple[Document, float]], query_year: Optional[int], query_has_no_year: bool) -> List[Document]:
        if not documents_with_scores: return []
        current_year = datetime.datetime.now().year
        processed_docs = []

        for doc, search_score in documents_with_scores:
            doc_year = doc.metadata.get("nam_ban_hanh")
            recency_score_component = 0.5
            if isinstance(doc_year, int):
                year_diff_from_current = current_year - doc_year
                if query_year:
                    year_diff_from_query = abs(doc_year - query_year)
                    if year_diff_from_query == 0: recency_score_component = 1.0
                    elif year_diff_from_query <= max(1, self.year_filter_margin): recency_score_component = 0.9 - (year_diff_from_query * 0.05)
                    else: recency_score_component = 0.6 - (min(year_diff_from_query, 5) * 0.05)
                elif query_has_no_year:
                    year_diff_from_current = current_year - doc_year
                    # ===>> CẢI TIẾN LOGIC ĐIỂM "DỐC" HƠN <<===
                    if year_diff_from_current < 0: recency_score_component = 0.0
                    elif year_diff_from_current <= 1: recency_score_component = 1.0 # Rất mới
                    elif year_diff_from_current <= 3: recency_score_component = 0.9 # Khá mới
                    elif year_diff_from_current <= 5: recency_score_component = 0.6 # Bắt đầu cũ
                    else: recency_score_component = 0.2 # Rất cũ

            final_score = (1.0 - self.recency_bias_weight) * search_score + self.recency_bias_weight * recency_score_component

            if not hasattr(doc, 'metadata') or doc.metadata is None: doc.metadata = {}
            doc.metadata['final_score_debug'] = final_score
            doc.metadata['search_score_debug'] = search_score
            doc.metadata['recency_component_debug'] = recency_score_component
            processed_docs.append(doc)

        processed_docs.sort(key=lambda x: x.metadata['final_score_debug'], reverse=True)

        logger.debug(f"Recency & Relevance. Query_has_no_year: {query_has_no_year}. Top 3:")
        for item_doc in processed_docs[:3]:
            logger.debug(
                f"  Title: {item_doc.metadata.get('title', 'N/A')}, "
                f"Year: {item_doc.metadata.get('nam_ban_hanh')}, "
                f"SrcScore: {item_doc.metadata.get('search_score_debug', 0.0):.3f}, "
                f"Recency: {item_doc.metadata.get('recency_component_debug', 0.0):.2f}, "
                f"Final: {item_doc.metadata.get('final_score_debug', 0.0):.3f}"
            )
        return processed_docs

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun, query_metadata: Optional[Dict] = None
    ) -> List[Document]:

        queries_to_search = [query]
        query_field = None
        if query_metadata and isinstance(query_metadata, dict): # Kiểm tra query_metadata có phải dict không
            query_field = query_metadata.get("field")

        # Nếu query_field vẫn là None từ query_metadata, thử lấy từ việc phân tích query gốc
        if not query_field or query_field == "khac":
            # temp_filters_for_field sẽ là một dict, có thể rỗng
            temp_filters_for_field, _ = self._extract_filters_and_query_year(query, None) # query_metadata ở đây là None để chỉ phân tích query
            if temp_filters_for_field and isinstance(temp_filters_for_field, dict): # Đảm bảo nó là dict
                query_field_from_temp = temp_filters_for_field.get("field")
                if query_field_from_temp and query_field_from_temp != "khac":
                    query_field = query_field_from_temp
            # Nếu vẫn không có field, query_field sẽ là None hoặc "khac" (nếu infer_field trả về "khac")

        logger.debug(f"Query field for expansion: {query_field}")

        if self.enable_query_expansion:
            # generate_expanded_queries nên có khả năng xử lý query_field là None
            expanded_set = generate_expanded_queries(query, query_field, self.max_expanded_queries)
            queries_to_search = list(expanded_set)

        # extracted_filters_from_original_query sẽ là một dict, có thể rỗng
        extracted_filters_from_original_query, query_year = self._extract_filters_and_query_year(query, query_metadata)

        # Trích xuất đối tượng và thêm vào bộ lọc
        subject = self._extract_subject_from_query(query)
        if subject:
            # Chúng ta sẽ tìm các tài liệu có metadata `doi_tuong_ap_dung` là `subject` HOẶC `chung`
            # Cấu trúc filter này phụ thuộc vào backend (Weaviate, ChromaDB, etc.)
            # Ví dụ cho Weaviate với toán tử OR:
            subject_filter = {
                "operator": "Or",
                "operands": [
                    {"path": ["doi_tuong_ap_dung"], "operator": "Equal", "valueString": subject},
                    {"path": ["doi_tuong_ap_dung"], "operator": "Equal", "valueString": "chung"}
                ]
            }
            # Kết hợp filter đối tượng với các filter đã có
            if "operator" in extracted_filters_from_original_query and extracted_filters_from_original_query["operator"] == "And":
                 extracted_filters_from_original_query["operands"].append(subject_filter)
            else:
                 # Nếu chưa có filter hoặc chỉ có 1 filter, gói chúng vào trong AND
                 existing_operands = [extracted_filters_from_original_query] if extracted_filters_from_original_query else []
                 extracted_filters_from_original_query = {
                     "operator": "And",
                     "operands": existing_operands + [subject_filter]
                 }

        weaviate_where_filter = self._build_weaviate_where_filter(extracted_filters_from_original_query or {}) # Đảm bảo là dict

        query_has_no_year_flag = query_year is None
        k_to_fetch = self.default_k * self.initial_fetch_k_multiplier
        if query_has_no_year_flag:
             k_to_fetch = max(k_to_fetch, self.default_k * (self.initial_fetch_k_multiplier + 1) )

        all_retrieved_docs_with_scores: Dict[str, Tuple[Document, float]] = {}

        for q_idx, current_query_to_search in enumerate(queries_to_search):
            # Phân bổ k cho mỗi query, đảm bảo tổng số lượng ứng viên không quá ít
            k_for_this_q = max(self.default_k // 2, k_to_fetch // len(queries_to_search) if queries_to_search else k_to_fetch)
            if query_has_no_year_flag and len(queries_to_search) > 1: # Nếu mở rộng query và không có năm, mỗi query con nên lấy nhiều hơn
                k_for_this_q = max(k_for_this_q, self.default_k)


            logger.info(f"Searching {q_idx+1}/{len(queries_to_search)}: '{current_query_to_search}', k_fetch={k_for_this_q}, where={weaviate_where_filter}")

            current_search_params = {"query": current_query_to_search, "k": k_for_this_q}
            if weaviate_where_filter:
                current_search_params["where"] = weaviate_where_filter

            docs_with_scores_for_current_query: List[Tuple[Document, float]] = []
            try:
                # Không truyền score_threshold vào đây nữa, sẽ lọc sau
                docs_with_scores_for_current_query = self.vectorstore.similarity_search_with_score(**current_search_params)
            except TypeError as te:
                 logger.warning(f"TypeError khi search cho query '{current_query_to_search}' (có thể do tham số 'where' không được hỗ trợ đúng cách bởi similarity_search_with_score cho backend này): {te}. Thử lại không có 'where'.")
                 current_search_params.pop("where", None)
                 try:
                     docs_with_scores_for_current_query = self.vectorstore.similarity_search_with_score(**current_search_params)
                 except Exception as e_retry_no_where:
                     logger.error(f"Lỗi khi truy xuất (retry không where) cho query '{current_query_to_search}': {e_retry_no_where}", exc_info=True)
            except Exception as e:
                logger.error(f"Lỗi khi truy xuất cho query '{current_query_to_search}': {e}", exc_info=True)

            for doc, score in docs_with_scores_for_current_query:
                doc_id_key = doc.id if hasattr(doc, 'id') and doc.id else doc.page_content[:100]
                if doc_id_key not in all_retrieved_docs_with_scores or score > all_retrieved_docs_with_scores[doc_id_key][1]:
                    all_retrieved_docs_with_scores[doc_id_key] = (doc, score)

        initial_docs_with_scores_list = list(all_retrieved_docs_with_scores.values())
        logger.info(f"Tổng cộng truy xuất được {len(initial_docs_with_scores_list)} chunks ứng viên từ tất cả các query.")

        if not initial_docs_with_scores_list:
            if query_year and ("nam_ban_hanh" in extracted_filters_from_original_query or "year_range_or" in extracted_filters_from_original_query):
                logger.info(f"Không tìm thấy kết quả với filter năm. Thử tìm không lọc theo năm...")
                filters_no_year = {k:v for k,v in extracted_filters_from_original_query.items() if k not in ["nam_ban_hanh", "year_range_or"]}
                weaviate_where_filter_no_year = self._build_weaviate_where_filter(filters_no_year)

                fallback_k = self.default_k * self.initial_fetch_k_multiplier
                search_params_no_year = {"query": query, "k": fallback_k}
                if weaviate_where_filter_no_year: search_params_no_year["where"] = weaviate_where_filter_no_year

                try:
                    initial_docs_with_scores_list = self.vectorstore.similarity_search_with_score(**search_params_no_year)
                    logger.info(f"Truy xuất được {len(initial_docs_with_scores_list)} chunks (fallback không lọc năm).")
                    if not initial_docs_with_scores_list: return []
                except Exception as e_no_year:
                    logger.error(f"Lỗi khi truy xuất (fallback không lọc năm): {e_no_year}", exc_info=True)
                    return []
            else:
                return []

        re_ranked_documents = self._apply_recency_and_relevance(initial_docs_with_scores_list, query_year, query_has_no_year_flag)

        final_results_before_threshold = re_ranked_documents
        if self.final_score_threshold_after_bias is not None:
            final_results_after_threshold = [
                doc for doc in final_results_before_threshold
                if doc.metadata.get('final_score_debug', 0.0) >= self.final_score_threshold_after_bias
            ]
            if len(final_results_after_threshold) < len(final_results_before_threshold):
                 logger.info(f"Áp dụng final_score_threshold={self.final_score_threshold_after_bias}. Lọc từ {len(final_results_before_threshold)} xuống {len(final_results_after_threshold)} docs.")
            re_ranked_documents = final_results_after_threshold

        final_results = re_ranked_documents[:self.default_k]
        logger.info(f"Trả về {len(final_results)} chunks (Query Expansion: {self.enable_query_expansion}, Recency Bias: {self.recency_bias_weight}).")
        return final_results