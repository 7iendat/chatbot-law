# import logging
# import re
# import sys
# from datetime import datetime
# from typing import List, Dict, Optional, Tuple, Set
# from langchain.retrievers import ContextualCompressionRetriever
# from langchain.retrievers.document_compressors import EmbeddingsFilter
# from langchain.schema.retriever import BaseRetriever
# from langchain.schema import Document
# from langchain.prompts import PromptTemplate
# from langchain_community.retrievers import BM25Retriever
# from langchain.retrievers import EnsembleRetriever
# from utils.synonym_map import SYNONYM_MAP, get_synonyms
# from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
# import requests
# from langchain_chroma import Chroma

# # Tăng giới hạn đệ quy của Python để phòng trường hợp
# sys.setrecursionlimit(3000)  # Tăng giới hạn đệ quy lên 3000

# # Cấu hình logger
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# def augment_query(query: str, llm, field_mapping: Dict, use_llm_paraphrase: bool = False) -> List[str]:
#     """Mở rộng truy vấn bằng từ đồng nghĩa và paraphrase (nếu bật)."""
#     query_lower = query.lower()
#     augmented_queries = {query}

#     # Phát hiện lĩnh vực từ truy vấn
#     query_fields = detect_field_from_query(query_lower, llm, field_mapping) or ["giao_thong"]

#     # Thêm các query với từ đồng nghĩa
#     for field in query_fields:
#         for common_term, legal_terms in SYNONYM_MAP.get(field, {}).items():
#             if common_term.lower() in query_lower:
#                 for legal_term in legal_terms:
#                     new_query = query_lower.replace(common_term.lower(), legal_term.lower())
#                     augmented_queries.add(new_query)

#     # Thêm query được paraphrase bởi LLM (nếu được bật)
#     if llm and use_llm_paraphrase:
#         try:
#             prompt = PromptTemplate(
#                 input_variables=["query"],
#                 template="Chuyển câu hỏi sau thành ngôn ngữ pháp lý chuyên ngành: {query}"
#             )
#             paraphrased_query = invoke_with_retry(llm, prompt.format(query=query))
#             augmented_queries.add(paraphrased_query)
#         except Exception as e:
#             logger.warning(f"🔸Lỗi khi paraphrase query: {e}")

#     return list(augmented_queries)

# def detect_field_from_query(query_lower: str, llm, field_mapping: Dict, use_llm: bool = True) -> List[str]:
#     """Phát hiện lĩnh vực từ câu hỏi dựa trên từ khóa hoặc LLM."""
#     if not query_lower:
#         return ["giao_thong"]  # Giá trị mặc định

#     detected_fields = set()

#     # Phát hiện lĩnh vực dựa trên từ khóa
#     for field_group, keywords in field_mapping.items():
#         if any(keyword.lower() in query_lower for keyword in keywords):
#             detected_fields.add(field_group)

#     # Sử dụng LLM để phát hiện lĩnh vực nếu không tìm thấy bằng từ khóa
#     if not detected_fields and llm and use_llm:
#         try:
#             fields_str = ", ".join(field_mapping.keys())
#             prompt = PromptTemplate(
#                 input_variables=["query", "fields"],
#                 template="Phân loại câu hỏi sau vào một trong các lĩnh vực pháp luật: {fields}\nCâu hỏi: {query}\nTrả lời chỉ tên lĩnh vực."
#             )
#             field_prediction = invoke_with_retry(llm, prompt.format(fields=fields_str, query=query_lower)).lower()

#             for field in field_mapping:
#                 if field.lower() in field_prediction or field_prediction in field.lower():
#                     detected_fields.add(field)
#                     break
#         except Exception as e:
#             logger.warning(f"🔸Lỗi khi dùng LLM để phân loại: {e}")

#     return list(detected_fields) or ["giao_thong"]

# @retry(
#     stop=stop_after_attempt(3),
#     wait=wait_exponential(multiplier=1, min=2, max=8),
#     retry=retry_if_exception_type((requests.exceptions.HTTPError, requests.exceptions.RequestException))
# )
# def invoke_with_retry(llm, prompt: str) -> str:
#     """Gọi LLM với cơ chế retry."""
#     try:
#         response = llm.invoke(prompt)
#         return response.content.strip() if hasattr(response, 'content') else str(response).strip()
#     except Exception as e:
#         logger.warning(f"LLM invoke failed: {e}")
#         return ""

# def extract_penalty_amount(penalty_text: Optional[str]) -> float:
#     """Trích xuất mức phạt cao nhất từ chuỗi văn bản."""
#     if not penalty_text or not isinstance(penalty_text, str):
#         return 0.0

#     amounts = re.findall(r'(\d+(?:\.\d+)?)\s*(?:triệu|nghìn|tỷ|tr|đồng|VND)', penalty_text, re.IGNORECASE)
#     numeric_values = []

#     for amount in amounts:
#         try:
#             value = float(amount.replace(".", "").replace(",", "."))
#             # Nhân với hệ số tương ứng
#             if "tỷ" in penalty_text.lower():
#                 value *= 1_000_000_000
#             elif "triệu" in penalty_text.lower() or "tr" in penalty_text.lower():
#                 value *= 1_000_000
#             elif "nghìn" in penalty_text.lower():
#                 value *= 1_000
#             numeric_values.append(value)
#         except ValueError:
#             continue

#     return max(numeric_values, default=0.0)

# def check_field_match(doc_field: Optional[str], query_fields: List[str], field_mapping: Dict) -> bool:
#     """Kiểm tra xem lĩnh vực của tài liệu có khớp với lĩnh vực truy vấn không."""
#     if not doc_field or not query_fields:
#         return False

#     doc_field_lower = doc_field.lower()

#     # Kiểm tra khớp trực tiếp
#     if doc_field_lower in {field.lower() for field in query_fields}:
#         return True

#     # Kiểm tra khớp gián tiếp thông qua từ khóa
#     for field_group, keywords in field_mapping.items():
#         if field_group.lower() in {field.lower() for field in query_fields}:
#             if doc_field_lower in {keyword.lower() for keyword in keywords}:
#                 return True

#     return False

# def score_document(doc: Document, query_fields: List[str], config: Dict, field_mapping: Dict, threshold_year: int) -> float:
#     """Tính điểm cho tài liệu dựa trên metadata."""
#     score = 0.0
#     metadata = getattr(doc, 'metadata', {})

#     # Điểm cho tính thời sự
#     year = metadata.get('year')
#     if year:
#         try:
#             doc_year = int(year)
#             if doc_year >= threshold_year:
#                 recency_score = min(1.0, (doc_year - threshold_year + 1) / (config['recent_years'] + 1))
#                 score += recency_score * config['recency_boost']
#         except (ValueError, TypeError):
#             pass

#     # Điểm cho lĩnh vực phù hợp
#     if query_fields and metadata.get('field'):
#         if check_field_match(metadata['field'], query_fields, field_mapping):
#             score += config['field_boost']

#     # Điểm cho mức phạt
#     if metadata.get('penalty') and config['penalty_boost'] > 0:
#         penalty_amount = extract_penalty_amount(metadata['penalty'])
#         if penalty_amount > 0:
#             penalty_score = min(1.0, penalty_amount / 1_000_000_000)
#             score += penalty_score * config['penalty_boost']

#     return score

# def get_all_documents(vectorstore, max_docs: int = 1000) -> List[Document]:
#     """
#     Lấy tất cả tài liệu từ vector store, hỗ trợ cả FAISS và Chroma.
#     """
#     try:
#         if isinstance(vectorstore, Chroma):
#             # Lấy tài liệu từ Chroma
#             results = vectorstore.get(include=["documents", "metadatas"])
#             documents = []
#             for doc, metadata in zip(results.get("documents", []), results.get("metadatas", [])):
#                 if doc:  # Đảm bảo doc không rỗng
#                     documents.append(Document(page_content=doc, metadata=metadata or {}))
#             return documents[:max_docs]
#         else:
#             logger.warning(f"🔸Vector store không được hỗ trợ: {type(vectorstore)}. Sử dụng similarity_search với k={max_docs}.")
#             return vectorstore.similarity_search("", k=max_docs)
#     except Exception as e:
#         logger.error(f"🔸Lỗi khi lấy tài liệu: {e}")
#         return []

# def create_retriever(vectorstore, embedding_model, llm, config: Optional[Dict] = None) -> Optional[BaseRetriever]:
#     """
#     Tạo và cấu hình retriever từ vector store với tính năng ưu tiên tài liệu mới và tối ưu hóa
#     cho tìm kiếm pháp luật đa lĩnh vực.
#     """
#     logger.info("🔸Đang tạo retriever tùy chỉnh cho dữ liệu pháp luật đa lĩnh vực...")

#     default_config = {
#         "recent_years": 3,
#         "primary_docs": 4,
#         "total_docs": 6,
#         "content_min_chars": 1000,
#         "search_type": "hybrid",
#         "field_boost": 0.3,
#         "recency_boost": 0.5,
#         "penalty_boost": 0.2,
#         "auto_detect_field": True,
#         "use_hybrid_search": True,
#         "use_llm_paraphrase": False,
#         "search_kwargs": {"k": 18, "fetch_k": 30, "lambda_mult": 0.7},
#         "field_mapping": {
#             "giao_thong": ["giao thông", "đường bộ", "đường thủy", "hàng không", "đường sắt", "vận tải", "đèn đỏ", "tín hiệu giao thông", "vượt đèn", "xe máy", "ô tô"],
#             "thue": ["thuế", "phí", "lệ phí", "ngân sách", "tài chính", "thuế thu nhập", "thuế VAT", "trốn thuế"],
#             "lao_dong": ["lao động", "việc làm", "nhân sự", "bảo hiểm xã hội", "bảo hiểm y tế", "lương", "thưởng", "nghỉ thai sản"],
#             "doanh_nghiep": ["doanh nghiệp", "công ty", "kinh doanh", "thương mại", "đầu tư", "khởi nghiệp", "đăng ký kinh doanh"],
#             "dat_dai": ["đất đai", "nhà ở", "bất động sản", "quy hoạch", "sở hữu đất", "thuê đất", "sổ đỏ", "chuyển nhượng đất"],
#             "hon_nhan": ["hôn nhân", "gia đình", "ly hôn", "con cái", "nuôi con", "thừa kế", "bạo lực gia đình"],
#             "hanh_chinh": ["hành chính", "thủ tục", "biên bản", "vi phạm hành chính", "khiếu nại", "tố cáo", "hộ khẩu", "căn cước"],
#             "hinh_su": ["hình sự", "tội phạm", "trộm cắp", "cướp", "lừa đảo", "tham nhũng", "ma túy", "giết người"],
#             "dan_su": ["dân sự", "hợp đồng", "giao dịch", "bồi thường", "nghĩa vụ dân sự", "nợ", "thừa kế"],
#             "moi_truong": ["môi trường", "ô nhiễm", "rác thải", "tài nguyên", "bảo vệ môi trường", "phạt môi trường"],
#             "y_te": ["y tế", "sức khỏe", "bệnh viện", "khám chữa bệnh", "dược phẩm", "bảo hiểm y tế", "vắc xin"],
#             "giao_duc": ["giáo dục", "học tập", "trường học", "đại học", "sinh viên", "học sinh", "học phí"],
#             "xay_dung": ["xây dựng", "công trình", "nhà thầu", "quy hoạch xây dựng", "cấp phép xây dựng", "xây không phép"],
#             "thuong_mai": ["thương mại", "mua bán", "xuất nhập khẩu", "thương hiệu", "cạnh tranh", "bán lẻ", "người tiêu dùng"],
#             "cong_nghe": ["công nghệ", "thông tin", "viễn thông", "internet", "phần mềm", "dữ liệu", "bảo mật", "sở hữu trí tuệ"]
#         }
#     }
#     config = {**default_config, **(config or {})}

#     # Kiểm tra cài đặt cơ bản
#     if config['total_docs'] < config['primary_docs']:
#         logger.error("🔸total_docs phải lớn hơn hoặc bằng primary_docs")
#         return None
#     if not vectorstore or not embedding_model:
#         logger.error("🔸vectorstore hoặc embedding_model không được None")
#         return None

#     threshold_year = datetime.now().year - config['recent_years']

#     try:
#         # Tạo retriever cơ sở (hybrid hoặc vector)
#         if config['search_type'] == "hybrid" and config['use_hybrid_search']:
#             # Tạo vector retriever
#             vector_search_kwargs = config['search_kwargs'].copy()
#             vector_retriever = vectorstore.as_retriever(
#                 search_type="mmr",
#                 search_kwargs=vector_search_kwargs
#             )

#             # Tạo keyword retriever (BM25)
#             documents = get_all_documents(vectorstore, max_docs=1000)
#             if not documents:
#                 logger.warning("🔸Không tìm thấy tài liệu nào trong vector store")
#                 documents = [Document(page_content="Tài liệu trống")]

#             texts = []
#             for doc in documents:
#                 if hasattr(doc, 'page_content') and doc.page_content:
#                     texts.append(doc.page_content)
#                 else:
#                     texts.append(str(doc))

#             if not texts:
#                 texts = ["Tài liệu trống"]

#             keyword_retriever = BM25Retriever.from_texts(texts, k=config['total_docs'] * 2)

#             # Kết hợp cả hai retriever
#             base_retriever = EnsembleRetriever(
#                 retrievers=[vector_retriever, keyword_retriever],
#                 weights=[0.6, 0.4]
#             )
#         else:
#             # Chỉ dùng vector retriever
#             base_retriever = vectorstore.as_retriever(
#                 search_type=config['search_type'],
#                 search_kwargs=config['search_kwargs']
#             )

#         # Tạo custom retriever
#         class CustomLegalRetriever(BaseRetriever):
#             def __init__(self):
#                 super().__init__()
#                 # Thêm cache để tránh đệ quy vô hạn
#                 self._query_cache = {}

#             def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
#                 # Kiểm tra cache để tránh đệ quy
#                 cache_key = str(query)
#                 if cache_key in self._query_cache:
#                     logger.info(f"🔸Sử dụng kết quả từ cache cho query: {query[:50]}...")
#                     return self._query_cache[cache_key]

#                 # Đánh dấu query này đang được xử lý (tránh đệ quy)
#                 self._query_cache[cache_key] = []

#                 try:
#                     # Đảm bảo query là chuỗi
#                     query = str(query) if query else ""
#                     if not query.strip():
#                         logger.warning("🔸Query rỗng, trả về danh sách trống")
#                         return []

#                     # Mở rộng query với các từ đồng nghĩa
#                     queries = augment_query(query, llm, config['field_mapping'], config['use_llm_paraphrase'])
#                     logger.info(f"🔸Mở rộng query: {len(queries)} biến thể")

#                     # Thu thập tài liệu từ tất cả các query
#                     unique_docs = {}  # Sử dụng dict để dễ dàng kiểm tra trùng lặp

#                     for q in queries:
#                         try:
#                             # Đảm bảo không gọi lại chính mình (tránh đệ quy)
#                             if hasattr(base_retriever, 'invoke'):
#                                 docs = base_retriever.invoke(q)
#                             elif hasattr(base_retriever, 'get_relevant_documents'):
#                                 docs = base_retriever.get_relevant_documents(q)
#                             else:
#                                 logger.warning(f"🔸Retriever không có method invoke hoặc get_relevant_documents")
#                                 continue

#                             for doc in docs:
#                                 # Tạo ID duy nhất cho tài liệu
#                                 doc_id = getattr(doc, 'metadata', {}).get('source', str(id(doc)))
#                                 unique_docs[doc_id] = doc
#                         except Exception as e:
#                             logger.warning(f"🔸Lỗi khi truy xuất cho query '{q[:30]}...': {e}")
#                             continue

#                     if not unique_docs:
#                         logger.warning("🔸Không tìm thấy tài liệu nào phù hợp")
#                         self._query_cache[cache_key] = []
#                         return []

#                     # Phát hiện lĩnh vực từ query
#                     query_fields = detect_field_from_query(
#                         query.lower(),
#                         llm,
#                         config['field_mapping'],
#                         config['auto_detect_field']
#                     )
#                     logger.info(f"🔸Lĩnh vực phát hiện: {', '.join(query_fields)}")

#                     # Tính điểm và sắp xếp tài liệu
#                     scored_docs = []
#                     for doc in unique_docs.values():
#                         score = score_document(doc, query_fields, config, config['field_mapping'], threshold_year)
#                         scored_docs.append((doc, score))

#                     # Sắp xếp theo điểm giảm dần
#                     scored_docs.sort(key=lambda x: x[1], reverse=True)

#                     # Phân loại tài liệu theo năm
#                     recent_docs, older_docs = [], []
#                     for doc, score in scored_docs:
#                         metadata = getattr(doc, 'metadata', {})
#                         doc_year = 0
#                         if metadata.get('year'):
#                             try:
#                                 doc_year = int(metadata.get('year', 0))
#                             except (ValueError, TypeError):
#                                 doc_year = 0

#                         if doc_year >= threshold_year:
#                             recent_docs.append((doc, score))
#                         else:
#                             older_docs.append((doc, score))

#                     logger.info(f"🔸Phân loại: {len(recent_docs)} tài liệu mới, {len(older_docs)} tài liệu cũ")

#                     # Chọn tài liệu mới trước
#                     selected_docs = []
#                     content_length = 0

#                     # Thêm tài liệu mới
#                     for doc, score in recent_docs[:config['primary_docs']]:
#                         selected_docs.append(doc)
#                         content_length += len(doc.page_content if hasattr(doc, 'page_content') else str(doc))
#                         logger.debug(f"🔸Chọn tài liệu mới: {getattr(doc, 'metadata', {}).get('source', '?')} (score: {score:.2f})")

#                     # Kiểm tra xem đã đủ nội dung chưa
#                     has_sufficient_content = content_length >= config['content_min_chars']
#                     logger.info(f"🔸Tài liệu mới: {len(selected_docs)}, nội dung: {content_length}, đủ: {has_sufficient_content}")

#                     # Bổ sung tài liệu cũ nếu cần
#                     if not has_sufficient_content or len(selected_docs) < config['primary_docs']:
#                         logger.info("🔸Bổ sung tài liệu cũ")
#                         for doc, score in older_docs:
#                             if len(selected_docs) < config['total_docs']:
#                                 selected_docs.append(doc)
#                                 content_length += len(doc.page_content if hasattr(doc, 'page_content') else str(doc))
#                                 logger.debug(f"🔸Chọn tài liệu cũ: {getattr(doc, 'metadata', {}).get('source', '?')} (score: {score:.2f})")
#                                 if content_length >= config['content_min_chars'] and len(selected_docs) >= config['primary_docs']:
#                                     break
#                             else:
#                                 break

#                     # Giới hạn số lượng tài liệu trả về
#                     selected_docs = selected_docs[:config['total_docs']]

#                     # Thêm metadata cho các tài liệu được chọn
#                     for i, doc in enumerate(selected_docs):
#                         if not hasattr(doc, 'metadata'):
#                             doc.metadata = {}
#                         doc.metadata['_retrieval_score'] = scored_docs[i][1] if i < len(scored_docs) else 0
#                         if query_fields:
#                             doc.metadata['_detected_fields'] = ', '.join(query_fields)

#                     logger.info(f"🔸Kết quả: {len(selected_docs)} tài liệu, {content_length} ký tự")

#                     # Lưu kết quả vào cache
#                     self._query_cache[cache_key] = selected_docs
#                     return selected_docs

#                 except Exception as e:
#                     logger.error(f"🔸Lỗi khi truy xuất tài liệu: {e}")
#                     self._query_cache[cache_key] = []
#                     return []

#         base_retriever = CustomLegalRetriever()

#         # Thêm bộ lọc embedding nếu được cấu hình
#         if config.get("use_embedding_filter", True) and embedding_model:
#             logger.info("🔸Áp dụng bộ lọc embedding")
#             embedding_filter = EmbeddingsFilter(
#                 embeddings=embedding_model,
#                 similarity_threshold=config.get("similarity_threshold", 0.7)
#             )
#             retriever = ContextualCompressionRetriever(
#                 base_compressor=embedding_filter,
#                 base_retriever=base_retriever
#             )
#         else:
#             retriever = base_retriever

#         # Thêm phương thức invoke nếu chưa có
#         if not hasattr(retriever, "invoke"):
#             # Tránh đệ quy bằng cách gọi trực tiếp _get_relevant_documents
#             retriever.invoke = lambda q: retriever._get_relevant_documents(q) if hasattr(retriever, '_get_relevant_documents') else base_retriever._get_relevant_documents(q)
#             logger.info("🔸Thêm phương thức invoke")

#         logger.info("🔸Tạo retriever thành công")
#         return retriever

#     except Exception as e:
#         logger.error(f"🔸Lỗi khi tạo retriever: {e}", exc_info=True)
#         return None

from langchain_core.vectorstores import VectorStoreRetriever
from typing import List
from langchain_core.documents import Document

class WeaviateHybridRetriever(VectorStoreRetriever):
    def _get_relevant_documents(
        self, query: str, *, run_manager=None
    ) -> List[Document]:
        """Get documents relevant to a query using hybrid search."""
        return self.vectorstore.similarity_search(
            query=query,
            k=self.search_kwargs.get("k", 4),
            by_hybrid=True,
            alpha=self.search_kwargs.get("alpha", 0.5)
        )