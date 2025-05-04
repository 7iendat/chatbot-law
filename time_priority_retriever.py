from langchain.retrievers import MultiQueryRetriever
from operator import itemgetter
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever
from typing import List, Dict, Any, Optional, Tuple
import datetime

class YearPrioritizedRetriever(BaseRetriever):
    """Retriever ưu tiên tài liệu theo năm mới nhất."""

    def __init__(self, base_retriever, llm, threshold=0.7, max_documents_per_year=3):
        """
        Khởi tạo retriever ưu tiên theo năm.

        Args:
            base_retriever: Retriever cơ bản để tìm kiếm văn bản
            llm: Language model để tạo các truy vấn tương tự
            threshold: Ngưỡng điểm số tương đồng để coi là kết quả phù hợp
            max_documents_per_year: Số lượng tài liệu tối đa lấy ra cho mỗi năm
        """
        super().__init__()
        self._base_retriever = base_retriever
        self._multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=llm
        )
        self._threshold = threshold
        self._max_documents_per_year = max_documents_per_year

    def _get_year(self, doc: Document) -> int:
        """Lấy năm từ metadata của tài liệu."""
        try:
            return int(doc.metadata.get("year", 0))
        except (ValueError, TypeError):
            return 0

    def _get_current_year(self) -> int:
        """Lấy năm hiện tại."""
        return datetime.datetime.now().year

    def _group_by_year(self, docs: List[Document]) -> Dict[int, List[Tuple[Document, float]]]:
        """Nhóm các tài liệu theo năm và sắp xếp theo điểm số."""
        result = {}
        for doc in docs:
            year = self._get_year(doc)
            score = doc.metadata.get("score", 0.0)
            if year not in result:
                result[year] = []
            result[year].append((doc, score))

        # Sắp xếp các tài liệu trong mỗi năm theo điểm số giảm dần
        for year in result:
            result[year] = sorted(result[year], key=itemgetter(1), reverse=True)

        return result

    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """
        Truy xuất tài liệu ưu tiên theo năm mới nhất.

        Args:
            query: Câu truy vấn
            run_manager: Quản lý callback cho quá trình truy xuất

        Returns:
            Danh sách tài liệu phù hợp
        """
        # Sử dụng MultiQueryRetriever để tạo các truy vấn tương đồng và lấy kết quả
        callbacks = run_manager.get_child() if run_manager else None
        all_docs = self._multi_query_retriever.get_relevant_documents(query, callbacks=callbacks)

        # Đảm bảo mỗi tài liệu có điểm số
        for doc in all_docs:
            if "score" not in doc.metadata:
                doc.metadata["score"] = 0.5  # Gán điểm mặc định

        # Nhóm tài liệu theo năm
        docs_by_year = self._group_by_year(all_docs)

        # Sắp xếp các năm theo thứ tự giảm dần (mới nhất trước)
        sorted_years = sorted(docs_by_year.keys(), reverse=True)

        result_docs = []
        current_year = self._get_current_year()

        # Đầu tiên, kiểm tra xem có tài liệu nào trong năm hiện tại vượt qua ngưỡng
        if current_year in docs_by_year:
            high_quality_current_docs = [
                doc for doc, score in docs_by_year[current_year][:self._max_documents_per_year]
                if score >= self._threshold
            ]
            if high_quality_current_docs:
                return high_quality_current_docs

        # Nếu không tìm thấy tài liệu chất lượng cao trong năm hiện tại,
        # xem xét các năm theo thứ tự giảm dần
        for year in sorted_years:
            top_docs = [doc for doc, _ in docs_by_year[year][:self._max_documents_per_year]]
            result_docs.extend(top_docs)

            # Kiểm tra xem đã tìm thấy đủ tài liệu chất lượng tốt chưa
            high_quality_docs = [
                doc for doc, score in docs_by_year[year][:self._max_documents_per_year]
                if score >= self._threshold
            ]

            if high_quality_docs:
                return result_docs  # Trả về ngay khi tìm thấy tài liệu chất lượng cao

        # Nếu không tìm thấy tài liệu nào vượt qua ngưỡng, trả về tất cả các tài liệu đã thu thập
        return result_docs

class SemanticSimilarityEnhancer(BaseRetriever):
    """Lớp nâng cao điểm số cho các tài liệu có ý nghĩa tương đối giống với câu hỏi."""

    def __init__(self, base_retriever: BaseRetriever, embedding_model):
        """
        Khởi tạo bộ nâng cao độ tương đồng ngữ nghĩa.

        Args:
            base_retriever: Bộ truy xuất cơ sở để lấy tài liệu ban đầu
            embedding_model: Mô hình nhúng văn bản để tính toán độ tương đồng
        """
        super().__init__()
        self._base_retriever = base_retriever
        self._embedding_model = embedding_model

    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """
        Lấy tài liệu từ bộ truy xuất cơ sở và nâng cao điểm số dựa trên độ tương đồng ngữ nghĩa.

        Args:
            query: Câu truy vấn
            run_manager: Quản lý callback cho quá trình truy xuất

        Returns:
            Danh sách tài liệu đã được điều chỉnh điểm số
        """
        # Lấy tài liệu từ bộ truy xuất cơ sở
        docs = self._base_retriever.get_relevant_documents(query, callbacks=run_manager.get_child() if run_manager else None)

        if not docs:
            return []

        # Nhúng câu truy vấn
        query_embedding = self._embedding_model.embed_query(query)

        enhanced_docs = []
        for doc in docs:
            # Nhúng nội dung tài liệu
            doc_embedding = self._embedding_model.embed_documents([doc.page_content])[0]

            # Tính toán độ tương đồng cosine
            semantic_similarity = self._cosine_similarity(query_embedding, doc_embedding)

            # Cập nhật hoặc tạo điểm số trong metadata
            if "score" in doc.metadata:
                # Nếu đã có điểm số, có thể kết hợp với điểm số hiện tại
                # Ví dụ: trung bình, tích, hoặc trọng số tùy chỉnh
                doc.metadata["original_score"] = doc.metadata["score"]
                doc.metadata["semantic_score"] = semantic_similarity
                doc.metadata["score"] = (doc.metadata["original_score"] + semantic_similarity) / 2
            else:
                doc.metadata["score"] = semantic_similarity

            enhanced_docs.append(doc)

        # Sắp xếp theo điểm số giảm dần
        enhanced_docs.sort(key=lambda x: x.metadata["score"], reverse=True)

        return enhanced_docs

    def _cosine_similarity(self, vec1, vec2):
        """Tính toán độ tương đồng cosine giữa hai vector."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        return dot_product / (norm1 * norm2) if norm1 * norm2 > 0 else 0.0

# Ví dụ sử dụng
def create_retriever(vector_store, embedding_model, llm):
    """
    Tạo chatbot luật với khả năng tìm kiếm ưu tiên theo năm.

    Args:
        vector_store: Kho lưu trữ vector đã được xây dựng
        embedding_model: Mô hình nhúng văn bản
        llm: Mô hình ngôn ngữ

    Returns:
        Retriever đã được cấu hình
    """
    # Tạo retriever cơ bản từ vector store
    base_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10}  # Lấy nhiều tài liệu hơn để có thể lọc
    )

    # Tạo bộ nâng cao độ tương đồng ngữ nghĩa
    enhancer = SemanticSimilarityEnhancer(base_retriever=base_retriever, embedding_model=embedding_model)

    # SemanticSimilarityEnhancer là một BaseRetriever,
    # nên chúng ta sử dụng nó trực tiếp
    enhanced_retriever = enhancer

    # Tạo retriever ưu tiên theo năm
    year_prioritized_retriever = YearPrioritizedRetriever(
        base_retriever=enhanced_retriever,
        llm=llm,
        threshold=0.65,  # Điều chỉnh ngưỡng phù hợp với dữ liệu của bạn
        max_documents_per_year=3
    )

    return year_prioritized_retriever