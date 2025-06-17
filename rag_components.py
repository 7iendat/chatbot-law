from langchain_huggingface import HuggingFaceEmbeddings
import config
import prompt_templete
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
import utils.utils as utils
from core.runnables import router_as_runnable
from langchain_core.documents import Document
import logging
from langchain_core.output_parsers import StrOutputParser
from typing import List,Any,Optional,Dict
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from utils.route_logic import route_logic
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage
from utils.process_data import filter_and_serialize_complex_metadata
import weaviate
import weaviate.classes.config as wvc_config
from weaviate.exceptions import WeaviateQueryException
import time

logger = logging.getLogger(__name__)

WEAVIATE_SCHEMA_CONFIG: List[Dict[str, Any]] = [
    # Tên trường, Kiểu dữ liệu trong Weaviate, Có nên vector hóa trường này không?
    {"name": "source", "dataType": wvc_config.DataType.TEXT, "vectorize": False},
    {"name": "title", "dataType": wvc_config.DataType.TEXT, "vectorize": True},
    {"name": "field", "dataType": wvc_config.DataType.TEXT, "vectorize": True},
    {"name": "so_hieu", "dataType": wvc_config.DataType.TEXT, "vectorize": False},
    {"name": "loai_van_ban", "dataType": wvc_config.DataType.TEXT, "vectorize": True},
    {"name": "ten_van_ban", "dataType": wvc_config.DataType.TEXT, "vectorize": True},
    {"name": "co_quan_ban_hanh", "dataType": wvc_config.DataType.TEXT, "vectorize": False},
    {"name": "ngay_ban_hanh_str", "dataType": wvc_config.DataType.TEXT, "vectorize": False},
    {"name": "nam_ban_hanh", "dataType": wvc_config.DataType.INT, "vectorize": False},
    {"name": "phan_code", "dataType": wvc_config.DataType.TEXT, "vectorize": False},
    {"name": "chuong_code", "dataType": wvc_config.DataType.TEXT, "vectorize": False},
    {"name": "muc_code", "dataType": wvc_config.DataType.TEXT, "vectorize": False},
    {"name": "dieu_code", "dataType": wvc_config.DataType.TEXT, "vectorize": False},
    {"name": "entity_type", "dataType": wvc_config.DataType.TEXT, "vectorize": False},
    {"name": "penalties", "dataType": wvc_config.DataType.TEXT, "vectorize": False},
    {"name": "cross_references", "dataType": wvc_config.DataType.TEXT, "vectorize": False},
]

# Hàm get_huggingface_embeddings giữ nguyên
def get_huggingface_embeddings(model_name: str, device: str = 'cpu'):
    logger.info(f"🔸Đang khởi tạo model embedding: {model_name} trên thiết bị {device}...")

    model_kwargs = {
        'device': device,
        'trust_remote_code': True  # thêm để đảm bảo load được những model custom
    }
    encode_kwargs = {
        'batch_size': 32,  # kích thước batch cho embedding
        'normalize_embeddings': True  # normalize để cosine similarity chuẩn
    }

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        logger.info("🔸Khởi tạo model embedding thành công.")
        return embeddings
    except Exception as e:
        logger.error(f"🔸Lỗi khi khởi tạo model embedding: {e}")
        raise Exception(f"Khởi tạo model embedding thất bại: {str(e)}")

# Begin New

def _create_weaviate_schema_if_not_exists(client: weaviate.WeaviateClient, collection_name: str):
    """Tạo schema cho collection một cách tường minh từ cấu hình WEAVIATE_SCHEMA_CONFIG."""
    if client.collections.exists(collection_name):
        logger.info(f"✅ Schema cho collection '{collection_name}' đã tồn tại.")
        return

    logger.info(f"🔸 Schema cho collection '{collection_name}' chưa tồn tại. Đang tạo từ cấu hình...")
    try:
        # Tự động tạo list các thuộc tính từ cấu hình
        properties = [
            wvc_config.Property(
                name=prop["name"],
                data_type=prop["dataType"],
                # Nếu vectorize=True, thì không skip (skip=False).
                # Nếu vectorize=False, thì skip (skip=True).
                skip_vectorization=not prop["vectorize"]
            )
            for prop in WEAVIATE_SCHEMA_CONFIG
        ]
        # Thêm trường 'text' mặc định của LangChain
        properties.append(wvc_config.Property(name="text", data_type=wvc_config.DataType.TEXT, skip_vectorization=False))

        client.collections.create(
            name=collection_name,
            properties=properties,
            # vectorizer_config=wvc_config.Configure.Vectorizer.text2vec_contextionary(
            #     # Khi tự cung cấp vector, ta nên đặt vectorizer là 'none'.
            #     # Tuy nhiên, nếu bạn muốn Weaviate vector hóa các trường có cờ vectorize=True,
            #     # bạn cần chỉ định một vectorizer ở đây (ví dụ: text2vec-transformers).
            #     # Để nhất quán với code ingest hiện tại (tự cung cấp vector), ta dùng 'none'.
            #     vectorize_collection_name=False
            # ),
            vectorizer_config=wvc_config.Configure.Vectorizer.none(),
            vector_index_config=wvc_config.Configure.VectorIndex.hnsw(
                distance_metric=wvc_config.VectorDistances.COSINE
            ),

        )
        logger.info(f"✅ Đã tạo schema cho collection '{collection_name}' thành công.")
    except WeaviateQueryException as e:
        logger.error(f"❌ Lỗi khi tạo schema cho collection '{collection_name}': {e}")
        raise

def _ingest_chunks_with_native_batching(client: weaviate.WeaviateClient, collection_name: str, chunks: List[Document], embeddings_model):
    """Sử dụng API batch gốc của Weaviate, an toàn và hiệu suất cao."""
    logger.info(f"🚀 Bắt đầu quá trình ingestion cho {len(chunks)} chunks...")

    texts_to_embed = [chunk.page_content for chunk in chunks]

    logger.info(f"🧠 Đang tạo embeddings cho {len(texts_to_embed)} chunks...")
    start_embed_time = time.time()
    chunk_vectors = embeddings_model.embed_documents(texts_to_embed)
    logger.info(f"⏱️  Thời gian tạo embedding: {time.time() - start_embed_time:.2f} giây.")

    # 3. CẢI TIẾN: Đảm bảo chỉ ingest các thuộc tính hợp lệ
    valid_property_names = {prop["name"] for prop in WEAVIATE_SCHEMA_CONFIG}
    valid_property_names.add("text") # Thêm trường 'text'

    with client.batch.dynamic() as batch:
        for i, chunk in enumerate(chunks):
            if not isinstance(chunk,Document) or not hasattr(chunk, 'id') or not chunk.id:
                logger.warning(f"Bỏ qua chunk ở vị trí {i} do không hợp lệ (sai type hoặc thiếu ID).")
                continue

            properties = {"text": chunk.page_content}
            # Lọc metadata để chỉ giữ lại các key hợp lệ đã định nghĩa trong schema
            filtered_metadata = {
                k: v for k, v in chunk.metadata.items() if k in valid_property_names
            }
            properties.update(filtered_metadata)

            batch.add_object(
                collection=collection_name,
                properties=properties,
                uuid=chunk.id,
                vector=chunk_vectors[i]
            )

    logger.info(f"✅ Batching hoàn tất. Đã gửi {len(chunks)} objects.")
    if batch.number_errors > 0:
        logger.error(f"❌ Có {batch.number_errors} lỗi xảy ra trong quá trình batching.")
        # Log ra 5 lỗi đầu tiên để dễ gỡ lỗi
        for i, error_msg in enumerate(batch.errors):
            if i >= 5: break
            logger.error(f"  - Lỗi {i+1}: {error_msg}")

def build_and_load_vectorstore(
    embeddings,
    collection_name: str,
    weaviate_client: weaviate.WeaviateClient,
    chunks: Optional[List[Document]] = None
) -> Optional[WeaviateVectorStore]:
    """
    Hàm điều phối: đảm bảo schema tồn tại, ingest dữ liệu nếu có,
    và cuối cùng trả về một đối tượng WeaviateVectorStore của LangChain.
    """
    if not weaviate_client or not weaviate_client.is_connected():
        logger.error("🔸 Weaviate client không được cung cấp hoặc không kết nối.")
        return None
    if not embeddings:
        logger.error("🔸 Model embedding không được cung cấp.")
        return None

    try:
        # Bước 1: Đảm bảo Schema tồn tại với cấu trúc đúng từ cấu hình
        _create_weaviate_schema_if_not_exists(weaviate_client, collection_name)

        # Bước 2: Ingest dữ liệu nếu `chunks` được cung cấp
        if chunks:
            logger.info(f"🔸 Có {len(chunks)} chunks được cung cấp để ingest.")

            # Tiền xử lý: Serialize các metadata phức tạp
            processed_chunks = filter_and_serialize_complex_metadata(chunks)

            # Ingest bằng phương pháp hiệu suất cao và an toàn
            _ingest_chunks_with_native_batching(
                client=weaviate_client,
                collection_name=collection_name,
                chunks=processed_chunks,
                embeddings_model=embeddings
            )
        else:
            logger.info("🔸 Không có chunks nào, chỉ tải vector store đã có.")

        # Bước 3: Tạo và trả về đối tượng wrapper của LangChain
        logger.info(f"🔸 Đang tạo LangChain wrapper cho collection '{collection_name}'...")

        # Tự động lấy danh sách metadata từ cấu hình
        all_metadata_fields = [prop["name"] for prop in WEAVIATE_SCHEMA_CONFIG]

        # 4. CẬP NHẬT: Khởi tạo WeaviateVectorStore theo API mới
        vectorstore = WeaviateVectorStore(
            client=weaviate_client,
            index_name=collection_name,
            text_key="text", # Phải khớp với tên trường trong schema
            attributes=all_metadata_fields # Quan trọng: để LangChain biết lấy các trường này
        )

        logger.info("✅ Vector store đã sẵn sàng để sử dụng.")
        return vectorstore

    except Exception as e:
        logger.error(f"❌ Lỗi nghiêm trọng trong quá trình xây dựng/tải vector store: {e}", exc_info=True)
        return None
# End new

def create_or_load_vectorstore(embeddings, weaviate_url, collection_name, weaviate_client, chunks=None):
    vectorstore = None

    if not embeddings:
        logger.error("🔸Không có model embedding để tạo/tải vector store.")
        return None

    logger.info(f"🔸Truy cập Weaviate tại: {weaviate_url} với collection: {collection_name}")

    try:
        # Kết nối tới Weaviate
        client = weaviate_client
        if not client:
            logger.error("🔸Không thể kết nối tới Weaviate.")
            return None
        # Tên collection cần kiểm tra
        collection_name = config.WEAVIATE_COLLECTION_NAME

        # Kiểm tra xem collection có tồn tại không
        collection_exists = client.collections.exists(collection_name)

        logger.info(f"Collection {collection_name} exists: {collection_exists}")

        if chunks is not None and not collection_exists:
            logger.info(f"🔸Tạo Weaviate collection mới từ {len(chunks)} chunks...")

            # Kiểm tra mẫu dữ liệu đầu tiên
            logger.info(f"🔸Chunk đầu tiên:\n{chunks[0].metadata}")
            logger.info(f"🔸Nội dung:\n{chunks[0].page_content[:500]}...")

            # Lọc metadata để đảm bảo tương thích với Weaviate
            chunks = filter_and_serialize_complex_metadata(chunks)
            logger.info(f"🔸Metadata chunk đầu tiên sau khi lọc/serialize:\n{chunks[0].metadata}")

             # KIỂM TRA TYPE
            if chunks:
                logger.info(f"Type của chunk đầu tiên: {type(chunks[0])}")
                # Kiểm tra xem có phải là langchain Document không
                from langchain_core.documents import Document as LangchainDocument
                is_langchain_doc = isinstance(chunks[0], LangchainDocument)
                logger.info(f"Chunk đầu tiên có phải là langchain_core.documents.Document không? {is_langchain_doc}")
                if not is_langchain_doc:
                    logger.error("!!! LỖI NGHIÊM TRỌNG: Chunks không phải là instance của langchain_core.documents.Document")
                    # In ra các attribute của object để xem nó là gì
                    try:
                        logger.error(f"Attributes của chunk[0]: {dir(chunks[0])}")
                        if hasattr(chunks[0], "metadata"):
                             logger.error(f"Metadata của chunk[0] (nếu có): {chunks[0].metadata}")
                        if hasattr(chunks[0], "page_content"):
                             logger.error(f"Page_content của chunk[0] (nếu có): {chunks[0].page_content[:100]}")
                    except:
                        pass # Bỏ qua nếu không thể dir()
                    return None # Dừng ở đây nếu type sai



            # Tạo vectorstore
            max_batch_size = 1000  # Kích thước batch an toàn
            total_chunks = len(chunks)
            logger.info("🔸Đang nhúng dữ liệu...")

            # Tạo collection mới
            vectorstore = WeaviateVectorStore.from_documents(
                documents=chunks[:1],  # Khởi tạo với 1 tài liệu để tạo schema
                embedding=embeddings,
                client=client,
                index_name=collection_name,
                text_key="text",  # Tên trường văn bản trong tài liệu

                # by_texts=False # Nếu dùng ids thì không cần by_texts, nhưng để rõ ràng
            )

            # Thêm tài liệu theo batch
            for i in range(1, total_chunks, max_batch_size):
                end_idx = min(i + max_batch_size, total_chunks)
                current_batch = chunks[i:end_idx]
                logger.info(f"🔸Đang xử lý batch {i//max_batch_size + 1}/{(total_chunks-1)//max_batch_size + 1}: từ {i} đến {end_idx-1}")

                try:
                    vectorstore.add_documents(current_batch)
                    logger.info(f"🔸Đã thêm batch {i//max_batch_size + 1} thành công")
                except Exception as batch_error:
                    logger.error(f"🔸Lỗi khi xử lý batch từ {i} đến {end_idx-1}: {str(batch_error)}")
                    # Thử với batch nhỏ hơn
                    smaller_batch_size = max_batch_size // 2
                    if smaller_batch_size >= 10:
                        logger.info(f"🔸Thử lại với batch size nhỏ hơn: {smaller_batch_size}")
                        for j in range(i, end_idx, smaller_batch_size):
                            end_j = min(j + smaller_batch_size, end_idx)
                            smaller_batch = chunks[j:end_j]
                            try:
                                vectorstore.add_documents(smaller_batch)
                                logger.info(f"🔸Đã thêm batch nhỏ từ {j} đến {end_j-1} thành công")
                            except Exception as small_batch_error:
                                logger.error(f"🔸Vẫn lỗi với batch nhỏ hơn từ {j} đến {end_j-1}: {str(small_batch_error)}")
                    else:
                        logger.error(f"🔸Batch size đã quá nhỏ, không thể giảm thêm. Bỏ qua batch này.")
            logger.info(f"🔸Tạo Weaviate collection thành công: {collection_name}")

        elif collection_exists:
            logger.info(f"🔸Tải Weaviate collection đã tồn tại: {collection_name}")
            vectorstore = WeaviateVectorStore(
                client=client,
                index_name=collection_name,
                embedding=embeddings,
                text_key="text",
                attributes=[ # Liệt kê TẤT CẢ các metadata bạn cần để retriever hoạt động
                    "nam_ban_hanh", "title", "source", "field", "loai_van_ban", "so_hieu",
                    "ten_van_ban", "ngay_ban_hanh_str", "co_quan_ban_hanh", "entity_type",
                    # Các trường serialize thành JSON cũng cần được liệt kê nếu muốn lấy về
                    "cross_references", "penalties"
                ]
            )
            logger.info("🔸Tải Weaviate collection thành công.")

        else:
            logger.error(f"🔸Collection '{collection_name}' không tồn tại và không có dữ liệu chunks để tạo mới.")
            return None

        logger.info("🔸Vectorstore sẵn sàng.")
        return vectorstore

    except Exception as e:
        if client:
            client.close()
            logger.info("🔸Đã đóng kết nối tới Weaviate.")
        logger.error(f"🔸Lỗi khi tạo/tải Weaviate vector store: {e}")
        return None


def get_google_llm(google_api_key):
    logger.info("🔸Đang khởi tạo LLM từ Google Generative AI...")
    if not google_api_key:
        logger.error("🔸Google API Key không được cung cấp.")
        return None
    try:
        def create_chat_google():
            return ChatGoogleGenerativeAI(
                model="gemini-2.5-flash-preview-05-20",
                google_api_key=google_api_key,
                temperature=0.0, # Điều chỉnh nhiệt độ nếu cần, 0.1-0.3 thường tốt cho RAG
                safety_settings={                 },
            )

        llm = create_chat_google()

        logger.info("🔸Khởi tạo Google Generative AI LLM thành công.")
        return llm
    except Exception as e:
        logger.error(f"🔸Lỗi khi khởi tạo Google Generative AI LLM: {e}")
        return None

def format_chat_history(chat_history):
    if not chat_history:
        return []
    formatted_history = []
    for message in chat_history:
        if isinstance(message, tuple) and len(message) == 2:
            human_msg, ai_msg = message
            formatted_history.append(HumanMessage(content=human_msg))
            formatted_history.append(AIMessage(content=ai_msg))
    return formatted_history


def create_qa_chain(llm, vectorstore, retriever, process_input_llm=None):
    if not llm or not vectorstore:
        logger.error("🔸Thiếu LLM hoặc Vector Store để tạo QA Chain.")
        return None

    try:
        logger.info("🔸Bắt đầu tạo ConversationalRetrievalChain...")

        # ----- PROMPTS -----
        generic_prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_templete.GENERAL_PROMPT),
            ("human", "{input}")
        ])

        condense_prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_templete.CONDENSE_QUESTION_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_templete.SYSTEM_PROMPT),
            ("human", prompt_templete.QA_PROMPT_TEMPLATE)
        ])

        # ----- TẠO CHAIN XỬ LÝ TÀI LIỆU -----
        document_chain = create_stuff_documents_chain(
            llm=llm,
            prompt=qa_prompt,
            document_variable_name="context",
            output_parser=StrOutputParser()
        )

        _retriever = retriever or vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 10, "score_threshold": 0.5} # Cấu hình retriever mặc định của bạn
        )

        # history_aware_retriever sẽ nhận {input: câu hỏi đã xử lý bởi Groq, chat_history: ...}
        history_aware_retriever = create_history_aware_retriever(
            llm=llm,
            retriever=_retriever,
            prompt=condense_prompt,
        )

        # legal_chain_retrieval_part sẽ xử lý câu hỏi đã được process_with_groq
        legal_chain_retrieval_part = create_retrieval_chain(
            history_aware_retriever, # Retriever này sẽ nhận câu hỏi đã được xử lý
            document_chain
        )

        def process_question_for_legal_chain(input_dict: Dict):
            # input_dict ở đây sẽ là {"input": cleaned_question, "chat_history": ...}
            # từ RunnablePassthrough trong router
            cleaned_question = input_dict.get("input", "")
            question_for_chain = utils.process_with_groq(process_input_llm, cleaned_question) # Gọi hàm của bạn
            logger.info(f"[Legal Branch] Question processed by Groq: {question_for_chain}")
            # Trả về dict mới cho history_aware_retriever và legal_chain_retrieval_part
            return {"input": question_for_chain, "chat_history": input_dict.get("chat_history", [])}

        # Full legal chain
        legal_chain_full = (
            RunnableLambda(process_question_for_legal_chain)
            | legal_chain_retrieval_part
        ).with_config({"run_name": "LegalChainWithGroqProcessing"})

        # ----- CHAIN TỔNG QUÁT (GENERAL) -----
        general_chain = (
            RunnablePassthrough.assign( # Giữ lại chat_history nếu cần, nhưng generic_prompt không dùng
                input_for_prompt=lambda x: x["input"]
            )
            | {
                "answer": generic_prompt | llm | StrOutputParser(),
                "context": lambda x: [] # Trả về context rỗng
              }
        ).with_config({"run_name": "GeneralChain"})


        def get_route_key_from_input_dict(input_dict_for_router: Dict) -> str:
            question_for_routing = input_dict_for_router.get("input", "")
            logger.info(f"DEBUG get_route_key: question_for_routing NHẬN ĐƯỢC = '{question_for_routing}'")
            # << Thêm toàn bộ các log debug bên trong route_logic như đã thảo luận ở lần trước >>
            route = route_logic({"input": question_for_routing})
            logger.info(f"DEBUG get_route_key: route_logic TRẢ VỀ = '{route}' CHO INPUT = '{question_for_routing}'")
            return route

        qa_router_chain = router_as_runnable( # Đảm bảo router_as_runnable của bạn hoạt động đúng
            routes={
                "general": general_chain,
                "legal": legal_chain_full # Sử dụng legal_chain đã bao gồm xử lý Groq
            },
            get_key=RunnableLambda(get_route_key_from_input_dict), # Truyền cleaned_question vào route_logic
            default=general_chain # Hoặc legal_chain tùy theo hành vi mặc định bạn muốn
        ).with_config({"run_name": "MainQARouter"})

        logger.info("🔸Đã tạo thành công QA Router Chain.")
        return qa_router_chain

    except Exception as e:
        logger.error(f"🔸Lỗi khi tạo QA Chain: {e}", exc_info=True)
        return None