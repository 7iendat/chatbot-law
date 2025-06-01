
import os
from langchain_huggingface import HuggingFaceEmbeddings
import json
# from langchain_chroma import Chroma
import config
import prompt_templete
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
import utils.utils as utils
from langchain_core.documents import Document
import logging
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores.utils import filter_complex_metadata
from typing import List, Dict
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from utils.route_logic import route_logic
# from langchain_core.prompts import PromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage



logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

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
        collection_name = "LawDocuments"

        # Kiểm tra xem collection có tồn tại không
        collection_exists = client.collections.exists(collection_name)

        print(f"Collection {collection_name} exists: {collection_exists}")

        if chunks is not None and not collection_exists:
            logger.info(f"🔸Tạo Weaviate collection mới từ {len(chunks)} chunks...")

            # Kiểm tra mẫu dữ liệu đầu tiên
            logger.info(f"🔸Chunk đầu tiên:\n{chunks[0].metadata}")
            logger.info(f"🔸Nội dung:\n{chunks[0].page_content[:500]}...")

            # Lọc metadata để đảm bảo tương thích với Weaviate
            chunks = filter_complex_metadata(chunks)

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
                attributes=[
                    'year',
                    'source',
                    'field',
                    'penalty',
                    'entity_type',
                    "so_hieu",
                    "loai_van_ban",
                    "ngay_ban_hanh",
                    "nam_ban_hanh"
                ]  # Tên trường văn bản trong tài liệu
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

# Hàm này sẽ được gọi bởi cả build script và API
# def create_or_load_chroma_vectorstore(embeddings, persist_directory, collection_name, chunks=None):
#     """
#     Tạo ChromaDB vector store nếu chunks được cung cấp và chưa tồn tại,
#     hoặc tải nếu đã tồn tại.
#     """
#     vectorstore = None

#     if not embeddings:
#         logger.error("🔸Không có model embedding để tạo/tải vector store.")
#         return None

#     logger.info(f"🔸Truy cập ChromaDB tại: {persist_directory} với collection: {collection_name}")

#     # Kiểm tra sự tồn tại của thư mục persist
#     db_exists = os.path.exists(persist_directory) and os.listdir(persist_directory)


#     if chunks is not None and not db_exists: # Chỉ tạo mới nếu có chunks và DB chưa tồn tại
#         logger.info(f"🔸Tạo ChromaDB mới từ {len(chunks)} chunks...")
#         try:
#             # Kiểm tra mẫu dữ liệu đầu tiên
#             logger.info(f"🔸Chunk đầu tiên:\n{chunks[0].metadata}")
#             logger.info(f"🔸Nội dung:\n{chunks[0].page_content[:500]}...")

#             chunks = filter_complex_metadata(chunks)
#             # Kích thước batch tối đa an toàn
#             max_batch_size = 1000  # Thấp hơn giới hạn 5461 để đảm bảo an toàn
#             total_chunks = len(chunks)
#             logger.info("🔸Đang nhúng dữ liệu...")
#             # Tạo vectorstore từ chunks
#             vectorstore = Chroma(
#                 embedding_function=embeddings,
#                 collection_name=collection_name,
#                 persist_directory=persist_directory
#             )
#             # Thêm tài liệu theo từng batch
#             for i in range(0, total_chunks, max_batch_size):
#                 end_idx = min(i + max_batch_size, total_chunks)
#                 current_batch = chunks[i:end_idx]

#                 logger.info(f"🔸Đang xử lý batch {i//max_batch_size + 1}/{(total_chunks-1)//max_batch_size + 1}: từ {i} đến {end_idx-1}")

#                 try:
#                     # Thêm batch hiện tại vào vectorstore
#                     vectorstore.add_documents(current_batch)

#                     logger.info(f"🔸Đã thêm và lưu batch {i//max_batch_size + 1} thành công")

#                 except Exception as batch_error:
#                     logger.error(f"🔸Lỗi khi xử lý batch từ {i} đến {end_idx-1}: {str(batch_error)}")

#                     # Thử với batch nhỏ hơn nếu có lỗi
#                     smaller_batch_size = max_batch_size // 2
#                     if smaller_batch_size >= 10:  # Đảm bảo batch không quá nhỏ
#                         logger.info(f"🔸Thử lại với batch size nhỏ hơn: {smaller_batch_size}")

#                         for j in range(i, end_idx, smaller_batch_size):
#                             end_j = min(j + smaller_batch_size, end_idx)
#                             smaller_batch = chunks[j:end_j]

#                             try:
#                                 vectorstore.add_documents(smaller_batch)
#                                 logger.info(f"🔸Đã thêm và lưu batch nhỏ từ {j} đến {end_j-1} thành công")
#                             except Exception as small_batch_error:
#                                 logger.error(f"🔸Vẫn lỗi với batch nhỏ hơn từ {j} đến {end_j-1}: {str(small_batch_error)}")
#                     else:
#                         logger.error(f"🔸Batch size đã quá nhỏ, không thể giảm thêm. Bỏ qua batch này.")
#             logger.info(f"🔸Tạo và lưu ChromaDB thành công vào: {persist_directory}")
#         except Exception as e:
#             logger.error(f"🔸Lỗi khi tạo ChromaDB mới: {e}")
#             return None
#     elif db_exists: # Nếu DB tồn tại, chỉ tải
#         logger.info(f"🔸Loading ChromaDB đã tồn tại từ: {persist_directory}")
#         try:
#             vectorstore = Chroma(
#                 collection_name=collection_name,
#                 embedding_function=embeddings,
#                 persist_directory=persist_directory
#             )
#             logger.info("🔸Loaded ChromaDB thành công.")


#         except Exception as e:
#             logger.error(f"🔸Lỗi khi tải ChromaDB: {e}")
#             return None
#     else: # Trường hợp không có chunks và DB cũng không tồn tại
#         logger.error(f"🔸Vector store tại '{persist_directory}' không tồn tại và không có dữ liệu chunks để tạo mới.")
#         return None

#     logger.info("🔸Vectorstore sẵn sàng.")

#     return vectorstore

def get_groq_llm(groq_api_key, temperature=0.2, max_new_tokens=1024): # Bỏ repo_id
    logger.info("🔸Đang khởi tạo LLM từ Groq...")

    if not groq_api_key:
        logger.error("🔸Groq API Key không được cung cấp.")
        return None

    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_templete.SYSTEM_PROMPT),
            ("human", "{input}")
        ])



        def create_chat_groq():
            return ChatGroq(
            groq_api_key=groq_api_key,
            temperature=temperature, # Điều chỉnh nhiệt độ nếu cần
            max_tokens=max_new_tokens,
            model_name=config.GROQ_MODEL_NAME, # Tham số này thay đổi thành 'max_tokens'
            # model_name="llama3-70b-8192", # << Mặc định, nhưng bạn có thể thử 'mixtral-8x7b-32768'
            #  context_length = 4096, # Thay đổi độ dài ngữ cảnh nếu cần
        )

        llm = create_chat_groq()

        llm_chain = prompt | llm

        logger.info("🔸Khởi tạo Groq LLM thành công.")
        return llm_chain
    except Exception as e:
        logger.error(f"🔸Lỗi khi khởi tạo Groq LLM: {e}")
        return None

def get_google_llm(google_api_key):
    logger.info("🔸Đang khởi tạo LLM từ Google Generative AI...")
    if not google_api_key:
        logger.error("🔸Google API Key không được cung cấp.")
        return None
    try:
        # System prompt đã được định nghĩa trong prompt_templete.SYSTEM_PROMPT
        # prompt = ChatPromptTemplate.from_messages([
        #     ("system", prompt_templete.SYSTEM_PROMPT),
        #     ("human", "{input}")
        # ])

        def create_chat_google():
            return ChatGoogleGenerativeAI(
                model="gemini-2.5-flash-preview-04-17", # Hoặc "gemini-pro" nếu 1.5 Pro chưa ổn định/có vấn đề
                google_api_key=google_api_key,
                temperature=0.1, # Điều chỉnh nhiệt độ nếu cần, 0.1-0.3 thường tốt cho RAG
                # convert_system_message_to_human=True # Có thể cần cho một số prompt phức tạp, thử nghiệm
                safety_settings={ # Tùy chỉnh cài đặt an toàn nếu cần
                    # HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
                    # HarmCategory.HARM_CATEGORY_DEROGATORY: HarmBlockThreshold.BLOCK_NONE,
                    # ... (thêm các category khác)
                    # Tham khảo: from google.generativeai.types import HarmCategory, HarmBlockThreshold
                }
            )

        llm = create_chat_google()
        # llm_chain = prompt | llm

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


        # ----- TẠO CHAIN TỔNG QUÁT -----
        # generic_chain = RunnableSequence(
        #     generic_prompt
        #     | llm
        #     | RunnableLambda(lambda x: {"answer": x})
        # ).with_config({"run_name": "GeneralChain"})

        # general_chain = (
        #     generic_prompt
        #     | llm # Dùng llm_model gốc
        #     | StrOutputParser()
        #     | RunnableLambda(lambda text_answer: {"answer": text_answer, "context": []}) # Thêm context rỗng cho đồng nhất
        # ).with_config({"run_name": "GeneralChain"})



        # Full legal chain
        # Nó sẽ nhận input là {"input": cleaned_question, "chat_history": ...}
        # sau đó process_question_for_legal_chain sẽ biến đổi "input"
        # rồi đưa vào legal_chain_retrieval_part
        legal_chain_full = (
            RunnableLambda(process_question_for_legal_chain)
            | legal_chain_retrieval_part
        ).with_config({"run_name": "LegalChainWithGroqProcessing"})

        # ----- CHAIN TỔNG QUÁT (GENERAL) -----
        # general_chain sẽ nhận trực tiếp {input: cleaned_question, chat_history: ...}
        # và chỉ sử dụng "input" cho generic_prompt
        general_chain = (
            RunnablePassthrough.assign( # Giữ lại chat_history nếu cần, nhưng generic_prompt không dùng
                input_for_prompt=lambda x: x["input"]
            )
            | {
                "answer": generic_prompt | llm | StrOutputParser(),
                "context": lambda x: [] # Trả về context rỗng
              }
        ).with_config({"run_name": "GeneralChain"})

        # ----- FORMAT METADATA (nếu cần hiển thị thêm) -----
        def format_metadata(docs: List[Document]) -> str:
            if not docs:
                return "Không có metadata tài liệu."
            return "\n".join(
                f"- Nguồn: {doc.metadata.get('source', 'Không rõ')}, Mức phạt: {doc.metadata.get('penalty', 'Không có')}"
                for doc in docs
            )
        def get_route_key(input_dict):
            return route_logic(input_dict)
        # ----- ROUTER -----
        # router = utils.router_as_runnable(
        #     routes={
        #         "general": general_chain,
        #         "legal": legal_chain
        #     },
        #     get_key=RunnableLambda(get_route_key),
        #     default=general_chain
        # ).with_config({"run_name": "QAChainRouter"})

        # logger.info("🔸Đã tạo thành công QA Router Chain.")
        # return router

        # ----- ROUTER -----
        # Input cho router sẽ là: {"input": cleaned_question, "chat_history": ...}
        # get_route_key sẽ chỉ dùng "input" (cleaned_question) để quyết định nhánh
        def get_route_key_from_input_dict(input_dict_for_router: Dict) -> str:
            question_for_routing = input_dict_for_router.get("input", "")
            logger.info(f"DEBUG get_route_key: question_for_routing NHẬN ĐƯỢC = '{question_for_routing}'")
            # << Thêm toàn bộ các log debug bên trong route_logic như đã thảo luận ở lần trước >>
            route = route_logic({"input": question_for_routing})
            logger.info(f"DEBUG get_route_key: route_logic TRẢ VỀ = '{route}' CHO INPUT = '{question_for_routing}'")
            return route


        qa_router_chain = utils.router_as_runnable( # Đảm bảo router_as_runnable của bạn hoạt động đúng
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