
import os
from langchain_huggingface import HuggingFaceEmbeddings
import json
# from langchain_chroma import Chroma
import config
import prompt_templete
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableSequence
import utils.utils as utils
from langchain_core.documents import Document
import logging
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores.utils import filter_complex_metadata
from typing import List, Dict, Any
from langchain_weaviate.vectorstores import WeaviateVectorStore




logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Hàm get_huggingface_embeddings giữ nguyên
def get_huggingface_embeddings(model_name: str, device: str = 'cpu'):
    """
    Hàm khởi tạo HuggingFaceEmbeddings với model chỉ định.

    Args:
        model_name (str): Tên model trên HuggingFace Hub.
        device (str, optional): 'cpu' hoặc 'cuda'. Defaults to 'cpu'.

    Returns:
        HuggingFaceEmbeddings: Instance đã khởi tạo thành công.

    Raises:
        Exception: Nếu khởi tạo thất bại.
    """
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
        logger.info("🔸 Khởi tạo model embedding thành công.")
        return embeddings
    except Exception as e:
        logger.error(f"🔸Lỗi khi khởi tạo model embedding: {e}")
        raise Exception(f"Khởi tạo model embedding thất bại: {str(e)}")

def create_or_load_vectorstore(embeddings, weaviate_url, collection_name, weaviate_client, chunks=None):
    """
    Tạo Weaviate vector store nếu chunks được cung cấp và chưa tồn tại,
    hoặc tải nếu đã tồn tại.
    """
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
                text_key="text",  # Tên trường văn bản trong tài liệu
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
    """Khởi tạo LLM từ Groq."""
    logger.info("🔸Đang khởi tạo LLM từ Groq...")

    if not groq_api_key:
        logger.error("🔸Groq API Key không được cung cấp.")
        return None

    try:

        prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_templete.SYSTEM_PROMPT),
            ("user", "{input}")
        ])
        # Tạo LLM từ Groq
        llm = prompt |  ChatGroq(
            groq_api_key=groq_api_key,
            temperature=temperature, # Điều chỉnh nhiệt độ nếu cần
            max_tokens=max_new_tokens,
            model_name=config.GROQ_MODEL_NAME, # Tham số này thay đổi thành 'max_tokens'
            # model_name="llama3-70b-8192", # << Mặc định, nhưng bạn có thể thử 'mixtral-8x7b-32768'
            #  context_length = 4096, # Thay đổi độ dài ngữ cảnh nếu cần
        )

        logger.info("🔸Khởi tạo Groq LLM thành công.")
        return llm
    except Exception as e:
        logger.error(f"🔸Lỗi khi khởi tạo Groq LLM: {e}")
        return None


def create_qa_chain(llm, vectorstore, retriever=None):
    if not llm or not vectorstore:
        logger.error("🔸Thiếu LLM hoặc Vector Store để tạo QA Chain.")
        return None

    logger.info("🔸Đang tạo ConversationalRetrievalChain...")

    try:
        # Prompt tổng quát cho câu hỏi không liên quan đến pháp luật
        generic_prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "Bạn là một trợ lý ảo AI thân thiện, bạn tên là ***Angel***, nhiệt tình và thông minh, được thiết kế để trả lời các câu hỏi tổng quát "
                "từ người dùng, bao gồm cả các chủ đề như công nghệ, đời sống, sức khỏe, du lịch, học tập, v.v. "
                "Nếu câu hỏi vượt ngoài phạm vi kiến thức chuyên sâu, hãy trả lời một cách lịch sự và đề xuất người dùng tìm kiếm thêm thông tin."
            )),
            ("human", "{question}")
        ])

        logger.info("🔸Đang tạo condense prompt ....")
        condense_prompt = ChatPromptTemplate.from_template(prompt_templete.CONDENSE_QUESTION_PROMPT)
        logger.info("🔸Đã tạo condense prompt thành công.")

        logger.info("🔸Đang tạo qa prompt ....")
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_templete.SYSTEM_PROMPT),
            ("human", prompt_templete.QA_PROMPT_TEMPLATE)
        ])
        logger.info("🔸Đã tạo qa prompt thành công.")

        # Khởi tạo ConversationalRetrievalChain cho câu hỏi pháp luật
        logger.info("🔸Đang tạo ConversationalRetrievalChain cho câu hỏi pháp luật...")
        legal_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever or vectorstore.as_retriever(),
            condense_question_prompt=condense_prompt,
            combine_docs_chain_kwargs={
                "prompt": qa_prompt
            },
            return_source_documents=True,
            output_key="answer",
            verbose=True
        )
        logger.info("🔸Đã tạo ConversationalRetrievalChain thành công.")

        # Sửa generic_chain để đảm bảo đầu ra là dictionary với key "answer"
        generic_chain = RunnableSequence(
            generic_prompt | llm | StrOutputParser() | (lambda x: {"answer": x})
        ).with_config({"run_name": "GeneralChain"})

        # Hàm định dạng metadata cho source_documents
        def format_metadata(docs: List[Document]) -> str:
            if not docs:
                logger.warning("🔸Không có tài liệu nào được cung cấp.")
                return "Không có metadata tài liệu."
            metadata_list = []
            for doc in docs:
                metadata = doc.metadata or {}
                penalty = metadata.get("penalty", "Không có thông tin mức phạt")
                source = metadata.get("source", "Không có thông tin nguồn")
                metadata_str = f"- Nguồn: {source}, Mức phạt: {penalty}"
                metadata_list.append(metadata_str)
            return "\n".join(metadata_list) or "Không có metadata tài liệu."

        # Hàm router
        def route_with_history(input: Any) -> Dict[str, Any]:
            logger.info(f"🔸Router input: {input} (type: {type(input)})")

            # Xử lý các loại đầu vào
            if isinstance(input, str):
                logger.warning(f"🔸String input received, converting to dict: {input}")
                input = {"question": input, "chat_history": []}
            elif isinstance(input, dict):
                if "question" not in input:
                    logger.error(f"🔸Dictionary input missing 'question' key: {input}")
                    raise ValueError("Input dictionary must contain 'question' key.")
            else:
                logger.error(f"🔸Invalid input type: {type(input)}, value: {input}")
                raise ValueError("Input must be a string or a dictionary with 'question' key.")

            route_key = utils.route_logic(input["question"])
            chain = {
                "general": generic_chain,
                "legal": legal_chain
            }.get(route_key, generic_chain)

            logger.info(f"🔸Selected chain: {route_key}")

            try:
                if route_key == "legal":
                    source_documents = retriever.invoke(input["question"]) if retriever else vectorstore.as_retriever().invoke(input["question"])
                    logger.info(f"Retrieved {len(source_documents)} source documents for question: {input['question']}")
                    chain_input = {
                        "question": input["question"],
                        "chat_history": input.get("chat_history", []),
                        "source_documents": format_metadata(source_documents)
                    }
                    result = chain.invoke(chain_input)
                    logger.info(f"🔸Raw legal_chain result: {result} (type: {type(result)})")
                else:
                    result = chain.invoke({"question": input["question"]})
                    logger.info(f"🔸Raw generic_chain result: {result} (type: {type(result)})")

                # Xử lý định dạng đầu ra
                if not isinstance(result, dict):
                    logger.warning(f"🔸Chain returned non-dict result: {result} (type: {type(result)})")
                    result = {"answer": str(result) if result else "Không thể xử lý câu hỏi. Vui lòng thử lại."}
                elif "answer" not in result:
                    logger.warning(f"🔸Chain result missing 'answer' key: {result}")
                    if "result" in result:
                        result = {"answer": str(result["result"])}
                    elif "text" in result:
                        result = {"answer": str(result["text"])}
                    else:
                        result = {"answer": "Không thể xử lý câu hỏi. Vui lòng thử lại."}

                # Xử lý chuỗi JSON nếu có
                if isinstance(result.get("answer"), str):
                    try:
                        parsed_answer = json.loads(result["answer"])
                        if isinstance(parsed_answer, dict) and "answer" in parsed_answer:
                            logger.warning(f"🔸Parsed JSON answer in chain result: {parsed_answer}")
                            result["answer"] = parsed_answer["answer"]
                    except json.JSONDecodeError:
                        pass

                logger.info(f"🔸Processed chain result: {result}")
                return result

            except Exception as chain_error:
                logger.error(f"🔸Chain invocation failed: {chain_error}")
                raise ValueError(f"Chain invocation failed: {str(chain_error)}")

        # Thiết lập router
        route_logic_runnable = RunnableLambda(lambda input: utils.route_logic(input))
        router = utils.router_as_runnable(
            routes={
                "general": generic_chain,
                "legal": legal_chain
            },
            get_key=route_logic_runnable,
            default=generic_chain.with_config({"run_name": "Default"})
        )
        logger.info("🔸Tạo chain thành công với router.")

        return router

    except Exception as e:
        logger.error(f"🔸Lỗi khi tạo QA Chain: {e}")
        return None