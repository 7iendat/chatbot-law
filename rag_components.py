# rag_components.py
import os

import uuid
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.vectorstores import Chroma
import config
import prompt_templete
from langchain.memory.chat_message_histories import RedisChatMessageHistory
# from langchain.chains import RetrievalQA
from langchain.chains import LLMChain,ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
# from langchain.schema.runnable.config import RunnableConfig
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
# from custom_output_parser import CustomOutputParser
from utils.utils import WrappedLLMChain
from langchain_core.runnables import RunnableLambda

import utils.utils as utils

# Hàm get_huggingface_embeddings giữ nguyên
def get_huggingface_embeddings(model_name, device='cpu'):
    # ... (code giữ nguyên từ trước) ...
    print(f"Đang khởi tạo model embedding: {model_name} trên thiết bị {device}...")
    model_kwargs = {'device': device}
    encode_kwargs = {'normalize_embeddings': True}
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        print("Khởi tạo model embedding thành công.")
        return embeddings
    except Exception as e:
        print(f"Lỗi khi khởi tạo model embedding: {e}")
        return None

# Hàm này sẽ được gọi bởi cả build script và API
def create_or_load_chroma_vectorstore(embeddings, persist_directory, collection_name, chunks=None):
    """
    Tạo ChromaDB vector store nếu chunks được cung cấp và chưa tồn tại,
    hoặc tải nếu đã tồn tại.
    """
    vectorstore = None

    if not embeddings:
        print("=> Lỗi: Không có model embedding để tạo/tải vector store.")
        return None

    print(f"=> Kiểm tra/Truy cập ChromaDB tại: {persist_directory} với collection: {collection_name}")

    # Kiểm tra sự tồn tại của thư mục persist
    db_exists = os.path.exists(persist_directory) and os.listdir(persist_directory)


    if chunks is not None and not db_exists: # Chỉ tạo mới nếu có chunks và DB chưa tồn tại
        print(f"Tạo ChromaDB mới từ {len(chunks)} chunks...")
        try:
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                collection_name=collection_name,
                persist_directory=persist_directory
            )
            vectorstore.persist()
            print(f"=> Tạo và lưu ChromaDB thành công vào: {persist_directory}")
        except Exception as e:
            print(f"=> Lỗi khi tạo ChromaDB mới: {e}")
            return None
    elif db_exists: # Nếu DB tồn tại, chỉ tải
        print(f"Tải ChromaDB đã tồn tại từ: {persist_directory}")
        try:
            vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=embeddings,
                persist_directory=persist_directory
            )
            print("=> Tải ChromaDB thành công.")


        except Exception as e:
            print(f"Lỗi khi tải ChromaDB: {e}")
            return None
    else: # Trường hợp không có chunks và DB cũng không tồn tại
        print(f"Lỗi: Vector store tại '{persist_directory}' không tồn tại và không có dữ liệu chunks để tạo mới.")
        return None

    print("=> Vectorstore sẵn sàng.")

    return vectorstore

def get_groq_llm(groq_api_key, temperature=0.2, max_new_tokens=1024): # Bỏ repo_id
    """Khởi tạo LLM từ Groq."""
    print("Đang khởi tạo LLM từ Groq...")

    if not groq_api_key:
        print("Lỗi: Groq API Key không được cung cấp.")
        return None

    try:

        prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_templete.SYSTEM_PROMPT),
            ("user", "{input}")
        ])
        # Tạo LLM từ Groq
        llm = prompt | ChatGroq(
            groq_api_key=groq_api_key,
            temperature=temperature, # Điều chỉnh nhiệt độ nếu cần
            max_tokens=max_new_tokens,
            model_name=config.GROQ_MODEL_NAME, # Tham số này thay đổi thành 'max_tokens'
            # model_name="llama3-70b-8192", # << Mặc định, nhưng bạn có thể thử 'mixtral-8x7b-32768'
            #  context_length = 4096, # Thay đổi độ dài ngữ cảnh nếu cần
        )
        print("Khởi tạo Groq LLM thành công.")
        return llm
    except Exception as e:
        print(f"Lỗi khi khởi tạo Groq LLM: {e}")
        return None

# Hàm create_qa_chain giữ nguyên
def create_qa_chain(llm, vectorstore,  search_k=5 ,redis_instance=None,session_id=None):
    # ... (code giữ nguyên từ trước) ...
    if not llm or not vectorstore:
        print("Lỗi: Thiếu LLM hoặc Vector Store để tạo QA Chain.")
        return None
    if not session_id:
        session_id = str(uuid.uuid4())  # Tạo một session_id mới nếu không có
    # print("Đang tạo RetrievalQA Chain...")
    print("Đang tạo ConversationalRetrievalChain...")

    # query_gen_prompt = PromptTemplate.from_template(config.QUERY_GEN_PROMPT_TEMPLATE)

    # query_gen_chain = query_gen_prompt | llm | config.OUTPUT_PARSER
    # print("query gen chain initialized (RunnableSequence).")

    # C1: Sử dụng retriever mặc định của vectorstore
    # retriever = vectorstore.as_retriever(search_kwargs={"k": search_k})
    # print(f"Retriever sẽ lấy {search_k} chunks.")

    # C2: Sử dụng MultiQueryRetriever
    multi_retriever = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(search_kwargs={"k": search_k}),
        llm=llm,
    )
    # print("Đã tạo MultiQueryRetriever.")
    # ... (phần còn lại giữ nguyên) ...
    # Prompt tùy chọn
    generic_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "Bạn là một trợ lý ảo AI thân thiện, bạn tên là ***Angel***, nhiệt tình và thông minh, được thiết kế để trả lời các câu hỏi tổng quát "
            "từ người dùng, bao gồm cả các chủ đề như công nghệ, đời sống, sức khỏe, du lịch, học tập, v.v. "
            "Nếu câu hỏi vượt ngoài phạm vi kiến thức chuyên sâu, hãy trả lời một cách lịch sự và đề xuất người dùng tìm kiếm thêm thông tin."
        )),
        ("human", "{question}")
    ])

    condense_question_prompt = PromptTemplate(
        input_variables=["chat_history", "question"],
        template=prompt_templete.CONDENSE_QUESTION_PROMPT
    )

    message_history = RedisChatMessageHistory(
        url=redis_instance,
        session_id=session_id
    )

    # Memory buffer để lưu lịch sử chat
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        chat_memory=message_history,
        k = 4,
        return_messages=True,
        output_key="answer"
    )

    try:
        # qa_chain = RetrievalQA.from_chain_type(
        #     llm=llm,
        #     chain_type="stuff",
        #     retriever=retriever,
        #     chain_type_kwargs=qa_prompt,
        #     return_source_documents=True,
        #     output_parser=CustomOutputParser()
        # )
        # print("Tạo QA Chain thành công.")
        # return qa_chain
        generic_chain =  WrappedLLMChain(LLMChain(llm=llm, prompt=generic_prompt, output_key="answer").with_config({"run_name": "General Info"}))

        legal_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=multi_retriever,
            memory=memory,
            condense_question_prompt=condense_question_prompt,
            return_source_documents=True,
            output_key="answer"
        )

        # Chain trả về duy nhất "answer"
        wrapped_legal_chain = WrappedLLMChain(legal_chain).with_config({"run_name": "Legal QA"})

        route_logic_runnable = RunnableLambda(lambda input: utils.route_logic(input))
        # Router setup
        router = utils.router_as_runnable(
            routes={
                # "general": generic_chain.with_config({"run_name": "General Info"}),
                "general": generic_chain,
                "legal": wrapped_legal_chain
            },
            get_key=route_logic_runnable,
            default=generic_chain.with_config({"run_name": "Default"})
        )
        print("=> Tạo chain thành công với memory.")

        # print("Router chain initialized.")
        return router
    except Exception as e:
        print(f"Lỗi khi tạo QA Chain: {e}")
        return None