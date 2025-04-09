# rag_components.py
import os
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

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
        print("Lỗi: Không có model embedding để tạo/tải vector store.")
        return None

    print(f"Kiểm tra/Truy cập ChromaDB tại: {persist_directory} với collection: {collection_name}")

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
            print(f"Tạo và lưu ChromaDB thành công vào: {persist_directory}")
        except Exception as e:
            print(f"Lỗi khi tạo ChromaDB mới: {e}")
            return None
    elif db_exists: # Nếu DB tồn tại, chỉ tải
        print(f"Tải ChromaDB đã tồn tại từ: {persist_directory}")
        try:
            vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=embeddings,
                persist_directory=persist_directory
            )
            print("Tải ChromaDB thành công.")
        except Exception as e:
            print(f"Lỗi khi tải ChromaDB: {e}")
            return None
    else: # Trường hợp không có chunks và DB cũng không tồn tại
        print(f"Lỗi: Vector store tại '{persist_directory}' không tồn tại và không có dữ liệu chunks để tạo mới.")
        return None

    return vectorstore

# Hàm get_huggingface_endpoint_llm giữ nguyên
def get_huggingface_endpoint_llm(repo_id, hf_api_token, temperature=0.2, max_new_tokens=1024):
    # ... (code giữ nguyên từ trước) ...
    print(f"Đang khởi tạo LLM từ Hugging Face Endpoint: {repo_id}")
    if not hf_api_token:
        print("Lỗi: Hugging Face API Token không được cung cấp.")
        return None
    try:
        llm = HuggingFaceEndpoint(
            repo_id=repo_id,
            huggingfacehub_api_token=hf_api_token,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            # --- THÊM DÒNG NÀY ---
            task="text-generation", # Chỉ định rõ task cho model
            # --------------------
            # Thêm các tham số khác nếu cần
            # top_k=10,
            # repetition_penalty=1.1
        )
        print("Khởi tạo LLM endpoint thành công.")
        return llm
    except Exception as e:
        print(f"Lỗi khi khởi tạo LLM endpoint: {e}")
        return None

# Hàm create_qa_chain giữ nguyên
def create_qa_chain(llm, vectorstore, search_k=5, prompt_template_str=None):
    # ... (code giữ nguyên từ trước) ...
    if not llm or not vectorstore:
        print("Lỗi: Thiếu LLM hoặc Vector Store để tạo QA Chain.")
        return None
    print("Đang tạo RetrievalQA Chain...")
    retriever = vectorstore.as_retriever(search_kwargs={"k": search_k})
    print(f"Retriever sẽ lấy {search_k} chunks.")
    # ... (phần còn lại giữ nguyên) ...
    chain_type_kwargs = {}
    if prompt_template_str:
        QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt_template_str)
        chain_type_kwargs = {"prompt": QA_CHAIN_PROMPT}
        print("Sử dụng prompt template tùy chỉnh.")
    else:
        print("Sử dụng prompt template mặc định của Langchain.")

    try:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs=chain_type_kwargs,
            return_source_documents=True
        )
        print("Tạo QA Chain thành công.")
        return qa_chain
    except Exception as e:
        print(f"Lỗi khi tạo QA Chain: {e}")
        return None