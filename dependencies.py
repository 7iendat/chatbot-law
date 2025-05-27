from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
import os
from dotenv import load_dotenv
from db.mongoDB import user_collection, blacklist_collection
import torch
import rag_components
from schemas.chat import AppState

from config import SECRET_KEY, ALGORITHM,EMBEDDING_MODEL_NAME, LLM_TEMPERATURE, LLM_MAX_NEW_TOKENS, WEAVIATE_COLLECTION_NAME, WEAVIATE_URL, SEARCH_K
from utils.utils import load_legal_dictionary
from groq import Groq
from typing import Annotated
from schemas.user import UserOut, UserRole
from fastapi import status
from datetime import datetime, timezone
import redis
# from time_priority_retriever import WeaviateHybridRetriever

import logging
from db.weaviateDB import connect_to_weaviate
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Bearer token security scheme
bearer_scheme = HTTPBearer()

def get_app_state(request: Request):
    # Kiểm tra xem app_state có thực sự tồn tại trên request.app.state không
    if not hasattr(request.app.state, 'app_state'):
        # Điều này chỉ xảy ra nếu lifespan không được gọi hoặc có lỗi trong lifespan
        print("Error in get_app_state: request.app.state.app_state is not set!") # Thêm log
        raise RuntimeError("Application state ('app_state') not found. Initialization failed?")

    # print(f"get_app_state: Returning app_state object. Type: {type(request.app.state.app_state)}") # Thêm log
    return request.app.state.app_state

def initialize_redis_client(): # Hàm riêng để khởi tạo và kiểm tra Redis
    redis_url = os.environ.get("REDIS_URL")
    if not redis_url:
        logger.error("🔸[Redis] REDIS_URL environment variable not set.")
        raise ValueError("REDIS_URL is not configured.") # Ném lỗi cấu hình

    try:
        logger.info(f"🔸[Redis] Attempting to connect to Redis at {redis_url}...")
        client = redis.Redis.from_url(redis_url, socket_connect_timeout=5, socket_timeout=5) # Thêm timeout
        client.ping()
        logger.info("🔸[Redis] Connected successfully and pinged.")
        return client
    except redis.exceptions.ConnectionError as e:
        logger.error(f"🔸[Redis] Connection failed for URL '{redis_url}': {e}")
        raise ConnectionError(f"Failed to connect to Redis: {e}") # Ném lỗi kết nối
    except Exception as e:
        logger.error(f"🔸[Redis] Error initializing Redis from URL '{redis_url}': {e}")
        raise RuntimeError(f"Error initializing Redis: {e}") # Ném lỗi chung

def initialize_api_components(app_state: AppState):
    """Khởi tạo các thành phần cần thiết cho API """
    logger.info("🔸Bắt đầu Khởi tạo API Components")

    load_dotenv()
    # --- Kiểm tra kết nối tới Redis ---
    app_state.process_input_llm = Groq(api_key=os.environ.get("PRE_PROCESS_INPUT_KEY"))
    try:
        app_state.redis = initialize_redis_client() # Gọi hàm khởi tạo redis
    except Exception as e:
        # Log lỗi ở đây đã được thực hiện trong initialize_redis_client
        # Chỉ cần ném lại lỗi để lifespan biết
        logger.error(f"☠️ LỖI NGHIÊM TRỌNG khi khởi tạo Redis trong initialize_api_components: {e}")
        raise # QUAN TRỌNG: Ném lại lỗi
    app_state.dict = load_legal_dictionary('./data/dictionary/legal_terms.json')
    app_state.weaviateDB = connect_to_weaviate()
    # --- Kiểm tra kết nối tới MongoDB ---
    if user_collection is  None or app_state.weaviateDB is None:
        logger.error("🔸Lỗi kết nối tới MongoDB hoặc Weaviate.")
        raise HTTPException(status_code=500, detail="Lỗi kết nối tới database.")

    app_state.groq_api_key = os.environ.get("GROQ_API_KEY") # Lấy key Groq

    if not app_state.groq_api_key:
        logger.error("🔸GROQ API Key không được cung cấp.")
        raise HTTPException(status_code=500, detail="Missing GROQ API Key")

    # app_state.google_api_key = os.environ.get("GOOGLE_API_KEY")
    # if not app_state.google_api_key:
    #     logger.error("🔸GG API Key không được cung cấp.")
    #     raise HTTPException(status_code=500, detail="Missing GG API Key")

    app_state.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"🔸Sử dụng thiết bị: {app_state.device}")

    # 1. Tải Embedding Model (giữ nguyên)
    print(f"Đang tải Embedding Model...")
    app_state.embeddings = rag_components.get_huggingface_embeddings(
        EMBEDDING_MODEL_NAME, app_state.device
    )
    if not app_state.embeddings:
        raise HTTPException(status_code=500, detail="Failed to load embedding model")

    # 2. Tải Vector Store (ChromaDB) (giữ nguyên)
    print(f"Đang tải Vector Store...")
    # app_state.vectorstore = rag_components.create_or_load_chroma_vectorstore(
    #     embeddings=app_state["embeddings"],
    #     persist_directory=CHROMA_PERSIST_DIR,
    #     collection_name=CHROMA_COLLECTION_NAME,
    #     chunks=None # Không cung cấp chunks khi khởi tạo API
    # )

    app_state.vectorstore = rag_components.create_or_load_vectorstore(
        embeddings=app_state.embeddings,
        weaviate_url=WEAVIATE_URL,
        collection_name=WEAVIATE_COLLECTION_NAME,
        weaviate_client=app_state.weaviateDB,
        chunks=None,
    )

    if not app_state.vectorstore:
         raise HTTPException(status_code=500, detail="Failed to load or create Vectorstore")

    # 3. Tải LLM (thay đổi để dùng Groq)
    logger.info(f"🔸Đang tải LLM (Groq)...")
    llm = rag_components.get_groq_llm(
        app_state.groq_api_key,
        temperature=LLM_TEMPERATURE,
        max_new_tokens=LLM_MAX_NEW_TOKENS
    )

    # llm = rag_components.get_google_llm(app_state.google_api_key)
    app_state.llm = llm
    logger.info(f"🔸Tải LLM (Groq) thanh cong")


    if not app_state.llm:
        raise HTTPException(status_code=500, detail="Failed to load LLM")

    # 4. Tạo retriever (giữ nguyên)
    logger.info(f"🔸Đang tạo retriever...")
    # app_state.retriever = create_retriever(
    #     app_state.vectorstore,
    #     app_state["embeddings"],
    #     llm,
    # )
    # app_state.retriever = create_retriever(
    #     app_state.vectorstore,
    #     app_state["embeddings"],
    #     llm,
    #     config={
    #         "recent_years": 5,
    #         "primary_docs": 4,
    #         "total_docs": 6,
    #         "content_min_chars": 1500,
    #         "use_llm_paraphrase": False
    #     }
    # )

    app_state.retriever = app_state.vectorstore.as_retriever(
        search_type="similarity_score_threshold",  # tìm kiếm dựa trên ngưỡng điểm tương đồng
        search_kwargs={
            "k": 10,          # lấy tối đa 3 tài liệu phù hợp nhất
            "score_threshold": 0.5  # ngưỡng điểm tương đồng tối thiểu (tùy chỉnh theo nhu cầu)
        }
    )


    # from time_priority_retriever import build_law_retriever
    # app_state.retriever = build_law_retriever(app_state.vectorstore)
    if app_state.retriever is None:
        raise HTTPException(status_code=500, detail="Failed to create retriever")
    logger.info(f"🔸Đã tạo retriever thành công.")

    # 5. Tạo QA Chain (giữ nguyên)
    logger.info(f"🔸Đang tạo QA Chain...")
    app_state.qa_chain = rag_components.create_qa_chain(
        app_state.llm,
        app_state.vectorstore,
        app_state.retriever,
    )
    if app_state.qa_chain is None:
        raise HTTPException(status_code=500, detail="Failed to create QA Chain")

    logger.info(f"🔸Khởi tạo API Components hoàn tất ")


# Lấy user hiện tại từ token
# def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
#     token = credentials.credentials

#     # Kiểm tra token có bị thu hồi (blacklist) không
#     if blacklist_collection.find_one({"token": token}):
#         raise HTTPException(status_code=401, detail="Token đã bị thu hồi")

#     try:
#         # Giải mã token
#         payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
#         email: str = payload.get("sub")

#         if email is None:
#             raise HTTPException(status_code=401, detail="Token không hợp lệ")

#         return email

#     except JWTError:
#         raise HTTPException(status_code=401, detail="Token không hợp lệ")

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)) -> UserOut:
    """
    Dependency để xác thực và lấy thông tin người dùng hiện tại từ token.

    Args:
        credentials: HTTP Authorization credentials từ request header

    Returns:
        UserOut: Thông tin người dùng đã xác thực

    Raises:
        HTTPException: Khi token không hợp lệ hoặc đã hết hạn
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Không thể xác thực người dùng",
        headers={"WWW-Authenticate": "Bearer"},
    )

    token = credentials.credentials

    # Kiểm tra token có bị thu hồi (blacklist) không
    if blacklist_collection.find_one({"token": token}):
        logger.warning(f"Phát hiện token đã bị thu hồi")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token đã bị thu hồi hoặc hết hạn",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        # Giải mã token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

        # Kiểm tra thông tin trong payload
        email: str = payload.get("sub")
        if email is None:
            logger.warning("Token không chứa sub (email)")
            raise credentials_exception

        # Kiểm tra thời hạn token
        exp = payload.get("exp")
        if exp is None:
            logger.warning("Token không chứa exp (expiration time)")
            raise credentials_exception

        # Kiểm tra token có hết hạn không (thêm kiểm tra dự phòng)
        if datetime.fromtimestamp(exp, tz=timezone.utc) < datetime.now(tz=timezone.utc):
            logger.warning(f"Token đã hết hạn")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token đã hết hạn",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Lấy thông tin người dùng từ database
        user_data = user_collection.find_one({"email": email}, {"password": 0})
        if user_data is None:
            logger.warning(f"Không tìm thấy người dùng với ID: {email}")
            raise credentials_exception


        # Tạo đối tượng UserOut từ dữ liệu user
        try:
            user = UserOut(**user_data)

            # Kiểm tra tài khoản có bị khóa không
            if hasattr(user, "is_active") and not user.is_active:
                logger.warning(f"Tài khoản đã bị khóa: {email}")
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Tài khoản đã bị khóa",
                    headers={"WWW-Authenticate": "Bearer"},
                )

            return user
        except Exception as e:
            logger.error(f"Lỗi khi chuyển đổi dữ liệu người dùng: {str(e)}")
            raise credentials_exception

    except JWTError as e:
        logger.error(f"Lỗi JWT: {str(e)}")
        raise credentials_exception
    except Exception as e:
        logger.error(f"Lỗi không xác định khi xác thực token: {str(e)}")
        raise credentials_exception


async def admin_required(
    current_user: Annotated[UserOut, Depends(get_current_user)]
) -> UserOut:
    """
    Dependency kiểm tra người dùng hiện tại có quyền admin hay không.
    Trả về thông tin người dùng nếu có quyền admin, nếu không raise HTTPException.

    Usage:
        @router.get("/admin-only")
        async def admin_route(user: UserOut = Depends(admin_required)):
            return {"message": "You have admin access"}
    """
    if not current_user.role or current_user.role not in [UserRole.ADMIN]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Bạn không có quyền truy cập chức năng này",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return current_user

async def super_admin_required(
    current_user: Annotated[UserOut, Depends(get_current_user)]
) -> UserOut:
    """
    Dependency kiểm tra người dùng hiện tại có quyền super admin hay không.
    Trả về thông tin người dùng nếu có quyền super admin, nếu không raise HTTPException.
    """
    if not current_user.role or current_user.role != UserRole.SUPER_ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Bạn không có quyền truy cập chức năng này",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return current_user

# Tạo một dependency factory để kiểm tra quyền động
def role_required(*allowed_roles):
    """
    Factory tạo dependency kiểm tra người dùng có thuộc một trong các role được cho phép hay không.

    Usage:
        @router.get("/staff-only")
        async def staff_route(user: UserOut = Depends(role_required(UserRole.ADMIN, UserRole.STAFF))):
            return {"message": "You have staff access"}
    """
    async def check_role(current_user: Annotated[UserOut, Depends(get_current_user)]) -> UserOut:
        if not current_user.role or current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Bạn không có quyền truy cập chức năng này",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return current_user

    return check_role