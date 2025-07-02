from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError, ExpiredSignatureError
import os
from dotenv import load_dotenv
from db.mongoDB import user_collection, blacklist_collection
import torch
import rag_components
from schemas.chat import AppState
from pydantic import ValidationError
from config import SECRET_KEY, ALGORITHM,EMBEDDING_MODEL_NAME,WEAVIATE_COLLECTION_NAME, WEAVIATE_URL
from utils.utils import load_legal_dictionary
import config
from langchain_groq import ChatGroq
from typing import Annotated, Optional
from schemas.user import UserOut, UserRole
from fastapi import status
from datetime import datetime, timezone
from db.redis import get_redis_client # Giả sử bạn đã định nghĩa hàm này trong db/redis.py
from utils.AdvancedLawRetriever import AdvancedLawRetriever
from services.reranker_service import get_reranker_compressor
from db.weaviateDB import connect_to_weaviate
import logging

logger = logging.getLogger(__name__)

handler = logging.StreamHandler()  # Gửi log đến stdout
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# Bearer token security scheme
bearer_scheme = HTTPBearer(auto_error=False)

def get_app_state(request: Request):
    if not hasattr(request.app.state, 'app_state'):
        print("Error in get_app_state: request.app.state.app_state is not set!")
        raise RuntimeError("Application state ('app_state') not found. Initialization failed?")
    return request.app.state.app_state

# def initialize_redis_client():
#     redis_url = os.environ.get("REDIS_URL")
#     if not redis_url:
#         logger.error("🔸[Redis] REDIS_URL environment variable not set.")
#         raise ValueError("REDIS_URL is not configured.")
#     try:
#         logger.info(f"🔸[Redis] Attempting to connect to Redis at {redis_url}...")
#         client = redis.Redis.from_url(redis_url, socket_connect_timeout=5, socket_timeout=5)
#         logger.info("🔸[Redis] Connected successfully and pinged.")
#         return client
#     except redis.exceptions.ConnectionError as e:
#         logger.error(f"🔸[Redis] Connection failed for URL '{redis_url}': {e}")
#         raise ConnectionError(f"Failed to connect to Redis: {e}")
#     except Exception as e:
#         logger.error(f"🔸[Redis] Error initializing Redis from URL '{redis_url}': {e}")
#         raise RuntimeError(f"Error initializing Redis: {e}")

async def initialize_api_components(app_state: AppState):
    """Khởi tạo các thành phần cần thiết cho API """
    logger.info("🔸Bắt đầu Khởi tạo API Components")

    load_dotenv()
    # --- Kiểm tra kết nối tới Redis ---
    app_state.process_input_llm = ChatGroq(model=config.GROQ_MODEL_NAME,temperature=0.2)
    try:
        app_state.redis = await get_redis_client() # Gọi hàm khởi tạo redis
    except Exception as e:
        logger.error(f"☠️ LỖI NGHIÊM TRỌNG khi khởi tạo Redis trong initialize_api_components: {e}")
        raise
    app_state.dict = load_legal_dictionary(config.LEGAL_DIC_FOLDER+ "/legal_terms.json")
    app_state.weaviateDB = connect_to_weaviate(run_diagnostics=False)
    # --- Kiểm tra kết nối tới MongoDB ---
    if user_collection is  None or app_state.weaviateDB is None:
        logger.error("🔸Lỗi kết nối tới MongoDB hoặc Weaviate.")
        raise HTTPException(status_code=500, detail="Lỗi kết nối tới database.")

    app_state.google_api_key = os.environ.get("GOOGLE_API_KEY")
    if not app_state.google_api_key:
        logger.error("🔸GG API Key không được cung cấp.")
        raise HTTPException(status_code=500, detail="Missing GG API Key")

    app_state.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"🔸Sử dụng thiết bị: {app_state.device}")

    # 1. Tải Embedding Model (giữ nguyên)
    print(f"Đang tải Embedding Model...")
    app_state.embeddings = rag_components.get_huggingface_embeddings(
        EMBEDDING_MODEL_NAME, app_state.device
    )
    if not app_state.embeddings:
        raise HTTPException(status_code=500, detail="Failed to load embedding model")

    # 2. Tải Vector Store
    print(f"Đang tải Vector Store...")
    app_state.vectorstore = rag_components.create_or_load_vectorstore(
        embeddings=app_state.embeddings,
        weaviate_url=WEAVIATE_URL,
        collection_name=WEAVIATE_COLLECTION_NAME,
        weaviate_client=app_state.weaviateDB,
        chunks=None,
    )

    if not app_state.vectorstore:
         raise HTTPException(status_code=500, detail="Failed to load or create Vectorstore")

    # 3. Tải LLM
    logger.info(f"🔸Đang tải LLM...")

    llm = rag_components.get_google_llm(app_state.google_api_key)
    app_state.llm = llm
    logger.info(f"🔸Tải LLM (Groq) thanh cong")

    if not app_state.llm:
        raise HTTPException(status_code=500, detail="Failed to load LLM")

    # 4. Tạo retriever (giữ nguyên)
    logger.info(f"🔸Đang tạo retriever...")

    app_state.reranker = get_reranker_compressor() # Singleton re-ranker

    app_state.retriever = AdvancedLawRetriever(
        client=app_state.weaviateDB,
        collection_name=WEAVIATE_COLLECTION_NAME,
        llm=app_state.llm,
        reranker=app_state.reranker, # Singleton re-ranker
        embeddings_model=app_state.embeddings
    )

    if app_state.retriever is None:
        raise HTTPException(status_code=500, detail="Failed to create retriever")
    logger.info(f"🔸Đã tạo retriever thành công.")

    # 5. Tạo QA Chain (giữ nguyên)
    logger.info(f"🔸Đang tạo QA Chain...")

    app_state.qa_chain = rag_components.create_qa_chain(
        llm=app_state.llm,
        retriever=app_state.retriever,
        process_input_llm=app_state.process_input_llm
    )
    if app_state.qa_chain is None:
        raise HTTPException(status_code=500, detail="Failed to create QA Chain")

    logger.info(f"🔸Khởi tạo API Components hoàn tất ")


async def get_access_token_from_cookie(request: Request) -> Optional[str]:
    """
    Lấy access token từ cookie 'access_token_cookie'.
    """
    token = request.cookies.get("access_token_cookie")
    logger.info(f"Token {token}")
    logger.debug(f"GET_ACCESS_TOKEN_FROM_COOKIE: Cookies nhận được: {request.cookies}")
    logger.info(f"GET_ACCESS_TOKEN_FROM_COOKIE: Token trích xuất từ 'access_token_cookie': {'PRESENT' if token else 'MISSING'}")
    return token

async def get_current_user(
    request: Request,
    token_from_cookie: Optional[str] = Depends(get_access_token_from_cookie),
    auth_header: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme) # Sẽ raise 403 nếu header Auth sai format
) -> UserOut:
    logger.warning("GET_CURRENT_USER: *** BẮT ĐẦU XÁC THỰC ***") # Sẽ không thấy log này nếu bearer_scheme raise 403

    token_to_verify: Optional[str] = None
    source_of_token: str = "NONE"

    if token_from_cookie:
        token_to_verify = token_from_cookie
        source_of_token = "COOKIE"
        logger.info("GET_CURRENT_USER: Sử dụng token từ cookie.")
    elif auth_header: # Chỉ dùng nếu không có token từ cookie
        token_to_verify = auth_header.credentials
        source_of_token = "AUTHORIZATION_HEADER"
        logger.info("GET_CURRENT_USER: Không có token từ cookie, sử dụng token từ Authorization header.")
    # Không cần else, 'if not token_to_verify' ở dưới sẽ xử lý

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Không thể xác thực người dùng. Vui lòng đăng nhập lại.",
        headers={"WWW-Authenticate": "Bearer"}, # Thêm header này là good practice
    )

    if not token_to_verify:
        logger.error(f"GET_CURRENT_USER: *** KHÔNG TÌM THẤY TOKEN (Nguồn: {source_of_token}) - RAISING 401 ***")
        raise credentials_exception

    logger.info(f"GET_CURRENT_USER: Token để verify (nguồn: {source_of_token}): {token_to_verify[:20]}...")

    # 1. Kiểm tra token trong blacklist
    try:
        logger.info("GET_CURRENT_USER: Đang kiểm tra blacklist...")
        is_blacklisted = blacklist_collection.find_one({"token": token_to_verify})
        if is_blacklisted:
            logger.error(f"GET_CURRENT_USER: *** TOKEN TRONG BLACKLIST - RAISING 401 ***")
            raise HTTPException( # Sử dụng credentials_exception hoặc một cái cụ thể hơn
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token đã bị thu hồi hoặc không hợp lệ.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        logger.info("GET_CURRENT_USER: Token không trong blacklist - OK")
    except HTTPException:
        raise
    except Exception as db_error:
        logger.error(f"GET_CURRENT_USER: *** LỖI DATABASE BLACKLIST: {db_error} ***")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Lỗi máy chủ khi kiểm tra trạng thái token."
        )

    # 2. Giải mã và xác thực JWT
    payload: Optional[dict] = None
    email: Optional[str] = None
    try:
        logger.info("GET_CURRENT_USER: Đang decode JWT...")
        if not SECRET_KEY: # Kiểm tra này quan trọng
            logger.error("GET_CURRENT_USER: *** SECRET_KEY CHƯA ĐƯỢC CẤU HÌNH ***")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Lỗi cấu hình máy chủ.")

        payload = jwt.decode(token_to_verify, SECRET_KEY, algorithms=[ALGORITHM])
        logger.info(f'CHECK: {payload}')
        email = payload.get("sub")
        exp = payload.get("exp")

        logger.info(f"GET_CURRENT_USER: JWT decode thành công - email: {email}, exp: {exp}")

        if not isinstance(email, str) or not email:
            logger.error("GET_CURRENT_USER: *** EMAIL KHÔNG HỢP LỆ TRONG TOKEN ***")
            raise credentials_exception # Sử dụng lại credentials_exception đã định nghĩa
        if not isinstance(exp, int): # Thường 'exp' là int (timestamp)
            logger.error("GET_CURRENT_USER: *** EXP KHÔNG HỢP LỆ TRONG TOKEN ***")
            raise credentials_exception

        expiration_datetime = datetime.fromtimestamp(exp, tz=timezone.utc)
        current_datetime_utc = datetime.now(tz=timezone.utc)

        if expiration_datetime < current_datetime_utc:
            logger.error(f"GET_CURRENT_USER: *** TOKEN HẾT HẠN - RAISING 401 ***")
            raise HTTPException( # Sử dụng credentials_exception hoặc một cái cụ thể hơn
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token đã hết hạn. Vui lòng đăng nhập lại.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        logger.info(f"GET_CURRENT_USER: Token còn hạn - OK")

    except ExpiredSignatureError: # Bắt lỗi cụ thể này từ PyJWT
        logger.error(f"GET_CURRENT_USER: *** TOKEN HẾT HẠN (ExpiredSignatureError) ***")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token đã hết hạn (JWT validation). Vui lòng đăng nhập lại.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except JWTError as e: # Bắt lỗi chung từ PyJWT
        logger.error(f"GET_CURRENT_USER: *** LỖI JWT: {e} ***")
        raise HTTPException( # Có thể dùng credentials_exception hoặc thông báo cụ thể hơn
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token không hợp lệ hoặc có vấn đề khi giải mã.", # Thông báo chung chung hơn
            headers={"WWW-Authenticate": "Bearer"},
        )
    except HTTPException: # Re-raise nếu là HTTPException đã được raise từ bên trong try
        raise
    except Exception as e_decode: # Bắt các lỗi không mong muốn khác
        logger.error(f"GET_CURRENT_USER: *** LỖI KHÔNG XÁC ĐỊNH KHI DECODE JWT: {e_decode} ***")
        raise credentials_exception # Trả về lỗi chung

    # 3. Lấy thông tin người dùng từ database
    user_data: Optional[dict] = None # Khởi tạo để tránh UnboundLocalError
    try:
        logger.info(f"GET_CURRENT_USER: Đang tìm user trong DB: {email.lower()}") # email đã được validate là str
        user_data = user_collection.find_one({"email": email.lower()}, {"password": 0, "_id": 0})
        # print(user_data) # Bỏ print trong production

        if user_data is None:
            logger.error(f"GET_CURRENT_USER: *** KHÔNG TÌM THẤY USER TRONG DB ({email.lower()}) - RAISING 401 ***")
            raise credentials_exception

        logger.info(f"GET_CURRENT_USER: Tìm thấy user - data: {user_data}")

    except HTTPException:
        raise
    except Exception as db_user_error:
        logger.error(f"GET_CURRENT_USER: *** LỖI DATABASE USER: {db_user_error} ***")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Lỗi máy chủ khi truy xuất thông tin người dùng."
        )

    # 4. Tạo đối tượng UserOut và kiểm tra is_active
    try:

        if user_data and ('username' not in user_data or not user_data.get('username')):
            user_data['username'] = email.lower().split('@')[0]
            logger.info(f"GET_CURRENT_USER: Set default username: {user_data['username']}")

        user = UserOut(**user_data)

        if not user.is_active:
            logger.error(f"GET_CURRENT_USER: *** TÀI KHOẢN BỊ KHÓA ({user.email}) - RAISING 403 ***")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, # 403 là phù hợp ở đây
                detail="Tài khoản của bạn đã bị khóa hoặc không hoạt động. Vui lòng liên hệ quản trị viên.",
            )

        logger.info(f"GET_CURRENT_USER: *** XÁC THỰC THÀNH CÔNG *** - User: {user.email}, Active: {user.is_active}, Role: {user.role}")

        return user

    except ValidationError as ve:
        logger.error(f"GET_CURRENT_USER: *** LỖI PYDANTIC VALIDATION: {ve.errors()} ***")
        logger.error(f"GET_CURRENT_USER: Dữ liệu gây lỗi: {user_data}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, # 422 khi dữ liệu không thể xử lý
            detail=f"Dữ liệu người dùng không hợp lệ từ DB: {ve.errors()}" # Có thể trả về lỗi cụ thể nếu an toàn
        )
    except HTTPException: # Re-raise
        raise
    except Exception as e_userout:
        logger.error(f"GET_CURRENT_USER: *** LỖI TẠO USEROUT HOẶC KIỂM TRA IS_ACTIVE: {e_userout} ***")
        raise credentials_exception # Lỗi chung nếu không rõ nguyên nhân

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
