from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import config
# from schemas.chat import AppState # Nên import từ nơi định nghĩa AppState
from schemas.chat import AppState # Giả sử AppState ở app_state.py cùng cấp
from routers.user import router as user_router
from routers.chat import router as chat_router
from routers.documents import router as docs_router
from routers.health_check import router as health_router
# from dependencies import initialize_api_components # Nên import từ nơi định nghĩa
from dependencies import initialize_api_components # Giả sử initialize_api_components ở utils.py cùng cấp
import logging # Thêm logging
import traceback

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("✅ [Lifespan] STARTING UP...")
    current_app_state_instance = AppState()
    initialization_successful = False
    try:
        logger.info("✅ [Lifespan] Calling initialize_api_components...")
        initialize_api_components(current_app_state_instance)
        app.state.app_state = current_app_state_instance
        initialization_successful = True
        logger.info("✅ [Lifespan] SUCCESSFULLY set app.state.app_state.")
        yield
        logger.info("✅ [Lifespan] SHUTTING DOWN (after yield)...")
    except Exception as e:
        logger.error(f"❌ [Lifespan] FATAL ERROR DURING STARTUP: {type(e).__name__} - {e}")
        logger.error(traceback.format_exc())
    finally:
        logger.info(f"✅ [Lifespan] EXITED. Initialization successful: {initialization_successful}")
        if initialization_successful:
            logger.info("✅ [Lifespan] Performing resource cleanup (if any)...")
            if hasattr(current_app_state_instance, 'redis') and current_app_state_instance.redis and hasattr(current_app_state_instance.redis, 'close'):
                try:
                    # await current_app_state_instance.redis.close() # Nếu async
                    logger.info("✅ [Lifespan] Redis connection closed (simulated/sync).")
                except Exception as e_close:
                    logger.error(f"⚠️ [Lifespan] Error closing Redis: {e_close}")
            if hasattr(current_app_state_instance, 'weaviateDB') and current_app_state_instance.weaviateDB and hasattr(current_app_state_instance.weaviateDB, 'close'):
                try:
                    # await current_app_state_instance.weaviateDB.close() # Nếu async
                    logger.info("✅ [Lifespan] WeaviateDB connection closed (simulated/sync).")
                except Exception as e_close:
                    logger.error(f"⚠️ [Lifespan] Error closing WeaviateDB: {e_close}")
        else:
            logger.warning("⚠️ [Lifespan] Skipping resource cleanup due to startup failure.")

app = FastAPI(
    title="Chatbot Hỏi Đáp Luật Việt Nam",
    version="1.2.0",
    lifespan=lifespan
)

# Cấu hình CORS (áp dụng cho tất cả các route của app chính)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thêm routers trực tiếp vào app chính với prefix /api
app.include_router(user_router, prefix="/api/user", tags=["Manager User"])
app.include_router(chat_router, prefix="/api/chat", tags=["Chatbot"])
app.include_router(docs_router, prefix="/api/documents", tags=["Documents"])
app.include_router(health_router, prefix="/api", tags=["Status"]) # Hoặc chỉ / nếu health check không cần /api

# Run with Uvicorn
if __name__ == "__main__":
    logger.info("=> Chạy FastAPI server với Uvicorn...")
    uvicorn.run(
        "main:app", # Đảm bảo "main" là tên file python của bạn
        host=config.API_HOST if hasattr(config, 'API_HOST') else "0.0.0.0",
        port=int(config.API_PORT) if hasattr(config, 'API_PORT') else 8000,
        reload=True # reload=True chỉ nên dùng cho development
    )