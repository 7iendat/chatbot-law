from fastapi import FastAPI
from contextlib import asynccontextmanager
import uvicorn
import config
from routers.auth import router as auth_router
from routers.chat import router as chat_router
from routers.documents import router as docs_router
from routers.health_check import router as health_router
from dependencies import initialize_api_components

app_state = {
    "embeddings": None,
    "vectorstore": None,
    "llm": None,
    "qa_chain": None,
    "device": "cpu",
    "groq_api_key": None,
    "dict": {},
    "redis": None,
    "retriever": None,
    "weaviateDB": None,
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_api_components(app_state)
    app.state.app_state = app_state
    yield
    app_state.clear()

app = FastAPI(
    title="Chatbot Hỏi Đáp Luật Việt Nam",
    version="1.2.0",
    lifespan=lifespan
)


# Include routers
app.include_router(auth_router, prefix="/auth", tags=["Auth"])
app.include_router(chat_router, prefix="/chat", tags=["Chatbot"])
app.include_router(docs_router, prefix="/documents", tags=["Documents"])
app.include_router(health_router, tags=["Status"])

if __name__ == "__main__":
    print(f"=> Chạy FastAPI server với Uvicorn...")
    uvicorn.run("main:app", host=config.API_HOST, port=config.API_PORT, reload=True)