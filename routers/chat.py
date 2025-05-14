from fastapi import APIRouter, Depends, HTTPException, Request
from schemas.chat import QueryRequest, AnswerResponse, ChatHistoryResponse
from dependencies import get_current_user
from services.chat_service import ask_question_service
from utils.utils import get_redis_history,delete_chat_from_redis
from dependencies import get_app_state
import logging
import uuid
from redis.asyncio import Redis

# Thiết lập logger
logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/create-chat")
async def create_chat(request: Request,user_email: str = Depends(get_current_user)):
    app_state = get_app_state(request=request)
    redis:Redis = app_state.get("redis")
    if not user_email:
        raise ValueError("user_email is missing")

    chat_id = str(uuid.uuid4())

    redis.hset(f"chat:{chat_id}:meta", mapping={"user": str(user_email)})
    return {"chat_id": chat_id}


@router.post("/", response_model=AnswerResponse)
async def chat_message(request_body: QueryRequest,request: Request, user_email:str=Depends(get_current_user)):
    app_state = get_app_state(request=request)
    result = ask_question_service(app_state,request_body, user_email)

    if not result:
        raise HTTPException(status_code=500, detail="Error during QA Chain invocation")
    return result

@router.delete("/chats/{chat_id}")
async def delete_chat(chat_id: str, request: Request, user_email: str = Depends(get_current_user)):
    app_state = get_app_state(request=request)
    redis = app_state["redis"]

    # Kiểm tra quyền trước khi xóa
    user_in_chat =  redis.hget(f"chat:{chat_id}:meta", "user")
    if user_in_chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")

    if user_in_chat.decode() != user_email:
        raise HTTPException(status_code=403, detail="Unauthorized")

    # Xóa chat
    delete_chat_from_redis(redis, chat_id)

    return {"detail": "Chat deleted successfully"}

@router.get("/{chat_id}/history",response_model=ChatHistoryResponse)
async def history(request: Request,chat_id: str):
    app_state = get_app_state(request=request)
    redis = app_state.get("redis")
    if not  redis.exists(f"chat:{chat_id}:messages"):
        raise HTTPException(status_code=404, detail="Chat not found")

    try:
        logger.info(f"Fetching history for session: {chat_id}")
        history = get_redis_history(redis, chat_id)
        return {"chat_id": chat_id, "history": history}
    except Exception as e:
        logger.error(f"Error fetching history for session {chat_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")