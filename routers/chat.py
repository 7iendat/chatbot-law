from fastapi import APIRouter, Depends, HTTPException, Request
from schemas.chat import QueryRequest, AnswerResponse, ChatHistoryResponse
from dependencies import get_current_user
from services.chat_service import ask_question_service
from utils.utils import get_redis_history
from dependencies import get_app_state
router = APIRouter()

@router.post("/ask", response_model=AnswerResponse)
async def ask(request_body: QueryRequest,request: Request, user=Depends(get_current_user)):
    app_state = get_app_state(request=request)
    result = ask_question_service(app_state,request_body, user)

    if not result:
        raise HTTPException(status_code=500, detail="Error during QA Chain invocation")
    return result

@router.get("/history/{session_id}",response_model=ChatHistoryResponse)
async def history(request: Request,session_id: str):
    app_state = get_app_state(request=request)
    redis = app_state.get("redis")
    history = get_redis_history(redis, session_id)
    return {"session_id": session_id, "history": history}