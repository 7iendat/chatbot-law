from fastapi import APIRouter,Request
from dependencies import get_app_state

router = APIRouter()

@router.get("/health")
async def health_check(request: Request):
    app_state = get_app_state(request=request)
    if app_state.qa_chain:
        return {"status": "OK", "message": "QA Chain is ready."}
    else:
        return {"status": "Initializing or Failed", "message": "QA Chain is not ready."}