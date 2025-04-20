from fastapi import APIRouter,Request

router = APIRouter()

@router.get("/health")
async def health_check(request: Request):
    app_state = request.app.state.app_state
    if app_state.get("qa_chain"):
        return {"status": "OK", "message": "QA Chain is ready."}
    else:
        return {"status": "Initializing or Failed", "message": "QA Chain is not ready."}