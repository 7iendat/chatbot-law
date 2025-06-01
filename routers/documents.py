from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException, Depends, Request
from services.document_service import (
    upload_document_and_schedule_processing
)
from dependencies import get_current_user

router = APIRouter()

@router.post("/upload")
async def upload_doc(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    request: Request = None,
    user=Depends(get_current_user)
):
    return await upload_document_and_schedule_processing(file, user, background_tasks, request)