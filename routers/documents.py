from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException, Depends, Request
import os
import time
import shutil
from schemas.user import UserOut
from dependencies import get_current_user
import logging
from typing import List
import config
from utils.utils import calculate_file_hash, check_if_hash_exists
from services.document_service import full_process_and_ingest_pipeline
from dependencies import get_app_state
logger = logging.getLogger(__name__)

router = APIRouter()

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc"}

@router.post("/upload/", status_code=202)
async def upload_and_ingest_documents(
    fastapi_request: Request,
    background_tasks: BackgroundTasks,
    current_user: UserOut = Depends(get_current_user),
    files: List[UploadFile] = File(..., description="Một hoặc nhiều file tài liệu cần upload.")
):
    """
    Endpoint duy nhất để upload một hoặc nhiều tài liệu.

    - **files**: Danh sách các file tài liệu cần upload.
    - API sẽ xử lý từng file trong nền và trả về ngay một báo cáo tổng hợp.
    - File trùng lặp (dựa trên nội dung) hoặc có định dạng không hỗ trợ sẽ bị bỏ qua.
    """

    app_state = get_app_state(request=fastapi_request)
    embedding_model = app_state.embeddings
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded.")

    accepted_files = []
    skipped_files = []

    for file in files:
        temp_file_path = None
        try:
            # Kiểm tra định dạng file
            file_extension = os.path.splitext(file.filename)[1].lower()
            if file_extension not in ALLOWED_EXTENSIONS:
                skipped_files.append({"filename": file.filename, "reason": "Unsupported file type"})
                continue

            # 1. Lưu file tạm để tính hash
            # Thêm timestamp để tránh xung đột tên file nếu upload nhiều file cùng tên trong 1 request
            temp_filename = f"temp_{int(time.time()*1000)}_{file.filename}"
            temp_file_path = os.path.join(config.PENDING_UPLOADS_FOLDER, temp_filename)
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # 2. Tính toán hash
            file_hash = calculate_file_hash(temp_file_path)

            # 3. Kiểm tra trùng lặp
            if check_if_hash_exists(file_hash):
                skipped_files.append({"filename": file.filename, "reason": "Duplicate file (content already processed)"})
                os.remove(temp_file_path) # Xóa file tạm
                continue

            # 4. File hợp lệ, chuẩn bị để xử lý
            final_filename = file.filename
            final_file_path = os.path.join(config.PENDING_UPLOADS_FOLDER, final_filename)
            # Xử lý nếu tên file đã tồn tại để tránh ghi đè
            if os.path.exists(final_file_path):
                 base, ext = os.path.splitext(final_filename)
                 final_filename = f"{base}_{file_hash[:8]}{ext}"
                 final_file_path = os.path.join(config.PENDING_UPLOADS_FOLDER, final_filename)

            os.rename(temp_file_path, final_file_path)
            temp_file_path = None # Đánh dấu là đã di chuyển

            # 5. Thêm tác vụ nền cho file này
            background_tasks.add_task(full_process_and_ingest_pipeline, final_file_path, file_hash,embedding_model)

            accepted_files.append({"filename": final_filename, "hash": file_hash})

        except Exception as e:
            logger.error(f"Error processing {file.filename} in upload batch: {e}", exc_info=True)
            skipped_files.append({"filename": file.filename, "reason": f"Server error: {str(e)}"})
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    # Nếu không có file nào được chấp nhận sau khi lọc
    if not accepted_files:
        raise HTTPException(
            status_code=400,
            detail={"message": "No valid new files were accepted for processing.", "skipped_files": skipped_files}
        )

    # Trả về kết quả tổng hợp
    return {
        "message": f"Request completed. Accepted {len(accepted_files)} files for background processing.",
        "accepted_files": accepted_files,
        "skipped_files": skipped_files
    }