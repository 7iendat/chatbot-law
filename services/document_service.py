import os
import gc
from fastapi import UploadFile, BackgroundTasks, HTTPException, Request
from langchain.schema import Document
import config
import utils.utils as utils
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def update_vectorstore_bg(new_file_path: str, app_state, filename: str):
    logger.info(f"🔸[BG Task] => Bắt đầu cập nhật với file: {new_file_path}", flush=True)

    if not os.path.exists(new_file_path):
        logger.error(f"🔸[BG Task] => File không tồn tại: {new_file_path}", flush=True)
        return

    try:
        ext = os.path.splitext(new_file_path)[1].lower()
        content = ""

        if ext == '.pdf':
            content = utils.extract_text_from_pdf_auto(new_file_path, lang='vie')
        elif ext in ['.txt', '.md']:
            with open(new_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        else:
            logger.error(f"🔸[BG Task] => Định dạng file '{ext}' chưa được hỗ trợ.", flush=True)
            return

        if not content.strip():
            logger.error(f"🔸[BG Task] => Không thể trích xuất nội dung từ file: {new_file_path}", flush=True)
            return


        year = utils.extract_year_from_filename(filename) or utils.extract_year_from_text(content)
        law_structure = utils.extract_law_structure(content)
        cleaned_content = utils.clean_text_optimized(content)
        field = utils.infer_field(cleaned_content)

        doc = Document(
            page_content=content,
            metadata = {
                "source": os.path.basename(new_file_path),
                "year": year,
                "field": field,
                "entity_type": utils.infer_entity_type(cleaned_content, field),
                "penalty": utils.extract_penalty(cleaned_content, field),
                **law_structure,  # Thêm số hiệu, loại văn bản, ngày ban hành
            }
        )
        doc_chunks = utils.split_by_law_structure(doc, max_chunk_size=config.CHUNK_SIZE * 2)

        if not doc_chunks:
            logger.error("🔸[BG Task] => Không có chunk nào được tạo.", flush=True)
            return

        embeddings = app_state.embeddings
        vectorstore = app_state.vectorstore

        if not embeddings or not vectorstore:
            logger.error(f"🔸Không tìm thấy embeddings hoặc vectorstore trong app_state.", flush=True)
            return

        vectorstore.add_documents(doc_chunks)
        del doc, doc_chunks, content
        gc.collect()
        logger.info(f"🔸[BG Task] => Đã thêm dữ liệu từ {os.path.basename(new_file_path)} vào Vector Store.", flush=True)

    except Exception as e:
        logger.error(f"🔸Lỗi khi cập nhật Vector Store: {e}", flush=True)

async def upload_and_index_document(file: UploadFile, user, background_tasks: BackgroundTasks, request: Request):
    filename = file.filename
    file_extension = os.path.splitext(filename)[1].lower()
    base_filename = os.path.splitext(filename)[0]
    output_filename = base_filename + ".txt"
    output_path = os.path.join(config.INPUT_TXT_FOLDER, output_filename)

    if any([
        os.path.exists(os.path.join(config.INPUT_TXT_FOLDER, filename)),
        os.path.exists(output_path)
    ]):
        return {"message": f"=> File '{filename}' hoặc bản .txt của nó đã tồn tại. Bỏ qua."}

    if file_extension not in [".txt", ".pdf"]:
        raise HTTPException(status_code=400, detail="=> Chỉ hỗ trợ file .txt và .pdf.")

    os.makedirs(config.INPUT_TXT_FOLDER, exist_ok=True)

    try:
        contents = await file.read()
        text = ""

        if file_extension == ".txt":
            text = contents.decode("utf-8")
        elif file_extension == ".pdf":
            if not all([utils.fitz, utils.Image, utils.io, utils.pytesseract]):
                raise HTTPException(status_code=500, detail="=> Thiếu thư viện xử lý PDF: fitz, Pillow, pytesseract...")
            temp_path = f"/tmp/{filename}"
            with open(temp_path, "wb") as f:
                f.write(contents)
            text = utils.extract_text_from_scanned_pdf(temp_path)
            os.remove(temp_path)

        cleaned_text = utils.clean_text_optimized(text)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(cleaned_text)

        app_state = request.app.state.app_state
        background_tasks.add_task(update_vectorstore_bg, new_file_path=output_path, app_state=app_state, filename=filename)

        return {"message": f"=> File '{filename}' đã được tải lên. Vector store đang được cập nhật nền."}

    except Exception as e:
        if os.path.exists(output_path):
            os.remove(output_path)
        raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý file: {str(e)}")