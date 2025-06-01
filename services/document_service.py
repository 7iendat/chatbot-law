import os
import gc
import tempfile
import asyncio
from typing import Optional, List, Dict, Tuple
from fastapi import UploadFile, BackgroundTasks, HTTPException, Request, File, Depends
from langchain.schema import Document
import config
import utils.utils as utils
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Class để xử lý tải lên và index documents"""

    def __init__(self):
        self.supported_extensions = ['.pdf', '.txt', '.md', '.docx']
        self.max_file_size = 50 * 1024 * 1024  # 50MB

    async def validate_file(self, file: UploadFile) -> None:
        # ... (giữ nguyên)
        if not file.filename:
            raise HTTPException(status_code=400, detail="Filename không được để trống")

        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension not in self.supported_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Chỉ hỗ trợ các định dạng: {', '.join(self.supported_extensions)}"
            )
        file.file.seek(0, 2)
        file_size = file.file.tell()
        file.file.seek(0)
        if file_size > self.max_file_size:
            raise HTTPException(
                status_code=413,
                detail=f"File quá lớn. Kích thước tối đa: {self.max_file_size // (1024*1024)}MB"
            )

    def check_file_exists(self, filename: str) -> bool:
        # ... (giữ nguyên)
        base_filename = os.path.splitext(filename)[0]
        possible_paths = [
            os.path.join(config.INPUT_TXT_FOLDER, filename),
            os.path.join(config.INPUT_TXT_FOLDER, base_filename + ".txt"),
            os.path.join(config.INPUT_TXT_FOLDER, base_filename + ".md")
        ]
        return any(os.path.exists(path) for path in possible_paths)

    async def extract_content(self, file: UploadFile, temp_path: str) -> str: # Hàm này trả về string
        """Trích xuất nội dung từ file"""
        file_extension = os.path.splitext(file.filename)[1].lower()
        extracted_text: str = "" # Khởi tạo là chuỗi rỗng

        try:
            if file_extension == '.txt':
                extracted_text = (await file.read()).decode('utf-8')
            elif file_extension == '.md':
                extracted_text = (await file.read()).decode('utf-8')
            elif file_extension == '.pdf':
                contents = await file.read()
                with open(temp_path, 'wb') as f:
                    f.write(contents)
                # utils.extract_text_from_pdf_auto trả về Tuple[str, Dict]
                # Chúng ta chỉ cần phần tử đầu tiên (str)
                pdf_extraction_result: Tuple[str, Dict] = utils.extract_text_from_pdf_auto(temp_path, lang='vie')
                extracted_text = pdf_extraction_result[0] # Lấy chuỗi văn bản
                # Bạn có thể muốn log hoặc sử dụng pdf_extraction_result[1] (metadata) ở đâu đó nếu cần
                logger.info(f"PDF extraction metadata: {pdf_extraction_result[1]}")

            elif file_extension == '.docx':
                contents = await file.read()
                with open(temp_path, 'wb') as f:
                    f.write(contents)
                # Giả sử utils.extract_text_from_docx cũng có thể trả về tuple
                # hoặc chỉ trả về string. Cần kiểm tra định nghĩa của nó.
                # Nếu nó chỉ trả về string:
                # extracted_text = utils.extract_text_from_docx(temp_path)
                # Nếu nó cũng trả về tuple (text, metadata_docx):
                docx_extraction_result = utils.extract_text_from_docx(temp_path) # Kiểm tra kiểu trả về của hàm này
                if isinstance(docx_extraction_result, tuple):
                    extracted_text = docx_extraction_result[0]
                    logger.info(f"DOCX extraction metadata (if any): {docx_extraction_result[1:]}")
                elif isinstance(docx_extraction_result, str):
                    extracted_text = docx_extraction_result
                else:
                    logger.error(f"Hàm extract_text_from_docx trả về kiểu không mong đợi: {type(docx_extraction_result)}")
                    # extracted_text sẽ vẫn là ""

            # Bây giờ extracted_text đã là một chuỗi (hoặc chuỗi rỗng)
            if not extracted_text or not extracted_text.strip():
                raise ValueError("Không thể trích xuất nội dung từ file hoặc file rỗng")

            return extracted_text

        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="File không đúng định dạng UTF-8")
        except ValueError as ve: # Bắt lỗi ValueError cụ thể hơn
            logger.warning(f"Lỗi giá trị khi trích xuất nội dung: {str(ve)}")
            raise HTTPException(status_code=400, detail=str(ve)) # Trả về thông báo lỗi từ ValueError
        except Exception as e:
            logger.error(f"Lỗi trích xuất nội dung không xác định: {str(e)}", exc_info=True) # Thêm exc_info để log traceback
            raise HTTPException(status_code=500, detail=f"Lỗi xử lý file: {str(e)}")

    def create_document_metadata(self, content: str, filename: str) -> dict:
        # ... (giữ nguyên)
        try:
            year = utils.extract_year_from_filename(filename) or utils.extract_year_from_text(content)
            law_structure = utils.extract_law_structure(content)
            cleaned_content = utils.clean_text_optimized(content)
            field = utils.infer_field(cleaned_content)
            metadata = {
                "source": filename, "year": year or "unknown", "field": field or "general",
                "entity_type": utils.infer_entity_type(cleaned_content, field),
                "penalty": utils.extract_penalty(cleaned_content, field),
                **law_structure,
            }
            return self.filter_metadata_for_weaviate(metadata)
        except Exception as e:
            logger.error(f"Lỗi tạo metadata: {str(e)}")
            return {
                "source": filename, "year": "unknown", "field": "general",
                "entity_type": "unknown", "penalty": "", "processed_at": utils.get_current_timestamp()
            }


    def filter_metadata_for_weaviate(self, metadata: dict) -> dict:
        # ... (giữ nguyên)
        filtered = {}
        for key, value in metadata.items():
            if value is not None:
                if isinstance(value, (str, int, float, bool)):
                    filtered[key] = value
                elif isinstance(value, list):
                    filtered[key] = ", ".join(str(v) for v in value)
                else:
                    filtered[key] = str(value)
        return filtered

    def create_chunks(self, document: Document) -> List[Document]:
        # ... (giữ nguyên)
        try:
            chunks = utils.split_by_law_structure(document, max_chunk_size=config.CHUNK_SIZE * 2)
            if not chunks:
                chunks = utils.simple_text_splitter(document, chunk_size=config.CHUNK_SIZE)
            logger.info(f"Đã tạo {len(chunks)} chunks từ document")
            return chunks
        except Exception as e:
            logger.error(f"Lỗi tạo chunks: {str(e)}")
            return [document]
async def update_vectorstore_background(
    file_content: str,
    filename: str,
    app_state,
    processor: DocumentProcessor
):
    """Background task để cập nhật vectorstore"""
    logger.info(f"🔸[BG Task] Bắt đầu xử lý file: {filename}")

    try:
        # Tạo document
        metadata = processor.create_document_metadata(file_content, filename)
        document = Document(page_content=file_content, metadata=metadata)

        # Tạo chunks
        chunks = processor.create_chunks(document)

        if not chunks:
            logger.error(f"🔸[BG Task] Không tạo được chunks từ file: {filename}")
            return

        # Lấy components từ app_state
        vectorstore = getattr(app_state, 'vectorstore', None)

        if not vectorstore:
            logger.error("🔸[BG Task] Không tìm thấy vectorstore trong app_state")
            return

        # Thêm documents vào vectorstore với batch processing
        await add_documents_to_vectorstore(vectorstore, chunks, filename)

        # Cleanup memory
        del document, chunks, file_content
        gc.collect()

        logger.info(f"🔸[BG Task] Hoàn thành xử lý file: {filename}")

    except Exception as e:
        logger.error(f"🔸[BG Task] Lỗi xử lý file {filename}: {str(e)}")

async def add_documents_to_vectorstore(vectorstore, chunks: List[Document], filename: str):
    """Thêm documents vào vectorstore với error handling"""
    max_batch_size = 100
    total_chunks = len(chunks)

    logger.info(f"🔸Thêm {total_chunks} chunks vào vectorstore")

    try:
        # Xử lý theo batch
        for i in range(0, total_chunks, max_batch_size):
            end_idx = min(i + max_batch_size, total_chunks)
            batch = chunks[i:end_idx]

            batch_num = i // max_batch_size + 1
            total_batches = (total_chunks - 1) // max_batch_size + 1

            logger.info(f"🔸Xử lý batch {batch_num}/{total_batches} ({len(batch)} chunks)")

            try:
                vectorstore.add_documents(batch)
                logger.info(f"🔸Batch {batch_num} thành công")

            except Exception as batch_error:
                logger.error(f"🔸Lỗi batch {batch_num}: {str(batch_error)}")

                # Thử thêm từng document một
                success_count = 0
                for doc in batch:
                    try:
                        vectorstore.add_documents([doc])
                        success_count += 1
                    except Exception as doc_error:
                        logger.error(f"🔸Lỗi thêm document: {str(doc_error)}")

                logger.info(f"🔸Batch {batch_num}: {success_count}/{len(batch)} documents thành công")

        logger.info(f"🔸Hoàn thành thêm chunks từ file: {filename}")

    except Exception as e:
        logger.error(f"🔸Lỗi nghiêm trọng khi thêm documents: {str(e)}")
        raise

async def upload_document_and_schedule_processing(
    file: UploadFile,
    user,
    background_tasks: BackgroundTasks,
    request: Request
):
    """Main function để upload và xử lý document"""
    processor = DocumentProcessor()

    try:
        # Validate file
        await processor.validate_file(file)

        filename = file.filename
        logger.info(f"🔸Bắt đầu upload file: {filename}")

        # Kiểm tra file đã tồn tại
        if processor.check_file_exists(filename):
            return {"message": f"File '{filename}' đã tồn tại. Bỏ qua.", "status": "skipped"}

        # Tạo thư mục output nếu chưa có
        os.makedirs(config.INPUT_TXT_FOLDER, exist_ok=True)

        # Sử dụng temporary file để xử lý
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
            temp_path = temp_file.name

            try:
                # Trích xuất nội dung
                content = await processor.extract_content(file, temp_path)

                # Clean content
                cleaned_content = utils.clean_text_optimized(content)

                # Lưu file đã processed
                base_filename = os.path.splitext(filename)[0]
                output_filename = base_filename + ".txt"
                output_path = os.path.join(config.INPUT_TXT_FOLDER, output_filename)

                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(cleaned_content)

                # Lấy app_state
                app_state = request.app.state.app_state

                # Schedule background processing
                background_tasks.add_task(
                    update_vectorstore_background,
                    file_content=cleaned_content,
                    filename=filename,
                    app_state=app_state,
                    processor=processor
                )

                logger.info(f"🔸Upload thành công: {filename}")

                return {
                    "message": f"File '{filename}' đã được tải lên thành công. Đang xử lý trong background.",
                    "status": "processing",
                    "filename": filename,
                    "output_path": output_filename
                }

            finally:
                # Cleanup temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"🔸Lỗi upload file {filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý file: {str(e)}")
