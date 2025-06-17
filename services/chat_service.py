from fastapi import Depends, HTTPException
from schemas.chat import QueryRequest, AnswerResponse, SourceDocument
from schemas.user import UserOut
from dependencies import get_current_user
import time
import json
from utils.utils import save_chat_to_redis, search_term_in_dictionary, minimal_preprocess_for_llm, save_chat_to_mongo, get_langchain_chat_history
import os
import logging
from db.mongoDB import conversations_collection
from datetime import datetime, timezone
import asyncio

logger = logging.getLogger(__name__)


async def ask_question_service(app_state, request: QueryRequest, user: UserOut = Depends(get_current_user)):
    chat_id = request.chat_id
    question_content = request.input # Giữ lại câu hỏi gốc của user để lưu

    # --- 1. Xác thực và kiểm tra metadata từ Redis ---
    meta_key = f"conversation_meta:{chat_id}"
    if not   app_state.redis.exists(meta_key): # Dùng await nếu redis client là async
        logger.warning(f"Metadata cho chat_id {chat_id} không tìm thấy trong Redis.")
        raise HTTPException(status_code=404, detail="Chat ID not found or session expired. Please reload the conversation.")

    user_in_redis =  app_state.redis.hget(meta_key, "user_id") # Key đã đổi thành user_id
    if not user_in_redis:
        logger.error(f"user_id không có trong metadata của chat {chat_id}.")
        raise HTTPException(status_code=404, detail="Chat metadata corrupted.")

    if user_in_redis.decode() != user.email:
        logger.warning(f"User {user.email} không được phép truy cập chat {chat_id} (thuộc về {user_in_redis.decode()}).")
        raise HTTPException(status_code=403, detail="Unauthorized to access this chat.")

    start_time = time.time()
    current_utc_time = datetime.now(timezone.utc) # Sử dụng UTC cho timestamp

    # --- 2. Tiền xử lý câu hỏi ---
    cleaned_question = minimal_preprocess_for_llm(question_content)

    # --- 3. Kiểm tra từ điển thuật ngữ (nếu có) ---
    if hasattr(app_state, 'dict') and app_state.dict:
        term_result = search_term_in_dictionary(cleaned_question, app_state.dict)
        if term_result:
            answer_def = term_result.get("definition", "Không thể tìm thấy định nghĩa.")
            assistant_response_time = datetime.now(timezone.utc)

            # Lưu vào Redis và MongoDB
            save_chat_to_redis(
                app_state.redis, chat_id, question_content, answer_def, current_utc_time, assistant_response_time
            )
            await save_chat_to_mongo(
                conversations_collection, chat_id, user.email, question_content, answer_def, current_utc_time, assistant_response_time
            )
            friendly_answer = f"Xin chào! Về câu hỏi '{question_content}' của bạn, tôi đã tìm thấy thông tin sau:\n\n{answer_def}\n\nHy vọng thông tin này hữu ích cho bạn. Bạn có muốn tìm hiểu thêm về chủ đề này hoặc có câu hỏi nào khác không? 😊"
            return AnswerResponse(
                answer=friendly_answer,
                sources=[
                    SourceDocument(
                        source="Thuật ngữ pháp lý",
                        page_content_preview=f"Định nghĩa thuật ngữ từ cơ sở dữ liệu"
                    )
                ],
                processing_time=round(time.time() - start_time, 2)
            )

    if not app_state.qa_chain:
        logger.error("QA Chain chưa được khởi tạo.")
        raise HTTPException(status_code=503, detail="Service Unavailable: QA Chain not ready.")


    try:
        redis_url = os.environ.get("REDIS_URL_LANGCHAIN", os.environ.get("REDIS_URL")) # Ưu tiên URL riêng cho Langchain nếu có
        if not redis_url:
            logger.error("REDIS_URL or REDIS_URL_LANGCHAIN not set for RedisChatMessageHistory.")
            raise ValueError("Redis URL for chat history is required.")


        chat_history_messages = await prepare_chat_history_optimized(
            app_state.redis,
            chat_id,
            max_messages=10
        )
        input_data_for_chain = {
            # "chat_history":  langchain_chat_history.messages, # Lấy messages đã được đồng bộ
            "chat_history":  chat_history_messages, # Lấy messages đã được đồng bộ
            "input": cleaned_question
        }

    except Exception as e:
        logger.error(f"Lỗi khi chuẩn bị chat history cho Langchain (chat_id: {chat_id}): {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Lỗi xử lý lịch sử chat.")


    # --- 5. Gọi QA Chain ---
    try:
        logger.debug(f"Input to QA Chain (chat_id: {chat_id}): {input_data_for_chain}")

        # Metadata cho LangSmith trace
        langsmith_metadata = {
            "user_email": user.email,
            "chat_id": chat_id,
            "original_question": question_content,
            "cleaned_question": cleaned_question,
            "request_id": request.request_id if hasattr(request, 'request_id') else "N/A" # Nếu bạn có request ID
        }

        chain_result =  app_state.qa_chain.invoke(input_data_for_chain, config={
                    "metadata": langsmith_metadata,
                    "run_name": f"AskService_QA_Invoke_ChatID_{chat_id[:8]}"
                    # "tags": ["production", "qa_service"]
                })

        # logger.info(f"QA Chain raw result (chat_id: {chat_id}): {chain_result}")

        # Xử lý kết quả từ chain (logic của bạn để trích xuất câu trả lời)
        assistant_response_content = ""
        if isinstance(chain_result, dict) and "answer" in chain_result:
            assistant_response_content = str(chain_result["answer"])
        elif isinstance(chain_result, str): # Một số chain có thể trả về string trực tiếp
            assistant_response_content = chain_result
        else:
            logger.error(f"QA Chain result không hợp lệ (chat_id: {chat_id}): {chain_result}")
            assistant_response_content = "Xin lỗi, tôi không thể xử lý yêu cầu này vào lúc này."
            # Không raise lỗi ở đây ngay, mà trả về thông báo lỗi cho user và log lại.

        if not assistant_response_content.strip():
             assistant_response_content = "Tôi không tìm thấy câu trả lời phù hợp."


    except Exception as chain_error:
        logger.error(f"Lỗi QA Chain (chat_id: {chat_id}): {chain_error}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý từ QA chain: {str(chain_error)[:100]}")

    assistant_response_time = datetime.now(timezone.utc)

    # --- 6. Lưu tin nhắn mới (câu hỏi của user và trả lời của AI) ---
    # Lưu vào key "conversation_messages:{chat_id}" của chúng ta
    save_chat_to_redis(
        app_state.redis, chat_id, question_content, assistant_response_content, current_utc_time, assistant_response_time
    )
    # Lưu vào MongoDB
    # Chạy ngầm hoặc sau khi trả lời user để không làm chậm response (nếu có thể)
    await save_chat_to_mongo(
        conversations_collection, chat_id, user.email, question_content, assistant_response_content, current_utc_time, assistant_response_time
    )

    end_time = time.time()

    logger.info(f"Trả lời cho chat {chat_id} bởi user {user.email}: {assistant_response_content[:100]}...")
    return AnswerResponse(
        answer=assistant_response_content,
        processing_time=round(end_time - start_time, 2)
    )

async def stream_chat_generator(
    app_state,
    chat_id: str,
    question_content: str,
    user_email: str
):
    """
    Generator function to stream chat responses.
    Yields data in Server-Sent Events (SSE) format.
    """
    start_time_total = time.time()
    current_utc_time = datetime.now(timezone.utc)
    full_answer_for_saving = "" # Để lưu toàn bộ câu trả lời vào DB

    try:
        # --- 1. Xác thực và kiểm tra metadata từ Redis (Tương tự ask_question_service) ---
        meta_key = f"conversation_meta:{chat_id}"
        if not  app_state.redis.exists(meta_key):
            logger.warning(f"Stream: Metadata cho chat_id {chat_id} không tìm thấy.")
            error_payload = {"error": "Chat ID not found or session expired. Please reload."}
            yield f"event: error\ndata: {json.dumps(error_payload)}\n\n"
            return

        user_in_redis_bytes =  app_state.redis.hget(meta_key, "user_id")
        if not user_in_redis_bytes:
            logger.error(f"Stream: user_id không có trong metadata của chat {chat_id}.")
            error_payload = {"error": "Chat metadata corrupted."}
            yield f"event: error\ndata: {json.dumps(error_payload)}\n\n"
            return

        user_in_redis = user_in_redis_bytes.decode()
        if user_in_redis != user_email:
            logger.warning(f"Stream: User {user_email} không được phép truy cập chat {chat_id}.")
            error_payload = {"error": "Unauthorized to access this chat."}
            yield f"event: error\ndata: {json.dumps(error_payload)}\n\n"
            return

        # --- 2. Tiền xử lý câu hỏi (Tương tự) ---
        cleaned_question = minimal_preprocess_for_llm(question_content)

        initial_processing_done_time = time.time()
        logger.info(f"Stream: Initial processing for {chat_id} took {initial_processing_done_time - start_time_total:.2f}s")

        # --- 3. Kiểm tra từ điển thuật ngữ (nếu có, và nó nhanh) ---
        if hasattr(app_state, 'dict') and app_state.dict:
            term_result = search_term_in_dictionary(cleaned_question, app_state.dict)
            if term_result:
                answer_def = term_result.get("definition", "Không thể tìm thấy định nghĩa.")
                assistant_response_time_dict = datetime.now(timezone.utc)
                full_answer_for_saving = answer_def # Gán cho lưu trữ

                # Stream toàn bộ định nghĩa như một chunk
                data_payload = {"token": answer_def, "is_final": True, "source": "dictionary"}
                yield f"data: {json.dumps(data_payload)}\n\n"
                # Có thể gửi event kết thúc riêng
                yield f"event: end_stream\ndata: {{}}\n\n" # Event kết thúc tùy chỉnh

                # Lưu vào Redis và MongoDB (sau khi stream)
                save_chat_to_redis(
                    app_state.redis, chat_id, question_content, full_answer_for_saving, current_utc_time, assistant_response_time_dict
                )
                asyncio.create_task(save_chat_to_mongo( # Chạy nền
                    conversations_collection, chat_id, user_email, question_content, full_answer_for_saving, current_utc_time, assistant_response_time_dict
                ))
                processing_time_dict = round(time.time() - start_time_total, 2)
                logger.info(f"Stream: Dictionary answer for {chat_id} sent in {processing_time_dict:.2f}s.")
                return # Kết thúc generator ở đây

        if not app_state.qa_chain: # qa_chain phải hỗ trợ streaming
            logger.error("Stream: QA Chain chưa được khởi tạo hoặc không hỗ trợ streaming.")
            error_payload = {"error": "Service Unavailable: QA Chain not ready for streaming."}
            yield f"event: error\ndata: {json.dumps(error_payload)}\n\n"
            return

        # --- 4. Lấy lịch sử chat cho Langchain Chain (Tương tự) ---
        try:
            langchain_chat_history = await  get_langchain_chat_history(app_state, chat_id)
            input_data_for_chain = {
                "chat_history": langchain_chat_history.messages,
                "input": cleaned_question
            }
        except Exception as e:
            logger.error(f"Stream: Lỗi khi chuẩn bị chat history (chat_id: {chat_id}): {e}", exc_info=True)
            error_payload = {"error": "Error processing chat history."}
            yield f"event: error\ndata: {json.dumps(error_payload)}\n\n"
            return

        # --- 5. Gọi QA Chain với streaming ---

        if not (hasattr(app_state.qa_chain, 'astream') or hasattr(app_state.qa_chain, 'stream')):
            logger.error(f"Stream: QA Chain (type: {type(app_state.qa_chain)}) không có phương thức astream hoặc stream.")
            error_payload = {"error": "QA Chain does not support streaming."}
            yield f"event: error\ndata: {json.dumps(error_payload)}\n\n"
            return

        chain_stream_method = app_state.qa_chain.astream if hasattr(app_state.qa_chain, 'astream') else app_state.qa_chain.stream

        logger.info(f"Stream: Invoking chain stream for {chat_id}...")
        stream_start_time = time.time()
        chunk_count = 0
        sources_streamed = False # Cờ để chỉ stream sources một lần

        async for chunk in chain_stream_method(input_data_for_chain):

            token = ""
            current_sources = None

            if isinstance(chunk, str):
                token = chunk
            elif hasattr(chunk, 'content'): # Giống AIMessageChunk
                token = chunk.content
            elif isinstance(chunk, dict):
                token = chunk.get("answer") or chunk.get("token") or chunk.get("content") or ""
                # Kiểm tra sources nếu chunk là dict và chưa stream sources
                if not sources_streamed and "source" in chunk:
                    current_sources = chunk["source"]

            if token:
                full_answer_for_saving += token
                data_payload = {"token": token, "is_final": False}
                yield f"data: {json.dumps(data_payload)}\n\n"
                chunk_count += 1

            # Stream sources nếu có và chưa được stream
            if current_sources and not sources_streamed:
                sources_list = []
                for doc in current_sources:
                    if hasattr(doc, 'metadata') and hasattr(doc, 'page_content'):
                        sources_list.append(SourceDocument(
                            source=doc.metadata.get('source', 'N/A'),
                            page_content_preview=doc.page_content[:200] + "..."
                        ).dict()) # Chuyển sang dict để JSON serialize
                if sources_list:
                    source_payload = {"sources": sources_list}
                    yield f"event: sources\ndata: {json.dumps(source_payload)}\n\n" # Event riêng cho sources
                    sources_streamed = True # Đánh dấu đã stream


        stream_end_time = time.time()
        logger.info(f"Stream: Chain streaming for {chat_id} completed in {stream_end_time - stream_start_time:.2f}s with {chunk_count} chunks.")

        # --- Gửi event kết thúc stream ---
        # Frontend có thể dùng event này để biết stream đã hoàn tất.
        # Hoặc, frontend có thể dựa vào một chunk đặc biệt như `{"is_final": true}`
        # Hoặc đơn giản là khi `EventSource.onmessage` không nhận được gì nữa sau một timeout.
        yield f"event: end_stream\ndata: {{ \"message\": \"Stream ended\" }}\n\n"


        # --- 6. Lưu tin nhắn hoàn chỉnh (sau khi stream xong) ---
        assistant_response_time = datetime.now(timezone.utc)
        if not full_answer_for_saving.strip() and chunk_count == 0: # Nếu không có token nào được stream
            full_answer_for_saving = "Tôi không tìm thấy câu trả lời phù hợp."
            # Stream câu trả lời mặc định này nếu chưa có gì
            data_payload = {"token": full_answer_for_saving, "is_final": True}
            yield f"data: {json.dumps(data_payload)}\n\n"
            yield f"event: end_stream\ndata: {{ \"message\": \"Stream ended with default message\" }}\n\n"


        logger.info(f"Stream: Full answer for {chat_id} to be saved: {full_answer_for_saving[:100]}...")
        save_chat_to_redis(
            app_state.redis, chat_id, question_content, full_answer_for_saving, current_utc_time, assistant_response_time
        )
        # Chạy lưu MongoDB ngầm để không block
        asyncio.create_task(save_chat_to_mongo(
            conversations_collection, chat_id, user_email, question_content, full_answer_for_saving, current_utc_time, assistant_response_time
        ))

        # Cập nhật Langchain history (nếu chain memory không tự làm)
        # await langchain_chat_history.aadd_user_message(question_for_chain)
        # await langchain_chat_history.aadd_ai_message(full_answer_for_saving)


    except HTTPException as e: # Bắt HTTPException đã được raise từ các hàm con
        logger.error(f"Stream: HTTPException for chat_id {chat_id}: {e.detail}", exc_info=True)
        error_payload = {"error": e.detail, "status_code": e.status_code}
        yield f"event: error_stream\ndata: {json.dumps(error_payload)}\n\n"
    except Exception as e:
        logger.error(f"Stream: Unhandled exception for chat_id {chat_id}: {e}", exc_info=True)
        error_payload = {"error": "An unexpected server error occurred during streaming."}
        yield f"event: error_stream\ndata: {json.dumps(error_payload)}\n\n"
    finally:
        # Đảm bảo generator kết thúc đúng cách.
        # EventSource trên client sẽ tự động đóng khi generator kết thúc.
        # Hoặc bạn có thể gửi một tín hiệu đóng rõ ràng nếu cần.
        # yield "event: close\ndata: Connection closed by server\n\n" # Không chuẩn SSE, nhưng một số client có thể hiểu
        logger.info(f"Stream: Generator for chat_id {chat_id} finished. Total time: {time.time() - start_time_total:.2f}s")


# Sử dụng GET cho EventSource theo chuẩn, truyền params qua query string
# EventSource chỉ hỗ trợ GET. Nếu bạn BẮT BUỘC phải dùng POST (ví dụ, câu hỏi quá dài cho URL),
# bạn sẽ cần một giải pháp phức tạp hơn, không dùng EventSource trực tiếp trên client
# mà dùng fetch API với ReadableStream và POST.


#helper

from typing import List, Optional,Any

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
async def prepare_chat_history_optimized(
    redis:Any,
    chat_id: str,
    max_messages: int = 10,  # Số lượng cặp tin nhắn (user+AI) tối đa để lấy
    max_tokens: Optional[int] = None, # (Tùy chọn nâng cao) Giới hạn token
    tokenizer: Optional[Any] = None # (Tùy chọn nâng cao) Tokenizer để đếm token
) -> List[BaseMessage]:
    """
    CẢI TIẾN: Lấy N tin nhắn gần nhất từ Redis để làm lịch sử chat.
    - Hiệu quả hơn bằng cách chỉ lấy một phần lịch sử.
    - An toàn hơn bằng cách kiểm soát độ dài ngữ cảnh.

    Args:
        redis: Client Redis bất đồng bộ.
        chat_id: ID của cuộc trò chuyện.
        max_messages: Số lượng tin nhắn tối đa để lấy từ cuối (ví dụ: 10 tin nhắn gần nhất).
        max_tokens: (Nâng cao) Giới hạn tổng số token của lịch sử.
        tokenizer: (Nâng cao) Tokenizer để sử dụng với max_tokens.

    Returns:
        Một danh sách các đối tượng tin nhắn của LangChain (HumanMessage, AIMessage).
    """
    messages_key = f"conversation_messages:{chat_id}"

    # 1. Lấy N tin nhắn gần nhất từ Redis
    # lrange(key, -N, -1) sẽ lấy N phần tử cuối cùng của list.
    # Lấy nhiều hơn một chút để đảm bảo có cặp user/ai hoàn chỉnh.
    num_to_fetch = max_messages + 2
    try:
        # Sử dụng lrange để lấy các tin nhắn gần nhất, hiệu quả hơn nhiều so với lấy tất cả
        raw_messages_json = await redis.lrange(messages_key, -num_to_fetch, -1)
        if not raw_messages_json:
            return []
    except Exception as e:
        logger.error(f"Lỗi khi đọc lịch sử chat từ Redis cho chat_id {chat_id}: {e}")
        return []

    # 2. Xây dựng danh sách tin nhắn cho LangChain
    langchain_messages: List[BaseMessage] = []
    total_tokens = 0

    # Lặp ngược từ cuối (tin nhắn mới nhất) để xử lý
    for msg_json_str in reversed(raw_messages_json):
        try:
            msg_data = json.loads(msg_json_str)
            content = msg_data.get("content", "")

            # (Tùy chọn nâng cao) Kiểm tra giới hạn token
            if max_tokens and tokenizer:
                num_tokens = len(tokenizer.encode(content))
                if total_tokens + num_tokens > max_tokens:
                    logger.warning(f"Đã đạt giới hạn token ({max_tokens}) cho lịch sử chat. Dừng lại.")
                    break # Dừng thêm tin nhắn
                total_tokens += num_tokens

            # Tạo đối tượng tin nhắn phù hợp
            if msg_data.get("role") == "user":
                langchain_messages.append(HumanMessage(content=content))
            elif msg_data.get("role") == "assistant":
                langchain_messages.append(AIMessage(content=content))

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Lỗi khi parse tin nhắn từ Redis: {e}. Bỏ qua tin nhắn này.")
            continue

    # 3. Đảo ngược lại danh sách để có đúng thứ tự (cũ -> mới)
    langchain_messages.reverse()

    # Cắt lại theo max_messages cuối cùng để đảm bảo số lượng chính xác
    return langchain_messages[-max_messages:]