from fastapi import Depends, HTTPException
from schemas.chat import QueryRequest, AnswerResponse, SourceDocument
from dependencies import get_current_user
import time
import json
from utils.utils import save_chat_to_redis, search_term_in_dictionary, process_with_groq,preprocess_input, save_chat_to_mongo, get_langchain_chat_history
import os
import logging
from langchain_community.chat_message_histories import RedisChatMessageHistory
from db.mongoDB import conversations_collection
from datetime import datetime, timezone
from schemas.chat import Message
import asyncio

logger = logging.getLogger(__name__)
# def ask_question_service(app_state, request: QueryRequest, user: str=Depends(get_current_user)):
#     chat_id = request.chat_id
#     question = request.input
#     if not app_state.redis.exists(f"chat:{chat_id}:meta"):
#         raise HTTPException(status_code=404, detail="Chat ID not found")

#     user_in_redis = app_state.redis.hget(f"chat:{chat_id}:meta", "user")
#     if not user_in_redis:
#         raise HTTPException(status_code=404, detail="Chat not found")

#     if user_in_redis.decode() != user.email:
#         raise HTTPException(status_code=403, detail="Unauthorized")

#     start_time = time.time()
#     # question = preprocess_vietnamese_query(question)['accented']
#     cleaned_question = preprocess_input(question)
#     question_processed = process_with_groq(app_state.process_input_llm,cleaned_question)

#     logger.info(f"check: => processed question: {question_processed}")
#     logger.info(f"check: => cleaned question: {cleaned_question}")

#     if not app_state.qa_chain:
#         raise HTTPException(status_code=503, detail="Service Unavailable: QA Chain chưa sẵn sàng.")

#     term_result = search_term_in_dictionary(question_processed, app_state.dict)

#     try:
#         # Initialize Redis memory for this chat_id
#         redis_url = os.environ.get("REDIS_URL")
#         if not redis_url:
#             logger.error("REDIS_URL not set.")
#             raise ValueError("Redis URL is required.")

#         chat_history = RedisChatMessageHistory(
#             url=redis_url,
#             session_id=chat_id,
#             ttl=86400,
#         )
#         # This function appears to be unused and may be causing the issue if referenced elsewhere
#         # def get_chat_history(_):
#         #     return chat_history.messages

#         if term_result:
#             answer_def = term_result.get("definition", "Không thể tạo câu trả lời.")
#             save_chat_to_redis(app_state.redis, chat_id, question_processed, answer_def)
#             return AnswerResponse(
#                 answer=answer_def,
#                 sources='legal_terms',
#                 processing_time=0.0
#             )

#         # Prepare input for qa_chain
#         input_data = {
#             "chat_history": chat_history.messages if chat_history.messages else [],
#             "input": question_processed
#         }
#         logger.debug(f"Input data: {input_data}, type: {type(input_data)}")

#         # Call qa_chain
#         try:
#             # This is likely where the error is occurring
#             result = app_state.qa_chain.invoke(input_data)
#             logger.info(f"🔸QA Chain raw result for chat_id {chat_id}: {result}")

#             # Handle unexpected output format
#             if isinstance(result, dict) and "answer" in result:
#                 if isinstance(result["answer"], str):
#                     try:
#                         # Try parsing answer as JSON
#                         parsed_answer = json.loads(result["answer"])
#                         if isinstance(parsed_answer, dict) and "answer" in parsed_answer:
#                             logger.warning(f"🔸Parsed JSON answer for chat_id {chat_id}: {parsed_answer}")
#                             result = {"answer": parsed_answer["answer"]}
#                         else:
#                             logger.warning(f"🔸Answer is a JSON string but not a valid answer dict: {result['answer']}")
#                     except json.JSONDecodeError:
#                         # Not a JSON string, use as is
#                         pass
#                 elif "question" in result:
#                     logger.warning(f"🔸Unexpected 'question' key in result for chat_id {chat_id}: {result}")
#                     result = {"answer": "Không thể xử lý câu hỏi. Vui lòng thử lại."}
#                 elif "raw" in result:
#                     logger.warning(f"🔸Unexpected 'raw' key in result for chat_id {chat_id}: {result}")
#                     result = {"answer": result["raw"]}

#             if not isinstance(result, dict) or "answer" not in result:
#                 logger.error(f"🔸Invalid QA Chain result for chat_id {chat_id}: {result}")
#                 raise ValueError("QA Chain did not return a valid response.")

#             response = result["answer"]
#             if not isinstance(response, str):
#                 logger.warning(f"🔸Response is not a string for chat_id {chat_id}: {response}")
#                 response = str(response)
#         except Exception as chain_error:
#             logger.error(f"🔸QA Chain error for chat_id {chat_id}: {chain_error}")
#             raise HTTPException(status_code=500, detail=f"QA Chain processing failed: {str(chain_error)}")

#         # Save response to chat history
#         chat_history.add_user_message(question_processed)
#         chat_history.add_ai_message(response)
#         end_time = time.time()

#         # Save to Redis
#         save_chat_to_redis(app_state.redis, chat_id, question_processed, response)
#         # Save to MongoDB
#         save_chat_to_mongo(
#             conversations_collection,
#             chat_id,
#             user.email,
#             question,
#             response

#         )
#         sources_list = []
#         if result.get("source_documents"):
#             for doc in result["source_documents"]:
#                 sources_list.append(SourceDocument(
#                     source=doc.metadata.get('source', 'N/A'),
#                     page_content_preview=doc.page_content[:200] + "..."
#                 ))

#         logger.info(f"🔸Answer: {response}")

#         return AnswerResponse(
#             answer=response,
#             sources=sources_list if sources_list else None,
#             processing_time=round(end_time - start_time, 2)
#         )
#     except Exception as e:
#         logger.error(f"Error during QA Chain invocation: {e}")
#         raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


async def ask_question_service(app_state, request: QueryRequest, user: str = Depends(get_current_user)):
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
    cleaned_question = preprocess_input(question_content)
    # Sử dụng câu hỏi đã xử lý bởi Groq cho chain, nhưng lưu câu hỏi gốc/cleaned vào DB/Redis
    question_for_chain = process_with_groq(app_state.process_input_llm, cleaned_question)
    logger.info(f"Question for chain (chat_id: {chat_id}): {question_for_chain}")


    # --- 3. Kiểm tra từ điển thuật ngữ (nếu có) ---
    if hasattr(app_state, 'dict') and app_state.dict: # Kiểm tra xem app_state.dict có tồn tại không
        term_result = search_term_in_dictionary(question_for_chain, app_state.dict)
        if term_result:
            answer_def = term_result.get("definition", "Không thể tìm thấy định nghĩa.")
            assistant_response_time = datetime.now(timezone.utc)

            # Lưu vào Redis và MongoDB
            save_chat_to_redis(
                app_state.redis, chat_id, question_content, answer_def, current_utc_time, assistant_response_time
            )
            save_chat_to_mongo(
                conversations_collection, chat_id, user.email, question_content, answer_def, current_utc_time, assistant_response_time
            )
            return AnswerResponse(
                answer=answer_def,
                sources='legal_terms', # Hoặc cấu trúc SourceDocument
                processing_time=round(time.time() - start_time, 2)
            )

    if not app_state.qa_chain:
        logger.error("QA Chain chưa được khởi tạo.")
        raise HTTPException(status_code=503, detail="Service Unavailable: QA Chain not ready.")

    # --- 4. Lấy lịch sử chat cho Langchain Chain ---
    # RedisChatMessageHistory sử dụng key riêng của nó, ví dụ: "message_store:{session_id}"
    # Nó không trực tiếp đọc từ "conversation_messages:{chat_id}" trừ khi bạn cấu hình key_prefix
    # Cách tiếp cận: Để RedisChatMessageHistory đọc/ghi vào key riêng của nó cho chain.
    # Dữ liệu trong "conversation_messages:{chat_id}" là "source of truth" để hiển thị và nạp lại.
    try:
        redis_url = os.environ.get("REDIS_URL_LANGCHAIN", os.environ.get("REDIS_URL")) # Ưu tiên URL riêng cho Langchain nếu có
        if not redis_url:
            logger.error("REDIS_URL or REDIS_URL_LANGCHAIN not set for RedisChatMessageHistory.")
            raise ValueError("Redis URL for chat history is required.")

        # Tạo RedisChatMessageHistory với session_id là chat_id
        # Langchain sẽ tự động tạo key kiểu message_store:your_chat_id
        # Hoặc bạn có thể thử set key_prefix nếu class hỗ trợ, nhưng thường là session_id.
        langchain_chat_history = RedisChatMessageHistory(
            url=redis_url,
            session_id=chat_id, # Quan trọng: phải là chat_id hiện tại
            ttl=86400,
        )
        # Lấy tin nhắn đã có trong history của Langchain (nếu có từ lần trước)
        # Lưu ý: Đây là history mà Langchain quản lý, có thể khác với "conversation_messages:{chat_id}"
        # nếu bạn không đồng bộ chúng.
        # Để đơn giản, khi load conversation (/c/{chat_id}), bạn có thể cũng nạp message vào key của Langchain.
        # Hoặc, dựa vào "conversation_messages:{chat_id}" để tạo lại Langchain history mỗi lần.
        # Cách tiếp cận ở đây: Dùng history mà Langchain đang có.

        # Khởi tạo lịch sử cho Langchain từ key messages chính của chúng ta (ĐỒNG BỘ HÓA)
        # Điều này đảm bảo Langchain sử dụng đúng lịch sử mà user thấy.
        messages_key = f"conversation_messages:{chat_id}"
        raw_messages_from_our_redis =  app_state.redis.lrange(messages_key, 0, -1)

        # Xóa history cũ trong Langchain key trước khi add lại, để tránh trùng lặp nếu user reload.
        langchain_chat_history.clear() # Dùng `clear()` nếu redis client đồng bộ

        for msg_json_str in raw_messages_from_our_redis:
            msg_data = json.loads(msg_json_str.decode()) # decode bytes to str
            message = Message(**msg_data) # Validate lại qua Pydantic
            if message.role == "user":
                langchain_chat_history.add_user_message(message.content)
            elif message.role == "assistant":
                langchain_chat_history.add_ai_message(message.content)

        input_data_for_chain = {
            "chat_history":  langchain_chat_history.messages, # Lấy messages đã được đồng bộ
            "input": question_for_chain
        }

    except Exception as e:
        logger.error(f"Lỗi khi chuẩn bị chat history cho Langchain (chat_id: {chat_id}): {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Lỗi xử lý lịch sử chat.")


    # --- 5. Gọi QA Chain ---
    try:
        logger.debug(f"Input to QA Chain (chat_id: {chat_id}): {input_data_for_chain}")
        # Đảm bảo qa_chain là async nếu bạn dùng await
        chain_result =  app_state.qa_chain.invoke(input_data_for_chain) # Giả sử chain có phương thức async
        # chain_result = app_state.qa_chain.invoke(input_data_for_chain) # Nếu chain là đồng bộ
        logger.info(f"QA Chain raw result (chat_id: {chat_id}): {chain_result}")

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
    save_chat_to_mongo(
        conversations_collection, chat_id, user.email, question_content, assistant_response_content, current_utc_time, assistant_response_time
    )
    # Langchain's RedisChatMessageHistory cũng sẽ tự lưu nếu chain được cấu hình với memory.
    # Tuy nhiên, việc chúng ta add lại vào langchain_chat_history ở trên là để đảm bảo ngữ cảnh cho lần gọi này.
    # Nếu chain của bạn tự động cập nhật memory (ví dụ ConversationBufferMemory với RedisChatMessageHistory),
    # thì không cần add_user_message/add_ai_message vào langchain_chat_history sau khi invoke.
    # Nhưng để chắc chắn, bạn có thể thêm:
    # await langchain_chat_history.aadd_user_message(question_for_chain)
    # await langchain_chat_history.aadd_ai_message(assistant_response_content)
    # Hãy kiểm tra tài liệu của memory component bạn đang dùng trong chain.

    end_time = time.time()

    # --- 7. Chuẩn bị sources (nếu có) ---
    sources_list = []
    if isinstance(chain_result, dict) and chain_result.get("source_documents"):
        for doc in chain_result["source_documents"]:
            if hasattr(doc, 'metadata') and hasattr(doc, 'page_content'):
                sources_list.append(SourceDocument(
                    source=doc.metadata.get('source', 'N/A'),
                    page_content_preview=doc.page_content[:200] + "..."
                ))

    logger.info(f"Trả lời cho chat {chat_id} bởi user {user.email}: {assistant_response_content[:100]}...")
    return AnswerResponse(
        answer=assistant_response_content,
        sources=sources_list if sources_list else None,
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
        if not await app_state.redis.exists(meta_key):
            logger.warning(f"Stream: Metadata cho chat_id {chat_id} không tìm thấy.")
            error_payload = {"error": "Chat ID not found or session expired. Please reload."}
            yield f"event: error\ndata: {json.dumps(error_payload)}\n\n"
            return

        user_in_redis_bytes = await app_state.redis.hget(meta_key, "user_id")
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
        cleaned_question = preprocess_input(question_content)
        # Sử dụng Groq để xử lý trước (nếu cần và nó nhanh)
        # Cân nhắc: Nếu process_with_groq chậm, nó có thể làm delay chunk đầu tiên.
        question_for_chain = process_with_groq(app_state.process_input_llm, cleaned_question)
        logger.info(f"Stream: Question for chain (chat_id: {chat_id}): {question_for_chain}")
        initial_processing_done_time = time.time()
        logger.info(f"Stream: Initial processing for {chat_id} took {initial_processing_done_time - start_time_total:.2f}s")


        # --- 3. Kiểm tra từ điển thuật ngữ (nếu có, và nó nhanh) ---
        if hasattr(app_state, 'dict') and app_state.dict:
            term_result = search_term_in_dictionary(question_for_chain, app_state.dict)
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
            # get_langchain_chat_history nên là async nếu redis client là async
            langchain_chat_history = await get_langchain_chat_history(app_state, chat_id)
            input_data_for_chain = {
                "chat_history": langchain_chat_history.messages,
                "input": question_for_chain
            }
        except Exception as e:
            logger.error(f"Stream: Lỗi khi chuẩn bị chat history (chat_id: {chat_id}): {e}", exc_info=True)
            error_payload = {"error": "Error processing chat history."}
            yield f"event: error\ndata: {json.dumps(error_payload)}\n\n"
            return

        # --- 5. Gọi QA Chain với streaming ---
        # Điều này phụ thuộc vào cách chain của bạn hỗ trợ streaming.
        # Giả sử chain.astream(input) hoặc chain.stream(input) trả về một async generator.
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
            # Xử lý chunk từ chain. Cấu trúc của chunk phụ thuộc vào chain của bạn.
            # Ví dụ phổ biến:
            # 1. Chunk là string (token):
            #    token = chunk
            # 2. Chunk là dict với key 'answer' hoặc 'content' cho token:
            #    token = chunk.get("answer") or chunk.get("content") or ""
            # 3. Chunk là một AIMessageChunk (Langchain)
            #    token = chunk.content
            # 4. Có thể có 'source_documents' trong chunk cuối hoặc chunk riêng.

            token = ""
            current_sources = None

            if isinstance(chunk, str):
                token = chunk
            elif hasattr(chunk, 'content'): # Giống AIMessageChunk
                token = chunk.content
            elif isinstance(chunk, dict):
                token = chunk.get("answer", chunk.get("token", "")) # Ưu tiên "answer", rồi "token"
                # Kiểm tra sources nếu chunk là dict và chưa stream sources
                if not sources_streamed and "source_documents" in chunk:
                    current_sources = chunk["source_documents"]

            if token:
                full_answer_for_saving += token
                data_payload = {"token": token}
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

            # Thêm một chút delay nhỏ để client có thời gian xử lý, tránh flood
            # await asyncio.sleep(0.01) # Tùy chọn, có thể không cần thiết


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
        yield f"event: error\ndata: {json.dumps(error_payload)}\n\n"
    except Exception as e:
        logger.error(f"Stream: Unhandled exception for chat_id {chat_id}: {e}", exc_info=True)
        error_payload = {"error": "An unexpected server error occurred during streaming."}
        yield f"event: error\ndata: {json.dumps(error_payload)}\n\n"
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