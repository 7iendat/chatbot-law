from fastapi import APIRouter, Depends, HTTPException, Request
from schemas.chat import QueryRequest, AnswerResponse, ChatHistoryResponse
from schemas.user import UserOut
from dependencies import get_current_user
from services.chat_service import ask_question_service, stream_chat_generator
from utils.utils import delete_chat_from_redis
from dependencies import get_app_state
import logging
import uuid
from redis.asyncio import Redis
from datetime import datetime, timezone
from db.mongoDB import conversations_collection
from fastapi.responses import StreamingResponse
from schemas.chat import Message,ConversationResponse
from typing import List
# Thiết lập logger
logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/create-chat")
async def create_chat(
    fastapi_request: Request, # Sử dụng Request từ FastAPI
    current_user: UserOut = Depends(get_current_user) # Sử dụng User model của bạn
):
    app_state = get_app_state(request=fastapi_request)
    redis_client: Redis = app_state.redis # Nên đặt tên rõ ràng là redis_client


    chat_id = str(uuid.uuid4())
    current_utc_time = datetime.now(timezone.utc) # Sử dụng UTC

    # --- Lưu metadata vào Redis với key đã thống nhất ---
    meta_key = f"conversation_meta:{chat_id}"
    conversation_meta_data = {
        "user_id": current_user.email, # Sử dụng key 'user_id' cho nhất quán
        "created_at": current_utc_time.isoformat(), # Lưu dưới dạng ISO string
        "updated_at": current_utc_time.isoformat(), # Ban đầu giống created_at
        "message_count": 0 # Số lượng tin nhắn ban đầu
    }

    try:
        # Sử dụng await nếu redis_client là async
        if hasattr(redis_client, 'hmset_async'): # Kiểm tra phương thức async (ví dụ)
            await redis_client.hmset(meta_key, conversation_meta_data)
            await redis_client.expire(meta_key, 86400) # TTL: 24 giờ
        else: # Client đồng bộ
            await redis_client.hmset(meta_key, conversation_meta_data)
            await redis_client.expire(meta_key, 86400) # TTL: 24 giờ

        logger.info(f"Đã tạo metadata cho chat_id {chat_id} trong Redis với key {meta_key}.")
    except Exception as e:
        logger.error(f"Lỗi khi lưu metadata vào Redis cho chat {chat_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Lỗi khi tạo metadata cho cuộc hội thoại.")

    # --- Lưu hội thoại rỗng vào MongoDB ---
    # (Đảm bảo messages ban đầu là list rỗng cho key messages chính của bạn)
    messages_key_in_mongo = "messages" # Key lưu trữ danh sách tin nhắn trong MongoDB

    conversation_doc = {
        "user_id": current_user.email,
        "conversation_id": chat_id,
        messages_key_in_mongo: [], # Danh sách tin nhắn rỗng
        "created_at": current_utc_time, # Lưu kiểu datetime object
        "updated_at": current_utc_time  # Lưu kiểu datetime object
    }

    try:
        conversations_collection.insert_one(conversation_doc)
        logger.info(f"Đã tạo hội thoại rỗng {chat_id} trong MongoDB cho user {current_user.email}.")
    except Exception as e:
        logger.error(f"Lỗi khi tạo hội thoại rỗng trong MongoDB cho chat {chat_id}: {e}", exc_info=True)
        # Cân nhắc xóa key meta trong Redis nếu MongoDB thất bại để tránh trạng thái không nhất quán
        try:
            if hasattr(redis_client, 'delete_async'):
                await redis_client.delete(meta_key)
            else:
                await redis_client.delete(meta_key)
            logger.info(f"Đã xóa meta key {meta_key} khỏi Redis do lỗi MongoDB.")
        except Exception as redis_del_err:
            logger.error(f"Lỗi khi xóa meta key {meta_key} khỏi Redis: {redis_del_err}")

        raise HTTPException(status_code=500, detail="Lỗi khi tạo cuộc hội thoại trong cơ sở dữ liệu.")

    return {"chat_id": chat_id}


@router.post("", response_model=AnswerResponse)
async def chat_message(request_body: QueryRequest,request: Request, user:UserOut=Depends(get_current_user)):
    app_state = get_app_state(request=request)
    result = await ask_question_service(app_state,request_body, user)

    if not result:
        raise HTTPException(status_code=500, detail="Error during QA Chain invocation")
    return result

@router.get("/stream") # Đổi thành GET
async def stream_chat_endpoint(
    chat_id: str,  # Lấy từ query param
    input: str,    # Lấy từ query param (tên param này phải khớp với FE)
    request: Request,
    user: UserOut = Depends(get_current_user) # Sửa kiểu user
):
    app_state = get_app_state(request=request)
    user_email = getattr(user, 'email', str(user)) # Lấy email an toàn

    # Kiểm tra input cơ bản
    if not chat_id or not input:
        raise HTTPException(status_code=400, detail="chat_id and input are required.")

    # Sử dụng EventSourceResponse (từ sse-starlette, cài đặt: pip install sse-starlette)
    # Nó xử lý các chi tiết của SSE tốt hơn StreamingResponse thô.
    # return EventSourceResponse(stream_chat_generator(app_state, chat_id, input, user_email))

    # Hoặc dùng StreamingResponse trực tiếp (đơn giản hơn nhưng ít tính năng SSE hơn)
    return StreamingResponse(
        stream_chat_generator(app_state, chat_id, input, user_email),
        media_type="text/event-stream"
    )

@router.delete("/chats/{chat_id}")
async def delete_chat(chat_id: str, request: Request, user: UserOut = Depends(get_current_user)):
    app_state = get_app_state(request=request)
    redis = app_state.redis

    meta_key = f"conversation_meta:{chat_id}"
    # Kiểm tra quyền trước khi xóa
    user_in_chat =  redis.hget(meta_key, "user_id")
    if user_in_chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")

    if user_in_chat.decode() != user.email:
        raise HTTPException(status_code=403, detail="Unauthorized")

    # Xóa chat
    delete_chat_from_redis(redis, chat_id)
    # Xóa hội thoại trong MongoDB
    result =  conversations_collection.delete_one({"conversation_id": chat_id, "user_id": user.email})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Chat not found in MongoDB")

    return {"message": "Chat deleted successfully"}



@router.get("/conversations", response_model=List[ConversationResponse])
async def get_conversations(user: UserOut = Depends(get_current_user)):
    try:
        logger.info(f"Attempting to get conversations for user: {user.email}")
        db_conversations_cursor = conversations_collection.find({"user_id": user.email})

        response_list = []


        for conv_doc in db_conversations_cursor: # Hoặc conversation_docs nếu bạn debug
            logger.debug(f"Processing conversation doc: {conv_doc}") # Log toàn bộ doc để xem cấu trúc
            all_messages = conv_doc.get("messages", [])
            logger.debug(f"Messages for this conversation: {all_messages}")


            response_list.append({
                "conversation_id": conv_doc["conversation_id"],
                "created_at": conv_doc["created_at"],
                "updated_at": conv_doc["updated_at"],
                "messages": all_messages # Trả về toàn bộ messages như yêu cầu
            })
        logger.info(f"Successfully processed {len(response_list)} conversations.")
        return response_list
    except Exception as e:
        logger.error(f"Error in get_conversations for user {user.email}: {e}", exc_info=True) # exc_info=True sẽ log cả traceback
        raise HTTPException(status_code=500, detail="An error occurred while fetching conversations.")


@router.get("/c/{chat_id}", response_model=ChatHistoryResponse)
async def load_conversation_and_sync_redis(
    fastapi_request: Request, # Đổi tên biến request
    chat_id: str, # Lấy trực tiếp từ path param
    current_user: UserOut = Depends(get_current_user) # Sử dụng User model
):
    app_state = get_app_state(request=fastapi_request)
    redis_client = app_state.redis # Nên là client async nếu có thể

    # 1. Kiểm tra hội thoại trong MongoDB
    try:
        conversation_doc = conversations_collection.find_one(
            {"conversation_id": chat_id, "user_id": current_user.email}
        )
        if not conversation_doc:
            logger.warning(f"Hội thoại {chat_id} không tồn tại hoặc không thuộc user {current_user.email}")
            raise HTTPException(
                status_code=404,
                detail="Hội thoại không tồn tại hoặc bạn không có quyền truy cập"
            )
    except Exception as e:
        logger.error(f"Lỗi MongoDB khi kiểm tra hội thoại {chat_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Lỗi kết nối cơ sở dữ liệu MongoDB")

    # 2. Chuẩn bị lịch sử tin nhắn từ MongoDB để trả về và nạp vào Redis
    raw_messages_from_db = conversation_doc.get("messages", [])
    validated_history_for_response: List[Message] = []
    for msg_data in raw_messages_from_db:
        try:
            # Validate và chuyển đổi timestamp nếu cần (Pydantic sẽ tự làm nếu input là datetime obj)
            validated_history_for_response.append(Message(**msg_data))
        except Exception as p_err:
            logger.warning(f"Bỏ qua message không hợp lệ trong chat {chat_id} từ DB: {msg_data}. Lỗi: {p_err}")


    # 3. Nạp/Đồng bộ tin nhắn và metadata vào Redis (Sử dụng key thống nhất)
    messages_redis_key = f"conversation_messages:{chat_id}"
    meta_redis_key = f"conversation_meta:{chat_id}"

    try:
        # Sử dụng pipeline cho hiệu quả
        # Giả sử redis_client là async
        with redis_client.pipeline() as pipe:
            pipe.delete(messages_redis_key) # Xóa messages cũ để nạp lại toàn bộ
            if validated_history_for_response:
                for msg_model in validated_history_for_response:
                    # Đảm bảo lưu trữ theo cấu trúc Pydantic `Message`
                    pipe.rpush(messages_redis_key, msg_model.model_dump_json()) # Pydantic V2
                    # hoặc .json() cho Pydantic V1
            pipe.expire(messages_redis_key, 86400)  # TTL: 24 giờ

            # Cập nhật/Tạo mới metadata
            # Lấy created_at, updated_at từ document MongoDB
            created_at_iso = conversation_doc["created_at"].isoformat()
            updated_at_iso = conversation_doc["updated_at"].isoformat()

            conversation_meta_data = {
                "user_id": current_user.email,
                "created_at": created_at_iso,
                "updated_at": updated_at_iso,
                "message_count": len(validated_history_for_response)
            }
            # Xóa meta cũ và đặt lại, hoặc dùng hmset để cập nhật
            pipe.delete(meta_redis_key)
            pipe.hset(meta_redis_key, conversation_meta_data)
            pipe.expire(meta_redis_key, 86400)


            pipe.execute()
        logger.info(f"Đã nạp và đồng bộ hội thoại {chat_id} vào Redis với {len(validated_history_for_response)} tin nhắn.")

    except Exception as e:
        logger.error(f"Lỗi khi nạp hội thoại {chat_id} vào Redis: {e}", exc_info=True)

    # 4. Trả về response
    return ChatHistoryResponse(
        chat_id=chat_id,
        history=validated_history_for_response,
        created_at=conversation_doc["created_at"], # Lấy từ doc MongoDB
        updated_at=conversation_doc["updated_at"], # Lấy từ doc MongoDB
        user_id=current_user.email
    )