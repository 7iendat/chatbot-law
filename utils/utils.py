# utils.py
import os
import logging
import regex as re
import json
from typing import List, Optional
from schemas.chat import  Message
from redis.asyncio import Redis
import bcrypt
from datetime import datetime, timedelta
from jose import jwt
from config import SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES
from typing import List, Dict,  Optional
from unidecode import unidecode
from db.mongoDB import user_collection
import secrets
from fastapi import HTTPException, status
from langchain_community.chat_message_histories import RedisChatMessageHistory
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)



async def save_chat_to_redis(
    r: Redis, # Hoặc redis.asyncio.Redis
    chat_id: str,
    user_question_content: str, # Nội dung câu hỏi gốc hoặc đã xử lý (tùy bạn)
    assistant_answer_content: str,
    user_question_timestamp: datetime, # Cung cấp timestamp
    assistant_answer_timestamp: datetime # Cung cấp timestamp
) -> bool:
    """
    Lưu tin nhắn mới của người dùng và trợ lý vào Redis với định dạng chuẩn hóa.
    Cập nhật 'updated_at' và 'message_count' trong metadata.
    """
    if not all([chat_id, user_question_content, assistant_answer_content]):
        logger.error("chat_id, user_question_content, và assistant_answer_content không được rỗng.")
        # raise ValueError("Đầu vào không hợp lệ.") # Hoặc trả về False
        return False

    messages_key = f"conversation_messages:{chat_id}"
    meta_key = f"conversation_meta:{chat_id}"

    try:
        # Tạo Pydantic models cho tin nhắn
        user_message = Message(role="user", content=user_question_content, timestamp=user_question_timestamp)
        assistant_message = Message(role="assistant", content=assistant_answer_content, timestamp=assistant_answer_timestamp)

        # Sử dụng pipeline cho các thao tác Redis
        pipe = await r.pipeline()
        await pipe.rpush(messages_key, user_message.model_dump_json())  # Pydantic V2
        await pipe.rpush(messages_key, assistant_message.model_dump_json()) # Pydantic V2
        # Hoặc .json() cho Pydantic V1

        # Đặt TTL cho key messages nếu nó mới được tạo (hoặc luôn refresh TTL)
        # Nếu bạn muốn TTL chỉ đặt một lần, bạn cần kiểm tra sự tồn tại của key trước
        # hoặc kiểm tra llen trước khi push (phức tạp hơn với pipeline).
        # Cách đơn giản là luôn đặt lại TTL.
        await pipe.expire(messages_key, 86400) # 24 giờ

        # Cập nhật metadata
        await pipe.hset(meta_key, "updated_at", assistant_answer_timestamp.isoformat())
        await pipe.hincrby(meta_key, "message_count", 2) # Tăng số lượng tin nhắn
        await pipe.expire(meta_key, 86400) # Refresh TTL cho meta

        await pipe.execute()

        logger.info(f"Đã lưu 2 tin nhắn mới vào {messages_key} và cập nhật {meta_key}.")
        return True

    except RedisError as e:
        logger.error(f"Lỗi Redis khi lưu tin nhắn cho chat_id {chat_id}: {e}", exc_info=True)
        raise # Re-raise để service xử lý HTTPException
    except Exception as e:
        logger.error(f"Lỗi không mong muốn khi lưu vào Redis cho chat_id {chat_id}: {e}", exc_info=True)
        # raise # Hoặc trả về False tùy theo cách bạn muốn xử lý
        return False


async def get_redis_history(r: Redis, chat_id: str, max_messages: int = 100) -> List[Message]:
    """
    Lấy lịch sử hội thoại từ Redis với định dạng chuẩn hóa.

    Args:
        r (Redis): Đối tượng Redis client.
        chat_id (str): ID của hội thoại.
        max_messages (int): Số tin nhắn tối đa trả về (mặc định 100).

    Returns:
        List[Message]: Danh sách tin nhắn (role, content, timestamp).

    Raises:
        ValueError: Nếu chat_id rỗng.
        redis.RedisError: Nếu có lỗi khi tương tác với Redis.
    """
    # Kiểm tra đầu vào
    if not chat_id:
        logger.error("chat_id không được rỗng")
        raise ValueError("chat_id là bắt buộc")

    messages_key = f"conversation_messages:{chat_id}"
    try:
        # Kiểm tra kết nối Redis
        r.ping()

        # Lấy tin nhắn (giới hạn max_messages từ cuối)
        history_raw =await r.lrange(messages_key, -max_messages, -1)
        chat_history = []

        for item in history_raw:
            try:
                parsed = json.loads(item)
                if not isinstance(parsed, dict):
                    logger.warning(f"Tin nhắn không phải dict trong {messages_key}: {item}")
                    continue

                # Kiểm tra các trường bắt buộc
                role = parsed.get("role")
                content = parsed.get("content")
                timestamp = parsed.get("timestamp")
                if not all([role, content, timestamp]):
                    logger.warning(f"Tin nhắn thiếu trường trong {messages_key}: {parsed}")
                    continue

                # Đảm bảo role hợp lệ
                if role not in ["user", "assistant"]:
                    logger.warning(f"Role không hợp lệ trong {messages_key}: {role}")
                    continue

                chat_history.append(Message(
                    role=role,
                    content=content,
                    timestamp=timestamp
                ))
            except json.JSONDecodeError as e:
                logger.error(f"Lỗi parse JSON trong {messages_key}: {item}, lỗi: {e}")
                continue
            except Exception as e:
                logger.error(f"Lỗi xử lý tin nhắn trong {messages_key}: {item}, lỗi: {e}")
                continue

        logger.info(f"Lấy {len(chat_history)} tin nhắn từ {messages_key}")
        return chat_history

    except r.RedisError as e:
        logger.error(f"Lỗi khi lấy lịch sử từ Redis cho chat_id {chat_id}: {e}")
        raise
    except Exception as e:
        logger.error(f"Lỗi không mong muốn khi lấy lịch sử từ Redis cho chat_id {chat_id}: {e}")
        return []


async def delete_chat_from_redis(r: Redis, chat_id: str) -> bool:
    """
    Xóa dữ liệu hội thoại và metadata từ Redis.

    Args:
        r (Redis): Đối tượng Redis client.
        chat_id (str): ID của hội thoại.

    Returns:
        bool: True nếu xóa thành công, False nếu thất bại.

    Raises:
        ValueError: Nếu chat_id rỗng.
        redis.RedisError: Nếu có lỗi khi tương tác với Redis.
    """
    # Kiểm tra đầu vào
    if not chat_id:
        logger.error("chat_id không được rỗng")
        raise ValueError("chat_id là bắt buộc")

    redis_key = f"conversation:{chat_id}"
    meta_key = f"chat:{chat_id}:meta"

    try:
        # Kiểm tra kết nối Redis
        r.ping()

        # Kiểm tra sự tồn tại của các key
        keys_to_delete = []
        if await r.exists(redis_key):
            keys_to_delete.append(redis_key)
        if await r.exists(meta_key):
            keys_to_delete.append(meta_key)

        if not keys_to_delete:
            logger.info(f"Không tìm thấy dữ liệu cho chat_id {chat_id} trong Redis")
            return True  # Không có gì để xóa, coi như thành công

        # Xóa các key
        deleted_count = await r.delete(*keys_to_delete)
        logger.info(f"Đã xóa {deleted_count} key cho chat_id {chat_id}: {keys_to_delete}")
        return True

    except r.RedisError as e:
        logger.error(f"Lỗi khi xóa dữ liệu từ Redis cho chat_id {chat_id}: {e}")
        raise
    except Exception as e:
        logger.error(f"Lỗi không mong muốn khi xóa dữ liệu từ Redis cho chat_id {chat_id}: {e}")
        return False

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.now() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def create_refresh_token(email: str) -> str:
    """
    Tạo và lưu refresh token vào cơ sở dữ liệu.

    Args:
        email (str): Địa chỉ email của người dùng.

    Returns:
        str: Refresh token được tạo.

    Raises:
        HTTPException: Nếu có lỗi khi lưu token.
    """
    try:
        # Generate refresh token
        refresh_token = secrets.token_urlsafe(32)
        expire = datetime.now() + timedelta(days=7)


        # Store refresh token in database
        result =  user_collection.update_one(
            {"email": email.lower()},
            {
                "$set": {
                    "refresh_token": refresh_token,
                    "refresh_token_expiry": expire,
                    "refresh_token_timestamp": datetime.now(),
                    "revoked": False
                }
            }
        )

        if result.modified_count != 1:
            logger.error(f"Không thể lưu refresh token cho email: {email}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Lỗi khi lưu refresh token."
            )

        logger.info(f"Refresh token created for email: {email}")
        return refresh_token

    except Exception as e:
        logger.error(f"Lỗi khi tạo refresh token: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Lỗi hệ thống khi tạo refresh token."
        )


def load_legal_dictionary(path: str = 'legal_terms.json') -> list:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['dictionary']

def is_definition_question(query: str) -> bool:
    definition_keywords = ["là gì", "định nghĩa", "nghĩa là gì", "hiểu thế nào", "khái niệm"]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in definition_keywords)


def normalize_text_for_matching(text: str) -> str:
    """
    Chuẩn hóa text cho việc so khớp: chữ thường, loại bỏ ký tự đặc biệt (chỉ giữ chữ và số),
    loại bỏ dấu tiếng Việt, chuẩn hóa khoảng trắng.
    """
    if not text or not isinstance(text, str):
        return ""
    text_no_diacritics = unidecode(text.lower()) # Chuyển không dấu và chữ thường
    # Loại bỏ tất cả ký tự không phải là chữ cái hoặc số hoặc khoảng trắng
    text_alphanumeric = re.sub(r'[^\w\s]', '', text_no_diacritics, flags=re.UNICODE)
    return re.sub(r'\s+', ' ', text_alphanumeric).strip()

def search_term_in_dictionary(query: str, dictionary: List[Dict]) -> Optional[Dict]:
    """
    Tìm kiếm thuật ngữ trong từ điển.
    Chỉ tìm nếu là câu hỏi định nghĩa.
    Cải thiện logic so khớp.
    """
    if not is_definition_question(query):
        logger.debug(f"'{query}' không phải câu hỏi định nghĩa, bỏ qua tìm từ điển.")
        return None

    if not dictionary:
        logger.warning("Từ điển rỗng, không thể tìm kiếm.")
        return None

    # Cố gắng trích xuất thuật ngữ chính từ câu hỏi định nghĩa
    # Ví dụ: "Khái niệm hợp đồng lao động là gì?" -> "hợp đồng lao động"
    # Đây là một regex đơn giản, có thể cần tinh chỉnh
    term_to_search_raw = query
    match = re.match(r"^(.*?)\s+(là gì|định nghĩa|nghĩa là gì|hiểu thế nào|khái niệm)\??$", query.lower().strip(), re.IGNORECASE)
    if match:
        term_to_search_raw = match.group(1).strip()
        logger.info(f"Trích xuất thuật ngữ từ câu hỏi định nghĩa: '{term_to_search_raw}'")

    query_normalized_for_match = normalize_text_for_matching(term_to_search_raw)
    if not query_normalized_for_match:
        logger.debug("Thuật ngữ tìm kiếm rỗng sau khi chuẩn hóa.")
        return None

    logger.info(f"Tìm kiếm thuật ngữ đã chuẩn hóa (không dấu): '{query_normalized_for_match}'")

    # Sắp xếp từ điển theo độ dài thuật ngữ giảm dần (để ưu tiên khớp cụm dài hơn)
    # và chuẩn hóa thuật ngữ từ điển một lần
    normalized_dictionary = []
    for entry in dictionary:
        term = entry.get("term")
        if term and isinstance(term, str):
            normalized_dictionary.append({
                "original_entry": entry,
                "normalized_term": normalize_text_for_matching(term)
            })

    # Sắp xếp theo độ dài thuật ngữ đã chuẩn hóa giảm dần
    # Điều này giúp "an toàn lao động" được khớp trước "an toàn" hoặc "lao động"
    # nếu query là "an toàn lao động là gì"
    normalized_dictionary.sort(key=lambda x: len(x["normalized_term"]), reverse=True)


    # Tìm kiếm khớp chính xác (sau khi chuẩn hóa cả query và term từ điển)
    for item in normalized_dictionary:
        if item["normalized_term"] == query_normalized_for_match:
            logger.info(f"Tìm thấy khớp chính xác (sau chuẩn hóa): '{item['original_entry']['term']}'")
            return item["original_entry"]

    # Tìm kiếm "chứa" (thuật ngữ từ điển là một phần của query đã chuẩn hóa)
    # Điều này hữu ích nếu query_normalized_for_match dài hơn thuật ngữ từ điển
    # Ví dụ: query_normalized = "dinh nghia an toan lao dong", term_normalized = "an toan lao dong"
    for item in normalized_dictionary:
        if item["normalized_term"] and item["normalized_term"] in query_normalized_for_match:
            logger.info(f"Tìm thấy khớp 'chứa' (từ điển trong query): '{item['original_entry']['term']}' (query norm: '{query_normalized_for_match}')")
            return item["original_entry"]


    logger.info(f"Không tìm thấy thuật ngữ '{query_normalized_for_match}' trong từ điển.")
    return None


def minimal_preprocess_for_llm(text: str) -> str:
    """
    Thực hiện tiền xử lý tối thiểu trước khi đưa vào LLM.
    Chỉ chuẩn hóa khoảng trắng và chuyển thành chữ thường.
    """
    if not text or not text.strip():
        # Vẫn cần kiểm tra input rỗng
        raise ValueError("Input không được rỗng")

    # 1. Chuẩn hóa khoảng trắng
    processed_text = re.sub(r'\s+', ' ', text).strip()

    # 2. Chuyển thành chữ thường để nhất quán
    processed_text = processed_text.lower()

    return processed_text

async def save_chat_to_mongo(conversations_collection,chat_id: str, user_email: str,user_question_content: str, # Nội dung câu hỏi
    assistant_answer_content: str, # Nội dung trả lời
    user_question_timestamp: datetime,
    assistant_answer_timestamp: datetime):
    user_message = {
        "role": "user",
        "content": user_question_content,
        "timestamp": user_question_timestamp
    }
    assistant_message = {
        "role": "assistant",
        "content": assistant_answer_content,
        "timestamp": assistant_answer_timestamp
    }
    conversation = conversations_collection.find_one({"conversation_id": chat_id})
    if not conversation:
        conversation = {
            "user_id": user_email,
            "conversation_id": chat_id,
            "messages": [user_message, assistant_message],
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }
        conversations_collection.insert_one(conversation)
    else:
        conversations_collection.update_one(
            {"conversation_id": chat_id},
            {
                "$push": {"messages": {"$each": [user_message, assistant_message]}},
                "$set": {"updated_at": datetime.now()}
            }
        )



async def get_langchain_chat_history(app_state, chat_id: str) -> RedisChatMessageHistory:
    """
    Retrieves and synchronizes chat history for Langchain.
    """
    redis_url = os.environ.get("REDIS_URL_LANGCHAIN", os.environ.get("REDIS_URL"))
    if not redis_url:
        raise ValueError("Redis URL for chat history is required.")

    # Đây là history mà Langchain sẽ sử dụng để đọc/ghi
    langchain_chat_history = RedisChatMessageHistory( # Hoặc RedisChatMessageHistoryAsync
        url=redis_url,
        session_id=chat_id,
        ttl=86400, # 1 day
    )

    # Đồng bộ hóa: Lấy từ key "source of truth" của chúng ta và nạp vào key của Langchain
    messages_key = f"conversation_messages:{chat_id}"
    # Sử dụng await nếu redis client của app_state là async
    raw_messages_from_our_redis =  app_state.redis.lrange(messages_key, 0, -1)

    # Xóa history cũ trong key của Langchain để tránh trùng lặp khi đồng bộ
    # Nếu dùng RedisChatMessageHistoryAsync: await langchain_chat_history.aclear()
    langchain_chat_history.clear() # Cho bản đồng bộ

    for msg_json_bytes in raw_messages_from_our_redis:
        msg_data = json.loads(msg_json_bytes.decode()) # decode bytes to str
        message = Message(**msg_data) # Validate

        if message.role == "user":
            # Nếu dùng RedisChatMessageHistoryAsync: await langchain_chat_history.aadd_user_message(message.content)
            langchain_chat_history.add_user_message(message.content)
        elif message.role == "assistant":
            # await langchain_chat_history.aadd_ai_message(message.content)
            langchain_chat_history.add_ai_message(message.content)

    return langchain_chat_history

# api/utils.py

import hashlib
import config

logger = logging.getLogger(__name__)

def calculate_file_hash(filepath: str) -> str:
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def check_if_hash_exists(file_hash: str) -> bool:
    if not os.path.exists(config.PROCESSED_HASH_LOG):
        return False
    try:
        with open(config.PROCESSED_HASH_LOG, "r") as f:
            processed_hashes = {line.strip() for line in f}
            return file_hash in processed_hashes
    except IOError as e:
        logger.error(f"Could not read hash log file: {e}")
        return False

def log_processed_hash(file_hash: str):
    try:
        with open(config.PROCESSED_HASH_LOG, "a") as f:
            f.write(file_hash + "\n")
    except IOError as e:
        logger.error(f"Could not write to hash log file: {e}")