from pymongo import MongoClient, errors
from dotenv import load_dotenv
import os
import logging
logger = logging.getLogger(__name__)
load_dotenv()

MONGO_URI = os.getenv("MONGODB_CLOUD_URI")
DB_NAME = os.getenv("DB_NAME")

try:
    # Kết nối MongoDB
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)

    # Trigger kết nối thử
    client.server_info()  # Gây lỗi nếu không kết nối được

    db = client[DB_NAME]
    user_collection = db["users"]
    blacklist_collection = db["token_blacklist"]
    conversations_collection = db["conversations"]

    # ⚠️ Tạo TTL index để MongoDB tự động xoá token khi tới hạn
    # Chỉ tạo index nếu chưa tồn tại
    if "expires_at_1" not in blacklist_collection.index_information():
        blacklist_collection.create_index("expires_at", expireAfterSeconds=0)
        logger.info("🔸Đã tạo TTL index cho 'expires_at' trong 'token_blacklist'.")

    logger.info("🔸Đã kết nối tới MongoDB Cloud thành công!")

except errors.ServerSelectionTimeoutError as e:
    logger.error("🔸Không thể kết nối tới MongoDB Cloud:")
    logger.error(f"🔸Error:{e}")
    user_collection = None
    blacklist_collection = None
