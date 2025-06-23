import redis.asyncio as aioredis # Sử dụng redis-py cho async
import config
import logging
logger = logging.getLogger(__name__)

# redis_url = os.environ.get("REDIS_URL")
# r = redis.Redis.from_url(redis_url)

# try:
#     r = redis.Redis.from_url(redis_url)
#     r.ping()  # Kiểm tra kết nối
#     logger.info("🔸[Redis] Connected successfully.")
# except redis.RedisError as e:
#     logger.error(f"🔸[Redis] Connection failed: {e}")
#     r = None

async def get_redis_client():
    """
    Tạo và trả về một client Redis kết nối đến URL trong config.
    Hỗ trợ cả Redis local và các dịch vụ cloud như Upstash.
    """
    redis_url = config.REDIS_URL
    if not redis_url:
        logger.error("❌ REDIS_URL not found in configuration.")
        raise ValueError("REDIS_URL is not set.")

    logger.info(f"🔸 Connecting to Redis at: ...{redis_url[-20:]}") # Log một phần URL để bảo mật

    try:
        # redis.asyncio.from_url sẽ tự động xử lý tất cả các thành phần:
        # - Giao thức (redis:// hoặc rediss://)
        # - Tên người dùng và mật khẩu
        # - Host và Port
        # - Bật SSL/TLS nếu URL là rediss:// (Upstash mặc định dùng)
        redis_client = aioredis.from_url(
            redis_url,
            encoding="utf-8",
            decode_responses=True # Tự động decode kết quả thành string
        )

        # Kiểm tra kết nối
        await redis_client.ping()

        logger.info("✅ Redis connection successful.")
        return redis_client
    except Exception as e:
        logger.error(f"❌ Failed to connect to Redis: {e}", exc_info=True)
        return None