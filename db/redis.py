import redis
import os
import logging
logger = logging.getLogger(__name__)

redis_url = os.environ.get("REDIS_URL")
r = redis.Redis.from_url(redis_url)

try:
    r = redis.Redis.from_url(redis_url)
    r.ping()  # Kiểm tra kết nối
    logger.info("🔸[Redis] Connected successfully.")
except redis.RedisError as e:
    logger.error(f"🔸[Redis] Connection failed: {e}")
    r = None