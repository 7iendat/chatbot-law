import redis
import os


redis_url = os.environ.get("REDIS_URL")
r = redis.Redis.from_url(redis_url)

try:
    r = redis.Redis.from_url(redis_url)
    r.ping()  # Kiểm tra kết nối
    print("=> [Redis] Connected successfully.")
except redis.RedisError as e:
    print(f"=> [Redis] Connection failed: {e}")
    r = None