import redis
import os
from dotenv import load_dotenv

redis_url = os.environ.get("REDIS_URL")
r = redis.Redis.from_url(redis_url)