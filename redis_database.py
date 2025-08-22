import os
import time
import urllib.parse as urlparse
import redis
from dotenv import load_dotenv

load_dotenv()

url = urlparse.urlparse(os.environ["REDISCLOUD_URL"])
r = redis.Redis(
    host=url.hostname,
    port=url.port,
    password=url.password,
    decode_responses=True
)

def save_user(chat_id, username):
    """Save user to the database."""
    r.hset(f"chat_id:{chat_id}", mapping={
        "username": username or "",
        "subscribed": "true",
        "last_active": str(int(time.time()))
    })
    r.sadd('subscribed_users', chat_id)

def save_user_to_database(chat_id, username):
    """Save user to the database."""
    r.hset(f"chat_id:{chat_id}", mapping={
        "username": username or "",
        "last_active": str(int(time.time()))
    })

def delete_user(chat_id):
    """Delete user from the database."""
    r.hdel(f"chat_id:{chat_id}", 'subscribed')
    r.srem('subscribed_users', chat_id)

def is_subscribed(chat_id):
    """Check if user is subscribed"""
    return r.hexists(f"chat_id:{chat_id}", 'subscribed')

def get_all_subscribed_chats():
    """Get all subscribed chat IDs"""
    return r.smembers('subscribed_users')

def get_all_chat_ids():
    """Get all chat IDs from the database."""
    return [key.split(":")[1] for key in r.keys("chat_id:*")]

def get_user_chat_id(username):
    """Get chat ID by username."""
    for key in r.keys("chat_id:*"):
        if r.hget(key, "username") == username:
            return key.split(":")[1]
    return None

def get_user_username(chat_id):
    """Get username by chat ID."""
    return r.hget(f"chat_id:{chat_id}", "username")
