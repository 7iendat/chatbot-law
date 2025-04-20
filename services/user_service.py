from db.mongoDB import user_collection, blacklist_collection
from fastapi import HTTPException


def get_users():
    try:
        users = user_collection.find({}, {"_id": 0, "password": 0})  # Không trả về mật khẩu
        user_list = list(users)

        return user_list
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi lấy danh sách người dùng: {str(e)}")