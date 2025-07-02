from db.mongoDB import user_collection
from fastapi import HTTPException, Request, Response
import math
from datetime import datetime, timedelta, timezone
from fastapi import status
import logging
import secrets
from bson import ObjectId
from passlib.context import CryptContext
from services.send_email import send_reset_password_email,send_verification_code_email,send_suspicious_activity_email
from schemas.user import LoginRequest
from utils.utils import create_access_token,create_refresh_token
import os


logger = logging.getLogger(__name__)
# Initialize password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_users(skip: int = 0, limit: int = 100, query: dict = None):
    """
    Lấy danh sách người dùng với phân trang và tìm kiếm nâng cao

    Args:
        skip (int): Số lượng bản ghi bỏ qua (để phân trang)
        limit (int): Số lượng bản ghi tối đa trả về
        query (dict, optional): Điều kiện tìm kiếm người dùng

    Returns:
        list: Danh sách người dùng đã được lọc và phân trang

    Raises:
        HTTPException: Nếu có lỗi xảy ra khi truy vấn cơ sở dữ liệu
    """
    try:
        # Khởi tạo query nếu chưa có
        search_query = query or {}

        # Thực hiện truy vấn với phân trang
        users_cursor = user_collection.find(
            search_query,
            {"_id": 0, "password": 0}  # Loại bỏ các trường nhạy cảm
        ).skip(skip).limit(limit)

        # Chuyển cursor thành danh sách
        user_list = list(users_cursor)

        # Thêm log để debug nếu cần
        logger.debug(f"Đã lấy {len(user_list)} người dùng với skip={skip}, limit={limit}")

        return user_list
    except Exception as e:
        logger.error(f"Lỗi khi lấy danh sách người dùng: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi lấy danh sách người dùng: {str(e)}"
        )

def count_users(query: dict = None):
    """
    Đếm tổng số người dùng thỏa mãn điều kiện tìm kiếm

    Args:
        query (dict, optional): Điều kiện tìm kiếm người dùng

    Returns:
        int: Tổng số người dùng
    """
    try:
        search_query = query or {}
        return user_collection.count_documents(search_query)
    except Exception as e:
        logger.error(f"Lỗi khi đếm người dùng: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi đếm người dùng: {str(e)}"
        )

async def get_paginated_users(
    skip: int = 0,
    limit: int = 10,
    search: str = None,
    sort_by: str = "created_at",
    sort_order: int = -1
):
    """
    Lấy danh sách người dùng với đầy đủ thông tin phân trang và tìm kiếm

    Args:
        skip (int): Số lượng bản ghi bỏ qua
        limit (int): Số lượng bản ghi tối đa trả về
        search (str, optional): Từ khóa tìm kiếm (tên, email, v.v.)
        sort_by (str): Trường để sắp xếp
        sort_order (int): Thứ tự sắp xếp (1: tăng dần, -1: giảm dần)

    Returns:
        dict: Kết quả phân trang với items và metadata
    """
    try:
        # Xây dựng query tìm kiếm nếu có từ khóa
        query = {}
        if search:
            # Tìm kiếm theo tên hoặc email
            query = {
                "$or": [
                    {"email": {"$regex": search, "$options": "i"}},
                    {"username": {"$regex": search, "$options": "i"}}
                ]
            }

        # Sắp xếp
        sort_criteria = [(sort_by, sort_order)]

        # Thực hiện truy vấn
        users_cursor = user_collection.find(
            query,
            {"_id": 0, "password": 0}  # Loại bỏ các trường nhạy cảm
        ).sort(sort_criteria).skip(skip).limit(limit)

        # Chuyển cursor thành danh sách
        user_list =  users_cursor.to_list(length=limit)

        # Đếm tổng số bản ghi
        total = user_collection.count_documents(query)

        # Tính toán thông tin phân trang
        total_pages = math.ceil(total / limit) if limit > 0 else 0
        current_page = math.floor(skip / limit) + 1 if limit > 0 else 1

        # Trả về kết quả với metadata phân trang
        return {
            "items": user_list,
            "metadata": {
                "total": total,
                "page": current_page,
                "page_size": limit,
                "pages": total_pages,
                "has_more": current_page < total_pages
            }
        }
    except Exception as e:
        logger.error(f"Lỗi khi lấy danh sách người dùng phân trang: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi lấy danh sách người dùng: {str(e)}"
        )

async def get_current_user_profile(email: str):
    """
    Lấy thông tin profile của người dùng hiện tại

    Args:
        email (str): Email của người dùng

    Returns:
        dict: Thông tin profile của người dùng
    """
    try:
        user =  user_collection.find_one({"email": email}, {"_id": 0, "password": 0})
        logger.info(f"check user user_service: {user}")
        if not user:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Người dùng không tồn tại")
        return user
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Lỗi khi lấy thông tin profile người dùng: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi lấy thông tin profile người dùng: {str(e)}"
        )


def delete_user(user_id: str):
    """
    Xóa người dùng theo ID

    Args:
        user_id (str): ID của người dùng cần xóa

    Returns:
        bool: True nếu xóa thành công, False nếu không tìm thấy người dùng
    """
    try:
        result = user_collection.delete_one({"_id": user_id})
        return result.deleted_count > 0
    except Exception as e:
        logger.error(f"Lỗi khi xóa người dùng: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi xóa người dùng: {str(e)}"
        )

def change_password(email: str, current_password: str, new_password: str):
    """
    Đổi mật khẩu của người dùng

    Args:
        user_id (str): ID của người dùng
        current_password (str): Mật khẩu hiện tại
        new_password (str): Mật khẩu mới

    Returns:
        bool: True nếu đổi mật khẩu thành công, False nếu không thành công
    """
    try:
        user = user_collection.find_one({"email": email})
        if not user:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Người dùng không tồn tại")



        if not pwd_context.verify(current_password, user["password"]):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Mật khẩu hiện tại không chính xác")

        # Hash the new password
        hashed_password = pwd_context.hash(new_password)
        # Update the password in the database
        result = user_collection.update_one({"email": email}, {"$set": {"password": hashed_password}})
        return result.modified_count > 0
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Lỗi khi đổi mật khẩu: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi đổi mật khẩu"
        )

async def reset_password_request(email: str) -> bool:
    """
    Yêu cầu đặt lại mật khẩu cho người dùng

    Args:
        email (str): Địa chỉ email của người dùng

    Returns:
        bool: True nếu yêu cầu thành công, False nếu không tìm thấy người dùng

    Raises:
        HTTPException: Nếu có lỗi trong quá trình xử lý
    """
    try:
        # Check if user exists
        user =  user_collection.find_one({"email": email.lower()})
        if not user:
            # Return True even if user not found to prevent email enumeration
            return True

        # Check for rate limiting (e.g., max 3 requests per hour)
        reset_requests =  user_collection.count_documents({
            "email": email.lower(),
            "reset_password_timestamp": {
                "$gte": datetime.now(tz=timezone.utc) - timedelta(hours=1)
            }
        })
        if reset_requests >= 3:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Quá nhiều yêu cầu đặt lại mật khẩu. Vui lòng thử lại sau."
            )

        # Generate secure reset token
        code = str(secrets.randbelow(1000000)).zfill(6)  # e.g., '123456'
        expiry = datetime.now() + timedelta(minutes=10)

        # Store reset token and timestamp in database
        user_collection.update_one(
            {"_id": ObjectId(user["_id"])},
            {
                "$set": {
                    "reset_password_code": code,
                    "reset_password_timestamp": datetime.now(tz=timezone.utc),
                    "reset_password_expiry": expiry,
                }
            }
        )

        # Send reset password email (uncomment and implement actual email sending)
        await send_reset_password_email(
            email=email,
            code=code,
            expiry=15
        )

        return True

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Lỗi khi yêu cầu đặt lại mật khẩu: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Lỗi hệ thống. Vui lòng thử lại sau."
        )



async def reset_password(code: str, newPassword: str) -> bool:
    """
    Hoàn tất đặt lại mật khẩu với token và mật khẩu mới.

    Args:
        code (str): code đặt lại mật khẩu.
        newPassword (str): Mật khẩu mới của người dùng.

    Returns:
        bool: True nếu đặt lại mật khẩu thành công, False nếu token không hợp lệ hoặc đã hết hạn.

    Raises:
        HTTPException: Nếu có lỗi hệ thống trong quá trình xử lý.
    """
    try:
        # Find user by reset token
        user =  user_collection.find_one({"reset_password_code": code})
        if not user:
            logger.warning(f"Code không hợp lệ: {code}")
            return False

        # Check if token is expired
        expiry = user.get("reset_password_expiry")
        if not expiry or expiry < datetime.now():
            logger.warning(f"Code đã hết hạn: {code}")
            return False

        # Validate new password (basic example, can add more rules)
        if len(newPassword) < 8:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Mật khẩu mới phải có ít nhất 8 ký tự"
            )

        # Hash the new password
        hashed_password = pwd_context.hash(newPassword)

        # Update user's password and clear reset token
        result =  user_collection.update_one(
            {"_id": ObjectId(user["_id"])},
            {
                "$set": {"password": hashed_password},
                "$unset": {
                    "reset_password_code": "",
                    "reset_password_expiry": "",
                    "reset_password_timestamp": ""
                }
            }
        )

        if result.modified_count != 1:
            logger.error(f"Không thể cập nhật mật khẩu cho user: {user['_id']}")
            return False

        logger.info(f"Đặt lại mật khẩu thành công cho email: {user['email']}")
        return True

    except Exception as e:
        logger.error(f"Lỗi khi đặt lại mật khẩu: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Lỗi hệ thống khi đặt lại mật khẩu. Vui lòng thử lại sau."
        )


async def generate_and_store_verification_code(email: str) -> bool:
    """
    Tạo và lưu mã xác minh đăng nhập vào cơ sở dữ liệu.

    Args:
        email (str): Địa chỉ email của người dùng.

    Returns:
        bool: True nếu lưu mã thành công.

    Raises:
        HTTPException: Nếu có lỗi hệ thống.
    """
    try:
        # Generate 6-digit code
        code = str(secrets.randbelow(1000000)).zfill(6)  # e.g., '123456'
        expiry = datetime.now() + timedelta(minutes=10)

        # Store code and expiry in database
        result =  user_collection.update_one(
            {"email": email.lower()},
            {
                "$set": {
                    "login_verification_code": code,
                    "login_verification_expiry": expiry,
                    "login_verification_timestamp": datetime.now()
                }
            }
        )

        if result.modified_count != 1:
            logger.error(f"Không thể lưu mã xác minh cho email: {email}")
            return False

        # Send verification email
        await send_verification_code_email(email, code)
        return True

    except Exception as e:
        logger.error(f"Lỗi khi tạo mã xác minh: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Lỗi hệ thống khi tạo mã xác minh. Vui lòng thử lại sau."
        )



async def verify_login_code(email: str, code: str, res: Response): # Bỏ kiểu trả về dict, để FastAPI tự suy luận hoặc dùng Pydantic model
    """
    Xác minh mã đăng nhập, set access và refresh tokens vào HttpOnly cookies,
    và trả về thông tin người dùng.

    Args:
        email (str): Địa chỉ email của người dùng.
        code (str): Mã xác minh do người dùng cung cấp.
        res (Response): Đối tượng Response của FastAPI để set cookies.

    Returns:
        dict: Thông tin người dùng và thông báo thành công.

    Raises:
        HTTPException: Nếu mã không hợp lệ, đã hết hạn hoặc có lỗi hệ thống.
    """
    try:
        user =  user_collection.find_one({ # Sử dụng await nếu user_collection là async (ví dụ Motor)
            "email": email.lower(),
            "login_verification_code": code
        })
        # Nếu user_collection là đồng bộ (ví dụ PyMongo)
        # user = user_collection.find_one({ ... })

        if not user:
            logger.warning(f"Mã xác minh không hợp lệ: {code} cho email: {email}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Mã xác minh không hợp lệ hoặc đã được sử dụng." # Thêm "đã được sử dụng"
            )

        expiry = user.get("login_verification_expiry")
        if not expiry or expiry < datetime.now(expiry.tzinfo if expiry.tzinfo else None): # So sánh aware với aware, naive với naive
            logger.warning(f"Mã xác minh đã hết hạn cho email: {email}")
            # Xóa mã đã hết hạn để tránh sử dụng lại
            user_collection.update_one(
                {"_id": ObjectId(user["_id"])},
                {
                    "$unset": {
                        "login_verification_code": "",
                        "login_verification_expiry": "",
                        "login_verification_timestamp": ""
                    }
                }
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Mã xác minh đã hết hạn."
            )

        try:
            token_expire_minutes_str = os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60") # Default là chuỗi "60"
            ACCESS_TOKEN_EXPIRE_MINUTES_INT = int(token_expire_minutes_str)
            if ACCESS_TOKEN_EXPIRE_MINUTES_INT <= 0:
                logger.warning(f"Giá trị ACCESS_TOKEN_EXPIRE_MINUTES ('{token_expire_minutes_str}') không hợp lệ, sử dụng mặc định 60 phút.")
                ACCESS_TOKEN_EXPIRE_MINUTES_INT = 60
        except ValueError:
            logger.error(f"Không thể chuyển đổi ACCESS_TOKEN_EXPIRE_MINUTES ('{os.getenv('ACCESS_TOKEN_EXPIRE_MINUTES')}') sang số nguyên. Sử dụng mặc định 60 phút.")
            ACCESS_TOKEN_EXPIRE_MINUTES_INT = 60

        logger.info(f"Sử dụng thời gian hết hạn cho access token: {ACCESS_TOKEN_EXPIRE_MINUTES_INT} phút.")
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES_INT) # Sử dụng biến đã chuyển đổi
        refresh_token_expires = timedelta(days=7) # Sử dụng biến đã chuyển đổi
        access_token = create_access_token(
            data={"sub": email.lower()}, expires_delta=access_token_expires
        ) # Giả sử create_access_token chấp nhận expires_delta

        IS_PRODUCTION = os.getenv("APP_ENVIRONMENT", "development").lower() == "production"
        # Generate refresh token (logic của bạn đã có)
        refresh_token_value = await  create_refresh_token(email.lower()) # Đảm bảo email là lowercase
        # --- SET COOKIES ---
        # Môi trường (ví dụ: "development", "production")
        # Bạn nên có một biến môi trường để xác định điều này
        # IS_PRODUCTION = os.getenv("ENVIRONMENT") == "production"
        # Tạm đặt là True, bạn nên lấy từ biến môi trường
        max_age_seconds = int(access_token_expires.total_seconds())
        logger.info(f"Setting access_token_cookie with Max-Age: {max_age_seconds} seconds")
        # Access Token Cookie
        res.set_cookie(
            key="access_token_cookie", # Tên cookie cho access token
            value=access_token,
            max_age=int(access_token_expires.total_seconds()), # Thời gian sống bằng access token
            httponly=True,
            secure=IS_PRODUCTION, # True trong production (HTTPS), False khi dev với HTTP
            samesite="lax", # Hoặc "strict"
            path="/",
        )

        # Refresh Token Cookie (logic của bạn đã có, điều chỉnh secure)
        res.set_cookie(
            key="refresh_token",
            value=refresh_token_value,
            max_age=int(refresh_token_expires.total_seconds()),
            httponly=True,
            secure=IS_PRODUCTION, # True trong production, False khi dev với HTTP
            samesite="lax",
            path="/api/user/refresh-token",
        )
        # --------------------

        # Clear verification code và cập nhật last_login
        update_result =  user_collection.update_one( # Sử dụng await nếu là async
            {"_id": ObjectId(user["_id"])},
            {
                "$unset": {
                    "login_verification_code": "",
                    "login_verification_expiry": "",
                    "login_verification_timestamp": ""
                },
                "$set": {
                    "last_login": datetime.now(timezone.utc if expiry.tzinfo else None), # Giữ timezone nhất quán
                    "login_type": "email", # Giả sử đây là đăng nhập bằng email
                }
            }
        )
        # if update_result.modified_count == 0:
            # logger.warning(f"Không thể cập nhật user sau khi xác minh: {email}")
            # Cân nhắc có nên raise lỗi ở đây không, hoặc chỉ log

        logger.info(f"Đăng nhập thành công cho email: {email}. Tokens đã được set vào cookies.")

        # Response trả về không còn chứa accessToken
        return {
            # "accessToken": access_token, # KHÔNG TRẢ VỀ ACCESS TOKEN TRONG BODY NỮA
            "user": {
                "email": email.lower(),
                "username": user.get("username", email.lower().split('@')[0]), # Cung cấp username mặc định nếu không có
                "role": user.get("role", "user"),
                "avatar_url": user.get("avatar_url", None), # Hoặc một avatar mặc định,
                "login_type": user.get("login_type", "email") # Giả sử đây là đăng nhập bằng email
            },
            "message": "Đăng nhập thành công. Tokens đã được lưu trữ an toàn."
        }

    except HTTPException as he:
        # Ghi log chi tiết hơn cho HTTPException nếu cần
        # logger.error(f"HTTPException trong verify_login_code cho {email}: {he.detail}", exc_info=not (he.status_code < 500))
        raise he
    except Exception as e:
        logger.error(f"Lỗi hệ thống khi xác minh mã đăng nhập cho {email}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Lỗi hệ thống, vui lòng thử lại sau."
        )


async def authenticate_user(request: LoginRequest) -> dict:
    """
    Xác thực người dùng dựa trên email và mật khẩu.

    Args:
        request (LoginRequest): Yêu cầu đăng nhập chứa email và mật khẩu.

    Returns:
        dict: Thông tin người dùng nếu xác thực thành công.

    Raises:
        HTTPException: Nếu thông tin đăng nhập không hợp lệ.
    """
    user =  user_collection.find_one({"email": request.email.lower()})
    if not user or not pwd_context.verify(request.password, user["password"]):
        logger.warning(f"Xác thực thất bại cho email: {request.email}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email hoặc mật khẩu không đúng",
            headers={"WWW-Authenticate": "Bearer"}
        )
    return user


async def refresh_access_token(req: Request  , res: Response ) -> dict:
    """
    Làm mới access token dựa trên refresh token từ cookie.

    Args:
        req (Request): FastAPI request object chứa cookie.

    Returns:
        JSONResponse: Phản hồi chứa access_token, refresh_token, username, email, role và set cookie mới.

    Raises:
        HTTPException: Nếu refresh token không hợp lệ, đã hết hạn hoặc có lỗi hệ thống.
    """
    try:
        # Retrieve refresh token from cookie
        refresh_token = req.cookies.get("refresh_token")
        if not refresh_token:
            logger.warning("Không tìm thấy refresh token trong cookie")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Không tìm thấy refresh token",
                headers={"WWW-Authenticate": "Bearer"}
            )

        # Find user by refresh token in database
        user =  user_collection.find_one({"refresh_token": refresh_token})


        if not user:
            logger.warning(f"Refresh token không hợp lệ: {refresh_token}")
            await send_suspicious_activity_email(user.get("email", "unknown") if user else "unknown")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Refresh token không hợp lệ",
                headers={"WWW-Authenticate": "Bearer"}
            )

        # Check if refresh token is expired
        expiry = user.get("refresh_token_expiry")
        if not expiry or expiry < datetime.now():
            logger.warning(f"Refresh token đã hết hạn cho email: {user['email']}")
            await send_suspicious_activity_email(user["email"])
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Refresh token đã hết hạn",
                headers={"WWW-Authenticate": "Bearer"}
            )

        email = user.get('email')
        # Generate new access token
        token_expire_minutes_str = os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60") # Default là chuỗi "60"
        ACCESS_TOKEN_EXPIRE_MINUTES_INT = int(token_expire_minutes_str)
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES_INT) # Sử dụng biến đã chuyển đổi
        access_token = create_access_token(
            data={"sub": email.lower()}, expires_delta=access_token_expires
        ) # Giả sử create_access_token chấp nhận expires_delta

        IS_PRODUCTION = os.getenv("APP_ENVIRONMENT", "development").lower() == "production"
        # Generate new refresh token (rotation)
        new_refresh_token = await create_refresh_token(email)

        # Update database with new refresh token
        user_collection.update_one(
            {"_id": user["_id"]},
            {
                "$set": {
                    "refresh_token": new_refresh_token,
                    "refresh_token_expiry": datetime.now() + timedelta(days=7),
                    "last_login": datetime.now(timezone.utc)
                }
            }
        )



         # Set access token mới vào cookie
        res.set_cookie(
            key="access_token_cookie", # Đảm bảo tên này khớp với client mong đợi
            value=access_token,
            max_age= access_token_expires, # tính bằng giây
            httponly=True,
            secure=IS_PRODUCTION,
            samesite='lax', # type: ignore
            path='/',
        )

        # Set new refresh token in cookie
        res.set_cookie(
            key="refresh_token",
            value=new_refresh_token,
            max_age=7 * 24 * 60 * 60,  # 7 days
            httponly=True,
            secure=IS_PRODUCTION,  # Set to False for local development
            samesite="lax",  # CSRF protection
            path="/api/user/refresh-token",
        )

        logger.info(f"Access token refreshed for email: {user['email']}")
        return {
            "success": True,
            "message": "Token đã được làm mới thành công.",
            "user": {
                "username": user['username'], # Lấy username, fallback về email
                "email": user['email'],
                "role": user['role'],
                "avatar_url": user['avatar_url']
            }
            # Không cần trả về token trong body nếu chúng đã được set trong HttpOnly cookies
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Lỗi khi làm mới access token: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Lỗi hệ thống khi làm mới access token"
        )


async def verify_forgot_password_code(email: str, code: str) -> dict:

    try:
        # Find user by email and code
        user =  user_collection.find_one({
            "email": email.lower(),
            "reset_password_code": code
        })
        if not user:
            logger.warning(f"Mã xác minh không hợp lệ: {code}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Mã xác minh không hợp lệ"
            )

        # Check if code is expired
        expiry = user.get("reset_password_expiry")
        if not expiry or expiry < datetime.now():
            logger.warning(f"Mã xác minh đã hết hạn cho email: {email}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Mã xác minh đã hết hạn"
            )



        return True
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Lỗi khi xác minh mã quen mat khau: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Lỗi hệ thống khi xác minh mã quen mat khau."
        )
