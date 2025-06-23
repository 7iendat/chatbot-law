from fastapi.responses import JSONResponse
from jose import JWTError, jwt
from datetime import datetime, timedelta, timezone
from fastapi import HTTPException, Depends, Request, status
from schemas.user import RegisterRequest, LoginRequest
from db.mongoDB import user_collection, blacklist_collection
from utils.utils import  verify_password
from config import SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext
from services.uplash import get_random_unsplash_image
from jose import JWTError
import logging

logger = logging.getLogger(__name__)
# Khởi tạo context mật khẩu

bearer_scheme = HTTPBearer(auto_error=False)

# Initialize password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
# Hàm tạo token truy cập
def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.now() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# Hàm đăng ký người dùng
async def register_user(user: RegisterRequest):

    try:
                # Kiểm tra username đã tồn tại chưa
        existing_user = user_collection.find_one({"email": user.email})
        if existing_user:
            raise HTTPException(status_code=400, detail="Tài khoản đã tồn tại.")

        # Mã hóa mật khẩu
        hashed_password = pwd_context.hash(user.password)

        avatar = await get_random_unsplash_image()
        # Lưu user mới
        user_collection.insert_one({
            "username": user.username,
            "password": hashed_password,
            "email": user.email,
            "avatar_url": avatar,
            "role": user.role or "user",
            "is_active": True,
        })

        return {"message": "Đăng ký thành công."}
    except HTTPException as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)
# Hàm xác thực đăng nhập
def authenticate_user(request: LoginRequest):
    user = user_collection.find_one({"email": request.email})
    if not user:
        raise HTTPException(status_code=401, detail="Sai tài khoản hoặc mật khẩu")

    if not verify_password(request.password, user["password"]):
        raise HTTPException(status_code=401, detail="Sai tài khoản hoặc mật khẩu")

    return user

# Hàm tạo và trả về JWT token sau khi đăng nhập thành công
def login_user(request: LoginRequest):
    user = authenticate_user(request)

    accessToken = create_access_token(data={"sub": request.email})

    return  {
        "accessToken": accessToken,
        "username": user["username"],
        "email": user["email"],
        "role": user.get("role", "user")
    }

# Ham dang xuat
async def logout_user(req: Request, credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)) :
    """
    Đăng xuất người dùng, vô hiệu hóa access token và refresh token.

    Args:
        req (Request): FastAPI request object chứa cookie.
        credentials (HTTPAuthorizationCredentials): Access token từ header Authorization.

    Returns:
        JSONResponse: Thông báo đăng xuất thành công và xóa cookie refresh_token.

    Raises:
        HTTPException: Nếu access token hoặc refresh token không hợp lệ.
    """
    try:
        # Decode access token to get payload
        token = credentials.credentials
        payload = verify_token(token)
        exp = payload.get("exp")
        if not exp:
            logger.warning("Access token không có thời gian hết hạn")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Access token không hợp lệ")

        # Blacklist access token
        await blacklist_collection.insert_one({
            "token": token,
            "expires_at": datetime.fromtimestamp(exp, tz=timezone.utc)
        })

        # Get refresh token from cookie
        refresh_token = req.cookies.get("refresh_token")
        if refresh_token:
            # Invalidate refresh token in database
            user = await user_collection.find_one({"refresh_token": refresh_token})
            if user:
                await user_collection.update_one(
                    {"_id": user["_id"]},
                    {"$set": {"refresh_token": None, "refresh_token_expiry": None, "revoked": True}}
                )
                logger.info(f"Refresh token invalidated for email: {user.get('email', 'unknown')}")
            else:
                logger.warning(f"Refresh token không hợp lệ trong database: {refresh_token}")

        # Prepare response and clear refresh token cookie
        response = JSONResponse(content={"message": "Đăng xuất thành công"}, status_code=200)
        response.set_cookie(
            key="refresh_token",
            value="",
            max_age=0,  # Expire cookie immediately
            httponly=True,
            secure=False,  # Set to False for local development
            samesite="lax",
            path="/",
        )

        logger.info("User logged out successfully")
        return response

    except JWTError as e:
        logger.error(f"Lỗi khi xác minh access token: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Access token không hợp lệ")
    except Exception as e:
        logger.error(f"Lỗi hệ thống khi đăng xuất: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Lỗi hệ thống khi đăng xuất"
        )

# Hàm kiểm tra xem token có hợp lệ không
def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Token không hợp lệ")
