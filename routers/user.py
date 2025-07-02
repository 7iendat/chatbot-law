from fastapi import APIRouter, Depends, HTTPException, Query,status, Response, Request
from fastapi.responses import RedirectResponse
from services.auth_service import (
    verify_token, register_user
)
from datetime import datetime , timezone
from services.user_service import get_paginated_users, delete_user,change_password,reset_password_request,reset_password, generate_and_store_verification_code, authenticate_user, verify_login_code,refresh_access_token,verify_forgot_password_code
from dependencies import  get_current_user, admin_required
from schemas.user import LoginRequest, LoginResponse, RegisterRequest, RegisterResponse, UserOut,PaginatedResponse,ProfileResponse,ChangePasswordRequest,PasswordResetRequest,PasswordReset,VerifyLoginRequest,VerifyForgotPassRequest, ResentVerifyCode,TokenValidationResponse,TokenValidationRequest

from starlette.config import Config
from authlib.integrations.starlette_client import OAuth
import os
import config
from db.mongoDB import user_collection
import uuid
from passlib.context import CryptContext
from datetime import datetime, timedelta, timezone
from utils.utils import create_access_token,create_refresh_token
import re
import logging
from dependencies import get_app_state

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/login")
async def login(request: LoginRequest):
    """
    Bắt đầu quá trình đăng nhập bằng cách xác thực và gửi mã xác minh qua email.

    Args:
        request (LoginRequest): Yêu cầu đăng nhập chứa email và mật khẩu.

    Returns:
        dict: Thông báo yêu cầu người dùng nhập mã xác minh.

    Raises:
        HTTPException: Nếu xác thực thất bại hoặc có lỗi hệ thống.
    """
    # Authenticate user
    await authenticate_user(request)

    # Generate and send verification code
    success = await generate_and_store_verification_code(request.email)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Không thể gửi mã xác minh. Vui lòng thử lại sau."
        )

    return {"message": "Mã xác minh đã được gửi đến email của bạn. Vui lòng kiểm tra và nhập mã để hoàn tất đăng nhập."}

@router.post("/resent-verification-code")
async def resend_verification_code(request:ResentVerifyCode ):
    """
    Gửi lại mã xác minh đăng nhập đến email của người dùng.

    Args:
        email (str): Địa chỉ email của người dùng.

    Returns:
        dict: Thông báo gửi lại mã xác minh thành công.

    Raises:
        HTTPException: Nếu có lỗi khi gửi mã xác minh.
    """
    success = await generate_and_store_verification_code(request.email)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Không thể gửi mã xác minh. Vui lòng thử lại sau."
        )
    return {"message": "Mã xác minh đã được gửi lại đến email của bạn."}

@router.post("/verify-login", response_model=LoginResponse)
async def verify_login(request: VerifyLoginRequest, res: Response):
    """
    Xác minh mã đăng nhập và trả về thông tin đăng nhập.

    Args:
        request (VerifyLoginRequest): Yêu cầu chứa email và mã xác minh.

    Returns:
        LoginResponse: Thông tin đăng nhập bao gồm access_token, username, email, role.

    Raises:
        HTTPException: Nếu mã không hợp lệ, đã hết hạn hoặc có lỗi hệ thống.
    """
    return await verify_login_code(request.email, request.code, res)

@router.post("/change-password")
async def user_change_password(
    request: ChangePasswordRequest,
    current_user: UserOut = Depends(get_current_user)
):
    """Đổi mật khẩu của người dùng hiện tại"""
    success = change_password(current_user.email, request.current_password, request.new_password)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Mật khẩu hiện tại không chính xác"
        )
    return {"message": "Đổi mật khẩu thành công"}

@router.post("/forgot-password")
async def forgot_password(request: PasswordResetRequest):
    """Yêu cầu đặt lại mật khẩu"""
    await reset_password_request(request.email)
    return {"message": "Chúng tôi đã gửi hướng dẫn đặt lại mật khẩu"}

@router.post("/forgot-password/verify-code")
async def verify_forgot_code(request: VerifyForgotPassRequest):
    """Xác minh mã đặt lại mật khẩu"""
    success = await verify_forgot_password_code(request.email, request.code)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Mã xác minh không hợp lệ hoặc đã hết hạn"
        )
    return {"message": "Mã xác minh hợp lệ"}

@router.post("/reset-password")
async def complete_password_reset(request: PasswordReset):
    """Hoàn tất đặt lại mật khẩu với token"""
    success = await reset_password(request.code, request.newPassword)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Token không hợp lệ hoặc đã hết hạn"
        )
    return {"message": "Đặt lại mật khẩu thành công"}

@router.post("/register", response_model=RegisterResponse)
async def register(request: RegisterRequest):
    return await register_user(request)

@router.post("/logout")
async def logout_user(response: Response):
    """
    Đăng xuất người dùng bằng cách xóa HttpOnly cookies.
    """

    response.delete_cookie(key="access_token_cookie", path="/", samesite="lax") # Đảm bảo các thuộc tính khớp với lúc set
    response.delete_cookie(key="refresh_token", path="/api/user/refresh-token", samesite="lax")
    # Có thể thêm logic thu hồi refresh token ở backend nếu cần

    return {"message": "Đăng xuất thành công"}

@router.get("/list_users", response_model=PaginatedResponse)
async def list_users(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    search: str = Query(None),
    sort_by: str = Query("created_at"),
    sort_order: int = Query(-1),
    current_user: UserOut = Depends(admin_required)
):
    """Lấy danh sách người dùng có phân trang và tìm kiếm"""
    return await get_paginated_users(
        skip=skip,
        limit=limit,
        search=search,
        sort_by=sort_by,
        sort_order=sort_order
    )


@router.get("/me", response_model=ProfileResponse)
async def get_profile(current_user: UserOut = Depends(get_current_user)):
    """Lấy thông tin profile của người dùng hiện tại"""

    return current_user

@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_user(
    user_id: int,
    current_user: UserOut = Depends(admin_required)
):
    """Xóa người dùng (chỉ admin)"""
    success = delete_user(user_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Không tìm thấy người dùng"
        )
    return {"detail": "Đã xóa người dùng thành công"}


@router.post("/refresh-token", response_model=LoginResponse)
async def refresh_token_endpoint(req: Request, res: Response):
    """
    Làm mới access token sử dụng refresh token.

    Args:
        request (RefreshTokenRequest): Yêu cầu chứa refresh token.

    Returns:
        LoginResponse: Thông tin đăng nhập bao gồm access_token, username, email, role.

    Raises:
        HTTPException: Nếu refresh token không hợp lệ hoặc đã hết hạn.
    """
    return await refresh_access_token(req, res)

@router.post("/validate-token", response_model=TokenValidationResponse)
async def validate_token(request: TokenValidationRequest):
    """
    Xác thực access token và trả về thông tin người dùng.

    Args:
        request (TokenValidationRequest): Request body chứa token cần xác thực.

    Returns:
        TokenValidationResponse: Object chứa thông tin validation result.

    Raises:
        HTTPException: Nếu token không hợp lệ hoặc đã hết hạn.
    """

    if not request.token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Access token không được cung cấp"
        )

    # Verify the token
    try:
        payload = verify_token(request.token)
        expired_at = payload.get("exp")

        # Kiểm tra thời gian hết hạn
        if expired_at < int(datetime.now(tz=timezone.utc).timestamp()):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Access token đã hết hạn"
            )

        # Trả về response object nếu token hợp lệ
        return TokenValidationResponse(valid=True, message="Token is valid")

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Access token không hợp lệ"
        ) from e


# --- Cấu hình Authlib ---
# Tạo một đối tượng config cho Authlib từ biến môi trường
auth_config = Config(environ=os.environ)
oauth = OAuth(auth_config)

# Đăng ký Google OAuth
try:
    oauth.register(
        name='google',
        server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
        client_id=os.getenv("GOOGLE_CLIENT_ID"),
        client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
        client_kwargs={'scope': 'openid email profile'}
    )
except Exception as e:
    logger.error(f"Failed to register Google OAuth: {e}")
    raise Exception("Google OAuth configuration failed")

@router.get('/login/google')
async def login_via_google(request: Request):
    """
    Endpoint bắt đầu quá trình đăng nhập.
    Nó sẽ chuyển hướng người dùng đến trang đăng nhập của Google.
    """
    try:
        redirect_uri = request.url_for('auth_google_callback')
        return await oauth.google.authorize_redirect(request, redirect_uri)
    except Exception as e:
        logger.error(f"Error initiating Google login: {e}")
        raise HTTPException(status_code=500, detail="Failed to initiate Google login")

@router.get('/google/callback', name='auth_google_callback')
async def auth_google_callback(request: Request):
    """
    Endpoint callback mà Google sẽ gọi lại sau khi người dùng xác thực.
    """
    try:
        # Lấy access token từ Google
        token = await oauth.google.authorize_access_token(request)
    except Exception as e:
        logger.error(f"Error authorizing access token from Google: {e}")
        raise HTTPException(status_code=400, detail="Could not authorize with Google.")

    # Lấy thông tin người dùng từ Google
    user_info = token.get('userinfo')
    if not user_info or not user_info.get('email'):
        raise HTTPException(status_code=400, detail="Could not retrieve user info from Google.")

    user_email = user_info['email']
    username = re.sub(r'[^a-zA-Z0-9]', '', user_email.split('@')[0])


    # Kiểm tra xem user đã tồn tại trong DB chưa
    db_user = user_collection.find_one({"email": user_email})

    if not db_user:
        placeholder_password = f"google-oauth2|{uuid.uuid4()}"
        hashed_password = pwd_context.hash(placeholder_password)
        # Nếu chưa, tạo user mới
        logger.info(f"New user from Google: {user_email}. Creating account...")


        # Tạo user mới với thông tin từ Google
        user_collection.insert_one({
            "email": user_email,
            "username": username,
            "password": hashed_password,  # Mật khẩu tạm thời, sẽ không dùng đến
            "avatar_url": user_info.get('picture', None),
            "role": "user",
            "is_active": True,
        })

        # Lấy lại user vừa tạo để đảm bảo có _id và các trường khác
        db_user = user_collection.find_one({"email": user_email})
        if not db_user: # Kiểm tra lại sau khi insert
             raise HTTPException(status_code=500, detail="Could not create and retrieve new user account.")


    # 1. Tạo một authorization code ngẫu nhiên, ngắn hạn
    auth_code = str(uuid.uuid4())

    # 2. Lưu email của user vào Redis với key là auth_code
    # Set thời gian hết hạn ngắn, ví dụ 1 phút (60 giây)
    app_state = get_app_state(request=request)
    redis_client = app_state.redis
    redis_client.set(f"google_auth_code:{auth_code}", user_email, ex=60)

    # 3. Chuyển hướng về trang callback của frontend, đính kèm code này
    frontend_callback_url = f"{config.FRONTEND_URL}/auth/callback?code={auth_code}"

    logger.info(f"Google auth successful for {user_email}. Redirecting to frontend with temp code.")
    return RedirectResponse(url=frontend_callback_url)


# === TẠO ENDPOINT MỚI ĐỂ ĐỔI CODE LẤY TOKEN ===
@router.post("/token/google")
async def exchange_google_code_for_token(request: Request,response: Response, code: str):
    """
    Frontend sẽ gọi endpoint này với code tạm thời để lấy JWT token và cookie.
    """
    app_state = get_app_state(request=request)
    redis_client = app_state.redis
    # 1. Lấy email từ Redis bằng code và xóa code ngay lập tức
    redis_key = f"google_auth_code:{code}"
    user_email_bytes =  redis_client.get(redis_key)
    if not user_email_bytes:
        raise HTTPException(status_code=400, detail="Invalid or expired authorization code.")

    redis_client.delete(redis_key) # Dùng một lần
    user_email = user_email_bytes.decode()

    user_collection.update_one({
        "email": user_email
    }, {
        "$set": {
            "last_login": datetime.now(timezone.utc),  # Cập nhật thời gian đăng nhập
            "login_type": "google"
        }
    })
    # Tạo JWT token cho người dùng
    access_token_expires = timedelta(minutes= int(config.ACCESS_TOKEN_EXPIRE_MINUTES))

    refresh_token_expires = timedelta(days=7)

    access_token = create_access_token(
        data={"sub": user_email.lower()}, expires_delta=access_token_expires
    )


    IS_PRODUCTION = os.getenv("APP_ENVIRONMENT", "development").lower() == "production"

    refresh_token_value = await  create_refresh_token(user_email.lower())



    # Access Token Cookie
    response.set_cookie(
            key="access_token_cookie", # Tên cookie cho access token
            value=access_token,
            max_age=int(access_token_expires.total_seconds()), # Thời gian sống bằng access token
            httponly=True,
            secure=IS_PRODUCTION, # True trong production (HTTPS), False khi dev với HTTP
            samesite="lax", # Hoặc "strict"
            path="/",
    )

    # Refresh Token Cookie (logic của bạn đã có, điều chỉnh secure)
    response.set_cookie(
            key="refresh_token",
            value=refresh_token_value,
            max_age=int(refresh_token_expires.total_seconds()),
            httponly=True,
            secure=IS_PRODUCTION, # True trong production, False khi dev với HTTP
            samesite="lax",
            path="/api/user/refresh-token",
        )

    user_info = user_collection.find_one({"email": user_email})
    user = {
        "email": user_info.get("email"),
        "username": user_info.get("username"),
        "role": user_info.get("role"),
        "avatar_url": user_info.get("avatar_url"),
        "login_type": user_info.get("login_type", "google"),
    }
    return {"message": "Login successful", "user": user}