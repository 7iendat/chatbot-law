from fastapi import APIRouter, Depends, HTTPException, Query,status, Response, Request
from services.auth_service import (
    verify_token, register_user
)
from datetime import datetime , timezone
from services.user_service import get_paginated_users, delete_user,change_password,reset_password_request,reset_password, generate_and_store_verification_code, authenticate_user, verify_login_code,refresh_access_token,verify_forgot_password_code
from dependencies import bearer_scheme, get_current_user, admin_required
from schemas.user import LoginRequest, LoginResponse, RegisterRequest, RegisterResponse, UserOut,PaginatedResponse,ProfileResponse,ChangePasswordRequest,PasswordResetRequest,PasswordReset,VerifyLoginRequest,VerifyForgotPassRequest, ResentVerifyCode,TokenValidationResponse,TokenValidationRequest


router = APIRouter()

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

# @router.post("/logout")
# async def logout(req: Request,credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
#     return await logout_user(req, credentials)

@router.post("/logout")
async def logout_user(response: Response):
    """
    Đăng xuất người dùng bằng cách xóa HttpOnly cookies.
    """

    response.delete_cookie(key="access_token_cookie", path="/", samesite="lax") # Đảm bảo các thuộc tính khớp với lúc set
    response.delete_cookie(key="refresh_token", path="/", samesite="lax")
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
