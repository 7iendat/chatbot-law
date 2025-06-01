from pydantic import BaseModel, EmailStr, Field, field_validator, HttpUrl
from typing import List, Optional
import re
from enum import Enum

# Define UserRole as an Enum
class UserRole(str, Enum):
    USER = "user"
    ADMIN = "admin"

class RegisterRequest(BaseModel):
    username: str = Field(
        ...,
        min_length=4,
        max_length=30,
        description="Tên đăng nhập của người dùng, chỉ chấp nhận chữ cái, số và dấu gạch dưới",
        example="john_doe123"
    )

    password: str = Field(
        ...,
        min_length=8,
        max_length=64,
        description="Mật khẩu của người dùng, yêu cầu ít nhất 8 ký tự, bao gồm chữ hoa, chữ thường, số và ký tự đặc biệt",
        example="StrongP@ss123"
    )

    email: EmailStr = Field(
        ...,
        description="Địa chỉ email hợp lệ của người dùng",
        example="user@example.com",
        pattern=r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    )

    role: UserRole = Field(
        default=UserRole.USER,
        description="Vai trò của người dùng trong hệ thống",
        example=UserRole.USER
    )

    avatar_url: Optional[HttpUrl] = Field(
        None,
        description="URL ảnh đại diện của người dùng",
        example="https://example.com/avatars/default.png"
    )

    is_active: bool = Field(
        default=True,
        description="Trạng thái hoạt động của người dùng",
        example=True
    )

    # Custom validators
    @field_validator("username")
    def username_alphanumeric(cls, v):
        if not re.match(r'^[a-zA-Z0-9_]+$', v):
            raise ValueError('Tên đăng nhập chỉ được chứa chữ cái, số và dấu gạch dưới')
        return v

    @field_validator('password')
    def password_strength(cls, v):
        if not re.search(r'[A-Z]', v):
            raise ValueError('Mật khẩu phải chứa ít nhất một ký tự viết hoa')
        if not re.search(r'[a-z]', v):
            raise ValueError('Mật khẩu phải chứa ít nhất một ký tự viết thường')
        if not re.search(r'[0-9]', v):
            raise ValueError('Mật khẩu phải chứa ít nhất một chữ số')
        if not re.search(r'[^a-zA-Z0-9]', v):
            raise ValueError('Mật khẩu phải chứa ít nhất một ký tự đặc biệt')
        return v

    @field_validator('role')
    def validate_role(cls, v):
        if v not in UserRole:
            raise ValueError(f'Vai trò không hợp lệ. Các vai trò được hỗ trợ: {", ".join([role.value for role in UserRole])}')
        return v

    @field_validator('avatar_url')
    def validate_avatar_url(cls, v):
        if v is None:
            return v
        try:
            from urllib.parse import urlparse
            parsed = urlparse(str(v))
            if not all([parsed.scheme, parsed.netloc]):
                raise ValueError('URL ảnh đại diện không hợp lệ')
        except Exception:
            raise ValueError('URL ảnh đại diện không hợp lệ')
        return v
class RegisterResponse(BaseModel):
    message: str

class LoginRequest(BaseModel):
    email: str
    password: str

class UserResponse(BaseModel):
    email: EmailStr
    username: str = "N/A"
    role: str
    avatar_url: Optional[str] = None

class LoginResponse(BaseModel):
    # accessToken: str
    # token_type: str = "bearer"
    user: UserResponse
    message: str


class VerifyLoginRequest(BaseModel):
    email: str
    code: str

class VerifyForgotPassRequest(BaseModel):
    email: str
    code: str

class UserOut(BaseModel):
    username: str
    email: EmailStr
    role: UserRole = Field(default=UserRole.USER, description="User role", example=UserRole.USER)
    avatar_url: Optional[str] = None
    is_active:bool = Field(default=True)

class RefreshTokenRequest(BaseModel):
    refresh_token: str

class PaginationMetadata(BaseModel):
    total: int
    page: int
    page_size: int
    pages: int
    has_more: bool

class PaginatedResponse(BaseModel):
    items: List[UserOut]
    metadata: PaginationMetadata

class ProfileResponse(BaseModel):
    username: str
    email: EmailStr
    role: UserRole = Field(default=UserRole.USER, description="User role", example=UserRole.USER)
    avatar_url: Optional[str] = Field(
        None,
        description="URL ảnh đại diện của người dùng",
        example="https://example.com/avatars/default.png"
    )
    is_active: bool = Field(
        default=True,
        description="Trạng thái hoạt động của người dùng",
    )

class ChangePasswordRequest(BaseModel):
    current_password: str = Field(..., min_length=8, description="Mật khẩu hiện tại", example="CurrentPassword123@")
    new_password: str = Field(
        ...,
        min_length=8,
        max_length=64,
        description="Mật khẩu của người dùng, yêu cầu ít nhất 8 ký tự, bao gồm chữ hoa, chữ thường, số và ký tự đặc biệt",
        example="NewPassword123@"
    )

    @field_validator('new_password')
    def password_strength(cls, v):
        if not re.search(r'[A-Z]', v):
            raise ValueError('Mật khẩu phải chứa ít nhất một ký tự viết hoa')
        if not re.search(r'[a-z]', v):
            raise ValueError('Mật khẩu phải chứa ít nhất một ký tự viết thường')
        if not re.search(r'[0-9]', v):
            raise ValueError('Mật khẩu phải chứa ít nhất một chữ số')
        if not re.search(r'[^a-zA-Z0-9]', v):
            raise ValueError('Mật khẩu phải chứa ít nhất một ký tự đặc biệt')
        return v

class PasswordResetRequest(BaseModel):
    email: EmailStr

class PasswordReset(BaseModel):
    code: str
    newPassword: str = Field(
        ...,
        min_length=8,
        max_length=64,
        description="Mật khẩu của người dùng, yêu cầu ít nhất 8 ký tự, bao gồm chữ hoa, chữ thường, số và ký tự đặc biệt",
        example="NewPassword123@"
    )

    @field_validator('newPassword')
    def password_strength(cls, v):
        if not re.search(r'[A-Z]', v):
            raise ValueError('Mật khẩu phải chứa ít nhất một ký tự viết hoa')
        if not re.search(r'[a-z]', v):
            raise ValueError('Mật khẩu phải chứa ít nhất một ký tự viết thường')
        if not re.search(r'[0-9]', v):
            raise ValueError('Mật khẩu phải chứa ít nhất một chữ số')
        if not re.search(r'[^a-zA-Z0-9]', v):
            raise ValueError('Mật khẩu phải chứa ít nhất một ký tự đặc biệt')
        return v

class ResentVerifyCode(BaseModel):
    email: str

class TokenValidationRequest(BaseModel):
    token: str

# Response model
class TokenValidationResponse(BaseModel):
    valid: bool
    message: str = None