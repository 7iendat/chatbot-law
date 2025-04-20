from pydantic import BaseModel,EmailStr
from typing import List, Optional
from pydantic import Field


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=4)
    password: str = Field(..., min_length=6)
    email: EmailStr
    role: Optional[str] = "user"

class RegisterResponse(BaseModel):
    message: str

class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    session_id: str
    username: str
    email: EmailStr
    role: Optional[str] = "user"

class UserOut(BaseModel):
    username: str
    email: EmailStr
    role: Optional[str] = "user"