from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from services.auth_service import (
    login_user, logout_user, register_user
)
from services.user_service import get_users
from dependencies import bearer_scheme, get_current_user
from schemas.auth import LoginRequest, LoginResponse, RegisterRequest, RegisterResponse, UserOut

router = APIRouter()

@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    return login_user(request)

@router.post("/register", response_model=RegisterResponse)
async def register(request: RegisterRequest):
    return register_user(request)

@router.post("/logout")
async def logout(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    return logout_user(credentials)

@router.get("/users", response_model=list[UserOut])
async def list_users():
    return get_users()