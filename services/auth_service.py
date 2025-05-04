import uuid
from jose import JWTError, jwt
from datetime import datetime, timedelta
from fastapi import HTTPException, Depends, Request
from schemas.auth import RegisterRequest, LoginRequest
from db.mongoDB import user_collection, blacklist_collection
from utils.utils import hash_password, verify_password
from config import SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Khởi tạo context mật khẩu

bearer_scheme = HTTPBearer()
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
        hashed_password = hash_password(user.password)

        # Lưu user mới
        user_collection.insert_one({
            "username": user.username,
            "password": hashed_password,
            "email": user.email,
            "role": user.role or "user"
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

    access_token = create_access_token(data={"sub": request.email})

    return  {
        "access_token": access_token,
        "username": user["username"],
        "email": user["email"],
        "role": user.get("role", "user")
    }

# Ham dang xuat
def logout_user(request: Request, credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    # Giải mã token để lấy thời gian hết hạn
    try:
        token = credentials.credentials
        payload = verify_token(token)
        exp = payload.get("exp")
        if not exp:
            raise HTTPException(status_code=400, detail="Token không hợp lệ")

        blacklist_collection.insert_one({
            "token": token,
            "expires_at": datetime.fromtimestamp(exp)
        })

        return {"message": "Đăng xuất thành công"}
    except JWTError:
        raise HTTPException(status_code=400, detail="Token không hợp lệ")

# Hàm kiểm tra xem token có hợp lệ không
def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Token không hợp lệ")