from datetime import timedelta, datetime, UTC
import os
from typing import Optional, Tuple
from pydantic import BaseModel
from fastapi import HTTPException, APIRouter, Depends, Response, Cookie, Request
from services.dependencies.rate_limiter import limiter

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
import jwt
from pwdlib import PasswordHash
import logging

import services.db_utils as db

pw_hasher = PasswordHash.recommended()
JWT_SECRET = os.getenv("JWT_SECRET")
SAME_SITE_SETTING = 'none' if not os.getenv('SITE_DOMAIN') else 'lax'


class LoginRequest(BaseModel):
    username: str
    password: str

router = APIRouter(prefix='/api/auth', tags=['auth'])
logger = logging.getLogger("uvicorn.error")

async def get_current_user(access_token: Optional[str] = Cookie(None)) -> int:
    """
    Extracts access token from cookie and attempts to decode it. Returns user ID on success.
    :param access_token: Dependency that extracts cookie from header (Dependency Injection upstream)
    :return: (int) user id corresponding to access token
    """
    if not access_token:
        logger.log(logging.INFO, "NO ACCESS TOKEN PROVIDED")
        raise HTTPException(status_code=401, detail={
            "message": "No access token provided",
            "error_code": "ACCESS_TOKEN_ABSENT"
        })

    try:
        payload = jwt.decode(access_token, JWT_SECRET, algorithms=["HS256"])
        user_id = payload.get("user_id")
        token_type = payload.get("type")
        username = payload.get("username")
        if token_type != "access":
            raise HTTPException(status_code=401, detail={
                "message": "Invalid token type",
                "error_code": "TOKEN_TYPE_MISMATCH"
            })
        if user_id is None:
            raise HTTPException(status_code=401, detail={
                "message": "Invalid Token provided",
                "error_code": "INVALID_TOKEN"
            })
        return user_id

    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail={
            "message": "Token Expired",
            "error_code": "TOKEN_EXPIRED"
        })

@router.post('/login')
@limiter.limit("30/minute")
async def login(request:Request, loginreq: LoginRequest, response:Response, db: AsyncSession = Depends(db.get_db_session)):
    access, refresh = await handle_login(loginreq, db)
    response.set_cookie('access_token',
                        value = access,
                        httponly=True,
                        secure=True,
                        samesite=SAME_SITE_SETTING,
                        path="/")
    response.set_cookie('refresh_token',
                        value = refresh,
                        httponly=True,
                        secure=True,
                        samesite=SAME_SITE_SETTING,
                        path="/api/auth")
    return {"message": "Login success"}

@router.post('/logout')
@limiter.limit("30/minute")
async def logout(request: Request, response: Response, user:int =  Depends(get_current_user), db: AsyncSession = Depends(db.get_db_session)):
    response.set_cookie('access_token',
                        value="",
                        max_age=0,
                        httponly=True,
                        secure=True,
                        samesite=SAME_SITE_SETTING,
                        path="/")
    response.set_cookie('refresh_token',
                        value="",
                        max_age=0,
                        httponly=True,
                        secure=True,
                        samesite=SAME_SITE_SETTING,
                        path="/api/auth")

    retry_count = 0
    invalidated = False
    while retry_count < 4:
        invalidated = await invalidate_refresh_token(user, db)
        if invalidated:
            break
        else:
            retry_count += 1

    return {"message": "Logout success"} if invalidated else {"message": "Logout failed"}

@router.post('/refresh')
@limiter.limit("10/minute")
async def refresh(request: Request,
                  response: Response,
                  refresh_token: Optional[str] = Cookie(None),
                  db: AsyncSession = Depends(db.get_db_session)):
    if not refresh_token:
        raise HTTPException(status_code=401, detail={
            "message": "Refresh token required.",
            "error_code": "NO_REFRESH_TOKEN"
        })

    try:
        payload = jwt.decode(refresh_token, JWT_SECRET, algorithms=["HS256"], options={"require": ["exp"]})
        user_id = payload.get("user_id")
        username = payload.get("username")
        token_type = payload.get("type")

        if token_type != "refresh":
            raise HTTPException(status_code=401, detail={
            "message": "Token provided is not a refresh token.",
            "error_code": "TOKEN_TYPE_MISMATCH"
        })
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail={
            "message": "Access token is expired",
            "error_code": "EXPIRED_ACCESS_TOKEN",
        })
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail={
            "message": "Invalid access token provided",
            "error_code": "INVALID_ACCESS_TOKEN",
        })

    refresh_valid = await validate_refresh_token(user_id, refresh_token, db_session=db)
    if not refresh_valid:
        raise HTTPException(status_code=401, detail={
            "message": "Unable to validate refresh token",
            "error_code": "INVALID_REFRESH_TOKEN"
        })
    user_data = {
        "username": username,
        "user_id": user_id,
    }
    new_access = await create_access_token(data=user_data)

    response.set_cookie('access_token',
                        value = new_access,
                        httponly=True,
                        secure=True,
                        samesite=SAME_SITE_SETTING,)

    return {"message": "Token refreshed successfully",
            "username": username,}

if os.getenv("SIGNUPS_ENABLED", "false").lower() == "true":
    @router.post('/signup')
    @limiter.limit("30/day")
    async def signup(request:Request, loginreq: LoginRequest, response: Response, db: AsyncSession = Depends(db.get_db_session)):
        username = loginreq.username
        password = loginreq.password
        new_user = await create_user(username=username, password=password, db_session=db)
        access, refresh = await handle_login(loginreq, db)

        response.set_cookie('access_token',
                            value = access,
                            httponly=True,
                            secure=True,
                            samesite=SAME_SITE_SETTING,
                            path="/")
        response.set_cookie('refresh_token',
                            value = refresh,
                            httponly=True,
                            secure=True,
                            samesite=SAME_SITE_SETTING,
                            path="/api/auth")

        return {"message": "Signup success",
                "user":{
                    "username": new_user.username,
                }}


async def _retrieve_user(username: str, db_session: AsyncSession) -> db.User | None:
    user = await db.get_user_by_username(username, db_session)
    if not user:
        return None
    return user

def hash_password(password: str) -> str:
    return pw_hasher.hash(password)

async def authenticate_and_retrieve_user(username: str, input_pw: str, db_session: AsyncSession) -> Tuple[int, bool]:
    user: db.User = await _retrieve_user(username, db_session)
    if not user:
        return None, False
    stored_pw = user.hashed_password
    user_id = user.user_id
    return user_id, pw_hasher.verify(input_pw, stored_pw)

async def handle_login(login_request: LoginRequest, db_session: AsyncSession):
    username = login_request.username
    password = login_request.password
    user_id, is_valid = await authenticate_and_retrieve_user(username, password, db_session)
    if not is_valid:
        raise HTTPException(status_code=401, detail={
            "message": "Incorrect username or password",
            "error_code": "INVALID_LOGIN"
        })
    login_info = {"username": username, "user_id": user_id}
    access = await create_access_token(data=login_info)
    refresh = await create_refresh_token(data=login_info, db_session=db_session)
    if not access or not refresh:
        raise HTTPException(status_code=401, detail={
            "message": "Unable to access or refresh token",
            "error_code": "TOKEN_ERR"
        })
    # assume this will be wrapped in another function for the actual endpoint that stores the refresh token and dispatches both tokens back to requestor.
    return access, refresh

async def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    token_ttl = expires_delta if expires_delta else timedelta(minutes=15)
    expire = datetime.now(tz=UTC) + token_ttl
    to_encode.update({"exp": expire, "type":"access"})
    return jwt.encode(to_encode,JWT_SECRET,algorithm="HS256")

async def create_refresh_token(data: dict, db_session: AsyncSession, expires_delta: Optional[timedelta] = None):
    to_encode = {}
    user_id = data.get("user_id", None)
    username = data.get("username", None)
    if not user_id:
        return None
    token_ttl = expires_delta if expires_delta else timedelta(days=14)
    expire = datetime.now(tz=UTC) + token_ttl
    to_encode.update({"exp": expire, "type":"refresh", "user_id": user_id, "username": username})
    token = jwt.encode(to_encode,JWT_SECRET,algorithm="HS256")
    await store_refresh_token(user_id=user_id, username=username, expiry=expire, token=token, db_session=db_session)
    return token

async def store_refresh_token(user_id:int, username:str, expiry:datetime, token:str, db_session: AsyncSession) -> bool:
    try:
        await db.add_or_update_refresh_token(user_id=user_id, username=username, expires=expiry, refresh_token=token, session=db_session)
    except IntegrityError:
        raise HTTPException(status_code=400, detail={
            "message": "Duplicate User ID",
            "error_code": "DUPLICATE_USER_ID",
        })
    except SQLAlchemyError:
        raise HTTPException(status_code=500, detail={
            "message": "Database Error",
            "error_code": "DB_ERROR"
        })
    return True

async def retrieve_refresh_info(user_id: int, db_session: AsyncSession) -> Optional[dict]:
    try:
        rt_entry = await db.get_refresh_token_by_uid(user_id, db_session)
        if not rt_entry:
            return None
    except SQLAlchemyError:
        raise HTTPException(status_code=500, detail="DB Error")
    user_id = rt_entry.user_id
    exp = rt_entry.expires
    token = rt_entry.refresh_token
    return {"user_id": user_id, "exp": exp, "token": token}

async def validate_refresh_token(user_id: int, token: str, db_session: AsyncSession) -> bool:
    refresh_info = await retrieve_refresh_info(user_id, db_session)
    if not refresh_info:
        return False
    stored_uid = refresh_info.get("user_id")
    stored_token = refresh_info.get("token")
    if not (user_id == stored_uid and token == stored_token):
        return False
    return True

async def invalidate_refresh_token(user_id: int, db_session: AsyncSession) -> bool:
    success = await db.delete_refresh_token(user_id, db_session)
    return bool(success)

async def create_user(username: str, password: str, db_session: AsyncSession) -> db.User:
    hashed_password = hash_password(password)
    try:
        user = await db.add_user(username, hashed_password, db_session)
    except IntegrityError as e:
        print(f"IntegrityError: {e}")
        raise HTTPException(status_code=400, detail={
            "message": "Username already exists",
            "error_code": "DUPLICATE_USERNAME",
        })
    except SQLAlchemyError as e:
        print(f"SQLAlchemyError: {e}")
        raise HTTPException(status_code=500, detail={
            "message": "Database Error",
            "error_code": "DB_ERROR",
            "debug": str(e),
            "error_type": type(e).__name__
        })
    return user

async def retrieve_uid_from_token(token: str) -> int:
    decoded = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    return decoded.get("user_id")
