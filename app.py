from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from services.dependencies.rate_limiter import limiter

from contextlib import asynccontextmanager
import os
from services import inference_engine
import services.dependencies.engine_manager as em
import services.auth_service as auth
import services.document_io as docs
from services.db_utils import init_db
import logging

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    logging.getLogger().setLevel(logging.INFO)

    em._engine_manager = em.EngineManager()
    print("Engine Manager initialised.")
    await init_db()
    print("DB Initialised. API Ready")

    yield

    print("Shutting down, cleaning up engines")
    await em._engine_manager.cleanup()
    print("All engines cleaned up")


app = FastAPI(lifespan=lifespan)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

allowed_origins = os.getenv("ALLOWED_ORIGINS").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(inference_engine.router)
app.include_router(auth.router)
app.include_router(docs.router)

@app.get('/health')
async def health():
    return {'status': 'ok'}

@app.get("/robots.txt", response_class=PlainTextResponse)
def robots():
    return """User-agent: *
Disallow: /"""
