"""
FastAPI Enterprise Application Entrypoint.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.core.logging import setup_logging, logger
from app.core.errors import CustomGPTException, custom_exception_handler, global_exception_handler
from app.services.gpt_service import gpt_service
from app.api.v1.router import api_router
from app.api.v1.endpoints import router as root_endpoints


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Setup logging & load model assets
    setup_logging()
    logger.info(f"Starting {settings.APP_NAME} in [{settings.ENV.upper()}] mode...")
    gpt_service.load_assets()
    yield
    logger.info("Shutting down application...")


app = FastAPI(
    title=settings.APP_NAME,
    description="Enterprise Multi-Modal Generative AI Platform",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for browser frontend cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception Handlers
app.add_exception_handler(CustomGPTException, custom_exception_handler)
app.add_exception_handler(Exception, global_exception_handler)

# Include v1 Router (/api/v1/chat, /api/v1/upload, /api/v1/health)
app.include_router(api_router, prefix="/api/v1")

# Root Backwards-Compatible Endpoints (/chat, /upload, /health)
app.include_router(root_endpoints)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
