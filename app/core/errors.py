"""
Centralized Exception Definitions and FastAPI Exception Handlers.
"""

from fastapi import Request, status
from fastapi.responses import JSONResponse
from app.core.logging import logger


class CustomGPTException(Exception):
    """Base exception class for application errors."""
    def __init__(self, message: str, status_code: int = 400):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class ModelNotLoadedException(CustomGPTException):
    def __init__(self, message: str = "PyTorch GPT Model weights or Tokenizer not loaded."):
        super().__init__(message=message, status_code=status.HTTP_503_SERVICE_UNAVAILABLE)


class UnsupportedFileTypeException(CustomGPTException):
    def __init__(self, message: str = "Unsupported file format. Upload PDF, PPTX, DOCX, TXT, or CSV."):
        super().__init__(message=message, status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE)


async def custom_exception_handler(request: Request, exc: CustomGPTException):
    logger.error(f"CustomGPTException triggered [{exc.status_code}]: {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.message, "path": str(request.url)}
    )


async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled Internal Error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": f"Internal Server Error: {str(exc)}", "path": str(request.url)}
    )
