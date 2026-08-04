import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("errors")


class AppError(Exception):
    """业务异常：可预期的错误，如"知识库不存在"。"""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def register_exception_handlers(app: FastAPI):
    @app.exception_handler(AppError)
    async def _app_error(request: Request, exc: AppError):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        logger.exception("unhandled error: %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content={"detail": "internal server error"})