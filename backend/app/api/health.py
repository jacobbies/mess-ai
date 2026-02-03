from fastapi import APIRouter, Request
from app.models.schemas import HealthResponse
from app.core.config import settings

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    s3_service = request.app.state.s3_service
    s3_connected = s3_service.check_connection()

    return HealthResponse(
        status="healthy" if s3_connected else "degraded",
        s3_connected=s3_connected,
        version=settings.APP_VERSION
    )