from fastapi import APIRouter, HTTPException, Request
from app.models.schemas import StreamResponse
from app.services.s3_keys import S3Keys

router = APIRouter()


@router.get("/stream/{track_id}", response_model=StreamResponse)
async def get_stream_url(track_id: str, request: Request):
    s3_service = request.app.state.s3_service

    try:
        audio_key = S3Keys.audio_smd(track_id)
        url = s3_service.get_audio_preurl(audio_key)

        return StreamResponse(
            url=url,
            track_id=track_id,
            expires_in=600
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate stream URL: {str(e)}")