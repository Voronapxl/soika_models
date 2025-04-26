from fastapi import APIRouter

from .soika_models_service import soika_models_service


soika_models_controller = APIRouter(prefix="/soika_models", tags=["Soika Models"])


@soika_models_controller.get("/extract_emotions")
async def get_emotions(text: str):

    result = await soika_models_service.extract_emotions(text)
    return result
