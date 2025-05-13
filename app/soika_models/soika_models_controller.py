from fastapi import APIRouter

from .emotions_request_model import TextPayload
from .soika_models_service import soika_models_service


soika_models_controller = APIRouter(prefix="/soika_models", tags=["Soika Models"])


@soika_models_controller.post("/extract_emotions")
async def get_emotions(payload: TextPayload):
    result = await soika_models_service.extract_emotions(payload.text)
    return result

@soika_models_controller.post("/extract_ner")
async def get_ner(payload: TextPayload):
    result = await soika_models_service.extract_ner(payload.text)
    return result