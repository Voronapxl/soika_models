from fastapi import APIRouter

from .soika_models_service import soika_models_service


soika_models_controller = APIRouter(prefix="/soika_models", tags=["Soika Models"])


@soika_models_controller.post("/extract_emotions")
async def get_emotions(text: str):

    result = await soika_models_service.extract_emotions(text)
    return result

@soika_models_controller.post("/extract_ner")
async def get_ner(text: str):

    result = await soika_models_service.extract_ner(text)
    return result