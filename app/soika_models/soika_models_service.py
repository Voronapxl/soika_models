import asyncio

from app.dependencies import models_initialization, gpu_handler


class SoikaModelsService:

    async def extract_emotions(self, text: str):
        """
        Function extracts emotions from text and returns list of emotions
        Args:
            text (str): text to extract emotions from
        Returns:
            list[dict]: dictionary with emotions and their probabilities
        """

        gpu_handler.check_cuda()
        model = models_initialization._classification_model
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(models_initialization.executor, model, text)
        return result


soika_models_service = SoikaModelsService()
