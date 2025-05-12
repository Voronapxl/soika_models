import asyncio
import re
from flair.data import Sentence

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

    async def extract_ner(self, text: str):
        gpu_handler.check_cuda()
        cleaned_text = re.sub(r"\[.*?]", "", text)
        sentence = Sentence(cleaned_text)
        if models_initialization._ner_model:
            print(models_initialization._ner_model)
        model = models_initialization._ner_model

        # loop = asyncio.get_running_loop()
        model.predict(sentence)

        return str(sentence)



soika_models_service = SoikaModelsService()
