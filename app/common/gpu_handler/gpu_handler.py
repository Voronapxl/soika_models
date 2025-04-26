import time

import torch
from loguru import logger
from iduconfig import Config

from app.common.exceptions import http_exception


class GPUHandler:

    def __init__(self, config: Config):

        self.config = config

    def check_cuda(self):

        max_retries = int(self.config.get("MAX_RETRIES"))
        retries = 0
        while retries < max_retries:
            if torch.cuda.is_available():
                return
            retries += 1
            logger.warning(f"GPU is not available, {retries} retry in 5 seconds")
            time.sleep(5)

        raise http_exception(
            504,
            "Couldn't start gpu calculations in reasonable time, with n retries",
            retries
        )
