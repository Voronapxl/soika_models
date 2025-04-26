import sys

from loguru import logger
from iduconfig import Config

from app.common.gpu_handler.gpu_handler import GPUHandler
from app.common.models.models import ModelsInit
from app.common.exceptions.http_exception_wrapper import http_exception


config = Config()
gpu_handler = GPUHandler(config)
models_initialization = ModelsInit()

logger.remove()
logger.add(sys.stderr)
logger.add(f'{config.get("LOGS_FILE")}.log')
