from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

from app.dependencies import config, http_exception, gpu_handler, models_initialization
from app.soika_models import soika_models_controller


@asynccontextmanager
async def lifespan(app: FastAPI):
    await models_initialization.init_models()
    yield

app = FastAPI(
    title=config.get("APP_NAME"),
    description=config.get("APP_DESCRIPTION"),
    version=config.get("APP_VERSION"),
    lifespan=lifespan
)

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=dict[str, str])
def read_root():
    return RedirectResponse(url='/docs')

@app.get("ping/")
async def root():
    """Check if the server is running"""

    return "pong"

@app.get("/logs")
async def get_logs():
    """
    Get logs file from app
    """

    try:
        return FileResponse(
            f"{config.get('LOG_FILE')}.log",
            media_type='application/octet-stream',
            filename=f"{config.get('LOG_FILE')}.log",
        )
    except FileNotFoundError as e:
        raise http_exception(
            status_code=404,
            msg="Log file not found",
            _input={"lof_file_name": f"{config.get('LOG_FILE')}.log"},
            _detail={"error": e.__str__()}
        )
    except Exception as e:
        raise http_exception(
            status_code=500,
            msg="Internal server error during reading logs",
            _input={"lof_file_name": f"{config.get('LOG_FILE')}.log"},
            _detail={"error": e.__str__()}
        )


app.include_router(soika_models_controller)
