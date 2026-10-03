import io
import logging
import os
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import torch
from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from api.schemas import Health, Prediction
from medvision.inference import Predictor

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_IMAGE_PIXELS = 4_000_000
logger = logging.getLogger(__name__)


def decode_image(content: bytes) -> Image.Image:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(content)) as image:
                if image.format not in {"PNG", "JPEG"}:
                    raise HTTPException(
                        status_code=415, detail="Only PNG and JPEG images are supported"
                    )
                if image.width * image.height > MAX_IMAGE_PIXELS:
                    raise HTTPException(status_code=413, detail="Image exceeds 4 million pixels")
                image.load()
                return image.convert("RGB")
    except (Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
        raise HTTPException(status_code=413, detail="Image dimensions are too large") from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="Invalid or corrupt image") from exc


def create_app(checkpoint: Path | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        torch.set_num_threads(2)
        path = (
            checkpoint
            if checkpoint is not None
            else Path(os.environ.get("MEDVISION_CHECKPOINT", "artifacts/model.pt"))
        )
        app.state.predictor = None
        if path.is_file():
            # Incompatible or corrupt artifacts fail startup.
            app.state.predictor = Predictor(path)
        else:
            logger.warning("Model checkpoint is missing; train a model or set MEDVISION_CHECKPOINT")
        yield
        app.state.predictor = None

    app = FastAPI(
        title="Production MedVision API",
        version="0.1.0",
        description=(
            "PathMNIST research and portfolio demonstration. Not intended for clinical use."
        ),
        lifespan=lifespan,
    )

    @app.get("/health", response_model=Health)
    def health(response: Response) -> Health:
        predictor = app.state.predictor
        if predictor is None:
            response.status_code = 503
        return Health(
            status="ready" if predictor is not None else "model_not_ready",
            model_loaded=predictor is not None,
            model_version=predictor.metadata["model_version"] if predictor is not None else None,
        )

    @app.post("/predict", response_model=Prediction)
    async def predict(file: Annotated[UploadFile, File()]) -> Prediction:
        try:
            predictor = app.state.predictor
            if predictor is None:
                raise HTTPException(status_code=503, detail="Model checkpoint is not loaded")
            content = await file.read(MAX_UPLOAD_BYTES + 1)
            if len(content) > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="Image exceeds 5 MiB")
            image = await run_in_threadpool(decode_image, content)
            result = await run_in_threadpool(predictor.predict, image)
            return Prediction(**result)
        finally:
            await file.close()

    return app


app = create_app()
