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

from api.artifacts import checkpoint_path
from api.schemas import BatchPrediction, Health, ModelInfo, Prediction
from medvision.inference import Predictor

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_IMAGE_PIXELS = 4_000_000
MAX_BATCH_FILES = 16
MAX_BATCH_BYTES = 20 * 1024 * 1024
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
        # An explicit path (for tests/embedding) takes precedence over environment settings.
        s3_uri = os.environ.get("MEDVISION_MODEL_S3_URI") if checkpoint is None else None
        try:
            with checkpoint_path(path, s3_uri) as resolved_path:
                if resolved_path.is_file():
                    # Incompatible or corrupt artifacts fail startup.
                    app.state.predictor = Predictor(resolved_path)
                else:
                    logger.warning(
                        "Model checkpoint is missing; train a model or set MEDVISION_CHECKPOINT"
                    )
            yield
        finally:
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

    def require_predictor() -> Predictor:
        if app.state.predictor is None:
            raise HTTPException(status_code=503, detail="Model checkpoint is not loaded")
        return app.state.predictor

    @app.get("/model-info", response_model=ModelInfo, response_model_exclude_none=True)
    def model_info() -> ModelInfo:
        return ModelInfo(**require_predictor().metadata)

    async def predict_upload(file: UploadFile, predictor: Predictor) -> Prediction:
        content = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Image exceeds 5 MiB")
        image = await run_in_threadpool(decode_image, content)
        try:
            result = await run_in_threadpool(predictor.predict, image)
            return Prediction(**result)
        finally:
            image.close()

    @app.post("/predict", response_model=Prediction)
    async def predict(file: Annotated[UploadFile, File()]) -> Prediction:
        try:
            return await predict_upload(file, require_predictor())
        finally:
            await file.close()

    @app.post("/predict/batch", response_model=list[BatchPrediction])
    async def predict_batch(files: Annotated[list[UploadFile], File()]) -> list[BatchPrediction]:
        try:
            predictor = require_predictor()
            if len(files) > MAX_BATCH_FILES:
                raise HTTPException(status_code=413, detail="Batch exceeds 16 images")
            if sum(file.size or 0 for file in files) > MAX_BATCH_BYTES:
                raise HTTPException(status_code=413, detail="Batch exceeds 20 MiB")
            results = []
            for index, file in enumerate(files):
                try:
                    result = await predict_upload(file, predictor)
                except HTTPException as exc:
                    raise HTTPException(
                        status_code=exc.status_code,
                        detail={"index": index, "filename": file.filename, "error": exc.detail},
                    ) from exc
                results.append(BatchPrediction(filename=file.filename, **result.model_dump()))
            return results
        finally:
            for file in files:
                await file.close()

    return app


app = create_app()
