from pydantic import BaseModel, Field


class Prediction(BaseModel):
    class_id: int = Field(ge=0, le=8)
    label: str
    confidence: float = Field(ge=0, le=1)
    probabilities: dict[str, float]
    model_version: str


class Health(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None


class ModelInfo(BaseModel):
    architecture: str
    model_version: str
    dataset: str
    class_names: list[str]
    preprocessing: str
    input_size: int | list[int] | None = None
    expected_input_format: str | None = None


class BatchPrediction(Prediction):
    filename: str | None
