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
