from pathlib import Path

import torch
from PIL import Image

from medvision.data import CLASS_NAMES, preprocess
from medvision.model import load_checkpoint


class Predictor:
    def __init__(self, checkpoint: Path):
        self.model, self.metadata = load_checkpoint(checkpoint)

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> dict:
        probabilities = self.model(preprocess(image).unsqueeze(0)).softmax(dim=1)[0]
        if not torch.isfinite(probabilities).all():
            raise ValueError("Model returned non-finite probabilities")
        class_id = int(probabilities.argmax())
        return {
            "class_id": class_id,
            "label": CLASS_NAMES[class_id],
            "confidence": float(probabilities[class_id]),
            "probabilities": dict(zip(CLASS_NAMES, probabilities.tolist(), strict=True)),
            "model_version": self.metadata["model_version"],
        }
