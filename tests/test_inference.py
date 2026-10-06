import pytest
import torch
from PIL import Image

from medvision.data import CLASS_NAMES
from medvision.inference import Predictor
from medvision.model import load_checkpoint


def test_checkpoint_inference_is_repeatable(checkpoint):
    predictor = Predictor(checkpoint)
    image = Image.new("RGB", (28, 28), color=(100, 150, 200))
    result = predictor.predict(image)
    assert predictor.predict(image) == result
    assert result["model_version"] == "test-fixture"
    assert result["label"] == CLASS_NAMES[result["class_id"]]
    assert result["confidence"] == pytest.approx(max(result["probabilities"].values()))
    assert sum(result["probabilities"].values()) == pytest.approx(1.0)
    assert set(result["probabilities"]) == set(CLASS_NAMES)


@pytest.mark.parametrize("key,value", [("class_names", []), ("preprocessing", "unknown")])
def test_incompatible_checkpoint_is_rejected(checkpoint, tmp_path, key, value):
    payload = torch.load(checkpoint, weights_only=True)
    payload[key] = value
    invalid = tmp_path / "invalid.pt"
    torch.save(payload, invalid)
    with pytest.raises(ValueError, match="metadata"):
        load_checkpoint(invalid)
