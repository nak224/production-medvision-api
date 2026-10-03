import argparse
import hashlib
import os
import tempfile
from pathlib import Path
from urllib.request import HTTPRedirectHandler, build_opener

import torch
from medmnist import INFO, PathMNIST
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import transforms

DATASET = "pathmnist"
IMAGE_SIZE = 28
CLASS_NAMES = tuple(INFO[DATASET]["label"][str(i)] for i in range(9))
PREPROCESSING = "rgb-resize28-bilinear-normalize0.5-v1"


class HTTPSOnlyRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not newurl.startswith("https://"):
            raise ValueError("Dataset redirect must use HTTPS")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def image_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE), interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )


def preprocess(image: Image.Image) -> torch.Tensor:
    return image_transform()(image.convert("RGB"))


def load_dataset(root: Path, split: str, limit: int | None = None, seed: int = 42) -> Dataset:
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split: {split}")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if not (root / "pathmnist.npz").is_file():
        raise FileNotFoundError("PathMNIST is missing; run `medvision-download --data-dir data`.")
    dataset = PathMNIST(root=str(root), split=split, download=False, transform=preprocess)
    if limit is not None and limit < len(dataset):
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(dataset), generator=generator)[:limit].tolist()
        return Subset(dataset, indices)
    return dataset


def _checksum(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, lambda: hashlib.md5(usedforsecurity=False)).hexdigest()


def download_data(root: Path) -> Path:
    """Download only from the official HTTPS URL and verify the published checksum."""
    root.mkdir(parents=True, exist_ok=True)
    destination = root / "pathmnist.npz"
    expected = INFO[DATASET]["MD5"]
    if destination.exists():
        if _checksum(destination) != expected:
            raise ValueError(
                f"Checksum mismatch for {destination}; inspect/remove the invalid file."
            )
        return destination

    url = INFO[DATASET]["url"]
    if not url.startswith("https://"):
        raise ValueError("Dataset download requires HTTPS")
    temporary = None
    try:
        with build_opener(HTTPSOnlyRedirect()).open(url, timeout=120) as response:
            if not response.url.startswith("https://"):
                raise ValueError("Dataset redirect must use HTTPS")
            with tempfile.NamedTemporaryFile(dir=root, suffix=".part", delete=False) as stream:
                temporary = Path(stream.name)
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
        if _checksum(temporary) != expected:
            raise ValueError("Downloaded dataset failed the official MD5 integrity check")
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and verify official 28px PathMNIST data")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    print(download_data(args.data_dir))


if __name__ == "__main__":
    main()
