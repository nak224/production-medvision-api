"""Resolve an optional S3 artifact without changing the checkpoint format."""

from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit

import boto3


@contextmanager
def checkpoint_path(local_path: Path, s3_uri: str | None = None) -> Iterator[Path]:
    if s3_uri is None:
        yield local_path
        return

    uri = urlsplit(s3_uri)
    if (
        uri.scheme != "s3"
        or not uri.netloc
        or not uri.path.lstrip("/")
        or uri.username is not None
        or ":" in uri.netloc
        or uri.query
        or uri.fragment
        or any(character.isspace() for character in uri.netloc)
    ):
        raise ValueError("MEDVISION_MODEL_S3_URI must be s3://bucket/path/model.pt")

    # Boto3 uses the standard credential chain, including the EC2 instance role.
    # The temporary directory is removed even if download or checkpoint loading fails.
    with TemporaryDirectory(prefix="medvision-model-") as directory:
        path = Path(directory) / "model.pt"
        with closing(boto3.client("s3")) as client:
            client.download_file(uri.netloc, uri.path[1:], str(path))
        yield path
