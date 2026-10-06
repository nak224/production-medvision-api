"""Build with this cloud machine's existing proxy and verified CA trust bundle."""

import os
import socket
import subprocess
from pathlib import Path
from urllib.parse import urlsplit


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    command = [
        "docker",
        "--config",
        "/workspace/.cache/medvision-docker",
        "build",
        "--network=host",
    ]
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY"):
        if name in os.environ:
            # Docker's predefined proxy arguments are excluded from image history.
            command.extend(("--build-arg", name))
    hosts = {
        urlsplit(os.environ[name]).hostname
        for name in ("HTTP_PROXY", "HTTPS_PROXY")
        if name in os.environ
    }
    for host in sorted(host for host in hosts if host):
        address = socket.getaddrinfo(host, None, family=socket.AF_INET)[0][4][0]
        command.extend(("--add-host", f"{host}:{address}"))
    if "SSL_CERT_FILE" in os.environ:
        # BuildKit mounts trust material temporarily; it is not copied into the image.
        command.extend(("--secret", "id=ca_bundle,src=" + os.environ["SSL_CERT_FILE"]))
    command.extend(("-t", "medvision-api:dev", "."))
    subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
