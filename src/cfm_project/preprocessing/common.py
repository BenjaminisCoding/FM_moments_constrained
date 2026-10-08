"""Checked source downloads and prepared-data manifests."""

import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sources(benchmark, raw, download=False):
    specs = json.loads((ROOT / "configs/preprocessing/sources.json").read_text())[
        benchmark
    ]
    raw.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        path = raw / spec["filename"]
        if not path.exists():
            if not download:
                raise FileNotFoundError(
                    f"{path}: download {spec['url']} or pass --download"
                )
            temporary = path.with_suffix(path.suffix + ".part")
            try:
                request = urllib.request.Request(
                    spec["url"],
                    headers={"User-Agent": "Mozilla/5.0 (compatible; cfm-project/0.1)"},
                )
                with (
                    urllib.request.urlopen(request, timeout=120) as response,
                    temporary.open("wb") as output,
                ):
                    for block in iter(lambda: response.read(1024 * 1024), b""):
                        output.write(block)
                if digest(temporary) != spec["sha256"]:
                    raise ValueError(f"Download checksum mismatch: {path.name}")
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        if digest(path) != spec["sha256"]:
            raise ValueError(f"Source checksum mismatch: {path}")
    return specs


def register(root, folder):
    path = root / "manifest.json"
    manifest = (
        json.loads(path.read_text()) if path.exists() else {"schema": 1, "files": {}}
    )
    for item in sorted(folder.rglob("*")):
        if item.is_file():
            manifest["files"][item.relative_to(root).as_posix()] = {
                "sha256": digest(item)
            }
    write_json(path, manifest)
