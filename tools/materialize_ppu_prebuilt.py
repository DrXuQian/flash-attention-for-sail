#!/usr/bin/env python3
"""Verify and materialize the checked-in PPU FlashAttention extension."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
PREBUILT_ROOT = REPO_ROOT / "prebuilt" / "ppu_10-ppu_15"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_key() -> str:
    public_torch = str(torch.__version__).split("+", 1)[0]
    torch_major_minor = ".".join(public_torch.split(".")[:2])
    cache_tag = sys.implementation.cache_tag.replace("-", "")
    abi = int(torch.compiled_with_cxx11_abi())
    return f"{cache_tag}-torch{torch_major_minor}-cxx11abi{abi}"


def check_field(label: str, actual, expected) -> None:
    if actual != expected:
        raise RuntimeError(f"{label}: got {actual!r}, expected {expected!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--destination",
        type=Path,
        default=REPO_ROOT,
        help="directory in which to materialize the importable .so",
    )
    args = parser.parse_args()

    key = runtime_key()
    prebuilt_dir = PREBUILT_ROOT / key
    manifest_path = prebuilt_dir / "manifest.json"
    if not manifest_path.is_file():
        available = sorted(path.parent.name for path in PREBUILT_ROOT.glob("*/manifest.json"))
        raise RuntimeError(f"no prebuilt for {key}; available manifests={available}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    build = manifest["build"]
    check_field("python_cache_tag", sys.implementation.cache_tag, build["python_cache_tag"])
    check_field(
        "torch_public_version",
        str(torch.__version__).split("+", 1)[0],
        build["torch_public_version"],
    )
    check_field("cxx11_abi", bool(torch.compiled_with_cxx11_abi()), build["cxx11_abi"])

    archive = prebuilt_dir / manifest["archive"]
    check_field("archive_size", archive.stat().st_size, manifest["archive_size"])
    check_field("archive_sha256", sha256(archive), manifest["archive_sha256"])

    zstd = shutil.which("zstd")
    if zstd is None:
        raise RuntimeError("zstd is required to materialize the checked-in PPU binary")
    destination_dir = args.destination.resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)
    artifact = destination_dir / manifest["artifact"]
    partial = artifact.with_name(artifact.name + ".partial")
    subprocess.run([zstd, "-d", "-f", str(archive), "-o", str(partial)], check=True)
    check_field("artifact_size", partial.stat().st_size, manifest["artifact_size"])
    check_field("artifact_sha256", sha256(partial), manifest["artifact_sha256"])
    os.chmod(partial, 0o755)
    os.replace(partial, artifact)
    print(f"[PPU FlashAttention prebuilt] PASS: runtime={key} artifact={artifact}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"[PPU FlashAttention prebuilt] FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
