"""Atomic writers and text artefacts.

Ref: Sec. 4.4 (artefacts are written once per run and reused by every read-out).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch


def atomic_write_bytes(path: str | Path, payload: bytes) -> Path:
    """Write bytes through a temporary file and replace the target atomically."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=f".{target.name}.")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
        tmp_path.chmod(0o644)
        tmp_path.replace(target)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return target


def atomic_write_text(path: str | Path, text: str) -> Path:
    return atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    blob = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    return atomic_write_text(path, blob)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_payload_digest(state: dict[str, Any]) -> str:
    """Digest tensor contents rather than container bytes.

    ``torch.save`` embeds container metadata, so hashing the file itself would
    change on every write even when the payload is identical.
    """
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        digest.update(key.encode("utf-8"))
        if isinstance(value, torch.Tensor):
            tensor = value.detach().to("cpu").contiguous()
            digest.update(str(tuple(tensor.shape)).encode("utf-8"))
            digest.update(str(tensor.dtype).encode("utf-8"))
            digest.update(tensor.numpy().tobytes())
        else:
            digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()
