"""Shared array fingerprints and stable single-cell time-label ordering."""
import hashlib
from typing import Any

import numpy as np


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    hasher = hashlib.sha256()
    hasher.update(str(contiguous.dtype).encode("utf-8"))
    hasher.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    hasher.update(contiguous.tobytes())
    return hasher.hexdigest()


def sort_unique_labels(labels: np.ndarray) -> list[Any]:
    label_list = labels.tolist()
    unique = list(dict.fromkeys(label_list))
    numeric_values: list[tuple[float, Any]] = []
    numeric_ok = True
    for label in unique:
        try:
            numeric_values.append((float(label), label))
        except (TypeError, ValueError):
            numeric_ok = False
            break
    if numeric_ok:
        numeric_values.sort(key=lambda item: item[0])
        return [item[1] for item in numeric_values]
    return sorted(unique, key=lambda value: str(value))
