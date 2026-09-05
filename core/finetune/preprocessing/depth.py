"""Versioned metric-depth visualization used by the Wan VAE input."""

from __future__ import annotations

import numpy as np


def encode_metric_depth(
    depth_mm: np.ndarray, *, max_depth_mm: int = 1300
) -> tuple[np.ndarray, np.ndarray]:
    depth_mm = np.asarray(depth_mm)
    if depth_mm.dtype != np.uint16 or depth_mm.ndim != 3:
        raise ValueError("metric depth must be uint16 with shape (T,H,W)")
    if max_depth_mm <= 1:
        raise ValueError("maximum metric depth must exceed one millimetre")
    valid = depth_mm != 0
    clipped = np.clip(depth_mm.astype(np.float32), 1, max_depth_mm)
    codes = np.rint(1.0 + (clipped - 1.0) * 254.0 / (max_depth_mm - 1.0)).astype(np.uint8)
    codes[~valid] = 0
    return np.repeat(codes[:, None], 3, axis=1), valid


def decode_metric_depth(
    encoded: np.ndarray, valid: np.ndarray, *, max_depth_mm: int = 1300
) -> np.ndarray:
    encoded = np.asarray(encoded)
    valid = np.asarray(valid, dtype=bool)
    if encoded.ndim != 4 or encoded.shape[1] != 3 or encoded.dtype != np.uint8:
        raise ValueError("encoded depth must be uint8 with shape (T,3,H,W)")
    if valid.shape != (encoded.shape[0], encoded.shape[2], encoded.shape[3]):
        raise ValueError("depth valid mask shape does not match")
    decoded = 1.0 + (encoded[:, 0].astype(np.float32) - 1.0) * (max_depth_mm - 1.0) / 254.0
    decoded[~valid] = 0
    return decoded
