"""Deterministic HSV flow visualization with white zero/invalid pixels."""

from __future__ import annotations

import numpy as np


def fit_flow_scale(flow: np.ndarray, valid: np.ndarray, *, percentile: float = 99.0) -> float:
    flow = np.asarray(flow, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if flow.ndim != 4 or flow.shape[-1] != 2 or valid.shape != flow.shape[:-1]:
        raise ValueError("flow and valid mask shapes are incompatible")
    if not 0 < percentile <= 100:
        raise ValueError("flow scale percentile must be in (0,100]")
    magnitudes = np.linalg.norm(flow, axis=-1)
    samples = magnitudes[valid]
    if samples.size == 0 or not np.isfinite(samples).all():
        raise ValueError("valid flow magnitudes are missing or non-finite")
    scale = float(np.percentile(samples, percentile))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("flow magnitude scale must be positive and finite")
    return scale


def encode_flow_hsv(flow: np.ndarray, valid: np.ndarray, *, magnitude_scale: float) -> np.ndarray:
    flow = np.asarray(flow, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if flow.ndim != 4 or flow.shape[-1] != 2 or valid.shape != flow.shape[:-1]:
        raise ValueError("flow and valid mask shapes are incompatible")
    if not np.isfinite(flow).all():
        raise ValueError("flow must be finite")
    if not np.isfinite(magnitude_scale) or magnitude_scale <= 0:
        raise ValueError("flow magnitude scale must be positive and finite")

    magnitude = np.linalg.norm(flow, axis=-1)
    hue = (np.arctan2(flow[..., 1], flow[..., 0]) + np.pi) / (2 * np.pi)
    saturation = np.clip(magnitude / magnitude_scale, 0, 1)
    saturation[~valid | (magnitude == 0)] = 0
    sector = (hue * 6.0) % 6.0
    integer = np.floor(sector).astype(np.int8)
    fraction = sector - integer
    low = 1.0 - saturation
    down = 1.0 - saturation * fraction
    up = 1.0 - saturation * (1.0 - fraction)
    one = np.ones_like(saturation)
    choices = (
        (one, up, low),
        (down, one, low),
        (low, one, up),
        (low, down, one),
        (up, low, one),
        (one, low, down),
    )
    output = np.empty(flow.shape[:-1] + (3,), dtype=np.float32)
    for value, channels in enumerate(choices):
        mask = integer == value
        for channel, component in enumerate(channels):
            output[..., channel][mask] = component[mask]
    output = np.rint(output * 255).astype(np.uint8)
    return output.transpose(0, 3, 1, 2)
