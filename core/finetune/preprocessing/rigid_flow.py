"""Rigid optical flow derived from metric depth and OpenCV w2c cameras."""

from __future__ import annotations

import numpy as np


def compute_rigid_flow(
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics_w2c: np.ndarray,
    *,
    relative_depth_tolerance: float = 0.03,
    occlusion_check: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    depth_m = np.asarray(depth_m, dtype=np.float32)
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    extrinsics_w2c = np.asarray(extrinsics_w2c, dtype=np.float32)
    if depth_m.ndim != 3:
        raise ValueError("metric depth must have shape (T,H,W)")
    frames, height, width = depth_m.shape
    if intrinsics.shape != (frames, 3, 3) or extrinsics_w2c.shape != (frames, 3, 4):
        raise ValueError("camera arrays do not match the depth sequence")
    if not np.isfinite(depth_m).all() or not np.isfinite(intrinsics).all() or not np.isfinite(extrinsics_w2c).all():
        raise ValueError("depth and camera arrays must be finite")
    if relative_depth_tolerance <= 0:
        raise ValueError("relative depth tolerance must be positive")

    flow = np.zeros((frames, height, width, 2), dtype=np.float32)
    valid = np.zeros((frames, height, width), dtype=bool)
    pixel_x, pixel_y = np.meshgrid(
        np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
    )
    homogeneous_pixels = np.stack((pixel_x, pixel_y, np.ones_like(pixel_x)), axis=0).reshape(3, -1)

    w2c_h = np.repeat(np.eye(4, dtype=np.float32)[None], frames, axis=0)
    w2c_h[:, :3] = extrinsics_w2c
    try:
        c2w_h = np.linalg.inv(w2c_h)
        inverse_intrinsics = np.linalg.inv(intrinsics)
    except np.linalg.LinAlgError:
        raise ValueError("camera matrix is not invertible") from None

    for target_index in range(1, frames):
        source_index = target_index - 1
        source_depth = depth_m[source_index].reshape(-1)
        source_valid = source_depth > 0
        camera_points = inverse_intrinsics[source_index] @ homogeneous_pixels
        camera_points *= source_depth[None]
        camera_h = np.concatenate((camera_points, np.ones((1, camera_points.shape[1]), dtype=np.float32)), axis=0)
        world_h = c2w_h[source_index] @ camera_h
        target_camera = w2c_h[target_index] @ world_h
        target_z = target_camera[2]
        projected = intrinsics[target_index] @ target_camera[:3]
        target_x = projected[0] / np.maximum(target_z, np.finfo(np.float32).eps)
        target_y = projected[1] / np.maximum(target_z, np.finfo(np.float32).eps)
        target_valid = (
            source_valid
            & (target_z > 0)
            & (target_x >= 0)
            & (target_x <= width - 1)
            & (target_y >= 0)
            & (target_y <= height - 1)
        )
        if occlusion_check:
            nearest_x = np.rint(target_x).astype(np.int64).clip(0, width - 1)
            nearest_y = np.rint(target_y).astype(np.int64).clip(0, height - 1)
            observed = depth_m[target_index, nearest_y, nearest_x]
            target_valid &= observed > 0
            relative_error = np.abs(observed - target_z) / np.maximum(observed, target_z)
            target_valid &= relative_error <= relative_depth_tolerance

        flat_flow = flow[target_index].reshape(-1, 2)
        flat_flow[:, 0] = target_x - homogeneous_pixels[0]
        flat_flow[:, 1] = target_y - homogeneous_pixels[1]
        flat_valid = valid[target_index].reshape(-1)
        flat_valid[:] = target_valid
        flat_flow[~target_valid] = 0
    return flow, valid
