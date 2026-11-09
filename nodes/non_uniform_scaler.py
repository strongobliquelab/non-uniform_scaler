"""Seamless non-uniform scale along one axis.

Within the affected span the local scale is a tent in source-pixel space:
1 at each boundary, and the scale factor at the middle. Content outside that
span is copied unchanged. Targeted-dimension mode picks the peak scale so the
continuous size lands on the requested length, then crops any leftover pixels
(or, if the result is still short, repeats the second end).
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

_SF_MAX = 64.0


def _iround(x: float) -> int:
    return int(math.floor(float(x) + 0.5))


def mask_bounds(mask: np.ndarray, direction: str, height: int, width: int):
    """Inclusive span of mask pixels greater than zero, or None if none are lit.

    Vertical uses the topmost and bottommost active rows. Horizontal uses the
    leftmost and rightmost active columns. A batch is the union of every mask.
    """
    m = np.asarray(mask)
    if m.ndim == 2:
        m = m[None, ...]
    if m.ndim != 3:
        return None
    if m.shape[1] != height or m.shape[2] != width:
        return None
    active = m > 0
    if direction == "vertical":
        idx = np.flatnonzero(active.any(axis=(0, 2)))
    else:
        idx = np.flatnonzero(active.any(axis=(0, 1)))
    if idx.size == 0:
        return None
    return int(idx[0]), int(idx[-1])


def clamp_span(start: int, end: int, length: int) -> Tuple[int, int]:
    """Inclusive span clamped into [0, length). Empty image -> (0, -1)."""
    if length <= 0:
        return 0, -1
    a = int(start)
    b = int(end)
    a = 0 if a < 0 else length - 1 if a >= length else a
    b = 0 if b < 0 else length - 1 if b >= length else b
    if a > b:
        a, b = b, a
    return a, b


def _continuous_length(width: int, scale_factor: float) -> float:
    return width * (1.0 + scale_factor) / 2.0


def output_span_length(width: int, scale_factor: float) -> int:
    if width <= 0:
        return 0
    sf = 0.0 if scale_factor < 0.0 else float(scale_factor)
    return max(1, _iround(_continuous_length(width, sf)))


def _inverse_left(y: float, width: int, scale_factor: float) -> float:
    """Inverse of C(x) = x + (sf - 1) * x^2 / width on the left half."""
    if width <= 0:
        return 0.0
    sf = float(scale_factor)
    if abs(sf - 1.0) < 1e-8:
        return y
    disc = 1.0 + 4.0 * (sf - 1.0) * y / width
    if disc < 0.0:
        disc = 0.0
    return (-1.0 + math.sqrt(disc)) * width / (2.0 * (sf - 1.0))


def source_coords(width: int, scale_factor: float, out_len: int) -> np.ndarray:
    """Map each output pixel center to a continuous source edge coordinate.

    Coordinate 0 is the leading edge of the affected span; ``width`` is the
    trailing edge. Pixel ``i`` of the source span is centered at ``i + 0.5``.
    """
    if out_len <= 0 or width <= 0:
        return np.zeros((0,), dtype=np.float64)
    sf = 0.0 if scale_factor < 0.0 else float(scale_factor)
    total = _continuous_length(width, sf)
    if total <= 0.0:
        return np.full((out_len,), width * 0.5, dtype=np.float64)
    half = total * 0.5
    js = (np.arange(out_len, dtype=np.float64) + 0.5) * (total / out_len)
    xs = np.empty(out_len, dtype=np.float64)
    for i, y in enumerate(js):
        if y <= half:
            x = _inverse_left(float(y), width, sf)
        else:
            x = width - _inverse_left(float(total - y), width, sf)
        if x < 0.0:
            x = 0.0
        elif x > width:
            x = float(width)
        xs[i] = x
    return xs


def resolve_scale_factor(
    length: int, span: int, scale_factor: float, mode: str, target: int
) -> float:
    if mode == "factor":
        return 0.0 if scale_factor < 0.0 else float(scale_factor)
    if span <= 0:
        return 1.0
    # The tent integrates to span * (1 + sf) / 2, so the size that lands on
    # `target` is known in closed form. Rounding may overshoot by a pixel;
    # the caller crops that. If it undershoots, raise sf.
    sf = 1.0 + 2.0 * (target - length) / span
    if sf < 0.0:
        sf = 0.0
    if sf > _SF_MAX:
        sf = _SF_MAX

    def size_of(value: float) -> int:
        return (length - span) + output_span_length(span, value)

    if size_of(sf) >= target or sf >= _SF_MAX:
        return sf
    lo = sf
    hi = _SF_MAX
    for _ in range(48):
        mid = (lo + hi) * 0.5
        if size_of(mid) >= target:
            hi = mid
        else:
            lo = mid
    return hi


def _sample_axis(image: np.ndarray, axis: int, src: np.ndarray) -> np.ndarray:
    """Bilinear sample along `axis` (1 = vertical, 2 = horizontal)."""
    length = image.shape[axis]
    if src.size == 0:
        shape = list(image.shape)
        shape[axis] = 0
        return np.zeros(shape, dtype=image.dtype)
    y0 = np.floor(src).astype(np.int64)
    t = (src - y0).astype(np.float64)
    y0 = np.clip(y0, 0, length - 1)
    y1 = np.clip(y0 + 1, 0, length - 1)
    if axis == 1:
        a = image[:, y0]
        b = image[:, y1]
        w = t.reshape(1, -1, 1, 1)
    else:
        a = image[:, :, y0]
        b = image[:, :, y1]
        w = t.reshape(1, 1, -1, 1)
    return (a * (1.0 - w) + b * w).astype(image.dtype, copy=False)


def _fit_length(image: np.ndarray, axis: int, target: int, crop_mode: str) -> np.ndarray:
    current = image.shape[axis]
    if target < 1:
        target = 1
    excess = current - target
    if excess > 0:
        if crop_mode == "first":
            c1, c2 = excess, 0
        elif crop_mode == "second":
            c1, c2 = 0, excess
        else:
            c1 = excess // 2
            c2 = excess - c1
        if axis == 1:
            return image[:, c1 : current - c2]
        return image[:, :, c1 : current - c2]
    if excess < 0:
        need = -excess
        if axis == 1:
            pad = np.repeat(image[:, -1:], need, axis=1)
            return np.concatenate([image, pad], axis=1)
        pad = np.repeat(image[:, :, -1:], need, axis=2)
        return np.concatenate([image, pad], axis=2)
    return image


def scale_image(
    image: np.ndarray,
    scaling_mode: str,
    direction: str,
    crop_mode: str,
    scale_factor: float,
    targeted_dimension: int,
    affected_start: int,
    affected_end: int,
    empty_span: bool = False,
) -> np.ndarray:
    """Scale a ComfyUI-style batch image of shape [B, H, W, C]."""
    if image.ndim != 4:
        raise ValueError("image must have shape [B, H, W, C]")
    axis = 1 if direction == "vertical" else 2
    length = int(image.shape[axis])
    if empty_span:
        start, end, span = 0, -1, 0
    else:
        start, end = clamp_span(affected_start, affected_end, length)
        span = end - start + 1 if end >= start else 0
    sf = resolve_scale_factor(
        length, span, float(scale_factor), scaling_mode, int(targeted_dimension)
    )
    if span <= 0 or abs(sf - 1.0) < 1e-8:
        warped = image
    else:
        out_len = output_span_length(span, sf)
        coords = source_coords(span, sf, out_len) + start - 0.5
        mid = _sample_axis(image, axis, coords)
        if axis == 1:
            parts = []
            if start > 0:
                parts.append(image[:, :start])
            parts.append(mid)
            if end + 1 < length:
                parts.append(image[:, end + 1 :])
            warped = np.concatenate(parts, axis=1) if parts else mid
        else:
            parts = []
            if start > 0:
                parts.append(image[:, :, :start])
            parts.append(mid)
            if end + 1 < length:
                parts.append(image[:, :, end + 1 :])
            warped = np.concatenate(parts, axis=2) if parts else mid
    if scaling_mode == "targeted_dimension":
        warped = _fit_length(warped, axis, int(targeted_dimension), crop_mode)
    return np.ascontiguousarray(warped)

def scale_latent(
    latent: np.ndarray,
    scaling_mode: str,
    direction: str,
    crop_mode: str,
    scale_factor: float,
    targeted_dimension: int,
    affected_start: int,
    affected_end: int,
    empty_span: bool = False,
    downsampling_factor: int = 8,
) -> np.ndarray:
    """Scale a ComfyUI-style batch latent of shape [B, C, H, W].
    targeted_dimension / affected_start / affected_end are interpreted in
    *pixel* space (the same numbers you would use on the image node).
    They are converted to latent units using downsampling_factor.
    """
    if latent.ndim != 4:
        raise ValueError("latent must have shape [B, C, H, W]")
    comp = max(1, int(downsampling_factor))
    # Convert user-facing pixel numbers → latent units
    target_lat = max(1, (int(targeted_dimension) + comp - 1) // comp)   # ceil
    start_lat  = int(affected_start) // comp
    end_lat    = int(affected_end)   // comp
    axis = 2 if direction == "vertical" else 3
    length = int(latent.shape[axis])
    if empty_span:
        start, end, span = 0, -1, 0
    else:
        start, end = clamp_span(start_lat, end_lat, length)
        span = end - start + 1 if end >= start else 0
    sf = resolve_scale_factor(
        length, span, float(scale_factor), scaling_mode, target_lat
    )
    if span <= 0 or abs(sf - 1.0) < 1e-8:
        warped = latent
    else:
        out_len = output_span_length(span, sf)
        coords = source_coords(span, sf, out_len) + start - 0.5
        mid = _sample_axis_latent(latent, axis, coords)
        if axis == 2:
            parts = []
            if start > 0:
                parts.append(latent[:, :, :start])
            parts.append(mid)
            if end + 1 < length:
                parts.append(latent[:, :, end + 1 :])
            warped = np.concatenate(parts, axis=2) if parts else mid
        else:
            parts = []
            if start > 0:
                parts.append(latent[:, :, :, :start])
            parts.append(mid)
            if end + 1 < length:
                parts.append(latent[:, :, :, end + 1 :])
            warped = np.concatenate(parts, axis=3) if parts else mid
    if scaling_mode == "targeted_dimension":
        warped = _fit_length_latent(warped, axis, target_lat, crop_mode)
    return np.ascontiguousarray(warped)

def _sample_axis_latent(latent: np.ndarray, axis: int, src: np.ndarray) -> np.ndarray:
    """Bilinear sample along `axis` (2 = vertical/H, 3 = horizontal/W) for [B, C, H, W]."""
    length = latent.shape[axis]
    if src.size == 0:
        shape = list(latent.shape)
        shape[axis] = 0
        return np.zeros(shape, dtype=latent.dtype)
    y0 = np.floor(src).astype(np.int64)
    t = (src - y0).astype(np.float64)
    y0 = np.clip(y0, 0, length - 1)
    y1 = np.clip(y0 + 1, 0, length - 1)
    if axis == 2:
        a = latent[:, :, y0]
        b = latent[:, :, y1]
        w = t.reshape(1, 1, -1, 1)
    else:
        a = latent[:, :, :, y0]
        b = latent[:, :, :, y1]
        w = t.reshape(1, 1, 1, -1)
    return (a * (1.0 - w) + b * w).astype(latent.dtype, copy=False)


def _fit_length_latent(latent: np.ndarray, axis: int, target: int, crop_mode: str) -> np.ndarray:
    current = latent.shape[axis]
    if target < 1:
        target = 1
    excess = current - target
    if excess > 0:
        if crop_mode == "first":
            c1, c2 = excess, 0
        elif crop_mode == "second":
            c1, c2 = 0, excess
        else:
            c1 = excess // 2
            c2 = excess - c1
        if axis == 2:
            return latent[:, :, c1 : current - c2]
        return latent[:, :, :, c1 : current - c2]
    if excess < 0:
        need = -excess
        if axis == 2:
            pad = np.repeat(latent[:, :, -1:], need, axis=2)
            return np.concatenate([latent, pad], axis=2)
        pad = np.repeat(latent[:, :, :, -1:], need, axis=3)
        return np.concatenate([latent, pad], axis=3)
    return latent