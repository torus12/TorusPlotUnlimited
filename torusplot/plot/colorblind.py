"""Colour-vision-deficiency (CVD) simulation for previews and palette checks.

Simulation uses the physiologically based model of Machado GM, Oliveira MM, Fernandes
LAF (2009) "A physiologically-based model for simulation of color vision deficiency",
*IEEE Trans Vis Comput Graph* 15(6):1291-1298, at severity 1.0 (dichromacy). The 3x3
matrices are applied in *linear* RGB. Achromatopsia is approximated by luminance.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from matplotlib import colors as mcolors

CVD_KINDS = ("deuteranopia", "protanopia", "tritanopia", "grayscale")

CVD_LABELS = {
    "deuteranopia": "Deuteranopia (green-blind, most common)",
    "protanopia": "Protanopia (red-blind)",
    "tritanopia": "Tritanopia (blue-blind, rare)",
    "grayscale": "Grayscale (print / total colour blindness)",
}

# Machado et al. (2009), Table 1, severity 1.0.
_MATRICES = {
    "protanopia": np.array(
        [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]]
    ),
    "deuteranopia": np.array(
        [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]]
    ),
    "tritanopia": np.array(
        [[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602], [0.004733, 0.691367, 0.303900]]
    ),
    "grayscale": np.array([[0.2126, 0.7152, 0.0722]] * 3),
}


def _to_linear(srgb: np.ndarray) -> np.ndarray:
    return np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)


def _to_srgb(linear: np.ndarray) -> np.ndarray:
    linear = np.clip(linear, 0.0, 1.0)
    return np.where(linear <= 0.0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - 0.055)


def simulate_rgb(rgb: np.ndarray, kind: str) -> np.ndarray:
    """Simulate CVD on an array of sRGB values in [0, 1] with shape (..., 3)."""
    if kind not in _MATRICES:
        raise ValueError(f"Unknown colour-vision type '{kind}'. Choose from {CVD_KINDS}.")
    rgb = np.asarray(rgb, dtype=float)
    out = _to_linear(rgb) @ _MATRICES[kind].T
    return np.clip(_to_srgb(out), 0.0, 1.0)


def simulate_colors(colors: Sequence[str], kind: str) -> list[str]:
    """Simulate how a list of colours appears under a colour-vision deficiency."""
    rgb = np.array([mcolors.to_rgb(c) for c in colors], dtype=float).reshape(-1, 3)
    return [mcolors.to_hex(c).upper() for c in simulate_rgb(rgb, kind)]


def simulate_image(image: np.ndarray, kind: str) -> np.ndarray:
    """Simulate CVD on an image array (H, W, 3 or 4), uint8 or float; returns uint8."""
    arr = np.asarray(image)
    is_int = arr.dtype.kind in "ui"
    data = arr.astype(float) / 255.0 if is_int else arr.astype(float)
    rgb = simulate_rgb(data[..., :3], kind)
    out = np.concatenate([rgb, data[..., 3:]], axis=-1) if data.shape[-1] == 4 else rgb
    return np.round(out * 255.0).astype(np.uint8)


def _lab(colors: Sequence[str]) -> np.ndarray:
    from skimage.color import rgb2lab

    rgb = np.array([mcolors.to_rgb(c) for c in colors], dtype=float).reshape(1, -1, 3)
    return rgb2lab(rgb)[0]


def min_color_distance(colors: Sequence[str], kind: str | None = None) -> float:
    """Smallest pairwise CIE76 Delta-E between colours (optionally after CVD simulation).

    Rule of thumb: below ~10 two colours are hard to tell apart in a figure.
    """
    cols = list(colors)
    if len(cols) < 2:
        return float("inf")
    if kind:
        cols = simulate_colors(cols, kind)
    lab = _lab(cols)
    diff = lab[:, None, :] - lab[None, :, :]
    dist = np.sqrt((diff**2).sum(axis=-1))
    dist[np.diag_indices_from(dist)] = np.inf
    return float(dist.min())


def palette_report(colors: Sequence[str], threshold: float = 10.0) -> dict[str, dict[str, float | bool]]:
    """Per vision type: minimum pairwise distance and whether the palette stays distinguishable."""
    report: dict[str, dict[str, float | bool]] = {}
    for kind in ("normal", *CVD_KINDS):
        d = min_color_distance(colors, None if kind == "normal" else kind)
        report[kind] = {"min_delta_e": d, "ok": bool(d >= threshold)}
    return report
