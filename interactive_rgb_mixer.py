"""
Interactive RGB / Perceptual LCh Mixer for SDO/AIA JPEG2000 (.jp2) images.

Provides a local Plotly Dash web application with:
  - Auto-Balance Presets: 7 data-driven, color-constancy and color-harmony rules
    computed from the loaded .jp2 channel statistics in perceptual Oklch space.
  - Option 1 (Global LCh Controls): 3 intuitive sliders (Global Hue Shift,
    Global Chroma/Saturation, Global Luminance/Lightness) whose tracks ARE
    live Lightroom-style gradient colorbars and whose draggable thumbs display
    the active color, moving the Option 2 per-waveband sliders in sync.
  - Option 2 (Per-Waveband LCh Controls): Individual Hue, Chroma, and Luminance
    sliders for each waveband (304 A, 171 A, 193 A), each riding directly on
    its own live gradient colorbar, corresponding to the 3 columns of `self.rgbmix`.
  - Real-time side-by-side or single-frame preview with ROI zoom presets and
    live export of the resulting 3x3 `self.rgbmix` matrix for `visualization.py`.

Usage:
    .venv/bin/python interactive_rgb_mixer.py
"""

import argparse
import base64
import copy
import glob
import os
import time
import cv2
import numpy as np
from dash import Dash, Input, Output, State, ctx, dcc, html, no_update

import visualization

# -----------------------------------------------------------------------------
# Default JP2 3x3 RGB mixing matrix from visualization.RGBMixer.set_jp2_default
# Columns correspond to input channels [304 (R), 171 (G), 193 (B)]
# Rows correspond to output channels [R, G, B]
# -----------------------------------------------------------------------------
DEFAULT_JP2_RGBMIX = np.array(
    [
        [1.045, 0.635, -0.295],
        [0.000, 0.955, 0.095],
        [0.000, 0.195, 0.565],
    ],
    dtype=np.float64,
)

# Slider bounds for Global (Option 1) controls
GLOBAL_H_MIN, GLOBAL_H_MAX = -180.0, 180.0
GLOBAL_C_MIN, GLOBAL_C_MAX = 0.0, 2.2
GLOBAL_L_MIN, GLOBAL_L_MAX = 0.2, 2.2

# Slider bounds for Per-Waveband (Option 2) controls
WB_H_MIN, WB_H_MAX = 0.0, 360.0
WB_C_MIN, WB_C_MAX = 0.0, 0.9
WB_L_MIN, WB_L_MAX = -0.3, 1.4

# -----------------------------------------------------------------------------
# Linear Perceptual Oklab / Oklch Opponent Transform
# Maps RGB basis vectors <-> Perceptual (L, a, b) opponent vectors where:
#   - Neutral gray [1, 1, 1]^T maps to [L=1, a=0, b=0]^T
#   - Polar coordinates (L, C=hypot(a,b), H=degrees(atan2(b,a))) give
#     perceptually uniform Lightness (L), Chroma/Saturation (C), and Hue (H).
#   - Because T is an invertible 3x3 matrix, any LCh change in Option 1 (Global)
#     or Option 2 (Per-Waveband) maps back to an exact 3x3 `self.rgbmix` matrix.
# -----------------------------------------------------------------------------
_OKLAB_M1 = np.array(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ],
    dtype=np.float64,
)

_OKLAB_M2 = np.array(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ],
    dtype=np.float64,
)

T_RGB2LAB = _OKLAB_M2 @ _OKLAB_M1
T_LAB2RGB = np.linalg.inv(T_RGB2LAB)


def rgbmix_to_waveband_lch(rgbmix: np.ndarray) -> list[dict[str, float]]:
    """Decompose the 3 columns of a 3x3 rgbmix matrix into (L, C, H) per waveband."""
    lab_cols = T_RGB2LAB @ rgbmix
    params = []
    for i in range(3):
        L, a, b = lab_cols[:, i]
        C = float(np.hypot(a, b))
        H = float(np.degrees(np.atan2(b, a)) % 360.0)
        params.append({"L": float(L), "C": C, "H": H})
    return params


def lch_to_rgbmix(
    wb_lch: list[dict[str, float]],
    global_hue_shift: float = 0.0,
    global_chroma_scale: float = 1.0,
    global_lum_scale: float = 1.0,
) -> np.ndarray:
    """
    Construct the 3x3 `self.rgbmix` matrix from Per-Waveband LCh settings
    optionally composed with Global LCh adjustments.
    """
    lab_cols = np.zeros((3, 3), dtype=np.float64)
    for i in range(3):
        L = wb_lch[i]["L"] * global_lum_scale
        C = wb_lch[i]["C"] * global_chroma_scale
        H_rad = np.radians(wb_lch[i]["H"] + global_hue_shift)
        lab_cols[0, i] = L
        lab_cols[1, i] = C * np.cos(H_rad)
        lab_cols[2, i] = C * np.sin(H_rad)
    return T_LAB2RGB @ lab_cols


def apply_global_to_anchor(
    anchor_wb: list[dict[str, float]],
    gh: float,
    gc: float,
    gl: float,
) -> list[dict[str, float]]:
    """Compute effective per-waveband (H, C, L) slider positions from anchor + global settings."""
    out = []
    for i in range(3):
        h_eff = round(float((anchor_wb[i]["H"] + gh) % 360.0), 1)
        c_eff = round(float(np.clip(anchor_wb[i]["C"] * gc, WB_C_MIN, WB_C_MAX)), 3)
        l_eff = round(float(np.clip(anchor_wb[i]["L"] * gl, WB_L_MIN, WB_L_MAX)), 3)
        out.append({"H": h_eff, "C": c_eff, "L": l_eff})
    return out


# -----------------------------------------------------------------------------
# Unified Perceptual Colorbar & Thumb Color Engine
# Ensures exact 1-to-1 visual correspondence across:
#   1. The Hue slider track & thumb
#   2. The Chroma slider track (Gray -> exact Hue thumb color) & thumb
#   3. The Luminance slider track (Dark -> exact Chroma thumb color -> Bright) & thumb
# -----------------------------------------------------------------------------
def vivid_rgb_at_hue(H_deg: float) -> np.ndarray:
    """
    Return the purest, maximum-chroma in-gamut RGB color [R, G, B] in [0, 1]^3
    for a perceptual Oklch Hue angle H_deg (with min(RGB)=0 and max(RGB)=1).
    """
    H_rad = np.radians(H_deg)
    u = T_LAB2RGB @ np.array([0.0, np.cos(H_rad), np.sin(H_rad)], dtype=np.float64)
    u_min, u_max = float(u.min()), float(u.max())
    return (u - u_min) / max(u_max - u_min, 1e-8)


def _rgb_vec_to_css(rgb: np.ndarray) -> str:
    u8 = np.round(np.clip(rgb, 0.0, 1.0) * 255.0).astype(int)
    return f"rgb({u8[0]}, {u8[1]}, {u8[2]})"


def evaluate_triplet_color_vec(
    H_deg: float,
    s_c: float,
    s_l: float,
    s_c_ref: float = 0.45,
    s_l_ref: float = 0.40,
) -> np.ndarray:
    """
    Evaluate the RGB color vector for (Hue angle H_deg, Chroma fraction s_c in [0, 1],
    Luminance fraction s_l in [0, 1]).
    """
    vivid = vivid_rgb_at_hue(H_deg)
    gray = np.array([0.50, 0.50, 0.50], dtype=np.float64)

    s_c_clamped = float(np.clip(s_c, 0.0, 1.0))
    if s_c_clamped <= s_c_ref:
        c_weight = 0.78 * (s_c_clamped / max(s_c_ref, 1e-8))
    else:
        c_weight = 0.78 + 0.22 * ((s_c_clamped - s_c_ref) / max(1.0 - s_c_ref, 1e-8))
    chroma_rgb = (1.0 - c_weight) * gray + c_weight * vivid

    s_l_clamped = float(np.clip(s_l, 0.0, 1.0))
    if s_l_clamped <= s_l_ref:
        t = s_l_clamped / max(s_l_ref, 1e-8)
        rgb = (0.12 + 0.88 * t) * chroma_rgb
    else:
        t = (s_l_clamped - s_l_ref) / max(1.0 - s_l_ref, 1e-8)
        rgb = (1.0 - 0.75 * t) * chroma_rgb + (0.75 * t) * np.ones(3, dtype=np.float64)

    return np.clip(rgb, 0.0, 1.0)


def make_triplet_gradients_and_swatch(
    h_cur: float,
    h_min: float,
    h_max: float,
    c_cur: float,
    c_min: float,
    c_max: float,
    c_ref: float,
    l_cur: float,
    l_min: float,
    l_max: float,
    l_ref: float,
    h_offset: float = 0.0,
    n_hue_stops: int = 25,
    n_cl_stops: int = 17,
) -> tuple[str, str, str, str]:
    """
    Generate the 3 matched CSS linear-gradient strings (Hue track, Chroma track, Luminance track)
    plus the exact cursor swatch color for any (Hue, Chroma, Luminance) slider triplet.
    """
    s_c_cur = (c_cur - c_min) / max(c_max - c_min, 1e-8)
    s_c_ref = (c_ref - c_min) / max(c_max - c_min, 1e-8)
    s_l_cur = (l_cur - l_min) / max(l_max - l_min, 1e-8)
    s_l_ref = (l_ref - l_min) / max(l_max - l_min, 1e-8)

    eff_h_cur = (h_offset + h_cur) % 360.0

    h_vals = np.linspace(h_min, h_max, n_hue_stops)
    h_stops = [
        _rgb_vec_to_css(evaluate_triplet_color_vec((h_offset + float(h)) % 360.0, 1.0, s_l_ref, s_c_ref, s_l_ref))
        for h in h_vals
    ]
    grad_h = f"linear-gradient(to right, {', '.join(h_stops)})"

    s_c_vals = np.linspace(0.0, 1.0, n_cl_stops)
    c_stops = [
        _rgb_vec_to_css(evaluate_triplet_color_vec(eff_h_cur, float(sc), s_l_cur, s_c_ref, s_l_ref))
        for sc in s_c_vals
    ]
    grad_c = f"linear-gradient(to right, {', '.join(c_stops)})"

    s_l_vals = np.linspace(0.0, 1.0, n_cl_stops)
    l_stops = [
        _rgb_vec_to_css(evaluate_triplet_color_vec(eff_h_cur, s_c_cur, float(sl), s_c_ref, s_l_ref))
        for sl in s_l_vals
    ]
    grad_l = f"linear-gradient(to right, {', '.join(l_stops)})"

    cursor_swatch = _rgb_vec_to_css(evaluate_triplet_color_vec(eff_h_cur, s_c_cur, s_l_cur, s_c_ref, s_l_ref))
    return grad_h, grad_c, grad_l, cursor_swatch


def dominant_hue(rgbmix: np.ndarray, pixels: np.ndarray) -> float:
    """
    Dominant perceived Hue angle (degrees) of the *rendered* image for a given rgbmix.

    Computed on real on-disk pixel samples (N, 3) after mixing and [0, 1] clipping, as the
    lightness-weighted mean opponent (a, b) vector: bright structures dominate the perceived
    color of the Sun, so they are weighted accordingly. Because a Global Hue Shift rotates
    every waveband's (a, b) column by the same angle, the rendered dominant hue becomes
    (dominant_hue + ΔH), which is exactly what the Global Hue colorbar displays.

    Falls back to the peak of the chroma-weighted hue histogram when the mean cast is
    near-neutral (e.g. zero-cast presets), where the mean vector has no stable angle.
    """
    lab = np.clip(pixels @ rgbmix.T, 0.0, 1.0) @ T_RGB2LAB.T
    w = np.clip(lab[:, 0], 0.0, None)
    ab = (lab[:, 1:] * w[:, None]).sum(axis=0) / max(float(w.sum()), 1e-8)
    if float(np.hypot(ab[0], ab[1])) > 5e-3:
        return float(np.degrees(np.atan2(ab[1], ab[0])) % 360.0)
    hues = np.degrees(np.atan2(lab[:, 2], lab[:, 1])) % 360.0
    hist, edges = np.histogram(hues, bins=72, range=(0.0, 360.0), weights=w * np.hypot(lab[:, 1], lab[:, 2]))
    k = int(np.argmax(hist))
    return float(0.5 * (edges[k] + edges[k + 1]))


def global_hue_track(
    anchor_wb: list[dict[str, float]],
    pixels: np.ndarray,
    gh_cur: float,
    gc: float = 1.0,
    gl: float = 1.0,
    n_stops: int = 25,
) -> tuple[str, float]:
    """
    Build the Global Hue Shift colorbar from the *rendered* Sun: each stop at shift ΔH shows the
    vivid color of the dominant hue the image actually takes at that shift (including gamut
    clipping, which bends a pure rotation for near-neutral palettes). Returns the CSS gradient
    and the rendered dominant hue at the current shift (used for the thumb and C/L tracks).
    """
    stops = np.linspace(GLOBAL_H_MIN, GLOBAL_H_MAX, n_stops)
    hues = [dominant_hue(lch_to_rgbmix(apply_global_to_anchor(anchor_wb, float(g), gc, gl)), pixels) for g in stops]
    grad = f"linear-gradient(to right, {', '.join(_rgb_vec_to_css(vivid_rgb_at_hue(h)) for h in hues)})"
    h_cur = dominant_hue(lch_to_rgbmix(apply_global_to_anchor(anchor_wb, float(gh_cur), gc, gl)), pixels)
    return grad, h_cur


def _primary_hue(rgb: list[float]) -> float:
    lab = T_RGB2LAB @ np.asarray(rgb, dtype=np.float64)
    return float(np.degrees(np.atan2(lab[2], lab[1])) % 360.0)


# Actual hue angles of the display primaries/secondaries in this (linear) opponent space,
# used to label the per-waveband Hue sliders truthfully (red is ~17°, not 0°; blue is ~286°, not 270°).
PRIMARY_HUE_MARKS = {
    round(_primary_hue(rgb), 1): name
    for rgb, name in [
        ([1, 0, 0], "R"),
        ([1, 1, 0], "Y"),
        ([0, 1, 0], "G"),
        ([0, 1, 1], "C"),
        ([0, 0, 1], "B"),
        ([1, 0, 1], "M"),
    ]
}


def lch_to_css_swatch(L: float, C: float, H_deg: float) -> str:
    """Convert a waveband's (L, C, H) into its matched display CSS rgb() color."""
    s_c = (C - WB_C_MIN) / (WB_C_MAX - WB_C_MIN)
    s_c_ref = 0.35 / (WB_C_MAX - WB_C_MIN)
    s_l = (L - WB_L_MIN) / (WB_L_MAX - WB_L_MIN)
    s_l_ref = (0.35 - WB_L_MIN) / (WB_L_MAX - WB_L_MIN)
    return _rgb_vec_to_css(evaluate_triplet_color_vec(H_deg, s_c, s_l, s_c_ref, s_l_ref))


def mix_metrics(rgbmix: np.ndarray, pixels: np.ndarray) -> dict[str, float]:
    """Rendered-image diagnostics on real on-disk pixel samples (no co-located-peak approximation)."""
    raw = pixels @ rgbmix.T
    mean_lab = T_RGB2LAB @ np.clip(raw, 0.0, 1.0).mean(axis=0)
    return {
        "a": float(mean_lab[1]),
        "b": float(mean_lab[2]),
        "cast_C": float(np.hypot(mean_lab[1], mean_lab[2])),
        "cast_H": float(np.degrees(np.atan2(mean_lab[2], mean_lab[1])) % 360.0),
        "clip_pct": float(100.0 * np.mean(np.any(raw > 1.0, axis=1))),
        "p995": np.percentile(raw, 99.5, axis=0),
    }


def _solve_chroma_for_cast(
    hues_deg: list[float],
    weights: np.ndarray,
    target_ab: tuple[float, float],
    c_sum: float,
) -> np.ndarray:
    """
    Solve for non-negative per-waveband chromas C_k (at fixed hues h_k) such that
        sum_k w_k C_k (cos h_k, sin h_k) = target_ab   and   sum_k C_k = c_sum.
    A negative C_k would silently flip that waveband's hue by 180° (and would then be
    clipped to 0 by the slider range), so negative solutions are resolved by dropping the
    offending waveband to C=0 and solving the remaining system in the least-squares sense.
    """
    rad = np.radians(np.asarray(hues_deg, dtype=np.float64))
    a_mat = np.vstack([weights * np.cos(rad), weights * np.sin(rad), np.ones(3)])
    rhs = np.array([target_ab[0], target_ab[1], c_sum], dtype=np.float64)
    active = np.ones(3, dtype=bool)
    c = np.zeros(3)
    for _ in range(3):
        sol, *_ = np.linalg.lstsq(a_mat[:, active], rhs, rcond=None)
        c[:] = 0.0
        c[active] = sol
        if np.all(c >= 0.0):
            break
        active &= c > 0.0
    return np.clip(c, 0.0, None)


def compute_auto_balance_presets(
    pixels: np.ndarray,
    grad_x: np.ndarray,
    grad_y: np.ndarray,
) -> dict[str, dict]:
    """
    Compute data-driven, color-constancy & color-harmony Auto-Balance presets in the opponent
    LCh space from real on-disk pixel samples (N, 3) of the [304, 171, 193] channels and their
    spatial gradients (N, 3) along x and y (for Gray-Edge).
    """
    means = pixels.mean(axis=0)
    stds = pixels.std(axis=0)

    def _normalize(wb_lch: list[dict[str, float]], target_max: float = 0.98, pct: float = 99.5) -> list[dict[str, float]]:
        # Scale the whole matrix (all L and C uniformly => hues and balance unchanged) so that the
        # brightest 0.5% of *actual* pixels reach target_max in their highest output channel.
        # The previous `rgbmix @ p99` bound assumed all channels peak at the same pixel, which
        # ignores negative mixing coefficients and spatially separated peaks.
        rgbmix = lch_to_rgbmix(wb_lch)
        peak = float(np.percentile(np.max(pixels @ rgbmix.T, axis=1), pct))
        scale = target_max / peak if peak > 1e-6 else 1.0
        return [
            {
                "H": round(float(w["H"]) % 360.0, 1),
                "C": round(float(np.clip(w["C"] * scale, WB_C_MIN, WB_C_MAX)), 3),
                "L": round(float(np.clip(w["L"] * scale, WB_L_MIN, WB_L_MAX)), 3),
            }
            for w in wb_lch
        ]

    # 1. Gray-World / G2V Solar Equilibrium: disk-mean color = slightly warm white.
    h_gw = [20.0, 122.0, 250.0]
    c_gw = _solve_chroma_for_cast(h_gw, means, (0.015, 0.025), 1.05)
    wb_gw = _normalize([{"H": h_gw[i], "C": float(c_gw[i]), "L": [0.24, 0.72, 0.14][i]} for i in range(3)])

    # 2. Split-Complementary Warm-Cool Harmony
    h_wc = [24.0, 106.0, 245.0]
    c_wc = _solve_chroma_for_cast(h_wc, means, (0.022, 0.035), 1.02)
    wb_wc = _normalize([{"H": h_wc[i], "C": float(c_wc[i]), "L": [0.26, 0.74, 0.12][i]} for i in range(3)])

    # 3. Iso-Perceptual Contrast (Equal Salience): the color difference produced by a 1-sigma
    #    fluctuation of waveband k is sigma_k * ||(L_k, C_k)||. Equalize that product across bands
    #    (the previous version multiplied unequal base vectors by 1/sigma, which left 171 Å with
    #    ~2x the salience of 193 Å).
    h_es = [18.0, 120.0, 248.0]
    base_es = np.array([[0.25, 0.34], [0.62, 0.34], [0.16, 0.28]])  # (L, C) direction per band
    delta_e = stds * np.linalg.norm(base_es, axis=1)
    k_es = np.mean(delta_e) / delta_e
    wb_es = _normalize(
        [{"H": h_es[i], "C": float(base_es[i, 1] * k_es[i]), "L": float(base_es[i, 0] * k_es[i])} for i in range(3)]
    )

    # 4. Triadic 120° Maximum Separation (Zero Net Cast): m_k * C_k equal => chroma vectors cancel.
    h_tr = [18.0, 138.0, 258.0]
    inv_mean = (1.0 / means) / np.mean(1.0 / means)
    c_tr = 0.33 * inv_mean
    wb_tr = _normalize([{"H": h_tr[i], "C": float(c_tr[i]), "L": [0.25, 0.70, 0.15][i]} for i in range(3)])

    # 5. White-Patch (Max-RGB) von Kries correction of the default palette: per-output-channel gains
    #    so that the brightest structures (99.5th pct of each output channel) render neutral white.
    raw_def = pixels @ DEFAULT_JP2_RGBMIX.T
    p_def = np.maximum(np.percentile(raw_def, 99.5, axis=0), 1e-3)
    wb_wp = _normalize(rgbmix_to_waveband_lch(np.diag(0.98 / p_def) @ DEFAULT_JP2_RGBMIX))

    # 6. Gray-Edge color constancy (van de Weijer et al. 2007) on the default palette: diagonal
    #    output gains so the mean gradient magnitude of the R, G and B outputs is equal, i.e. the
    #    average color of *edges* (loops, network boundaries) is neutral, instead of the average of
    #    all pixels, which is dominated by the large, smooth quiet-Sun disk.
    e_out = np.mean(np.hypot(grad_x @ DEFAULT_JP2_RGBMIX.T, grad_y @ DEFAULT_JP2_RGBMIX.T), axis=0)
    g_ge = np.mean(e_out) / np.maximum(e_out, 1e-6)
    wb_ge = _normalize(rgbmix_to_waveband_lch(np.diag(g_ge) @ DEFAULT_JP2_RGBMIX))

    # 7. Decorrelation Stretch (PCA), the classic multispectral planetary-imaging technique:
    #    choose the (a, b) opponent rows within the plane orthogonal to the disk-mean vector
    #    (=> zero net cast), whitened so the two chroma axes are uncorrelated with equal variance
    #    (=> differences BETWEEN wavebands get maximal, balanced color separation). The (a, b)
    #    frame is then rotated/reflected so 304 Å stays red and 304 -> 171 -> 193 runs red -> green -> blue.
    cov = np.cov(pixels, rowvar=False)
    _, _, vt = np.linalg.svd(means.reshape(1, 3))
    q = vt[1:].T  # 3x2 orthonormal basis of {r : r . means = 0}
    evals, evecs = np.linalg.eigh(q.T @ cov @ q)
    w_ab = q @ evecs @ np.diag(1.0 / np.sqrt(np.maximum(evals, 1e-12)))  # 3x2: chroma vector per waveband
    lab_def = (pixels @ DEFAULT_JP2_RGBMIX.T) @ T_RGB2LAB.T
    target_std = 1.25 * float(np.sqrt(0.5 * (lab_def[:, 1].var() + lab_def[:, 2].var())))
    chroma_cols = (w_ab * target_std).T  # 2x3, column k = (a_k, b_k)

    def _orient(cols: np.ndarray) -> np.ndarray:
        h = np.arctan2(cols[1], cols[0])
        rot = np.radians(20.0) - h[0]
        cr, sr = np.cos(rot), np.sin(rot)
        out = np.array([[cr, -sr], [sr, cr]]) @ cols
        hd = np.degrees(np.arctan2(out[1], out[0])) % 360.0
        return out, (hd[1] - hd[0]) % 360.0 < (hd[2] - hd[0]) % 360.0

    chroma_cols, ordered = _orient(chroma_cols)
    if not ordered:
        chroma_cols, _ = _orient(np.diag([1.0, -1.0]) @ chroma_cols)
    lab_dcs = np.vstack([[w["L"] for w in DEFAULT_WB_LCH], chroma_cols])
    wb_dcs = _normalize(rgbmix_to_waveband_lch(T_LAB2RGB @ lab_dcs))

    m_def = mix_metrics(DEFAULT_JP2_RGBMIX, pixels)

    return {
        "default": {
            "label": "Default set_jp2_default()",
            "wb_lch": copy.deepcopy(DEFAULT_WB_LCH),
            "desc": (
                "Original empirical matrix from visualization.py. "
                f"Disk-mean cast C={m_def['cast_C']:.3f} at H={m_def['cast_H']:.0f}° (a={m_def['a']:+.3f}, b={m_def['b']:+.3f}); "
                f"{m_def['clip_pct']:.1f}% of on-disk pixels clip; output p99.5 = "
                f"({m_def['p995'][0]:.2f}, {m_def['p995'][1]:.2f}, {m_def['p995'][2]:.2f})."
            ),
        },
        "gray_world": {
            "label": "1. G2V Gray-World",
            "wb_lch": wb_gw,
            "desc": "Data-Driven Gray-World Equilibrium: Balances channel chromas against on-disk mean intensities so overlapping bright structures sum to warm G2V solar white without clipping.",
        },
        "warm_cool": {
            "label": "2. Warm–Cool Harmony",
            "wb_lch": wb_wc,
            "desc": "Split-Complementary Thermal Harmony: Shifts 171 Å loops to pure Amber-Gold (106°) and 304 Å to Vermilion (24°), opposed by complementary Cerulean-Blue (245°) for the 1.5 MK corona.",
        },
        "equal_salience": {
            "label": "3. Equal Salience (ΔE)",
            "wb_lch": wb_es,
            "desc": "Iso-Perceptual Contrast: Weights each channel inversely to its image standard deviation (1/σ) so 304 Å prominences, 171 Å loops, and 193 Å plumes have equal visual salience.",
        },
        "triadic_120": {
            "label": "4. 120° Oklch Triad",
            "wb_lch": wb_tr,
            "desc": "Maximum Chromatic Separation: Spaces 304, 171, and 193 Å exactly 120° apart (18°, 138°, 258°) with mean-compensated chroma for exact zero net color cast (a=0, b=0).",
        },
        "white_patch": {
            "label": "5. White-Patch (Max-RGB)",
            "wb_lch": wb_wp,
            "desc": "Von Kries White-Patch correction of the default palette: per-output-channel gains make the brightest 0.5% of structures (flares, AR cores, limb brightenings) render neutral white instead of tinted, while keeping the default hue ordering.",
        },
        "gray_edge": {
            "label": "6. Gray-Edge",
            "wb_lch": wb_ge,
            "desc": "Gray-Edge color constancy on the default palette: per-output-channel gains make the mean color of spatial gradients (loop edges, network boundaries) neutral, rather than the mean of all pixels, so fine structure is balanced instead of the smooth quiet-Sun background. Because 193 Å structure is much smoother than the 304 Å network, this yields a cooler palette.",
        },
        "decorrelation": {
            "label": "7. Decorrelation Stretch",
            "wb_lch": wb_dcs,
            "desc": "PCA Decorrelation Stretch (multispectral imaging): chroma axes are whitened so inter-waveband differences get maximal, balanced color separation with zero net disk cast; luminance kept from the default, 304 Å kept red.",
        },
    }


def generate_color_wheel_uri(size: int = 148) -> str:
    """Generate a smooth perceptual Oklch color wheel using the exact same vivid Hue mapping."""
    yy, xx = np.mgrid[0:size, 0:size]
    nx = (xx - (size - 1) / 2.0) / ((size - 1) / 2.0)
    ny = ((size - 1) / 2.0 - yy) / ((size - 1) / 2.0)
    r = np.hypot(nx, ny)
    angles = np.atan2(ny, nx)

    u = np.stack([np.zeros_like(angles), np.cos(angles), np.sin(angles)], axis=-1) @ T_LAB2RGB.T
    u_min = np.min(u, axis=-1, keepdims=True)
    u_max = np.max(u, axis=-1, keepdims=True)
    vivid = (u - u_min) / np.maximum(u_max - u_min, 1e-8)

    r_clamped = np.clip(r, 0.0, 1.0)[..., None]
    gray = np.full_like(vivid, 0.50)
    rgb = (1.0 - r_clamped) * gray + r_clamped * vivid

    bg = np.array([30.0 / 255.0, 41.0 / 255.0, 59.0 / 255.0])
    alpha = np.clip((1.0 - r) * (size / 3.0), 0.0, 1.0)[..., None]
    comp = rgb * alpha + bg * (1.0 - alpha)
    bgr_u8 = (comp[:, :, ::-1] * 255.0).astype(np.uint8)
    ok, buf = cv2.imencode(".png", bgr_u8)
    if not ok:
        raise RuntimeError("Failed to encode Oklch color wheel PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.tobytes()).decode('ascii')}"


DEFAULT_WB_LCH = rgbmix_to_waveband_lch(DEFAULT_JP2_RGBMIX)
COLOR_WHEEL_URI = generate_color_wheel_uri(148)


class JP2ImageCache:
    """Pre-loads the three 4096x4096 JP2 channels, computes statistics, and caches fast views."""

    def __init__(self, jp2_files: list[str]):
        self.jp2_files = jp2_files
        raw_channels = [visualization.load_jp2(f).astype(np.float32) / 255.0 for f in jp2_files]
        self.full_stack = np.stack(raw_channels, axis=-1)
        h, w, _ = self.full_stack.shape

        self.views: dict[str, np.ndarray] = {
            "full_1024": cv2.resize(self.full_stack, (1024, 1024), interpolation=cv2.INTER_AREA),
            "full_1536": cv2.resize(self.full_stack, (1536, 1536), interpolation=cv2.INTER_AREA),
            "crop_bottom_half": cv2.resize(
                self.full_stack[1935 : min(h, 1935 + 2160), 128 : min(w, 128 + 3840), :],
                (1280, 720),
                interpolation=cv2.INTER_AREA,
            ),
            "zoom_prominence": self.full_stack[2300:3324, 200:1224, :].copy(),
            "zoom_active_region": self.full_stack[1050:2074, 2800:3824, :].copy(),
        }

        preview = self.views["full_1024"].astype(np.float64)
        mask = np.max(preview, axis=-1) > 0.05
        self.disk_pixels = preview[mask]
        self.channel_means = np.mean(self.disk_pixels, axis=0)
        self.channel_stds = np.std(self.disk_pixels, axis=0)
        # Small fixed subsample for per-callback diagnostics (dominant hue, cast, clipping).
        stride = max(1, len(self.disk_pixels) // 40000)
        self.sample_pixels = self.disk_pixels[::stride].copy()
        # Even smaller subsample for the 26 renders behind the Global Hue colorbar.
        self.track_pixels = self.sample_pixels[::4].copy()
        # Per-channel spatial gradients at the same pixels (Gray-Edge statistic). Gradients are linear,
        # so the gradient of any mixed output channel is simply rgbmix @ (gx, gy).
        # Standard Gray-Edge pre-smoothing (sigma=2 px) suppresses pixel noise in the gradients.
        smooth = cv2.GaussianBlur(preview, (0, 0), 2.0)
        self.disk_grad_x = np.stack([cv2.Sobel(smooth[:, :, k], cv2.CV_64F, 1, 0, ksize=3)[mask] for k in range(3)], axis=-1)
        self.disk_grad_y = np.stack([cv2.Sobel(smooth[:, :, k], cv2.CV_64F, 0, 1, ksize=3)[mask] for k in range(3)], axis=-1)
        self.presets = compute_auto_balance_presets(self.disk_pixels, self.disk_grad_x, self.disk_grad_y)

        self.default_uris: dict[str, str] = {
            view_name: self.render_uri(view_name, DEFAULT_JP2_RGBMIX)
            for view_name in self.views
        }

    def render_uri(
        self,
        view_name: str,
        rgbmix: np.ndarray,
        scalemin: float = 0.0,
        jpeg_quality: int = 92,
    ) -> str:
        """Apply 3x3 rgbmix to cached float32 stack and return a base64 JPEG data URI."""
        stack = self.views[view_name]
        mixed = stack @ rgbmix.astype(np.float32).T
        np.clip(mixed, 0.0, 1.0, out=mixed)
        mixed *= 255.0
        if scalemin > 0:
            mixed = (mixed - scalemin) * (255.0 / (255.0 - scalemin))
            np.clip(mixed, 0.0, 255.0, out=mixed)

        bgr_u8 = mixed[:, :, ::-1].astype(np.uint8)
        ok, buf = cv2.imencode(".jpg", bgr_u8, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        if not ok:
            raise RuntimeError("Failed to encode preview image as JPEG")
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"


def _format_dark_marks(marks: dict) -> dict:
    """Ensure slider mark labels have light gray contrast on the dark background."""
    return {
        k: {"label": str(v), "style": {"color": "#cbd5e1", "fontSize": "11px"}}
        for k, v in marks.items()
    }


def _slider_wrap_style(gradient: str, thumb_color: str) -> dict:
    """Set CSS custom properties on the slider wrapper so the track IS the colorbar and the thumb IS the cursor swatch."""
    return {
        "--track-bg": gradient,
        "--thumb-bg": thumb_color,
    }


def _slider_row(
    label: str,
    slider_id: str,
    min_val: float,
    max_val: float,
    step: float,
    val: float,
    marks: dict,
    initial_gradient: str,
    initial_thumb_color: str,
):
    return html.Div(
        style={"marginBottom": "10px"},
        children=[
            html.Div(
                style={"display": "flex", "justifyContent": "space-between", "marginBottom": "2px"},
                children=[
                    html.Span(label, style={"fontSize": "13px", "fontWeight": "600", "color": "#e2e8f0"}),
                    html.Span(
                        id=f"{slider_id}-val",
                        style={"fontSize": "12px", "fontFamily": "monospace", "fontWeight": "600", "color": "#38bdf8"},
                    ),
                ],
            ),
            # Wrapper applies --track-bg and --thumb-bg directly onto the dcc.Slider track and thumb
            html.Div(
                id=f"{slider_id}-wrap",
                className="color-slider-wrap",
                style=_slider_wrap_style(initial_gradient, initial_thumb_color),
                children=[
                    dcc.Slider(
                        id=slider_id,
                        min=min_val,
                        max=max_val,
                        step=step,
                        value=val,
                        included=False,
                        marks=_format_dark_marks(marks),
                        updatemode="drag",
                        tooltip={"placement": "bottom", "always_visible": False},
                    )
                ],
            ),
        ],
    )


def _wheel_pin_style(H_deg: float, C: float, swatch_color: str) -> dict:
    """Compute CSS position for a waveband pin on the 148x148 polar color wheel."""
    r_norm = float(np.clip(C / WB_C_MAX, 0.0, 0.96))
    rad = np.radians(H_deg)
    x_pct = 50.0 + 50.0 * r_norm * float(np.cos(rad))
    y_pct = 50.0 - 50.0 * r_norm * float(np.sin(rad))
    return {
        "position": "absolute",
        "left": f"{x_pct:.1f}%",
        "top": f"{y_pct:.1f}%",
        "transform": "translate(-50%, -50%)",
        "backgroundColor": swatch_color,
        "color": "#0f172a",
        "fontWeight": "800",
        "fontSize": "10px",
        "padding": "2px 5px",
        "borderRadius": "10px",
        "border": "2px solid #f8fafc",
        "boxShadow": "0 1px 4px rgba(0,0,0,0.85)",
        "pointerEvents": "none",
        "whiteSpace": "nowrap",
    }


DARK_THEME_INDEX_STRING = """<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            :root, body {
                background-color: #0f172a;
                color: #e2e8f0;
                margin: 0;
                padding: 0;
                --Dash-Text-Primary: #e2e8f0 !important;
                --Dash-Text-Strong: #f8fafc !important;
                --Dash-Text-Weak: #cbd5e1 !important;
                --Dash-Text-Disabled: #94a3b8 !important;
                --Dash-Stroke-Strong: #64748b !important;
                --Dash-Stroke-Weak: #334155 !important;
                --Dash-Fill-Interactive-Strong: #38bdf8 !important;
                --Dash-Fill-Interactive-Weak: #334155 !important;
                --Dash-Fill-Inverse-Strong: #1e293b !important;
                --Dash-Fill-Primary-Hover: rgba(56, 189, 248, 0.16) !important;
                --Dash-Fill-Primary-Active: rgba(56, 189, 248, 0.26) !important;
                --Dash-Fill-Disabled: #1e293b !important;
            }
            /* Style the dcc.Slider track itself as the Lightroom-style colorbar */
            .color-slider-wrap .dash-slider-root {
                padding: 6px 0 24px 0 !important;
            }
            .color-slider-wrap .dash-slider-track {
                height: 12px !important;
                border-radius: 6px !important;
                background: var(--track-bg) !important;
                border: 1px solid #475569 !important;
                box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.5) !important;
            }
            .color-slider-wrap .dash-slider-range {
                background: transparent !important;
            }
            .color-slider-wrap .dash-slider-thumb {
                width: 18px !important;
                height: 18px !important;
                background-color: var(--thumb-bg, #38bdf8) !important;
                border: 2.5px solid #ffffff !important;
                box-shadow: 0 0 5px rgba(0, 0, 0, 0.9) !important;
            }
            .color-slider-wrap .dash-slider-dot {
                background-color: rgba(255, 255, 255, 0.35) !important;
                width: 4px !important;
                height: 4px !important;
            }
            .dash-slider-mark,
            .dash-slider-mark-outside-selection,
            .rc-slider-mark-text,
            .rc-slider-mark-text-active {
                color: #cbd5e1 !important;
                font-size: 11px !important;
            }
            .dash-dropdown,
            .dash-dropdown-trigger,
            .dash-dropdown-content,
            .dash-dropdown-options,
            .dash-dropdown-option,
            .dash-dropdown-value,
            .dash-dropdown-search {
                background-color: #1e293b !important;
                color: #e2e8f0 !important;
                border-color: #475569 !important;
            }
            .dash-dropdown-option:hover {
                background-color: #334155 !important;
                color: #ffffff !important;
            }
            label, .dash-radioitems label {
                color: #e2e8f0 !important;
                cursor: pointer;
            }
            .preset-btn:hover {
                background-color: #334155 !important;
                border-color: #38bdf8 !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""


def create_dash_app(cache: JP2ImageCache, output_dir: str) -> Dash:
    app = Dash(__name__, title="AIA-Reloaded Interactive RGB / LCh Mixer")
    app.index_string = DARK_THEME_INDEX_STRING

    init_gh_grad, base_dom_h = global_hue_track(DEFAULT_WB_LCH, cache.track_pixels, 0.0)
    _, init_gc_grad, init_gl_grad, init_g_swatch = make_triplet_gradients_and_swatch(
        0.0, GLOBAL_H_MIN, GLOBAL_H_MAX,
        1.0, GLOBAL_C_MIN, GLOBAL_C_MAX, 1.0,
        1.0, GLOBAL_L_MIN, GLOBAL_L_MAX, 1.0,
        h_offset=base_dom_h,
    )
    init_gh_hue_color = _rgb_vec_to_css(vivid_rgb_at_hue(base_dom_h))

    waveband_meta = [
        {"idx": 0, "wavel": "304 Å", "short": "304", "role": "Red Input — Chromosphere & Prominences"},
        {"idx": 1, "wavel": "171 Å", "short": "171", "role": "Green Input — Quiet Corona & Coronal Loops"},
        {"idx": 2, "wavel": "193 Å", "short": "193", "role": "Blue Input — Hot Corona & Flare Loops"},
    ]

    wb_cards = []
    for wb in waveband_meta:
        i = wb["idx"]
        init = DEFAULT_WB_LCH[i]
        wh_grad, wc_grad, wl_grad, w_swatch = make_triplet_gradients_and_swatch(
            init["H"], WB_H_MIN, WB_H_MAX,
            init["C"], WB_C_MIN, WB_C_MAX, 0.35,
            init["L"], WB_L_MIN, WB_L_MAX, 0.35,
            h_offset=0.0,
        )
        wh_vivid = _rgb_vec_to_css(vivid_rgb_at_hue(init["H"]))
        wb_cards.append(
            html.Div(
                style={
                    "backgroundColor": "#1e293b",
                    "border": "1px solid #334155",
                    "borderRadius": "8px",
                    "padding": "12px",
                    "marginBottom": "10px",
                },
                children=[
                    html.Div(
                        style={"display": "flex", "alignItems": "center", "justifyContent": "space-between", "marginBottom": "8px"},
                        children=[
                            html.Div(
                                children=[
                                    html.Span(wb["wavel"], style={"fontWeight": "700", "fontSize": "14px", "color": "#f8fafc"}),
                                    html.Span(f"  ({wb['role']})", style={"fontSize": "11px", "color": "#cbd5e1"}),
                                ]
                            ),
                            html.Div(
                                id=f"wb-{i}-swatch",
                                title="Effective basis color for this waveband",
                                style={
                                    "width": "24px",
                                    "height": "24px",
                                    "borderRadius": "50%",
                                    "border": "2px solid #cbd5e1",
                                    "backgroundColor": w_swatch,
                                },
                            ),
                        ],
                    ),
                    _slider_row(
                        f"{wb['wavel']} Hue Angle (H°)",
                        f"wb-{i}-h",
                        WB_H_MIN,
                        WB_H_MAX,
                        0.1,
                        round(init["H"], 1),
                        PRIMARY_HUE_MARKS,
                        wh_grad,
                        wh_vivid,
                    ),
                    _slider_row(
                        f"{wb['wavel']} Chroma / Saturation (C)",
                        f"wb-{i}-c",
                        WB_C_MIN,
                        WB_C_MAX,
                        0.001,
                        round(init["C"], 3),
                        {0.0: "0 (Gray)", 0.35: "0.35", 0.7: "0.70"},
                        wc_grad,
                        w_swatch,
                    ),
                    _slider_row(
                        f"{wb['wavel']} Luminance Weight (L)",
                        f"wb-{i}-l",
                        WB_L_MIN,
                        WB_L_MAX,
                        0.001,
                        round(init["L"], 3),
                        {-0.2: "-0.2", 0.0: "0", 0.5: "0.5", 1.0: "1.0"},
                        wl_grad,
                        w_swatch,
                    ),
                ],
            )
        )

    preset_btn_style = {
        "backgroundColor": "#0f172a",
        "color": "#e2e8f0",
        "border": "1px solid #475569",
        "borderRadius": "6px",
        "padding": "7px 10px",
        "cursor": "pointer",
        "fontWeight": "600",
        "fontSize": "12px",
        "textAlign": "left",
    }

    init_anchor = apply_global_to_anchor(DEFAULT_WB_LCH, 0.0, 1.0, 1.0)

    app.layout = html.Div(
        style={
            "backgroundColor": "#0f172a",
            "color": "#f1f5f9",
            "minHeight": "100vh",
            "fontFamily": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
            "padding": "16px 24px",
        },
        children=[
            # Single authoritative store holding anchor_wb and last_synced_wb
            dcc.Store(
                id="mixer-state-store",
                data={
                    "anchor_wb": copy.deepcopy(init_anchor),
                    "last_synced_wb": copy.deepcopy(init_anchor),
                },
            ),
            # Top Header Bar
            html.Div(
                style={
                    "display": "flex",
                    "justifyContent": "space-between",
                    "alignItems": "center",
                    "borderBottom": "1px solid #334155",
                    "paddingBottom": "12px",
                    "marginBottom": "16px",
                },
                children=[
                    html.Div(
                        children=[
                            html.H2(
                                "SDO/AIA Interactive Perceptual LCh & RGB Mixer (.jp2)",
                                style={"margin": "0", "fontSize": "20px", "fontWeight": "700", "color": "#f8fafc"},
                            ),
                            html.Div(
                                f"Loaded samples: {', '.join(os.path.basename(f) for f in cache.jp2_files)}",
                                style={"fontSize": "12px", "color": "#cbd5e1", "marginTop": "2px"},
                            ),
                        ]
                    ),
                    html.Div(
                        style={"display": "flex", "gap": "10px", "alignItems": "center"},
                        children=[
                            html.Button(
                                "Reset All to set_jp2_default()",
                                id="btn-reset",
                                n_clicks=0,
                                style={
                                    "backgroundColor": "#334155",
                                    "color": "#f8fafc",
                                    "border": "1px solid #64748b",
                                    "borderRadius": "6px",
                                    "padding": "8px 14px",
                                    "cursor": "pointer",
                                    "fontWeight": "600",
                                    "fontSize": "12px",
                                },
                            ),
                            html.Button(
                                "Save Full 4096×4096 JPEG to samples/",
                                id="btn-save-full",
                                n_clicks=0,
                                style={
                                    "backgroundColor": "#0284c7",
                                    "color": "#ffffff",
                                    "border": "none",
                                    "borderRadius": "6px",
                                    "padding": "8px 14px",
                                    "cursor": "pointer",
                                    "fontWeight": "600",
                                    "fontSize": "12px",
                                },
                            ),
                        ],
                    ),
                ],
            ),
            # Main 2-Column Content Area
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "440px 1fr", "gap": "20px", "alignItems": "start"},
                children=[
                    # LEFT PANEL: Auto-Balance + Controls + Matrix Output
                    html.Div(
                        children=[
                            # Auto-Balancing Rules Card
                            html.Div(
                                style={
                                    "backgroundColor": "#1e293b",
                                    "border": "1px solid #a7f3d0",
                                    "borderRadius": "8px",
                                    "padding": "12px 14px",
                                    "marginBottom": "14px",
                                },
                                children=[
                                    html.Div(
                                        "Auto-Balance Rules (Data-Driven & Color Harmony)",
                                        style={"fontWeight": "700", "fontSize": "14px", "color": "#a7f3d0", "marginBottom": "6px"},
                                    ),
                                    html.Div(
                                        "Click any rule to auto-compute optimal (H, C, L) from the JP2 image statistics and update all Option 1 & Option 2 sliders:",
                                        style={"fontSize": "11px", "color": "#cbd5e1", "marginBottom": "10px"},
                                    ),
                                    html.Div(
                                        style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "8px", "marginBottom": "10px"},
                                        children=[
                                            html.Button(
                                                cache.presets["gray_world"]["label"],
                                                id="btn-auto-gray-world",
                                                className="preset-btn",
                                                n_clicks=0,
                                                style=preset_btn_style,
                                            ),
                                            html.Button(
                                                cache.presets["warm_cool"]["label"],
                                                id="btn-auto-warm-cool",
                                                className="preset-btn",
                                                n_clicks=0,
                                                style=preset_btn_style,
                                            ),
                                            html.Button(
                                                cache.presets["equal_salience"]["label"],
                                                id="btn-auto-equal-salience",
                                                className="preset-btn",
                                                n_clicks=0,
                                                style=preset_btn_style,
                                            ),
                                            html.Button(
                                                cache.presets["triadic_120"]["label"],
                                                id="btn-auto-triadic",
                                                className="preset-btn",
                                                n_clicks=0,
                                                style=preset_btn_style,
                                            ),
                                            html.Button(
                                                cache.presets["white_patch"]["label"],
                                                id="btn-auto-white-patch",
                                                className="preset-btn",
                                                n_clicks=0,
                                                style=preset_btn_style,
                                            ),
                                            html.Button(
                                                cache.presets["gray_edge"]["label"],
                                                id="btn-auto-gray-edge",
                                                className="preset-btn",
                                                n_clicks=0,
                                                style=preset_btn_style,
                                            ),
                                            html.Button(
                                                cache.presets["decorrelation"]["label"],
                                                id="btn-auto-decorrelation",
                                                className="preset-btn",
                                                n_clicks=0,
                                                style=preset_btn_style,
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        id="auto-balance-desc",
                                        children=cache.presets["default"]["desc"],
                                        style={
                                            "backgroundColor": "#0f172a",
                                            "border": "1px solid #334155",
                                            "borderRadius": "6px",
                                            "padding": "8px 10px",
                                            "fontSize": "11px",
                                            "color": "#cbd5e1",
                                            "lineHeight": "1.45",
                                        },
                                    ),
                                ],
                            ),
                            # Option 1: Global LCh Controls + Color Wheel
                            html.Div(
                                style={
                                    "backgroundColor": "#1e293b",
                                    "border": "1px solid #38bdf8",
                                    "borderRadius": "8px",
                                    "padding": "14px",
                                    "marginBottom": "14px",
                                },
                                children=[
                                    html.Div(
                                        style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "8px"},
                                        children=[
                                            html.Span(
                                                "Option 1 — Global Perceptual LCh (3 Sliders)",
                                                style={"fontWeight": "700", "fontSize": "14px", "color": "#38bdf8"},
                                            ),
                                            html.Div(
                                                style={"display": "flex", "alignItems": "center", "gap": "8px"},
                                                children=[
                                                    html.Div(
                                                        id="global-swatch",
                                                        title="Current Global Solar Tone (matches Chroma & Luminance slider thumbs)",
                                                        style={
                                                            "width": "22px",
                                                            "height": "22px",
                                                            "borderRadius": "50%",
                                                            "border": "2px solid #f8fafc",
                                                            "backgroundColor": init_g_swatch,
                                                        },
                                                    ),
                                                    html.Button(
                                                        "Reset Global",
                                                        id="btn-reset-global",
                                                        n_clicks=0,
                                                        style={
                                                            "backgroundColor": "#334155",
                                                            "color": "#e2e8f0",
                                                            "border": "1px solid #64748b",
                                                            "borderRadius": "4px",
                                                            "padding": "3px 8px",
                                                            "cursor": "pointer",
                                                            "fontSize": "11px",
                                                        },
                                                    ),
                                                ],
                                            ),
                                        ],
                                    ),
                                    # Interactive Perceptual Color Wheel showing 304, 171, 193 positions
                                    html.Div(
                                        style={
                                            "display": "flex",
                                            "alignItems": "center",
                                            "gap": "14px",
                                            "backgroundColor": "#0f172a",
                                            "border": "1px solid #334155",
                                            "borderRadius": "8px",
                                            "padding": "10px 12px",
                                            "marginBottom": "12px",
                                        },
                                        children=[
                                            html.Div(
                                                style={
                                                    "position": "relative",
                                                    "width": "136px",
                                                    "height": "136px",
                                                    "flexShrink": "0",
                                                    "backgroundImage": f"url('{COLOR_WHEEL_URI}')",
                                                    "backgroundSize": "cover",
                                                    "borderRadius": "50%",
                                                    "border": "1px solid #475569",
                                                },
                                                children=[
                                                    html.Div(
                                                        style={
                                                            "position": "absolute",
                                                            "left": "50%",
                                                            "top": "50%",
                                                            "width": "6px",
                                                            "height": "6px",
                                                            "transform": "translate(-50%, -50%)",
                                                            "borderRadius": "50%",
                                                            "backgroundColor": "#f8fafc",
                                                            "opacity": "0.6",
                                                        }
                                                    ),
                                                    html.Div(
                                                        "304",
                                                        id="wheel-pin-0",
                                                        style=_wheel_pin_style(
                                                            init_anchor[0]["H"],
                                                            init_anchor[0]["C"],
                                                            lch_to_css_swatch(init_anchor[0]["L"], init_anchor[0]["C"], init_anchor[0]["H"]),
                                                        ),
                                                    ),
                                                    html.Div(
                                                        "171",
                                                        id="wheel-pin-1",
                                                        style=_wheel_pin_style(
                                                            init_anchor[1]["H"],
                                                            init_anchor[1]["C"],
                                                            lch_to_css_swatch(init_anchor[1]["L"], init_anchor[1]["C"], init_anchor[1]["H"]),
                                                        ),
                                                    ),
                                                    html.Div(
                                                        "193",
                                                        id="wheel-pin-2",
                                                        style=_wheel_pin_style(
                                                            init_anchor[2]["H"],
                                                            init_anchor[2]["C"],
                                                            lch_to_css_swatch(init_anchor[2]["L"], init_anchor[2]["C"], init_anchor[2]["H"]),
                                                        ),
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                style={"fontSize": "11px", "color": "#cbd5e1", "lineHeight": "1.45"},
                                                children=[
                                                    html.Div(
                                                        "Perceptual Oklch Color Wheel",
                                                        style={"fontWeight": "700", "color": "#f8fafc", "fontSize": "12px", "marginBottom": "4px"},
                                                    ),
                                                    html.Div("• Hue thumb picks the vivid color on the Hue track"),
                                                    html.Div("• Chroma track grades Gray → that exact Hue color"),
                                                    html.Div("• Luminance track grades Dark → Chroma thumb → Bright"),
                                                ],
                                            ),
                                        ],
                                    ),
                                    _slider_row(
                                        "Global Hue Shift (ΔH°)",
                                        "global-h",
                                        GLOBAL_H_MIN,
                                        GLOBAL_H_MAX,
                                        0.5,
                                        0.0,
                                        {-180: "-180°", -90: "-90°", 0: "0°", 90: "+90°", 180: "+180°"},
                                        init_gh_grad,
                                        init_gh_hue_color,
                                    ),
                                    _slider_row(
                                        "Global Chroma / Saturation Multiplier (×C)",
                                        "global-c",
                                        GLOBAL_C_MIN,
                                        GLOBAL_C_MAX,
                                        0.01,
                                        1.0,
                                        {0.0: "0.0×", 0.5: "0.5×", 1.0: "1.0×", 1.5: "1.5×", 2.0: "2.0×"},
                                        init_gc_grad,
                                        init_g_swatch,
                                    ),
                                    _slider_row(
                                        "Global Luminance / Lightness Multiplier (×L)",
                                        "global-l",
                                        GLOBAL_L_MIN,
                                        GLOBAL_L_MAX,
                                        0.01,
                                        1.0,
                                        {0.5: "0.5×", 1.0: "1.0×", 1.5: "1.5×", 2.0: "2.0×"},
                                        init_gl_grad,
                                        init_g_swatch,
                                    ),
                                ],
                            ),
                            # Option 2: Per-Waveband LCh Controls
                            html.Details(
                                open=True,
                                style={
                                    "backgroundColor": "#0f172a",
                                    "border": "1px solid #334155",
                                    "borderRadius": "8px",
                                    "padding": "10px",
                                    "marginBottom": "14px",
                                },
                                children=[
                                    html.Summary(
                                        "Option 2 — Per-Waveband LCh Controls (304 / 171 / 193 Å)",
                                        style={
                                            "fontWeight": "700",
                                            "fontSize": "14px",
                                            "color": "#f8fafc",
                                            "cursor": "pointer",
                                            "marginBottom": "10px",
                                        },
                                    ),
                                    html.Div(
                                        "Each waveband defines one column of self.rgbmix. Sliders move automatically with Option 1 and Auto-Balance rules, and can also be fine-tuned individually.",
                                        style={"fontSize": "11px", "color": "#cbd5e1", "marginBottom": "10px"},
                                    ),
                                    *wb_cards,
                                ],
                            ),
                            # Resulting self.rgbmix Python snippet
                            html.Div(
                                style={
                                    "backgroundColor": "#1e293b",
                                    "border": "1px solid #334155",
                                    "borderRadius": "8px",
                                    "padding": "12px",
                                },
                                children=[
                                    html.Div(
                                        style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "6px"},
                                        children=[
                                            html.Span(
                                                "Resulting self.rgbmix Matrix (Copy to visualization.py)",
                                                style={"fontWeight": "700", "fontSize": "13px", "color": "#a7f3d0"},
                                            ),
                                            dcc.Clipboard(
                                                target_id="rgbmix-code",
                                                title="Copy self.rgbmix code",
                                                style={"cursor": "pointer", "fontSize": "14px", "color": "#38bdf8"},
                                            ),
                                        ],
                                    ),
                                    html.Pre(
                                        id="rgbmix-code",
                                        style={
                                            "backgroundColor": "#090d16",
                                            "color": "#e2e8f0",
                                            "padding": "10px",
                                            "borderRadius": "6px",
                                            "fontSize": "12px",
                                            "fontFamily": "monospace",
                                            "margin": "0",
                                            "overflowX": "auto",
                                        },
                                    ),
                                    html.Div(
                                        id="balance-metrics",
                                        style={"fontSize": "11px", "fontFamily": "monospace", "color": "#cbd5e1", "marginTop": "8px"},
                                    ),
                                    html.Div(
                                        id="save-status",
                                        style={"fontSize": "12px", "color": "#38bdf8", "marginTop": "6px"},
                                    ),
                                ],
                            ),
                        ]
                    ),
                    # RIGHT PANEL: Interactive Viewport & Comparison Controls
                    html.Div(
                        children=[
                            # Viewport Toolbar
                            html.Div(
                                style={
                                    "display": "flex",
                                    "flexWrap": "wrap",
                                    "gap": "16px",
                                    "alignItems": "center",
                                    "justifyContent": "space-between",
                                    "backgroundColor": "#1e293b",
                                    "border": "1px solid #334155",
                                    "borderRadius": "8px",
                                    "padding": "10px 16px",
                                    "marginBottom": "14px",
                                },
                                children=[
                                    html.Div(
                                        style={"display": "flex", "alignItems": "center", "gap": "10px"},
                                        children=[
                                            html.Span("Region / Resolution:", style={"fontSize": "13px", "fontWeight": "600", "color": "#e2e8f0"}),
                                            dcc.Dropdown(
                                                id="view-select",
                                                options=[
                                                    {"label": "Full Sun (1024×1024 Fast Preview)", "value": "full_1024"},
                                                    {"label": "Full Sun (1536×1536 High-Res Preview)", "value": "full_1536"},
                                                    {"label": "Bottom Half 16:9 Crop (Prominence + Corona)", "value": "crop_bottom_half"},
                                                    {"label": "1:1 Native Zoom — Erupting Prominence (304 Å)", "value": "zoom_prominence"},
                                                    {"label": "1:1 Native Zoom — Active Region Loops (171/193 Å)", "value": "zoom_active_region"},
                                                ],
                                                value="full_1024",
                                                clearable=False,
                                                style={"width": "340px", "fontSize": "13px", "color": "#e2e8f0"},
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        style={"display": "flex", "alignItems": "center", "gap": "10px"},
                                        children=[
                                            html.Span("Comparison Mode:", style={"fontSize": "13px", "fontWeight": "600", "color": "#e2e8f0"}),
                                            dcc.RadioItems(
                                                id="compare-mode",
                                                options=[
                                                    {"label": " Side-by-Side (Modified vs. Default)", "value": "side_by_side"},
                                                    {"label": " Single Large View (Modified)", "value": "single"},
                                                    {"label": " Hold Default Only", "value": "default_only"},
                                                ],
                                                value="side_by_side",
                                                inline=True,
                                                style={"fontSize": "13px", "display": "flex", "gap": "12px", "color": "#e2e8f0"},
                                                labelStyle={"color": "#e2e8f0", "marginRight": "8px"},
                                            ),
                                        ],
                                    ),
                                    html.Span(
                                        id="render-timing",
                                        style={"fontSize": "12px", "fontFamily": "monospace", "color": "#cbd5e1"},
                                    ),
                                ],
                            ),
                            # Image Display Container
                            html.Div(
                                id="image-display-grid",
                                style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"},
                                children=[
                                    # Modified Image Card
                                    html.Div(
                                        id="modified-card",
                                        style={
                                            "backgroundColor": "#090d16",
                                            "border": "1px solid #38bdf8",
                                            "borderRadius": "8px",
                                            "padding": "10px",
                                            "textAlign": "center",
                                        },
                                        children=[
                                            html.Div(
                                                "Modified Mix (Live LCh -> self.rgbmix)",
                                                style={"fontSize": "13px", "fontWeight": "700", "color": "#38bdf8", "marginBottom": "8px"},
                                            ),
                                            html.Img(
                                                id="img-modified",
                                                style={"width": "100%", "maxHeight": "78vh", "objectFit": "contain", "borderRadius": "4px"},
                                            ),
                                        ],
                                    ),
                                    # Default Reference Card
                                    html.Div(
                                        id="default-card",
                                        style={
                                            "backgroundColor": "#090d16",
                                            "border": "1px solid #334155",
                                            "borderRadius": "8px",
                                            "padding": "10px",
                                            "textAlign": "center",
                                        },
                                        children=[
                                            html.Div(
                                                "Default set_jp2_default() Reference",
                                                style={"fontSize": "13px", "fontWeight": "600", "color": "#cbd5e1", "marginBottom": "8px"},
                                            ),
                                            html.Img(
                                                id="img-default",
                                                style={"width": "100%", "maxHeight": "78vh", "objectFit": "contain", "borderRadius": "4px"},
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                        ]
                    ),
                ],
            ),
        ],
    )

    # -------------------------------------------------------------------------
    # Single Atomic Master Callback:
    # Updates slider positions, renders image, and sets --track-bg & --thumb-bg
    # directly on each slider wrapper so the slider rides directly ON its colorbar.
    # -------------------------------------------------------------------------
    @app.callback(
        Output("wb-0-h", "value"),
        Output("wb-0-c", "value"),
        Output("wb-0-l", "value"),
        Output("wb-1-h", "value"),
        Output("wb-1-c", "value"),
        Output("wb-1-l", "value"),
        Output("wb-2-h", "value"),
        Output("wb-2-c", "value"),
        Output("wb-2-l", "value"),
        Output("global-h", "value"),
        Output("global-c", "value"),
        Output("global-l", "value"),
        Output("img-modified", "src"),
        Output("img-default", "src"),
        Output("image-display-grid", "style"),
        Output("modified-card", "style"),
        Output("default-card", "style"),
        Output("rgbmix-code", "children"),
        Output("balance-metrics", "children"),
        Output("render-timing", "children"),
        Output("auto-balance-desc", "children"),
        Output("global-h-val", "children"),
        Output("global-c-val", "children"),
        Output("global-l-val", "children"),
        Output("wb-0-h-val", "children"),
        Output("wb-0-c-val", "children"),
        Output("wb-0-l-val", "children"),
        Output("wb-1-h-val", "children"),
        Output("wb-1-c-val", "children"),
        Output("wb-1-l-val", "children"),
        Output("wb-2-h-val", "children"),
        Output("wb-2-c-val", "children"),
        Output("wb-2-l-val", "children"),
        Output("global-swatch", "style"),
        Output("wb-0-swatch", "style"),
        Output("wb-1-swatch", "style"),
        Output("wb-2-swatch", "style"),
        Output("wheel-pin-0", "style"),
        Output("wheel-pin-1", "style"),
        Output("wheel-pin-2", "style"),
        Output("global-h-wrap", "style"),
        Output("global-c-wrap", "style"),
        Output("global-l-wrap", "style"),
        Output("wb-0-h-wrap", "style"),
        Output("wb-0-c-wrap", "style"),
        Output("wb-0-l-wrap", "style"),
        Output("wb-1-h-wrap", "style"),
        Output("wb-1-c-wrap", "style"),
        Output("wb-1-l-wrap", "style"),
        Output("wb-2-h-wrap", "style"),
        Output("wb-2-c-wrap", "style"),
        Output("wb-2-l-wrap", "style"),
        Output("mixer-state-store", "data"),
        Input("global-h", "value"),
        Input("global-c", "value"),
        Input("global-l", "value"),
        Input("wb-0-h", "value"),
        Input("wb-0-c", "value"),
        Input("wb-0-l", "value"),
        Input("wb-1-h", "value"),
        Input("wb-1-c", "value"),
        Input("wb-1-l", "value"),
        Input("wb-2-h", "value"),
        Input("wb-2-c", "value"),
        Input("wb-2-l", "value"),
        Input("btn-reset", "n_clicks"),
        Input("btn-reset-global", "n_clicks"),
        Input("btn-auto-gray-world", "n_clicks"),
        Input("btn-auto-warm-cool", "n_clicks"),
        Input("btn-auto-equal-salience", "n_clicks"),
        Input("btn-auto-triadic", "n_clicks"),
        Input("btn-auto-white-patch", "n_clicks"),
        Input("btn-auto-gray-edge", "n_clicks"),
        Input("btn-auto-decorrelation", "n_clicks"),
        Input("view-select", "value"),
        Input("compare-mode", "value"),
        State("mixer-state-store", "data"),
    )
    def master_update(
        gh,
        gc,
        gl,
        w0h,
        w0c,
        w0l,
        w1h,
        w1c,
        w1l,
        w2h,
        w2c,
        w2l,
        _n_reset,
        _n_reset_global,
        _n_gw,
        _n_wc,
        _n_es,
        _n_tr,
        _n_wp,
        _n_ge,
        _n_dcs,
        view_name,
        compare_mode,
        store_data,
    ):
        t0 = time.perf_counter()
        trigger = ctx.triggered_id

        anchor_wb = copy.deepcopy(store_data["anchor_wb"])
        last_synced_wb = copy.deepcopy(store_data["last_synced_wb"])

        current_wb_inputs = [
            {"H": round(float(w0h), 1), "C": round(float(w0c), 3), "L": round(float(w0l), 3)},
            {"H": round(float(w1h), 1), "C": round(float(w1c), 3), "L": round(float(w1l), 3)},
            {"H": round(float(w2h), 1), "C": round(float(w2c), 3), "L": round(float(w2l), 3)},
        ]

        if isinstance(trigger, str) and trigger.startswith("wb-") and current_wb_inputs == last_synced_wb:
            return (no_update,) * 53

        preset_map = {
            "btn-reset": "default",
            "btn-auto-gray-world": "gray_world",
            "btn-auto-warm-cool": "warm_cool",
            "btn-auto-equal-salience": "equal_salience",
            "btn-auto-triadic": "triadic_120",
            "btn-auto-white-patch": "white_patch",
            "btn-auto-gray-edge": "gray_edge",
            "btn-auto-decorrelation": "decorrelation",
        }

        desc_out = no_update
        update_wb_sliders = False
        update_global_sliders = False

        if trigger in preset_map:
            preset = cache.presets[preset_map[trigger]]
            anchor_wb = apply_global_to_anchor(preset["wb_lch"], 0.0, 1.0, 1.0)
            wb_lch = copy.deepcopy(anchor_wb)
            last_synced_wb = copy.deepcopy(wb_lch)
            gh, gc, gl = 0.0, 1.0, 1.0
            update_wb_sliders = True
            update_global_sliders = True
            desc_out = preset["desc"]
        elif trigger == "btn-reset-global":
            wb_lch = apply_global_to_anchor(anchor_wb, 0.0, 1.0, 1.0)
            last_synced_wb = copy.deepcopy(wb_lch)
            gh, gc, gl = 0.0, 1.0, 1.0
            update_wb_sliders = True
            update_global_sliders = True
        elif trigger in {"global-h", "global-c", "global-l"}:
            wb_lch = apply_global_to_anchor(anchor_wb, float(gh), float(gc), float(gl))
            last_synced_wb = copy.deepcopy(wb_lch)
            update_wb_sliders = True
        elif isinstance(trigger, str) and trigger.startswith("wb-"):
            wb_lch = current_wb_inputs
            for i in range(3):
                if abs(wb_lch[i]["H"] - last_synced_wb[i]["H"]) > 1e-3:
                    anchor_wb[i]["H"] = float((wb_lch[i]["H"] - float(gh)) % 360.0)
                if abs(wb_lch[i]["C"] - last_synced_wb[i]["C"]) > 1e-4 and float(gc) > 1e-4:
                    anchor_wb[i]["C"] = float(wb_lch[i]["C"] / float(gc))
                if abs(wb_lch[i]["L"] - last_synced_wb[i]["L"]) > 1e-4 and abs(float(gl)) > 1e-4:
                    anchor_wb[i]["L"] = float(wb_lch[i]["L"] / float(gl))
            last_synced_wb = copy.deepcopy(wb_lch)
        else:
            wb_lch = current_wb_inputs

        rgbmix = lch_to_rgbmix(wb_lch)
        mod_uri = cache.render_uri(view_name, rgbmix)
        def_uri = cache.default_uris[view_name]
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        base_mod_style = {
            "backgroundColor": "#090d16",
            "border": "1px solid #38bdf8",
            "borderRadius": "8px",
            "padding": "10px",
            "textAlign": "center",
        }
        base_def_style = {
            "backgroundColor": "#090d16",
            "border": "1px solid #334155",
            "borderRadius": "8px",
            "padding": "10px",
            "textAlign": "center",
        }

        if compare_mode == "side_by_side":
            grid_style = {"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"}
        elif compare_mode == "single":
            grid_style = {"display": "grid", "gridTemplateColumns": "1fr", "gap": "16px"}
            base_def_style = {**base_def_style, "display": "none"}
        else:
            grid_style = {"display": "grid", "gridTemplateColumns": "1fr", "gap": "16px"}
            base_mod_style = {**base_mod_style, "display": "none"}

        code_str = (
            "self.rgbmix = np.array([\n"
            f"    [{rgbmix[0, 0]:7.4f}, {rgbmix[0, 1]:7.4f}, {rgbmix[0, 2]:7.4f}],\n"
            f"    [{rgbmix[1, 0]:7.4f}, {rgbmix[1, 1]:7.4f}, {rgbmix[1, 2]:7.4f}],\n"
            f"    [{rgbmix[2, 0]:7.4f}, {rgbmix[2, 1]:7.4f}, {rgbmix[2, 2]:7.4f}],\n"
            "])"
        )

        mets = mix_metrics(rgbmix, cache.sample_pixels)
        p995 = mets["p995"]
        metrics_str = (
            f"Disk Mean Cast (a, b): ({mets['a']:+.3f}, {mets['b']:+.3f})  |  "
            f"Clipped px: {mets['clip_pct']:.1f}%  |  "
            f"p99.5 RGB: ({p995[0]:.2f}, {p995[1]:.2f}, {p995[2]:.2f})"
        )

        # Option 1 Global Hue track: centred on the dominant hue of the image rendered with the
        # stationary anchor (ΔH = 0), so the color under the cursor is the Sun's resulting hue.
        grad_gh, sun_h = global_hue_track(anchor_wb, cache.track_pixels, float(gh), float(gc), float(gl))
        _, grad_gc, grad_gl, g_swatch = make_triplet_gradients_and_swatch(
            0.0, GLOBAL_H_MIN, GLOBAL_H_MAX,
            float(gc), GLOBAL_C_MIN, GLOBAL_C_MAX, 1.0,
            float(gl), GLOBAL_L_MIN, GLOBAL_L_MAX, 1.0,
            h_offset=sun_h,
        )
        gh_vivid_cursor = _rgb_vec_to_css(vivid_rgb_at_hue(sun_h))

        # Compute Option 2 Per-Waveband track gradients, thumb colors, swatches, and wheel pins
        wb_wraps = []
        swatches = []
        wheel_pins = []
        swatch_base = {
            "width": "24px",
            "height": "24px",
            "borderRadius": "50%",
            "border": "2px solid #cbd5e1",
        }
        for i in range(3):
            wh, wc, wl = wb_lch[i]["H"], wb_lch[i]["C"], wb_lch[i]["L"]
            b_h, b_c, b_l, w_sw = make_triplet_gradients_and_swatch(
                wh, WB_H_MIN, WB_H_MAX,
                wc, WB_C_MIN, WB_C_MAX, 0.35,
                wl, WB_L_MIN, WB_L_MAX, 0.35,
                h_offset=0.0,
            )
            wh_viv = _rgb_vec_to_css(vivid_rgb_at_hue(wh))
            wb_wraps.extend(
                [
                    _slider_wrap_style(b_h, wh_viv),
                    _slider_wrap_style(b_c, w_sw),
                    _slider_wrap_style(b_l, w_sw),
                ]
            )
            swatches.append({**swatch_base, "backgroundColor": w_sw})
            wheel_pins.append(_wheel_pin_style(wh, wc, w_sw))

        wb_slider_outs = (
            (
                wb_lch[0]["H"], wb_lch[0]["C"], wb_lch[0]["L"],
                wb_lch[1]["H"], wb_lch[1]["C"], wb_lch[1]["L"],
                wb_lch[2]["H"], wb_lch[2]["C"], wb_lch[2]["L"],
            )
            if update_wb_sliders
            else (no_update,) * 9
        )
        global_slider_outs = (0.0, 1.0, 1.0) if update_global_sliders else (no_update,) * 3

        return (
            *wb_slider_outs,
            *global_slider_outs,
            mod_uri,
            def_uri,
            grid_style,
            base_mod_style,
            base_def_style,
            code_str,
            metrics_str,
            f"Render: {elapsed_ms:.1f} ms",
            desc_out,
            f"{float(gh):+.1f}° → Sun hue {sun_h:.0f}°",
            f"{float(gc):.2f}×",
            f"{float(gl):.2f}×",
            f"{wb_lch[0]['H']:.1f}°",
            f"{wb_lch[0]['C']:.3f}",
            f"{wb_lch[0]['L']:.3f}",
            f"{wb_lch[1]['H']:.1f}°",
            f"{wb_lch[1]['C']:.3f}",
            f"{wb_lch[1]['L']:.3f}",
            f"{wb_lch[2]['H']:.1f}°",
            f"{wb_lch[2]['C']:.3f}",
            f"{wb_lch[2]['L']:.3f}",
            {
                "width": "22px",
                "height": "22px",
                "borderRadius": "50%",
                "border": "2px solid #f8fafc",
                "backgroundColor": g_swatch,
            },
            swatches[0],
            swatches[1],
            swatches[2],
            wheel_pins[0],
            wheel_pins[1],
            wheel_pins[2],
            _slider_wrap_style(grad_gh, gh_vivid_cursor),
            _slider_wrap_style(grad_gc, g_swatch),
            _slider_wrap_style(grad_gl, g_swatch),
            *wb_wraps,
            {"anchor_wb": anchor_wb, "last_synced_wb": last_synced_wb},
        )

    @app.callback(
        Output("save-status", "children"),
        Input("btn-save-full", "n_clicks"),
        State("wb-0-h", "value"),
        State("wb-0-c", "value"),
        State("wb-0-l", "value"),
        State("wb-1-h", "value"),
        State("wb-1-c", "value"),
        State("wb-1-l", "value"),
        State("wb-2-h", "value"),
        State("wb-2-c", "value"),
        State("wb-2-l", "value"),
        prevent_initial_call=True,
    )
    def save_full_res(
        _n_clicks,
        w0h,
        w0c,
        w0l,
        w1h,
        w1c,
        w1l,
        w2h,
        w2c,
        w2l,
    ):
        wb_lch = [
            {"H": float(w0h), "C": float(w0c), "L": float(w0l)},
            {"H": float(w1h), "C": float(w1c), "L": float(w1l)},
            {"H": float(w2h), "C": float(w2c), "L": float(w2l)},
        ]
        rgbmix = lch_to_rgbmix(wb_lch)

        mixer = visualization.RGBMixer(
            data_files=[[f] for f in cache.jp2_files],
            jp2=True,
            outputdir=output_dir,
        )
        mixer.set_jp2_default()
        mixer.rgbmix = rgbmix
        # Write the plain RGB-mixed output (exactly what the preview shows). The default lab=(1,1,1)
        # path is not an identity: it min/max-stretches the image and round-trips it through an
        # 8-bit CIELab quantization, so the saved file would differ from the on-screen preview.
        mixer.lab = None
        mixer.filename_lab = None
        mixer.filename_rgb = "im_custom_lch_jp2"
        mixer.process_rgb(0)
        out_path = os.path.join(output_dir, "im_custom_lch_jp2_0000.jpeg")
        return f"Saved full 4096×4096 image to: {out_path}"

    return app


def find_sample_jp2_files(samples_dir: str) -> list[str]:
    files = []
    for wavel in ["304", "171", "193"]:
        matches = sorted(glob.glob(os.path.join(samples_dir, f"*.{wavel}*.jp2")))
        if not matches:
            raise FileNotFoundError(f"No .jp2 file found for wavelength {wavel} in {samples_dir}")
        files.append(matches[0])
    return files


def main():
    parser = argparse.ArgumentParser(description="Launch the interactive AIA JP2 RGB / LCh mixer Dash app.")
    parser.add_argument(
        "--samples-dir",
        default=os.path.abspath("samples"),
        help="Directory containing sample .304., .171., and .193. jp2 files (default: ./samples)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8050, help="Port number (default: 8050)")
    parser.add_argument("--debug", action="store_true", help="Run Dash in debug mode")
    args = parser.parse_args()

    jp2_files = find_sample_jp2_files(args.samples_dir)
    print(f"Loading JP2 files: {jp2_files}")
    cache = JP2ImageCache(jp2_files)
    app = create_dash_app(cache, output_dir=os.path.abspath(args.samples_dir))
    print(f"Starting Dash server at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
