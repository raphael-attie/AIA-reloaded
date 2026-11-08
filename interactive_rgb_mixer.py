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
import json
import re
import threading
from collections import OrderedDict
from typing import NamedTuple
import os
import time
import cv2
import numpy as np
from dash import Dash, Input, Output, State, ctx, dcc, html, no_update

import visualization

# sunpy's standard SDO/AIA colormaps (sunpy.visualization.colormaps; needs `sunpy[visualization]`).
from sunpy.visualization import colormaps as sunpy_cm

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

# Input gamma slider bounds: each input channel x in [0, 1] is mapped to x**(1/gamma) before
# mixing, as visualization.scale_rgb does with RGBMixer.gamma_rgb. gamma > 1 lifts faint loops a
# lot while bright structures (x near 1) barely change; gamma < 1 deepens faint areas.
GAMMA_MIN, GAMMA_MAX = 0.5, 3.0
# Sigmoidal contrast strength (visualization.sigmoid_contrast) applied after gamma; 0 = off.
CONTRAST_MIN, CONTRAST_MAX = 0.0, 12.0

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
    visible: np.ndarray | None = None,
) -> tuple[str, float]:
    """
    Build the Global Hue Shift colorbar from the *rendered* Sun: each stop at shift ΔH shows the
    vivid color of the dominant hue the image actually takes at that shift (including gamut
    clipping, which bends a pure rotation for near-neutral palettes). Returns the CSS gradient
    and the rendered dominant hue at the current shift (used for the thumb and C/L tracks).
    `visible` (3 x 0/1) drops hidden input channels exactly as the preview does.
    """
    col = np.ones(3) if visible is None else np.asarray(visible, dtype=np.float64)
    stops = np.linspace(GLOBAL_H_MIN, GLOBAL_H_MAX, n_stops)
    hues = [dominant_hue(lch_to_rgbmix(apply_global_to_anchor(anchor_wb, float(g), gc, gl)) * col, pixels) for g in stops]
    grad = f"linear-gradient(to right, {', '.join(_rgb_vec_to_css(vivid_rgb_at_hue(h)) for h in hues)})"
    h_cur = dominant_hue(lch_to_rgbmix(apply_global_to_anchor(anchor_wb, float(gh_cur), gc, gl)) * col, pixels)
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


def compute_auto_balance_presets(pixels: np.ndarray) -> dict[str, dict]:
    """
    Compute the Auto-Balance presets (Warm-Cool Harmony, White-Patch) in the opponent LCh space
    from real on-disk pixel samples (N, 3) of the selected (R, G, B) input channels.
    """
    means = pixels.mean(axis=0)

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

    # Split-Complementary Warm-Cool Harmony
    h_wc = [24.0, 106.0, 245.0]
    c_wc = _solve_chroma_for_cast(h_wc, means, (0.022, 0.035), 1.02)
    wb_wc = _normalize([{"H": h_wc[i], "C": float(c_wc[i]), "L": [0.26, 0.74, 0.12][i]} for i in range(3)])

    # White-Patch (Max-RGB) von Kries correction of the default palette: per-output-channel gains
    #    so that the brightest structures (99.5th pct of each output channel) render neutral white.
    raw_def = pixels @ DEFAULT_JP2_RGBMIX.T
    p_def = np.maximum(np.percentile(raw_def, 99.5, axis=0), 1e-3)
    wb_wp = _normalize(rgbmix_to_waveband_lch(np.diag(0.98 / p_def) @ DEFAULT_JP2_RGBMIX))

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
        "warm_cool": {
            "label": "Warm–Cool Harmony",
            "wb_lch": wb_wc,
            "desc": "Split-Complementary Thermal Harmony: Shifts 171 Å loops to pure Amber-Gold (106°) and 304 Å to Vermilion (24°), opposed by complementary Cerulean-Blue (245°) for the 1.5 MK corona.",
        },
        "white_patch": {
            "label": "White-Patch (Max-RGB)",
            "wb_lch": wb_wp,
            "desc": "Von Kries White-Patch correction of the default palette: per-output-channel gains make the brightest 0.5% of structures (flares, AR cores, limb brightenings) render neutral white instead of tinted, while keeping the default hue ordering.",
        },
    }


# -----------------------------------------------------------------------------
# Option 1 + Option 2 preset files (JSON), saved under <project>/presets/
# -----------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PRESETS_DIR = os.path.join(PROJECT_DIR, "presets")
PRESET_FORMAT = "aia-reloaded-lch-preset"


def preset_path(name: str) -> str:
    """Validate a user-typed preset filename and return its path inside PRESETS_DIR (.json enforced)."""
    name = (name or "").strip()
    if not name.lower().endswith(".json"):
        name += ".json"
    stem = name[:-5]
    if not stem or not re.fullmatch(r"[A-Za-z0-9._ -]+", stem) or stem.startswith("."):
        raise ValueError("use letters, digits, spaces, '.', '_' or '-' only (no folders)")
    return os.path.join(PRESETS_DIR, name)


def build_preset(anchor_wb, wb_lch, gh, gc, gl, triplet, rgbmix) -> dict:
    """Serialize Option 1 (global) and Option 2 (per-waveband) settings. `anchor_wb` is the per-waveband
    state before the global adjustment, so loading restores both slider sets and their coupling."""
    def _wb(w):
        return [{"H": round(float(x["H"]), 4), "C": round(float(x["C"]), 5), "L": round(float(x["L"]), 5)} for x in w]

    return {
        "format": PRESET_FORMAT,
        "version": 1,
        "option1_global": {"hue_shift_deg": float(gh), "chroma_scale": float(gc), "luminance_scale": float(gl)},
        "option2_wavebands": _wb(wb_lch),
        "anchor_wavebands": _wb(anchor_wb),
        "info": {
            "wavelengths_rgb": list(triplet),
            "rgbmix": np.round(rgbmix, 6).tolist(),
            "saved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }


def parse_preset(obj: dict) -> tuple[list[dict[str, float]], tuple[float, float, float]]:
    """Return (anchor_wb, (gh, gc, gl)) from a preset dict; values are clipped to the slider ranges."""
    if not isinstance(obj, dict) or obj.get("format") != PRESET_FORMAT:
        raise ValueError(f"not an {PRESET_FORMAT} file")
    g = obj["option1_global"]
    gh = float(np.clip(float(g["hue_shift_deg"]), GLOBAL_H_MIN, GLOBAL_H_MAX))
    gc = float(np.clip(float(g["chroma_scale"]), GLOBAL_C_MIN, GLOBAL_C_MAX))
    gl = float(np.clip(float(g["luminance_scale"]), GLOBAL_L_MIN, GLOBAL_L_MAX))
    src = obj.get("anchor_wavebands")
    if src is None:  # derive the anchor from the effective per-waveband values
        src = [
            {"H": (w["H"] - gh) % 360.0, "C": w["C"] / gc if gc > 1e-6 else w["C"], "L": w["L"] / gl if abs(gl) > 1e-6 else w["L"]}
            for w in obj["option2_wavebands"]
        ]
    if len(src) != 3:
        raise ValueError("expected 3 wavebands")
    anchor = [
        {
            "H": float(w["H"]) % 360.0,
            "C": float(np.clip(float(w["C"]), WB_C_MIN, WB_C_MAX)),
            "L": float(np.clip(float(w["L"]), WB_L_MIN, WB_L_MAX)),
        }
        for w in src
    ]
    return anchor, (gh, gc, gl)


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

    bg = np.array([17.0 / 255.0, 17.0 / 255.0, 17.0 / 255.0])  # wheel corner color = panel background (#111111)
    alpha = np.clip((1.0 - r) * (size / 3.0), 0.0, 1.0)[..., None]
    comp = rgb * alpha + bg * (1.0 - alpha)
    bgr_u8 = (comp[:, :, ::-1] * 255.0).astype(np.uint8)
    ok, buf = cv2.imencode(".png", bgr_u8)
    if not ok:
        raise RuntimeError("Failed to encode Oklch color wheel PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.tobytes()).decode('ascii')}"


DEFAULT_WB_LCH = rgbmix_to_waveband_lch(DEFAULT_JP2_RGBMIX)

# Output channel of each rgbmix column (column k of rgbmix takes the wavelength assigned to slot k).
RGB_SLOTS = [
    {"rgb": "R", "name": "Red", "color": "#f87171"},
    {"rgb": "G", "name": "Green", "color": "#4ade80"},
    {"rgb": "B", "name": "Blue", "color": "#60a5fa"},
]
DEFAULT_TRIPLET = ("304", "171", "193")
AIA_WAVELENGTHS = ["94", "131", "171", "193", "211", "304", "335", "1600", "1700", "4500"]
AIA_ROLES = {
    "94": "Flaring Corona (Fe XVIII, ~6 MK)",
    "131": "Flare Plasma & Transition Region (Fe VIII / XXI, ~0.4 & 10 MK)",
    "171": "Quiet Corona & Coronal Loops (Fe IX, ~0.6 MK)",
    "193": "Corona & Hot Flare Plasma (Fe XII / XXIV, ~1.2 & 20 MK)",
    "211": "Active-Region Corona (Fe XIV, ~2 MK)",
    "304": "Chromosphere & Prominences (He II, ~50 kK)",
    "335": "Active-Region Corona (Fe XVI, ~2.5 MK)",
    "1600": "Upper Photosphere & Transition Region (C IV)",
    "1700": "Temperature Minimum & Photosphere (UV continuum)",
    "4500": "Photosphere (white light)",
}


def _svg_uri(svg: str) -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")


# Eye icons for the per-channel visibility toggles (open = channel shown, closed = hidden).
# Placeholder thumbnail for an empty R/G/B slot.
EMPTY_CHANNEL_URI = _svg_uri(
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><rect width="100" height="100" fill="#0b1220"/>'
    '<circle cx="50" cy="50" r="38" fill="none" stroke="#2a2a2a" stroke-width="2" stroke-dasharray="5 5"/>'
    '<text x="50" y="55" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#525252">empty</text></svg>'
)
EYE_OPEN_URI = _svg_uri(
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#e5e5e5" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M2 12s3.6-6.5 10-6.5S22 12 22 12s-3.6 6.5-10 6.5S2 12 2 12z"/><circle cx="12" cy="12" r="3"/></svg>'
)
EYE_CLOSED_URI = _svg_uri(
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#a3a3a3" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M2 10c2.7 3.6 6.1 5.4 10 5.4s7.3-1.8 10-5.4"/>'
    '<path d="M5.5 13.6 4 16M9.5 15.2 9 18M14.5 15.2 15 18M18.5 13.6 20 16"/></svg>'
)


def channel_visible(n_clicks) -> bool:
    """Eye toggle state: every click flips it; starts visible."""
    return not (int(n_clicks or 0) % 2)


def visibility_mask(n0, n1, n2) -> np.ndarray:
    """Column mask for rgbmix: 1 for shown input channels, 0 for hidden ones."""
    return np.array([1.0 if channel_visible(n) else 0.0 for n in (n0, n1, n2)])


def aia_cmap_name(wavel: str) -> str:
    return f"sdoaia{wavel}"


def sunpy_lut_bgr_u8(cmap_name: str) -> np.ndarray:
    """256-entry BGR uint8 lookup table sampled from a sunpy colormap (for cv2 encoding)."""
    cmap = sunpy_cm.cmlist.get(cmap_name)
    if cmap is None:  # no sunpy colormap for this channel: plain grayscale
        return np.repeat(np.arange(256, dtype=np.uint8)[:, None], 3, axis=1)
    rgba = cmap(np.linspace(0.0, 1.0, 256))
    return np.round(rgba[:, 2::-1] * 255.0).astype(np.uint8)
COLOR_WHEEL_URI = generate_color_wheel_uri(148)


VIEW_NAMES = ["full_1024", "full_1536", "crop_bottom_half", "zoom_prominence", "zoom_active_region"]


def _make_views(full: np.ndarray) -> dict[str, np.ndarray]:
    """Preview views of one full-resolution 4096x4096 image (2D) or stack (3D)."""
    h, w = full.shape[:2]
    return {
        "full_1024": cv2.resize(full, (1024, 1024), interpolation=cv2.INTER_AREA),
        "full_1536": cv2.resize(full, (1536, 1536), interpolation=cv2.INTER_AREA),
        "crop_bottom_half": cv2.resize(
            full[1935 : min(h, 1935 + 2160), 128 : min(w, 128 + 3840)],
            (1280, 720),
            interpolation=cv2.INTER_AREA,
        ),
        "zoom_prominence": full[2300:3324, 200:1224].copy(),
        "zoom_active_region": full[1050:2074, 2800:3824].copy(),
    }


def _encode_uri(bgr_u8: np.ndarray, jpeg_quality: int = 92) -> str:
    """Encode a BGR uint8 image as a base64 JPEG data URI."""
    ok, buf = cv2.imencode(".jpg", bgr_u8, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    if not ok:
        raise RuntimeError("Failed to encode preview image as JPEG")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


# Image pixels per full-resolution (4096 px) pixel for each preview view, so spatial filters (sharpening
# radius, MGN scales) act on the same physical sizes in previews as in the full-resolution output.
VIEW_SCALE = {
    "full_1024": 1024 / 4096,
    "full_1536": 1536 / 4096,
    "crop_bottom_half": 1280 / 3840,
    "zoom_prominence": 1.0,
    "zoom_active_region": 1.0,
}


class ToneParams(NamedTuple):
    """Per-input-channel processing before mixing: tone curve, then MGN structure, then unsharp mask."""

    gamma: float = 1.0
    contrast: float = 0.0
    structure: float = 0.0
    amount: float = 0.0
    radius: float = 1.0
    detail: float = 25.0
    masking: float = 0.0

    @property
    def spatial(self) -> bool:
        return self.structure > 1e-6 or self.amount > 1e-6

    @property
    def sharpen(self) -> tuple[float, float, float, float]:
        return (self.amount, self.radius, self.detail, self.masking)

    def key(self) -> tuple:
        return (round(self.gamma, 3), round(self.contrast, 2), round(self.structure, 3),
                round(self.amount, 3), round(self.radius, 2), round(self.detail, 1), round(self.masking, 1))


class TripletData:
    """Mixing stacks, on-disk statistics and Auto-Balance presets for one (R, G, B) wavelength triplet."""

    def __init__(self, triplet: tuple, channel_views: dict[str, dict[str, np.ndarray]], jp2_files: list):
        """`triplet` holds a wavelength per R/G/B slot, or None for an empty slot (an all-zero input)."""
        self.triplet = triplet
        self.jp2_files = jp2_files
        any_views = channel_views[next(w for w in triplet if w is not None)]
        self.views: dict[str, np.ndarray] = {
            view_name: np.stack(
                [channel_views[w][view_name] if w is not None else np.zeros_like(any_views[view_name]) for w in triplet],
                axis=-1,
            )
            for view_name in VIEW_NAMES
        }

        self._preview = self.views["full_1024"].astype(np.float64)
        self._mask = np.max(self._preview, axis=-1) > 0.05
        self.disk_pixels = self._preview[self._mask]
        # Small fixed subsample for per-callback diagnostics (dominant hue, cast, clipping).
        stride = max(1, len(self.disk_pixels) // 40000)
        self._stride = stride
        self.sample_pixels = self.disk_pixels[::stride].copy()
        # Even smaller subsample for the 26 renders behind the Global Hue colorbar.
        self.track_pixels = self.sample_pixels[::4].copy()
        self._tone_views: dict[tuple, np.ndarray] = {}
        self._mgn_views: dict[tuple, np.ndarray] = {}
        self._edge_samples: dict[tuple, list] = {}
        self._presets: dict[tuple, dict[str, dict]] = {}
        self._mids: dict[tuple, np.ndarray] = {}

    @property
    def presets(self) -> dict[str, dict]:
        """Presets on the un-toned inputs (computed on first use, so frame changes stay fast)."""
        return self.presets_for(ToneParams())

    def contrast_mid(self, gamma: float, contrast: float) -> np.ndarray:
        """Per-channel pivots of the sigmoidal contrast, chosen so each channel's on-disk mean level
        is unchanged (visualization.mean_preserving_midpoint). The mix being linear, contrast then
        changes neither the mean brightness nor the color balance, whatever the Option 1/2 sliders.
        (A single shared pivot shifted each channel's level differently, so the brightness and
        color change depended on the mixing weights.)"""
        key = (round(gamma, 3), round(contrast, 2))
        if key not in self._mids:
            x = self.sample_pixels ** (1.0 / key[0])
            self._mids[key] = np.array([visualization.mean_preserving_midpoint(x[:, k], key[1]) for k in range(3)])
        return self._mids[key]

    def tone(self, x: np.ndarray, gamma: float, contrast: float) -> np.ndarray:
        """Input tone curve: gamma x**(1/gamma), then per-channel sigmoidal contrast (= visualization.scale_rgb)."""
        if abs(gamma - 1.0) > 1e-6:
            x = np.power(x, x.dtype.type(1.0 / gamma))
        if contrast <= 1e-6:
            return x
        mids = self.contrast_mid(gamma, contrast)
        return np.stack([visualization.sigmoid_contrast(x[..., k], contrast, mids[k]) for k in range(3)], axis=-1)

    def _mgn_view(self, view_name: str, tp: ToneParams, toned: np.ndarray) -> np.ndarray:
        """MGN local component of the tone-mapped view (cached: dragging the Structure or sharpening
        sliders reuses it)."""
        key = (view_name, round(tp.gamma, 3), round(tp.contrast, 2))
        if key not in self._mgn_views:
            if len(self._mgn_views) >= 2:
                self._mgn_views.pop(next(iter(self._mgn_views)))
            scale = VIEW_SCALE.get(view_name, 1.0)
            self._mgn_views[key] = np.stack([visualization.mgn_detail(toned[..., k], scale) for k in range(3)], axis=-1)
        return self._mgn_views[key]

    def tone_view(self, view_name: str, tp: ToneParams) -> np.ndarray:
        """Processed input stack for a view: tone curve, MGN structure, unsharp mask (= visualization.scale_rgb,
        with spatial sizes scaled to the view). Only the most recent results are kept."""
        if abs(tp.gamma - 1.0) < 1e-6 and tp.contrast <= 1e-6 and not tp.spatial:
            return self.views[view_name]
        key = (view_name, *tp.key())
        if key not in self._tone_views:
            if len(self._tone_views) >= 2:
                self._tone_views.pop(next(iter(self._tone_views)))
            out = self.tone(self.views[view_name], tp.gamma, tp.contrast)
            if tp.structure > 1e-6:
                out = np.clip(out + np.float32(tp.structure) * self._mgn_view(view_name, tp, out), 0.0, 1.0)
            if tp.amount > 1e-6:
                scale = VIEW_SCALE.get(view_name, 1.0)
                thrs = self.mask_thresholds(tp) if tp.masking > 0 else [None] * 3
                out = np.stack(
                    [
                        visualization.unsharp_mask(out[..., k], tp.amount, tp.radius, tp.detail, tp.masking, scale, thrs[k])
                        for k in range(3)
                    ],
                    axis=-1,
                )
            self._tone_views[key] = out
        return self._tone_views[key]

    def mask_thresholds(self, tp: ToneParams) -> list:
        """Absolute masking thresholds per channel, from the full-resolution channels after the tone curve
        (MGN structure is not included, to keep this fast). Used by every preview view and by the
        full-resolution save, so Masking selects the same structures everywhere. The full-res edge
        strengths are cached per (gamma, contrast, radius); moving the Masking slider only re-takes a percentile."""
        key = (round(tp.gamma, 3), round(tp.contrast, 2), round(tp.radius, 2))
        if key not in self._edge_samples:
            if len(self._edge_samples) >= 2:
                self._edge_samples.pop(next(iter(self._edge_samples)))
            mids = self.contrast_mid(tp.gamma, tp.contrast)
            samples = []
            for k, f in enumerate(self.jp2_files):
                if f is None:
                    samples.append(None)
                    continue
                x = (visualization.load_jp2(f) / 255.0).astype(np.float32)
                if abs(tp.gamma - 1.0) > 1e-6:
                    x = np.power(x, np.float32(1.0 / tp.gamma))
                x = visualization.sigmoid_contrast(x, tp.contrast, mids[k])
                g = visualization.edge_strength(x, tp.radius, 1.0)
                samples.append(g[::4, ::4].ravel().copy())
            self._edge_samples[key] = samples
        return [
            None if g is None else float(np.percentile(g, min(tp.masking, 99.9)))
            for g in self._edge_samples[key]
        ]

    def tone_pixels(self, tp: ToneParams) -> tuple[np.ndarray, np.ndarray]:
        """(sample, track) on-disk pixel subsamples after processing, for diagnostics and the Global Hue
        colorbar. Spatial filters need neighbors, so they are taken from the processed full-sun view."""
        if not tp.spatial:
            return self.tone(self.sample_pixels, tp.gamma, tp.contrast), self.tone(self.track_pixels, tp.gamma, tp.contrast)
        sample = self.tone_view("full_1024", tp).astype(np.float64)[self._mask][:: self._stride]
        return sample, sample[::4]

    def presets_for(self, tp: ToneParams) -> dict[str, dict]:
        """Auto-Balance presets computed on the processed inputs actually being mixed (cached)."""
        key = tp.key()
        if key not in self._presets:
            if tp.spatial:
                preview = self.tone_view("full_1024", tp).astype(np.float64)
            else:
                preview = self.tone(self._preview, tp.gamma, tp.contrast)
            self._presets[key] = compute_auto_balance_presets(preview[self._mask])
        return self._presets[key]

    def render_uri(
        self,
        view_name: str,
        rgbmix: np.ndarray,
        tp: ToneParams = ToneParams(),
        scalemin: float = 0.0,
        jpeg_quality: int = 92,
    ) -> str:
        """Apply the per-channel processing and the 3x3 rgbmix to the cached float32 stack; return a JPEG data URI."""
        mixed = self.tone_view(view_name, tp) @ rgbmix.astype(np.float32).T
        np.clip(mixed, 0.0, 1.0, out=mixed)
        mixed *= 255.0
        if scalemin > 0:
            mixed = (mixed - scalemin) * (255.0 / (255.0 - scalemin))
            np.clip(mixed, 0.0, 255.0, out=mixed)
        return _encode_uri(mixed[:, :, ::-1].astype(np.uint8), jpeg_quality)


class JP2ImageCache:
    """
    Pre-loads every available 4096x4096 JP2 wavelength once as small preview views, plus each
    wavelength rendered alone in its sunpy SDO/AIA colormap. Mixing stacks, statistics and presets
    for a selected (R, G, B) triplet are built on demand and kept for the most recent triplets.
    """

    MAX_TRIPLETS = 4

    def __init__(self, jp2_by_wavel: dict[str, str]):
        self.jp2_by_wavel = jp2_by_wavel
        self.wavelengths = list(jp2_by_wavel)
        self.channel_views: dict[str, dict[str, np.ndarray]] = {}
        self._channel_uris: dict[tuple[str, str], str] = {}
        for wavel, path in jp2_by_wavel.items():
            full = visualization.load_jp2(path).astype(np.float32) / 255.0
            self.channel_views[wavel] = _make_views(full)
        self._triplets: dict[tuple, TripletData] = {}

    def channel_uri(self, wavel: str, view_name: str) -> str:
        """The wavelength alone in its sunpy SDO/AIA colormap (encoded on first request)."""
        key = (wavel, view_name)
        if key not in self._channel_uris:
            lut = sunpy_lut_bgr_u8(aia_cmap_name(wavel))
            v = self.channel_views[wavel][view_name]
            self._channel_uris[key] = _encode_uri(lut[np.round(np.clip(v, 0.0, 1.0) * 255.0).astype(np.uint8)])
        return self._channel_uris[key]

    def get(self, triplet) -> TripletData:
        triplet = tuple(norm_wavel(w) for w in triplet)
        if all(w is None for w in triplet):
            raise ValueError("at least one R/G/B slot must have a wavelength")
        if triplet not in self._triplets:
            if len(self._triplets) >= self.MAX_TRIPLETS:
                self._triplets.pop(next(iter(self._triplets)))
            files = [self.jp2_by_wavel[w] if w is not None else None for w in triplet]
            self._triplets[triplet] = TripletData(triplet, self.channel_views, files)
        return self._triplets[triplet]


# -----------------------------------------------------------------------------
# Data sources: the bundled samples (one frame) and event folders written by get_data.py
# (<data_dir>/event_*/<wavelength>/*.jp2, one file per frame). Frames are loaded on demand.
# -----------------------------------------------------------------------------
EMPTY_SLOT = "none"  # dropdown value of an empty R/G/B slot
_TIMESTAMP_RE = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{6})")


def norm_wavel(w):
    """Dropdown value -> wavelength string, or None for an empty slot."""
    return None if w in (None, "", EMPTY_SLOT) else str(w)


class DataSource:
    """A set of time-aligned JP2 frames per AIA wavelength."""

    def __init__(self, name: str, label: str, files_by_wavel: dict[str, list[str]], frame_times: list[str], output_dir: str, is_event: bool):
        self.name = name
        self.label = label
        self.files_by_wavel = files_by_wavel
        self.wavelengths = [w for w in AIA_WAVELENGTHS if w in files_by_wavel]
        self.frame_times = frame_times
        self.n_frames = len(frame_times)
        self.output_dir = output_dir
        self.is_event = is_event

    def files_for(self, frame: int) -> dict[str, str]:
        frame = int(np.clip(frame, 0, self.n_frames - 1))
        return {w: self.files_by_wavel[w][frame] for w in self.wavelengths}

    def frame_label(self, frame: int) -> str:
        t = self.frame_times[int(np.clip(frame, 0, self.n_frames - 1))]
        m = re.fullmatch(r"(\d{4}-\d{2}-\d{2})T(\d{2})(\d{2})(\d{2})", t)
        return f"{m.group(1)} {m.group(2)}:{m.group(3)}:{m.group(4)} UTC" if m else t


def samples_source(samples_dir: str) -> DataSource:
    files = find_sample_jp2_files(samples_dir)
    return DataSource(
        "samples", f"Samples ({os.path.basename(os.path.normpath(samples_dir))}/)",
        {w: [f] for w, f in files.items()}, ["sample"], samples_dir, is_event=False,
    )


def event_source(event_dir: str) -> DataSource | None:
    """Event folder from get_data.py: one sub-folder per wavelength. Frames are the timestamps
    (parsed from the file names) present in every channel; if file names carry no timestamp,
    frames are aligned by sorted order."""
    per_wavel = {}
    for w in AIA_WAVELENGTHS:
        files = sorted(glob.glob(os.path.join(event_dir, w, "*.jp2")))
        if files:
            per_wavel[w] = files
    if not per_wavel:
        return None
    stamped = {w: {m.group(1): f for f in fs if (m := _TIMESTAMP_RE.search(os.path.basename(f)))} for w, fs in per_wavel.items()}
    if all(len(stamped[w]) == len(per_wavel[w]) for w in per_wavel):
        times = sorted(set.intersection(*(set(d) for d in stamped.values())))
        files_by_wavel = {w: [stamped[w][t] for t in times] for w in per_wavel}
    else:
        n = min(len(fs) for fs in per_wavel.values())
        times = [f"frame {i + 1}" for i in range(n)]
        files_by_wavel = {w: fs[:n] for w, fs in per_wavel.items()}
    if not times:
        return None
    name = os.path.basename(os.path.normpath(event_dir))
    label = f"{name}  ({', '.join(files_by_wavel)} Å · {len(times)} frames)"
    return DataSource(name, label, files_by_wavel, times, os.path.join(event_dir, "mixer_exports"), is_event=True)


def discover_sources(samples_dir: str, data_dir: str | None) -> "OrderedDict[str, DataSource]":
    sources: OrderedDict[str, DataSource] = OrderedDict()
    sources["samples"] = samples_source(samples_dir)
    if data_dir and os.path.isdir(data_dir):
        for event_dir in sorted(glob.glob(os.path.join(data_dir, "event_*"))):
            src = event_source(event_dir) if os.path.isdir(event_dir) else None
            if src is not None:
                sources[src.name] = src
    return sources


class FrameStore:
    """Loads (source, frame) on demand into a JP2ImageCache; keeps the most recent few."""

    MAX_FRAMES = 3

    def __init__(self, sources: "OrderedDict[str, DataSource]"):
        self.sources = sources
        self._frames: OrderedDict[tuple[str, int], JP2ImageCache] = OrderedDict()
        self._lock = threading.Lock()  # concurrent callbacks must not decode the same frame twice

    def source(self, name: str) -> DataSource:
        return self.sources.get(name) or next(iter(self.sources.values()))

    def get(self, name: str, frame) -> JP2ImageCache:
        src = self.source(name)
        key = (src.name, int(np.clip(int(frame or 0), 0, src.n_frames - 1)))
        with self._lock:
            if key in self._frames:
                self._frames.move_to_end(key)
            else:
                if len(self._frames) >= self.MAX_FRAMES:
                    self._frames.popitem(last=False)
                self._frames[key] = JP2ImageCache(src.files_for(key[1]))
            return self._frames[key]


def gamma_track_gradient(n_stops: int = 16, faint: float = 0.2) -> str:
    """Gamma track: brightness a faint input (x = 0.2) takes at each gamma, so the bar shows the lift."""
    stops = []
    for g in np.linspace(GAMMA_MIN, GAMMA_MAX, n_stops):
        v = int(round(255.0 * faint ** (1.0 / g)))
        stops.append(f"rgb({v}, {v}, {v})")
    return f"linear-gradient(to right, {', '.join(stops)})"


def contrast_track_gradient(n_stops: int = 16) -> str:
    """Contrast track: a gray ramp passed through the sigmoid at mid strength, so the bar shows
    how mid-tones get stretched while black and white stay put."""
    x = np.linspace(0.0, 1.0, n_stops)
    y = visualization.sigmoid_contrast(x, 0.5 * CONTRAST_MAX, 0.5)
    return f"linear-gradient(to right, {', '.join(f'rgb({int(round(v * 255))}, {int(round(v * 255))}, {int(round(v * 255))})' for v in y)})"


def _format_dark_marks(marks: dict) -> dict:
    """Ensure slider mark labels have light gray contrast on the dark background."""
    return {
        k: {"label": str(v), "style": {"color": "#d4d4d4", "fontSize": "11px"}}
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
                    html.Span(label, id=f"{slider_id}-label", style={"fontSize": "13px", "fontWeight": "600", "color": "#e5e5e5"}),
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
        "color": "#000000",
        "fontWeight": "800",
        "fontSize": "10px",
        "padding": "2px 5px",
        "borderRadius": "10px",
        "border": "2px solid #fafafa",
        "boxShadow": "0 1px 4px rgba(0,0,0,0.85)",
        "pointerEvents": "none",
        "whiteSpace": "nowrap",
    }


def _display_grid_style(compare_mode: str) -> dict:
    """Side-by-side gives the mix 3x the width of the stacked channel column, so the three
    stacked channel images add up to the same height as the mixed image."""
    cols = "3fr 1fr" if compare_mode == "side_by_side" else "1fr"
    return {"display": "grid", "gridTemplateColumns": cols, "gap": "16px", "alignItems": "start"}


def _channels_stack_style(compare_mode: str) -> dict:
    """Stack the 3 channels vertically next to the mix, or in a row when shown alone."""
    cols = "1fr" if compare_mode == "side_by_side" else "1fr 1fr 1fr"
    return {"display": "grid", "gridTemplateColumns": cols, "gap": "8px"}


DARK_THEME_INDEX_STRING = """<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            :root, body {
                background-color: #000000;
                color: #e5e5e5;
                margin: 0;
                padding: 0;
                --Dash-Text-Primary: #e5e5e5 !important;
                --Dash-Text-Strong: #fafafa !important;
                --Dash-Text-Weak: #d4d4d4 !important;
                --Dash-Text-Disabled: #a3a3a3 !important;
                --Dash-Stroke-Strong: #525252 !important;
                --Dash-Stroke-Weak: #2a2a2a !important;
                --Dash-Fill-Interactive-Strong: #38bdf8 !important;
                --Dash-Fill-Interactive-Weak: #2a2a2a !important;
                --Dash-Fill-Inverse-Strong: #111111 !important;
                --Dash-Fill-Primary-Hover: rgba(56, 189, 248, 0.16) !important;
                --Dash-Fill-Primary-Active: rgba(56, 189, 248, 0.26) !important;
                --Dash-Fill-Disabled: #111111 !important;
            }
            /* Style the dcc.Slider track itself as the Lightroom-style colorbar */
            .color-slider-wrap .dash-slider-root {
                padding: 6px 0 24px 0 !important;
            }
            .color-slider-wrap .dash-slider-track {
                height: 12px !important;
                border-radius: 6px !important;
                background: var(--track-bg) !important;
                border: 1px solid #3a3a3a !important;
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
            /* Frame (time) slider: a visible rail, with the elapsed part highlighted */
            .frame-slider-wrap .dash-slider-track {
                height: 6px !important;
                border-radius: 3px !important;
                background: #3a3a3a !important;
            }
            .frame-slider-wrap .dash-slider-range {
                background: #8a8a8a !important;
            }
            /* Frame number box: same height, text size and border as the ◀ ▶ buttons beside it, and
               lifted (margin-bottom, like the buttons) so it centers on the rail, not on rail + marks. */
            .frame-slider-wrap input[type=number] {
                height: 24px !important;
                width: 46px !important;
                box-sizing: border-box !important;
                padding: 0 4px !important;
                margin-bottom: 16px !important;
                font-size: 12px !important;
                text-align: center !important;
                color: #e5e5e5 !important;
                background-color: #000000 !important;
                border: 1px solid #3a3a3a !important;
                border-radius: 6px !important;
            }
            .frame-slider-wrap .dash-slider-thumb {
                background-color: #e5e5e5 !important;
                border: 2px solid #000000 !important;
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
                color: #d4d4d4 !important;
                font-size: 11px !important;
            }
            .dash-dropdown,
            .dash-dropdown-trigger,
            .dash-dropdown-content,
            .dash-dropdown-options,
            .dash-dropdown-option,
            .dash-dropdown-value,
            .dash-dropdown-search {
                background-color: #111111 !important;
                color: #e5e5e5 !important;
                border-color: #3a3a3a !important;
            }
            /* Invisible focus helper of dcc.Dropdown: keep it from overflowing narrow dropdowns
               (the per-channel wavelength selectors) and creating a horizontal page scroll. */
            .dash-dropdown-focus-target {
                width: 1px !important;
                left: 0 !important;
            }
            .eye-btn:hover {
                border-color: #38bdf8 !important;
            }
            .dash-dropdown-option:hover {
                background-color: #2a2a2a !important;
                color: #ffffff !important;
            }
            label, .dash-radioitems label {
                color: #e5e5e5 !important;
                cursor: pointer;
            }
            .preset-btn:hover {
                background-color: #2a2a2a !important;
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


def create_dash_app(store: "FrameStore") -> Dash:
    app = Dash(__name__, title="AIA-Reloaded Interactive RGB / LCh Mixer")
    app.index_string = DARK_THEME_INDEX_STRING

    init_src = store.source("samples")
    init_cache = store.get(init_src.name, 0)
    init_triplet = default_triplet(init_src)
    init_data = init_cache.get(init_triplet)
    init_gh_grad, base_dom_h = global_hue_track(DEFAULT_WB_LCH, init_data.track_pixels, 0.0)
    _, init_gc_grad, init_gl_grad, init_g_swatch = make_triplet_gradients_and_swatch(
        0.0, GLOBAL_H_MIN, GLOBAL_H_MAX,
        1.0, GLOBAL_C_MIN, GLOBAL_C_MAX, 1.0,
        1.0, GLOBAL_L_MIN, GLOBAL_L_MAX, 1.0,
        h_offset=base_dom_h,
    )
    init_gh_hue_color = _rgb_vec_to_css(vivid_rgb_at_hue(base_dom_h))

    waveband_meta = [
        {"idx": k, "wavel": f"{w} Å", "short": w, "role": waveband_role(k, w)}
        for k, w in enumerate(init_triplet)
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
                    "backgroundColor": "#111111",
                    "border": "1px solid #2a2a2a",
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
                                    html.Span(wb["wavel"], id=f"wb-{i}-title", style={"fontWeight": "700", "fontSize": "14px", "color": "#fafafa"}),
                                    html.Span(f"  ({wb['role']})", id=f"wb-{i}-role", style={"fontSize": "11px", "color": "#d4d4d4"}),
                                ]
                            ),
                            html.Div(
                                id=f"wb-{i}-swatch",
                                title="Effective basis color for this waveband",
                                style={
                                    "width": "24px",
                                    "height": "24px",
                                    "borderRadius": "50%",
                                    "border": "2px solid #d4d4d4",
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
        "backgroundColor": "#000000",
        "color": "#e5e5e5",
        "border": "1px solid #3a3a3a",
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
            "backgroundColor": "#000000",
            "color": "#f5f5f5",
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
                    "borderBottom": "1px solid #2a2a2a",
                    "paddingBottom": "12px",
                    "marginBottom": "16px",
                },
                children=[
                    html.Div(
                        children=[
                            html.H2(
                                "SDO/AIA Interactive Perceptual LCh & RGB Mixer (.jp2)",
                                style={"margin": "0", "fontSize": "20px", "fontWeight": "700", "color": "#fafafa"},
                            ),
                            html.Div(
                                loaded_files_text(init_src, 0),
                                id="loaded-files",
                                style={"fontSize": "12px", "color": "#d4d4d4", "marginTop": "2px"},
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
                                    "backgroundColor": "#2a2a2a",
                                    "color": "#fafafa",
                                    "border": "1px solid #525252",
                                    "borderRadius": "6px",
                                    "padding": "8px 14px",
                                    "cursor": "pointer",
                                    "fontWeight": "600",
                                    "fontSize": "12px",
                                },
                            ),
                            html.Button(
                                "Save Full-Res JPEG",
                                id="btn-save-full",
                                title="Save the current frame at full resolution: samples/ for the samples, <event>/mixer_exports/ for an event",
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
                            # Data Source Card: bundled samples or an event folder (time series) from get_data.py
                            html.Div(
                                style={
                                    "backgroundColor": "#111111",
                                    "border": "1px solid #2a2a2a",
                                    "borderRadius": "8px",
                                    "padding": "12px 14px 2px 14px",
                                    "marginBottom": "14px",
                                },
                                children=[
                                    html.Div(
                                        "Data Source",
                                        style={"fontWeight": "700", "fontSize": "14px", "color": "#f5f5f5", "marginBottom": "6px"},
                                    ),
                                    dcc.Dropdown(
                                        id="source-select",
                                        options=[{"label": src.label, "value": name} for name, src in store.sources.items()],
                                        value=init_src.name,
                                        clearable=False,
                                        searchable=False,
                                        style={"fontSize": "12px", "color": "#e5e5e5", "marginBottom": "8px"},
                                    ),
                                    # Row: time slider (left) + ◀ ▶ (right); frame number & time right below.
                                    # Hidden for single-frame sources (the samples).
                                    html.Div(
                                        id="frame-row",
                                        style=frame_row_style(init_src),
                                        children=[
                                            html.Div(
                                                className="frame-slider-wrap",
                                                style={"flex": "1", "minWidth": "0"},
                                                children=dcc.Slider(
                                                    id="frame-slider",
                                                    min=0,
                                                    max=max(init_src.n_frames - 1, 1),
                                                    step=1,
                                                    value=0,
                                                    disabled=init_src.n_frames < 2,
                                                    marks=frame_marks(init_src),
                                                    updatemode="mouseup",  # each new frame decodes 3 full-res JP2s (~1 s)
                                                    # no value tooltip: the number box shows it, and a tooltip would cover the label below
                                                ),
                                            ),
                                            html.Div(
                                                style={"display": "flex", "gap": "4px", "flexShrink": "0", "marginBottom": "16px"},
                                                children=[
                                                    html.Button("◀", id="btn-frame-prev", n_clicks=0, title="Previous frame", className="preset-btn", style=frame_step_btn_style()),
                                                    html.Button("▶", id="btn-frame-next", n_clicks=0, title="Next frame", className="preset-btn", style=frame_step_btn_style()),
                                                ],
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        id="frame-label",
                                        children=frame_label_text(init_src, 0),
                                        style={"fontSize": "12px", "fontFamily": "monospace", "color": "#38bdf8", "marginBottom": "10px"},
                                    ),
                                ],
                            ),
                            # Auto-Balancing Rules Card
                            html.Div(
                                style={
                                    "backgroundColor": "#111111",
                                    "border": "1px solid #2a2a2a",
                                    "borderRadius": "8px",
                                    "padding": "12px 14px",
                                    "marginBottom": "14px",
                                },
                                children=[
                                    html.Div(
                                        "Auto-Balance Rules & Presets",
                                        style={"fontWeight": "700", "fontSize": "14px", "color": "#f5f5f5", "marginBottom": "6px"},
                                    ),
                                    html.Div(
                                        "Apply a rule computed from the image statistics, or save / load your Option 1 & Option 2 settings as a .json preset (in presets/):",
                                        style={"fontSize": "11px", "color": "#d4d4d4", "marginBottom": "10px"},
                                    ),
                                    html.Div(
                                        style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "8px", "marginBottom": "8px"},
                                        children=[
                                            html.Button(
                                                init_data.presets["warm_cool"]["label"],
                                                id="btn-auto-warm-cool",
                                                className="preset-btn",
                                                n_clicks=0,
                                                style=preset_btn_style,
                                            ),
                                            html.Button(
                                                init_data.presets["white_patch"]["label"],
                                                id="btn-auto-white-patch",
                                                className="preset-btn",
                                                n_clicks=0,
                                                style=preset_btn_style,
                                            ),
                                            html.Button(
                                                "💾 Save Preset…",
                                                id="btn-save-preset",
                                                className="preset-btn",
                                                n_clicks=0,
                                                title="Save the Option 1 & Option 2 sliders to presets/<name>.json",
                                                style=preset_btn_style,
                                            ),
                                            dcc.Upload(
                                                id="preset-upload",
                                                accept=".json,application/json",
                                                multiple=False,
                                                children=html.Button(
                                                    "📂 Load Preset…",
                                                    className="preset-btn",
                                                    title="Load Option 1 & Option 2 sliders from a .json preset (e.g. in presets/)",
                                                    style={**preset_btn_style, "width": "100%"},
                                                ),
                                            ),
                                        ],
                                    ),
                                    dcc.Store(id="preset-save-request"),
                                    html.Div(
                                        id="preset-status",
                                        style={"fontSize": "11px", "color": "#38bdf8", "marginBottom": "8px", "minHeight": "14px"},
                                    ),
                                    html.Div(
                                        id="auto-balance-desc",
                                        children=init_data.presets["default"]["desc"],
                                        style={
                                            "backgroundColor": "#000000",
                                            "border": "1px solid #2a2a2a",
                                            "borderRadius": "6px",
                                            "padding": "8px 10px",
                                            "fontSize": "11px",
                                            "color": "#d4d4d4",
                                            "lineHeight": "1.45",
                                        },
                                    ),
                                ],
                            ),
                            # Option 1: Global LCh Controls + Color Wheel
                            html.Div(
                                style={
                                    "backgroundColor": "#111111",
                                    "border": "1px solid #2a2a2a",
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
                                                style={"fontWeight": "700", "fontSize": "14px", "color": "#f5f5f5"},
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
                                                            "border": "2px solid #fafafa",
                                                            "backgroundColor": init_g_swatch,
                                                        },
                                                    ),
                                                    html.Button(
                                                        "Reset Global",
                                                        id="btn-reset-global",
                                                        n_clicks=0,
                                                        style={
                                                            "backgroundColor": "#2a2a2a",
                                                            "color": "#e5e5e5",
                                                            "border": "1px solid #525252",
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
                                            "backgroundColor": "#000000",
                                            "border": "1px solid #2a2a2a",
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
                                                    "border": "1px solid #3a3a3a",
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
                                                            "backgroundColor": "#fafafa",
                                                            "opacity": "0.6",
                                                        }
                                                    ),
                                                    html.Div(
                                                        init_triplet[0],
                                                        id="wheel-pin-0",
                                                        style=_wheel_pin_style(
                                                            init_anchor[0]["H"],
                                                            init_anchor[0]["C"],
                                                            lch_to_css_swatch(init_anchor[0]["L"], init_anchor[0]["C"], init_anchor[0]["H"]),
                                                        ),
                                                    ),
                                                    html.Div(
                                                        init_triplet[1],
                                                        id="wheel-pin-1",
                                                        style=_wheel_pin_style(
                                                            init_anchor[1]["H"],
                                                            init_anchor[1]["C"],
                                                            lch_to_css_swatch(init_anchor[1]["L"], init_anchor[1]["C"], init_anchor[1]["H"]),
                                                        ),
                                                    ),
                                                    html.Div(
                                                        init_triplet[2],
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
                                                style={"fontSize": "11px", "color": "#d4d4d4", "lineHeight": "1.45"},
                                                children=[
                                                    html.Div(
                                                        "Perceptual Oklch Color Wheel",
                                                        style={"fontWeight": "700", "color": "#fafafa", "fontSize": "12px", "marginBottom": "4px"},
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
                                    "backgroundColor": "#000000",
                                    "border": "1px solid #2a2a2a",
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
                                            "color": "#fafafa",
                                            "cursor": "pointer",
                                            "marginBottom": "10px",
                                        },
                                    ),
                                    html.Div(
                                        "Each waveband defines one column of self.rgbmix. Sliders move automatically with Option 1 and Auto-Balance rules, and can also be fine-tuned individually.",
                                        style={"fontSize": "11px", "color": "#d4d4d4", "marginBottom": "10px"},
                                    ),
                                    *wb_cards,
                                ],
                            ),
                            # Sharpening & Structure (collapsible, like Option 2): per input channel, after the tone curve
                            html.Details(
                                open=True,
                                style={
                                    "backgroundColor": "#000000",
                                    "border": "1px solid #2a2a2a",
                                    "borderRadius": "8px",
                                    "padding": "10px",
                                    "marginBottom": "14px",
                                },
                                children=[
                                    html.Summary(
                                        "Sharpening & Structure",
                                        style={"fontWeight": "700", "fontSize": "14px", "color": "#fafafa", "cursor": "pointer", "marginBottom": "10px"},
                                    ),
                                    html.Div(
                                        "Applied to each input channel after the Tone Curve, before mixing (so the mix gets no color fringes). "
                                        "Radii are in full-resolution pixels: fine sharpening is best judged in the 1:1 zoom views, since "
                                        "full-sun previews are downsampled 2.7–4×.",
                                        style={"fontSize": "11px", "color": "#d4d4d4", "marginBottom": "10px"},
                                    ),
                                    html.Div(
                                        "Structure — Multi-scale Gaussian Normalization (MGN)",
                                        style={"fontSize": "12px", "fontWeight": "700", "color": "#e5e5e5", "marginBottom": "4px"},
                                    ),
                                    _slider_row(
                                        "Structure (MGN weight; 0 = off)", "structure", 0.0, 0.2, 0.002, 0.0,
                                        {0: "0", 0.05: "0.05", 0.1: "0.1", 0.15: "0.15", 0.2: "0.2"}, GRAY_TRACK, "#d4d4d4",
                                    ),
                                    html.Div(
                                        "Sharpening — unsharp mask",
                                        style={"fontSize": "12px", "fontWeight": "700", "color": "#e5e5e5", "margin": "4px 0"},
                                    ),
                                    _slider_row(
                                        "Amount (0 = off)", "sharp-amount", 0.0, 3.0, 0.05, 0.0,
                                        {0: "0", 1: "1", 2: "2", 3: "3"}, GRAY_TRACK, "#d4d4d4",
                                    ),
                                    _slider_row(
                                        "Radius (full-res px)", "sharp-radius", 0.5, 4.0, 0.1, 1.0,
                                        {0.5: "0.5", 1: "1", 2: "2", 3: "3", 4: "4"}, GRAY_TRACK, "#d4d4d4",
                                    ),
                                    _slider_row(
                                        "Detail (halo allowance %; low = halo suppression)", "sharp-detail", 0.0, 100.0, 1.0, 25.0,
                                        {0: "0", 25: "25", 50: "50", 75: "75", 100: "100"}, GRAY_TRACK, "#d4d4d4",
                                    ),
                                    _slider_row(
                                        "Masking (edge mask %; 0 = sharpen everywhere)", "sharp-masking", 0.0, 100.0, 1.0, 0.0,
                                        {0: "0", 25: "25", 50: "50", 75: "75", 100: "100"}, GRAY_TRACK, "#d4d4d4",
                                    ),
                                ],
                            ),
                            # Resulting self.rgbmix Python snippet
                            html.Div(
                                style={
                                    "backgroundColor": "#111111",
                                    "border": "1px solid #2a2a2a",
                                    "borderRadius": "8px",
                                    "padding": "12px",
                                },
                                children=[
                                    html.Div(
                                        style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "6px"},
                                        children=[
                                            html.Span(
                                                "Resulting self.rgbmix Matrix (Copy to visualization.py)",
                                                style={"fontWeight": "700", "fontSize": "13px", "color": "#f5f5f5"},
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
                                            "backgroundColor": "#000000",
                                            "color": "#e5e5e5",
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
                                        style={"fontSize": "11px", "fontFamily": "monospace", "color": "#d4d4d4", "marginTop": "8px"},
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
                                    "backgroundColor": "#111111",
                                    "border": "1px solid #2a2a2a",
                                    "borderRadius": "8px",
                                    "padding": "10px 16px",
                                    "marginBottom": "14px",
                                },
                                children=[
                                    html.Div(
                                        style={"display": "flex", "alignItems": "center", "gap": "10px"},
                                        children=[
                                            html.Span("Region / Resolution:", style={"fontSize": "13px", "fontWeight": "600", "color": "#e5e5e5"}),
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
                                                style={"width": "340px", "fontSize": "13px", "color": "#e5e5e5"},
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        style={"display": "flex", "alignItems": "center", "gap": "10px"},
                                        children=[
                                            html.Span("Comparison:", style={"fontSize": "13px", "fontWeight": "600", "color": "#e5e5e5"}),
                                            dcc.Dropdown(
                                                id="compare-mode",
                                                options=[
                                                    {"label": "Side-by-Side (Mix + Input Channels)", "value": "side_by_side"},
                                                    {"label": "Single Large View (Mix)", "value": "single"},
                                                    {"label": "Input Channels Only", "value": "channels_only"},
                                                ],
                                                value="side_by_side",
                                                clearable=False,
                                                searchable=False,
                                                style={"width": "250px", "fontSize": "13px", "color": "#e5e5e5"},
                                            ),
                                        ],
                                    ),
                                    html.Span(
                                        id="render-timing",
                                        style={"fontSize": "12px", "fontFamily": "monospace", "color": "#d4d4d4"},
                                    ),
                                ],
                            ),
                            # Image Display Container
                            html.Div(
                                id="image-display-grid",
                                style=_display_grid_style("side_by_side"),
                                children=[
                                    # Mix column: the mixed image with its Tone Curve panel right below it, so the
                                    # panel spans the mix only (not the input-channels column).
                                    html.Div(
                                        id="modified-column",
                                        children=[
                                            # Modified Image Card
                                            html.Div(
                                                id="modified-card",
                                                style={
                                                    "backgroundColor": "#000000",
                                                    "border": "1px solid #2a2a2a",
                                                    "borderRadius": "8px",
                                                    "padding": "10px",
                                                    "textAlign": "center",
                                                },
                                                children=[
                                                    html.Div(
                                                        "Modified Mix (Live LCh -> self.rgbmix)",
                                                        style={"fontSize": "13px", "fontWeight": "700", "color": "#f5f5f5", "marginBottom": "8px"},
                                                    ),
                                                    html.Img(
                                                        id="img-modified",
                                                        style={"width": "100%", "maxHeight": "78vh", "objectFit": "contain", "borderRadius": "4px"},
                                                    ),
                                                ],
                                            ),
                                            # Tone panel (spans the mix column only): input gamma, then sigmoidal
                                            # contrast to restore the mid-tone contrast that gamma flattens.
                                            html.Div(
                                                style={
                                                    "backgroundColor": "#111111",
                                                    "border": "1px solid #2a2a2a",
                                                    "borderRadius": "8px",
                                                    "padding": "12px 16px 4px 16px",
                                                    "marginTop": "14px",
                                                },
                                                children=[
                                                    html.Div(
                                                        "Tone Curve (applied to each input channel before mixing)",
                                                        style={"fontWeight": "700", "fontSize": "14px", "color": "#fafafa", "marginBottom": "8px"},
                                                    ),
                                                    _slider_row(
                                                        "Gamma γ — lift faint structures (x → x^(1/γ))",
                                                        "gamma",
                                                        GAMMA_MIN,
                                                        GAMMA_MAX,
                                                        0.05,
                                                        1.0,
                                                        {0.5: "0.5", 1.0: "1.0", 1.5: "1.5", 2.0: "2.0", 2.5: "2.5", 3.0: "3.0"},
                                                        gamma_track_gradient(),
                                                        "#d4d4d4",
                                                    ),
                                                    _slider_row(
                                                        "Contrast — sigmoidal S-curve, brightness-neutral per channel (0 = off)",
                                                        "contrast",
                                                        CONTRAST_MIN,
                                                        CONTRAST_MAX,
                                                        0.1,
                                                        0.0,
                                                        {0: "0", 3: "3", 6: "6", 9: "9", 12: "12"},
                                                        contrast_track_gradient(),
                                                        "#d4d4d4",
                                                    ),
                                                ],
                                            ),
                                        ],
                                    ),
                                    # Input Channels Card: 304 / 171 / 193 each in its sunpy SDO/AIA colormap
                                    html.Div(
                                        id="default-card",
                                        style={
                                            "backgroundColor": "#000000",
                                            "border": "1px solid #2a2a2a",
                                            "borderRadius": "8px",
                                            "padding": "10px",
                                            "textAlign": "center",
                                        },
                                        children=[
                                            html.Div(
                                                "Input Channels (sunpy SDO/AIA colormaps)",
                                                style={"fontSize": "13px", "fontWeight": "600", "color": "#d4d4d4", "marginBottom": "8px"},
                                            ),
                                            html.Div(
                                                id="channels-stack",
                                                style=_channels_stack_style("side_by_side"),
                                                children=[
                                                    html.Figure(
                                                        style={"margin": "0"},
                                                        children=[
                                                            html.Div(
                                                                style={"display": "flex", "alignItems": "center", "gap": "6px", "marginBottom": "4px"},
                                                                children=[
                                                                    html.Span(
                                                                        slot["rgb"],
                                                                        title=f"{slot['name']} output channel (rgbmix column {k})",
                                                                        style={
                                                                            "fontWeight": "800",
                                                                            "fontSize": "12px",
                                                                            "color": "#000000",
                                                                            "backgroundColor": slot["color"],
                                                                            "borderRadius": "4px",
                                                                            "padding": "2px 6px",
                                                                        },
                                                                    ),
                                                                    dcc.Dropdown(
                                                                        id=f"ch-select-{k}",
                                                                        options=channel_options(init_src),
                                                                        value=init_triplet[k] or EMPTY_SLOT,
                                                                        clearable=False,
                                                                        searchable=False,
                                                                        style={"flex": "1", "minWidth": "0", "fontSize": "12px", "color": "#e5e5e5"},
                                                                    ),
                                                                ],
                                                            ),
                                                            # Thumbnail with the eye toggle overlaid on its top-right corner.
                                                            html.Div(
                                                                style={"position": "relative"},
                                                                children=[
                                                                    html.Img(
                                                                        id=f"img-channel-{k}",
                                                                        src=init_cache.channel_uri(init_triplet[k], "full_1024") if init_triplet[k] else EMPTY_CHANNEL_URI,
                                                                        style={"width": "100%", "objectFit": "contain", "borderRadius": "4px", "display": "block"},
                                                                    ),
                                                                    html.Button(
                                                                        html.Img(
                                                                            id=f"ch-eye-icon-{k}",
                                                                            src=EYE_OPEN_URI,
                                                                            style={"width": "18px", "height": "18px", "display": "block"},
                                                                        ),
                                                                        id=f"ch-eye-{k}",
                                                                        n_clicks=0,
                                                                        title="Hide this channel from the mix (preview only; sliders unchanged)",
                                                                        className="eye-btn",
                                                                        style={
                                                                            "position": "absolute",
                                                                            "top": "4px",
                                                                            "right": "4px",
                                                                            "backgroundColor": "rgba(0, 0, 0, 0.7)",
                                                                            "border": "1px solid #3a3a3a",
                                                                            "borderRadius": "6px",
                                                                            "padding": "3px 4px",
                                                                            "cursor": "pointer",
                                                                        },
                                                                    ),
                                                                ],
                                                            ),
                                                            html.Figcaption(
                                                                channel_caption(k, init_triplet[k]),
                                                                id=f"ch-caption-{k}",
                                                                style={"fontSize": "11px", "color": "#d4d4d4", "marginTop": "3px"},
                                                            ),
                                                        ],
                                                    )
                                                    for k, slot in enumerate(RGB_SLOTS)
                                                ],
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
        Output("channels-stack", "style"),
        Output("image-display-grid", "style"),
        Output("modified-column", "style"),
        Output("default-card", "style"),
        Output("rgbmix-code", "children"),
        Output("balance-metrics", "children"),
        Output("render-timing", "children"),
        Output("gamma-val", "children"),
        Output("contrast-val", "children"),
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
        Output("preset-upload", "contents"),
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
        Input("btn-auto-warm-cool", "n_clicks"),
        Input("btn-auto-white-patch", "n_clicks"),
        Input("preset-upload", "contents"),
        Input("view-select", "value"),
        Input("compare-mode", "value"),
        Input("ch-select-0", "value"),
        Input("ch-select-1", "value"),
        Input("ch-select-2", "value"),
        Input("gamma", "value"),
        Input("contrast", "value"),
        Input("ch-eye-0", "n_clicks"),
        Input("ch-eye-1", "n_clicks"),
        Input("ch-eye-2", "n_clicks"),
        Input("source-select", "value"),
        Input("frame-slider", "value"),
        *[Input(sid, "value") for sid in ENHANCE_SLIDER_IDS],
        State("mixer-state-store", "data"),
        State("preset-upload", "filename"),
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
        _n_wc,
        _n_wp,
        upload_contents,
        view_name,
        compare_mode,
        ch0,
        ch1,
        ch2,
        gamma,
        contrast,
        eye0,
        eye1,
        eye2,
        source_name,
        frame,
        structure,
        sh_amount,
        sh_radius,
        sh_detail,
        sh_masking,
        store_data,
        upload_filename,
    ):
        t0 = time.perf_counter()
        trigger = ctx.triggered_id
        try:
            data = store.get(source_name, frame).get((ch0, ch1, ch2))
        except ValueError:  # every slot empty: nothing to mix
            return (no_update,) * len(ctx.outputs_list)
        gamma = float(gamma) if gamma else 1.0
        contrast = float(contrast) if contrast else 0.0
        tp = make_tone_params(gamma, contrast, structure, sh_amount, sh_radius, sh_detail, sh_masking)
        sample_px, track_px = data.tone_pixels(tp)

        anchor_wb = copy.deepcopy(store_data["anchor_wb"])
        last_synced_wb = copy.deepcopy(store_data["last_synced_wb"])

        current_wb_inputs = [
            {"H": round(float(w0h), 1), "C": round(float(w0c), 3), "L": round(float(w0l), 3)},
            {"H": round(float(w1h), 1), "C": round(float(w1c), 3), "L": round(float(w1l), 3)},
            {"H": round(float(w2h), 1), "C": round(float(w2c), 3), "L": round(float(w2l), 3)},
        ]

        if isinstance(trigger, str) and trigger.startswith("wb-") and current_wb_inputs == last_synced_wb:
            return (no_update,) * len(ctx.outputs_list)

        preset_map = {
            "btn-reset": "default",
            "btn-auto-warm-cool": "warm_cool",
            "btn-auto-white-patch": "white_patch",
        }

        # A new wavelength assignment changes the statistics behind the default description.
        desc_out = (
            data.presets_for(tp)["default"]["desc"]
            if trigger in {"ch-select-0", "ch-select-1", "ch-select-2", "source-select"}
            else no_update
        )
        update_wb_sliders = False
        update_global_sliders = False
        clear_upload = no_update

        loaded = None
        if trigger == "preset-upload" and upload_contents:
            clear_upload = None  # so loading the same file again re-triggers
            try:
                _, b64 = upload_contents.split(",", 1)
                loaded = parse_preset(json.loads(base64.b64decode(b64).decode("utf-8")))
            except (ValueError, KeyError, TypeError, json.JSONDecodeError, UnicodeDecodeError) as exc:
                desc_out = f"Could not load preset '{upload_filename}': {exc}"

        if loaded is not None:
            anchor_wb, (gh, gc, gl) = loaded
            wb_lch = apply_global_to_anchor(anchor_wb, gh, gc, gl)
            last_synced_wb = copy.deepcopy(wb_lch)
            update_wb_sliders = True
            update_global_sliders = True
            desc_out = f"Loaded preset '{upload_filename}' (Option 1 & Option 2 sliders)."
        elif trigger in preset_map:
            preset = data.presets_for(tp)[preset_map[trigger]]
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
        # Hidden channels are dropped from what is displayed only (their rgbmix column is zeroed);
        # sliders, anchors and the exported matrix are left untouched, so un-hiding restores the mix.
        visible = visibility_mask(eye0, eye1, eye2)
        shown_mix = rgbmix * visible
        mod_uri = data.render_uri(view_name, shown_mix, tp)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        base_mod_style = {}
        base_def_style = {
            "backgroundColor": "#000000",
            "border": "1px solid #2a2a2a",
            "borderRadius": "8px",
            "padding": "10px",
            "textAlign": "center",
        }

        grid_style = _display_grid_style(compare_mode)
        if compare_mode == "single":
            base_def_style = {**base_def_style, "display": "none"}
        elif compare_mode == "channels_only":
            base_mod_style = {"display": "none"}

        code_str = (
            "self.rgbmix = np.array([\n"
            f"    [{rgbmix[0, 0]:7.4f}, {rgbmix[0, 1]:7.4f}, {rgbmix[0, 2]:7.4f}],\n"
            f"    [{rgbmix[1, 0]:7.4f}, {rgbmix[1, 1]:7.4f}, {rgbmix[1, 2]:7.4f}],\n"
            f"    [{rgbmix[2, 0]:7.4f}, {rgbmix[2, 1]:7.4f}, {rgbmix[2, 2]:7.4f}],\n"
            "])\n"
            f"self.gamma_rgb = ({gamma:.2f}, {gamma:.2f}, {gamma:.2f})\n"
            f"self.contrast = {contrast:.1f}\n"
            f"self.contrast_mid = ({', '.join(f'{v:.4f}' for v in data.contrast_mid(gamma, contrast))})\n"
            f"self.structure = {tp.structure:.3f}\n"
            f"self.sharpen = ({tp.amount:.2f}, {tp.radius:.1f}, {tp.detail:.0f}, {tp.masking:.0f})  # amount, radius, detail, masking"
        )
        hidden = [f"{w} Å" for w, v in zip(data.triplet, visible) if w and not v]
        if hidden:
            code_str += f"\n# preview currently hides {', '.join(hidden)} (matrix above is the full mix)"

        mets = mix_metrics(shown_mix, sample_px)
        p995 = mets["p995"]
        metrics_str = (
            f"Disk Mean Cast (a, b): ({mets['a']:+.3f}, {mets['b']:+.3f})  |  "
            f"Clipped px: {mets['clip_pct']:.1f}%  |  "
            f"p99.5 RGB: ({p995[0]:.2f}, {p995[1]:.2f}, {p995[2]:.2f})"
        )

        # Option 1 Global Hue track: centred on the dominant hue of the image rendered with the
        # stationary anchor (ΔH = 0), so the color under the cursor is the Sun's resulting hue.
        grad_gh, sun_h = global_hue_track(anchor_wb, track_px, float(gh), float(gc), float(gl), visible=visible)
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
            "border": "2px solid #d4d4d4",
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
        global_slider_outs = (float(gh), float(gc), float(gl)) if update_global_sliders else (no_update,) * 3

        return (
            *wb_slider_outs,
            *global_slider_outs,
            mod_uri,
            _channels_stack_style(compare_mode),
            grid_style,
            base_mod_style,
            base_def_style,
            code_str,
            metrics_str,
            f"Render: {elapsed_ms:.1f} ms",
            f"{gamma:.2f}",
            f"{contrast:.1f}" if contrast > 0 else "off",
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
                "border": "2px solid #fafafa",
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
            clear_upload,
        )

    # Channel images and every wavelength label depend only on the view and the wavelength
    # assignment, so they are sent only when those change (not on every slider drag).
    channel_outputs = []
    for k in range(3):
        channel_outputs += [
            Output(f"img-channel-{k}", "src"),
            Output(f"ch-caption-{k}", "children"),
            Output(f"wheel-pin-{k}", "children"),
            Output(f"wb-{k}-title", "children"),
            Output(f"wb-{k}-role", "children"),
            Output(f"wb-{k}-h-label", "children"),
            Output(f"wb-{k}-c-label", "children"),
            Output(f"wb-{k}-l-label", "children"),
        ]

    @app.callback(
        *channel_outputs,
        Input("view-select", "value"),
        Input("ch-select-0", "value"),
        Input("ch-select-1", "value"),
        Input("ch-select-2", "value"),
        Input("source-select", "value"),
        Input("frame-slider", "value"),
        prevent_initial_call=True,
    )
    def update_channel_panel(view_name, ch0, ch1, ch2, source_name, frame):
        frame_cache = store.get(source_name, frame)
        out = []
        for k, wavel in enumerate(norm_wavel(w) for w in (ch0, ch1, ch2)):
            name = f"{wavel} Å" if wavel else "Empty slot"
            out += [
                frame_cache.channel_uri(wavel, view_name) if wavel else EMPTY_CHANNEL_URI,
                channel_caption(k, wavel),
                wavel or "—",
                name,
                f"  ({waveband_role(k, wavel)})",
                f"{name} Hue Angle (H°)",
                f"{name} Chroma / Saturation (C)",
                f"{name} Luminance Weight (L)",
            ]
        return tuple(out)

    # Selecting a data source: R/G/B defaults (event folders: 304 / 171 / 131 when present, else
    # whatever is available, empty when fewer than 3 channels), channel lists and the frame slider.
    @app.callback(
        Output("ch-select-0", "options"),
        Output("ch-select-1", "options"),
        Output("ch-select-2", "options"),
        Output("ch-select-0", "value"),
        Output("ch-select-1", "value"),
        Output("ch-select-2", "value"),
        Output("frame-slider", "max"),
        Output("frame-slider", "value"),
        Output("frame-slider", "marks"),
        Output("frame-slider", "disabled"),
        Output("frame-row", "style"),
        Input("source-select", "value"),
        prevent_initial_call=True,
    )
    def on_source_change(source_name):
        src = store.source(source_name)
        opts = channel_options(src)
        trip = [w or EMPTY_SLOT for w in default_triplet(src)]
        return (opts, opts, opts, *trip, max(src.n_frames - 1, 1), 0, frame_marks(src), src.n_frames < 2, frame_row_style(src))

    # ◀ / ▶ step one frame (typing in the slider's box applies when it loses focus).
    @app.callback(
        Output("frame-slider", "value", allow_duplicate=True),
        Input("btn-frame-prev", "n_clicks"),
        Input("btn-frame-next", "n_clicks"),
        State("frame-slider", "value"),
        State("source-select", "value"),
        prevent_initial_call=True,
    )
    def step_frame(_prev, _next, frame, source_name):
        src = store.source(source_name)
        step = -1 if ctx.triggered_id == "btn-frame-prev" else 1
        new = int(np.clip(int(frame or 0) + step, 0, src.n_frames - 1))
        return new if new != frame else no_update

    @app.callback(
        Output("frame-label", "children"),
        Output("loaded-files", "children"),
        Input("source-select", "value"),
        Input("frame-slider", "value"),
        prevent_initial_call=True,
    )
    def update_frame_label(source_name, frame):
        src = store.source(source_name)
        return frame_label_text(src, frame or 0), loaded_files_text(src, frame or 0)

    @app.callback(
        *[Output(f"{sid}-val", "children") for sid in ENHANCE_SLIDER_IDS],
        *[Input(sid, "value") for sid in ENHANCE_SLIDER_IDS],
    )
    def update_enhance_labels(structure, amount, radius, detail, masking):
        return (
            f"{float(structure or 0):.3f}" if structure else "off",
            f"{float(amount or 0):.2f}" if amount else "off",
            f"{float(radius or 1):.1f} px",
            f"{float(detail or 0):.0f}%",
            f"{float(masking or 0):.0f}%" if masking else "off",
        )

    # Save Preset: ask for a filename in the browser, then write the JSON server-side into presets/.
    app.clientside_callback(
        """
        function(n) {
            if (!n) { return window.dash_clientside.no_update; }
            const d = new Date();
            const pad = (x) => String(x).padStart(2, "0");
            const def = "preset_" + d.getFullYear() + pad(d.getMonth() + 1) + pad(d.getDate()) + "_"
                + pad(d.getHours()) + pad(d.getMinutes()) + pad(d.getSeconds()) + ".json";
            const name = window.prompt("Save Option 1 & Option 2 preset as (.json, saved in presets/):", def);
            if (!name) { return window.dash_clientside.no_update; }
            return {"name": name, "t": Date.now()};
        }
        """,
        Output("preset-save-request", "data"),
        Input("btn-save-preset", "n_clicks"),
        prevent_initial_call=True,
    )

    @app.callback(
        Output("preset-status", "children"),
        Input("preset-save-request", "data"),
        State("global-h", "value"),
        State("global-c", "value"),
        State("global-l", "value"),
        State("wb-0-h", "value"),
        State("wb-0-c", "value"),
        State("wb-0-l", "value"),
        State("wb-1-h", "value"),
        State("wb-1-c", "value"),
        State("wb-1-l", "value"),
        State("wb-2-h", "value"),
        State("wb-2-c", "value"),
        State("wb-2-l", "value"),
        State("mixer-state-store", "data"),
        State("ch-select-0", "value"),
        State("ch-select-1", "value"),
        State("ch-select-2", "value"),
        prevent_initial_call=True,
    )
    def save_preset(request, gh, gc, gl, w0h, w0c, w0l, w1h, w1c, w1l, w2h, w2c, w2l, store_data, ch0, ch1, ch2):
        if not request:
            return no_update
        try:
            path = preset_path(request.get("name"))
        except ValueError as exc:
            return f"Preset not saved — invalid filename: {exc}."
        if os.path.exists(path):
            return f"Preset not saved — {os.path.relpath(path, PROJECT_DIR)} already exists; choose another name."
        wb_lch = [
            {"H": float(w0h), "C": float(w0c), "L": float(w0l)},
            {"H": float(w1h), "C": float(w1c), "L": float(w1l)},
            {"H": float(w2h), "C": float(w2c), "L": float(w2l)},
        ]
        preset = build_preset(
            store_data["anchor_wb"], wb_lch, float(gh), float(gc), float(gl), tuple(norm_wavel(w) for w in (ch0, ch1, ch2)), lch_to_rgbmix(wb_lch)
        )
        os.makedirs(PRESETS_DIR, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(preset, f, indent=2)
        return f"Saved preset to {os.path.relpath(path, PROJECT_DIR)}"

    # Eye toggles: swap the icon and dim the hidden channel's thumbnail.
    eye_outputs = []
    for k in range(3):
        eye_outputs += [
            Output(f"ch-eye-icon-{k}", "src"),
            Output(f"ch-eye-{k}", "title"),
            Output(f"img-channel-{k}", "style"),
        ]

    @app.callback(
        *eye_outputs,
        Input("ch-eye-0", "n_clicks"),
        Input("ch-eye-1", "n_clicks"),
        Input("ch-eye-2", "n_clicks"),
        prevent_initial_call=True,
    )
    def update_eye_toggles(n0, n1, n2):
        out = []
        for n in (n0, n1, n2):
            shown = channel_visible(n)
            out += [
                EYE_OPEN_URI if shown else EYE_CLOSED_URI,
                "Hide this channel from the mix (preview only; sliders unchanged)" if shown else "Show this channel in the mix again",
                {"width": "100%", "objectFit": "contain", "borderRadius": "4px", "display": "block", "opacity": "1" if shown else "0.25"},
            ]
        return tuple(out)

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
        State("ch-select-0", "value"),
        State("ch-select-1", "value"),
        State("ch-select-2", "value"),
        State("gamma", "value"),
        State("contrast", "value"),
        State("ch-eye-0", "n_clicks"),
        State("ch-eye-1", "n_clicks"),
        State("ch-eye-2", "n_clicks"),
        State("source-select", "value"),
        State("frame-slider", "value"),
        *[State(sid, "value") for sid in ENHANCE_SLIDER_IDS],
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
        ch0,
        ch1,
        ch2,
        gamma,
        contrast,
        eye0,
        eye1,
        eye2,
        source_name,
        frame,
        structure,
        sh_amount,
        sh_radius,
        sh_detail,
        sh_masking,
    ):
        src = store.source(source_name)
        frame = int(frame or 0)
        triplet = tuple(norm_wavel(w) for w in (ch0, ch1, ch2))
        if all(w is None for w in triplet):
            return "Nothing to save: all R/G/B slots are empty."
        visible = visibility_mask(eye0, eye1, eye2)
        gamma = float(gamma) if gamma else 1.0
        contrast = float(contrast) if contrast else 0.0
        tp = make_tone_params(gamma, contrast, structure, sh_amount, sh_radius, sh_detail, sh_masking)
        wb_lch = [
            {"H": float(w0h), "C": float(w0c), "L": float(w0l)},
            {"H": float(w1h), "C": float(w1c), "L": float(w1l)},
            {"H": float(w2h), "C": float(w2c), "L": float(w2l)},
        ]
        rgbmix = lch_to_rgbmix(wb_lch) * visible  # save what the preview shows (hidden channels left out)
        frame_cache = store.get(src.name, frame)
        mids = frame_cache.get(triplet).contrast_mid(gamma, contrast)

        # Same pipeline as RGBMixer/process_rgb_image for jp2 (visualization.scale_rgb, then RGB -> BGR),
        # with the fixed 0..255 input range the preview uses, and all-zero inputs for empty slots.
        files = src.files_for(frame)
        raw = [visualization.load_jp2(files[w]) if w else None for w in triplet]
        shape = next(r.shape for r in raw if r is not None)
        raw = [r if r is not None else np.zeros(shape) for r in raw]
        im = visualization.scale_rgb(
            raw, np.zeros(3), np.full(3, 255.0), gamma_rgb=(gamma, gamma, gamma), rgbmix=rgbmix,
            contrast=contrast, contrast_mid=tuple(mids), structure=tp.structure, sharpen=tp.sharpen,
            sharpen_mask_thr=frame_cache.get(triplet).mask_thresholds(tp) if tp.amount > 0 and tp.masking > 0 else None,
        )
        bgr = np.clip(np.flip(im, axis=2), 0, 255).astype(np.uint8)

        hidden = "".join(f"_hide{w}" for w, v in zip(triplet, visible) if w and not v)
        stem = f"im_custom_lch_jp2_{'_'.join(w or 'none' for w in triplet)}{hidden}_gamma{gamma:.2f}_contrast{contrast:.1f}"
        if tp.structure > 1e-6:
            stem += f"_mgn{tp.structure:.3f}"
        if tp.amount > 1e-6:
            stem += f"_sharp{tp.amount:.2f}-r{tp.radius:.1f}-d{tp.detail:.0f}-m{tp.masking:.0f}"
        suffix = src.frame_times[frame] if src.is_event else "0000"
        os.makedirs(src.output_dir, exist_ok=True)
        out_path = os.path.join(src.output_dir, f"{stem}_{suffix}.jpeg")
        cv2.imwrite(out_path, bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        return f"Saved full {bgr.shape[1]}×{bgr.shape[0]} image to: {out_path}"

    return app


def find_sample_jp2_files(samples_dir: str) -> dict[str, str]:
    """Map every AIA wavelength with a `*.<wavel>.jp2` file in samples_dir to that file."""
    files = {}
    for wavel in AIA_WAVELENGTHS:
        matches = sorted(glob.glob(os.path.join(samples_dir, f"*.{wavel}.jp2")))
        if matches:
            files[wavel] = matches[0]
    if len(files) < 3:
        raise FileNotFoundError(f"Need .jp2 files for at least 3 AIA wavelengths in {samples_dir}, found {list(files)}")
    return files


EVENT_PREFERRED_TRIPLET = ("304", "171", "131")


def default_triplet(src: "DataSource") -> tuple:
    """Default R/G/B wavelengths for a data source.

    Samples: 304/171/193, the assignment set_jp2_default() is tuned for (else as for events).
    Event folders: R=304, G=171, B=131 when present; a slot whose preferred channel is missing
    takes the next available channel (AIA order) not already used; with fewer than 3 channels
    the remaining slots stay empty (None)."""
    wavelengths = src.wavelengths
    if not src.is_event and all(w in wavelengths for w in DEFAULT_TRIPLET):
        return DEFAULT_TRIPLET
    out = [w if w in wavelengths else None for w in EVENT_PREFERRED_TRIPLET]
    remaining = [w for w in wavelengths if w not in out]
    for k in range(3):
        if out[k] is None and remaining:
            out[k] = remaining.pop(0)
    return tuple(out)


def channel_options(src: "DataSource") -> list[dict]:
    return [{"label": f"{w} Å", "value": w} for w in src.wavelengths] + [{"label": "— empty —", "value": EMPTY_SLOT}]


def frame_marks(src: "DataSource") -> dict:
    if src.n_frames < 2:
        return {}
    idx = sorted({0, (src.n_frames - 1) // 2, src.n_frames - 1})
    return {i: {"label": src.frame_label(i).split(" ")[1][:5] if src.is_event else str(i + 1), "style": {"color": "#d4d4d4", "fontSize": "11px"}} for i in idx}


ENHANCE_SLIDER_IDS = ["structure", "sharp-amount", "sharp-radius", "sharp-detail", "sharp-masking"]
GRAY_TRACK = "linear-gradient(to right, #262626, #a3a3a3)"


def make_tone_params(gamma, contrast, structure, amount, radius, detail, masking) -> "ToneParams":
    return ToneParams(
        gamma=float(gamma) if gamma else 1.0,
        contrast=float(contrast) if contrast else 0.0,
        structure=float(structure) if structure else 0.0,
        amount=float(amount) if amount else 0.0,
        radius=float(radius) if radius else 1.0,
        detail=float(detail) if detail is not None else 25.0,
        masking=float(masking) if masking else 0.0,
    )


def frame_row_style(src: "DataSource") -> dict:
    return {"display": "flex" if src.n_frames > 1 else "none", "alignItems": "center", "gap": "8px"}


def frame_step_btn_style() -> dict:
    return {
        "backgroundColor": "#000000",
        "color": "#e5e5e5",
        "border": "1px solid #3a3a3a",
        "borderRadius": "6px",
        "padding": "2px 9px",
        "cursor": "pointer",
        "fontSize": "12px",
    }


def frame_label_text(src: "DataSource", frame: int) -> str:
    if src.n_frames < 2:
        return "Single frame (samples)"
    return f"Frame {int(frame) + 1} / {src.n_frames}  ·  {src.frame_label(frame)}"


def loaded_files_text(src: "DataSource", frame: int) -> str:
    files = src.files_for(frame)
    return f"Loaded: {src.name} — {', '.join(os.path.basename(f) for f in files.values())}"


def waveband_role(slot: int, wavel) -> str:
    if not wavel:
        return f"{RGB_SLOTS[slot]['name']} Input — empty (no channel assigned)"
    return f"{RGB_SLOTS[slot]['name']} Input — {AIA_ROLES.get(wavel, 'AIA channel')}"


def channel_caption(slot: int, wavel) -> str:
    if not wavel:
        return f"(empty) → {RGB_SLOTS[slot]['rgb']} input"
    return f"{wavel} Å → {RGB_SLOTS[slot]['rgb']} input  ·  {aia_cmap_name(wavel)}"


def main():
    parser = argparse.ArgumentParser(description="Launch the interactive AIA JP2 RGB / LCh mixer Dash app.")
    parser.add_argument(
        "--samples-dir",
        default=os.path.abspath("samples"),
        help="Directory containing sample *.<wavelength>.jp2 files, e.g. sample.304.jp2 (default: ./samples)",
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.expanduser("~/data/AIA"),
        help="Folder holding event_* folders from get_data.py, selectable as data sources (default: ~/data/AIA)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8050, help="Port number (default: 8050)")
    parser.add_argument("--debug", action="store_true", help="Run Dash in debug mode")
    args = parser.parse_args()

    sources = discover_sources(os.path.abspath(args.samples_dir), args.data_dir)
    for src in sources.values():
        print(f"Data source: {src.label}")
    app = create_dash_app(FrameStore(sources))
    print(f"Starting Dash server at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
