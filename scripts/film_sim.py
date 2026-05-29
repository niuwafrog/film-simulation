"""
Film Simulation — apply classic film stock looks to digital images.

Uses proper film-style processing:
  - Linear color space manipulation
  - 3x3 color matrix transforms
  - Film characteristic curve (shoulder/toe)
  - Three-way color grading (shadows/midtones/highlights)
  - Per-hue selectivity
  - Luminance-matched film grain
  - Halation bloom

Usage:
  python film_sim.py <input> <output> --profile <name> [--strength 0-1]
  python film_sim.py --list
"""

import argparse
import json
import os
import sys
import math

import numpy as np
from PIL import Image, ImageFilter

try:
    _RESAMPLE = Image.Resampling.BILINEAR
except AttributeError:
    _RESAMPLE = Image.BILINEAR


# ═══════════════════════════════════════════════════════════════════
#  Color science primitives
# ═══════════════════════════════════════════════════════════════════

def _lin_to_srgb(c):
    """Linear → sRGB (per-channel, safe)."""
    c = np.clip(c, 0, 1)
    mask = c <= 0.0031308
    return np.where(mask, c * 12.92, 1.055 * (c ** (1.0 / 2.4)) - 0.055)


def _srgb_to_lin(c):
    """sRGB → linear (per-channel, safe)."""
    c = np.clip(c, 0, 1)
    mask = c <= 0.04045
    return np.where(mask, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _rgb_to_gray(r, g, b, w=(0.2126, 0.7152, 0.0722)):
    return r * w[0] + g * w[1] + b * w[2]


# ═══════════════════════════════════════════════════════════════════
#  Film curve — classic five-point characteristic curve
# ═══════════════════════════════════════════════════════════════════

def _film_curve(channel, toe=0.08, shoulder=0.92, toe_slope=0.15,
                shoulder_slope=0.10, contrast=1.0, lift=0.0, gamma=1.0):
    """
    Film-style characteristic curve with adjustable toe, shoulder, contrast.

    toe:         point where shadows start rolling off (lower = more linear in shadows)
    shoulder:    point where highlights start rolling off
    toe_slope:   how aggressively shadows roll off (lower = more compression)
    shoulder_slope: how aggressively highlights roll off
    contrast:    overall slope in midtones
    lift:        raise/lower blacks (positive = faded blacks)
    gamma:       overall gamma adjustment
    """
    c = np.clip(channel, 0, 1)

    # Apply contrast in linear space
    c = (c - 0.5) * contrast + 0.5 + lift
    c = np.clip(c, 0, 1)

    # Gamma
    c = np.power(np.clip(c, 1e-6, 1), gamma)

    # Toe (shadow roll-off)
    toe_mask = c < toe
    if np.any(toe_mask):
        c[toe_mask] = toe * (1 - toe_slope) * (c[toe_mask] / toe) ** 2 \
                      + toe * toe_slope * (c[toe_mask] / toe)

    # Shoulder (highlight roll-off)
    shoulder_mask = c > shoulder
    if np.any(shoulder_mask):
        t = (c[shoulder_mask] - shoulder) / (1 - shoulder)
        c[shoulder_mask] = shoulder + (1 - shoulder) * (
            t * (1 - shoulder_slope) + shoulder_slope * (1 - np.exp(-t * 4)) / (1 - np.exp(-4))
        )

    return np.clip(c, 0, 1)


# ═══════════════════════════════════════════════════════════════════
#  Color matrix transform  (RGB → M→ RGB  in linear space)
# ═══════════════════════════════════════════════════════════════════

def _apply_color_matrix(r, g, b, matrix_3x3):
    """Apply 3x3 color matrix in linear space."""
    r2 = r * matrix_3x3[0][0] + g * matrix_3x3[0][1] + b * matrix_3x3[0][2]
    g2 = r * matrix_3x3[1][0] + g * matrix_3x3[1][1] + b * matrix_3x3[1][2]
    b2 = r * matrix_3x3[2][0] + g * matrix_3x3[2][1] + b * matrix_3x3[2][2]
    return np.clip(r2, 0, 1), np.clip(g2, 0, 1), np.clip(b2, 0, 1)


# ═══════════════════════════════════════════════════════════════════
#  Three-way color grading  (shadows / midtones / highlights)
# ═══════════════════════════════════════════════════════════════════

def _three_way_grade(r, g, b, shadows=(0,0,0), midtones=(0,0,0), highlights=(0,0,0)):
    """
    Apply independent RGB offsets to shadows, midtones, highlights.

    Each is a (dr, dg, db) tuple — values added in linear space.
    Uses luminance-based blending for smooth transitions.
    """
    gray = _rgb_to_gray(r, g, b)

    # Shadow weight: strongest near 0, falls off by 0.3
    sh_w = np.clip(1 - gray / 0.3, 0, 1)
    sh_w = sh_w * sh_w  # steeper falloff

    # Highlight weight: strongest near 1, falls off below 0.7
    hl_w = np.clip((gray - 0.6) / 0.3, 0, 1)
    hl_w = hl_w * hl_w

    # Midtone weight: bell curve centered at 0.45
    mt_w = np.exp(-((gray - 0.45) ** 2) / 0.08)
    mt_w = mt_w * (1 - sh_w) * (1 - hl_w)

    r += sh_w * shadows[0] + mt_w * midtones[0] + hl_w * highlights[0]
    g += sh_w * shadows[1] + mt_w * midtones[1] + hl_w * highlights[1]
    b += sh_w * shadows[2] + mt_w * midtones[2] + hl_w * highlights[2]

    return np.clip(r, 0, 1), np.clip(g, 0, 1), np.clip(b, 0, 1)


# ═══════════════════════════════════════════════════════════════════
#  Per-hue selectivity
# ═══════════════════════════════════════════════════════════════════

def _hue_adjust(r, g, b, adjustments):
    """
    Apply per-hue saturation/hue-shift adjustments.

    adjustments: list of (hue_center_deg, hue_width_deg, sat_mult, hue_shift_deg)
      hue_center: 0-360 degrees
      hue_width:   width of the affected range
      sat_mult:    saturation multiplier (0=gray, 1=unchanged)
      hue_shift_deg: shift hue by this many degrees
    """
    # Convert to HSL
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    l = (mx + mn) / 2.0
    d = mx - mn

    eps = 1e-8
    s = np.where(d < eps, 0.0, d / np.maximum(1 - np.abs(2 * l - 1), eps))

    h = np.zeros_like(s)
    mask = d > eps
    if np.any(mask):
        rm, gm, bm, dm = r[mask], g[mask], b[mask], d[mask]
        mxm = mx[mask]
        h[mask] = np.where(
            mxm == rm,
            ((gm - bm) / dm) % 6,
            np.where(mxm == gm, (bm - rm) / dm + 2.0, (rm - gm) / dm + 4.0)
        )
    h_deg = (h / 6.0) * 360.0

    result_s = s.copy()
    result_h = h_deg.copy()

    for center, width, sat_mult, hue_shift in adjustments:
        half = width / 2.0
        # distance on color wheel
        diff = np.abs((h_deg - center + 180) % 360 - 180)
        weight = np.clip(1 - diff / half, 0, 1)
        # smooth edge
        weight = weight * weight * (3 - 2 * weight)

        if abs(sat_mult - 1.0) > 0.001:
            result_s = np.where(weight > 0, s * (1 - weight) + s * weight * sat_mult, result_s)

        if abs(hue_shift) > 0.1:
            result_h = np.where(weight > 0, (h_deg + weight * hue_shift) % 360, result_h)

    # Clamp saturation
    result_s = np.clip(result_s, 0, 1)
    h_rad = result_h / 360.0 * 6.0

    c = (1 - np.abs(2 * l - 1)) * result_s
    x = c * (1 - np.abs((h_rad % 2) - 1))
    m = l - c / 2

    out_r = np.zeros_like(r)
    out_g = np.zeros_like(r)
    out_b = np.zeros_like(r)

    hi = h_rad.astype(int) % 6
    for i in range(6):
        mask_i = hi == i
        if i == 0:
            out_r[mask_i], out_g[mask_i], out_b[mask_i] = c[mask_i], x[mask_i], 0
        elif i == 1:
            out_r[mask_i], out_g[mask_i], out_b[mask_i] = x[mask_i], c[mask_i], 0
        elif i == 2:
            out_r[mask_i], out_g[mask_i], out_b[mask_i] = 0, c[mask_i], x[mask_i]
        elif i == 3:
            out_r[mask_i], out_g[mask_i], out_b[mask_i] = 0, x[mask_i], c[mask_i]
        elif i == 4:
            out_r[mask_i], out_g[mask_i], out_b[mask_i] = x[mask_i], 0, c[mask_i]
        else:
            out_r[mask_i], out_g[mask_i], out_b[mask_i] = c[mask_i], 0, x[mask_i]

    out_r += m
    out_g += m
    out_b += m

    return np.clip(out_r, 0, 1), np.clip(out_g, 0, 1), np.clip(out_b, 0, 1)


# ═══════════════════════════════════════════════════════════════════
#  Tools: grain, halation, fade
# ═══════════════════════════════════════════════════════════════════

def _add_grain(img, amount):
    """Add luminance-matched film grain (subtle, natural texture)."""
    if amount < 0.05:
        return img
    h, w = img.shape[:2]
    gray = img.mean(axis=2, keepdims=True)

    # Single-scale fine grain — film grain is fine, not chunky
    noise = np.random.randn(h, w, 1) * 0.04 * amount

    # Grain is weakest in shadows (blocked up) and highlights (featureless sky)
    # and strongest in midtones
    grain_envelope = 4 * gray * (1 - gray)
    noise = noise * grain_envelope

    return np.clip(img + noise, 0, 1)


def _add_halation(img, radius=25, intensity=0.12):
    """Red-dominant bloom around bright areas (Cinestill signature)."""
    gray = np.mean(img, axis=2)
    bright = np.clip((gray - 0.65) / 0.35, 0, 1)
    bloom = np.array(Image.fromarray((bright * 255).astype(np.uint8))
                     .filter(ImageFilter.GaussianBlur(radius=radius))) / 255.0
    hal = np.zeros_like(img)
    hal[..., 0] = bloom * intensity * 1.6
    hal[..., 1] = bloom * intensity * 0.3
    hal[..., 2] = bloom * intensity * 0.1
    return np.clip(img + hal, 0, 1)


def _fade(img, amount):
    """Fade blacks → vintage look."""
    return img * (1 - amount) + amount * 0.45


def _vignette(img, amount):
    """Apply subtle lens vignetting (darkened corners)."""
    if amount < 0.01:
        return img
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / np.sqrt(cx ** 2 + cy ** 2)
    falloff = 1 - amount * (dist ** 2)
    return np.clip(img * falloff[..., np.newaxis], 0, 1)


def _cartoonize(img, edge_thickness=0.5, color_levels=8):
    """Convert photo to cartoon/anime look using edge detection + color quantization.

    edge_thickness: 0-1, how prominent edges are
    color_levels:   number of quantization levels per channel (4-16)
    """
    h, w = img.shape[:2]
    gray = (img.mean(axis=2) * 255).astype(np.uint8)

    # Edge detection via Sobel
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    gx = np.zeros_like(gray, dtype=np.float32)
    gy = np.zeros_like(gray, dtype=np.float32)
    for dy in range(3):
        for dx in range(3):
            shifted = np.roll(np.roll(gray, dy - 1, axis=0), dx - 1, axis=1)
            gx += shifted * sobel_x[dy, dx]
            gy += shifted * sobel_y[dy, dx]
    edges = np.sqrt(gx ** 2 + gy ** 2)

    # Normalize edges
    edge_max = edges.max() if edges.max() > 0 else 1
    edges = np.clip(edges / edge_max, 0, 1)

    # Invert: edges become black lines
    edge_mask = 1 - edges * edge_thickness
    edge_mask = np.clip(edge_mask, 0, 1)

    # Bilateral-like smoothing: median filter + Gaussian
    flat = (img * 255).astype(np.uint8)
    # Apply median filter per channel for edge-preserving smoothing
    from PIL import ImageFilter
    flat_pil = Image.fromarray(flat)
    median = flat_pil.filter(ImageFilter.MedianFilter(size=5))
    smooth = np.array(median).astype(np.float32) / 255.0

    # Color quantization
    smooth = (smooth * (color_levels - 1)).round() / (color_levels - 1)

    # Apply edge lines
    result = smooth * edge_mask[..., np.newaxis]
    return np.clip(result, 0, 1)


def _cartoon_inked(img, edge_threshold=0.15, color_levels=6, line_width=1):
    """Cartoon with bold ink line art — thick black outlines, posterized flat colors.

    edge_threshold: lower = more edges visible (0.05-0.3)
    color_levels:   number of quantization levels (3-8)
    line_width:     dilation iterations to thicken lines (0-2)
    """
    h, w = img.shape[:2]
    gray = (img.mean(axis=2) * 255).astype(np.uint8)

    # Edge detection via Sobel with stronger response
    sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    gx = np.zeros_like(gray, dtype=np.float32)
    gy = np.zeros_like(gray, dtype=np.float32)
    for dy in range(3):
        for dx in range(3):
            gx += np.roll(np.roll(gray, dy - 1, 0), dx - 1, 1) * sx[dy, dx]
            gy += np.roll(np.roll(gray, dy - 1, 0), dx - 1, 1) * sy[dy, dx]
    edges = np.sqrt(gx ** 2 + gy ** 2)
    edges = edges / (edges.max() + 1e-8)

    # Binary edge threshold for clean ink lines
    ink_mask = (edges > edge_threshold).astype(np.uint8)

    # Thicken lines via dilation
    if line_width > 0:
        from PIL import ImageFilter
        ink_img = Image.fromarray(ink_mask * 255)
        for _ in range(line_width):
            ink_img = ink_img.filter(ImageFilter.MaxFilter(3))
        ink_mask = np.array(ink_img) > 127

    # Color quantization (flat posterized colors)
    flat = (img * (color_levels - 1)).round() / (color_levels - 1)

    # Apply black ink lines on top
    result = flat.copy()
    result[ink_mask] = 0.0  # pure black lines

    return np.clip(result, 0, 1)


def _lineart(img, sensitivity=0.08, line_width=0, clean_bg=True):
    """Convert photo to anti-aliased line art — smooth edges, no pixel jaggies.

    Uses Sobel magnitude as continuous line darkness (not binary threshold),
    producing smooth, anti-aliased strokes.
    sensitivity: 0.03-0.2, higher = more lines visible
    line_width:  0 = thin, 1+ = thicker
    """
    from PIL import ImageFilter, Image
    gray_np = (img.mean(axis=2) * 255).astype(np.uint8)

    # Pre-blur
    blurred = np.array(Image.fromarray(gray_np).filter(
        ImageFilter.GaussianBlur(radius=0.6)), dtype=np.float32)

    # Sobel
    sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    gx = np.zeros_like(blurred); gy = np.zeros_like(blurred)
    for dy in range(3):
        for dx in range(3):
            gx += np.roll(np.roll(blurred, dy - 1, 0), dx - 1, 1) * sx[dy, dx]
            gy += np.roll(np.roll(blurred, dy - 1, 0), dx - 1, 1) * sy[dy, dx]
    mag = np.sqrt(gx**2 + gy**2)
    mag = mag / (mag.max() + 1e-8)

    # Anti-aliased line: continuous value (0=white, 1=black)
    # High Sobel mag → dark line, low mag → white
    lo = sensitivity * 0.4
    hi = sensitivity * 1.5
    lines = np.clip((mag - lo) / (hi - lo), 0, 1)
    lines = lines * lines  # contrast curve for cleaner lines

    # Remove noise floor (isolated faint pixels)
    if clean_bg:
        binary = lines > 0.15
        opened = np.array(Image.fromarray((binary * 255).astype(np.uint8))
                          .filter(ImageFilter.MinFilter(3))) > 127
        restored = np.array(Image.fromarray((opened * 255).astype(np.uint8))
                            .filter(ImageFilter.MaxFilter(3))) > 127
        lines = lines * restored

    # Subtle smoothing of the line mask to reduce jaggies
    lines = np.array(Image.fromarray((lines * 255).astype(np.uint8))
                     .filter(ImageFilter.GaussianBlur(radius=0.4))) / 255.0

    # Thicken
    if line_width > 0:
        binary = lines > 0.5
        for _ in range(line_width):
            binary = np.array(Image.fromarray((binary * 255).astype(np.uint8))
                              .filter(ImageFilter.MaxFilter(3))) > 127
        lines = np.maximum(lines, binary.astype(float) * 0.9)

    result = 1.0 - np.stack([lines, lines, lines], axis=-1)
    return np.clip(result, 0, 1)


def _pixelate(img, block_size=8, color_levels=4, dither=False):
    """Retro game / pixel art effect — lower resolution + color quantization.

    Downsamples the image to a blocky low-res version using nearest-neighbor
    interpolation, then quantizes colors to a limited palette — like classic
    retro game consoles (NES, Game Boy, CGA, etc.).

    block_size:    how many pixels per block (4-32). Larger = more pixelated.
                   The image is downscaled by 1/block_size then upscaled back.
    color_levels:  number of quantization levels per channel (2-8).
                   Lower = more aggressive posterization (e.g. 2 = 8 total colors).
    dither:        whether to apply Floyd-Steinberg dithering for smoother
                   transitions between quantized colors.
    """
    h, w = img.shape[:2]
    if block_size < 2:
        block_size = 2

    # --- 1. Downscale to low resolution with nearest-neighbor ---
    small_w = max(1, w // block_size)
    small_h = max(1, h // block_size)
    img_8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    small = np.array(
        Image.fromarray(img_8).resize((small_w, small_h), Image.Resampling.NEAREST)
    ).astype(np.float32) / 255.0

    # --- 2. Color quantization (posterization) ---
    if dither and color_levels < 8:
        # Apply Floyd-Steinberg dithering before quantization
        small = _floyd_steinberg(small, color_levels)
    else:
        simple_levels = max(2, int(color_levels))
        small = (small * (simple_levels - 1)).round() / (simple_levels - 1)

    # --- 3. Upscale back with nearest-neighbor (preserving pixel edges) ---
    result = np.array(
        Image.fromarray((np.clip(small, 0, 1) * 255).astype(np.uint8))
        .resize((w, h), Image.Resampling.NEAREST)
    ).astype(np.float32) / 255.0

    return np.clip(result, 0, 1)


def _floyd_steinberg(img, levels):
    """Apply Floyd-Steinberg error diffusion dithering.

    img:     float array (0-1) shape (H, W, C)
    levels:  number of quantization levels per channel (2-8)
    """
    h, w = img.shape[:2]
    n = max(2, int(levels))
    result = img.copy()

    for y in range(h):
        for x in range(w):
            old = result[y, x].copy()
            new = (old * (n - 1)).round() / (n - 1)
            result[y, x] = new
            error = old - new

            # Distribute error to neighboring pixels (Floyd-Steinberg pattern)
            if x + 1 < w:
                result[y, x + 1] += error * 7.0 / 16.0
            if y + 1 < h:
                if x > 0:
                    result[y + 1, x - 1] += error * 3.0 / 16.0
                result[y + 1, x] += error * 5.0 / 16.0
                if x + 1 < w:
                    result[y + 1, x + 1] += error * 1.0 / 16.0

    return np.clip(result, 0, 1)


def _retro_palette_quantize(img, palette_name="nes"):
    """Quantize image to a fixed retro game console palette.

    palette_name: one of 'nes', 'gameboy', 'cga', 'cga-high', 'zxs', 'pico8'
    """
    PALETTES = {
        "nes": [
            [0x00, 0x00, 0x00], [0xFC, 0xFC, 0xFC], [0xF8, 0xF8, 0xF8], [0xBC, 0xBC, 0xBC],
            [0xA0, 0xA0, 0xA0], [0x80, 0x80, 0x80], [0x60, 0x60, 0x60], [0x40, 0x40, 0x40],
            [0x00, 0x00, 0xFC], [0x00, 0x00, 0xBC], [0x00, 0x00, 0xA0], [0x00, 0x00, 0x80],
            [0x00, 0x00, 0x60], [0x00, 0x00, 0x40], [0x00, 0x00, 0x20], [0x20, 0x20, 0x20],
            [0xFC, 0x00, 0x00], [0xBC, 0x00, 0x00], [0xA0, 0x00, 0x00], [0x80, 0x00, 0x00],
            [0x60, 0x00, 0x00], [0x40, 0x00, 0x00], [0x20, 0x00, 0x00], [0x00, 0xFC, 0x00],
            [0x00, 0xBC, 0x00], [0x00, 0xA0, 0x00], [0x00, 0x80, 0x00], [0x00, 0x60, 0x00],
            [0x00, 0x40, 0x00], [0x00, 0x20, 0x00], [0xFC, 0xFC, 0x00], [0xBC, 0xBC, 0x00],
            [0xA0, 0xA0, 0x00], [0x80, 0x80, 0x00], [0x60, 0x60, 0x00], [0x40, 0x40, 0x00],
            [0x20, 0x20, 0x00], [0xFC, 0x00, 0xFC], [0xBC, 0x00, 0xBC], [0xA0, 0x00, 0xA0],
            [0x80, 0x00, 0x80], [0x60, 0x00, 0x60], [0x40, 0x00, 0x40], [0x20, 0x00, 0x20],
            [0x00, 0xFC, 0xFC], [0x00, 0xBC, 0xBC], [0x00, 0xA0, 0xA0], [0x00, 0x80, 0x80],
            [0x00, 0x60, 0x60], [0x00, 0x40, 0x40], [0x00, 0x20, 0x20], [0xFC, 0xA0, 0x00],
            [0xFC, 0x80, 0x00], [0xFC, 0x60, 0x00], [0xFC, 0x40, 0x00], [0xFC, 0x20, 0x00],
            [0xA0, 0x50, 0x00], [0x80, 0x40, 0x00], [0x60, 0x30, 0x00], [0x40, 0x20, 0x00],
        ],
        "gameboy": [
            [0x0F, 0x38, 0x0F], [0x30, 0x62, 0x30],
            [0x8B, 0xAC, 0x0F], [0x9B, 0xBC, 0x0F],
            [0xCB, 0xD8, 0x6C], [0xE0, 0xE8, 0xC0],
        ],
        "cga": [
            [0x00, 0x00, 0x00], [0x00, 0xAA, 0x00], [0xAA, 0x00, 0x00], [0xAA, 0x55, 0x00],
            [0x00, 0x00, 0xAA], [0x00, 0xAA, 0xAA], [0xAA, 0x00, 0xAA], [0xAA, 0xAA, 0xAA],
            [0x55, 0x55, 0x55], [0x55, 0xFF, 0x55], [0xFF, 0x55, 0x55], [0xFF, 0xFF, 0x55],
            [0x55, 0x55, 0xFF], [0x55, 0xFF, 0xFF], [0xFF, 0x55, 0xFF], [0xFF, 0xFF, 0xFF],
        ],
        "cga-high": [
            [0x00, 0x00, 0x00], [0x00, 0xFF, 0x00], [0xFF, 0x00, 0x00], [0xFF, 0xFF, 0x00],
            [0x00, 0x00, 0xFF], [0x00, 0xFF, 0xFF], [0xFF, 0x00, 0xFF], [0xFF, 0xFF, 0xFF],
        ],
        "zxs": [
            [0x00, 0x00, 0x00], [0x00, 0x00, 0xD7], [0xD7, 0x00, 0x00],
            [0xD7, 0x00, 0xD7], [0x00, 0xD7, 0x00], [0x00, 0xD7, 0xD7],
            [0xD7, 0xD7, 0x00], [0xD7, 0xD7, 0xD7],
        ],
        "pico8": [
            [0x00, 0x00, 0x00], [0x1D, 0x2B, 0x53], [0x7E, 0x25, 0x53], [0x00, 0x87, 0x51],
            [0xAB, 0x52, 0x36], [0x5F, 0x57, 0x4F], [0xC2, 0xC3, 0xC7], [0xFF, 0xF1, 0xE8],
            [0xFF, 0x00, 0x4D], [0xFF, 0xA3, 0x00], [0xFF, 0xEC, 0x27], [0x00, 0xE4, 0x36],
            [0x29, 0xAD, 0xFF], [0x83, 0x76, 0x9C], [0xFF, 0x77, 0xA8], [0xFF, 0xCC, 0xAA],
        ],
    }

    pal = np.array(PALETTES.get(palette_name, PALETTES["nes"]), dtype=np.float32) / 255.0
    img_flat = img.reshape(-1, 3)
    # For each pixel, find nearest palette color
    distances = np.sum((img_flat[:, np.newaxis, :] - pal[np.newaxis, :, :]) ** 2, axis=2)
    nearest = np.argmin(distances, axis=1)
    result = pal[nearest].reshape(img.shape)
    return result.astype(np.float32)


def _remove_bg(img, bg_color=(255, 255, 255), tolerance=30):
    """Remove a solid background color and make it transparent (RGBA output).

    img:        (H, W, 3) float array (0-1) or uint8 (0-255)
    bg_color:   RGB tuple of the color to remove
    tolerance:  how far from bg_color to still consider as background (0-255)

    Returns:    (H, W, 4) uint8 array with alpha channel
    """
    # Handle both float and uint8 input
    if img.dtype == np.float32 or img.dtype == np.float64:
        img_8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    else:
        img_8 = np.array(img, dtype=np.uint8)

    h, w = img_8.shape[:2]
    r, g, b = img_8[..., 0].astype(float), img_8[..., 1].astype(float), img_8[..., 2].astype(float)
    dist = np.sqrt((r - bg_color[0])**2 + (g - bg_color[1])**2 + (b - bg_color[2])**2)

    # Smooth alpha transition
    alpha = np.clip(dist / tolerance * 255, 0, 255).astype(np.uint8)
    # Fully opaque where distance > tolerance
    alpha = np.where(dist > tolerance, 255, alpha)

    return np.dstack([img_8, alpha])


def _trim_alpha(img_rgba, shrink_px=1):
    """Shrink alpha inward to clean up edge aliasing/halos.

    img_rgba:   (H, W, 4) uint8 RGBA array
    shrink_px:  number of pixels to erode/shrink the alpha mask

    Detects the bounding box of non-transparent content and shrinks it
    inward by shrink_px pixels. Also applies a light Gaussian blur to
    the alpha edge to feather residual white fringe.
    """
    alpha = img_rgba[..., 3]

    # Find content bounding box (rows/cols with any non-transparent pixel)
    rows = np.any(alpha > 0, axis=1)
    cols = np.any(alpha > 0, axis=0)
    if not np.any(rows) or not np.any(cols):
        return img_rgba  # nothing visible

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    # Shrink inward
    y_min = min(y_min + shrink_px, y_max)
    y_max = max(y_max - shrink_px, y_min)
    x_min = min(x_min + shrink_px, x_max)
    x_max = max(x_max - shrink_px, x_min)

    cropped = img_rgba[y_min:y_max+1, x_min:x_max+1].copy()

    # Feather alpha edges with a light blur to clean white fringe
    from PIL import ImageFilter
    alpha_crop = cropped[..., 3]
    # Blur alpha slightly, then threshold to keep core, feather edges
    alpha_blur = np.array(
        Image.fromarray(alpha_crop).filter(ImageFilter.GaussianBlur(radius=0.6))
    )
    # Blend: keep core fully opaque, edges get blurred alpha
    cropped[..., 3] = alpha_blur

    return cropped


def _comic_print(img, amount, dot_size=4):
    """Comic/manga halftone print effect — Ben-Day dots, bold colors, inked edges.

    amount:   0-1, intensity
    dot_size: pixel size of halftone dots (3-8)
    """
    if amount < 0.01:
        return img
    h, w = img.shape[:2]

    # 1. Boost contrast and saturation (comic ink vibrancy)
    img = (img - 0.5) * 1.3 + 0.5
    gray = img.mean(axis=2, keepdims=True)
    img = gray + (img - gray) * 1.4
    img = np.clip(img, 0, 1)

    # 2. Posterize to limited color levels
    levels = max(3, int(8 - amount * 3))
    img = (img * (levels - 1)).round() / (levels - 1)

    # 3. Edge inking — Sobel edges become black lines
    gray_uint8 = (img.mean(axis=2) * 255).astype(np.uint8)
    sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    gx = np.zeros_like(gray_uint8, dtype=np.float32)
    gy = np.zeros_like(gray_uint8, dtype=np.float32)
    for dy in range(3):
        for dx in range(3):
            gx += np.roll(np.roll(gray_uint8, dy - 1, 0), dx - 1, 1) * sx[dy, dx]
            gy += np.roll(np.roll(gray_uint8, dy - 1, 0), dx - 1, 1) * sy[dy, dx]
    edges = np.sqrt(gx ** 2 + gy ** 2)
    edges = edges / (edges.max() + 1e-8)
    ink = (edges > 0.25).astype(np.float32) * 0.3 * amount
    img = img * (1 - ink[..., np.newaxis])

    # 4. Halftone Ben-Day dots in darker areas
    dot_size = max(3, int(dot_size))
    for y in range(0, h, dot_size):
        for x in range(0, w, dot_size):
            cy = min(y + dot_size // 2, h - 1)
            cx = min(x + dot_size // 2, w - 1)
            b = img[cy, cx].mean()
            if b < 0.65:
                r = int(dot_size // 2 * (1 - b * 1.2))
                r = max(1, min(r, dot_size // 2))
                yy, xx = np.ogrid[-dot_size//2:dot_size//2+1, -dot_size//2:dot_size//2+1]
                if yy.size > 0 and xx.size > 0:
                    dot = np.sqrt(yy**2 + xx**2) <= r
                    y0, y1 = y, min(y + dot_size, h)
                    x0, x1 = x, min(x + dot_size, w)
                    dh, dw = y1 - y0, x1 - x0
                    img[y0:y1, x0:x1] *= np.where(dot[:dh, :dw], 0.55, 1.0)[..., np.newaxis]

    return np.clip(img, 0, 1)


# ═══════════════════════════════════════════════════════════════════
#  Film profiles
# ═══════════════════════════════════════════════════════════════════

PROFILES = {}

PROFILES["classic-chrome"] = {
    "label": "Fujifilm Classic Chrome",
    "description": "Muted, warm shadows, cool midtones, reds preserved",
    "color_matrix": [
        [1.08, -0.05, -0.03],
        [-0.04, 1.03,  0.01],
        [0.01,  0.03,  0.96],
    ],
    "curve": {
        "toe": 0.10, "shoulder": 0.88,
        "toe_slope": 0.15, "shoulder_slope": 0.10,
        "contrast": 0.95, "lift": 0.01, "gamma": 0.97,
    },
    "grade": {
        "shadows":    [0.020, 0.005, -0.005],
        "midtones":   [-0.005, 0.002, 0.010],
        "highlights": [0.005, 0.002, -0.002],
    },
    "hue_adjust": [
        (200, 50, 0.65, 3),
        (120, 40, 0.70, 8),
        (30,  20, 0.90, 0),
    ],
    "grain": 0.30,
    "halation": 0.0,
    "fade": 0.0,
    "saturation": 0.75,
}

PROFILES["velvia"] = {
    "label": "Fujifilm Velvia",
    "description": "Vivid, high saturation, high contrast — landscape film",
    "color_matrix": [
        [1.15, -0.10, -0.05],
        [-0.08, 1.12, -0.04],
        [-0.03, -0.05, 1.08],
    ],
    "curve": {
        "toe": 0.06, "shoulder": 0.90,
        "toe_slope": 0.08, "shoulder_slope": 0.06,
        "contrast": 1.10, "lift": 0.0, "gamma": 1.0,
    },
    "hue_adjust": [
        (120, 50, 1.20, 0),
        (200, 40, 1.15, 0),
        (0,   25, 1.10, 0),
    ],
    "grain": 0.15,
    "saturation": 1.20,
}

PROFILES["provia"] = {
    "label": "Fujifilm Provia",
    "description": "Natural, balanced, versatile all-purpose film",
    "color_matrix": [
        [1.08, -0.05, -0.03],
        [-0.03, 1.05, -0.02],
        [-0.01, -0.02, 1.03],
    ],
    "curve": {
        "toe": 0.06, "shoulder": 0.90,
        "toe_slope": 0.08, "shoulder_slope": 0.06,
        "contrast": 1.05, "lift": 0.0, "gamma": 1.0,
    },
    "grade": {
        "shadows":    [0.0, 0.0, 0.0],
        "midtones":   [0.0, 0.0, 0.0],
        "highlights": [0.0, 0.0, 0.0],
    },
    "hue_adjust": [],
    "grain": 0.15,
    "halation": 0.0,
    "fade": 0.0,
    "saturation": 1.0,
}

PROFILES["astia"] = {
    "label": "Fujifilm Astia",
    "description": "Soft contrast, gentle saturation — ideal for portraits",
    "color_matrix": [
        [1.05, -0.03, -0.02],
        [-0.02, 1.03, -0.01],
        [-0.01, -0.01, 1.02],
    ],
    "curve": {
        "toe": 0.10, "shoulder": 0.88,
        "toe_slope": 0.15, "shoulder_slope": 0.10,
        "contrast": 0.90, "lift": 0.01, "gamma": 0.95,
    },
    "grade": {
        "shadows":    [0.01, 0.0, 0.0],
        "midtones":   [0.0, 0.0, 0.01],
        "highlights": [0.01, 0.0, 0.0],
    },
    "hue_adjust": [
        (30, 30, 0.80, 0),   # oranges → slightly muted (good for skin)
    ],
    "grain": 0.20,
    "halation": 0.0,
    "fade": 0.0,
    "saturation": 0.85,
}

PROFILES["portra"] = {
    "label": "Kodak Portra",
    "description": "Warm skin tones, soft contrast, pastel palette",
    "color_matrix": [
        [1.10, -0.06, -0.04],
        [-0.04, 1.02,  0.02],
        [-0.01,  0.04, 0.97],
    ],
    "curve": {
        "toe": 0.12, "shoulder": 0.86,
        "toe_slope": 0.18, "shoulder_slope": 0.12,
        "contrast": 0.92, "lift": 0.015, "gamma": 0.93,
    },
    "grade": {
        "shadows":    [0.025, 0.005, -0.015],
        "midtones":   [0.01,  0.0,    0.0],
        "highlights": [0.02,  0.005, -0.01],
    },
    "hue_adjust": [
        (200, 50, 0.55, 5),    # blues desaturate
        (120, 45, 0.65, 10),   # greens desaturate, warm shift
        (30,  30, 0.85, 0),    # oranges/skin slightly muted
    ],
    "grain": 0.40,
    "halation": 0.0,
    "fade": 0.02,
    "saturation": 0.80,
}

PROFILES["tri-x"] = {
    "label": "Kodak Tri-X 400",
    "description": "Classic B&W, contrasty, grainy",
    "bw": True,
    "bw_weights": [0.30, 0.59, 0.11],
    "curve": {
        "toe": 0.10, "shoulder": 0.82,
        "toe_slope": 0.12, "shoulder_slope": 0.15,
        "contrast": 1.15, "lift": 0.0, "gamma": 0.95,
    },
    "grain": 0.80,
    "halation": 0.0,
    "fade": 0.0,
}

PROFILES["gold"] = {
    "label": "Kodak Gold 200",
    "description": "Warm consumer film, golden highlights",
    "color_matrix": [
        [1.12, -0.08, -0.04],
        [-0.05, 1.04,  0.01],
        [-0.02,  0.02, 1.00],
    ],
    "curve": {
        "toe": 0.10, "shoulder": 0.87,
        "toe_slope": 0.15, "shoulder_slope": 0.10,
        "contrast": 0.98, "lift": 0.01, "gamma": 0.95,
    },
    "grade": {
        "shadows":    [0.02, 0.0, -0.01],
        "midtones":   [0.015, 0.0, -0.005],
        "highlights": [0.025, 0.01, -0.01],
    },
    "hue_adjust": [
        (200, 50, 0.60, 5),    # blues desaturate
        (120, 45, 0.70, 15),   # greens warm
        (30,  25, 1.05, 2),    # oranges/skin preserved
    ],
    "grain": 0.45,
    "halation": 0.0,
    "fade": 0.02,
    "saturation": 0.85,
}

PROFILES["cinestill"] = {
    "label": "Cinestill 800T",
    "description": "Cinematic tungsten, teal shadows, halation",
    "color_matrix": [
        [1.08, -0.04, -0.04],
        [-0.05, 1.04,  0.01],
        [-0.01,  0.02, 0.99],
    ],
    "curve": {
        "toe": 0.07, "shoulder": 0.89,
        "toe_slope": 0.08, "shoulder_slope": 0.10,
        "contrast": 1.05, "lift": 0.0, "gamma": 0.95,
    },
    "grade": {
        "shadows":    [-0.01, -0.005, 0.03],
        "midtones":   [-0.005, 0.0,   0.01],
        "highlights": [0.015, 0.005, -0.01],
    },
    "hue_adjust": [
        (200, 45, 0.75, -3),
        (30,  25, 1.05, 3),
    ],
    "grain": 0.30,
    "halation": 0.10,
    "saturation": 0.90,
}

PROFILES["bw-high-contrast"] = {
    "label": "B&W High Contrast",
    "description": "Dramatic black and white",
    "bw": True,
    "bw_weights": [0.35, 0.55, 0.10],
    "curve": {
        "toe": 0.05, "shoulder": 0.80,
        "toe_slope": 0.05, "shoulder_slope": 0.08,
        "contrast": 1.40, "lift": 0.0, "gamma": 1.0,
    },
    "grain": 0.50,
    "halation": 0.0,
    "fade": 0.0,
}

PROFILES["faded"] = {
    "label": "Faded Vintage",
    "description": "Faded, yellow/pink vintage film look",
    "color_matrix": [
        [0.95,  0.05,  0.0],
        [-0.02, 0.98,  0.04],
        [0.02,  0.06,  0.92],
    ],
    "curve": {
        "toe": 0.20, "shoulder": 0.82,
        "toe_slope": 0.30, "shoulder_slope": 0.15,
        "contrast": 0.80, "lift": 0.06, "gamma": 0.90,
    },
    "grade": {
        "shadows":    [0.04, 0.01, -0.02],
        "midtones":   [0.03, 0.01, -0.01],
        "highlights": [0.05, 0.02, -0.01],
    },
    "hue_adjust": [
        (200, 50, 0.40, 10),    # blues very desaturated
        (120, 45, 0.50, 20),   # greens desaturated warm
        (30,  30, 0.75, 0),    # oranges muted
    ],
    "grain": 0.55,
    "halation": 0.0,
    "fade": 0.10,
    "saturation": 0.55,
}

PROFILES["portra-bw"] = {
    "label": "Kodak Portra (B&W conversion)",
    "description": "Kodak Portra conversion to B&W with smooth tonality",
    "bw": True,
    "bw_weights": [0.35, 0.55, 0.10],
    "curve": {
        "toe": 0.08, "shoulder": 0.88,
        "toe_slope": 0.12, "shoulder_slope": 0.10,
        "contrast": 1.0, "lift": 0.01, "gamma": 0.95,
    },
    "grain": 0.50,
    "halation": 0.0,
    "fade": 0.01,
}

PROFILES["ccd"] = {
    "label": "CCD Sensor",
    "description": "Early digital CCD look — cool tones, punchy blues, abrupt highlight clip",
    "color_matrix": [
        [1.02,  0.01, -0.03],
        [0.0,   1.02, -0.02],
        [-0.03, 0.04,  0.99],
    ],
    "curve": {
        "toe": 0.08, "shoulder": 0.82,
        "toe_slope": 0.10, "shoulder_slope": 0.05,
        "contrast": 1.05, "lift": 0.005, "gamma": 0.98,
    },
    "grade": {
        "shadows":    [0.005, 0.0, 0.015],
        "midtones":   [-0.005, 0.0, 0.010],
        "highlights": [0.0, 0.0, 0.005],
    },
    "hue_adjust": [
        (200, 40, 1.15, 2),
        (120, 35, 0.90, 0),
        (30,  20, 1.0, 0),
    ],
    "grain": 0.15,
    "saturation": 0.95,
    "vignette": 0.0,
}

PROFILES["leica"] = {
    "label": "Leica Look",
    "description": "Leica rendering — micro contrast, warm tones, smooth transitions, 3D pop",
    "color_matrix": [
        [1.06, -0.03, -0.03],
        [-0.02, 1.04, -0.02],
        [-0.01, 0.02,  0.99],
    ],
    "curve": {
        "toe": 0.07, "shoulder": 0.90,
        "toe_slope": 0.10, "shoulder_slope": 0.08,
        "contrast": 1.08, "lift": 0.005, "gamma": 0.95,
    },
    "grade": {
        "shadows":    [0.015, 0.005, -0.005],
        "midtones":   [0.005, 0.0, 0.005],
        "highlights": [0.010, 0.005, -0.005],
    },
    "hue_adjust": [
        (200, 35, 0.85, 3),
        (120, 30, 0.90, 5),
        (30,  20, 1.0, 0),
    ],
    "grain": 0.12,
    "saturation": 0.90,
    "vignette": 0.08,
}

PROFILES["cartoon"] = {
    "label": "Cartoon",
    "description": "Turn photo into cartoon/anime style with cel-shading and edge lines",
    "special": "cartoonize",
    "edge_thickness": 0.55,
    "color_levels": 7,
}

PROFILES["cartoon-inked"] = {
    "label": "Cartoon Inked",
    "description": "Cartoon with bold ink line art — thick black outlines, flat colors",
    "special": "cartoonize_inked",
    "edge_threshold": 0.15,
    "color_levels": 6,
}

PROFILES["lineart"] = {
    "label": "Line Art",
    "description": "Pure line art / sketch — XDoG-based, white bg with black strokes",
    "special": "lineart",
    "line_sensitivity": 0.04,
    "line_width": 0,
    "clean_bg": True,
}

PROFILES["print"] = {
    "label": "Comic Print",
    "description": "Comic book / manga halftone print — Ben-Day dots, bold inks, inked edges",
    "special": "comic_print",
    "print_amount": 0.60,
    "dot_size": 8,
}

PROFILES["pixel"] = {
    "label": "Retro Pixel Art",
    "description": "Retro game pixelation — lower resolution + color posterization, like 8-bit/16-bit game art",
    "special": "pixelate",
    "block_size": 8,
    "color_levels": 4,
    "dither": False,
}

PROFILES["pixel-dithered"] = {
    "label": "Retro Pixel Art (Dithered)",
    "description": "Pixel art with Floyd-Steinberg dithering for smoother color transitions",
    "special": "pixelate",
    "block_size": 8,
    "color_levels": 3,
    "dither": True,
}

PROFILES["pixel-nes"] = {
    "label": "NES Palette Pixel Art",
    "description": "Pixel art quantized to authentic NES color palette",
    "special": "pixelate_retro",
    "block_size": 8,
    "palette": "nes",
}

PROFILES["pixel-gameboy"] = {
    "label": "Game Boy Pixel Art",
    "description": "Classic Game Boy look — 4-shade green palette, pixelated",
    "special": "pixelate_retro",
    "block_size": 6,
    "palette": "gameboy",
}

PROFILES["pixel-pico8"] = {
    "label": "PICO-8 Pixel Art",
    "description": "PICO-8 fantasy console look — 16-color palette, chunky pixels",
    "special": "pixelate_retro",
    "block_size": 6,
    "palette": "pico8",
}


# ═══════════════════════════════════════════════════════════════════
#  Main processing pipeline
# ═══════════════════════════════════════════════════════════════════

def apply_profile(img_np, profile, strength=1.0):
    """Apply a film profile to an 8-bit RGB image."""
    img = img_np.astype(np.float32) / 255.0

    # --- 1. B&W conversion (early, before color work) ---
    if profile.get("bw"):
        w = profile["bw_weights"]
        gray = img[..., 0] * w[0] + img[..., 1] * w[1] + img[..., 2] * w[2]
        img = np.stack([gray, gray, gray], axis=-1)

    # --- 2. Linearize ---
    r_lin = _srgb_to_lin(img[..., 0])
    g_lin = _srgb_to_lin(img[..., 1])
    b_lin = _srgb_to_lin(img[..., 2])

    # --- 3. Color matrix transform ---
    matrix = profile.get("color_matrix")
    if matrix:
        r_lin, g_lin, b_lin = _apply_color_matrix(r_lin, g_lin, b_lin, matrix)

    # --- 4. Three-way color grade ---
    grade = profile.get("grade")
    if grade:
        r_lin, g_lin, b_lin = _three_way_grade(
            r_lin, g_lin, b_lin,
            grade.get("shadows", (0,0,0)),
            grade.get("midtones", (0,0,0)),
            grade.get("highlights", (0,0,0)),
        )

    # --- 5. Convert back to sRGB for curve ---
    r = _lin_to_srgb(r_lin)
    g = _lin_to_srgb(g_lin)
    b = _lin_to_srgb(b_lin)

    # --- 6. Film curve ---
    cv = profile.get("curve", {})
    r = _film_curve(r, **cv)
    g = _film_curve(g, **cv)
    b = _film_curve(b, **cv)
    img = np.stack([r, g, b], axis=-1)

    # --- 7. Global saturation ---
    sat = profile.get("saturation", 1.0)
    if abs(sat - 1.0) > 0.001 and not profile.get("bw"):
        gray = img.mean(axis=2, keepdims=True)
        img = gray + (img - gray) * sat

    # --- 8. Per-hue adjustments ---
    hue_adj = profile.get("hue_adjust", [])
    if hue_adj and not profile.get("bw"):
        r, g, b = img[..., 0], img[..., 1], img[..., 2]
        r, g, b = _hue_adjust(r, g, b, hue_adj)
        img = np.stack([r, g, b], axis=-1)

    # --- 9. Halation ---
    hal = profile.get("halation", 0)
    if hal > 0:
        img = _add_halation(img, intensity=hal)

    # --- 10. Grain ---
    grain = profile.get("grain", 0)
    if grain > 0:
        img = _add_grain(img, grain)

    # --- 11. Vignette ---
    vig = profile.get("vignette", 0)
    if vig > 0:
        img = _vignette(img, vig)

    # --- 12. Fade ---
    fade = profile.get("fade", 0)
    if fade > 0:
        img = _fade(img, fade)

    # --- 13. Special effects (cartoon / print) ---
    special = profile.get("special")
    if special == "cartoonize":
        img = _cartoonize(img, edge_thickness=profile.get("edge_thickness", 0.5),
                          color_levels=profile.get("color_levels", 8))
    elif special == "cartoonize_inked":
        img = _cartoon_inked(img, edge_threshold=profile.get("edge_threshold", 0.15),
                             color_levels=profile.get("color_levels", 6),
                             line_width=profile.get("line_width", 1))
    elif special == "lineart":
        img = _lineart(img, sensitivity=profile.get("line_sensitivity", 0.04),
                       line_width=profile.get("line_width", 0),
                       clean_bg=profile.get("clean_bg", True))
    elif special == "comic_print":
        img = _comic_print(img, amount=profile.get("print_amount", 0.5),
                           dot_size=profile.get("dot_size", 4))
    elif special == "pixelate":
        img = _pixelate(img, block_size=profile.get("block_size", 8),
                        color_levels=profile.get("color_levels", 4),
                        dither=profile.get("dither", False))
    elif special == "pixelate_retro":
        img = _pixelate(img, block_size=profile.get("block_size", 8),
                        color_levels=8, dither=False)
        img = _retro_palette_quantize(img, profile.get("palette", "nes"))

    # --- 14. Strength blend ---
    if strength < 1.0:
        orig = img_np.astype(np.float32) / 255.0
        img = orig * (1 - strength) + img * strength

    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def list_profiles():
    data = {}
    for name, p in PROFILES.items():
        data[name] = {"label": p["label"], "description": p["description"]}
    print(json.dumps(data, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Apply film simulation to an image")
    parser.add_argument("input", nargs="?", help="Input image path")
    parser.add_argument("output", nargs="?", help="Output image path (default: input_{profile}.ext)")
    parser.add_argument("--profile", "-p", default="classic-chrome", help="Film profile name")
    parser.add_argument("--strength", "-s", type=float, default=1.0, help="Effect strength 0-1")
    parser.add_argument("--block-size", type=int, default=0, help="Pixel block size for pixel/retro profiles (overrides profile default)")
    parser.add_argument("--color-levels", type=int, default=0, help="Color quantization levels for pixel profiles (overrides profile default)")
    parser.add_argument("--dither", action="store_true", default=None, help="Enable dithering for pixel profiles")
    parser.add_argument("--no-dither", action="store_false", dest="dither", default=None, help="Disable dithering for pixel profiles")
    parser.add_argument("--palette", type=str, default=None, help="Retro palette name: nes, gameboy, cga, pico8")
    parser.add_argument("--remove-bg", nargs="?", const="255,255,255", default=None,
                        help="Remove background color and output RGBA PNG. Optionally specify color as R,G,B (default: 255,255,255 white)")
    parser.add_argument("--bg-tolerance", type=int, default=30, help="Tolerance for background removal 0-255 (default: 30)")
    parser.add_argument("--shrink", type=int, default=0,
                        help="Shrink/trim inward by N pixels to clean edge aliasing (default: 0)")
    parser.add_argument("--list", action="store_true", help="List available profiles")
    parser.add_argument("--quality", type=int, default=95, help="JPEG save quality")
    args = parser.parse_args()

    if args.list:
        list_profiles()
        return

    if args.profile not in PROFILES:
        print(f"Unknown profile '{args.profile}'. Use --list to see available.", file=sys.stderr)
        sys.exit(1)

    if not args.output:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_{args.profile}{ext}"

    # Merge CLI overrides into profile
    profile = dict(PROFILES[args.profile])
    if args.block_size > 0:
        profile["block_size"] = args.block_size
    if args.color_levels > 0:
        profile["color_levels"] = args.color_levels
    if args.dither is not None:
        profile["dither"] = args.dither
    if args.palette is not None:
        profile["palette"] = args.palette

    print(f"Applying: {PROFILES[args.profile]['label']}")
    img = Image.open(args.input).convert("RGB")
    result = apply_profile(np.array(img), profile, strength=args.strength)

    # --- Post-processing: background removal + trim ---
    need_alpha = args.remove_bg is not None or args.shrink > 0

    if args.remove_bg is not None:
        bg_color = (255, 255, 255)
        if args.remove_bg != "":
            try:
                parts = [int(x.strip()) for x in args.remove_bg.split(",")]
                if len(parts) == 3:
                    bg_color = tuple(parts)
            except ValueError:
                pass
        result = _remove_bg(result, bg_color=bg_color, tolerance=args.bg_tolerance)
        # Ensure output is PNG if we have alpha
        base_no_ext, _ = os.path.splitext(args.output)
        args.output = base_no_ext + ".png"
        print(f"  Background removed (color={bg_color}, tolerance={args.bg_tolerance})")

    if args.shrink > 0:
        if not need_alpha:
            # Convert to RGBA first
            result = np.dstack([result, np.full((result.shape[0], result.shape[1]), 255, dtype=np.uint8)])
            need_alpha = True
        result = _trim_alpha(result, shrink_px=args.shrink)
        # Update output to PNG if not already
        base_no_ext, ext = os.path.splitext(args.output)
        if ext.lower() != ".png":
            args.output = base_no_ext + ".png"
        print(f"  Trimmed/shrunk by {args.shrink}px")

    if need_alpha:
        Image.fromarray(result, "RGBA").save(args.output)
    else:
        Image.fromarray(result).save(args.output, quality=args.quality)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
