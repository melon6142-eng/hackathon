#!/usr/bin/env python3
"""DAKER microstructure transform gallery + subtle-difference amplification."""
from __future__ import annotations

import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
_KFONT = "/usr/share/fonts/truetype/sand-box/google/Nanum Gothic/NanumGothic-Regular.ttf"
font_manager.fontManager.addfont(_KFONT)
plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from skimage.feature import peak_local_max, structure_tensor, structure_tensor_eigenvalues
from skimage.filters import gaussian, sobel
from skimage.measure import label, regionprops
from skimage.morphology import disk, local_minima
from skimage.segmentation import watershed

ROOT = Path("/workspace/daker")
TRAIN_DIR = ROOT / "open" / "train"
CSV_PATH = ROOT / "open" / "train.csv"
OUT = ROOT / "runs" / "viz"
ROWS_DIR = OUT / "rows"
PANELS_DIR = OUT / "panels"
AMP_DIR = OUT / "amplify"
N_THREADS = 2

# Stage colors: brighter -> cooler, darker -> hotter
STAGE_COLORS = {
    "brightest": (180, 220, 255),  # pale blue
    "bright": (120, 200, 255),
    "stage0": (200, 200, 200),
    "stage1": (255, 220, 80),
    "stage2": (255, 140, 40),
    "stage3": (230, 50, 30),
    "pore": (80, 0, 100),
}
STAGE_ORDER = [
    ("brightest", "> +1σ", STAGE_COLORS["brightest"]),
    ("bright", "+0.5~+1σ", STAGE_COLORS["bright"]),
    ("stage0", "±0.5σ", STAGE_COLORS["stage0"]),
    ("stage1", "−0.5~−1σ", STAGE_COLORS["stage1"]),
    ("stage2", "−1~−1.5σ", STAGE_COLORS["stage2"]),
    ("stage3", "−1.5~−2.5σ", STAGE_COLORS["stage3"]),
    ("pore", "< −2.5σ", STAGE_COLORS["pore"]),
]


def load_meta():
    rows = [(r["ID"], float(r["hardness"])) for r in csv.DictReader(open(CSV_PATH))]
    rows.sort(key=lambda x: x[1])
    return rows


def load_gray(iid: str) -> np.ndarray:
    p = TRAIN_DIR / f"{iid}.png"
    im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise FileNotFoundError(p)
    return im.astype(np.float32)


def pick_samples(rows):
    low6 = rows[:6]
    high6 = rows[-6:]
    mids = sorted(rows, key=lambda x: abs(x[1] - 194.0))[:4]
    # stable order: high desc, mid by closeness, low asc
    high6 = list(reversed(high6))
    low6 = list(low6)
    samples = []
    for iid, h in high6:
        samples.append(("high", iid, h))
    for iid, h in mids:
        samples.append(("mid", iid, h))
    for iid, h in low6:
        samples.append(("low", iid, h))
    return samples


def compute_global_stats(rows):
    """Scan all train images for global percentiles and absolute stage thresholds."""
    # reservoir of pixels for percentiles
    rng = np.random.default_rng(0)
    samples = []
    means = []
    stds = []
    for iid, h in rows:
        g = load_gray(iid)
        means.append(float(g.mean()))
        stds.append(float(g.std()))
        flat = g.reshape(-1)
        idx = rng.choice(flat.size, size=min(4000, flat.size), replace=False)
        samples.append(flat[idx])
    pix = np.concatenate(samples)
    p1, p99 = np.percentile(pix, [1, 99])
    # Absolute brightness stages using global mean/std of pixel values
    g_mean = float(pix.mean())
    g_std = float(pix.std() + 1e-6)
    # also collect for narrow stretch: typical mid-gray band
    p20, p80 = np.percentile(pix, [20, 80])
    return {
        "p1": float(p1),
        "p99": float(p99),
        "p20": float(p20),
        "p80": float(p80),
        "g_mean": g_mean,
        "g_std": g_std,
        "img_mean_mean": float(np.mean(means)),
        "img_mean_std": float(np.std(means) + 1e-6),
    }


def clahe_img(g: np.ndarray) -> np.ndarray:
    u8 = np.clip(g, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(u8)


def common_scale(g: np.ndarray, lo: float, hi: float, gamma: float = 0.85) -> np.ndarray:
    x = (g - lo) / max(hi - lo, 1e-6)
    x = np.clip(x, 0, 1)
    x = np.power(x, gamma)
    return (x * 255).astype(np.uint8)


def narrow_scale(g: np.ndarray, lo: float = 120.0, hi: float = 170.0) -> np.ndarray:
    """Aggressive stretch of a tight band so 2~3 gray-level diffs pop."""
    x = (g - lo) / max(hi - lo, 1e-6)
    x = np.clip(x, 0, 1)
    return (x * 255).astype(np.uint8)


def stage_from_z(z: np.ndarray) -> np.ndarray:
    """Return RGB stage map from z-scores. Darker = hotter."""
    out = np.zeros(z.shape + (3,), dtype=np.uint8)
    out[:] = STAGE_COLORS["stage0"]
    out[z > 1.0] = STAGE_COLORS["brightest"]
    out[(z > 0.5) & (z <= 1.0)] = STAGE_COLORS["bright"]
    out[(z >= -0.5) & (z <= 0.5)] = STAGE_COLORS["stage0"]
    out[(z >= -1.0) & (z < -0.5)] = STAGE_COLORS["stage1"]
    out[(z >= -1.5) & (z < -1.0)] = STAGE_COLORS["stage2"]
    out[(z >= -2.5) & (z < -1.5)] = STAGE_COLORS["stage3"]
    out[z < -2.5] = STAGE_COLORS["pore"]
    return out


def relative_stage_map(g: np.ndarray) -> np.ndarray:
    sm = gaussian(g, sigma=2.0, preserve_range=True)
    med = float(np.median(sm))
    std = float(sm.std() + 1e-6)
    z = (sm - med) / std
    return stage_from_z(z)


def absolute_stage_map(g: np.ndarray, g_mean: float, g_std: float) -> np.ndarray:
    sm = gaussian(g, sigma=2.0, preserve_range=True)
    z = (sm - g_mean) / g_std
    return stage_from_z(z)


def add_legend_bar(rgb: np.ndarray, title: str = "밝기 단계") -> np.ndarray:
    h, w = rgb.shape[:2]
    bar_h = 28
    canvas = np.zeros((h + bar_h, w, 3), dtype=np.uint8)
    canvas[:h] = rgb
    canvas[h:] = 30
    n = len(STAGE_ORDER)
    for i, (key, label, col) in enumerate(STAGE_ORDER):
        x0 = int(i * w / n)
        x1 = int((i + 1) * w / n)
        canvas[h + 2 : h + bar_h - 2, x0 + 1 : x1 - 1] = col
    # text via PIL
    im = Image.fromarray(canvas)
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/sand-box/google/Nanum Gothic/NanumGothic-Regular.ttf", 10)
    except Exception:
        font = ImageFont.load_default()
    for i, (key, label, col) in enumerate(STAGE_ORDER):
        x0 = int(i * w / n) + 3
        draw.text((x0, h + 6), label, fill=(0, 0, 0) if i < 4 else (255, 255, 255), font=font)
    return np.array(im)


def watershed_seg(g: np.ndarray):
    # invert so grains (brighter) become basins after smoothing
    sm = gaussian(g, sigma=1.2, preserve_range=True)
    # distance on bright regions (grains)
    thr = np.percentile(sm, 45)
    mask = sm >= thr
    # edges as barriers
    edges = sobel(sm / 255.0)
    # markers: local maxima of distance transform inside mask
    dist = ndi.distance_transform_edt(mask)
    # suppress tiny peaks
    coords = peak_local_max(dist, min_distance=4, labels=mask.astype(np.int32), exclude_border=False)
    markers = np.zeros_like(g, dtype=np.int32)
    for i, (r, c) in enumerate(coords, start=1):
        markers[r, c] = i
    if markers.max() == 0:
        # fallback: local minima of smoothed image (grain centers often slightly darker? try maxima of -edges)
        coords = peak_local_max(-edges, min_distance=5, exclude_border=False)
        for i, (r, c) in enumerate(coords, start=1):
            markers[r, c] = i
    labels_ws = watershed(edges, markers, mask=mask)
    # also assign dark leftover? leave as 0 (boundary/pore)
    n_grains = int(labels_ws.max())
    props = regionprops(labels_ws)
    areas = [p.area for p in props if p.area >= 8]
    mean_area = float(np.mean(areas)) if areas else 0.0
    # colorize
    rng = np.random.default_rng(42)
    colors = rng.integers(40, 255, size=(max(n_grains + 1, 2), 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]
    rgb = colors[labels_ws]
    # black boundaries
    from skimage.segmentation import find_boundaries

    bnd = find_boundaries(labels_ws, mode="outer")
    rgb[bnd] = [0, 0, 0]
    return rgb, n_grains, mean_area, labels_ws


def boundary_strength_map(g: np.ndarray):
    sm = gaussian(g, sigma=1.0, preserve_range=True)
    # gradient magnitude
    gx = ndi.sobel(sm, axis=1)
    gy = ndi.sobel(sm, axis=0)
    grad = np.hypot(gx, gy)
    p99 = float(np.percentile(grad, 99))
    # darkness along edges: low intensity where grad high
    # boundary "darkness" = how dark the pixel is when it's on an edge
    edge_mask = grad >= np.percentile(grad, 85)
    # strength = grad * (1 - normalized brightness) to emphasize dark thick boundaries
    inv = 1.0 - (sm - sm.min()) / max(sm.max() - sm.min(), 1e-6)
    strength = grad * (0.4 + 0.6 * inv)
    # thickness proxy: blur of edge mask
    thick = gaussian(edge_mask.astype(np.float32), sigma=1.5)
    heat = strength * (0.5 + thick)
    # normalize for display
    lo, hi = np.percentile(heat, [5, 99.5])
    x = (heat - lo) / max(hi - lo, 1e-6)
    x = np.clip(x, 0, 1)
    cmap = plt.get_cmap("inferno")
    rgb = (cmap(x)[..., :3] * 255).astype(np.uint8)
    return rgb, p99, float(grad[edge_mask].mean()) if edge_mask.any() else 0.0


def direction_maps(g: np.ndarray, labels_ws: np.ndarray):
    sm = gaussian(g / 255.0, sigma=1.5, preserve_range=True)
    Axx, Axy, Ayy = structure_tensor(sm, sigma=2.0, mode="reflect")
    # orientation: 0.5 * atan2(2Axy, Axx-Ayy)
    orient = 0.5 * np.arctan2(2.0 * Axy, Axx - Ayy)  # [-pi/2, pi/2]
    # eigenvalues for coherence
    l1, l2 = structure_tensor_eigenvalues([Axx, Axy, Ayy])  # l1 >= l2
    coherence = (l1 - l2) / (l1 + l2 + 1e-8)
    coherence = np.clip(coherence, 0, 1)
    # HSV: hue from orientation, sat from coherence, val constant-ish
    hue = (orient + np.pi / 2) / np.pi  # 0..1
    hsv = np.stack([hue, coherence, np.clip(0.35 + 0.65 * (1 - sm), 0.2, 1.0)], axis=-1)
    rgb = matplotlib.colors.hsv_to_rgb(hsv)
    rgb_u8 = (rgb * 255).astype(np.uint8)

    # EBSD-style: per-grain mean orientation
    ebsd = np.zeros_like(rgb_u8)
    for lab in range(1, int(labels_ws.max()) + 1):
        m = labels_ws == lab
        if m.sum() < 8:
            continue
        # circular mean of orientation (double angle)
        ang = orient[m] * 2
        mean_ang = 0.5 * np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())
        coh = float(coherence[m].mean())
        h = (mean_ang + np.pi / 2) / np.pi
        color = matplotlib.colors.hsv_to_rgb([h, min(1.0, 0.3 + 0.7 * coh), 0.85])
        ebsd[m] = (np.array(color) * 255).astype(np.uint8)
    # black boundaries
    from skimage.segmentation import find_boundaries

    bnd = find_boundaries(labels_ws, mode="outer")
    ebsd[bnd] = [0, 0, 0]
    return rgb_u8, ebsd


def correlation_band_map(g: np.ndarray, band_lo: float, band_hi: float) -> np.ndarray:
    """Highlight pixels in the brightness band most correlated with hardness in red."""
    base = np.stack([g, g, g], axis=-1).astype(np.uint8)
    # slight dim
    base = (base.astype(np.float32) * 0.55).astype(np.uint8)
    mask = (g >= band_lo) & (g <= band_hi)
    out = base.copy()
    out[mask, 0] = np.clip(out[mask, 0].astype(np.int16) + 160, 0, 255).astype(np.uint8)
    out[mask, 1] = (out[mask, 1] * 0.3).astype(np.uint8)
    out[mask, 2] = (out[mask, 2] * 0.3).astype(np.uint8)
    return out


def to_rgb(gray_u8: np.ndarray) -> np.ndarray:
    if gray_u8.ndim == 2:
        return np.stack([gray_u8] * 3, axis=-1)
    return gray_u8


def put_label(rgb: np.ndarray, text: str, top: bool = True) -> np.ndarray:
    im = Image.fromarray(rgb)
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/sand-box/google/Nanum Gothic/NanumGothic-Regular.ttf", 12)
    except Exception:
        font = ImageFont.load_default()
    y = 2 if top else rgb.shape[0] - 16
    # shadow
    draw.text((3, y + 1), text, fill=(0, 0, 0), font=font)
    draw.text((2, y), text, fill=(255, 255, 255), font=font)
    return np.array(im)


def hstack_pad(imgs, gap=4, bg=20):
    h = max(im.shape[0] for im in imgs)
    parts = []
    for im in imgs:
        if im.shape[0] < h:
            pad = np.full((h - im.shape[0], im.shape[1], 3), bg, dtype=np.uint8)
            im = np.vstack([im, pad])
        parts.append(im)
        parts.append(np.full((h, gap, 3), bg, dtype=np.uint8))
    parts = parts[:-1]
    return np.hstack(parts)


def make_row(sample, gstats, band_lo, band_hi):
    tag, iid, hard = sample
    g = load_gray(iid)
    orig = to_rgb(np.clip(g, 0, 255).astype(np.uint8))
    clahe = to_rgb(clahe_img(g))
    common = to_rgb(common_scale(g, gstats["p1"], gstats["p99"], gamma=0.85))
    narrow = to_rgb(narrow_scale(g, lo=120.0, hi=170.0))
    rel = relative_stage_map(g)
    rel_l = add_legend_bar(rel)
    abs_m = absolute_stage_map(g, gstats["g_mean"], gstats["g_std"])
    abs_l = add_legend_bar(abs_m)
    corr = correlation_band_map(g, band_lo, band_hi)
    seg_rgb, n_grains, mean_area, labels_ws = watershed_seg(g)
    seg_rgb = put_label(seg_rgb, f"n={n_grains} meanA={mean_area:.0f}px")
    bnd_rgb, p99_grad, _ = boundary_strength_map(g)
    bnd_rgb = put_label(bnd_rgb, f"top1% grad={p99_grad:.1f}")
    dir_rgb, ebsd_rgb = direction_maps(g, labels_ws)

    # labels on each panel
    panels = [
        put_label(orig, "원본"),
        put_label(clahe, "CLAHE"),
        put_label(common, "공통스케일 1-99%"),
        put_label(narrow, "좁은스케일 120-170"),
        put_label(rel_l, "상대단계"),
        put_label(abs_l, "절대단계"),
        put_label(corr, f"상관대역 {band_lo:.0f}-{band_hi:.0f}"),
        put_label(seg_rgb, "분할"),
        put_label(bnd_rgb, "경계강도"),
        put_label(dir_rgb, "방향(픽셀)"),
        put_label(ebsd_rgb, "방향(입자)"),
    ]
    # resize each to 256 width for row (keep aspect; already 256)
    # title strip
    row = hstack_pad(panels, gap=3)
    title_h = 26
    canvas = np.full((row.shape[0] + title_h, row.shape[1], 3), 18, dtype=np.uint8)
    canvas[title_h:] = row
    im = Image.fromarray(canvas)
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/sand-box/google/Nanum Gothic/NanumGothic-Regular.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    title = f"[{tag}] {iid}  hardness={hard:.3f}"
    draw.text((6, 4), title, fill=(255, 230, 120), font=font)
    out_path = ROWS_DIR / f"{tag}_{iid}_h{hard:.1f}.png"
    im.save(out_path)
    # also return compact panels for summary
    compact = {
        "tag": tag,
        "iid": iid,
        "hard": hard,
        "orig": orig,
        "common": common,
        "narrow": narrow,
        "stage": rel,  # without legend for compact
        "seg": seg_rgb,
        "bnd": bnd_rgb,
        "dir": dir_rgb,
        "n_grains": n_grains,
        "mean_area": mean_area,
        "p99_grad": p99_grad,
        "mean": float(g.mean()),
        "std": float(g.std()),
    }
    # features for scatter
    sm = gaussian(g, sigma=1.0, preserve_range=True)
    gx = ndi.sobel(sm, axis=1)
    gy = ndi.sobel(sm, axis=0)
    grad = np.hypot(gx, gy)
    edge = grad >= np.percentile(grad, 90)
    edge_dark = float((255 - sm[edge]).mean()) if edge.any() else 0.0
    compact["edge_dark"] = edge_dark
    return str(out_path), compact


def find_correlation_band(rows):
    """Find gray-level bins whose occupancy correlates most with hardness."""
    # Use 32 bins over 0-255
    bins = np.linspace(0, 256, 33)
    centers = 0.5 * (bins[:-1] + bins[1:])
    H = []
    hard = []
    for iid, h in rows:
        g = load_gray(iid)
        hist, _ = np.histogram(g, bins=bins, density=True)
        H.append(hist)
        hard.append(h)
    H = np.array(H)
    hard = np.array(hard)
    hard_z = (hard - hard.mean()) / (hard.std() + 1e-6)
    corrs = []
    for j in range(H.shape[1]):
        x = H[:, j]
        if x.std() < 1e-12:
            corrs.append(0.0)
        else:
            corrs.append(float(np.corrcoef(x, hard)[0, 1]))
    corrs = np.array(corrs)
    # pick contiguous band around strongest |corr| bins (top 3)
    abs_c = np.abs(corrs)
    # Prefer the strongest positive-correlation peak (darker midtones for hardness)
    best = int(np.argmax(corrs))  # strongest positive
    if corrs[best] < 0.05:
        best = int(np.argmax(abs_c))
    lo_i = best
    hi_i = best
    # grow while neighbor keeps >= 60% of peak |corr| and same sign
    thr = 0.60 * abs(corrs[best])
    sign = 1 if corrs[best] >= 0 else -1
    while lo_i > 0 and abs(corrs[lo_i - 1]) >= thr and (corrs[lo_i - 1] * sign) > 0:
        lo_i -= 1
    while hi_i < len(centers) - 1 and abs(corrs[hi_i + 1]) >= thr and (corrs[hi_i + 1] * sign) > 0:
        hi_i += 1
    # ensure at least ~16 gray levels wide
    while (hi_i - lo_i + 1) < 2 and hi_i < len(centers) - 1:
        hi_i += 1
    band_lo = float(bins[lo_i])
    band_hi = float(bins[hi_i + 1])
    return band_lo, band_hi, centers, corrs, bins


def standardized_hist(g, g_mean, g_std, bins):
    z = (g - g_mean) / g_std
    # map z to display bins over [-3,3]
    hist, _ = np.histogram(z, bins=bins, density=True)
    return hist


def make_amplify_panels(rows, gstats, band_lo, band_hi, centers, corrs):
    # group
    rows_s = sorted(rows, key=lambda x: x[1])
    low18 = rows_s[:18]
    high18 = rows_s[-18:]
    # mid: hardness within 192-196 or closest 40
    mid = [r for r in rows_s if 190 <= r[1] <= 198]
    if len(mid) < 20:
        mid = sorted(rows_s, key=lambda x: abs(x[1] - 194))[:40]

    zbins = np.linspace(-3, 3, 49)
    zc = 0.5 * (zbins[:-1] + zbins[1:])

    def mean_hist(group):
        acc = []
        for iid, h in group:
            g = load_gray(iid)
            acc.append(standardized_hist(g, gstats["g_mean"], gstats["g_std"], zbins))
        return np.mean(acc, axis=0)

    h_high = mean_hist(high18)
    h_low = mean_hist(low18)
    h_mid = mean_hist(mid)

    # (a) overlaid histograms + difference
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), dpi=140)
    ax = axes[0]
    ax.plot(zc, h_low, color="#3b82f6", lw=2, label=f"저경도 평균 (n={len(low18)})")
    ax.plot(zc, h_mid, color="#9ca3af", lw=2, label=f"중간 평균 (n={len(mid)})")
    ax.plot(zc, h_high, color="#ef4444", lw=2, label=f"고경도 평균 (n={len(high18)})")
    ax.set_xlabel("표준화 밝기 z = (pixel − μ_global) / σ_global")
    ax.set_ylabel("밀도")
    ax.set_title("전체 train 기준 표준화 밝기 히스토그램")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax = axes[1]
    ax.plot(zc, h_high - h_low, color="#f59e0b", lw=2, label="고 − 저")
    ax.plot(zc, h_high - h_mid, color="#ef4444", lw=1.5, ls="--", label="고 − 중")
    ax.plot(zc, h_low - h_mid, color="#3b82f6", lw=1.5, ls="--", label="저 − 중")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("표준화 밝기 z")
    ax.set_ylabel("밀도 차이")
    ax.set_title("분포 차이 곡선")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    hist_path = AMP_DIR / "hist_high_low_mid.png"
    fig.savefig(hist_path)
    plt.close(fig)

    # correlation per bin bar
    fig, ax = plt.subplots(figsize=(8, 3.5), dpi=140)
    colors = ["#ef4444" if c > 0 else "#3b82f6" for c in corrs]
    ax.bar(centers, corrs, width=(centers[1] - centers[0]) * 0.9, color=colors, alpha=0.85)
    ax.axvspan(band_lo, band_hi, color="red", alpha=0.15, label=f"선택 대역 {band_lo:.0f}-{band_hi:.0f}")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("밝기 구간 (0–255)")
    ax.set_ylabel("경도와의 상관")
    ax.set_title("밝기 구간별 경도 상관 (선택 대역 강조)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    corr_path = AMP_DIR / "corr_by_gray_bin.png"
    fig.savefig(corr_path)
    plt.close(fig)

    # (d) scatter features vs hardness for all train
    feats = []
    for iid, h in rows:
        g = load_gray(iid)
        sm = gaussian(g, sigma=1.0, preserve_range=True)
        gx = ndi.sobel(sm, axis=1)
        gy = ndi.sobel(sm, axis=0)
        grad = np.hypot(gx, gy)
        edge = grad >= np.percentile(grad, 90)
        edge_dark = float((255 - sm[edge]).mean()) if edge.any() else 0.0
        feats.append((h, float(g.mean()), float(g.std()), edge_dark))
    feats = np.array(feats)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), dpi=140)
    titles = ["평균 밝기", "밝기 표준편차", "경계 어두움 (상위10% 경계)"]
    for i, ax in enumerate(axes):
        ax.scatter(feats[:, i + 1], feats[:, 0], s=10, alpha=0.45, c="#64748b", edgecolors="none")
        # highlight high/low 18
        hi_idx = np.argsort(feats[:, 0])[-18:]
        lo_idx = np.argsort(feats[:, 0])[:18]
        ax.scatter(feats[hi_idx, i + 1], feats[hi_idx, 0], s=18, c="#ef4444", label="고18", zorder=3)
        ax.scatter(feats[lo_idx, i + 1], feats[lo_idx, 0], s=18, c="#3b82f6", label="저18", zorder=3)
        r = np.corrcoef(feats[:, i + 1], feats[:, 0])[0, 1]
        ax.set_xlabel(titles[i])
        ax.set_ylabel("hardness")
        ax.set_title(f"{titles[i]}  r={r:.3f}")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)
    fig.tight_layout()
    scat_path = AMP_DIR / "scatter_features_vs_hardness.png"
    fig.savefig(scat_path)
    plt.close(fig)

    return str(hist_path), str(corr_path), str(scat_path), feats


def make_summary(compacts, hist_path, corr_path, scat_path):
    # pick 2 high, 2 mid, 2 low
    highs = [c for c in compacts if c["tag"] == "high"][:2]
    mids = [c for c in compacts if c["tag"] == "mid"][:2]
    lows = [c for c in compacts if c["tag"] == "low"][:2]
    picks = highs + mids + lows

    cell = 180  # each panel ~180px -> 6*180=1080 + gaps < 2400
    gap = 4
    cols = ["orig", "common", "stage", "seg", "bnd", "dir"]
    col_titles = ["원본", "공통스케일", "단계", "분할", "경계", "방향"]

    def resize(im, s=cell):
        return np.array(Image.fromarray(im).resize((s, s), Image.NEAREST))

    row_h = cell + 22
    title_h = 28
    n_cols = len(cols)
    width = n_cols * cell + (n_cols - 1) * gap + 16
    # ensure <= 2400
    assert width <= 2400, width
    height = title_h + len(picks) * (row_h + 8) + 8
    canvas = np.full((height, width, 3), 16, dtype=np.uint8)
    # header
    im = Image.fromarray(canvas)
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/sand-box/google/Nanum Gothic/NanumGothic-Regular.ttf", 14)
        font_s = ImageFont.truetype("/usr/share/fonts/truetype/sand-box/google/Nanum Gothic/NanumGothic-Regular.ttf", 11)
    except Exception:
        font = ImageFont.load_default()
        font_s = font
    draw.text((8, 6), "DAKER 요약: 고2 / 중2 / 저2  |  원본·공통스케일·단계·분할·경계·방향", fill=(255, 220, 120), font=font)
    canvas = np.array(im)

    y = title_h
    for c in picks:
        # row label strip
        strip = np.full((row_h, width, 3), 22, dtype=np.uint8)
        x = 8
        panels = [resize(c[k]) for k in cols]
        # stage without legend already
        row_img = hstack_pad(panels, gap=gap)
        # place
        rh, rw = row_img.shape[:2]
        strip[20 : 20 + rh, 8 : 8 + rw] = row_img
        # titles
        sim = Image.fromarray(strip)
        d = ImageDraw.Draw(sim)
        d.text((8, 2), f"[{c['tag']}] {c['iid']}  hardness={c['hard']:.3f}  n={c['n_grains']} meanA={c['mean_area']:.0f} mean={c['mean']:.1f} std={c['std']:.1f}", fill=(230, 230, 230), font=font_s)
        for i, t in enumerate(col_titles):
            d.text((8 + i * (cell + gap) + 4, 20 + cell - 14), t, fill=(255, 255, 255), font=font_s)
        strip = np.array(sim)
        canvas[y : y + row_h, :] = strip
        y += row_h + 8

    summary_path = OUT / "summary_6rows.png"
    Image.fromarray(canvas).save(summary_path)

    # Amplification board: hist + corr + scatter + narrow examples
    # Build a combined amplify summary
    hist_im = np.array(Image.open(hist_path).convert("RGB"))
    corr_im = np.array(Image.open(corr_path).convert("RGB"))
    scat_im = np.array(Image.open(scat_path).convert("RGB"))

    # narrow scale examples: 2 high + 2 low side by side
    narrow_panels = []
    for c in highs[:2] + lows[:2]:
        g = load_gray(c["iid"])
        n = to_rgb(narrow_scale(g, 120, 170))
        n = put_label(n, f"{c['tag']} {c['iid']} h={c['hard']:.1f} mean={c['mean']:.1f}")
        narrow_panels.append(np.array(Image.fromarray(n).resize((220, 220), Image.NEAREST)))
    narrow_row = hstack_pad(narrow_panels, gap=6)

    # also correlation band examples
    band_panels = []
    for c in highs[:2] + lows[:2]:
        g = load_gray(c["iid"])
        b = correlation_band_map(g, band_lo := 0, band_hi := 0)  # placeholder overwritten below
    # fix: use actual band from outer scope — passed via closure; recompute from file name? pass args
    return summary_path, hist_im, corr_im, scat_im, narrow_row


def make_amplify_summary(compacts, hist_path, corr_path, scat_path, band_lo, band_hi):
    highs = [c for c in compacts if c["tag"] == "high"][:2]
    lows = [c for c in compacts if c["tag"] == "low"][:2]
    hist_im = np.array(Image.open(hist_path).convert("RGB"))
    corr_im = np.array(Image.open(corr_path).convert("RGB"))
    scat_im = np.array(Image.open(scat_path).convert("RGB"))

    def prep(c, kind):
        g = load_gray(c["iid"])
        if kind == "narrow":
            im = to_rgb(narrow_scale(g, 120, 170))
            lab = f"좁은120-170 {c['tag']} {c['iid']} h={c['hard']:.1f} μ={c['mean']:.1f}"
        else:
            im = correlation_band_map(g, band_lo, band_hi)
            lab = f"상관대역 {c['tag']} {c['iid']} h={c['hard']:.1f}"
        return put_label(np.array(Image.fromarray(im).resize((200, 200), Image.NEAREST)), lab)

    narrow_row = hstack_pad([prep(c, "narrow") for c in highs + lows], gap=6)
    band_row = hstack_pad([prep(c, "band") for c in highs + lows], gap=6)

    # resize large plots to similar width
    target_w = max(narrow_row.shape[1], 1100)

    def fit_w(im, w):
        h = int(im.shape[0] * w / im.shape[1])
        return np.array(Image.fromarray(im).resize((w, h), Image.BILINEAR))

    hist_im = fit_w(hist_im, target_w)
    corr_im = fit_w(corr_im, target_w)
    scat_im = fit_w(scat_im, target_w)
    # pad narrow/band to target_w
    def pad_w(im, w):
        if im.shape[1] >= w:
            return im
        pad = np.full((im.shape[0], w - im.shape[1], 3), 16, dtype=np.uint8)
        return np.hstack([im, pad])

    narrow_row = pad_w(narrow_row, target_w)
    band_row = pad_w(band_row, target_w)

    gap = np.full((12, target_w, 3), 16, dtype=np.uint8)
    # titles
    def titled(im, text):
        bar = np.full((24, im.shape[1], 3), 20, dtype=np.uint8)
        p = Image.fromarray(bar)
        d = ImageDraw.Draw(p)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/sand-box/google/Nanum Gothic/NanumGothic-Regular.ttf", 13)
        except Exception:
            font = ImageFont.load_default()
        d.text((6, 4), text, fill=(255, 220, 120), font=font)
        return np.vstack([np.array(p), im])

    parts = [
        titled(hist_im, "(a) 표준화 밝기 히스토그램 + 차이 곡선"),
        gap,
        titled(corr_im, "(b) 밝기 구간–경도 상관 / 선택 대역"),
        gap,
        titled(band_row, "(b) 상관 대역 픽셀 강조 (고2·저2)"),
        gap,
        titled(narrow_row, "(c) 좁은 공통 스케일 120–170 (미세 밝기차 증폭, 고2·저2)"),
        gap,
        titled(scat_im, "(d) 평균밝기 · 표준편차 · 경계어두움 vs 경도"),
    ]
    board = np.vstack(parts)
    path = OUT / "summary_amplify.png"
    Image.fromarray(board).save(path)
    return path


def write_html(samples, row_paths, summary_path, amp_summary, hist_path, corr_path, scat_path, band_lo, band_hi, gstats):
    cards = []
    for (tag, iid, hard), rp in zip(samples, row_paths):
        rel = Path(rp).relative_to(OUT)
        cards.append(
            f'<div class="card"><div class="t">[{tag}] {iid} · hardness={hard:.3f}</div>'
            f'<a href="{rel}"><img src="{rel}" loading="lazy"></a></div>'
        )
    html = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<title>DAKER 미세구조 변환 갤러리</title>
<style>
body{{font-family:system-ui,sans-serif;background:#111;color:#eee;margin:16px}}
h1,h2{{color:#fbbf24}}
img{{max-width:100%;border-radius:6px;background:#000}}
.card{{margin:18px 0;padding:10px;background:#1a1a1a;border-radius:8px}}
.t{{font-weight:700;margin-bottom:6px}}
.meta{{color:#aaa;font-size:14px;line-height:1.5}}
a{{color:#93c5fd}}
</style></head><body>
<h1>DAKER 미세구조 변환 갤러리</h1>
<p class="meta">샘플: 고경도 6 · 중간(~194) 4 · 저경도 6<br>
공통스케일: train 픽셀 1–99% → 0–255 + γ=0.85<br>
좁은스케일: 120–170 강제 스트레치 (미세 밝기차 증폭)<br>
상관 대역: {band_lo:.1f}–{band_hi:.1f}<br>
전역 μ={gstats['g_mean']:.2f} σ={gstats['g_std']:.2f} · p1={gstats['p1']:.1f} p99={gstats['p99']:.1f}
</p>
<h2>요약</h2>
<div class="card"><img src="{Path(summary_path).name}"></div>
<div class="card"><img src="{Path(amp_summary).name}"></div>
<h2>증폭 패널</h2>
<div class="card"><img src="{Path(hist_path).relative_to(OUT)}"></div>
<div class="card"><img src="{Path(corr_path).relative_to(OUT)}"></div>
<div class="card"><img src="{Path(scat_path).relative_to(OUT)}"></div>
<h2>줄별 비교</h2>
{''.join(cards)}
</body></html>"""
    (OUT / "index.html").write_text(html, encoding="utf-8")


def main():
    ROWS_DIR.mkdir(parents=True, exist_ok=True)
    PANELS_DIR.mkdir(parents=True, exist_ok=True)
    AMP_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_meta()
    samples = pick_samples(rows)
    print("samples:", [(t, i, round(h, 2)) for t, i, h in samples], flush=True)

    print("computing global stats...", flush=True)
    gstats = compute_global_stats(rows)
    print("gstats", gstats, flush=True)

    print("finding correlation band...", flush=True)
    band_lo, band_hi, centers, corrs, bins = find_correlation_band(rows)
    print(f"correlation band: {band_lo:.1f}-{band_hi:.1f}", flush=True)
    print("top corr bins:", sorted(zip(centers, corrs), key=lambda x: -abs(x[1]))[:5], flush=True)

    print("amplify panels...", flush=True)
    hist_path, corr_path, scat_path, feats = make_amplify_panels(rows, gstats, band_lo, band_hi, centers, corrs)

    print("building rows with 2 threads...", flush=True)
    row_paths = [None] * len(samples)
    compacts = [None] * len(samples)
    with ThreadPoolExecutor(max_workers=N_THREADS) as ex:
        futs = {ex.submit(make_row, s, gstats, band_lo, band_hi): i for i, s in enumerate(samples)}
        for fut in as_completed(futs):
            i = futs[fut]
            path, compact = fut.result()
            row_paths[i] = path
            compacts[i] = compact
            print("done", compact["tag"], compact["iid"], flush=True)

    print("summary...", flush=True)
    # rebuild summary without the broken first draft
    highs = [c for c in compacts if c["tag"] == "high"][:2]
    mids = [c for c in compacts if c["tag"] == "mid"][:2]
    lows = [c for c in compacts if c["tag"] == "low"][:2]
    picks = highs + mids + lows
    cell = 180
    gap = 4
    cols = ["orig", "common", "stage", "seg", "bnd", "dir"]
    col_titles = ["원본", "공통스케일", "단계", "분할", "경계", "방향"]

    def resize(im, s=cell):
        return np.array(Image.fromarray(im).resize((s, s), Image.NEAREST))

    row_h = cell + 22
    title_h = 28
    n_cols = len(cols)
    width = n_cols * cell + (n_cols - 1) * gap + 16
    height = title_h + len(picks) * (row_h + 8) + 8
    canvas = np.full((height, width, 3), 16, dtype=np.uint8)
    im = Image.fromarray(canvas)
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/sand-box/google/Nanum Gothic/NanumGothic-Regular.ttf", 14)
        font_s = ImageFont.truetype("/usr/share/fonts/truetype/sand-box/google/Nanum Gothic/NanumGothic-Regular.ttf", 11)
    except Exception:
        font = ImageFont.load_default()
        font_s = font
    draw.text((8, 6), "DAKER 요약: 고2 / 중2 / 저2  |  원본·공통스케일·단계·분할·경계·방향", fill=(255, 220, 120), font=font)
    canvas = np.array(im)
    y = title_h
    for c in picks:
        strip = np.full((row_h, width, 3), 22, dtype=np.uint8)
        panels = [resize(c[k]) for k in cols]
        row_img = hstack_pad(panels, gap=gap)
        rh, rw = row_img.shape[:2]
        strip[20 : 20 + rh, 8 : 8 + rw] = row_img
        sim = Image.fromarray(strip)
        d = ImageDraw.Draw(sim)
        d.text(
            (8, 2),
            f"[{c['tag']}] {c['iid']}  hardness={c['hard']:.3f}  n={c['n_grains']} meanA={c['mean_area']:.0f} mean={c['mean']:.1f} std={c['std']:.1f}",
            fill=(230, 230, 230),
            font=font_s,
        )
        for i, t in enumerate(col_titles):
            d.text((8 + i * (cell + gap) + 4, 20 + cell - 14), t, fill=(255, 255, 255), font=font_s)
        canvas[y : y + row_h, :] = np.array(sim)
        y += row_h + 8
    summary_path = OUT / "summary_6rows.png"
    Image.fromarray(canvas).save(summary_path)
    print("summary width", width, flush=True)

    amp_summary = make_amplify_summary(compacts, hist_path, corr_path, scat_path, band_lo, band_hi)
    write_html(samples, row_paths, summary_path, amp_summary, hist_path, corr_path, scat_path, band_lo, band_hi, gstats)

    meta = {
        "samples": [{"tag": t, "id": i, "hardness": h} for t, i, h in samples],
        "gstats": gstats,
        "band": [band_lo, band_hi],
        "summary": str(summary_path),
        "amplify_summary": str(amp_summary),
        "rows": row_paths,
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("DONE", flush=True)
    print("summary", summary_path)
    print("amplify", amp_summary)
    print("index", OUT / "index.html")


if __name__ == "__main__":
    main()
