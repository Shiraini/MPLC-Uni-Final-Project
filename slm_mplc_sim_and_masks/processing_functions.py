# ===== processing_functions.py =====
import os, glob, itertools
import cv2
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


# ---------- File helpers ----------
def first_existing(root, stem):
    """Return first image path in root whose filename starts with stem."""
    for ext in ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.bmp'):
        cand = sorted(glob.glob(os.path.join(root, f"{stem}*{ext}")))
        if cand:
            return cand[0]
    raise FileNotFoundError(f"No image for {stem} under {root}")

# ---------- Image loaders (no per-image normalization) ----------
def load_gray_linear(path, dark=None, exposure_scale=1.0):
    """
    Load grayscale image as float (linear camera counts).
    Optionally subtract a dark frame (same resolution/exposure),
    and apply a single global exposure_scale factor if needed.
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    img = img.astype(np.float64)
    if dark is not None:
        img = np.clip(img - dark, a_min=0, a_max=None)
    return img * float(exposure_scale)

# ---------- Math utilities ----------
def gaussian_2d(coords, x0, y0, sx, sy, A, c):
    x, y = coords
    g = A * np.exp(-(((x - x0) ** 2) / (2 * sx ** 2) + ((y - y0) ** 2) / (2 * sy ** 2))) + c
    return g

def find_gaussian_spot(image):
    """
    Find (cx, cy, r) for a bright Gaussian-like spot in a 2D array.
    Works on a *linear* image. Internally normalizes a cropped patch for the fit.
    Returns integer pixel center and a conservative radius (px).
    """
    img = image
    # Start from brightest pixel
    y0, x0 = np.unravel_index(np.argmax(img), img.shape)
    # Crop around peak for robust fit
    R = 25
    y_min, y_max = max(0, y0 - R), min(img.shape[0], y0 + R + 1)
    x_min, x_max = max(0, x0 - R), min(img.shape[1], x0 + R + 1)
    crop = img[y_min:y_max, x_min:x_max].copy()
    if crop.size == 0 or crop.max() <= 0:
        # Fallback: just return brightest pixel with small radius
        return int(x0), int(y0), 6

    # Normalize crop for numeric stability
    crop = crop - crop.min()
    m = crop.max()
    if m > 0:
        crop = crop / m

    yy, xx = np.mgrid[0:crop.shape[0], 0:crop.shape[1]]
    p0 = (crop.shape[1] / 2, crop.shape[0] / 2, 5.0, 5.0, 1.0, 0.0)
    bounds = ((0, 0, 1, 1, 0, -0.5),
              (crop.shape[1], crop.shape[0], 30, 30, 2.0, 0.5))
    try:
        popt, _ = curve_fit(gaussian_2d,
                            (xx.ravel(), yy.ravel()),
                            crop.ravel(), p0=p0, bounds=bounds, maxfev=2000)
        x_fit, y_fit, sx, sy, *_ = popt
        cx = x_min + x_fit
        cy = y_min + y_fit
        # radius: ~ 1.2 * 3 * average sigma  (covers >99% power with margin)
        r = 1.2 * 3.0 * float(sx + sy) / 2.0
        return int(round(cx)), int(round(cy)), int(max(5, min(80, round(r))))
    except Exception:
        # Fallback if fit fails
        return int(x0), int(y0), 10

def integrate_circle(img, cx, cy, r):
    """Sum image values inside a circle of radius r centered at (cx, cy)."""
    H, W = img.shape
    y, x = np.ogrid[:H, :W]
    mask = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
    return float(img[mask].sum())

def draw_ports_bgr(img_gray, centers, r, thickness=2):
    """
    Return a BGR visualization of img_gray with circles at each center.
    centers: list of (cx, cy); r: int
    """
    img_u8 = np.clip(img_gray, 0, None)
    if img_u8.max() > 0:
        img_u8 = (img_u8 / img_u8.max() * 255.0).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]  # R, G, B, Y
    for k, (cx, cy) in enumerate(centers):
        color = colors[k % len(colors)]
        cv2.circle(img_bgr, (int(cx), int(cy)), int(r), color, thickness, lineType=cv2.LINE_AA)
        cv2.circle(img_bgr, (int(cx), int(cy)), max(1, thickness), color, -1, lineType=cv2.LINE_AA)
    return img_bgr


# Load grayscale image, subtract background, normalize to [0,1]
def load_gray_norm(path, bg_percentile=5):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    img = img.astype(float)
    bg = np.percentile(img, bg_percentile)
    img = np.clip(img - bg, a_min=0, a_max=None)
    mx = img.max()
    if mx > 0:
        img /= mx
    return img


def compute_crosstalk_matrix(out_paths, MODES, centre_list, radius_px, SAVE_DIR):
    powers = []
    r_px = radius_px * 1 / 3.6
    # r_px = radius_px
    for m in MODES:
        img = load_gray_norm(out_paths[m])
        row = [integrate_circle(img, cx, cy, r_px) for (cx, cy) in centre_list]
        powers.append(row)
    P = np.array(powers, dtype=float)  # Raw power matrix: input modes × detected ports

    row_sums = P.sum(axis=1, keepdims=True) + 1e-12
    T = P / row_sums  # Normalized transfer matrix (rows sum to 1)

    # Best column permutation to maximize diagonal sum
    best_perm, best_score = None, -1
    for perm in itertools.permutations(range(len(MODES))):
        score = T[range(len(MODES)), perm].sum()
        if score > best_score:
            best_score, best_perm = score, perm

    T = T[:, best_perm]
    P = P[:, best_perm]
    col_labels = [MODES[j] for j in best_perm]

    XT_dB = 10 * np.log10(np.clip(T, 1e-6, 1.0))  # Crosstalk matrix in dB
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(XT_dB, vmin=-40, vmax=0, cmap='viridis')
    ax.set_xticks(range(4))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(4))
    ax.set_yticklabels(MODES)
    ax.set_xlabel('Detected output')
    ax.set_ylabel('Injected input')
    fig.colorbar(im, label='Power [dB]')
    plt.title(f'Crosstalk matrix (dB)')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'crosstalk_matrix.png'), dpi=200)
    plt.show()
