# ===== processing_results.py =====
# Computes ports, absolute power matrix P, crosstalk T, IL, MDL,
# and saves annotated output images with four port circles.

import os, itertools
import numpy as np
import matplotlib.pyplot as plt
import cv2

from processing_functions import (
    first_existing, load_gray_linear, find_gaussian_spot,
    integrate_circle, draw_ports_bgr, compute_crosstalk_matrix
)

# ---------------- User paths & settings ----------------
INPUT_DIR  = 'lab_results/28.8/2planes4modes/inputs'        # camera frames of injected modes (pre-sorter), same exposure
OUTPUT_DIR = 'lab_results/28.8/2planes4modes/non-filtered'  # camera frames at the output plane, same exposure
SAVE_DIR   = './annotated_outputs'                          # where annotated images will be saved
os.makedirs(SAVE_DIR, exist_ok=True)

MODES = ['HG10', 'HG01', 'HG11', 'HG22']       # ordering of injected inputs and desired ports
DARK_PATH = None                               # e.g. '../lab_results/dark.tif' if you have it
EXPOSURE_SCALE = 1.0                           # keep 1.0 if all images share the same exposure/gain

# ------------------------------------------------------
# Load optional dark frame
DARK = None
if DARK_PATH and os.path.isfile(DARK_PATH):
    d = cv2.imread(DARK_PATH, cv2.IMREAD_GRAYSCALE)
    if d is not None:
        DARK = d.astype(np.float64)

# Map mode -> image path
out_paths = {m: first_existing(OUTPUT_DIR, m) for m in MODES}
in_paths  = {m: first_existing(INPUT_DIR,  m) for m in MODES}

# ---------- 1) Detect ports (centers) and a common radius ----------
centers = []
r_list  = []
for m in MODES:
    img = load_gray_linear(out_paths[m], dark=DARK, exposure_scale=EXPOSURE_SCALE)
    cx, cy, r = find_gaussian_spot(img)
    centers.append((cx, cy))
    r_list.append(r)

# Use a conservative common radius (median) across modes
r_px = int(max(4, np.median(r_list)))

# ---------- 2) Build absolute power matrix P (inputs × ports) ----------
# P[i, j] = power recorded in port j when input i is launched
P = np.zeros((len(MODES), len(MODES)), dtype=np.float64)
for i, m in enumerate(MODES):
    img = load_gray_linear(out_paths[m], dark=DARK, exposure_scale=EXPOSURE_SCALE)
    P[i, :] = [integrate_circle(img, cx, cy, r_px) for (cx, cy) in centers]

# ---------- 3) Permute columns to maximize diagonal (assign ports) ----------
# This associates each column (physical port) with a logical mode label
best_perm, best_score = None, -1.0
for perm in itertools.permutations(range(len(MODES))):
    score = P[range(len(MODES)), perm].sum()
    if score > best_score:
        best_score, best_perm = score, perm

P = P[:, best_perm]
col_labels = [MODES[j] for j in best_perm]  # logical labels of ports after permutation

# ---------- 4) Crosstalk matrix T (row-normalized), and dB heatmap ----------
row_sums = P.sum(axis=1, keepdims=True) + 1e-12
T = P / row_sums


compute_crosstalk_matrix(out_paths, MODES, centers, r_px, SAVE_DIR)

# ---------- 5) IL (per mode & average) from absolute energies ----------
# Input energies measured from "input images" (same units/exposure)
Ein = np.array([load_gray_linear(in_paths[m], dark=DARK, exposure_scale=EXPOSURE_SCALE).sum()
                for m in MODES])
Tout = P.sum(axis=1)

gains = (Tout + 1e-12) / (Ein + 1e-12)                   # modal power transmission factors
IL_per_mode_dB = -10.0 * np.log10(gains)                 # IL_i in dB (loss is positive)
IL_avg_dB       = IL_per_mode_dB.mean()
MDL_dB          = 10.0 * np.log10(gains.max() / gains.min())  # power-based MDL

print("\n=== Performance metrics ===")
for m, ILdB, g in zip(MODES, IL_per_mode_dB, gains):
    print(f"IL[{m}] = {ILdB:5.2f} dB   (transmission = {100*g:5.2f}%)")
print(f"Average IL = {IL_avg_dB:5.2f} dB")
print(f"MDL        = {MDL_dB:5.2f} dB  (based on modal power gains)")

# ---------- 6) Leakage per input (off-diagonal fraction) ----------
diag_idx = np.argmax(T, axis=1)
print("\n=== Leakage per input (fraction of total for that input) ===")
for i, m in enumerate(MODES):
    good = T[i, diag_idx[i]]
    leak = 1.0 - good
    print(f"{m}: desired port = {col_labels[diag_idx[i]]:>4s}, good = {good:0.3f}, leakage = {leak:0.3f}")

# ---------- 7) Save annotated output images with four port circles ----------
# Use *original* geometric centers (before permutation) for drawing, since those are physical port locations.
# But also save a legend mapping physical port -> logical label.
legend_txt = os.path.join(SAVE_DIR, 'port_mapping.txt')
with open(legend_txt, 'w') as f:
    f.write("Physical port index -> Logical label (after permutation)\n")
    for k, j in enumerate(best_perm):
        f.write(f"Port {k}: {MODES[j]}\n")

# Draw circles at physical centers
for m in MODES:
    img = load_gray_linear(out_paths[m], dark=DARK, exposure_scale=EXPOSURE_SCALE)
    vis = draw_ports_bgr(img, centers, r_px, thickness=2)
    base = os.path.basename(out_paths[m])
    out_path = os.path.join(SAVE_DIR, f"annotated_{m}_{base}")
    cv2.imwrite(out_path, vis)
    # Also show interactively
    cv2.imshow('annotated', vis)
    cv2.waitKey(300)  # brief flash so you can see them
cv2.destroyAllWindows()

print(f"\nSaved annotated images and crosstalk heatmap to: {os.path.abspath(SAVE_DIR)}")
print(f"Port mapping legend: {legend_txt}")
