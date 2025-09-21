# ===== processing_results.py =====
from processing_functions import *
import numpy as np
import matplotlib.pyplot as plt
import itertools

INPUT_DIR = '../lab_results/28.8/3planes4modes/inputs'
OUTPUT_DIR = '../lab_results/28.8/3planes4modes/non-filtered'
MODES = ['HG10','HG01','HG11','HG22']  # Used modes

# Dictionaries mapping each mode to its image file path
out_paths = {m: first_existing(OUTPUT_DIR, m) for m in MODES}
in_paths = {m: first_existing(INPUT_DIR,  m) for m in MODES}

# Spot centers and average radius for all modes
centres = {}
r_accum = 0
for m in MODES:
    cx, cy, r = find_gaussian_spot(out_paths[m])
    centres[m] = (cx, cy)
    r_accum += r
r_px = max(3, int(round(r_accum / len(MODES))))

powers = []
centre_list = list(centres.values())
for m in MODES:
    img = load_gray_norm(out_paths[m])
    row = [integrate(img, cx, cy, r_px) for (cx, cy) in centre_list]
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

XT_dB = 10*np.log10(np.clip(T, 1e-6, 1.0))  # Crosstalk matrix in dB
fig, ax = plt.subplots(figsize=(4,4))
im = ax.imshow(XT_dB, vmin=-40, vmax=0, cmap='viridis')
ax.set_xticks(range(4)); ax.set_xticklabels(col_labels)
ax.set_yticks(range(4)); ax.set_yticklabels(MODES)
ax.set_xlabel('Detected output')
ax.set_ylabel('Injected input')
fig.colorbar(im, label='Power [dB]')
plt.title('Crosstalk matrix (dB)')
plt.tight_layout(); plt.show()

svals = np.linalg.svd(T, compute_uv=False)
mdl_dB = 10*np.log10(np.max(svals)/np.min(svals))  # Compute mode-dependent loss
print(f"Mode-dependent loss (MDL) ≃ {mdl_dB:4.2f} dB")

# Calculate insertion loss per mode from input vs. output energy
for i, m in enumerate(MODES):
    Ein  = sum_energy(in_paths[m])
    Eout = P[i, :].sum()
    ILm  = -10*np.log10((Eout+1e-12)/(Ein+1e-12))
    print(f"IL[{m}] ≃ {ILm:.2f} dB")

print(f"Average IL ≃ {np.mean([-10*np.log10((P[i,:].sum()+1e-12)/(sum_energy(in_paths[m])+1e-12)) for i,m in enumerate(MODES)]):.2f} dB")
