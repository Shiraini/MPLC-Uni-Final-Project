import numpy as np
import matplotlib.pyplot as plt
import pickle
from init import downsample, file_name
from scipy.ndimage import zoom
from PIL import Image

# === Load Trained MPLC ===
with open(file_name, "rb") as f:
    system = pickle.load(f)
print("Loaded trained MPLC system")

# === Display Downsampled Masks ===
mask_before_upsampling = []
for i, plane in enumerate(system.planes):
    phase = plane.phase  # shape: (Ny, Nx)
    norm_phase = (np.mod(phase, 2 * np.pi)) / (2 * np.pi)  # Normalize to [0,1]
    mask_before_upsampling.append(norm_phase)
combined_mask = np.concatenate(mask_before_upsampling, axis=1)
plt.figure(figsize=(12, 5))
plt.imshow(combined_mask, cmap='gray')
plt.title(f'Downsampled Combined SLM Phase Layout ({len(mask_before_upsampling)} Masks)')
plt.axis('off')
plt.show()


# === Upsample Phase Masks ===
normalized_masks = []
for i, plane in enumerate(system.planes):
    phasor = np.exp(1j * plane.phase)
    # interpolate real & imag separately
    re_up = zoom(phasor.real, downsample, order=3, mode='wrap')
    im_up = zoom(phasor.imag, downsample, order=3, mode='wrap')
    phasor_up = re_up + 1j*im_up
    # back to phase in radians
    phi_up = np.angle(phasor_up)
    # normalise to 0–1 for 8-bit output
    norm_phase = (phi_up % (2*np.pi)) / (2*np.pi)
    normalized_masks.append(norm_phase)


# === Combine Horizontally for SLM display ===
# Creates zero-padding (gaps) between phase masks for SLM layout calibration
def conc(x1, x2, ps=316):
    print(f"{x1}-{ps} + 960={(x1-ps + 960)}")
    gap1 = np.zeros((ps*2, x1-ps + 960))
    gap2 = np.zeros((ps * 2, (x2 - ps) - (x1 + ps)))
    gap3 = np.zeros((ps * 2, 960 - (x2 + ps)))
    return gap1, gap2, gap3


# x1, x2: manual calibration values based on optical alignment or pixel measurements
gap1, gap2, gap3 = conc(-500, 479, int((normalized_masks[0].shape[0])/2))
combined_mask = np.concatenate([gap1, normalized_masks[0], gap2, normalized_masks[1], gap3], axis=1)

# Display the final SLM-compatible layout
plt.figure(figsize=(12, 5))
plt.imshow(combined_mask, cmap='gray')
plt.title('Upsampled Combined SLM Phase Layout (3 Masks)')
plt.axis('off')
plt.show()

# === Save Image for SLM Upload ===
Image.fromarray((combined_mask * 255).astype(np.uint8)).save(f"{file_name}.png")
print("Saved: 3_planes_combined.png")