import numpy as np
import matplotlib.pyplot as plt
import pickle
from init import pp, shape, downsample, file_name
from scipy.ndimage import zoom

# === Load Trained MPLC ===
with open(file_name, "rb") as f:
    system = pickle.load(f)
print("Loaded trained MPLC system")

############################# Masks when it is downsampled
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
#############################

# masks = [zoom(mask.phase, downsample, order=4) for mask in system.planes]
# === Extract Phase Masks ===
normalized_masks = []
for i, plane in enumerate(system.planes):
    # upsampled = zoom(plane.phase, downsample, order=4)
    # norm_phase = (np.mod(upsampled, 2 * np.pi)) / (2 * np.pi)  # Normalize to [0,1]
    # normalized_masks.append(norm_phase)

    phasor = np.exp(1j * plane.phase)
    # --- 2. interpolate real & imag separately
    re_up = zoom(phasor.real, downsample, order=3, mode='wrap')
    im_up = zoom(phasor.imag, downsample, order=3, mode='wrap')
    phasor_up = re_up + 1j*im_up
    # --- 3. back to phase in radians
    phi_up = np.angle(phasor_up)
    # --- 4. normalise to 0–1 for 8-bit output
    norm_phase = (phi_up % (2*np.pi)) / (2*np.pi)
    normalized_masks.append(norm_phase)

# === Display Individual Phase Masks ===
# fig, axs = plt.subplots(1, 3, figsize=(15, 5))
# for i, mask in enumerate(normalized_masks):
#     axs[i].imshow(mask, cmap='gray')
#     axs[i].set_title(f'Mask {i+1} (normalized)')
#     axs[i].axis('off')
# plt.tight_layout()
# plt.show()

# === Combine Horizontally for SLM display ===
gap = np.zeros((1100, 200))
print(len(normalized_masks))
# combined_mask = np.concatenate([normalized_masks[0], gap, normalized_masks[1]], axis=1)
combined_mask = np.concatenate(normalized_masks, axis=1)

plt.figure(figsize=(12, 5))
plt.imshow(combined_mask, cmap='gray')
plt.title('Upsampled Combined SLM Phase Layout (3 Masks)')
plt.axis('off')
plt.show()

# === Save Image for SLM Upload ===
from PIL import Image
Image.fromarray((combined_mask * 255).astype(np.uint8)).save('2planes_1100in_1000x1000.png')
print("Saved: 3_planes_combined.png")