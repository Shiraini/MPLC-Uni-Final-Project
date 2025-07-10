import numpy as np
import matplotlib.pyplot as plt
import pickle
from init import pp, shape

# === Load Trained MPLC ===
with open("MPLCv13", "rb") as f:
    system = pickle.load(f)
print("Loaded trained MPLC system")

# === Extract Phase Masks ===
normalized_masks = []
for i, plane in enumerate(system.planes):
    phase = plane.phase  # shape: (Ny, Nx)
    norm_phase = (np.mod(phase, 2 * np.pi)) / (2 * np.pi)  # Normalize to [0,1]
    normalized_masks.append(norm_phase)

# === Display Individual Phase Masks ===
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for i, mask in enumerate(normalized_masks):
    axs[i].imshow(mask, cmap='gray')
    axs[i].set_title(f'Mask {i+1} (normalized)')
    axs[i].axis('off')
plt.tight_layout()
plt.show()

# === Combine Horizontally for SLM display ===
combined_mask = np.concatenate(normalized_masks, axis=1)

plt.figure(figsize=(12, 5))
plt.imshow(combined_mask, cmap='gray')
plt.title('Combined SLM Phase Layout (3 Masks)')
plt.axis('off')
plt.show()

# === Save Image for SLM Upload ===
from PIL import Image
Image.fromarray((combined_mask * 255).astype(np.uint8)).save('3_planes_combined.png')
print("Saved: 3_planes_combined.png")
