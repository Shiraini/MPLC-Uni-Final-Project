import numpy as np
import matplotlib.pyplot as plt
import pickle
from init import downsample, file_name, n_planes
from scipy.ndimage import zoom, rotate
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

# def flip(x):
#     return np.fliplr(x)
#
# def generate_rotated_images(image, start_angle=-45, end_angle=45, step=5):
#     rotated_images = {}
#     for angle in range(start_angle, end_angle + 1, step):
#         rotated = rotate(image, angle, reshape=False, order=3, mode='wrap', prefilter=True)
#         rotated = np.clip(rotated, 0, 1)
#         rotated_images[angle] = rotated
#     return rotated_images

def create_combined_mask(normalized_masks, positions, total_width=1920):
    # Assume all masks are same height and width
    mask_height, mask_width = normalized_masks[0].shape
    combined_mask = np.zeros((mask_height, total_width))
    # Shift x from range [-960, 960] to index [0, 1920]
    x_offset = total_width // 2
    for mask, pos in zip(normalized_masks, positions):
        start_x = int(pos + x_offset - mask_width // 2)
        end_x = start_x + mask_width
        # Error checking
        if start_x < 0 or end_x > total_width:
            raise ValueError(f"Mask at position {pos} exceeds image bounds [{start_x}:{end_x}]")
        combined_mask[:, start_x:end_x] = mask
    return combined_mask

def create_combined_mask_xy(normalized_masks, centers_xy, canvas_w=1920, canvas_h=1080, fill=0.0, blend=False):
    if len(normalized_masks) != len(centers_xy):
        raise ValueError("normalized_masks and centers_xy must have the same length.")

    mh, mw = normalized_masks[0].shape
    for m in normalized_masks:
        if m.shape != (mh, mw):
            raise ValueError("All masks must have the same shape.")

    if canvas_h is None:
        canvas_h = mh

    combined = np.full((canvas_h, canvas_w), fill, dtype=np.float32)
    weight   = np.full((canvas_h, canvas_w), 0.0, dtype=np.float32) if blend else None

    x0 = canvas_w // 2
    y0 = canvas_h // 2

    for mask, (cx, cy) in zip(normalized_masks, centers_xy):
        # Convert center (cx,cy) with origin at canvas center -> top-left indices
        start_x = int(cx + x0 - mw // 2)
        end_x   = start_x + mw
        start_y = int(cy + y0 - mh // 2)
        end_y   = start_y + mh

        # Bounds check
        if start_x < 0 or start_y < 0 or end_x > canvas_w or end_y > canvas_h:
            raise ValueError(f"Mask centered at ({cx},{cy}) exceeds canvas bounds: "
                             f"x[{start_x}:{end_x}], y[{start_y}:{end_y}], "
                             f"canvas ({canvas_w}x{canvas_h}).")

        if blend:
            combined[start_y:end_y, start_x:end_x] += mask
            weight[start_y:end_y, start_x:end_x]   += 1.0
        else:
            combined[start_y:end_y, start_x:end_x] = mask

    if blend:
        # Avoid divide-by-zero (background stays 'fill' where weight==0)
        nonzero = weight > 0
        combined[nonzero] = combined[nonzero] / weight[nonzero]

    return combined

# if(n_planes > 1):
#     positions = [(5,1), (671,-13),(-692,-33)]
#     # normalized_masks[1] = np.fliplr(normalized_masks[1])
#     combined_mask = create_combined_mask_xy(normalized_masks, positions)
#     # combined_mask = create_combined_mask(normalized_masks, positions, total_width=1920)
# else:
#     combined_mask = normalized_masks[0]
#     # combined_mask = flip(normalized_masks[0])
positions = [-823 , 774]
combined_mask = create_combined_mask(normalized_masks, positions)

# === Combine Horizontally for SLM display ===
# Creates zero-padding (gaps) between phase masks for SLM layout calibration
# def conc(x1, x2, ps=316):
#     gap1 = np.zeros((ps*2, x1-ps + 960))
#     gap2 = np.zeros((ps * 2, (x2 - ps) - (x1 + ps)))
#     gap3 = np.zeros((ps * 2, 960 - (x2 + ps)))
#     return gap1, gap2, gap3
#
# if(n_planes == 3):
#     # x1, x2: manual calibration values based on optical alignment or pixel measurements
#     gap1, gap2, gap3 = conc(-500, 479, int((normalized_masks[0].shape[0])/2))
#     combined_mask = np.concatenate([gap1, normalized_masks[0], gap2, normalized_masks[1], gap3], axis=1)
# else:
#     combined_mask = normalized_masks[0]

# Display the final SLM-compatible layout
plt.figure(figsize=(12, 5))
plt.imshow(combined_mask, cmap='gray')
plt.title('Upsampled Combined SLM Phase Layout (3 Masks)')
plt.axis('off')
plt.show()

# === Save Image for SLM Upload ===
Image.fromarray((combined_mask * 255).astype(np.uint8)).save(f"{file_name}.png")
print("Saved: 3_planes_combined.png")

# if(n_planes == 1):
#     rotated_images = generate_rotated_images(combined_mask, -5, 5, 1)
#     for angle in rotated_images.keys():
#         rotated_images[angle] = np.squeeze(rotated_images[angle])
#         Image.fromarray((rotated_images[angle] * 255).astype(np.uint8)).save(f"{angle}.png")
#         print("Saved: 3_planes_combined.png")

