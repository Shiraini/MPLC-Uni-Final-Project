from processing_functions import *

# full-scale, 16-bit or 8-bit grayscale files
paths = dict(HG00='../lab_results/2planes_3modes/HG00.jpg',
             HG10='../lab_results/2planes_3modes/HG10.jpg',
             HG01='../lab_results/2planes_3modes/HG01.jpg')

# (x, y) pixel coordinates of the three output ports on the sensor
centres = {}
r_px = 0
for HG in list(paths.keys()):
    cx, cy, r = find_gaussian_spot(paths[HG])
    centres[HG] = (cx, cy)
    # print(f"HG={HG}, (cx,cy)=({cx}, {cy})")
    r_px += r

r_px = r_px/3               # aperture radius (adjust once)
# print(f"r_px={r_px}")

powers = []                          # rows: input modes   cols: outputs
for key in ['HG00', 'HG01', 'HG10']:  # <- order = rows
    raw = cv2.imread(paths[key], cv2.IMREAD_GRAYSCALE).astype(float)
    raw -= raw.min()                 # crude dark subtraction if needed
    row = [integrate(raw, *c, r_px) for c in list(centres.values())]
    powers.append(row)
P = np.array(powers)

T = P / P.sum(axis=1, keepdims=True)      # each row sums to 1
XT_dB = 10*np.log10(T)                    # convenient dB view

labels = ['00', '01', '10']              # mode labels for both axes

fig, ax = plt.subplots(figsize=(4,4))
im = ax.imshow(XT_dB, vmin=-40, vmax=0)  # default colour-map
ax.set_xticks(range(3))
ax.set_xticklabels(labels)
ax.set_yticks(range(3))
ax.set_yticklabels(labels)

ax.set_xlabel('Detected output mode')
ax.set_ylabel('Injected input mode')
fig.colorbar(im, label='Power [dB]')
plt.title('Crosstalk matrix (dB)')
plt.tight_layout()
plt.show()

insertion_loss_dB = -10*np.log10(P.sum()/cv2.imread(paths['HG00'],
                                     cv2.IMREAD_GRAYSCALE).astype(float).sum())
mdl_dB = 10*np.log10(np.max(np.linalg.svd(T, compute_uv=False)) /
                     np.min(np.linalg.svd(T, compute_uv=False)))
print(f'Insertion loss ≃ {insertion_loss_dB:5.2f} dB')
print(f'Mode-dependent loss (MDL) ≃ {mdl_dB:4.1f} dB')

# The function isn't working correct right now
detect_sorted_mode_from_map('../lab_results/2planes_3modes/HG10.jpg', centres)
"""Add a function that it's input is an image with the gaussian beam. 
the purpose of the function is to recognize which mode was sorted, 
print it and plot the same image with a circle around the relevant gaussian beam"""