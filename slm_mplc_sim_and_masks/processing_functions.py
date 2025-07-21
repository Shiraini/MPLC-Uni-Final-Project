import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter


def find_gaussian_spot(image_path):
    # Load and normalize image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(float)
    img -= img.min()
    img /= img.max()
    # Find the brightest pixel (initial guess)
    y0, x0 = np.unravel_index(np.argmax(img), img.shape)
    # Crop a small region around the peak for Gaussian fit
    crop_radius = 20
    x_min, x_max = x0 - crop_radius, x0 + crop_radius + 1
    y_min, y_max = y0 - crop_radius, y0 + crop_radius + 1
    cropped = img[y_min:y_max, x_min:x_max]
    # Build x, y coordinate arrays for fitting
    x = np.arange(cropped.shape[1])
    y = np.arange(cropped.shape[0])
    x, y = np.meshgrid(x, y)
    def gaussian_2d(coords, x0, y0, sigma_x, sigma_y, A, offset):
        x, y = coords
        return A * np.exp(-(((x - x0) ** 2) / (2 * sigma_x ** 2) +
                            ((y - y0) ** 2) / (2 * sigma_y ** 2))) + offset
    # Initial guess
    guess = (crop_radius, crop_radius, 5, 5, 1, 0)
    try:
        popt, _ = curve_fit(gaussian_2d,
                            (x.ravel(), y.ravel()),
                            cropped.ravel(), p0=guess)
        x_fit, y_fit, sigma_x, sigma_y, *_ = popt
        # Convert fit center to image coordinates
        cx = x_min + x_fit
        cy = y_min + y_fit
        radius = np.mean([sigma_x, sigma_y]) * 2.355  # FWHM estimate
        return int(round(cx)), int(round(cy)), int(round(radius/2))
    except RuntimeError:
        print("Fit failed.")
        return None


def integrate(img, cx, cy, r):
    y,x = np.ogrid[:img.shape[0], :img.shape[1]]
    mask = (x-cx)**2 + (y-cy)**2 <= r**2
    return img[mask].sum()


def detect_sorted_mode_from_map(image_path,
                                      port_map,
                                      position_thresh=20,
                                      similarity_thresh=0.6):
    def find_gaussian_spot(image_path):
        from scipy.optimize import curve_fit
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(float)
        img -= img.min(); img /= img.max()
        y0, x0 = np.unravel_index(np.argmax(img), img.shape)
        crop_radius = 20
        x_min, x_max = x0 - crop_radius, x0 + crop_radius + 1
        y_min, y_max = y0 - crop_radius, y0 + crop_radius + 1
        cropped = img[y_min:y_max, x_min:x_max]
        x = np.arange(cropped.shape[1])
        y = np.arange(cropped.shape[0])
        x, y = np.meshgrid(x, y)
        def gaussian_2d(coords, x0, y0, sx, sy, A, off):
            x, y = coords
            return A*np.exp(-(((x-x0)**2)/(2*sx**2)+((y-y0)**2)/(2*sy**2)))+off
        guess = (crop_radius, crop_radius, 5, 5, 1, 0)
        try:
            popt, _ = curve_fit(gaussian_2d, (x.ravel(), y.ravel()), cropped.ravel(), p0=guess)
            x_fit, y_fit, sx, sy, *_ = popt
            cx = x_min + x_fit
            cy = y_min + y_fit
            radius = np.mean([sx, sy]) * 2.355
            return int(round(cx)), int(round(cy)), int(round(radius/2))
        except RuntimeError:
            print("Fit failed.")
            return None

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read {image_path}")
    img = img.astype(float)
    img -= img.min()
    if img.max() == 0:
        print("Image is empty.")
        return None
    img /= img.max()

    # 1. Use find_gaussian_spot to locate the brightest beam
    result = find_gaussian_spot(image_path)
    if result is None:
        return None
    cx, cy, r_px = result

    # 2. Compare detected centre to each known port
    def distance(p1, p2):
        return np.hypot(p1[0]-p2[0], p1[1]-p2[1])

    closest_mode = None
    closest_dist = float('inf')
    for mode, (px, py) in port_map.items():
        d = distance((cx, cy), (px, py))
        if d < closest_dist:
            closest_mode = mode
            closest_dist = d

    if closest_dist > position_thresh:
        print("❓ Detected beam doesn't match any known port.")
        return None

    # 3. Measure power at all known ports
    h, w = img.shape
    Y, X = np.ogrid[:h, :w]
    powers = {}
    for mode, (px, py) in port_map.items():
        mask = (X - px)**2 + (Y - py)**2 <= r_px**2
        powers[mode] = img[mask].sum()

    max_power = max(powers.values())
    selected_modes = [m for m, p in powers.items() if p >= similarity_thresh * max_power]

    # 4. Show visual result
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    for mode, (px, py) in port_map.items():
        color = 'lime' if mode in selected_modes else 'red'
        circ = plt.Circle((px, py), r_px, fill=False, color=color, linewidth=2)
        ax.add_patch(circ)
        ax.text(px, py, mode, color=color, fontsize=9, ha='center', va='center')
    ax.set_axis_off()
    plt.title("Detected Mode(s)")
    plt.show()

    # 5. Report
    if not selected_modes:
        print("❓ No clear beam matched.")
        return None
    if len(selected_modes) == 1:
        print(f"✅ Detected mode: {selected_modes[0]}")
        return selected_modes[0]
    print(f"✅ Detected superposition: {' + '.join(selected_modes)}")
    return selected_modes

