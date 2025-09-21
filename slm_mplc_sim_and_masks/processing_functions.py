import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
import os
import glob


#Images path and initiallization
def first_existing(root, stem):
    """
    searches a given folder for the first file whose name starts with a given stem and returns its path,
    checking common image extensions
    """
    for ext in ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.bmp'):
        cand = sorted(glob.glob(os.path.join(root, f"{stem}*{ext}")))
        if cand:
            return cand[0]
    raise FileNotFoundError(f"No image for {stem} under {root}")


def load_gray_norm(path, bg_percentile=5):
    """
    Loads an image in grayscale, subtracts low-percentile background, and normalizes pixel values to [0,1].
    """
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


# Gaussian fits for finiding the location of the outputs modes
def gaussian_2d(coords, x0, y0, sigma_x, sigma_y, A, offset):
    x, y = coords
    return A * np.exp(-(((x - x0) ** 2) / (2 * sigma_x ** 2) +
                        ((y - y0) ** 2) / (2 * sigma_y ** 2))) + offset


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


def sum_energy(path):
    return load_gray_norm(path).sum()