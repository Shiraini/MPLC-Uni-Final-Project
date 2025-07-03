# Imports
import pandas as pd
import matplotlib.pyplot as plt
from google.colab import files
import numpy as np
from scipy.optimize import curve_fit, root_scalar, root, least_squares

# Functions

def gaussian(x, I0, x0, w):
    return I0 * np.exp(-2 * ((x - x0) ** 2) / w ** 2)

def data_extract(distance, gray):
  # Step 1: Use the horizontal profile (already extracted earlier)
  x = distance.to_numpy()
  y = gray.to_numpy()

  # Step 2: Initial guess [I0, x0, w]
  I0_guess = np.max(y)
  x0_guess = x[np.argmax(y)]
  w_guess = 2.0  # just a start

  popt, _ = curve_fit(gaussian, x, y, p0=[I0_guess, x0_guess, w_guess])
  I0_fit, x0_fit, w0_fit = popt

  # Step 3: Plot the fit
  x_fit = np.linspace(np.min(x), np.max(x), 1000)
  y_fit = gaussian(x_fit, *popt)

  plt.figure(figsize=(10, 5))
  plt.plot(x, y, 'o', label='Data')
  plt.plot(x_fit, y_fit, '-', label='Gaussian Fit', color='red')
  plt.title('Gaussian Fit to Horizontal Beam Profile')
  plt.xlabel('Distance (inches)')
  plt.ylabel('Gray Value')
  plt.legend()
  plt.grid(True)
  plt.show()

  wz_m = w0_fit * 1e-6
  print(f"Beam waist w0 = {w0_fit:.2f} um")

  return wz_m

def system(vars, w1, w2, d):
    z1, z0 = vars
    w0 = np.sqrt(633e-9 / (np.pi * z0))
    eq1 = w1 - w0 * np.sqrt(z1**2 + z0**2)
    eq2 = w2 - w0 * np.sqrt((z1 + d)**2 + z0**2)
    return [eq1, eq2]

def get_z0_z1_w0(w_direct, w_60cm, wavelength):
  w1 = w_direct     # beam radius at unknown z1
  w2 = w_60cm       # beam radius at z1 + 0.6
  d = 0.6           # 60 cm between measurements

# CODE

  # Initial guesses for [z1, z0]
  initial_guess = [0.5, 0.1]

  sol = root(system, initial_guess, args=(w1, w2, d))

  z1_solution, z0_solution = sol.x
  w0_solution = np.sqrt(wavelength / (np.pi * z0_solution))

  print(f"Estimated z₁: {z1_solution:.4f} m")
  print(f"Estimated z₀ (Rayleigh range): {z0_solution:.4f} m")
  print(f"Estimated beam waist ω₀: {w0_solution*1e6:.2f} um")
  return z0_solution, z1_solution, w0_solution

def gaussian_beam_system(vars, w1, w2, d, wavelength):
    z1, z0 = vars

    # Prevent non-physical values
    if z0 <= 0:
        return [1e6, 1e6]

    try:
        w0 = np.sqrt(wavelength / (np.pi * z0))
        eq1 = w1 - w0 * np.sqrt(z1**2 + z0**2)
        eq2 = w2 - w0 * np.sqrt((z1 + d)**2 + z0**2)
        return [eq1, eq2]
    except Exception as e:
        print(f"Error in function evaluation: {e}")
        return [1e6, 1e6]  # Fallback for safety

def get_z0_z1_w0(w_direct, w_60cm, wavelength, d=0.6, verbose=True):
    # Initial guess: [z1, z0] in meters
    initial_guess = [0.2, 0.1]

    # Run least squares solver
    result = least_squares(
        gaussian_beam_system,
        x0=initial_guess,
        args=(w_direct, w_60cm, d, wavelength),
        bounds=([0, 1e-5], [10, 10])  # z1, z0 must be positive
    )

    if not result.success:
        raise RuntimeError("Optimization failed: " + result.message)

    z1_solution, z0_solution = result.x
    w0_solution = np.sqrt(wavelength / (np.pi * z0_solution))

    if verbose:
        print(f"Estimated z₁ (position from waist): {z1_solution:.4f} m")
        print(f"Estimated z₀ (Rayleigh range):       {z0_solution:.4f} m")
        print(f"Estimated beam waist ω₀:             {w0_solution*1e6:.2f} um")

    return z0_solution, z1_solution, w0_solution



wavelength = 633e-9  # 633 nm for red He-Ne laser

# Step 1: Upload the file - Histogram data using ImageJ with right scale
uploaded = files.upload()

# Step 2: Load the Excel file (replace the filename if different)
excel_file = list(uploaded.keys())[0]
df = pd.read_excel(excel_file, sheet_name='HG00')

# Step 3: Extract vertical and horizontal data (first block)
# Columns: A - Distance (Vertical), B - Gray (Vertical)
#          C - Distance (Horizontal), D - Gray (Horizontal)
vertical_distance = df.iloc[2:, 0].dropna().astype(float)
vertical_gray = df.iloc[2:, 1].dropna().astype(float)

horizontal_distance = df.iloc[2:, 2].dropna().astype(float)
horizontal_gray = df.iloc[2:, 3].dropna().astype(float)

# Step 4: Plot
plt.figure(figsize=(10, 5))
plt.plot(vertical_distance, vertical_gray, label='Vertical', marker='o')
plt.plot(horizontal_distance, horizontal_gray, label='Horizontal', marker='s')
plt.title('Gray Value vs. Distance (Sheet: HG00)')
plt.xlabel('Distance (inches)')
plt.ylabel('Gray Value')
plt.legend()
plt.grid(True)
plt.show()

# Extract 60 cm profile data
vertical_distance_60 = df.iloc[2:, 5].dropna().astype(float)
vertical_gray_60 = df.iloc[2:, 6].dropna().astype(float)

horizontal_distance_60 = df.iloc[2:, 7].dropna().astype(float)
horizontal_gray_60 = df.iloc[2:, 8].dropna().astype(float)

print("Direct")
w_direct = data_extract(horizontal_distance, horizontal_gray)
print("\n\n60 cm")
w_60cm = data_extract(horizontal_distance_60, horizontal_gray_60)
print("\n\n")
z0, z1, w0 = get_z0_z1_w0(w_direct, w_60cm, wavelength)

