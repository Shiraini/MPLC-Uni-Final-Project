import matplotlib.pyplot as plt
import numpy as np
from scipy.special import hermite
from mode import Mode
from phase_plane import Plane
from MPLC import MPLCSystem
from animator import animate
import pickle
import copy

# === Simulation Parameters ===
downsample = 8
lr = 0.15                           # learning rate
dx = 0.5E-3                         # x-shift for target modes
dy = np.sqrt(3)/2 * dx              # y-shift for target modes
# d = [0E-2, 11.0E-2, 60.0E-2]        # distance between planes
d = [0E-2, 53.4E-2]
# 43.3, 10.4
# d = 6.3E-2
Nx = 256
Ny = 256
win = 1000E-6                        # input beam waist
wout = 150E-6                       # output beam waist
iter = 50                           # number of training iterations
n_planes = 1                     # number of phase planes
width = 50
pad = 50
shape = [Ny, Nx]
wl = 632E-9                         # wavelength [m]
pp = 7.5E-6 #-5.3 approx                           # pixel pitch
file_name = f"MPLC_{n_planes}MFD={win}_pp={pp}_d={d}_HG11"



# === Optional Absorber Function ===
def raised_cosine_absorber(shape, pad, width):
    # Creates raised cosine edge taper to suppress boundary effects
    Ny, Nx = shape
    Y, X = np.meshgrid(np.arange(Ny), np.arange(Nx), indexing='ij')
    # Compute distance to nearest edge
    dist_top    = Y
    dist_bottom = Ny - 1 - Y
    dist_left   = X
    dist_right  = Nx - 1 - X
    dist_to_edge = np.minimum(np.minimum(dist_top, dist_bottom),np.minimum(dist_left, dist_right))
    # Compute window
    absorber = np.ones_like(dist_to_edge, dtype=np.float32)
    # Where to apply taper
    taper_zone = (dist_to_edge < pad) & (dist_to_edge >= (pad - width))
    dark_zone  = dist_to_edge < (pad - width)
    # Raised cosine transition
    u = (pad - dist_to_edge[taper_zone]) / width
    absorber[taper_zone] = 0.5 * (1 + np.cos(np.pi * u))
    # Fully dark zone
    absorber[dark_zone] = 0.0
    return absorber


absorber = raised_cosine_absorber((Ny/downsample, Nx/downsample), pad/downsample, width/downsample)
# absorber = None

# === Define Target Modes (spatially shifted Gaussians) ===
target00 = Mode(0, 0, shape, pp, wout, wl, (dx, dx), absorber, downsample)
target10 = Mode(0, 0, shape, pp, wout, wl, (0, dx), absorber, downsample)
target01 = Mode(0, 0, shape, pp, wout, wl, (dx, 0), absorber, downsample)
target11 = Mode(0, 0, shape, pp, wout, wl, (0, 0), absorber, downsample)
target20 = Mode(0, 0, shape, pp, wout, wl, (-dx, dx), absorber, downsample)
target02 = Mode(0, 0, shape, pp, wout, wl, (dx, -dx), absorber, downsample)
target21 = Mode(0, 0, shape, pp, wout, wl, (-dx, 0), absorber, downsample)
target12 = Mode(0, 0, shape, pp, wout, wl, (0, -dx), absorber, downsample)
target22 = Mode(0, 0, shape, pp, wout, wl, (-dx, -dx), absorber, downsample)

# === Define Input HG Modes ===
HG00 = Mode(0, 0, shape, pp, win, wl, (0, 0), absorber, downsample)
HG10 = Mode(1, 0, shape, pp, win, wl, (0, 0), absorber, downsample)
HG01 = Mode(0, 1, shape, pp, win, wl, (0, 0), absorber, downsample)
HG11 = Mode(1, 1, shape, pp, win, wl, (0, 0), absorber, downsample)
HG20 = Mode(2, 0, shape, pp, win, wl, (0, 0), absorber, downsample)
HG02 = Mode(0, 2, shape, pp, win, wl, (0, 0), absorber, downsample)
HG21 = Mode(2, 1, shape, pp, win, wl, (0, 0), absorber, downsample)
HG12 = Mode(1, 2, shape, pp, win, wl, (0, 0), absorber, downsample)
HG22 = Mode(2, 2, shape, pp, win, wl, (0, 0), absorber, downsample)

# === Create Superposition Target ===
supertarget = copy.deepcopy(target00)
supertarget.field += copy.deepcopy(target01.field) + copy.deepcopy(target10.field) + copy.deepcopy(target02.field) + copy.deepcopy(target20.field) + copy.deepcopy(target11.field)

# === Define Planes ===
planes = [Plane(shape, pp, None, downsample) for _ in range(n_planes)]

# === Prepare Inputs and Targets ===
targets = [target00, target10, target01, target11, target20, target02, target21, target12, target22]
inputs = [HG00, HG10, HG01, HG11, HG20, HG02, HG21, HG12, HG22]

# === Create Superposition Mode ===
inputs_super = copy.deepcopy(inputs)
super_field = np.sum([mode.field for mode in inputs_super], axis=0)
supermode = copy.deepcopy(HG10)
supermode.field = super_field

in_list = [HG10, HG01, HG20, HG02]
tar_list = [target10, target01, target20, target02]

