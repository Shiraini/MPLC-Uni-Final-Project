import matplotlib.pyplot as plt
import numpy as np
from scipy.special import hermite
from mode import HGMode, LGMode
from phase_plane import Plane
from MPLC import MPLCSystem
from animator import animate
import pickle
import copy

# === Simulation Parameters ===
downsample = 4
lr = 0.002                           # learning rate 0.0005
dx = 0.5E-3                         # x-shift for target modes
dy = np.sqrt(3)/2 * dx              # y-shift for target modes
# d = [0E-2, 8.4E-2, 8.5E-2]  # distance between planes
d = [0E-2, 4.1E-2, 4.1E-2, 8E-2] # distance between planes
# Nx = 512  # 2 planes
# Ny = 512  # 2 planes
Nx = 400  # 3 planes
Ny = 400  # 3 planes
win = 450E-6                        # input beam waist
wout = 60E-6                       # output beam waist
iter = 50                           # number of training iterations
n_planes = 3                    # number of phase planes
width = 150
pad = 50
shape = [Ny, Nx]
wl = 632E-9                         # wavelength [m]
pp = 7.5E-6                        # pixel pitch
file_name = f"MPLC_{n_planes}_half_good_modes"



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
absorber_test = raised_cosine_absorber((Ny, Nx), pad, width)

target00 = HGMode(0, 0, shape, pp, wout, wl, (-dx, -dx), absorber, downsample)
target10 = HGMode(0, 0, shape, pp, wout, wl, (-dx, 0), absorber, downsample)
target01 = HGMode(0, 0, shape, pp, wout, wl, (0, -dx), absorber, downsample)
target11 = HGMode(0, 0, shape, pp, wout, wl, (0, 0), absorber, downsample)
target20 = HGMode(0, 0, shape, pp, wout, wl, (-dx, dx), absorber, downsample)
target02 = HGMode(0, 0, shape, pp, wout, wl, (dx, -dx), absorber, downsample)
target21 = HGMode(0, 0, shape, pp, wout, wl, (0, dx), absorber, downsample)
target12 = HGMode(0, 0, shape, pp, wout, wl, (dx, 0), absorber, downsample)
target22 = HGMode(0, 0, shape, pp, wout, wl, (dx, dx), absorber, downsample)


# === Define Input HG Modes ===
HG00 = HGMode(0, 0, shape, pp, win, wl, (0, 0), absorber, downsample)
HG10 = HGMode(1, 0, shape, pp, win, wl, (0, 0), absorber, downsample)
HG01 = HGMode(0, 1, shape, pp, win, wl, (0, 0), absorber, downsample)
HG11 = HGMode(1, 1, shape, pp, win, wl, (0, 0), absorber, downsample)
HG20 = HGMode(2, 0, shape, pp, win, wl, (0, 0), absorber, downsample)
HG02 = HGMode(0, 2, shape, pp, win, wl, (0, 0), absorber, downsample)
HG21 = HGMode(2, 1, shape, pp, win, wl, (0, 0), absorber, downsample)
HG12 = HGMode(1, 2, shape, pp, win, wl, (0, 0), absorber, downsample)
HG22 = HGMode(2, 2, shape, pp, win, wl, (0, 0), absorber, downsample)

LG00 = LGMode(0, 0, shape, pp, win, wl, (0, 0), absorber, downsample)
LG1_1 = LGMode(1, -1, shape, pp, win, wl, (0, 0), absorber, downsample)
LG10 = LGMode(1, 0, shape, pp, win, wl, (0, 0), absorber, downsample)
LG11 = LGMode(1, 1, shape, pp, win, wl, (0, 0), absorber, downsample)
LG2_2 = LGMode(2, -2, shape, pp, win, wl, (0, 0), absorber, downsample)
LG2_1 = LGMode(2, -1, shape, pp, win, wl, (0, 0), absorber, downsample)
LG20 = LGMode(2, 0, shape, pp, 465E-6, wl, (0, 0), absorber, downsample)
LG21 = LGMode(2, 1, shape, pp, win, wl, (0, 0), absorber, downsample)
LG22 = LGMode(2, 2, shape, pp, win, wl, (0, 0), absorber, downsample)

# === Create Superposition Target ===
supertarget = copy.deepcopy(target00)
supertarget.field += copy.deepcopy(target01.field) + copy.deepcopy(target10.field) + copy.deepcopy(target02.field) + copy.deepcopy(target20.field) + copy.deepcopy(target11.field)

# === Define Planes ===
planes = [Plane(shape, pp, None, downsample) for _ in range(n_planes)]

# === Prepare Inputs and Targets ===
targets = [target00, target10, target01, target11, target20, target02, target21, target12, target22]
HGinputs = [HG00, HG10, HG01, HG11, HG20, HG02, HG21, HG12, HG22]
LGinputs = [LG00, LG1_1, LG10, LG11, LG2_2, LG2_1, LG20, LG21, LG22]

# === Create Superposition Mode ===
inputs_super = copy.deepcopy(HGinputs)
super_field = np.sum([mode.field for mode in inputs_super], axis=0)
supermode = copy.deepcopy(HG10)
supermode.field = super_field