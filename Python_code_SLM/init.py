import matplotlib.pyplot as plt
import numpy as np
from scipy.special import hermite
from mode import Mode
from phase_plane import Plane
from MPLC import MPLCSystem
from animator import animate
import pickle
import copy

lr = 0.15
dx = 1E-3
dy = 0.5E-3
d = 10E-2  # distance between planes
Nx = 600
Ny = 600
win = 400E-6
wout = 300E-6
iter = 50
n_planes = 3
width = 50
pad = 50
Nx += 2*pad
Ny += 2*pad
shape = [Ny, Nx]

def raised_cosine_absorber(shape, pad, width):
    Ny, Nx = shape
    Y, X = np.meshgrid(np.arange(Ny), np.arange(Nx), indexing='ij')

    # Compute distance to nearest edge
    dist_top    = Y
    dist_bottom = Ny - 1 - Y
    dist_left   = X
    dist_right  = Nx - 1 - X

    dist_to_edge = np.minimum(np.minimum(dist_top, dist_bottom),
                              np.minimum(dist_left, dist_right))

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
absorber = raised_cosine_absorber((Nx, Ny), pad, width)

# target00 = Mode(0, 0, shape, 8E-6, wout, 632E-9, (-dx, dy))
target00 = Mode(0, 0, shape, 8E-6, wout, 632E-9, (0, 0.5E-3), absorber)
target01 = Mode(0, 0, shape, 8E-6, wout, 632E-9, (0, 0.5E-3), absorber)
target10 = Mode(0, 0, shape, 8E-6, wout, 632E-9, (1E-3,0.5E-3), absorber)
target11 = Mode(0, 0, shape, 8E-6, wout, 632E-9, (-1E-3, -0.5E-3), absorber)
target20 = Mode(0, 0, shape, 8E-6, wout, 632E-9, (0, -0.5E-3), absorber)
target02 = Mode(0, 0, shape, 8E-6, wout, 632E-9, (1E-3, -0.5E-3), absorber)

# HG00 = Mode(0, 0, shape, 8E-6, win, 632E-9, (0, 0))
HG00 = Mode(0, 0, shape, 8E-6, win, 632E-9, (0, 0), absorber)
HG01 = Mode(0, 1, shape, 8E-6, win, 632E-9, (0, 0), absorber)
HG10 = Mode(1, 0, shape, 8E-6, win, 632E-9, (0, 0), absorber)
HG11 = Mode(1, 1, shape, 8E-6, win, 632E-9, (0, 0), absorber)
HG20 = Mode(2, 0, shape, 8E-6, win, 632E-9, (0, 0), absorber)
HG02 = Mode(0, 2, shape, 8E-6, win, 632E-9, (0, 0), absorber)

supertarget = copy.deepcopy(target01)
supertarget.field += copy.deepcopy(target10.field) + copy.deepcopy(target02.field) + copy.deepcopy(target20.field) + copy.deepcopy(target11.field)

planes = [Plane(shape, 8E-6, None) for _ in range(n_planes)]

targets = [target01, target10, target11, target20, target02]
inputs = [HG01, HG10, HG11, HG20, HG02]

inputs_super = copy.deepcopy(inputs)

for mode in inputs_super:
    power = np.sum(np.abs(mode.field)**2)
    mode.field /= np.sqrt(power)
super_field = np.sum([mode.field for mode in inputs_super], axis=0)

supermode = copy.deepcopy(HG10)
supermode.field = super_field
