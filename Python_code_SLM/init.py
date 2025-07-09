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
dx = 1.1E-3
dy = np.sqrt(3)/2 * dx
d = 7E-2  # distance between planes
Nx = 540
Ny = 900
win = 400E-6
wout = 400E-6
iter = 50
n_planes = 3
width = 50
pad = 50
Nx += 2*pad
Ny += 2*pad
shape = [Ny, Nx]
wl = 632E-9
pp = 8E-6 #pixel pitch

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


target00 = Mode(0, 0, shape, pp, wout, wl, (dx, dx), absorber)
target10 = Mode(0, 0, shape, pp, wout, wl, (0, dx), absorber)
target01 = Mode(0, 0, shape, pp, wout, wl, (dx, 0), absorber)
target20 = Mode(0, 0, shape, pp, wout, wl, (-dx, dx), absorber)
target11 = Mode(0, 0, shape, pp, wout, wl, (0, 0), absorber)
target02 = Mode(0, 0, shape, pp, wout, wl, (dx, -dx), absorber)


HG00 = Mode(0, 0, shape, pp, win, wl, (0, 0), absorber)
HG10 = Mode(1, 0, shape, pp, win, wl, (0, 0), absorber)
HG01 = Mode(0, 1, shape, pp, win, wl, (0, 0), absorber)
HG20 = Mode(2, 0, shape, pp, win, wl, (0, 0), absorber)
HG11 = Mode(1, 1, shape, pp, win, wl, (0, 0), absorber)
HG02 = Mode(0, 2, shape, pp, win, wl, (0, 0), absorber)

supertarget = copy.deepcopy(target00)
supertarget.field += copy.deepcopy(target01.field) + copy.deepcopy(target10.field) + copy.deepcopy(target02.field) + copy.deepcopy(target20.field) + copy.deepcopy(target11.field)

planes = [Plane(shape, 8E-6, None) for _ in range(n_planes)]

targets = [target00, target10, target01, target20, target11, target02]
inputs = [HG00, HG10, HG01, HG20, HG11, HG02]

inputs_super = copy.deepcopy(inputs)

# for mode in inputs_super:
#     power = np.sum(np.abs(mode.field)**2)
#     mode.field /= np.sqrt(power)
super_field = np.sum([mode.field for mode in inputs_super], axis=0)

supermode = copy.deepcopy(HG10)
supermode.field = super_field

supertarget.visualize()


