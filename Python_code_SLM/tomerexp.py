import matplotlib.pyplot as plt
import numpy as np
from scipy.special import hermite
from mode import Mode
from phase_plane import Plane
from MPLC import MPLCSystem
import copy
from animator import animate
from init import absorber
from exped import *

d = 40E-2

target()
my_target_field = np.load('target.npy')
my_target = Mode(0, 0, shape, 8E-6, 1500E-6, 632E-9, (0, 0), absorber)
my_target.field = my_target_field * np.exp(1j * np.angle(my_target.field))

empty = MPLCSystem([Plane(shape, 8E-6, None)], d, lr=1)
snapshots = empty.sort(my_target, True, 250)
animate(snapshots, my_target.X, my_target.Y, save_as='one_square_tomer.mp4', mode='linear')