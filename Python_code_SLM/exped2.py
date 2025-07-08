from exped import target, GRID
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import hermite
from mode import Mode
from phase_plane import Plane
from MPLC import MPLCSystem
import copy
from animator import animate
from init import absorber,shape



lr = 0.1
dx = 1E-3
dy = 0.5E-3
d = 10E-2  # distance between planes
win = 400E-6
wout = 300E-6
iter = 50
n_planes = 1

planes = [Plane(shape, 8E-6, None) for _ in range(n_planes)]

target()
my_target_field = [np.load('target.npy')]
my_target = [Mode(0, 0, shape, 8E-6, 1500E-6, 632E-9, (0, 0), absorber)]
my_target[0].field = my_target_field[0] * np.exp(1j * np.angle(my_target[0].field))
input = [Mode(0, 0, shape, 8E-6, 1500E-6, 632E-9, (0, 0),absorber)]

MPLC_fun = MPLCSystem(planes, d, lr)
print('Commence Training...')
MPLC_fun.fit_fontaine(input, my_target, iter)
print('Training Completed!')

snapshots = MPLC_fun.sort(input[0], True, 50)
animate(snapshots, input[0].X, input[0].Y, save_as='fun_target_dB.mp4', mode='linear')

area = 8E-6 * 8E-6
inital_power = abs(input[0].field)**2
MPLC_fun.sort(input[0])
output_power = abs(input[0].field)**2
inital_power = inital_power.sum()*area
output_power = output_power.sum()*area
print(f'input power = {inital_power}; output power = {output_power}')
