import matplotlib.pyplot as plt
import numpy as np
from scipy.special import hermite
from mode import Mode
from phase_plane import Plane
from init import *
import copy

with open("MPLCv12", "rb") as f:
    loaded_sys = pickle.load(f)
print('uploaded MPLC succesfully')

snapshots = loaded_sys.sort(supermode, True, 50)
animate(snapshots, supermode.X, supermode.Y, save_as=f'superposition norm MPLCv12 (6).mp4')

loaded_sys.compute_transfer_matrix(inputs[:6], targets[:6])
il, mdl = loaded_sys.compute_IL_MDL_from_T()
print(f"IL = {il} dB; MDL = {mdl} dB")
loaded_sys.visualize_crosstalk_matrix()

# snapshots = loaded_sys.sort(supermode, True, 50)
# animate(snapshots, supermode.X, supermode.Y, save_as=f'HG10 generate MPLCv11.mp4')