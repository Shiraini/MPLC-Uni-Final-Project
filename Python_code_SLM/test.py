import matplotlib.pyplot as plt
import numpy as np
from scipy.special import hermite
from mode import Mode
from phase_plane import Plane
from init import *
import copy

with open("MPLCv9", "rb") as f:
    loaded_sys = pickle.load(f)
print('uploaded MPLC succesfully')

# snapshots = loaded_sys.sort(supermode, True, 50)
# animate(snapshots, supermode.X, supermode.Y, save_as=f'superposition norm MPLCv9.mp4')
#
# loaded_sys.compute_transfer_matrix(inputs, targets)
# il, mdl = loaded_sys.compute_IL_MDL_from_T()
# print(f"IL = {il} dB; MDL = {mdl} dB")
# loaded_sys.visualize_crosstalk_matrix()

snapshots = loaded_sys.sort(HG00, True, 50)
animate(snapshots, supermode.X, supermode.Y, save_as=f'HG00 sort MPLCv9.mp4')