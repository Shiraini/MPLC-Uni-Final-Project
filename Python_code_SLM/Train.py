import matplotlib.pyplot as plt
import numpy as np
from scipy.special import hermite
from mode import Mode
from phase_plane import Plane
from MPLC import MPLCSystem
from animator import animate
import pickle
import copy
from init import *

if __name__ == "__main__":

    MPLC = MPLCSystem(planes, d, lr)
    print('Commence Training...')
    MPLC.fit_fontaine(inputs, targets, iter)
    print('Training Completed!')

    # snapshots = MPLC.sort(test11, True, 100)
    # animate(snapshots, HG11.X, HG11.Y, save_as='HG10 sort MPLCv10.mp4')


    with open(file_name, "wb") as f:
        pickle.dump(MPLC, f)


