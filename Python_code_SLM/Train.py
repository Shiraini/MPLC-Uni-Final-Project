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

    test11 = Mode(1, 1, shape, pp,win, wl, (0,0))

    MPLC = MPLCSystem(planes, d, lr)
    print('Commence Training...')
    MPLC.fit_fontaine(targets, inputs, iter)
    print('Training Completed!')

    # snapshots = MPLC.sort(test11, True, 100)
    # animate(snapshots, HG11.X, HG11.Y, save_as='HG10 sort MPLCv10.mp4')


    with open("MPLCv10", "wb") as f:
        pickle.dump(MPLC, f)


