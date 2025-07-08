import matplotlib.pyplot as plt
import numpy as np
from scipy.special import hermite
from mode import Mode
from phase_plane import Plane
from init import *

if __name__ == "__main__":
    with open("MPLC 3P tri", "rb") as f:
        loaded_sys = pickle.load(f)

    snapshots = loaded_sys.sort(supermode, True, 100)
    animate(snapshots, supermode.X, supermode.Y, save_as='supermode sort tri.mp4', mode='linear')