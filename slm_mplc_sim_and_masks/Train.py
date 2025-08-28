from numpy.ma.core import append

from init import *

if __name__ == "__main__":
    # Initialize MPLC system with phase planes, spacing d, and learning rate lr
    MPLC = MPLCSystem(planes, d, lr)
    print('Commence Training...')
    # MPLC.fit_fontaine(in_list, tar_list, iter)
    good_modes = []
    inp = []
    good_modes_tar = []
    tar = []
    inp.extend(HGinputs[1:4])
    inp.extend([HGinputs[8]])
    tar.extend(targets[1:4])
    tar.extend([targets[8]])

    good_modes.extend(HGinputs[1:6])
    good_modes.extend([HGinputs[8]])
    good_modes_tar.extend(targets[1:6])
    good_modes_tar.extend([targets[8]])


    MPLC.fit_fontaine(inp, tar, iter)
    # MPLC.fit_fontaine(HGinputs, targets, iter)
    print('Training Completed!')

    # snapshots = MPLC.sort(test11, True, 100)
    # animate(snapshots, HG11.X, HG11.Y, save_as='HG10 sort MPLCv10.mp4')

    # Save trained MPLC system to file
    with open(file_name, "wb") as f:
        pickle.dump(MPLC, f)


