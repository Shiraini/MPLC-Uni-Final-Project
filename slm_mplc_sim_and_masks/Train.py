from init import *

if __name__ == "__main__":
    # Initialize MPLC system with phase planes, spacing d, and learning rate lr
    MPLC = MPLCSystem(planes, d, lr)
    print('Commence Training...')
    # Train MPLC using Fontaine-style wavefront matching
    # MPLC.fit_fontaine(in_list, tar_list, iter)
    # MPLC.fit_fontaine(inputs[:4], targets[:4], iter)
    MPLC.fit_fontaine([inputs[3]], [targets[3]], iter)
    print('Training Completed!')

    # snapshots = MPLC.sort(test11, True, 100)
    # animate(snapshots, HG11.X, HG11.Y, save_as='HG10 sort MPLCv10.mp4')

    # Save trained MPLC system to file
    with open(file_name, "wb") as f:
        pickle.dump(MPLC, f)


