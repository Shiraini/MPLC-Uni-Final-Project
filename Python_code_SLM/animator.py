import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors

pp = 8E-6  # pixel pitch [m]


def animate(
    snapshots,            # list of 2D complex field frames
    X, Y,                 # coordinate grids
    mode="linear",        # "linear" | "clip" | "dB"
    clip_percent=99.0,    # used if mode == "clip"
    vmin_db=-40,          # used if mode == "dB"
    interval=60,
    save_as=None,
):
    """
    Visualise intensity + phase movie.

    mode:
        "linear"  -> colour-bar spans [0, global max]
        "clip"    -> spans [0, percentile clip_percent of global intensities]
        "dB"      -> log-scale in dB relative to launch peak
    """

    # ---------- common prep ----------
    x_vals = X[0] * 1e3
    y_vals = Y[:, 0] * 1e3
    extent = [x_vals[0], x_vals[-1], y_vals[0], y_vals[-1]]

    # ---------- choose scaling ----------
    intensities = [np.abs(s) ** 2 * pp**2 for s in snapshots]
    launch_peak = np.max(intensities[0])

    if mode == "linear":
        vmin_i, vmax_i = 0.0, np.max(intensities[0])
        def norm_I(I):
            return I

        cmap_int = "inferno"
        norm_obj = mcolors.Normalize(vmin=vmin_i, vmax=vmax_i)
        colorbar_label = "Intensity"

    elif mode == "clip":
        flat = np.concatenate([I.ravel() for I in intensities])
        vmax_i = np.percentile(flat, clip_percent)
        vmin_i = 0.0
        def norm_I(I):
            return I

        cmap_int = "inferno"
        norm_obj = mcolors.Normalize(vmin=vmin_i, vmax=vmax_i)
        colorbar_label = f"Intensity (≤ {clip_percent}th pct)"

    elif mode == "dB":
        def to_db(I):
            return 10 * np.log10(I / launch_peak + 1e-12)
        def norm_I(I):
            return to_db(I)

        cmap_int = "inferno"
        norm_obj = mcolors.Normalize(vmin=vmin_db, vmax=0.0)
        colorbar_label = f"Intensity [dB] (0 dB = launch peak)"

    else:
        raise ValueError("mode must be 'linear', 'clip', or 'dB'")

    # ---------- figure ----------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    im1 = ax1.imshow(
        norm_I(intensities[0]),
        cmap=cmap_int,
        norm=norm_obj,
        extent=extent,
        origin="lower",
    )
    im2 = ax2.imshow(
        np.angle(snapshots[0]),
        cmap="twilight",
        extent=extent,
        origin="lower",
        vmin=-np.pi,
        vmax=np.pi,
    )

    ax1.set_title("Intensity")
    ax2.set_title("Phase")
    ax1.set_xlabel("x [mm]"); ax1.set_ylabel("y [mm]")
    ax2.set_xlabel("x [mm]")

    fig.colorbar(im1, ax=ax1, shrink=0.8, label=colorbar_label)
    fig.colorbar(im2, ax=ax2, shrink=0.8, label="Phase [rad]")

    # ---------- animation update ----------
    def update(i):
        im1.set_data(norm_I(intensities[i]))
        im2.set_data(np.angle(snapshots[i]))
        ax1.set_title(f"Intensity — frame {i}")
        ax2.set_title(f"Phase — frame {i}")
        return [im1, im2]

    ani = animation.FuncAnimation(
        fig, update, frames=len(snapshots), interval=interval, blit=False
    )

    if save_as:
        ani.save(save_as, writer="ffmpeg" if save_as.endswith(".mp4") else "pillow", fps=30)
    else:
        plt.show()
