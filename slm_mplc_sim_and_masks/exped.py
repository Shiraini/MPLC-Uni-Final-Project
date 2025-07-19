import matplotlib.pyplot as plt
import numpy as np
from scipy.special import hermite
from mode import Mode
from phase_plane import Plane
from MPLC import MPLCSystem
from animator import animate
import pickle
import copy
import numpy as np
import matplotlib.pyplot as plt
from init import shape

GRID   = shape[0]
SIGMA  = 3             # Gaussian σ in pixels
ERASE  = -1               # scale factor for right-click
SAVE_FILE = "target.npy"
def target():
    def make_kernel(sigma):
        R = int(np.ceil(3 * sigma))          # radius ≈3σ captures 99 %
        y, x = np.mgrid[-R:R+1, -R:R+1]
        g = np.exp(-(x**2 + y**2) / (2*sigma**2))
        return g / g.max()                   # peak = 1

    KERNEL = make_kernel(SIGMA)
    Kr, Kc = KERNEL.shape

    def stamp(canvas, cx, cy, scale):
        """
        Add 'scale * KERNEL' centered at (cx,cy) into canvas, clip 0..1.
        """
        r0, c0 = cy - Kr//2, cx - Kc//2
        rs = slice(max(0, r0), min(GRID, r0 + Kr))
        cs = slice(max(0, c0), min(GRID, c0 + Kc))
        ks = (slice(rs.start - r0, rs.stop - r0),
              slice(cs.start - c0, cs.stop - c0))
        canvas[rs, cs] = np.clip(canvas[rs, cs] + scale * KERNEL[ks], 0, 1)

    # ------------------------------------------------------------------ UI
    canvas = np.zeros((GRID, GRID), np.float32)
    fig, ax = plt.subplots(figsize=(5,5))
    im  = ax.imshow(canvas, cmap='inferno', origin='lower', vmin=0, vmax=1)
    ax.set_title("Paint=L MB  |  Erase=R MB  |  Save=s")
    ax.set_xticks([]); ax.set_yticks([])

    def draw_event(event):
        if event.inaxes is not ax: return
        cx, cy = int(round(event.xdata)), int(round(event.ydata))
        if event.button == 1:   # paint
            stamp(canvas, cx, cy, +1.0)
        elif event.button == 3: # erase
            stamp(canvas, cx, cy, ERASE)
        im.set_data(canvas)
        fig.canvas.draw_idle()

    def key_event(event):
        if event.key == 's':
            np.save(SAVE_FILE, canvas)
            print(f"Saved {SAVE_FILE}  (shape {canvas.shape})")
            plt.close(fig)

    cid1 = fig.canvas.mpl_connect('button_press_event', draw_event)
    cid2 = fig.canvas.mpl_connect('motion_notify_event', draw_event)
    cid3 = fig.canvas.mpl_connect('key_press_event',    key_event)

    plt.show()



