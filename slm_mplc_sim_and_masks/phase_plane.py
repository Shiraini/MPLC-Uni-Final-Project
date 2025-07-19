import matplotlib.pyplot as plt
import numpy as np


class Plane:
    def __init__(self, grid_shape, pixel_pitch, phase, downsample=4):
        # Initialize plane with shape, pixel size, phase mask, and downsampling
        self.Ny, self.Nx = grid_shape
        self.pixel_pitch = pixel_pitch
        self.downsample = downsample
        self.X, self.Y = self._make_grid()
        self.phase = np.zeros((len(self.Y), len(self.X))) if phase is None else phase

    def _make_grid(self):
        # Create 2D coordinate grid (centered, downsampled)
        Lx = self.Nx * self.pixel_pitch
        Ly = self.Ny * self.pixel_pitch
        x = np.linspace(-Lx/2, Lx/2, self.Nx)[::self.downsample]
        y = np.linspace(-Ly/2, Ly/2, self.Ny)[::self.downsample]
        mesh = np.meshgrid(x, y)
        return mesh

    def visualize(self, scale_mm=True):
        # Plot the wrapped phase map
        x_vals = self.X[0, :] * 1e3 if scale_mm else self.X[0, :]
        y_vals = self.Y[:, 0] * 1e3 if scale_mm else self.Y[:, 0]
        units = 'mm' if scale_mm else 'm'

        wrapped_phase = np.mod(self.phase, 2 * np.pi)

        plt.figure(figsize=(6, 5))
        plt.imshow(wrapped_phase, cmap='twilight', extent=[
            x_vals[0], x_vals[-1], y_vals[0], y_vals[-1]
        ])
        plt.title('Phase Mask φ(x, y)')
        plt.xlabel(f'x [{units}]')
        plt.ylabel(f'y [{units}]')
        plt.colorbar(label='Phase [rad]')
        plt.tight_layout()
        plt.show()

    def apply(self, mode, back=False):
        # Apply (or reverse) the phase mask to a mode
        mode.field = mode.field * np.exp(1j * self.phase) if back is False else mode.field * np.exp(-1j * self.phase)
