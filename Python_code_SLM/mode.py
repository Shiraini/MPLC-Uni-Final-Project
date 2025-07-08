import matplotlib.pyplot as plt
import numpy as np
from scipy.special import hermite


class Mode:
    def __init__(self, m, n, grid_shape, pixel_pitch, w0, wavelength, center=(0, 0), absorber=None):
        self.m = m
        self.n = n
        self.Ny, self.Nx = grid_shape
        self.pixel_pitch = pixel_pitch
        self.w0 = w0
        self.wavelength = wavelength
        self.k = 2 * np.pi / wavelength
        self.z = 0
        self.center = center
        self.absorber = absorber
        self.X, self.Y = self._make_grid()
        self.field = self._generate_field()
        if self.absorber is not None:
            self.field *= self.absorber

    def _make_grid(self):
            Lx = self.Nx * self.pixel_pitch
            Ly = self.Ny * self.pixel_pitch
            x = np.linspace(-Lx/2, Lx/2, self.Nx)
            y = np.linspace(-Ly/2, Ly/2, self.Ny)
            return np.meshgrid(x, y)


    def _generate_field(self):
        # Returns the HG mode complex field (assume z=0 plane for now)

        X, Y, w0 = self.X, self.Y, self.w0
        dx, dy = self.center
        X = X - dx
        Y = Y - dy

        ξ = np.sqrt(2) * X / w0
        η = np.sqrt(2) * Y / w0

        Hm = hermite(self.m)(ξ)
        Hn = hermite(self.n)(η)

        envelope = np.exp(-(X ** 2 + Y ** 2) / w0 ** 2)

        mode = Hm * Hn * envelope
        return mode

    def visualize(self, scale_mm=True):
        """Display intensity and phase of the mode field."""
        x_vals = self.X[0, :] * 1e3 if scale_mm else self.X[0, :]
        y_vals = self.Y[:, 0] * 1e3 if scale_mm else self.Y[:, 0]
        units = 'mm' if scale_mm else 'm'

        # Intensity plot
        plt.figure(figsize=(6, 5))
        plt.imshow(np.abs(self.field) ** 2, cmap='inferno', extent=[
            x_vals[0], x_vals[-1], y_vals[0], y_vals[-1]
        ],origin='lower')
        plt.title(f'Intensity |HG({self.m},{self.n})|²')
        plt.xlabel(f'x [{units}]')
        plt.ylabel(f'y [{units}]')
        plt.colorbar(label='Intensity')
        plt.tight_layout()

        # # Phase plot
        # plt.figure(figsize=(6, 5))
        # plt.imshow(np.angle(self.field), cmap='twilight', extent=[
        #     x_vals[0], x_vals[-1], y_vals[0], y_vals[-1]
        # ],origin='lower')
        # plt.title(f'Phase ∠HG({self.m},{self.n})')
        # plt.xlabel(f'x [{units}]')
        # plt.ylabel(f'y [{units}]')
        # plt.colorbar(label='Phase [rad]')
        # plt.tight_layout()
        plt.show()


    def propagate(self, z):
        # Free-space propagation over distance z using angular spectrum
        #move to angular spectrum
        fx = np.fft.fftfreq(self.Nx, d=self.pixel_pitch)  # [1/m]
        fy = np.fft.fftfreq(self.Ny, d=self.pixel_pitch)
        FX, FY = np.meshgrid(fx, fy)
        spectra = np.fft.fft2(self.field)

        # Argument inside square root
        sqrt_arg = (1 / self.wavelength) ** 2 - FX ** 2 - FY ** 2
        sqrt_arg = np.maximum(0, sqrt_arg)  # suppress evanescent waves
        phase = 2 * np.pi * z * np.sqrt(sqrt_arg)
        H = np.exp(1j * phase)
        propagated_spectra = spectra * H
        self.field = np.fft.ifft2(propagated_spectra)
        if self.absorber is not None:
            self.field *= self.absorber


    def propagate_segmented(self, z, steps=10):
        dz = z / steps
        for _ in range(steps):
            self.propagate(dz)
            yield self.field.copy()


