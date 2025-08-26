import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import envelope
from scipy.special import hermite, eval_genlaguerre


class HGMode:
    def __init__(self, m, n, grid_shape, pixel_pitch, w0, wavelength, center=(0, 0), absorber=None, downsample=4):
        # Initialize Hermite-Gaussian mode parameters
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
        self.downsample = downsample
        self.X, self.Y = self._make_grid()
        self.field = self._generate_field()
        self.norm = np.sqrt(np.sum(abs(self.field)**2 * self.pixel_pitch**2))
        self.field /= self.norm

        # Apply optional absorber (e.g., aperture mask)
        if self.absorber is not None:
            self.field *= self.absorber

    @property
    def power(self):
        # Return total optical power (integrated intensity)
        return np.sum(abs(self.field)**2 * self.pixel_pitch**2)

    def _make_grid(self):
        # Create downsampled spatial grid centered at (0,0)
        Lx = self.Nx * self.pixel_pitch
        Ly = self.Ny * self.pixel_pitch
        x = np.linspace(-Lx/2, Lx/2, self.Nx)[::self.downsample]
        y = np.linspace(-Ly/2, Ly/2, self.Ny)[::self.downsample]
        X, Y = np.meshgrid(x, y)
        return X, Y


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
        # Display intensity (and optionally phase) of the mode
        x_vals = self.X[0, :] * 1e3 if scale_mm else self.X[0, :]
        y_vals = self.Y[:, 0] * 1e3 if scale_mm else self.Y[:, 0]
        units = 'mm' if scale_mm else 'm'

        # Intensity plot
        plt.figure(figsize=(6, 5))
        plt.imshow(np.abs(self.field) ** 2, cmap='inferno', extent=[x_vals[0], x_vals[-1], y_vals[0], y_vals[-1]], origin='lower')
        plt.title(f'Intensity |HG({self.m},{self.n})|²')
        plt.xlabel(f'x [{units}]')
        plt.ylabel(f'y [{units}]')
        plt.colorbar(label='Intensity')
        plt.tight_layout()

        # Phase plot
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
        # Propagate the field over distance z using angular spectrum
        pp = self.pixel_pitch * self.downsample
        fx = np.fft.fftfreq(len(self.X), d=pp)
        fy = np.fft.fftfreq(len(self.Y), d=pp)
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
        # Propagate in small steps and yield field after each
        dz = z / steps
        for _ in range(steps):
            self.propagate(dz)
            yield self.field.copy()



class LGMode:
    def __init__(self, p, l, grid_shape, pixel_pitch, w0, wavelength, center=(0, 0), absorber=None, downsample=4):
        # Initialize Hermite-Gaussian mode parameters
        self.p = p
        self.l = l
        self.Ny, self.Nx = grid_shape
        self.pixel_pitch = pixel_pitch
        self.w0 = w0
        self.wavelength = wavelength
        self.k = 2 * np.pi / wavelength
        self.z = 0
        self.center = center
        self.absorber = absorber
        self.downsample = downsample
        self.X, self.Y = self._make_grid()
        self.field = self._generate_field()
        self.norm = np.sqrt(np.sum(abs(self.field)**2 * self.pixel_pitch**2))
        self.field /= self.norm

        # Apply optional absorber (e.g., aperture mask)
        if self.absorber is not None:
            self.field *= self.absorber

    @property
    def power(self):
        # Return total optical power (integrated intensity)
        return np.sum(abs(self.field)**2 * self.pixel_pitch**2)

    def _make_grid(self):
        # Create downsampled spatial grid centered at (0,0)
        Lx = self.Nx * self.pixel_pitch
        Ly = self.Ny * self.pixel_pitch
        x = np.linspace(-Lx/2, Lx/2, self.Nx)[::self.downsample]
        y = np.linspace(-Ly/2, Ly/2, self.Ny)[::self.downsample]
        X, Y = np.meshgrid(x, y)
        return X, Y


    def _generate_field(self):
        # Returns the HG mode complex field (assume z=0 plane for now)
        X, Y, w0 = self.X, self.Y, self.w0
        dx, dy = self.center
        X = X - dx
        Y = Y - dy
        r = np.hypot(X,Y)
        phi = np.arctan2(Y,X)
        abs_l = abs(self.l)
        x_arg = 2.0 * (r**2)/(w0**2)
        Lpl = eval_genlaguerre(self.p, abs_l, x_arg)
        radial = (np.sqrt(2.0) * r/w0)**abs_l * Lpl
        envelope = np.exp(-(r**2 / (w0**2)))
        az_phase = np.exp(1j * self.l * phi)
        mode = radial * envelope * az_phase
        return mode

    def visualize(self, scale_mm=True):
        # Display intensity (and optionally phase) of the mode
        x_vals = self.X[0, :] * 1e3 if scale_mm else self.X[0, :]
        y_vals = self.Y[:, 0] * 1e3 if scale_mm else self.Y[:, 0]
        units = 'mm' if scale_mm else 'm'

        # Intensity plot
        plt.figure(figsize=(6, 5))
        plt.imshow(np.abs(self.field) ** 2, cmap='inferno', extent=[x_vals[0], x_vals[-1], y_vals[0], y_vals[-1]], origin='lower')
        plt.title(f'Intensity |LG({self.p},{self.l})|²')
        plt.xlabel(f'x [{units}]')
        plt.ylabel(f'y [{units}]')
        plt.colorbar(label='Intensity')
        plt.tight_layout()

        # Phase plot
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
        # Propagate the field over distance z using angular spectrum
        pp = self.pixel_pitch * self.downsample
        fx = np.fft.fftfreq(len(self.X), d=pp)
        fy = np.fft.fftfreq(len(self.Y), d=pp)
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
        # Propagate in small steps and yield field after each
        dz = z / steps
        for _ in range(steps):
            self.propagate(dz)
            yield self.field.copy()



