import numpy as np
import pyshtools.expand as shtools
class SphereHarmonics:

    def __init__(self, surface_data, normalization_method='zero'):
        # validate normalization method is zero or mean
        if normalization_method not in ['zero', 'mean']:
            raise ValueError("normalization_method must be either 'zero' or 'mean'")
        self.normalization_method = normalization_method
        self.surface_data = surface_data
        self.harmonics = self.process()

    def process(self):
        # check if latitua and longitua are even number from surface
        if self.surface_data.shape[1] % 2 or self.surface_data.shape[0] % 2:
            raise ValueError("The number of samples in latitude and longitude, n, must be even")
        # check if latitua and longitua are equal from surface
        if self.surface_data.shape[1] == self.surface_data.shape[0]:
            s = 1
        elif self.surface_data.shape[1] == 2 * self.surface_data.shape[0]:
            s = 2
        else:
            raise ValueError("Spherical grid must be either (NxN) or (Nx2N)")

        # normalize surface
        if self.normalization_method == 'mean':
            self.surface_data = self.surface_data/np.mean(self.surface_data)
        self.harmonics = shtools.SHExpandDH(self.surface_data, sampling=s)
        if self.normalization_method == 'zero':
            # surface divided by the first harmonic
            self.harmonics = self.harmonics/self.harmonics[0, 0, 0]
        return self.harmonics

    def updates_surface(self, surface_data):
        # recomputes surface from harmonics
        self.surface_data = surface_data
        self.harmonics = self.process()


