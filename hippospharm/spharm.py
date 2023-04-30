
import pyshtools.expand as shtools
class SphereHarmonics:

    def __init__(self, surface):
        self.surface = surface
        self.normalization_method = 'zero' # or mean


    def process(self):
        # check if latitua and longitua are even number from surface
        if self.surface.shape[1] % 2 or self.surface.shape[0] % 2:
            raise ValueError("The number of samples in latitude and longitude, n, must be even")
        # check if latitua and longitua are equal from surface
        if self.surface.shape[1] == self.surface.shape[0]:
            s = 1
        elif self.surface.shape[1] == 2 * self.surface.shape[0]:
            s = 2
        else:
            raise ValueError("Spherical grid must be either (NxN) or (Nx2N)")

        # normalize surface
        if self.normalization_method == 'mean':
            self.surface = self.surface/np.mean(self.surface)
        self.harmonics = shtools.SHExpandDH(self.surface, sampling=s)
        if self.normalization_method == 'zero':
            # surface divided by the first harmonic
            self.harmonics = self.harmonics/self.harmonics[0, 0, 0]
        return self.harmonics


