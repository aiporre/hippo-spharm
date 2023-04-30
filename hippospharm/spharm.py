import numpy as np
import pandas as pd
import pyshtools as pysh
import matplotlib.pyplot as plt


class SphereHarmonics:

    def __init__(self, surface_data, normalization_method='zero'):
        # validate normalization method is zero or mean
        if normalization_method not in ['zero', 'mean']:
            raise ValueError("normalization_method must be either 'zero' or 'mean'")
        self.normalization_method = normalization_method
        self.surface_data = surface_data
        self.harmonics = self.process()
        self.extract_harmdata()
        self.lmax = self.harmonics.shape[1]

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
            self.surface_data = self.surface_data / np.mean(self.surface_data)
        self.harmonics = pysh.expand.SHExpandDH(self.surface_data, sampling=s)
        if self.normalization_method == 'zero':
            # surface divided by the first harmonic
            self.harmonics = self.harmonics / self.harmonics[0, 0, 0]
        return self.harmonics

    def updates_surface(self, surface_data):
        # recomputes surface from harmonics
        self.surface_data = surface_data
        self.harmonics = self.process()
        self.extract_harmdata()
        
    def plot_spectrum(self):
        # plot spectrum
        clm = pysh.SHCoeffs.from_array(self.harmonics)
        clm.plot_spectrum(unit='per_l', xscale='log', yscale='log', show=False)
        plt.show()

    def extract_harmdata(self):
        # save harmonics to csv file
        # collects amplitude power real, imaginary, phase order and harmonic pairs
        harm = self.harmonics
        harmdata = pd.DataFrame()
        for degree in range(len(harm[0])):
            for order in range(degree + 1):
                harmdata = harmdata.append(pd.Series({'degree': int(degree),
                                                      'order': int(order),
                                                      'value': harm[0][degree, order]}), ignore_index=True)

            for order in range(1, degree + 1):
                harmdata = harmdata.append(pd.Series({'degree': int(degree),
                                                      'order': -int(order),
                                                      'value': harm[1][degree, order]}), ignore_index=True)

        harmdata['amplitude'] = np.abs(harmdata['value'])
        harmdata['power'] = harmdata['amplitude'] ** 2
        harmdata['real'] = np.real(harmdata['value'])
        harmdata['imag'] = np.imag(harmdata['value'])
        harmdata['degree'] = np.int_(np.real(harmdata['degree']))
        harmdata['order'] = np.int_(np.real(harmdata['order']))
        harmdata['harmonic'] = ''
        for i in range(len(harmdata)):
            harmdata.at[i, 'harmonic'] = 'm=' + str(harmdata.iloc[i]['degree']) \
                                         + ' n=' + str(harmdata.iloc[i]['order'])
        self.harmdata = harmdata


    def save_to_csv(self, filename, directory):
        # creates dictories if not exist
        import os
        if not os.path.exists(directory):
            os.makedirs(directory)
        # saves harmonics to csv file
        self.harmdata.to_csv(directory + '/' + filename, index=False)