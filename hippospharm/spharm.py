import os
import numpy as np
import pandas as pd
# import pyshtools as pysh
import matplotlib.pyplot as plt
import seaborn as sns

class SphereHarmonics:

    def __init__(self, surface_data, normalization_method='zero'):
        # validate normalization method is zero or mean
        if normalization_method is not None and normalization_method not in ['zero', 'mean']:
            raise ValueError("normalization_method must be either 'zero' or 'mean'")
        self.normalization_method = normalization_method
        self.surface_data = surface_data
        self.harmonics = self.process()
        self.extract_harmdata()
        self.lmax = self.harmonics.shape[1]
        self.name = 'spharm'
        self.sampling = None

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
        self.sampling = s

        # normalize surface
        if self.normalization_method == 'mean':
            self.surface_data = self.surface_data / np.mean(self.surface_data)
        self.harmonics = pysh.expand.SHExpandDHC(self.surface_data, sampling=s)
        if self.normalization_method == 'zero':
            # surface divided by the first harmonic
            self.harmonics = self.harmonics / self.harmonics[0, 0, 0]
        return self.harmonics

    def updates_surface(self, surface_data):
        # recomputes surface from harmonics
        self.surface_data = surface_data
        self.harmonics = self.process()
        self.extract_harmdata()
        
    def plot_spectrum(self, show=False):
        # plot spectrum
        clm = pysh.SHCoeffs.from_array(self.harmonics)
        #clm.plot_spectrum(unit='per_l', xscale='log', yscale='log', show=False)
        clm.plot_spectrum(show=False)
        if show:
            plt.show()
        # grid = clm.expand()
        # fig, ax = grid.plot(show=False)
        # plt.show()

    def extract_harmdata(self):
        # save harmonics to csv file
        # collects amplitude power real, imaginary, phase order and harmonic pairs
        harm = self.harmonics
        harmdata = pd.DataFrame(columns=['degree', 'order', 'value'])
        for degree in range(len(harm[0])):
            for order in range(degree + 1):
                harmdata.loc[len(harmdata)] = pd.Series({'degree': int(degree), 'order': int(order),'value': harm[0][degree, order]})


            for order in range(1, degree + 1):
                harmdata.loc[len(harmdata)] = pd.Series({'degree': int(degree), 'order': -int(order),'value': harm[1][degree, order]})


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


    def compute_power_spectrum(self, norm=False, filename=None, directory=None):
        stat = self.harmdata.groupby(['degree']).sum().reset_index()
        if norm:
            maxline = stat[stat['degree'] == 0].iloc[0]
            for col in stat.columns:
                if col != 'degree':
                    stat.loc[:, col] = stat[col] / maxline[col]
        stat['amplitude'] = np.sqrt(stat['power'])
        stat['harmonic'] = stat['degree']
        self.frequency_spectrum = stat
        if filename is not None and directory is not None:
            # create directory
            if not os.path.exists(directory):
                os.makedirs(directory)
            # generate a name for the frequency spectrum
            self.frequency_spectrum['Name'] = filename.split('.')[0]
            self.frequency_spectrum.to_csv(os.path.join(directory, filename), sep='\t', index=False)

    # def plot_spectrum(self, value='amplitude', title=None, cutoff=None, logscale=False, show=True, **kwargs):
    #     norm = kwargs.pop('norm', False)
    #     stat = self.harmdata
    #     if norm:
    #         stat.loc[:, value] = np.array(stat[value]) / stat[value].iloc[0]
    #     if cutoff is not None:
    #         stat = stat[stat.degree < cutoff]
    #     if logscale:
    #         stat.loc[:, value] = np.log(stat[value])
    #     hm = stat.pivot(columns='degree', index='order', values=value)
    #     plt.clf()
    #     plt.figure(figsize=(6, 5))
    #     pl = sns.heatmap(hm, **kwargs)
    #     if title is None:
    #         if self.name is not None:
    #             title = self.name + '; value = ' + value
    #         else:
    #             title = 'value = ' + value
    #     plt.title(title)
    #     if show:
    #         plt.show()
    #     return pl.figure
    def plot_spectrum2(self, show=True, **kwargs):
        # plot spectrum
        clm = pysh.SHCoeffs.from_array(self.harmonics)
        #clm.plot_spectrum(unit='per_l', xscale='log', yscale='log', show=False)
        clm.plot_spectrum2d(show=False, **kwargs)
        if show:
            plt.show()

    def compute_inverse_surface(self, lmax):
        # compute the array of harmonics
        # harmarray = self.convert_harmdata_to_harmarray()
        # compute inverse surface from harmonics
        # R = pysh.expand.MakeGridDHC(self.harmonics, lmax_calc=lmax, sampling=self.sampling).real
        R = pysh.SHCoeffs.from_array(self.harmonics).expand(lmax=lmax).data.real
        R = R[:-1,:-1] # remove the last two rows and columns
        return R
    
    
    def convert_harmdata_to_harmarray(self):
        """
        Convert the spectrum from the table form to pyshtools format.
        """
        harmdata = self.harmdata
        size = len(harmdata['degree'].unique())
        harm = np.zeros([2, size, size], dtype=complex)
        for degree in range(len(harm[0])):
            for order in range(degree + 1):
                line = harmdata[(harmdata['degree'] == degree) & (harmdata['order'] == order)].iloc[0]
                harm[0][degree, order] = line['real'] + 1j * line['imag']

            for order in range(1, degree + 1):
                line = harmdata[(harmdata['degree'] == degree) & (harmdata['order'] == -order)].iloc[0]
                harm[1][degree, order] = line['real'] + 1j * line['imag']

        self.harmarray = harm
        return harm
    
    def compute_features(self , cutoff=None, static_features='amplitude', rotation_invariant=True):
        stat = self.harmdata
        if cutoff is None:
            cutoff = np.max(stat.degree)
        if static_features == 'real_imag':
            static_features = ['real', 'imag']
        else:
            static_features = [static_features]
        if rotation_invariant:
            self.compute_power_spectrum(norm=False)
            stat = self.frequency_spectrum

        features = []
        for value in static_features:
            features = features + list(stat[value][stat.degree < cutoff + 1])
        return np.array(features)
