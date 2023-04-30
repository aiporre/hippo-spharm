# import numpy as np
# import pyshtools
# import matplotlib.pyplot as plt
# 
# # Define the function on the sphere
# N = 128  # DH grid resolution
# data = np.random.randn(N, N)
# 
# # Expand the function in terms of spherical harmonics
# lmax = 20  # maximum degree of the expansion
# coeffs = pyshtools.expand.SHExpandDH(data, sampling=2, lmax_calc=lmax)
# 
# # Compute the power spectrum
# spectrum = pyshtools.spectral_spectrum(coeffs)
# 
# # Plot the power spectrum
# fig, ax = plt.subplots()
# ax.plot(spectrum['degrees'], spectrum['power'])
# ax.set_xlabel('Degree l')
# ax.set_ylabel('Power')
# ax.set_xlim(0, lmax)
# ax.set_ylim(0, None)
# ax.grid()
# plt.show()
import matplotlib.pyplot as plt
import numpy as np
import pyshtools as pysh

pysh.utils.figstyle(rel_width=0.75)

# Generate a random field on the sphere
lmax = 100


degrees = np.arange(101, dtype=float)
degrees[0] = np.inf
power = degrees**(-2)

clm = pysh.SHCoeffs.from_random(power, seed=12345)
clm.plot_spectrum(unit='per_l', xscale='log', yscale='log', show=False)
plt.show()



fig, ax = clm.plot_spectrum2d(cmap_rlimits=(1.e-7, 0.1),
                              show=False)

plt.show()
clm_ortho = clm.convert(normalization='ortho',
                        csphase=-1,
                        lmax=50)
print(clm_ortho)
# compute powe prectra of two func and plot them along with the power spectra of the two functions


fig, ax = clm.plot_spectrum(legend='4$\pi$ normalized',
                            show=False)
clm_ortho.plot_spectrum(ax=ax,
                        linestyle='dashed',
                        legend='Orthonormalized')
ax.plot(clm.degrees(), power, '-k')
limits = ax.set_xlim(0, 100)

plt.show()

# plot expansion coefficients
grid = clm.expand()
grid.plot(show=False)
plt.show()