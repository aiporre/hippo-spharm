import numpy as np
import pyshtools as pysh

# Define the parameters of the ellipsoid
a = 6.371e6  # semi-major axis
b = 6.356e6  # semi-minor axis
c = 6.378e6  # polar radius

# Generate the ellipsoid grid
grid = pysh.ellipsoid(a=a, b=b, c=c, nlat=91, nlon=181, degrees=True)

# Compute the spherical harmonics coefficients
clm, slm = pysh.SHExpandLSQ(grid, lmax=50)

# Print the first few coefficients
print('C_00:', clm[0, 0])
print('C_20:', clm[2, 0])
print('S_21:', slm[2, 1])

# Compute the power spectrum of the coefficients
spectrum = pysh.Spectrum(clm, slm)

# Plot the power spectrum
fig, ax = spectrum.plot_spectrum(unit='per_l', show=True)
