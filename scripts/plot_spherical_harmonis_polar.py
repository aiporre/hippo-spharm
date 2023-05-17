import sys

import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm
import argparse

import pyshtools as pysh
from hippospharm.segmentation import BrainImage

argparser = argparse.ArgumentParser()

argparser.add_argument('-d', '--dir', type=str, help='dataset directory')
argparser.add_argument('-s', '--subject', type=str, help='dataset file')
argparser.add_argument('-N', '--num', type=int, help='number of harmonics')
argparser.add_argument('-l', '--lmax', type=int, help='number of harmonics for reconstruction', default=10)
args = argparser.parse_args()


def get_harmonics(l, m, theta, phi):
    # Calculate the real spherical harmonics for l=2, m=1
    Ylm = sph_harm(m, l, theta, phi).real
    return Ylm


# Define the parameters
theta = np.linspace(0, 2 * np.pi, 100)
phi = np.linspace(0, np.pi, 50)

# Create a grid of theta and phi values
theta, phi = np.meshgrid(theta, phi)
# Convert to Cartesian coordinates
x = np.sin(phi) * np.cos(theta)
y = np.sin(phi) * np.sin(theta)
z = np.cos(phi)

# get surface from directory
data_dir = args.dir
subject = args.subject
# /home/doom/Documents/Phd/data/aging_caroline_landalle/ds002872-download/sub-01/anat/sub-01_hipp.nii.gz""
segmentation_file = os.path.join(data_dir, subject, 'anat', subject + '_hipp.nii.gz')
data_file = os.path.join(data_dir, subject, 'anat', subject + '_corrected.nii.gz')
# create a surface
brain_image = BrainImage(data_file, segmentation_file)
spacing = brain_image.get_spacing()
brain_image.plot_3d_mask(show=True)
sys.exit(0)

hippocampus = brain_image.get_hippocampus('right')
surface = hippocampus.get_isosurface(show=True, method='boundary', spacing=spacing, N=(500, 1000), presample=2)
plt.figure()
ax=surface.plot_scatter(show=False)
ax.set_title('hippocampus surface scatter')

# plot surface.R
fig, ax = plt.subplots()
ax.imshow(surface.R)
harmonics = surface.get_harmonics()
grid = pysh.SHCoeffs.from_array(harmonics.harmonics).expand()
grid.plot()
# grid.plot3d()
harmonics.plot_spectrum()
harmonics.plot_spectrum2(show=False)
features = harmonics.compute_features(cutoff=args.num)

plt.figure()
reconstructed_surface = surface.get_inverse_surface(lmax=None)
reconstructed_surface.plot_scatter(show=False)
plt.title('reconstructed surface')

fig, ax = plt.subplots()
ax.imshow(reconstructed_surface.R)
plt.title('reconstructed surface grid')

# plot orignal surface in spheres
plt.figure()
surface.plot_sphere(show=False, title='original surface in sphere')

plt.figure()
reconstructed_surface.plot_sphere(show=False, title='reconstructed surface in sphere')

# reconstructed_surface.plot(show=True, save='surf_org.vtk')
# surface.plot(show=True, save='surf_harm.vtk')
# plt.show()
N = args.num
center = N
plot_ncols = 2*N-1
plot_nrows = N
fig = plt.figure(figsize=(8, 8))

for i in range(0,N):
    if i == 0:
        print('coordinate', i+1, center)
        print('coef', i, 0)
        # coords = (N, K, K*(i)+center)
        coords = (i+1, center)
        print('subplot coodrinate is : ', coords)
        Ylm = get_harmonics(i, 0, theta, phi)
        # Create the polar plot
        cmap = plt.get_cmap('bwr')
        norms = plt.Normalize(-abs(Ylm).max(), abs(Ylm).max())
        colors = cmap(norms(Ylm))
        plot_index = (coords[0]-1) * plot_ncols + coords[1]
        print('argments ot add suplot: ', plot_nrows, plot_ncols, plot_index)
        ax = fig.add_subplot(plot_nrows, plot_ncols, plot_index, projection='3d')
        ax.set_title('l = ' + str(i) + ', m = ' + str(0))
        ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=colors, alpha=0.8, linewidth=0)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_axis_off()
    else:
        L = 2*i + 1
        K = int(L/2)
        for j in range(L):
            print('coordinate', (i+1, center -K + j ))
            print('coef', i, j)
            # coords = (N, K, K * (i) + center - K + j)
            coords = (i+1, center - K + j)
            print('subplot coodrinate is : ', coords)
            Ylm = get_harmonics(i, j-K, theta, phi)
            # Create the polar plot
            plot_index = (coords[0]-1) * plot_ncols + coords[1]
            print('argments ot add suplot: ', plot_nrows, plot_ncols, plot_index)
            ax = fig.add_subplot(plot_nrows, plot_ncols, plot_index, projection='3d')
            ax.set_title('l=' + str(i) + ', m=' + str(j - K))
            cmap = plt.get_cmap('bwr')
            norms = plt.Normalize(-abs(Ylm).max(), abs(Ylm).max())
            colors = cmap(norms(Ylm))
            ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=colors, alpha=0.8, linewidth=0)
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([-1, 1])
            ax.set_axis_off()
plt.show()
