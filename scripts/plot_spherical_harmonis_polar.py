import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm

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


# plt.show()
N = 4
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
