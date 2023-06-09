import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
from scipy.interpolate import griddata

from hippospharm.spharm import SphereHarmonics
from hippospharm.utils import transform_cartesian_to_spherical, transform_spherical_to_cartesian


def interpolate_to_grid(r, theta, phi, N):
    # compute grid size
    grid_size = N[0]
    grid_size2 = N[1]

    # make a lattice
    elevation = np.linspace(0, np.pi, grid_size, endpoint=False)
    azimuth = np.linspace(0, 2 * np.pi, grid_size2, endpoint=False)
    azimuth, elevation = np.meshgrid(azimuth, elevation)

    # make a list of shape points (theta and phi angles) and values (radius)
    values = r

    points = np.array([theta, phi]).transpose()

    # add 0 and pi to theta
    points = np.concatenate((points, np.array([[0, 0], [0, 2 * np.pi], [np.pi, 0], [np.pi, 2 * np.pi]])), axis=0)
    rmin = np.mean(r[np.where(theta == theta.min())])
    rmax = np.mean(r[np.where(theta == theta.max())])
    values = np.concatenate((values, np.array([rmin, rmin, rmax, rmax])), axis=0)

    # add shape points shifted to the left and right in the longitude dimension, to fill the edges
    points = np.concatenate((points, points - [0, 2 * np.pi], points + [0, 2 * np.pi]), axis=0)
    values = np.concatenate((values, values, values), axis=0)
    # points = np.concatenate((points, points - [0, 2 * np.pi], points + [0, 2 * np.pi], points - [np.pi,0], points + [np.pi, 0]), axis=0)
    # values = np.concatenate((values, values, values, values, values), axis=0)

    # make list of lattice points
    xi = np.asarray([[elevation[i, j], azimuth[i, j]] for i in range(len(elevation)) for j in range(len(elevation[0]))])

    # # interpolate the shape points on the lattice
    grid = griddata(points, values, xi, method='linear')
    grid = grid.reshape((grid_size, grid_size2))
    # fig = plt.figure()
    # plt.subplot(121)
    # plt.scatter(points[:,0], points[:,1], c=values, s=1)
    # plt.title('original sampling points')
    # plt.subplot(122)
    # plt.imshow(grid)
    # plt.title('interpolated')
    #
    #
    # plt.figure()
    # ax = plt.axes(projection='3d')
    # cm = plt.get_cmap('viridis')
    # norm = plt.Normalize(vmin=grid.min(), vmax=grid.max())
    # colors = cm(norm(grid.flatten()))
    # x, y, z = transform_spherical_to_cartesian(grid.flatten(), elevation.flatten(), azimuth.flatten())
    # ax.scatter(x, y, z, c=colors, s=1)
    # plt.title('interpolated points')
    #
    # plt.figure()
    # ax = plt.axes(projection='3d')
    # cm = plt.get_cmap('viridis')
    # norm = plt.Normalize(vmin=values.min(), vmax=values.max())
    # colors = cm(norm(values))
    # x, y, z = transform_spherical_to_cartesian(1, points[:,0], points[:,1])
    # ax.scatter(x, y, z, c=colors, s=1)
    # ax.set_title('original points back to sphere')
    #
    # plt.figure()
    # ax = plt.axes(projection='3d')
    # cm = plt.get_cmap('viridis')
    # norm = plt.Normalize(vmin=values.min(), vmax=values.max())
    # colors = cm(norm(values))
    # x, y, z = transform_spherical_to_cartesian(values, points[:,0], points[:,1])
    # ax.scatter(x, y, z, c=colors, s=1)
    # ax.set_title('original points from the reorgazaized data')
    #
    # plt.figure()
    # ax = plt.axes(projection='3d')
    # x, y, z = transform_spherical_to_cartesian(r, theta, phi)
    # norm = plt.Normalize(vmin=r.min(), vmax=r.max())
    # colors = cm(norm(r))
    # ax.scatter(x, y, z, c=colors, s=1)
    # ax.set_title('original points from input')
    # plt.show()
    #
    return grid, elevation, azimuth


class Surface:
    def __init__(self, data=None, filename=None, grid=None, N=None, spacing=(1,1,1)):
        self.x = 0
        self.y = 0
        self.z = 0
        self.spacing = spacing
        if filename is not None:
            self._readfile(filename)
        elif data is not None:
            # assert data is an int or a tuple (N,N) or (N,2*N)
            if isinstance(N, int):
                self.N = (N, N)
            elif isinstance(N, tuple):
                assert (len(N) == 2), "N must be a tuple of length 2 or int"
                assert N[0] == N[1] or 2*N[0] == N[1], "N must be a tuple (n,n) or (n,2n)"
                self.N = N
            else:
                raise ValueError("N must be a tuple of length 2 or int")
            self._readdata(data)
        elif grid is not None:
            self._readgrid(grid)


    def _readfile(self, filename):
        # reads file
        # for ...
        pass

    def _readdata(self, data):
        # resamples data into an spherical grid
        self.x = data[:, 0]*self.spacing[0]
        self.y = data[:, 1]*self.spacing[1]
        self.z = data[:, 2]*self.spacing[2]
        # center data
        self.center = np.array([np.mean(self.x), np.mean(self.y), np.mean(self.z)])
        self.x = self.x - np.mean(self.x)
        self.y = self.y - np.mean(self.y)
        self.z = self.z - np.mean(self.z)
        # compute spherical coordinates
        self.r_vect, self.theta_vect, self.phi_vect = transform_cartesian_to_spherical(self.x, self.y, self.z)
        # compute grid
        self._resample()


    def _resample(self):
        self.R, self.Theta, self.Phi = interpolate_to_grid(self.r_vect, self.theta_vect, self.phi_vect, self.N)
        # recompute spherical as flatten of theta and phi
        self.theta_vect = self.Theta.flatten()
        self.phi_vect = self.Phi.flatten()
        # compute cartesian coordinates
        self.X = self.R * np.sin(self.Theta) * np.cos(self.Phi)
        self.Y = self.R * np.sin(self.Theta) * np.sin(self.Phi)
        self.Z = self.R * np.cos(self.Theta)
        self.x = self.X.flatten()
        self.y = self.Y.flatten()
        self.z = self.Z.flatten()

    def _readgrid(self, grid):
        # assert that grid is NxN or Nx2*N
        assert grid.shape[0] == grid.shape[1] or 2*grid.shape[0] == grid.shape[1], f'grid must be a NxN or Nx2N array. Got {grid.shape}'
        # assert that grid is a 2D array
        assert len(grid.shape) == 2, 'grid must be a 2D array'
        self.R = grid
        N = grid.shape[0]
        N2 = grid.shape[1]
        theta = np.linspace(0, np.pi, N, endpoint=False)  # polar elevation
        phi = np.linspace(0, 2 * np.pi, N2, endpoint=False)  # azimuth
        self.Phi, self.Theta = np.meshgrid(phi, theta)

        # compute cartesian coordinates
        self.X = self.R * np.sin(self.Theta) * np.cos(self.Phi)
        self.Y = self.R * np.sin(self.Theta) * np.sin(self.Phi)
        self.Z = self.R * np.cos(self.Theta)
        self.r_vect = self.R.flatten()
        self.phi_vect = self.Phi.flatten()
        self.theta_vect = self.Theta.flatten()
        # convert x, y z from flatten
        self.x = self.X.flatten()
        self.y = self.Y.flatten()
        self.z = self.Z.flatten()

        # data is centerd in the origin
        self.center = np.array([0, 0, 0])

    def process(self):
        pass

    def save(self, output_filename):
        # save file
        pass

    def get_inverse_surface(self, lmax=None):
        # compute inverse surface
        R = self.spharm.compute_inverse_surface(lmax)
        return Surface(grid=R)


    def plot(self, ax=None, show=False, save=None):
        # plot elipsoid
        # ax = plt.axes(projection = '3d')
        # #ax.plot_trisurf(self.x, self.y, self.z, cmap='viridis', edgecolor='none')
        # #ax.scatter(self.x, self.y, self.z, c=self.z, cmap='viridis', linewidth=0.5)
        #

        if ax is None:
            ax = plt.axes(projection='3d')
        x, y, z = self.x, self.y, self.z
        # X, Y, Z = np.meshgrid(x, y, z)
        ax.plot_surface(self.X, self.Y, self.Z, cmap='viridis', edgecolor='none')
        # ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
        # ax.set_box_aspect(self.spacing)
        ax.set_box_aspect([max(x) - min(x), max(y) - min(y), max(z) - min(z)])
        # ----------------------------

        # import pyvista as pv
        # points = np.stack((self.x, self.y, self.z), axis=1)
        #
        # # Convert the point cloud to a PyVista PolyData object
        # cloud = pv.PolyData(points)
        #
        # # Visualize the point cloud
        # cloud.plot()
        #
        # # Generate a mesh from the point cloud using Delaunay triangulation
        # # Adjust the alpha parameter to control the distance under which two points are linked
        # volume = cloud.delaunay_3d(alpha=2.)
        #
        # # Extract the surface mesh from the volume
        # mesh = volume.extract_geometry()
        #
        # if save is not None:
        #     mesh.save(save)
        #
        # # Visualize the mesh
        # mesh.plot()
        if show:
            plt.show()

    def plot_scatter(self, ax=None, show=False):
        if ax is None:
            ax = plt.axes(projection='3d')
        ax.scatter(self.x, self.y, self.z, c=self.z, cmap='viridis', linewidth=0.5)
        if show:
            plt.show()
        return ax

    def plot_sphere(self, ax=None, show=False, title='Sphere'):
        if ax is None:
            ax = plt.axes(projection='3d')
        ax.set_title(title)
        cmap = plt.get_cmap('bwr')
        norms = plt.Normalize(self.R.min(), self.R.max())
        colors = cmap(norms(self.R))
        # Convert to Cartesian coordinates
        x, y, z = transform_spherical_to_cartesian(1, self.Theta, self.Phi)
        ax.plot_surface(x, y, z, facecolors=colors, linewidth=0, antialiased=False, shade=False)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.set_axis_off()
        if show:
            plt.show()
        return ax

    def get_harmonics(self, normalization_method=None):
        # compute harmonics
        self.spharm = SphereHarmonics(self.R, normalization_method=normalization_method)
        return self.spharm

class Ellipsoid(Surface):
    def __init__(self, a=1, b=1, c=3, N=20):
        super().__init__(filename=None)
        self.x = 0
        self.y = 0
        self.z = 0
        self._generate_ellipsoid(a, b, c, N)
        self.spharm = None

    def _generate_ellipsoid(self, a, b, c, N):
        """
        Generates an elipsoid
        
        Spherical cords are coordinates (radius r, inclination θ, azimuth φ),
        where r ∈ [0, ∞), θ ∈ [0, π], φ ∈ [0, 2π), by
        \begin{aligned}
            x & = r\sin \theta \,\cos \varphi ,\\
            y & = r\sin \theta \,\sin \varphi ,\\
            z & = r\cos \theta .\end{aligned}
        \end{aligment}
        :param a: x axis scale
        :param b: y axis scale
        :param c: z axis scale
        :param N: number of points in the theta phi grid N*N
        """
        phi = np.linspace(0, 2 * np.pi, N, endpoint=False)  # azimuth
        theta = np.linspace(0, np.pi, N, endpoint=False)  # polar elevation
        self.Phi, self.Theta = np.meshgrid(phi, theta)
        cos, sin = lambda x: np.cos(x), lambda x: np.sin(x)
        x_unit, y_unit, z_unit = sin(self.Theta) * cos(self.Phi), sin(self.Theta)*sin(self.Phi), cos(self.Theta)
        self.R = np.sqrt(1 / ((x_unit/a) **2 + (y_unit/b) **2 + (z_unit/c) **2))
        # flat everything to make triplets
        self.phi_vect = self.Phi.flatten()
        self.theta_vect = self.Theta.flatten()
        self.r_vect = self.R.flatten()

        # compute cartesial cordinates as well for plotting
        self.x = self.r_vect * sin(self.theta_vect) * cos(self.phi_vect)
        self.y = self.r_vect * sin(self.theta_vect) * sin(self.phi_vect)
        self.z = self.r_vect * cos(self.theta_vect)

        # comput now for the meshgrid
        self.X = self.R * sin(self.Theta) * cos(self.Phi)
        self.Y = self.R * sin(self.Theta) * sin(self.Phi)
        self.Z = self.R * cos(self.Theta)


class Sphere(Surface):
    def __init__(self, radius, N=20):
        super().__init__(filename=None)
        self.x = 0
        self.y = 0
        self.z = 0
        self._generate_sphere(radius, N)
        self.spharm = None

    def _generate_sphere(self, radius, N):
        """
        Generates a sphere

        Spherical cords are coordinates (radius r, inclination θ, azimuth φ),
        where r ∈ [0, radius), θ ∈ [0, π], φ ∈ [0, 2π), by
        \begin{aligned}
            x & = r\sin \theta \,\cos \varphi ,\\
            y & = r\sin \theta \,\sin \varphi ,\\
            z & = r\cos \theta .\end{aligned}
        \end{aligment}
        :param radius: sphere radius
        :param N: number of points in the theta phi grid N*N
        """
        phi = np.linspace(0, 2 * np.pi, N, endpoint=False)  # azimuth
        theta = np.linspace(0, np.pi, N, endpoint=False)  # polar elevation
        self.Phi, self.Theta = np.meshgrid(phi, theta)
        cos, sin = lambda x: np.cos(x), lambda x: np.sin(x)
        x_unit, y_unit, z_unit = sin(self.Theta) * cos(self.Phi), sin(self.Theta) * sin(self.Phi), cos(self.Theta)
        self.R = np.ones(self.Phi.shape) * radius

        # flat everything to make triplets
        self.phi_vect = self.Phi.flatten()
        self.theta_vect = self.Theta.flatten()
        self.r_vect = self.R.flatten()

        # compute cartesial cordinates as well for plotting
        self.x = self.r_vect * sin(self.theta_vect) * cos(self.phi_vect)
        self.y = self.r_vect * sin(self.theta_vect) * sin(self.phi_vect)
        self.z = self.r_vect * cos(self.theta_vect)

        # comput now for the meshgrid
        self.X = self.R * sin(self.Theta) * cos(self.Phi)
        self.Y = self.R * sin(self.Theta) * sin(self.Phi)
        self.Z = self.R * cos(self.Theta)

if __name__ == '__main__':
    # create elipsoid and plot
    ellipsoid = Ellipsoid()
    ellipsoid.plot()
    plt.show()
    # create sphere and plot
    sphere = Sphere(1, N=10)
    sphere.plot()
    plt.show()

    # get inverse of the elipsoid
    h = ellipsoid.get_harmonics()
    ellipsoid_inv = ellipsoid.get_inverse_surface()
    ellipsoid_inv.plot()
    plt.show()

    print('the features are: ', h.compute_features())
