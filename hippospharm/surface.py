import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
from scipy.interpolate import griddata

from hippospharm.spharm import SphereHarmonics
from hippospharm.utils import transform_cartesian_to_spherical


def interpolate_to_grid(r, theta, phi, N):
    # compute grid size
    grid_size = N[0]

    # make a lattice
    I = np.linspace(0, np.pi, grid_size, endpoint=False)
    J = np.linspace(0, 2 * np.pi, grid_size, endpoint=False)
    J, I = np.meshgrid(J, I)

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

    # make list of lattice points
    xi = np.asarray([[I[i, j], J[i, j]] for i in range(len(I)) for j in range(len(I[0]))])

    # interpolate the shape points on the lattice
    grid = griddata(points, values, xi, method='linear')
    grid = grid.reshape((grid_size, grid_size))

    return grid, J, I


class Surface:
    def __init__(self, data=None, filename=None, grid=None, N=None):
        self.x = 0
        self.y = 0
        self.z = 0
        if filename is not None:
            self._readfile(filename)
        elif data is not None:
            # assert data is an int or a tuple (N,N) or (N,2*N)
            if isinstance(N, int):
                self.N = (N, N)
            elif isinstance(N, tuple):
                assert (len(N) == 2), "N must be a tuple of length 2 or int"
                assert N[0] == N[1] or N[0] == 2*N[1], "N must be a tuple (n,n) or (n,2n)"
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
        self.x = data[:, 0]
        self.y = data[:, 1]
        self.z = data[:, 2]
        # center data
        self.center = np.array([np.mean(self.x), np.mean(self.y), np.mean(self.z)])
        self.x = self.x - np.mean(self.x)
        self.y = self.y - np.mean(self.y)
        self.z = self.z - np.mean(self.z)
        # compute spherical coordinates
        self.r_vect, self.theta_vect, self.phi_vect = transform_cartesian_to_spherical(self.x, self.y, self.z)
        # compute grid
        self.R, self.Theta, self.Phi = interpolate_to_grid(self.r_vect, self.theta_vect, self.phi_vect, self.N)
        # recompute spherical as flatten of theta and phi
        self.theta_vect = self.Theta.flatten()
        self.phi_vect = self.Phi.flatten()
        # compute cartesian coordinates
        self.x = self.R * np.sin(self.Theta) * np.cos(self.Phi)
        self.y = self.R * np.sin(self.Theta) * np.sin(self.Phi)
        self.z = self.R * np.cos(self.Theta)

    def _readgrid(self, grid):
        # assert that grid is NxN or Nx2*N
        assert grid.shape[0] == grid.shape[1] or grid.shape[0] == 2*grid.shape[1], 'grid must be a NxN or Nx2N array'
        # assert that grid is a 2D array
        assert len(grid.shape) == 2, 'grid must be a 2D array'
        self.R = grid
        N = grid.shape[0]
        phi = np.linspace(0, 2 * np.pi, N, endpoint=False)  # azimuth
        theta = np.linspace(0, np.pi, N, endpoint=False)  # polar elevation
        self.Phi, self.Theta = np.meshgrid(phi, theta)

        # compute cartesian coordinates
        self.x = self.R * np.sin(self.Theta) * np.cos(self.Phi)
        self.y = self.R * np.sin(self.Theta) * np.sin(self.Phi)
        self.z = self.R * np.cos(self.Theta)
        self.r_vect = self.R.flatten()
        self.phi_vect = self.Phi.flatten()
        self.theta_vect = self.Theta.flatten()

        # data is centerd in the origin
        self.center = np.array([0, 0, 0])

    def process(self):
        pass

    def save(self, output_filename):
        # save file
        pass
    def plot(self):
        # plot elipsoid
        ax = plt.axes(projection = '3d')
        #ax.plot_trisurf(self.x, self.y, self.z, cmap='viridis', edgecolor='none')
        #ax.scatter(self.x, self.y, self.z, c=self.z, cmap='viridis', linewidth=0.5)
        ax.plot_surface(self.X, self.Y, self.Z, cmap='viridis', edgecolor='none')

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
