import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
from hippospharm.spharm import SphereHarmonics

class Surface:
    def __init__(self, data=None, filename=None, grid=None):
        self.x = 0
        self.y = 0
        self.z = 0
        if filename is not None:
            self._readfile(filename)
        elif data is not None:
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
        self.r_vect = np.sqrt(self.x**2 + self.y**2 + self.z**2)
        # TODO: compute theta and phi

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
