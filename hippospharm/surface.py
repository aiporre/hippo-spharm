import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d

class Surface:
    def __init__(self, filename):
        self.x = 0
        self.y = 0
        self.z = 0
        self._readfile(filename)

    def _readfile(self, filename):
        # reads file
        # for ...
        pass

    def process(self):
        pass

    def save(self, output_filename):
        # save file
        pass


class Ellipsoid:
    def __init__(self, a=1, b=1, c=3, N=20):
        self.x = 0
        self.y = 0
        self.z = 0
        self._generate_ellipsoid(a, b, c, N)

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

    def plot(self):
        # plot elipsoid
        ax = plt.axes(projection = '3d')
        #ax.plot_trisurf(self.x, self.y, self.z, cmap='viridis', edgecolor='none')
        #ax.scatter(self.x, self.y, self.z, c=self.z, cmap='viridis', linewidth=0.5)
        ax.plot_surface(self.X, self.Y, self.Z, cmap='viridis', edgecolor='none')

if __name__ == '__main__':
    # create elipsoid and plot
    ellipsoid = Ellipsoid()
    ellipsoid.plot()
    plt.show()
