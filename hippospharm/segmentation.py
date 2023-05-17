import numpy as np
import SimpleITK as sitk
from matplotlib import pyplot as plt
import plotly.graph_objects as go
from scipy.interpolate import RegularGridInterpolator
from skimage import measure
from skimage.segmentation import find_boundaries

from hippospharm.surface import Surface


class Image:
    def __init__(self, data=None, filename=None):
        self.image = 0
        if filename is not None:
            self._readfile(filename)
        elif data is not None:
            self._readdata(data)

    def _read_image(self, filename):
        # Load the image
        image_sitk = sitk.ReadImage(filename)
        # Get the image data as a numpy array
        img = sitk.GetArrayFromImage(image_sitk)
        return img

    def _readdata(self, data):
        # resamples data into an spherical grid
        self.image = data


    def plot_XY(self, ax=None, show=False):

        # Generate a 3D numpy array with random values

        # Get the index of the central XY plane
        z_index = self.image.shape[0] // 2

        # Extract the central XY plane
        img_xy_plane = self.image[z_index, :, :]
        if ax is None:
            # Create a figure with one subplot
            fig, ax1 = plt.subplots(ncols=1, figsize=(10, 5))
        else:
            ax1 = ax

        # Plot the first image in the left subplot
        ax1.imshow(img_xy_plane, cmap='gray')
        ax1.set_title('brain image xy')

        # Add x and y axis labels to both subplots
        ax1.set_xlabel('X axis')
        ax1.set_ylabel('Y axis')
        ax1.axis('off')

        # Show the plot
        if show:
            plt.show()
        return ax1

    def plot_3d_web(self, show=False):
        # create a 3D plot with plotly
        N, M, P = self.image.shape
        X, Y, Z = np.meshgrid(np.arange(N), np.arange(M), np.arange(P))
        values = self.image.flatten()
        isomin = np.min(values)-0.1
        isomax = np.max(values)+0.1

        fig = go.Figure(data=go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=values.flatten(),
            isomin=isomin,
            isomax=isomax,
            opacity=0.1,  # needs to be small to see through all surfaces
            surface_count=21,  # needs to be a large number for good volume rendering
        ))
        if show:
            fig.show()
    def plot_3d(self, show=False):
        # create plot 3d with matplotlib
        # N, M, P = self.image.shape
        # X, Y, Z = np.meshgrid(np.arange(N), np.arange(M), np.arange(P))
        # values = self.image.flatten()
        # isomin = np.min(values) - 0.1
        # isomax = np.max(values) + 0.1

        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # ax.voxels(self.image, edgecolor='k')
        # plt.show()
        # fig = plt.figure()fig = plt.figure()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # Get the coordinates of the non-zero elements in the volume mask
        x, y, z = self.image.nonzero()
        # Plot the non-zero elements as points in the 3D space
        ax.scatter(x, y, z)
        if show:
            plt.show()

    def get_isosurface(self, value=0.5, show=False, method='marching_cubes', spacing=(1, 1, 1), N=None, presample=None,
                       crop=True):
        assert method in ['marching_cubes', 'boundary'], 'method must be marching_cubes or boundary'
        volume = self.image
        # crop around the volume
        if crop:
            # Get the indices of the non-zero elements in the volume mask
            x, y, z = volume.nonzero()
            # Get the min and max values for each dimension
            x_min, x_max = x.min(), x.max()
            y_min, y_max = y.min(), y.max()
            z_min, z_max = z.min(), z.max()
            # Crop the volume around the non-zero elements
            volume = volume[x_min:x_max, y_min:y_max, z_min:z_max]

        if presample is not None:
            assert isinstance(presample, int), 'presample must be an integer'
            assert presample > 0, 'presample must be positive'
            # presample the image 3d mask to increase the number of points
            # create a 3d grid
            N, M, P = volume.shape
            # create a regular grid interpolator
            interpolator = RegularGridInterpolator((np.arange(N), np.arange(M), np.arange(P)), volume,
                                                   method='nearest', bounds_error=False, fill_value=0)
            # create a new grid with presample points
            step_n, step_m, step_p = N * presample, M * presample, P * presample
            Xp, Yp, Zp = np.meshgrid(np.linspace(0, N, step_n), np.linspace(0, M, step_m), np.linspace(0, P, step_p),
                                     indexing='ij')
            # interpolate the image
            volume = interpolator((Xp, Yp, Zp))
            volume = np.round(volume).astype(int)

        # Generate a random 3D numpy array representing a volumek
        if method == 'marching_cubes':
            assert len(np.unique(volume)) == 2, 'image must be binary'
            # Set the isovalue for the isosurface
            iso_value = value

            # Get the vertices and faces of the isosurface using marching cubes
            vertices, faces, _, _ = measure.marching_cubes(volume, iso_value, spacing=spacing)
            # put vertices in N x 3 array
            data = np.array(vertices)
            # creates a plotly figure
            if show:
                # Create a figure and an Axes3D object
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')

                # Plot the isosurface using the vertices and faces
                ax.plot_trisurf(vertices[:, 0], vertices[:, 1], faces, vertices[:, 2])

                # Set the limits for the x, y, and z axes
                # ax.set_xlim([0, volume.shape[0]])
                # ax.set_ylim([0, volume.shape[1]])
                # ax.set_zlim([0, volume.shape[2]])
                ax.set_box_aspect(spacing)

                # Set the labels for the x, y, and z axes
                ax.set_xlabel('X axis')
                ax.set_ylabel('Y axis')
                ax.set_zlabel('Z axis')

                # Show the plot
                if show:
                    plt.show()
        else:
            # check if image has only one value
            if not np.unique(volume).max() == 1 and not np.unique(volume).min() == 0 and len(np.unique(volume)) != 2:
                raise ValueError('image has more than one value')
            border = find_boundaries(volume)*volume
            l = np.unique(border)[1]
            print('---->>> label', l)
            vertices = np.array(np.where(border == l))
            for iv in range(3):
                vertices[iv] = vertices[iv] * spacing[iv]
            # make data a N x 3 array
            data = np.array(vertices.transpose())
            # plot 3d scatter
            if show:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(vertices[0], vertices[1], vertices[2])
                plt.show()

        # creates instance of surface with data
        if N is None:
            N = data.shape[0]
            print('Waring: N is None, setting N to ', N)
        surface = Surface(data=data, N=N, spacing=
                          spacing)
        return surface

class BrainImage(Image):
    def __init__(self, filename, mask_file=None, mask=None):
        self.image = self._read_image(filename)
        self.filename = filename
        if mask_file is not None:
            self.mask = self._read_image(mask_file)
        if mask_file is None and mask is None:
            print('segmenting hippocampus....')
            self.mask = np.ones(self.image.shape)  # TODO: write code to segment the brain cortices
            print('done')

    def get_spacing(self):
        temp_image = sitk.ReadImage(self.filename)
        spacing = temp_image.GetSpacing()
        return spacing

    def get_hippocampus(self, side='left'):
        '''
        Returns the hippocampus mask with a given side. If side is not specified, it returns the left hippocampus.
        Side left is all voxels with 1 as value and the left are the all the voxels with 2 as value.
        :param side:
        :return: int array with 1 for hippocampus and 0 for background
        '''
        # convert side to int
        if side == 'left':
            side = 1
        elif side == 'right':
            side = 2
        else:
            raise ValueError('side must be left or right')
        # get hippocampus mask
        data =  np.where(self.mask == side, 1, 0)
        hippo = Image(data=data)
        return hippo



    def plot_XY(self, show=True):

        # Generate a 3D numpy array with random values

        # Get the index of the central XY plane
        z_index = self.image.shape[0] // 2

        # Extract the central XY plane
        img_xy_plane = self.image[z_index, :, :]
        mask_xy_plane = self.mask[z_index, :, :]

        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))

        # Plot the first image in the left subplot
        ax1.imshow(img_xy_plane, cmap='gray')
        ax1.set_title('brain image xy')

        # Plot the second image in the right subplot
        ax2.imshow(mask_xy_plane, cmap='gray')
        ax2.set_title('mask image xy')

        # Add x and y axis labels to both subplots
        ax1.set_xlabel('X axis')
        ax1.set_ylabel('Y axis')
        ax2.set_xlabel('X axis')
        ax2.set_ylabel('Y axis')
        ax1.axis('off')
        ax2.axis('off')

        # Show the plot
        if show:
            plt.show()


if __name__ == '__main__':
    # read image in data path
    img_file = './data/segmented/HFH_001.img'
    mask_file = './data/segmented/HFH_001_Hipp_Labels.img'
    img = BrainImage(img_file, mask_file=mask_file)
    img.plot_XY()

    # get hippo right
    hippo = img.get_hippocampus(side='right')
    # hippo.plot_XY(show=True)
    # hippo.plot_3d(show=True)

    # get hippo left
    hippo = img.get_hippocampus(side='left')
    # hippo.plot_XY(show=True)
    # hippo.plot_3d(show=True)

    # get isosurface
    surface = hippo.get_isosurface(show=True, method='marching_cubes', N=100)

    # test get harmonics
    print('calculating harmonics....')
    harmonics = surface.get_harmonics()
    print('done')
    print('plotting harmonics....')
    harmonics.plot_spectrum()
    harmonics.plot_spectrum2()
    print('done')

    # compute the harmonics for the surface
    h = surface.get_harmonics()
    surface.plot()
    plt.show()
    # get the inverser surface
    surface_inverse = surface.get_inverse_surface()
    surface_inverse.plot()
    plt.show()
