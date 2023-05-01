import numpy as np
import SimpleITK as sitk
from matplotlib import pyplot as plt
import plotly.graph_objects as go

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
class BrainImage(Image):
    def __init__(self, filename, mask_file=None, mask=None):
        self.image = self._read_image(filename)
        if mask_file is not None:
            self.mask = self._read_image(mask_file)
        if mask_file is None and mask is None:
            print('segmenting hippocampus....')
            self.mask = np.ones(self.image.shape)  # TODO: write code to segment the brain cortices
            print('done')


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
    hippo.plot_XY(show=True)
    hippo.plot_3d(show=True)

    # get hippo left
    hippo = img.get_hippocampus(side='left')
    hippo.plot_XY(show=True)
    hippo.plot_3d(show=True)
