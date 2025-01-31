import os
import sys
import matplotlib.pyplot as plt
from hippospharm.segmentation import BrainImage, Mesh


def plot_hippocampus(filepath, maskpath):
    if not os.path.isfile(filepath):
        raise ValueError(f"File {filepath} does not exist")
    if not os.path.isfile(maskpath):
        raise ValueError(f"File {maskpath} does not exist")

    brain_image = BrainImage(filepath, mask_file=maskpath)
    spacing = brain_image.get_spacing()
    print(f"Spacing: {spacing}")

    # Plot right hippocampus
    right_hipp = brain_image.get_hippocampus('right')
    right_hipp.save('right_hippocampus.raw')
    vertices_r, faces_r = right_hipp.get_isosurface(value=0.5, presample=1, show=False, method='marching_cubes', N=500, spacing=spacing, as_surface=False,)
    surface = Mesh(vertices_r, faces_r)
    surface.plot(show=True)
    surface.save('right_hippocampus.obj')


    # right_hipp.plot_XY(show=True)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_trisurf(right_surface.x, right_surface.y, right_surface.z, triangles=right_surface.triangles, cmap='viridis', edgecolor='none')
    # ax.set_title('Right Hippocampus')
    # plt.show()
    #
    # # Plot left hippocampus
    # left_hipp = brain_image.get_hippocampus('left')
    # left_surface = left_hipp.get_isosurface(show=False, method='marching_cubes', N=500, spacing=spacing)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_trisurf(left_surface.x, left_surface.y, left_surface.z, triangles=left_surface.triangles, cmap='viridis', edgecolor='none')
    # ax.set_title('Left Hippocampus')
    # plt.show()

if __name__ == '__main__':
    if len(sys.argv) < 3:
        raise ValueError('Usage: python plot_hippocampus.py <path to corrected.nii.gz> <path to segmented.nii.gz file>')
    filepath = sys.argv[1]
    maskpath = sys.argv[2]
    plot_hippocampus(filepath, maskpath)