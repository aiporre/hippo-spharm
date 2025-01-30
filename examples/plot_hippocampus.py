import os
import sys
import matplotlib.pyplot as plt
from hippospharm.segmentation import BrainImage

def plot_hippocampus(filepath, maskpath):
    if not os.path.isfile(filepath):
        raise ValueError(f"File {filepath} does not exist")
    if not os.path.isfile(maskpath):
        raise ValueError(f"File {maskpath} does not exist")

    brain_image = BrainImage(filepath, mask_file=maskpath)
    spacing = brain_image.get_spacing()

    # Plot right hippocampus
    right_hipp = brain_image.get_hippocampus('right')
    right_surface = right_hipp.get_isosurface(show=False, method='marching_cubes', N=500, spacing=spacing)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(right_surface.x, right_surface.y, right_surface.z, triangles=right_surface.triangles, cmap='viridis', edgecolor='none')
    ax.set_title('Right Hippocampus')
    plt.show()

    # Plot left hippocampus
    left_hipp = brain_image.get_hippocampus('left')
    left_surface = left_hipp.get_isosurface(show=False, method='marching_cubes', N=500, spacing=spacing)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(left_surface.x, left_surface.y, left_surface.z, triangles=left_surface.triangles, cmap='viridis', edgecolor='none')
    ax.set_title('Left Hippocampus')
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) < 3:
        raise ValueError('Usage: python plot_hippocampus.py <path to corrected.nii.gz> <path to segmented.nii.gz file>')
    filepath = sys.argv[1]
    maskpath = sys.argv[2]
    plot_hippocampus(filepath, maskpath)