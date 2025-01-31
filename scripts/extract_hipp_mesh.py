# this script creates a cvs file with the features of the harmonics for each image
# the features are computed in each hippocampus right and left.

import os
from hippospharm.segmentation import BrainImage, Mesh
import multiprocessing
import tqdm
import sys
# get argument datapath from sys.argv
# use sys.argv[1] or a Environment variable called DATAPATH
datapath = os.environ.get('DATAPATH')
if datapath is None:
    if len(sys.argv) < 2:
        raise ValueError('Usage: python extract_features.py <path to data> or create an environment variable DATAPATH')
    datapath = sys.argv[1]
print(' Processing files in ', datapath)
subs = [f for f in os.listdir(datapath) if f.startswith('sub')]
# list of subjects files fond in datapath
print('Found ', len(subs), ' subjects')
# find all corrected files
files_corrected = [os.path.join(datapath, sub, 'anat', f) for sub in subs for f in os.listdir(os.path.join(datapath, sub, 'anat')) if f.endswith('corrected.nii.gz')]
# find all segmentation files
files_hipp = [os.path.join(datapath, sub, 'anat', f) for sub in subs for f in os.listdir(os.path.join(datapath, sub, 'anat')) if f.endswith('seg.nii.gz')]
print('Found ', len(files_hipp), ' hippocampus segmentation files')
for i, f in enumerate(files_hipp):
    print(f'{i} : {f}')
print('--------------------')
print('processing....')
# create a list of BrainImages

# brain_images = [BrainImage(filename, mask_file=mask_file) for filename, mask_file in tqdm.tqdm(zip(files_corrected, files_hipp), desc='loading images', total=len(files_corrected))]
# create a folder called models in datapath
models_path = os.path.join(datapath, 'models')
if not os.path.exists(models_path):
    os.makedirs(models_path)
    print('Created models folder')
else:
    print('models folder already exists models will be overwritten')



for filename, mask_file, sub in tqdm.tqdm(zip(files_corrected, files_hipp, subs), desc='loading images', total=len(files_corrected)):
    print('---->> processing: ', filename)
    brain_image = BrainImage(filename, mask_file=mask_file)
    print('brain_image', filename)
    print('mask fiel', mask_file)
    # create the model name file in models folder with sub-XX_hip.obj
    model_name_prefix = os.path.join(models_path, sub + '_hip')
    spacing = brain_image.get_spacing()
    # create a list of right hippocampus
    right_hipp = brain_image.get_hippocampus('right')
    # get features for each surface printing a progress bar with tqdm
    N = 500 
    V_r, F_r = right_hipp.get_isosurface(value=0.5, presample=1, show=False, method='marching_cubes', N=500, spacing=spacing, as_surface=False)
    surface = Mesh(V_r, F_r)
    surface.save(model_name_prefix + '_right.obj')
    # create a list of left hippocampus
    left_hipp = brain_image.get_hippocampus('left')
    V_l, F_l = left_hipp.get_isosurface(value=0.5, presample=1, show=False, method='marching_cubes', N=500, spacing=spacing, as_surface=False)
    surface = Mesh(V_l, F_l)
    surface.save(model_name_prefix + '_left.obj')


