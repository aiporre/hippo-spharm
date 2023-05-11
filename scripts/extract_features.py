# this script creates a cvs file with the features of the harmonics for each image
# the features are computed in each hippocampus right and left.

import os
from hippospharm.segmentation import BrainImage
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

# init list of features
features = [] # list of np.arrays contianing the features
subject = [] # list of subject names as strings
side = [] # list of strings 'right' or 'left'
def get_features(surface):
    # get harmonics
    harmonics = surface.get_harmonics()
    # get features
    return harmonics.compute_features()

for filename, mask_file, sub in tqdm.tqdm(zip(files_corrected, files_hipp, subs), desc='loading images', total=len(files_corrected)):
    brain_image = BrainImage(filename, mask_file=mask_file)
    # create a list of right hippocampus
    right_hipp = brain_image.get_hippocampus('right')
    # get features for each surface printing a progress bar with tqdm
    surface = right_hipp.get_isosurface(show=False, method='marching_cubes', N=100)

    feat_vector = get_features(surface)
    features.append(feat_vector)
    subject.append(sub)
    side.append('right')
    # create a list of left hippocampus
    left_hipp = brain_image.get_hippocampus('left')
    print('file', filename, 'left_hipp', left_hipp)
    surface = left_hipp.get_isosurface(show=False, method='marching_cubes', N=100)
    feat_vector = get_features(surface)
    features.append(feat_vector)
    subject.append(sub)
    side.append('left')
# # create a list of right hippocampus
# right_hipp = [b.get_hippocampus('right') for b in tqdm.tqdm(brain_images, desc='extracting right hippocampus', total=len(brain_images))]
# # get features for each surface printing a progress bar with tqdm
# for surface, sub in tqdm.tqdm(zip(right_hipp, subs), desc='features from right hippocampus', total=len(right_hipp)):
#     feat_vector = get_features(surface)
#     features.append(feat_vector)
#     subject.append(sub)
#     side.append('right')
#
#
# # create a list of left hippocampus
# left_hipp = [b.get_hippocampus('left') for b in tqdm.tqdm(brain_images, desc='extracting left hippocampus', total=len(brain_images))]
# for surface, sub in tqdm.tqdm(zip(left_hipp, subs), desc='features from left hippocampus', total=len(left_hipp)):
#     feat_vector = get_features(surface)
#     features.append(feat_vector)
#     subject.append(sub)
#     side.append('left')

# create dataframe and save in csv with features in individual columns
import pandas as pd
N = len(features[0])
column_names = ['feat_' + str(i) for i in range(N)]
df = pd.DataFrame(features, columns=column_names)
df['subject'] = subject
df['side'] = side
df.to_csv('features.csv', index=False)
# print statistics of the dataframe and the size of csv file
print(df.describe())
print('size of csv file', os.path.getsize('features.csv')/1024/1024, 'MB')
