# this script creates a cvs file with the features of the harmonics for each image
# the features are computed in each hippocampus right and left.
import argparse
import os

import pandas as pd

from hippospharm.segmentation import BrainImage, Mesh
import tqdm


# get argument datapath from sys.argv

# Argument parser
parser = argparse.ArgumentParser(description='Extract features from hippocampus images')
parser.add_argument('datapath', nargs='?', default=os.environ.get('DATAPATH'), help='Path to the data directory or set the environment variable DATAPATH')
parser.add_argument('-s', '--sessions', action='store_true', help='Check sessions')
parser.add_argument('-o', '--overwrite', action='store_true', help='Overwrite models')
parser.add_argument('-t', '--target', type=str, help='target image to extact: brain, corrected, reoriented, mni')

args = parser.parse_args()
datapath = args.datapath
is_find_sessions = args.sessions
target = args.target

print('program started:', args)

def get_mri_session(sub, suffix):
    ## look for all the files of sessions
    sessions = [f for f in os.listdir(os.path.join(datapath,sub)) if f.startswith('ses')]
    session_paths = [os.path.join(datapath, sub, session, 'anat') for session in sessions]
    mri_files = []
    for session_path in session_paths:
        files = os.listdir(session_path)
        mm = [f for f in files if f.endswith(f'_{suffix}.nii.gz')]
        if len(mm) == 0:
            print(f'No {suffix} file found in {session_path}')
            mri_file = None
            mri_files.append(mri_file)
        else:
            mri_file = mm[0]
            mri_files.append(os.path.join(session_path, mri_file))
    return mri_files

def get_mri(sub, suffix):
    files = os.listdir(os.path.join(datapath,sub,'anat'))
    # TODO: if zero or more than one file is found, raise an exception
    mri_file = [f for f in files if f.endswith(f'_{suffix}.nii.gz')][0]
    return os.path.join(datapath, sub, 'anat', mri_file)

print(' Processing files in ', datapath)
subs = [f for f in os.listdir(datapath) if f.startswith('sub')]
# list of subjects files fond in datapath
print('Found ', len(subs), ' subjects')
if is_find_sessions:
    files_corrected = []
    files_hipp = []
    for sub in subs:
        suffix = target
        files_corrected += get_mri_session(sub, suffix)
        if target == 'brain':
            suffix = 'seg'
        else:
            suffix = f'{target}_seg'
        files_hipp += get_mri_session(sub, suffix)
    # filter none values in pairs
    ffs = [(f1, f2) for f1, f2 in zip(files_corrected, files_hipp) if f1 is not None and f2 is not None]
    files_corrected, files_hipp = zip(*ffs)
else:
    # TODO: use get_mri for dynamic file selection, now it is hardcoded to corrected and seg
    # Use get_mri for dynamic file selection
    # find all corrected files
    suffix = target
    files_corrected = [get_mri(sub, target) for sub in subs]
    # find all segmentation files
    suffix = f"{target}_seg"
    files_hipp = [get_mri(sub, f"{target}_seg") for sub in subs]
print('Found ', len(files_hipp), ' hippocampus segmentation files')
#for i, f in enumerate(files_hipp):
#    print(f'{i} : {f}')
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
    if args.overwrite:
        print('models folder already exists, models will be overwritten')
    else:
        print('models folder already exists, models will not be overwritten')


failed_list = []
reason = []
values = zip(files_corrected, files_hipp, subs) if not is_find_sessions else zip(files_corrected, files_hipp, [1]*len(files_corrected))
# filter files that have already been processed
if not args.overwrite:
    values_filtered = []
    cnt = 0
    for filename, mask_file, sub in values:
        # create the model name file in models folder with sub-XX_hip.obj
        cnt = cnt + 1
        if is_find_sessions:
            fname= os.path.basename(filename)
            sub_session = fname.rsplit('_', 1)[0].replace("_", "-")
            model_name_prefix = os.path.join(models_path, sub_session + '_hip')
        else:
            model_name_prefix = os.path.join(models_path, sub + '_hip')
        # are both output there
        if os.path.exists(model_name_prefix + '_right.obj') and os.path.exists(model_name_prefix + '_left.obj'):
            continue
        # add the file to the list
        values_filtered.append((filename, mask_file, sub))
    print('reduced files to process from', cnt, 'to', len(values_filtered))
    values = values_filtered



for filename, mask_file, sub in tqdm.tqdm(values, desc='loading images', total=len(files_corrected)):
    print('---->> processing: ', filename)
    # check if lock file exists
    f_lock = filename.replace(".nii.gz", ".meshlock")
    if os.path.exists(f_lock):
        print('lock file exists, skipping', f_lock)
        continue
    # then create a lock file
    with open(f_lock, 'w+') as f:
        f.write('lock file')
    try:
        print('brain_image', filename)
        print('mask file', mask_file)
        # create the model name file in models folder with sub-XX_hip.obj
        if is_find_sessions:
            fname= os.path.basename(filename)
            sub_session = fname.rsplit('_', 1)[0].replace("_", "-")
            model_name_prefix = os.path.join(models_path, sub_session + '_hip')
        else:
            model_name_prefix = os.path.join(models_path, sub + '_hip')
        # EXTRACT THE HIPPOCAMPUS as mesh
        # create a BrainImage object
        brain_image = BrainImage(filename, mask_file=mask_file)
        spacing = brain_image.get_spacing()
        assert all(spacing == [1, 1, 1]), 'Spacing is not consistent'
        # create a list of right hippocampus
        right_hipp = brain_image.get_hippocampus('right')
        # get features for each surface printing a progress bar with tqdm
        N = 500
        V_r, F_r = right_hipp.get_isosurface(value=0.5, presample=1, show=False, method='marching_cubes', N=500, spacing=spacing, as_surface=False)
        surface = Mesh(V_r, F_r)
        if not os.path.exists(model_name_prefix + '_right.obj') or args.overwrite:
            surface.save(model_name_prefix + '_right.obj')
        # create a list of left hippocampus
        left_hipp = brain_image.get_hippocampus('left')
        V_l, F_l = left_hipp.get_isosurface(value=0.5, presample=1, show=False, method='marching_cubes', N=500, spacing=spacing, as_surface=False)
        surface = Mesh(V_l, F_l)
        # surface.save(model_name_prefix + '_left.obj')
        if not os.path.exists(model_name_prefix + '_left.obj') or args.overwrite:
            surface.save(model_name_prefix + '_left.obj')
    except Exception as e:
        print('Failed to process ', filename)
        print(e)
        failed_list.append(filename)
        reason.append(str(e))
    # remove the lock file
    if os.path.exists(f_lock):
        os.remove(f_lock)
        assert not os.path.exists(f_lock), f"{f_lock} still there "
    else:
        print('warining: lock file not found but program finsihed fine', f_lock)
print('Failed to process the following files:')
df = pd.DataFrame({'filename': failed_list, 'reason': reason})
# print the whole table
pd.set_option('display.max_colwidth', None)
print(df)


