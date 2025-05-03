# this script cleans up files in the ADNI BIDS dataset

import os
import sys
import multiprocessing
import tqdm
import time
import argparse

# Argument parser
parser = argparse.ArgumentParser(description='Brain extraction using BET')
parser.add_argument('dataset_path', type=str, help='Dataset path')
parser.add_argument('-p', '--processes', type=int, default=1, help='Number of processes')
parser.add_argument('-t', '--target', help='ending of file to delete', default='_crop.nii.gz',
                    choices=['_crop.nii.gz', '_brain.nii.gz', '_mask.nii.gz', '_reoriented.nii.gz',
                             '_corrected.nii.gz', 'log', 'qc', 'pred_process', '_seg.nii.gz',
                             '_hipp_pred_bin.nii.gz'])
parser.add_argument('-s', '--sessions', action='store_true', help='Check sessions')
args = parser.parse_args()

dataset_path = args.dataset_path
processes = args.processes
target = args.target
is_find_sessions = args.sessions
print(' searching for ', target, ' in the path: ', dataset_path)
if processes == 1:
    print('Running in single process')
else:
    print('Running in parallel with', processes, 'processes')
print('Searching for sessions:', is_find_sessions)
print('press `y` to confirm, the files cannot be recovered')
confirm = input()
if confirm != 'y':
    print('Exiting')
    sys.exit(0)
else:
    print('Confirmed')


# Ensure the dataset path is a directory
if not os.path.isdir(dataset_path):
    print("Error: Directory is missing", dataset_path, "is not a valid directory")
    sys.exit(0)

subs = [f for f in os.listdir(dataset_path) if f.startswith('sub')]

if len(subs) == 0:
    print(f"Directory {dataset_path} has no sub-x directories containing data")
    print("Done")
    sys.exit(0)

# Function to get MRI file
def get_mri(sub: str, suffix: str):
    files = os.listdir(os.path.join(dataset_path, sub, 'anat'))
    _mri_file = [f for f in files if f.endswith(suffix)]
    if len(_mri_file) == 0:
        print(f'file not found for {sub}')
        return None
    else:
        return os.path.join(dataset_path, sub, 'anat', _mri_file[0])

def get_mri_session(sub: str, suffix: str):
    ## look for all the files of sessions
    sessions = [f for f in os.listdir(os.path.join(dataset_path,sub)) if f.startswith('ses')]
    session_paths = [os.path.join(dataset_path,sub, session, 'anat') for session in sessions]
    mri_files = []
    for session_path in session_paths:
        files = os.listdir(session_path)
        _mri_file = [f for f in files if f.endswith(suffix)]
        if len(_mri_file) == 0:
            print(f'file not found for {session_path}')
            continue
        else:
            mri_file = str(_mri_file[0])
            mri_files.append(os.path.join(str(session_path), mri_file))
    return mri_files

# Make a list of outputs changing the suffix to _brain.nii.gz
def make_output(f_input, sub):
    var_path = os.path.dirname(f_input)
    var_path = os.path.join(var_path, sub + '_brain.nii.gz')
    return var_path
# make out for sessions
def make_output_sessions(f_input, sub):
    var_path = os.path.dirname(f_input)
    f_prefix = os.path.basename(f_input).rsplit('_', 1)[0]
    var_path = os.path.join(var_path, f_prefix + '_brain.nii.gz')
    return var_path

# Make a list of inputs
if is_find_sessions:
    files_input = []
    for sub in subs:
        mri = get_mri_session(sub, suffix=target)
        files_input += mri
else:
    # files_input = [get_mri(sub, suffix=target) for sub in subs]
    files_input = []
    for sub in subs:
        mri = get_mri(sub, suffix=target)
        if mri is not None:
            files_input.append(mri)
if len(files_input) == 0:
    print('No files found')
    sys.exit(0)
print('one file input', files_input[0])
# first get the ROI of head then extract the brain

# Make a list of commands for brain extraction
commands = []
for f_in in files_input:
    if os.path.exists(f_in) and f_in.endswith(target) and os.path.isfile(f_in):
        # this is a command from FSL bet extraction
        commands.append(['rm', f_in])
    elif os.path.exists(f_in) and f_in.endswith(target) and os.path.isdir(f_in):
        commands.append(['rm', '-r', f_in])
    elif not os.path.exists(f_in):
        print(f'file {f_in} does not exist')

# Function to execute command
def execute_command(*c):
    print('command', c)
    c = list(c)
    start_time = time.time()
    f_in = c[-1]
    if os.path.exists(f_in):
        os.system(' '.join(c))
        print(f'file is deleted {f_in}')
    else:
        print(f"{f_in} not exists, skipping")
    print("--- %s seconds ---" % (time.time() - start_time))
# print one command and ask for confirmation
if len(commands) == 0:
    print('No files found')
    sys.exit(0)
print('one command', commands[0])
print('press `y` to confirm, the files cannot be recovered')
confirm = input()
if confirm != 'y':
    print('Exiting')
    sys.exit(0)
# Run the commands
if processes == 1:
    for c in tqdm.tqdm(commands, desc=f'deleting {target}', total=len(commands)):
        execute_command(*c)
else:
    with multiprocessing.Pool(processes=processes) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(execute_command, commands), desc=f'deleting {target}', total=len(commands)):
            pass

