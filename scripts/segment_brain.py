import os
import sys
import multiprocessing
import subprocess
import tqdm
import time
import argparse

from filelock import FileLock

# Argument parser
parser = argparse.ArgumentParser(description='Brain extraction using BET')
parser.add_argument('dataset_path', type=str, help='Dataset path')
parser.add_argument('-p', '--processes', type=int, default=1, help='Number of processes')
# parser.add_argument('-r', '--reoriented', action='store_true', help='use reoriented extraction, default is bias correction')
parser.add_argument('-T', '--target', type=str, default="correction", help='Target suffix for input files, default is "correction". If reoriented images are used, set this to "reoriented", if isotrpic images are used, set this to "isotropic"')
parser.add_argument('-s', '--sessions', action='store_true', help='Check sessions')
parser.add_argument('-t', '--tool', type=str, default='bet', help='Tool to use for brain extraction. default is bet (FSL)', choices=['bet', 'hd-bet'])
parser.add_argument('-c', '--crop', action='store_true', help='Do only crop,')
args = parser.parse_args()

dataset_path = args.dataset_path
processes = args.processes
#is_reoriented = args.reoriented
target = args.target
is_find_sessions = args.sessions
bet_tool = args.tool
only_crop = args.crop


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
def get_mri(sub):
    files = os.listdir(os.path.join(dataset_path, sub, 'anat'))
    mri_file = [f for f in files if f.endswith('.nii.gz')][0]
    return os.path.join(dataset_path, sub, 'anat', mri_file)

def get_mri_session(sub):
    ## look for all the files of sessions
    sessions = [f for f in os.listdir(os.path.join(dataset_path,sub)) if f.startswith('ses')]
    session_paths = [os.path.join(dataset_path,sub, session, 'anat') for session in sessions]
    mri_files = []
    for session_path in session_paths:
        files = os.listdir(session_path)
        if target == 'reoriented':
            _files = [f for f in files if f.endswith('_reoriented.nii.gz')]
        elif target == 'isotropic':
            _files = [f for f in files if f.endswith('_isotropic.nii.gz')]
        else:
            _files = [f for f in files if f.endswith('_corrected.nii.gz')]
        if len(_files) == 0:
            # skip this session if no reoriented file is found
            print(f"No reoriented file found in {session_path}, skipping")
            continue
        mri_file = _files[0]
        mri_files.append(os.path.join(session_path, mri_file))

    return mri_files

# Make a list of outputs changing the suffix to _brain.nii.gz
def make_output(f_input, sub):
    var_path = os.path.dirname(f_input)
    var_path = os.path.join(var_path, sub + '_brain.nii.gz')
    return var_path
# make out for sessionsr
def make_output_sessions(f_input):
    var_path = os.path.dirname(f_input)
    f_prefix = os.path.basename(f_input).rsplit('_', 1)[0]
    var_path = os.path.join(var_path, f_prefix + '_brain.nii.gz')
    return var_path
# Make a list of inputs
if is_find_sessions:
    files_input = []
    for sub in subs:
        files_input += get_mri_session(sub)
    files_brain = [make_output_sessions(f_in) for f_in in files_input]
    print('one file input', files_input[0])
    print('one file output', files_brain[0])
else:
    files_input = [get_mri(sub) for sub in subs]
    files_brain = [make_output(f_in, sub) for f_in, sub in zip(files_input, subs)]

# first get the ROI of head then extract the brain

# Make a list of commands for brain extraction
commands = []
for f_in, f_out in zip(files_input, files_brain):
    if not os.path.exists(f_out):
        # this is a command from FSL bet extraction
        f_crop = f_out.replace('_brain.nii.gz', '_crop.nii.gz')
        if not os.path.exists(f_crop):
            commands.append(['robustfov','-i', f_in, '-r', f_crop])
        if only_crop:
            pass
        else:
            if not os.path.exists(f_out):
                if bet_tool == 'bet':
                    commands.append(['bet', f_crop, f_out, '-R', '-m', '-f', '0.3'])
                else:
                    commands.append(['hd-bet', f_crop, f_out, '-R', '-m', '-f', '0.3'])
# stats in the commands and files
print('Number of commands', len(commands))
print('Number of files', len(files_brain))
print('Number of files input', len(files_input))
print('number of subjects', len(subs))

# Function to execute command
def execute_command(*c):
    print('command', c)
    c = list(c)
    start_time = time.time()
    command_name = c[0]
    if command_name == 'robustfov':
        f_out = c[-1]
    elif command_name == 'bet' or command_name == 'hd-bet':
        f_out = c[2]
    else:
        raise Exception('Unknown command')
    f_lock = f_out + '.lock'
    if not os.path.exists(f_out) and not os.path.exists(f_lock):
        # run the command
        with FileLock(f_lock):
            print('running command', c)
            os.system(' '.join(c))
        print(f'file out generated {f_out}')
    elif os.path.exists(f_lock):
        print('lock file exists', f_lock)
    else:
        print(f"{f_out} exists, skipping")
    print("--- %s seconds ---" % (time.time() - start_time))

# Run the commands
if processes == 1:
    for c in tqdm.tqdm(commands, desc='Brain extraction', total=len(commands)):
        execute_command(*c)
else:
    with multiprocessing.Pool(processes=processes) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(execute_command, commands), desc='Brain extraction', total=len(commands)):
            pass
