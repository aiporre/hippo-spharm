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
parser.add_argument('-r', '--reoriented', action='store_true', help='use reoriented extraction, default is bias correction')
parser.add_argument('-s', '--sessions', action='store_true', help='Check sessions')
parser.add_argument('-t', '--tool', type=str, default='bet', help='Tool to use for brain extraction. default is bet (FSL)', choices=['bet', 'hd-bet'])
args = parser.parse_args()

dataset_path = args.dataset_path
processes = args.processes
is_reoriented = args.reoriented
is_find_sessions = args.sessions
bet_tool = args.tool


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
        if is_reoriented:
            mri_file = [f for f in files if f.endswith('_reoriented.nii.gz')][0]
        else:
            mri_file = [f for f in files if f.endswith('_corrected.nii.gz')][0]
        mri_files.append(os.path.join(session_path, mri_file))

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
        files_input += get_mri_session(sub)
    files_brain = [make_output_sessions(f_in, sub) for f_in, sub in zip(files_input, subs)]
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
        commands.append(['robustfov','-i', f_in, '-r', f_crop])
        if bet_tool == 'bet':
            commands.append(['bet', f_crop, f_out, '-R', '-m', '-f', '0.3'])
        else:
            commands.append(['hd-bet', f_crop, f_out, '-R', '-m', '-f', '0.3'])

# Function to execute command
def execute_command(*c):
    print('command', c)
    c = list(c)
    start_time = time.time()
    f_out = c[-1]
    if not os.path.exists(f_out):
        os.system(' '.join(c))
        print(f'file out generated {f_out}')
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