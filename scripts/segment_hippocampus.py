import os
import sys
import multiprocessing
import tqdm
import time
from hippmapper.cli import main
import argparse
from filelock import FileLock

# arg
parser = argparse.ArgumentParser(description='hipp segmentation')
parser.add_argument('dataset_path', type=str, help='dataset path')
parser.add_argument('-p', '--processes', type=int, default=1, help='number of processes')
parser.add_argument('-t', '--target', type=str, default="brain", help='target image to extact: brain, corrected, reoriented, mni')
parser.add_argument('-s', '--sessions', action='store_true', help='Check sessions')
args = parser.parse_args()

dataset_path = args.dataset_path
PROCESSES = max(1, multiprocessing.cpu_count()-3)
processes = args.processes
is_find_sessions = args.sessions
target_type = args.target
# check if the target type is in the list
if target_type not in ['brain', 'corrected', 'reoriented', 'mni']:
    print('target_type is not in the list check the help command -h')
    print('exit 0')
    sys.exit(0)

if processes == -1:
    processes = PROCESSES
    print('Using PROCESSES =', processes, ' out of ', multiprocessing.cpu_count(), ' cores available')

# assert last argument is a directory
if not os.path.isdir(dataset_path):
    print("Error: Directory is missing", dataset_path, " is not a valid direcot")
    sys.exit(0)

if not os.path.isdir(dataset_path):
    print("Error: Directory is missing", dataset_path, " is not a valid direcot")
    sys.exit(0)


subs = [f for f in os.listdir(dataset_path) if f.startswith('sub')]

if len(subs) == 0:
    print(f"Directory {dataset_path} has not sub-x directories containing data")
    print("Done")
    sys.exit(0)

# find the inputs as dict
def get_mri(sub):
    files = os.listdir(os.path.join(dataset_path,sub,'anat'))
    if target_type == 'brain':
        mri_file = [f for f in files if f.endswith('_brain.nii.gz')][0]
    elif target_type == 'reoriented':
        mri_file = [f for f in files if f.endswith('_reoriented.nii.gz')][0]
    elif target_type == 'corrected':
        mri_file = [f for f in files if f.endswith('_corrected.nii.gz')][0]
    elif target_type == 'mni':
        mri_file = [f for f in files if f.endswith('_mni.nii.gz')][0]
    else:
        raise Exception(f'Unknown target type: {target_type}')
    return os.path.join(dataset_path, sub, 'anat', mri_file)

def get_mri_session(sub):
    ## look for all the files of sessions
    sessions = [f for f in os.listdir(os.path.join(dataset_path,sub)) if f.startswith('ses')]
    session_paths = [os.path.join(dataset_path,sub, session, 'anat') for session in sessions]
    mri_files = []
    for session_path in session_paths:
        files = os.listdir(session_path)
        if target_type == 'brain':
            mri_file = [f for f in files if f.endswith('_brain.nii.gz')][0]
        elif target_type == 'reoriented':
            mri_file = [f for f in files if f.endswith('_reoriented.nii.gz')][0]
        elif target_type == 'corrected':
            mri_file = [f for f in files if f.endswith('_corrected.nii.gz')][0]
        elif target_type == 'mni':
            mri_file = [f for f in files if f.endswith('_mni.nii.gz')][0]
        else:
            raise Exception(f'Unknown target type: {target_type}')
        mri_files.append(os.path.join(session_path, mri_file))

    return mri_files

# make a list of inputs
if is_find_sessions:
    files_input = []
    for sub in subs:
        files_input += get_mri_session(sub)
else:
    files_input = [get_mri(sub) for sub in subs]

print('one mri file' , files_input[0])


# make a list of outputs changing a suffix -corrected.nii.gz
def make_output(f_input, sub,  suffix):
    var_path = os.path.dirname(f_input)
    print('subs', sub)
    var_path = os.path.join(var_path,  sub + '_' + suffix + '.nii.gz')
    return var_path


def make_output_sessions(f_input, suffix):
    var_path = os.path.dirname(f_input)
    f_prefix = os.path.basename(f_input).rsplit('_', 1)[0]
    var_path = os.path.join(var_path, f_prefix + f'_{suffix}.nii.gz')
    return var_path


# if brain_extraction:
#     files_corrected = [make_output(f_in, sub, 'brain') for f_in, sub in zip(files_input, subs)]
# else:
#     files_corrected = [make_output(f_in, sub, 'corrected') for f_in, sub in zip(files_input, subs)]
if is_find_sessions:
    files_hipp = [make_output_sessions(f_in, 'seg') for f_in in files_input]
else:
    files_hipp = [make_output(f_in, sub, 'seg') for f_in, sub in zip(files_input, subs)]

print('first file input,', files_input[0])
# print('first file corrected,', files_corrected[0])
print('first file files_hipp,', files_hipp[0])


# make a list of commands for hippocampus segmentation
commands = []
for f_in, f_out in zip(files_input, files_hipp):
    if not os.path.exists(f_out):
        commands.append(['seg_hipp', '-t1', f_in, '-o', f_out])
print(' we have ', len(commands), 'commands to run')
print(' number of files was ', len(files_input))

# run the commands

def execute_command(*c):
    print('command', c)
    c = list(c)
    start_time = time.time()
    f_out = c[-1]
    f_lock = f_out.replace(".nii.gz", ".lock")
    if not os.path.exists(f_out) and not os.path.exists(f_lock):
        lock = FileLock(f_lock)
        with lock:
            main(c)
        print(f'file out generated {f_out}')
    elif os.path.exists(f_lock):
        print(f"{f_lock} is blocking the process. skipping")
    else:
        print(f"{f_out} exists, skipping")

    print("--- %s seconds ---" % (time.time() - start_time))

def execute_command_multiprocessing(*c):
    print('command', c)
    c = list(c[0])
    start_time = time.time()
    f_out = c[-1]
    f_lock = f_out.replace(".nii.gz", ".lock")
    if not os.path.exists(f_out) and not os.path.exists(f_lock):
        lock = FileLock(f_lock)
        with lock:
            main(c)
        print(f'file out generated {f_out}')
    elif os.path.exists(f_lock):
        print(f"{f_lock} is blocking the process. skipping")
    else:
        print(f"{f_out} exists, skipping")
    print(f'file out generated {f_out}')
    print("--- %s seconds ---" % (time.time() - start_time))

if processes == 1:
    for c in tqdm.tqdm(commands, desc='hipp segmentation', total=len(commands)):
        # time this loop
        print('''\n
            # ------------------------> Hippocampus segmentation <------------------------
           ''')
        start_time = time.time()
        # main(c)
        # Create a multiprocessing Process object for my_function
        # execute_command(c)
        p = multiprocessing.Process(target=execute_command, args=(c))

        # Start the process
        p.start()

        # Wait for the process to complete
        p.join()

        print('Process completed')
        print("--- %s seconds ---" % (time.time() - start_time))
        # nice footer with ascii art
        print('''\n
            # ------------------------------------------------------------------------------
             ''')
else:
    # create a pool of processes=PROCESSES to run comand with arguments commands in parallel
    with multiprocessing.Pool(processes=processes) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(execute_command_multiprocessing, commands), desc='hipp segmentation', total=len(commands)):
            pass
