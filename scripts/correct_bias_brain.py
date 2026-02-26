import os
import sys
import multiprocessing
import tqdm
from hippmapper.cli import main

from filelock import FileLock

def usage():
    print("Usage python segmentation.py <dataset with sub-dirs> [options]")
    print(" -h, --help display this message")
    print(" -s, --look for sessions")

def find_sessions(subs_list):
    # this corrections is necesary for the ADNI dataset which as more giles.
    sessions = []
    for sub in subs_list:
        session_dirs = [os.path.join(sub,f) for f in os.listdir(os.path.join(dataset_path,sub)) if f.startswith('ses')]
        # add al the sessions found 
        sessions.extend(session_dirs)
    return sessions
    

def get_mri(sub):
    files = os.listdir(os.path.join(dataset_path,sub,'anat'))
    mri_file = [f for f in files if f.endswith('_T1w.nii.gz')][0]
    return os.path.join(dataset_path,sub, 'anat', mri_file)

# parsing argumtens
if "-h" in sys.argv or "--help" in sys.argv:
    usage()
    sys.exit(0)

if len(sys.argv) < 2:
    print("Error: Directory is missing")
    usage()
    sys.exit(1)
elif len(sys.argv) == 2 or len(sys.argv) == 3:
    # assert the second argument doesn't start with -s
    if sys.argv[1].startswith('-s'):
        print("Error: Direcry is missing")
        usage()
        sys.exit(1)
elif len(sys.argv)>3:
    print('more than 3 arguments is not allowed')
    usage()
    sys.exit(1)

dataset_path = sys.argv[1]
if len(sys.argv) == 3:
    options = sys.argv[2]
    is_find_sessions = True if options.startswith('-s') else False
else:
    is_find_sessions = False

if not os.path.isdir(dataset_path):
    print("Error: Directory is missing", dataset_path, " is not a valid direcot")
    usage()
    sys.exit(0)

# look for paths to proecess
subs = [f for f in os.listdir(dataset_path) if f.startswith('sub')]
if is_find_sessions:
    print('completing the session in the subs directories')
    subs = find_sessions(subs)
print(f'processing subs : {len(subs)}')


if len(subs) == 0:
    print(f"Directory {dataset_path} has not sub-x directories containing data")
    print("Done")
    sys.exit(0)

# find the inputs as dict

print('one mri file' , get_mri(subs[0]))

# make a list of inputs

files_input = [get_mri(sub) for sub in subs]

# make a list of outputs changing a suffix -corrected.nii.gz
def make_output(f_input, sub,  suffix):
    var_path = os.path.dirname(f_input)
    var_path = os.path.join(var_path,  sub + '_' + suffix + '.nii.gz')
    return var_path
if is_find_sessions:
    files_corrected = [make_output(f_in, sub.replace('/', '_'), 'corrected') for f_in, sub in zip(files_input, subs)]
else:
    files_corrected = [make_output(f_in, sub, 'corrected') for f_in, sub in zip(files_input, subs)]

print('first file input,', files_input[0])
print('first file corrected,', files_corrected[0])

# make a list of commands for bias correction
commands = []
for f_in, f_out in zip(files_input, files_corrected):
    if not os.path.exists(f_out):
        commands.append(['bias_corr', '-i', f_in, '-o', f_out])
# run the commands
for c in tqdm.tqdm(commands, desc='bias correction', total=len(commands)):
    f_out = c[-1]
    f_lock = f_out + '.lock'
    # create a lock file
    # check if the file exists
    if not os.path.exists(f_out) and not os.path.exists(f_lock):
        # run the command
        print('running command', c)
        with FileLock(f_lock):
            main(c)
    elif os.path.exists(f_out):
        print('output file already exists', f_out)
    elif os.path.exists(f_lock):
        print('lock file exists', f_lock)

print('You might want to extract the brain using a FLS tool, e.g. FSL BET')
