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
parser.add_argument('-t', '--target', type=str, default="brain", help='target image to extact: bn, corrected, reoriented')
parser.add_argument('-s', '--sessions', action='store_true', help='Check sessions')
args = parser.parse_args()

dataset_path = args.dataset_path
processes = args.processes
target_name = args.target
is_find_sessions = args.sessions

on_brain = False
if target_name == "brain":
    print("finding the brain extracted nii.gz files")
    target_ending = "_brain.nii.gz"
    on_brain =True
elif target_name == "corrected":
    print("finding the corrected bias nii.gz files")
    target_ending = "_corrected.nii.gz"
elif target_name == "reoriented":
    print("finding the reoriented nii.gz files")
    target_ending = "_reoriented.nii.gz"
else:
    print('target_name is not in the list check the help command -h')
    print('exit 0')
    sys.exit(0)
# try to use the environment variable $MNI_DIR
# if not set, use the default path
mni_templates_dir = os.environ.get('MNI_DIR', "data/mni_templates")
if not os.path.exists(mni_templates_dir):
    print("Error: Directory if temlates MNI is missing", mni_templates_dir, "is not a valid directory")
    sys.exit(0)
if on_brain:
    mni_reference = os.path.join(mni_templates_dir, "MNI152_T1_1mm_brain.nii.gz")
else:
    mni_reference = os.path.join(mni_templates_dir, "MNI152_T1_1mm.nii.gz")

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
    mri_file = [f for f in files if f.endswith(target_ending)][0]
    return os.path.join(dataset_path, sub, 'anat', mri_file)

def get_mri_session(sub):
    ## look for all the files of sessions
    sessions = [f for f in os.listdir(os.path.join(dataset_path,sub)) if f.startswith('ses')]
    session_paths = [os.path.join(dataset_path,sub, session, 'anat') for session in sessions]
    mri_files = []
    for session_path in session_paths:
        files = os.listdir(session_path)
        mri_file = [f for f in files if f.endswith(target_ending)][0]
        mri_files.append(os.path.join(session_path, mri_file))
    return mri_files

# Make a list of outputs changing the suffix to _brain.nii.gz
def make_output(f_input, sub):
    var_path = os.path.dirname(f_input)
    var_path = os.path.join(var_path, sub + '_mni.nii.gz')
    return var_path
# make out for sessions
def make_output_sessions(f_input):
    var_path = os.path.dirname(f_input)
    f_prefix = os.path.basename(f_input).rsplit('_', 1)[0]
    var_path = os.path.join(var_path, f_prefix + '_mni.nii.gz')
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
#print('files_input', files_input)
#print('files_brain', files_brain)
for f_in, f_out in zip(files_input, files_brain):
    if not os.path.exists(f_out):
        # this is a command from FSL bet extraction
        # create a block file
        # this is a command from FSL bet extraction

        f_block = f_out.replace('_mni.nii.gz', '_mni.lock')
        commands.append([
            'command', 'mni',
            'input', f_in,
            'output', f_out,
            'reference', mni_reference,
            'block', f_block
            ])
        # delete the block
print('number of commands', len(commands))
print('number of input files', len(files_input))
print('number of output files', len(files_brain))

# Function to execute command
def _execute_command(command, f_out=None):
    print('command', command)
    c = command.split(" ")
    start_time = time.time()
    f_out = c[-1] if f_out is None else f_out
    if not os.path.exists(f_out):
        os.system(' '.join(c))
        print(f'file out generated {f_out}')
    else:
        print(f"{f_out} exists, skipping")
    print("--- %s seconds ---" % (time.time() - start_time))

# Function to execute command
def map_nmi(f_in, f_out, f_ref):
    # input_image = ants.image_read(f_in)
    # reference_image = ants.image_read(f_ref)
    # registration = ants.registration(fixed=input_image, moving=reference_image, type_of_transform='SyN')
    # write the output
    # ants.image_write(registration['warpedmovout'], f_out)
    curr_dir = os.path.dirname(f_in)
    segs_dir = os.path.join(curr_dir, "segments")
    f_in_seg = os.path.join(segs_dir, os.path.basename(f_in).replace(".nii.gz", "_synthseg.nii.gz"))
    f_reg_seg = f_ref.replace(".nii.gz","_synthseg.nii.gz")
    command_1 = f"mri_synthseg --i {f_in} --o {segs_dir} --parc --cpu"
    command_2 = f"mri_easyreg --ref {f_ref} --flo {f_in} --flo_seg {f_in_seg} --ref_seg {f_reg_seg}  --flo_reg {f_out}"
    # mri_synthseg --i brain.nii.gz --o brain_seg --parc --cpu
    print('make parcellation')
    _execute_command(command_1, segs_dir)
    print('register to MNI...')
    _execute_command(command_2, f_out)

    


def execute_command(*c):
    print('command', c)
    u = list(c)
    c ={'output': u[5], 'input': u[3], 'reference': u[7], 'block': u[9]}

    start_time = time.time()
    f_out = c['output']
    f_in = c['input']
    f_ref = c['reference']
    f_block = c['block']
    if not os.path.exists(f_out) and not os.path.exists(f_block):
        with open(f_block, 'w') as f:
            f.write('1')
        map_nmi(f_in, f_out, f_ref)
        print(f'file out generated {f_out}')
        # delete the block
        os.remove(f_block)
    else:
        print(f"{f_out} exists, skipping")
    print("--- %s seconds ---" % (time.time() - start_time))

# Run the commands
if processes == 1:
    for c in tqdm.tqdm(commands, desc='MNI projection', total=len(commands)):
        execute_command(*c)
else:
    with multiprocessing.Pool(processes=processes) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(execute_command, commands), desc='MNI projection', total=len(commands)):
            pass
