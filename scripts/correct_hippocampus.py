import os
import sys
import multiprocessing
import tqdm
from hippmapper.cli import main


def usage():
    print("Usage python segmentation.py <dataset with sub-dirs>")
    print(" -h, --help display this message")

if "-h" in sys.argv or "--help" in sys.argv:
    usage()
    sys.exit(0)

if len(sys.argv) < 2:
    print("Error: Directory is missing")
    usage()
    sys.exit(0)

dataset_path = sys.argv[1]

if not os.path.isdir(dataset_path):
    print("Error: Directory is missing", dataset_path, " is not a valid direcot")
    usage()
    sys.exit(0)


subs = [f for f in os.listdir(dataset_path) if f.startswith('sub')]

if len(subs) == 0:
    print(f"Directory {dataset_path} has not sub-x directories containing data")
    print("Done")
    sys.exit(0)

# find the inputs as dict
def get_mri(sub):
    files = os.listdir(os.path.join(dataset_path,sub,'anat'))
    mri_file = [f for f in files if f.endswith('_brain.nii.gz')][0]
    return os.path.join(dataset_path,sub, 'anat', mri_file)

print('one mri file' , get_mri(subs[0]))

# make a list of inputs
files_input = [get_mri(sub) for sub in subs]

# make a list of outputs changing a suffix -corrected.nii.gz
def make_output(f_input, sub,  suffix):
    var_path = os.path.dirname(f_input)
    var_path = os.path.join(var_path,  sub + '_' + suffix + '.nii.gz')
    return var_path
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
    main(c)

print('You might want to extract the brain using a FLS tool, e.g. FSL BET')