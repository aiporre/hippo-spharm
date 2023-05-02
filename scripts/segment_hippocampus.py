import os

import tqdm
from hippmapper.cli import main

subs = [f for f in os.listdir('./') if f.startswith('sub')]
# find the inputs as dict
def get_mri(sub):
    files = os.listdir(os.path.join(sub,'anat'))
    mri_file = [f for f in files if f.endswith('nii.gz')][0]
    return os.path.join(sub, 'anat', mri_file)

print('one mri file' , get_mri(subs[0]))

# make a list of inputs
files_input = [get_mri(sub) for sub in subs]
# make a list of outputs changing a suffix -corrected.nii.gz
def make_output(f_input, sub,  suffix):
    return os.path.join(os.path.dirname(f_input), sub + '_' + suffix + '.nii.gz')
files_corrected = [make_output(f_in, sub, 'corrected') for f_in, sub in zip(files_input, subs)]
files_hipp = [make_output(f_in, sub, 'hipp') for f_in, sub in zip(files_input, subs)]

# make a list of commands for bias correction
commands = []
for f_in, f_out in zip(files_input, files_corrected):
    if not os.path.exists(f_out):hipp segmentation
        commands.append(['bias_corr', '-i', f_in, '-o', f_out])
# run the commands
for c in tqdm.tqdm(commands, desc='bias correction', total=len(commands)):
    main(c)

# make a list of commands for hippocampus segmentation
commands = []
for f_in, f_out in zip(files_corrected, files_hipp):
    if not os.path.exists(f_out):
        commands.append(['seg_hipp', '-t1', f_in, '-o', f_out])

# run the commands
for c in tqdm.tqdm(commands, desc='hipp segmentation', total=len(commands)):
    main(c)
# running the commnands apply each images?
# args = ['--help']
# main(args)
