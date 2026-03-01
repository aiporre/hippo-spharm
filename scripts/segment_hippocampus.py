import os
import sys
import multiprocessing
import tqdm
import time
# import subprocess
import argparse
from filelock import FileLock
from random import shuffle
from hippospharm.freesurfer_hipp_segmentation import  free_surfer_hipp_segmentation
# arg
parser = argparse.ArgumentParser(description='hipp segmentation')
parser.add_argument('dataset_path', type=str, help='dataset path')
parser.add_argument('-p', '--processes', type=int, default=1, help='number of processes')
parser.add_argument('-t', '--target', type=str, default="brain",
                    help='target image to extact: brain, corrected, reoriented, mni')
parser.add_argument('--brain_target', type=str, default=None, help="if target brain you need to tell which corrected_brain or reoriented_brain to use. If plain is used it will look for _brain.nii.gz without suffix")
parser.add_argument('-s', '--sessions', action='store_true', help='Check sessions')
parser.add_argument("--tool", type=str, default="hippmapper", help="tool to use for hippocampus segmentation",
                    choices=["hippmapper", "freesurfer"])
args = parser.parse_args()

dataset_path = args.dataset_path
PROCESSES = max(1, multiprocessing.cpu_count() - 3)
processes = args.processes
is_find_sessions = args.sessions
target_type = args.target
if target_type == 'brain':
    # you need to tell me which brain reoriented or corrected _brain.
    if args.brain_target is None:
        print("Error: if target is brain you need to tell which brain to use with --brain_target corrected or reoriented")
        sys.exit(0)
    if args.brain_target not in ['corrected', 'reoriented', 'plain']:
        print("Error: brain_target must be either corrected or reoriented or plain (no suffix) check the help command -h")
        sys.exit(0)

tool = args.tool
# check if the target type is in the list
if target_type not in ['brain', 'corrected', 'reoriented', 'mni', 't1w']:
    print('target_type is not in the list check the help command -h')
    print('exit 0')
    sys.exit(0)

if processes == -1:
    processes = PROCESSES
    print('Using PROCESSES =', processes, ' out of ', multiprocessing.cpu_count(), ' cores available')
else:
    print('Using processes =', processes, ' out of ', multiprocessing.cpu_count(), ' cores available')

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
    files = os.listdir(os.path.join(dataset_path, sub, 'anat'))
    if target_type == 'brain':
        if args.brain_target == 'corrected':
            mri_candidates = [f for f in files if f.endswith('_corrected_brain.nii.gz')]
        elif args.brain_target == 'reoriented':
            mri_candidates = [f for f in files if f.endswith('_reoriented_brain.nii.gz')]
        elif args.brain_target == 'plain':
            mri_candidates = [f for f in files if f.endswith('_brain.nii.gz')]
        else:
            raise Exception(f'Unknown brain target type: {args.brain_target}')
    elif target_type == 'reoriented':
        mri_candidates = [f for f in files if f.endswith('_reoriented.nii.gz')]
    elif target_type == 'corrected':
        mri_candidates = [f for f in files if f.endswith('_corrected.nii.gz')]
    elif target_type == 'mni':
        mri_candidates = [f for f in files if f.endswith('_mni.nii.gz')]
    elif target_type == 't1w':
        mri_candidates = [f for f in files if f.endswith('_T1w.nii.gz')]
    else:
        raise Exception(f'Unknown target type: {target_type}')
    if not mri_candidates:
        print(f"No {target_type} MRI found in {os.path.join(dataset_path, sub, 'anat')}, skipping")
        return None
    if len(mri_candidates) > 1:
        print(f"Warning: multiple {target_type} MRI found in {os.path.join(dataset_path, sub, 'anat')}, using the first one")
        print('there were your options the candidates were : ')
        one = True
        for c in mri_candidates:
            if one:
                print(' - ', c, ' <-- this one will be used')
                one = False
            else:
                print(f" - {c}")
    mri_file = mri_candidates[0]
    return os.path.join(dataset_path, sub, 'anat', mri_file)


def get_mri_session(sub):
        ## look for all the files of sessions
        sessions = [f for f in os.listdir(os.path.join(dataset_path, sub)) if f.startswith('ses')]
        session_paths = [os.path.join(dataset_path, sub, session, 'anat') for session in sessions]
        mri_files = []
        for session_path in session_paths:
            if not os.path.isdir(session_path):
                continue
            files = os.listdir(session_path)
            if target_type == 'brain':
                if args.brain_target == 'corrected':
                    mri_candidates = [f for f in files if f.endswith('_corrected_brain.nii.gz')]
                elif args.brain_target == 'reoriented':
                    mri_candidates = [f for f in files if f.endswith('_reoriented_brain.nii.gz')]
                elif args.brain_target == 'plain':
                    mri_candidates = [f for f in files if f.endswith('_brain.nii.gz')]
                else:
                    raise Exception(f'Unknown brain target type: {args.brain_target}')
            elif target_type == 'reoriented':
                mri_candidates = [f for f in files if f.endswith('_reoriented.nii.gz')]
            elif target_type == 'corrected':
                mri_candidates = [f for f in files if f.endswith('_corrected.nii.gz')]
            elif target_type == 'mni':
                mri_candidates = [f for f in files if f.endswith('_mni.nii.gz')]
            elif target_type == 't1w':
                mri_candidates = [f for f in files if f.endswith('_T1w.nii.gz')]
            else:
                raise Exception(f'Unknown target type: {target_type}')
            if not mri_candidates:
                print(f"No {target_type} MRI found in {session_path}, skipping")
                continue
            if len(mri_candidates) > 1:
                print(f"Warning: multiple {target_type} MRI found in {session_path}, using the first one")
                one = True
                for c in mri_candidates:
                    if one:
                        print(' - ', c, ' <-- this one will be used')
                        one = False
                    else:
                        print(f" - {c}")
            mri_files.append(os.path.join(session_path, mri_candidates[0]))

        return mri_files


# make a list of inputs
if is_find_sessions:
    files_input = []
    for sub in subs:
        files_input += get_mri_session(sub)
else:
    files_input = [get_mri(sub) for sub in subs]
# clean up the none files
files_input = [f for f in files_input if f is not None]
if len(files_input) == 0:
    print(f"No {target_type} MRI found in any of the subs, exiting")
    sys.exit(0)

print('one mri file', files_input[0])


# Build output path in the same directory: filename will be `{sub}_{suffix}.nii.gz`
def make_output(f_input, sub, suffix):
    var_path = os.path.dirname(f_input)
    print('subs', sub)
    var_path = os.path.join(var_path, sub + '_' + suffix + '.nii.gz')
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
if tool == "hippmapper":
    if target_type == 'brain':
        out_suffix = 'seg'
    else:
        out_suffix = f'{target_type}_seg'
elif tool == "freesurfer":
    if target_type == 'brain':
        out_suffix = 'fsseg'
    else:
        out_suffix = f'{target_type}_fsseg'
else:
    raise Exception(f'Unknown tool {tool} for hippocampus segmentation')

if is_find_sessions:
    files_hipp = [make_output_sessions(f_in, out_suffix) for f_in in files_input]
else:
    files_hipp = [make_output(f_in, sub, out_suffix) for f_in, sub in zip(files_input, subs)]

print('first file input,', files_input[0])
# print('first file corrected,', files_corrected[0])
print('first file files_hipp,', files_hipp[0])

# make a list of commands for hippocampus segmentation
commands = []
for f_in, f_out in zip(files_input, files_hipp):
    if not os.path.exists(f_out):
        if tool == "hippmapper":
            commands.append(['seg_hipp', '-t1', f_in, '-o', f_out])
        elif tool == "freesurfer":
            if PROCESSES>1:
                freesurfer_subject_dir = os.path.join(dataset_path, 'freesurfer_subjects')
                commands.append(['bash', freesurfer_subject_dir, './scripts/freesurfer_hipp_segmentation.sh', f_in, f_out])
            else:
                commands.append(['bash', './scripts/freesurfer_hipp_segmentation.sh', f_in, f_out])
print(' we have ', len(commands), 'commands to run')
print(' number of files was ', len(files_input))
print('shuffle commands')
shuffle(commands)
print('ready...goo....')
# run the commands
import time


def execute_command(c):
    print('command', c)
    #c = list(c)
    start_time = time.time()
    f_out = c[-1]
    f_lock = f_out.replace(".nii.gz", ".lock")
    if not os.path.exists(f_out) and not os.path.exists(f_lock):
        with open(f_lock, 'w+') as f:
            f.write('')
        if c[0] == 'seg_hipp':
            print(f'Running hippocampus segmentation with HippMapp3r for {f_out}')
            from hippmapper.cli import main
            main(c)
        elif c[0] == 'bash':
            print(f'Running hippocampus segmentation with FreeSurfer for {f_out}')
            # # run bash with subprocessA
            # command_out = subprocess.run(c,
            #     stdout=subprocess.PIPE,
            #     stderr=subprocess.PIPE,
            #     universal_newlines=True)
            # print(command_out.stdout)
            # if command_out.returncode != 0:
            #     print(f"Error running command: {' '.join(c)}")
            #     print(command_out.stderr)
            freesurfer_subject_dir = os.path.join(dataset_path, 'freesurfer_subjects')
            os.environ['SUBJECTS_DIR'] = freesurfer_subject_dir
            if not os.path.exists(freesurfer_subject_dir):
                os.makedirs(freesurfer_subject_dir)
            free_surfer_hipp_segmentation(c[2], c[3], freesurfer_subject_dir)
        else:
            print(f'Unknown command {c[0]}')
        print(f'file out generated {f_out}')
        os.remove(f_lock)
        assert not os.path.exists(f_lock), f"{f_lock} still there "
    elif os.path.exists(f_lock):
        print(f"{f_lock} is blocking the process. skipping")
    else:
        print(f"{f_out} exists, skipping")

    print("--- %s seconds ---" % (time.time() - start_time))


def execute_command_multiprocessing(c):
    print('command', c)
    # c = list(c[0])
    start_time = time.time()
    f_out = c[-1]
    f_lock = f_out.replace(".nii.gz", ".lock")
    if not os.path.exists(f_out) and not os.path.exists(f_lock):
        with open(f_lock, 'w+') as f:
            f.write('')
        if c[0] == 'seg_hipp':
            print(f'Running hippocampus segmentation with HippMapp3r for {f_out}')
            from hippmapper.cli import main
            main(c)
        elif c[0] == 'bash':
            print(f'Running hippocampus segmentation with FreeSurfer for {f_out}')
            # run bash with subprocess
            # command_out = subprocess.run(c,
            #     stdout=subprocess.PIPE,
            #     stderr=subprocess.PIPE,
            #     universal_newlines=True)
            # print(command_out.stdout)
            # if command_out.returncode != 0:
            #     print(f"Error running command: {' '.join(c)}")
            #     print(command_out.stderr)
            freesurfer_subject_dir = c[1]
            os.environ['SUBJECTS_DIR'] = freesurfer_subject_dir
            if not os.path.exists(freesurfer_subject_dir):
                os.makedirs(freesurfer_subject_dir)
            free_surfer_hipp_segmentation(c[3], c[4], freesurfer_subject_dir)
        else:
            print(f'Unknown command {c}')

        print(f'file out generated {f_out}')
        os.remove(f_lock)
        assert not os.path.exists(f_lock), f"{f_lock} still there "
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
        #execute_command(c)
        p = multiprocessing.Process(target=execute_command, args=(c,))

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
        for _ in tqdm.tqdm(pool.imap_unordered(execute_command_multiprocessing, commands), desc='hipp segmentation',
                           total=len(commands)):
            pass
