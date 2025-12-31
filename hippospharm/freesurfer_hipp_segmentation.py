import os
from nipype.interfaces.freesurfer import ReconAll
import nibabel as nb
import shutil

def free_surfer_hipp_segmentation(input_file, output_file):
    # check if input directory exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input directory {input_file} does not exist.")
    # check it is nii.gz file
    if not input_file.endswith('.nii.gz'):
        raise ValueError(f"Input file {input_file} is not a .nii.gz file.")
    input_file = os.path.abspath(input_file)
    base_name = os.path.basename(input_file).replace('.nii.gz', '')
    # variables for freesurfer
    subject_id = f"FS_{base_name}"
    # make subject_id alfanumeric
    subject_id = ''.join(e for e in subject_id if e.isalnum() or e == '_')
    SUBJECTS_DIR = os.path.dirname(input_file)
    print(f"Using SUBJECTS_DIR: {SUBJECTS_DIR}")
    os.environ['SUBJECTS_DIR'] = SUBJECTS_DIR
    if os.path.exists(os.path.join(SUBJECTS_DIR, subject_id)):
        # remove existing directory
        shutil.rmtree(os.path.join(SUBJECTS_DIR, subject_id))
    # run freesurfer recon-all with hippocampal segmentation
    recon = ReconAll()
    recon.inputs.subject_id = subject_id
    recon.inputs.directive = 'all'
    recon.inputs.T1_files = input_file
    result = recon.run()
    # read and compose 1 2 hippocampal segmentation
    fs_mask_path = os.path.join(SUBJECTS_DIR, subject_id, 'mri', 'aseg.mgz')
    fs_mask = nb.load(fs_mask_path)
    fs_data = fs_mask.get_fdata()
    left_hipp = (fs_data == 17).astype(int)
    right_hipp = (fs_data == 53).astype(int)
    combined_hipp = left_hipp + right_hipp * 2
    combined_img = nb.Nifti1Image(combined_hipp, fs_mask.affine, fs_mask.header)
    nb.save(combined_img, output_file)
    print(f"Hippocampal segmentation saved to {output_file}")