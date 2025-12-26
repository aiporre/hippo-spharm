import os
import sys
import multiprocessing
import tqdm
import time
import argparse
import nibabel as nib
import numpy as np
import pandas as pd

# --- Argument Parser ---
parser = argparse.ArgumentParser(
    description='Compute hippocampus and intracranial volumes from existing native and MNI-space masks.',
    formatter_class=argparse.RawTextHelpFormatter
)
parser.add_argument('dataset_path', type=str, help='Path to the BIDS dataset directory.')
parser.add_argument('-p', '--processes', type=int, default=multiprocessing.cpu_count() - 1,
                    help='Number of parallel processes to use.')
parser.add_argument('-s', '--sessions', action='store_true', help='Process individual sessions instead of subjects.')
parser.add_argument('-o', '--output', type=str, default='volume_statistics.csv', help='Name for the output CSV file.')


# --- Helper Functions ---

def find_sessions(subs_list, base_path):
    """Find all session directories within a list of subject directories."""
    sessions = []
    for sub in subs_list:
        try:
            session_dirs = [os.path.join(sub, f) for f in os.listdir(os.path.join(base_path, sub)) if
                            f.startswith('ses-')]
            if session_dirs:
                sessions.extend(session_dirs)
            else:
                # If a subject has no session folders, process the subject itself
                sessions.append(sub)
        except FileNotFoundError:
            print(f"Warning: Could not find directory for subject {sub}. Skipping.")
    return sessions


def get_mask_path(subject_id, base_path, mask_suffix):
    """
    Find the path to a specific mask for a given subject or session.
    Example: subject_id='sub-001/ses-01', mask_suffix='_seg.nii.gz'
    -> looks for .../sub-001/ses-01/anat/sub-001_ses-01_seg.nii.gz
    """
    try:
        # Create a flattened ID for the filename, e.g., 'sub-001_ses-01'
        flat_id = subject_id.replace(os.path.sep, '_')

        # Construct the expected path and filename
        anat_path = os.path.join(base_path, subject_id, 'anat')
        mask_filename = f"{flat_id}{mask_suffix}"
        full_path = os.path.join(anat_path, mask_filename)

        if os.path.exists(full_path):
            return full_path
    except FileNotFoundError:
        pass
    # Return None if not found
    return None


def compute_subject_volumes(subject_id_and_path):
    """
    Computes all volumes for a single subject using their segmentation and brain masks.
    """
    subject_id, base_path = subject_id_and_path

    # Find paths for all four potential files
    native_seg_path = get_mask_path(subject_id, base_path, '_seg.nii.gz')
    native_brain_mask_path = get_mask_path(subject_id, base_path, '_brain_mask.nii.gz')
    mni_brain_path = get_mask_path(subject_id, base_path, '_mni.nii.gz')
    mni_seg_path = get_mask_path(subject_id, base_path, '_mni_seg.nii.gz')

    # Initialize all results to 0
    results = {
        'subject_id': subject_id,
        'left_hipp_voxels': 0, 'left_hipp_mm3': 0.0,
        'right_hipp_voxels': 0, 'right_hipp_mm3': 0.0,
        'icv_voxels': 0, 'icv_mm3': 0.0,
        'mni_left_hipp_voxels': 0, 'mni_left_hipp_mm3': 0.0,
        'mni_right_hipp_voxels': 0, 'mni_right_hipp_mm3': 0.0,
        'mni_brain_voxels': 0, 'mni_brain_mm3': 0.0,
    }

    try:
        # --- Process Native Space Hippocampus Segmentation ---
        #print('compute 1')
        if native_seg_path:
            img = nib.load(native_seg_path)
            voxel_vol = np.prod(img.header.get_zooms())
            data = img.get_fdata(dtype=float)
            #print(data.shape)
            results['left_hipp_voxels'] = np.sum(data == 2)
            results['left_hipp_mm3'] = results['left_hipp_voxels'] * voxel_vol
            results['right_hipp_voxels'] = np.sum(data == 1)
            results['right_hipp_mm3'] = results['right_hipp_voxels'] * voxel_vol
        else:
            print(f"Warning: Native seg mask not found for {subject_id}")
            pass
        #print('compute ICV') 
        # --- Process Native Space Brain Mask (ICV) ---
        if native_brain_mask_path:
            img = nib.load(native_brain_mask_path)
            voxel_vol = np.prod(img.header.get_zooms())
            data = img.get_fdata(dtype=float)
            #print(data.shape)
            results['icv_voxels'] = np.count_nonzero(data)
            results['icv_mm3'] = results['icv_voxels'] * voxel_vol
        else:
            print(f"Warning: Native brain mask not found for {subject_id}")
            pass

        #print('compute mni vol space') 
        # --- Process MNI Space Brain Volume ---
        if mni_brain_path:
            img = nib.load(mni_brain_path)
            voxel_vol = np.prod(img.header.get_zooms())
            data = img.get_fdata(dtype=float)
            #print(data.shape)
            results['mni_brain_voxels'] = np.count_nonzero(data)
            results['mni_brain_mm3'] = results['mni_brain_voxels'] * voxel_vol
        else:
            print(f"Warning: MNI brain file not found for {subject_id}")
            pass

        #print('compute mni hip space') 
        # --- Process MNI Space Hippocampus Segmentation ---
        if mni_seg_path:
            img = nib.load(mni_seg_path)
            voxel_vol = np.prod(img.header.get_zooms())
            data = img.get_fdata(dtype=float)
            #print(data.shape)
            results['mni_left_hipp_voxels'] = np.sum(data == 2)
            results['mni_left_hipp_mm3'] = results['mni_left_hipp_voxels'] * voxel_vol
            results['mni_right_hipp_voxels'] = np.sum(data == 1)
            results['mni_right_hipp_mm3'] = results['mni_right_hipp_voxels'] * voxel_vol
        else:
            print(f"Warning: MNI seg mask not found for {subject_id}")
            pass

        return results

    except Exception as e:
        print(f"Error processing {subject_id}: {e}")
        return None


# --- Main Execution Block ---
if __name__ == "__main__":
    args = parser.parse_args()

    if not os.path.isdir(args.dataset_path):
        print(f"Error: Directory not found at '{args.dataset_path}'")
        sys.exit(1)

    subjects = sorted([f for f in os.listdir(args.dataset_path) if f.startswith('sub-')])

    if not subjects:
        print(f"No 'sub-*' directories found in '{args.dataset_path}'. Exiting.")
        sys.exit(0)

    if args.sessions:
        print('Processing by session...')
        subjects = find_sessions(subjects, args.dataset_path)

    print(f'Found {len(subjects)} subjects/sessions to process.')

    tasks = [(sub, args.dataset_path) for sub in subjects]
    # tasks = tasks[:10]
    all_results = []
    print(f"Starting volume computation with {args.processes} processes...")
    with multiprocessing.Pool(processes=args.processes) as pool:
        with tqdm.tqdm(total=len(subjects), desc="Computing Volumes") as pbar:
            for result in pool.imap_unordered(compute_subject_volumes, tasks):
                if result:
                    all_results.append(result)
                pbar.update()
    print(all_results)
    if not all_results:
        print("\nNo volumes were computed. Please check for mask files and potential warnings.")
        sys.exit(0)

    df = pd.DataFrame(all_results)
    # Reorder columns for better readability
    column_order = [
        'subject_id',
        'left_hipp_voxels', 'left_hipp_mm3',
        'right_hipp_voxels', 'right_hipp_mm3',
        'icv_voxels', 'icv_mm3',
        'mni_left_hipp_voxels', 'mni_left_hipp_mm3',
        'mni_right_hipp_voxels', 'mni_right_hipp_mm3',
        'mni_brain_voxels', 'mni_brain_mm3'
    ]
    df = df[column_order]
    df = df.sort_values(by='subject_id').reset_index(drop=True)

    output_csv_path = os.path.join(args.dataset_path, args.output)
    df.to_csv(output_csv_path, index=False)

    print("\nProcessing complete.")
    print(f"Volume statistics saved to: {output_csv_path}")
    print("\nFirst 5 rows of the output table:")
    print(df.head().to_string())
