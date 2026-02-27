import argparse
import os
import sys

import nibabel as nib

TARGET = ("L", "P", "I")


def list_subs(path: str):
    return [f for f in os.listdir(path) if f.startswith("sub")]


def _pick_first_matching_file(dir_path: str, suffix: str) -> str:
    files = os.listdir(dir_path)
    matches = [f for f in files if f.endswith(suffix)]
    if not matches:
        raise FileNotFoundError(f"No matching file ending with `{suffix}` in `{dir_path}`")
    return os.path.join(dir_path, matches[0])


def get_mri(dataset_path: str, sub: str, brain_extraction: bool) -> str:
    anat_dir = os.path.join(dataset_path, sub, "anat")
    suffix = "_brain.nii.gz" if brain_extraction else "_corrected.nii.gz"
    return _pick_first_matching_file(anat_dir, suffix)


def get_mri_session(dataset_path: str, sub: str, brain_extraction: bool):
    sub_path = os.path.join(dataset_path, sub)
    if not os.path.isdir(sub_path):
        return []

    sessions = [f for f in os.listdir(sub_path) if f.startswith("ses")]
    suffix = "_brain.nii.gz" if brain_extraction else "_corrected.nii.gz"

    out: list[str] = []
    for ses in sessions:
        anat_dir = os.path.join(sub_path, ses, "anat")
        if not os.path.isdir(anat_dir):
            continue
        try:
            out.append(_pick_first_matching_file(anat_dir, suffix))
        except FileNotFoundError:
            # session exists but doesn't have the expected file
            continue
    return out


def make_output(f_input: str, sub: str, suffix: str) -> str:
    var_path = os.path.dirname(f_input)
    return os.path.join(var_path, f"{sub}_{suffix}.nii.gz")


def make_output_sessions(f_input: str, suffix: str) -> str:
    var_path = os.path.dirname(f_input)
    f_prefix = os.path.basename(f_input).rsplit("_", 1)[0]
    return os.path.join(var_path, f"{f_prefix}_{suffix}.nii.gz")


def check_orientation(sub: str, f: str, is_search_sessions: bool) -> bool:
    img = nib.load(f)
    orig_axcodes = nib.aff2axcodes(img.affine)
    print("current orientation", orig_axcodes, "file:", f)

    if orig_axcodes == TARGET:
        print("Already in target orientation")
        return False  # unchanged

    # Orientation transform: current -> target
    curr_ornt = nib.orientations.axcodes2ornt(orig_axcodes)
    target_ornt = nib.orientations.axcodes2ornt(TARGET)
    ornt_trans = nib.orientations.ornt_transform(curr_ornt, target_ornt)

    data_reoriented = nib.orientations.apply_orientation(img.get_fdata(), ornt_trans)

    # Update affine for the voxel reordering
    inv_aff = nib.orientations.inv_ornt_aff(ornt_trans, img.shape)
    new_affine = img.affine.dot(inv_aff)

    out_img = nib.Nifti1Image(data_reoriented, new_affine, header=img.header)

    if is_search_sessions:
        f_out = make_output_sessions(f, "reoriented")
    else:
        f_out = make_output(f, sub, "reoriented")

    nib.save(out_img, f_out)

    new_axcodes = nib.aff2axcodes(out_img.affine)
    print("new orientation", new_axcodes, "saved:", f_out)
    return new_axcodes == TARGET


def gather_pairs(dataset_path: str, is_search_sessions: bool, brain_extraction: bool):
    subs = list_subs(dataset_path)
    if not subs:
        print(f"Directory `{dataset_path}` has no `sub-*` directories containing data")
        sys.exit(0)

    pairs: list[tuple[str, str]] = []
    if is_search_sessions:
        for sub in subs:
            for f in get_mri_session(dataset_path, sub, brain_extraction):
                pairs.append((sub, f))
    else:
        for sub in subs:
            pairs.append((sub, get_mri(dataset_path, sub, brain_extraction)))

    if not pairs:
        print("No input files found (check dataset structure / suffix expectations).")
        sys.exit(0)

    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Check and fix orientation of nifti files in the dataset")
    parser.add_argument("dataset_path", help="Path to the dataset")
    parser.add_argument("-b", "--brain", action="store_true", help="Use brain extracted images")
    parser.add_argument("-s", "--sessions", action="store_true", help="Check sessions")
    args = parser.parse_args()

    pairs = gather_pairs(args.dataset_path, args.sessions, args.brain)

    summary: dict[str, str] = {}
    for sub, f in pairs:
        try:
            fixed = check_orientation(sub, f, args.sessions)
        except Exception as e:
            print(f"Error processing `{f}`: {e}")
            summary[os.path.basename(f)] = f"error: {e}"
            continue

        if fixed:
            summary[os.path.basename(f)] = "was fixed"
        else:
            summary[os.path.basename(f)] = "already correct"

    # print("Summary table")
    # for k, v in summary.items():
    #     print("__________________________________________________")
    #     print(f"{k} | {v} |")
    # print("__________________________________________________")
    # save to csv file
    import csv
    with open(os.path.join(args.dataset_path, "orientation_check_summary.csv"), "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Filename", "Status"])
        for k, v in summary.items():
            writer.writerow([k, v])



if __name__ == "__main__":
    main()