#!/usr/bin/env python3
# coding: utf-8
import argparse
import os
import sys
import nibabel as nib

try:
    # nibabel >= 2.1
    from nibabel.processing import resample_to_output
except Exception:
    resample_to_output = None

def parse_args():
    p = argparse.ArgumentParser(description="Resample NIfTI to isotropic voxel size (default = min voxel size). "
                                          "Produces <input>_isotropic.nii.gz unless -o is given.")
    p.add_argument("in_img", help="Input NIfTI (.nii or .nii.gz)")
    p.add_argument("-s", "--spacing", type=float, default=None,
                   help="Target isotropic voxel size in mm (default: min(input voxel size))")
    p.add_argument("-i", "--interp", choices=("linear", "nearest"), default="linear",
                   help="Interpolation mode: 'linear' (default) or 'nearest' (for labels)")
    p.add_argument("-o", "--out", default=None, help="Output filename (optional). If not provided, "
                                                     "input_basename_isotropic.nii.gz is used")
    p.add_argument("-f", "--force", action="store_true", help="Overwrite existing output")
    return p.parse_args()

def main():
    args = parse_args()
    in_path = args.in_img
    if not os.path.exists(in_path):
        print(f"Error: input file does not exist: {in_path}")
        sys.exit(1)

    img = nib.load(in_path)
    orig_zooms = img.header.get_zooms()[:3]
    if len(orig_zooms) < 3:
        print("Error: input image does not have 3 spatial dimensions")
        sys.exit(1)

    target_spacing = args.spacing if args.spacing is not None else min(orig_zooms)
    if target_spacing <= 0:
        print("Error: target spacing must be positive")
        sys.exit(1)

    order = 0 if args.interp == "nearest" else 3

    # prepare output path
    if args.out:
        out_path = args.out
    else:
        base = os.path.basename(in_path)
        base_noext = base
        if base_noext.endswith('.nii.gz'):
            base_noext = base_noext[:-7]
        elif base_noext.endswith('.nii'):
            base_noext = base_noext[:-4]
        out_dir = os.path.dirname(in_path) or "."
        out_path = os.path.join(out_dir, f"{base_noext}_isotropic.nii.gz")

    if os.path.exists(out_path) and not args.force:
        print(f"{out_path} already exists. Use -f to overwrite.")
        sys.exit(0)

    if resample_to_output is None:
        print("nibabel.processing.resample_to_output is not available in this nibabel installation.")
        print("Please upgrade nibabel or install a newer version.")
        sys.exit(1)

    print(f"Input: {in_path}")
    print(f"Original spacing: {orig_zooms}, original shape: {img.shape[:3]}")
    print(f"Resampling to isotropic spacing: {target_spacing} mm (interp={args.interp}) ...")

    resampled = resample_to_output(img, voxel_sizes=(target_spacing, target_spacing, target_spacing), order=order)

    print(f"Resulting spacing: {resampled.header.get_zooms()[:3]}, shape: {resampled.shape[:3]}")
    nib.save(resampled, out_path)
    print(f"Saved isotropic image: {out_path}")

if __name__ == "__main__":
    main()
