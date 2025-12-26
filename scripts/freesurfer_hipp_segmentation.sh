#!/bin/sh

# This script performs hippocampal subfield segmentation using FreeSurfer's
# segmentHA_T1.sh command. It requires a T1-weighted MRI image as input
# and outputs a hippocampal segmentation file give as second input
# Usage: ./freesurfer_hipp_segmentation.sh <T1_image>

set -euo pipefail

input_image=$1
output_hippo=$2
# validation inputs
if [ -z "$input_image" ] || [ -z "$output_hippo" ]; then
  echo "Usage: $0 <T1_image.nii.gz> <output_hippo.nii.gz>"
  exit 1
fi
# check input file exists
if [ ! -f "$input_image" ]; then
  echo "Input file $input_image does not exist"
  exit 1
fi
# check input file is nii.gz
if [[ "$input_image" != *.nii.gz ]]; then
  echo "Input file $input_image is not a .nii.gz file"
  exit 1
fi

# Get the directory and base name of the input image
input_dir=$(dirname "$input_image")
base_name=$(basename "$input_image" .nii.gz)
input_nii=$(realpath "$input_image")
# Set the output directory
export SUBJECTS_DIR="$input_dir"
# subject id is the base name of the input image
subject_id="FS_${base_name}"
# remove the special characters from subject id
subject_id=$(echo "$subject_id" | tr -cd '[:alnum:]_')
# run command reconall
recon-all -i "$input_nii" -s "$subject_id" -all
# extract hippocampus masks 17 and 53 and join into one file
fs_mask_dir="$SUBJECTS_DIR/$subject_id/mri"
aseg_file="$fs_mask_dir/aseg.mgz"
if [ ! -f "$aseg_file" ]; then
  echo "aseg.mgz file not found in $fs_mask_dir"
  exit 1
fi
cd "$fs_mask_dir"
# create left hippocampus mask
mri_binarize --i "$aseg_file" --match 17 --o "$lh_hippocampus.nii.gz" --binval 1
# create right hippocampus mask
mri_binarize --i "$aseg_file" --match 53 --o "$rh_hippocampus.nii.gz" --binval 2
# combine left and right hippocampus masks
fslmaths "$lh_hippocampus.nii.gz" -add "$rh_hippocampus.nii.gz" "hippo.mgz"
# move the output file to input directory
mri_convert "hippo.mgz" "$output_hippo"
echo "Hippocampal segmentation saved to $output_hippo"
# clean up intermediate files
rm "$lh_hippocampus.nii.gz" "$rh_hippocampus.nii.gz" "hippo.mgz"
echo "Done"

