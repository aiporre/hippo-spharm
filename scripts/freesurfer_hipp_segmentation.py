#!/usr/bin/env python3
"""Python equivalent of freesurfer_hipp_segmentation.sh

This provides a function `freesurfer_hipp_segmentation(input_image, output_hippo, run_cmds=True)`
that mirrors the behavior of the original bash script but exposes a dry-run option
and raises exceptions on errors instead of exiting the interpreter.

It also includes a CLI-compatible `main()` so the script can be run from the
command line like the original shell script.

Note: The function will call external tools (recon-all, mri_binarize, fslmaths,
mri_convert). If these are not installed or available in PATH, the function will
raise a RuntimeError (unless run_cmds is False).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional


def _check_executable(name: str) -> bool:
    return shutil.which(name) is not None


class FreeSurferSegmentationError(RuntimeError):
    pass


def _run(cmd: list[str], cwd: Optional[str] = None, dry_run: bool = False) -> subprocess.CompletedProcess:
    """Run a command using subprocess.run, raising on non-zero exit unless dry_run.

    Returns the CompletedProcess when executed; in dry_run mode it only prints the command
    and returns a CompletedProcess-like dummy with returncode 0.
    """
    cmd_str = " ".join(shlex_quote(c) for c in cmd)
    print(f"Running: {cmd_str} (cwd={cwd})")
    if dry_run:
        # Return a dummy CompletedProcess-like object
        cp = subprocess.CompletedProcess(cmd, 0)
        return cp
    try:
        cp = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd, text=True)
        return cp
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {cmd_str}\nstdout:\n{e.stdout}\nstderr:\n{e.stderr}")
        raise FreeSurferSegmentationError(f"Command failed: {cmd_str}") from e

import shlex

def shlex_quote(s: str) -> str:
    """Simple quote for printing commands (does not affect execution)."""
    return shlex.quote(s)

def freesurfer_hipp_segmentation(input_image: str, output_hippo: str, run_cmds: bool = True) -> str:
    """Perform hippocampal segmentation using FreeSurfer commands.

    Parameters:
    - input_image: path to a T1-weighted image (must end with .nii.gz)
    - output_hippo: path where the final hippocampus segmentation (nii.gz) will be written
    - run_cmds: if False, do a dry run (print commands but do not execute them)

    Returns the absolute path of the produced `output_hippo` on success.

    Raises FreeSurferSegmentationError on failure.
    """
    # Validate inputs
    if not input_image or not output_hippo:
        raise ValueError("Both input_image and output_hippo must be provided")

    input_path = Path(input_image)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file {input_image} does not exist")

    if not input_image.endswith('.nii.gz'):
        raise ValueError(f"Input file {input_image} is not a .nii.gz file")

    input_dir = str(input_path.parent)
    base_name = input_path.name
    # remove .nii.gz suffix in a robust way
    if base_name.endswith('.nii.gz'):
        base_name = base_name[: -len('.nii.gz')]

    input_nii = str(input_path.resolve())

    # Set SUBJECTS_DIR to the directory containing the input image
    subjects_dir = input_dir
    os.environ['SUBJECTS_DIR'] = subjects_dir

    # subject id = FS_<base_name>, keep only alnum and underscore
    subject_id = f"FS_{base_name}"
    subject_id = re.sub(r'[^A-Za-z0-9_]', '', subject_id)

    subject_dir = Path(subjects_dir) / subject_id
    fs_mask_dir = subject_dir / 'mri'
    aseg_path = fs_mask_dir / 'aseg.mgz'

    # Check for required executables if we'll run commands
    required_cmds = ['recon-all', 'mri_binarize', 'fslmaths', 'mri_convert']
    if run_cmds:
        missing = [c for c in required_cmds if not _check_executable(c)]
        if missing:
            raise FreeSurferSegmentationError(f"Missing required executables in PATH: {', '.join(missing)}")

    # Run or skip recon-all based on presence of subject directory and aseg.mgz
    if subject_dir.is_dir():
        print(f"Existing FreeSurfer subject directory found: {subject_dir}")
        if not aseg_path.exists():
            print(f"aseg.mgz file not found at {aseg_path}, running recon-all -s {subject_id} -all")
            cmd = ['recon-all', '-s', subject_id, '-all']
            _run(cmd, cwd=None, dry_run=not run_cmds)
        else:
            print(f"aseg.mgz found at {aseg_path}, skipping recon-all")
    else:
        print(f"Creating FreeSurfer subject directory by running recon-all -i {input_nii} -s {subject_id} -all")
        cmd = ['recon-all', '-i', input_nii, '-s', subject_id, '-all']
        _run(cmd, cwd=None, dry_run=not run_cmds)

    # After recon-all, ensure aseg.mgz exists
    if not aseg_path.exists():
        raise FreeSurferSegmentationError(f"aseg.mgz file not found at expected location: {aseg_path}")

    # Work inside the FS mri directory
    cwd = str(fs_mask_dir)

    # create left hippocampus mask (label 17)
    lh_out = 'lh_hippocampus.nii.gz'
    rh_out = 'rh_hippocampus.nii.gz'
    hippo_mgz = 'hippo.mgz'

    cmd_lh = ['mri_binarize', '--i', str(aseg_path), '--match', '17', '--o', lh_out, '--binval', '1']
    _run(cmd_lh, cwd=cwd, dry_run=not run_cmds)

    # create right hippocampus mask (label 53)
    cmd_rh = ['mri_binarize', '--i', str(aseg_path), '--match', '53', '--o', rh_out, '--binval', '2']
    _run(cmd_rh, cwd=cwd, dry_run=not run_cmds)

    # combine left and right masks
    cmd_add = ['fslmaths', lh_out, '-add', rh_out, hippo_mgz]
    _run(cmd_add, cwd=cwd, dry_run=not run_cmds)

    # convert to desired output file
    cmd_convert = ['mri_convert', hippo_mgz, str(output_hippo)]
    _run(cmd_convert, cwd=cwd, dry_run=not run_cmds)

    # cleanup intermediate files (best-effort)
    try:
        if run_cmds:
            (fs_mask_dir / lh_out).unlink(missing_ok=True)
            (fs_mask_dir / rh_out).unlink(missing_ok=True)
            (fs_mask_dir / hippo_mgz).unlink(missing_ok=True)
        else:
            print(f"Dry-run: would remove {lh_out}, {rh_out}, {hippo_mgz} from {cwd}")
    except Exception as e:
        # non-fatal cleanup failure
        print(f"Warning: failed to remove intermediate files: {e}")

    print(f"Hippocampal segmentation saved to {output_hippo}")
    return str(Path(output_hippo).resolve())


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run freesurfer hippocampal segmentation (Python port of the bash script)')
    parser.add_argument('input_image', help='Input T1 image (nii.gz)')
    parser.add_argument('output_hippo', help='Output hippocampus segmentation (nii.gz)')
    parser.add_argument('--dry-run', action='store_true', help='Print commands but do not execute them')

    args = parser.parse_args()
    try:
        freesurfer_hipp_segmentation(args.input_image, args.output_hippo, run_cmds=not args.dry_run)
    except Exception as exc:
        print(f"ERROR: {exc}")
        raise

