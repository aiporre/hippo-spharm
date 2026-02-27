# hippo-spharm
spherical harmonics of hippocampus for feature extraction

## pipeline

1. correct bias script in scripts folder
2. run from (repo/iodine) brain extraction from hippomapper see paper for refeerences (link to code)[https://github.com/AICONSlab/HippMapp3r]
3. run hippocampus extractions
4. run hippocampus mesh
5. run fix mesh script 
6. run feature extraction.

## to fix mesh
set a fix number of meshes, and fixes holes 

```bash
export REMESHBIN=/path/to/code/Manifold/build/manifold
python scripts/fix_meshes /path/to/models
```
## installation of FSL

```bash
wget https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/7.4.1/freesurfer-linux-ubuntu20_amd64-7.4.1.tar.gz
tar -xzvf freesurfer-linux-ubuntu20_amd64-7.4.1.tar.gz
mv freesurfer $HOME/
```
copy this into your bashrc
```bash
export FREESURFER_HOME=$HOME/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh
```
# IXI pipeline
1. run bias correction on the T1 image using the script in the scripts folder
2. run reorientation script in the scripts folder
3. run brain extraction using the FSL BET tool with isotropic because voxels to large
4. run hippocampus extraction using the FSL FIRST tool with isotropic because voxels to large
```
export DATAPATH=`/media/sauron/GG2/datasets/IXIDATA/IXI_BIDS`
conda activate hippmapper 
# correct bias
# this creates a _corrected.nii.gz file
python scripts/correct_bias.py `$DATAPATH` -s

# check orientation
# this creates a _reoriented.nii.gz file
python scripts/check_data_orientation.py `$DATAPATH` -s 

# brain segmentation
# this creates a _reoriented_brain.nii.gz file
python scripts/segment_brain.py `$DATAPATH` -s -r -i

# hippocampus segmentation
# this creates the hippocampus segmenation reoriented_brain_seg.nii.gz file


python scripts/segment_hippocampus.py `$DATAPATH` -s -t brain --brain_target reoriented
```


# installation

```
conda create -n hippospharm python=3.10 pip
```

fix meshes requires it own conda env


```
conda create -n hippospharm python=3.7 pip
pip install quad_mesh_simpify==1.1.5
```

alternative for it worked better just to get the repo and build the wheel and intall it. TODO: put commands here.


