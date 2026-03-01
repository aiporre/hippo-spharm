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
export DATAPATH="/media/$USER/GG2/datasets/IXIDATA/IXI_BIDS"
conda activate hippmapper 
# correct bias
# this creates a _corrected.nii.gz file
python scripts/correct_bias.py $DATAPATH -s

# check orientation
# this creates a _reoriented.nii.gz file
python scripts/check_data_orientation.py $DATAPATH -s 

# brain segmentation
# this creates a _reoriented_brain.nii.gz file
python scripts/segment_brain.py $DATAPATH -s -r -i

# hippocampus segmentation
# this creates the hippocampus segmenation reoriented_brain_seg.nii.gz file


python scripts/segment_hippocampus.py $DATAPATH -s -t brain --brain_target reoriented
```

# HCP aging pipeline
1. run the hpa.py conversion from the hcp_convert library (here copy link to my fork) to convert the HCP data to BIDS format

> NOTE: all data is there just need to run hippocampus segmentation
2. run hippmapper pipeline on the converted data
3. run extract mesh
4. run fix mesh script
5. run feature extraction this is the bsa code 

commands we used

```bash
export DATAPATH="/media/$USER/GG2/datasets/HCP_aging"
# convert to bids
conda activate clinica
python hca.py --hca_dir /home/$USER/Documents/Phd/data/HPC_Aging/raw --output_dir /media/$USER/mirko/HCP_aging/HCPAging_BIDS --method move --extract_dir /media/$USER/mirko/temp/HCP_aging_extractions/
# run segment hippocampus
conda activate hippmapper
python scripts/segment_hippocampus.py $DATAPATH -t brain
# run mesh extraction
python scripts/extract_hipp_mesh.py /media/$USER/mirko/HCP_aging/HCPAging_BIDS -t brain
# run fix mesh script
conda activate remesh
python scripts/fix_meshes.py $DATAPATH ./scripts/manifold -N 3000 --skip -t 0

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

## install remesh evironment
There are some incopatibilies in the scripts so we need a separate environment for remeshing. 

```bash

conda create -n remesh python=3.10 pip numpy==1.26.0 scipy==1.15.3
 git clone https://github.com/jannessm/quadric-mesh-simplification.git
 
 cd quadric-mesh-simplification/
 pip install requirements
 pip install -r requirements.txt 
 python setup.py build_ext --inplace
 pip install .
```
after this you need to install something like this:

```css
Package             Version
------------------- -------
Cython              3.2.4
fast_simplification 0.1.13
networkx            3.4.2
numpy               1.26.0
packaging           25.0
pip                 26.0.1
quad_mesh_simplify  1.1.5
scipy               1.15.3
setuptools          80.10.2
tqdm                4.67.3
trimesh             4.11.2
```
