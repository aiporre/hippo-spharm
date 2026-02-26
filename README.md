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


