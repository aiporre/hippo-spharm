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
 
