# this script fix the meshes with the code in shape analyss repo
import argparse
import os
from tqdm import tqdm
from hippospharm.fix_mesh import fix_mesh

# get argument datapath from sys.argv

# Argument parser
parser = argparse.ArgumentParser(description='Extract features from hippocampus images')
parser.add_argument('datapath', nargs='?', default=os.environ.get('DATAPATH'), help='Path to the data directory or set the environment variable DATAPATH')
parser.add_argument('remeshbin', nargs='?', default=os.environ.get('REMESHBIN'), help='Path to the remesh binary or set the environment variable REMESHBIN')
parser.add_argument('-N', '--num', type=int, default=6890, help='Number of vertices in the remeshed mesh')
parser.add_argument('--skip', action='store_true', help='Skip meshes that cannot be fixed')




args = parser.parse_args()
datapath = args.datapath
mesh_bin = args.remeshbin
target_vertices = args.num
is_skip_fails = args.skip
# create models path
print(datapath)
models_path = os.path.join(datapath, 'models')
fix_path = os.path.join(datapath, 'fixmodels')

# collect all obj files in models path
files = [f for f in os.listdir(models_path) if f.endswith('.obj')]
# create a folder called fixmodels in datapath
if not os.path.exists(fix_path):
    os.makedirs(fix_path)
    print('Created fixmodels folder')
else:
    print('models folder already exists models will be overwritten')
print('--------------------')
print('processing....')
# create list of commands
for i, f in enumerate(tqdm(files)):
    print(f'{i} : {f}')
    # create a mesh object
    input_mesh_path = os.path.join(models_path, f)
    # fix the mesh
    if is_skip_fails:
        try:
            mesh = fix_mesh(input_mesh_path, target_vertices=target_vertices, remesh_bin=mesh_bin)
        except:
            print(f"Failed to fix {f}")
            continue
    else:
        mesh = fix_mesh(input_mesh_path, target_vertices=target_vertices, remesh_bin=mesh_bin)
    # save the mesh
    mesh.export(os.path.join(fix_path, f))






