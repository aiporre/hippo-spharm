# this script fix the meshes with the code in shape analyss repo
import subprocess
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
parser.add_argument('--keep-dirs', action='store_true', help='keeps the file hierachy of the mesh when fixing it')
parser.add_argument('--suffix', '-s', type=str, default='.obj', help='Suffix of the mesh files to fix, default is .obj')

args = parser.parse_args()
datapath = args.datapath
mesh_bin = args.remeshbin
target_vertices = args.num
is_skip_fails = args.skip
keep_dirs = args.keep_dirs
suffix = args.suffix
# create models path
print(datapath)
models_path = os.path.join(datapath, 'models')
fix_path = os.path.join(datapath, 'fixmodels')

# collect all obj files in models path
if keep_dirs:
    # looks files from the models path recursively keeping example plane/train/data.obj
    files = []
    for root, dirs, files_in_dir in os.walk(models_path):
        for f in files_in_dir:
            if f.endswith(suffix):
                ff = os.path.join(root, f)
                ff = ff.replace(models_path, '')
                if ff.startswith(os.sep):
                    ff = ff[1:]
                # remove the models path from the beginning of the path
                files.append(ff)
else:
    files = [f for f in os.listdir(models_path) if f.endswith(suffix)]
if len(files) == 0:
    print(f"No {suffix} files found in {models_path}. Please check the directory.")
    exit(0)
print(files[0])
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
    print('---->> model path', models_path)
    print('---->> mesh_file', input_mesh_path)
    if suffix != '.obj':
        print(f"Warning: the suffix {suffix} is not .obj, this may cause issues with the remeshing process.")
        print('convert to temporary .obj file')
        # convert the mesh to obj file
        import trimesh
        mesh = trimesh.load(input_mesh_path)
        input_mesh_path = input_mesh_path.replace(suffix, '.obj')
        if os.path.exists(input_mesh_path):
            print(f"Warning: {input_mesh_path} already exists, it wont be overwritten.")
        else:
            mesh.export(input_mesh_path)

    # fix the mesh
    if is_skip_fails:
        try:
            mesh = fix_mesh(input_mesh_path, target_vertices=target_vertices, remesh_bin=mesh_bin)
        except Exception as e:
            print(e)
            print(f"Failed to fix {f}")
            continue
    else:
        mesh = fix_mesh(input_mesh_path, target_vertices=target_vertices, remesh_bin=mesh_bin)
    # make last evaluation of the mesh to check number of vertices
    if mesh.vertices.shape[0] != target_vertices:
        print('running blender remesh to fix the mesh')
        temp_file = os.path.join(fix_path, f.replace(suffix, '_temp.obj'))
        mesh.export(temp_file)
        command = ['blender', '--background', '--python-exit-code', '1', '--python', 'scripts/blender_remesh.py',
                  '--', temp_file, str(target_vertices) ]
        # runinng command
        print('running command', command)
        command_out = subprocess.run(command, capture_output=True, text=True)
        if command_out.returncode != 0:
            print(f"Failed to remesh {f} with blender. Error: {command_out.stderr}")
            continue
    # save the mesh
    if keep_dirs:
        # create the directory structure in fix_path
        dir_structure = os.path.dirname(f)

        if dir_structure:
            os.makedirs(os.path.join(fix_path, dir_structure), exist_ok=True)
        file = os.path.join(fix_path, f)
        print('---->> save mesh to', file)
        mesh.export(os.path.join(fix_path, f))
    else:
        # save the mesh in fix_path
        # without directory structure
        mesh.export(os.path.join(fix_path, f))






