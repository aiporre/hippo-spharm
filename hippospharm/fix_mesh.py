import os
import subprocess
import trimesh
import numpy as np


def quadric_decimation_garland(resampled_mesh, target_vertices):
    # use Garland and Heckbert's quadric error metrics for mesh simplification
    from quad_mesh_simplify import simplify_mesh
    positions = np.array(resampled_mesh.vertices).astype(float)
    face = np.array(resampled_mesh.faces).astype("uint32")
    print(f'running garland quadric decimation to fix the mesh: from {positions.shape[0]} to {target_vertices} vertices')
    new_positions, new_face = simplify_mesh(positions, face, target_vertices)
    resampled_mesh = trimesh.Trimesh(vertices=new_positions, faces=new_face, process=False)
    print('number of vertices after garland quadric decimation:', resampled_mesh.vertices.shape[0])
    return resampled_mesh

def mesh_fix(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Fixes mesh holes and smooths the mesh using Laplacian smoothing.

    Parameters
    ----------
    mesh: trimesh.Trimesh
        The input mesh to be processed.

    Returns
    -------
    trimesh.Trimesh:
        The processed mesh.
    """
    import pymeshfix
    meshfix = pymeshfix.MeshFix(mesh.vertices, mesh.faces)
    meshfix.repair(verbose=True)
    # make a trimesh object
    fixed_mesh = trimesh.Trimesh(vertices=meshfix.v, faces=meshfix.f, process=False)
    # smooth the mesh
    return fixed_mesh



def fix_mesh(mesh_filename:str, target_vertices:int=6890, remesh_bin=None, suffix='.obj', tolerance_num_vertices=10, plus=False, use_mesh_fix=True) -> trimesh.Trimesh:
    """
    Fixes mesh holes, smooths the mesh using Laplacian smoothing, and resamples it to the target number of vertices.

    Parameters
    ----------
    mesh: trimesh.Trimesh
        The input mesh to be processed.
    target_vertices: int
        The target number of vertices for the resampled mesh.

    Returns
    -------
    trimesh.Trimesh:
        The processed mesh.
    """
    assert remesh_bin is not None, "manifold_bin_path is not provided"

    # get a temp folder
    temp_dir = "./temp_meshes"
    if not suffix.startswith('.'):
        suffix = "." + suffix
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    fbname = os.path.basename(mesh_filename).replace(suffix, '')
    manifold_output_filename = f"{temp_dir}/{fbname}_manifold_output{suffix}"
    output_filename = f"{temp_dir}/{fbname}_output{suffix}"

    # Load the original mesh and check if remeshing is necessary
    original_mesh = trimesh.load_mesh(mesh_filename)
    broken_faces = trimesh.repair.broken_faces(original_mesh)
    num_broken = len(broken_faces) if hasattr(broken_faces, '__len__') else int(broken_faces)
    num_vertices = original_mesh.vertices.shape[0]
    print(f"Original mesh: {num_broken} broken faces, {num_vertices} vertices.")

    if num_broken == 0 and abs(num_vertices - target_vertices) < tolerance_num_vertices:
        print("Mesh already has no holes and is within target vertex tolerance. Skipping remeshing.")
        # export to output file for consistency and return
        original_mesh.export(output_filename)
        return original_mesh

    # remeshing required, ensure remesh binary is provided
    assert remesh_bin is not None, "manifold_bin_path is not provided but remeshing is required"

    # execute remeshing binary and wait for it to finish
    if plus:
        result = subprocess.run([f"{remesh_bin}", "--input", mesh_filename, "--output", manifold_output_filename],
                                check=True, capture_output=True, text=True)
    else:
        result = subprocess.run([f"{remesh_bin}", mesh_filename, manifold_output_filename],
                                check=True, capture_output=True, text=True)
    print("stdout:", result.stdout)
    print("stderr:", result.stderr)
    mesh = trimesh.load_mesh(manifold_output_filename)
    broken_faces = trimesh.repair.broken_faces(mesh)
    print(f"Number of broken triangles found manifold output {broken_faces} and number of vertices {mesh.vertices.shape[0]}.")
    # if there are broken faces try to fix them
    if len(broken_faces) > 0:
        print("Fixing them.. filling holes")
        trimesh.repair.fill_holes(mesh)
        if use_mesh_fix:
            print("Using pymeshfix to further fix the mesh")
            mesh = mesh_fix(mesh)
        broken_faces = trimesh.repair.broken_faces(mesh)
        print(f"Number of broken triangles after filling holes {broken_faces}.")
    # if still broken faces just load the original mesh
    if len(broken_faces) > 0:
        print("Still broken faces after filling holes, loading original mesh")
        mesh = trimesh.load_mesh(mesh_filename)
        print("Number of broken triangles in original mesh:", trimesh.repair.broken_faces(mesh))
    broken_faces = trimesh.repair.broken_faces(mesh)
    if len(broken_faces) > 0:
        print("Warning: still broken faces in the mesh, proceeding anyway.")
    # check first is if with garland quadric decimation we can reach the target vertices directly
    if mesh.vertices.shape[0] > target_vertices and mesh.vertices.shape[0] < 10*target_vertices:
        print('attempting to reach target vertices directly with garland quadric decimation')
        resampled_mesh = quadric_decimation_garland(mesh, target_vertices)
        broken_faces = trimesh.repair.broken_faces(resampled_mesh)
        if len(broken_faces) == 0 and np.abs(resampled_mesh.vertices.shape[0] - target_vertices) < tolerance_num_vertices:
            print('successfully reached target vertices with garland quadric decimation')
            mesh = resampled_mesh
            mesh.export(output_filename)
            return mesh
        else:
            print('failed to reach target vertices with garland quadric decimation, proceeding with iterative approach " \n',
                  f'current number of vertices: {resampled_mesh.vertices.shape[0]}, broken faces: {broken_faces}')
    # compute the number of faces
    def within_range(curr_num_vertices):
        return np.abs(target_vertices - curr_num_vertices) < tolerance_num_vertices
    # check before
    if len(broken_faces) == 0 and within_range(mesh.vertices.shape[0]):
        print('mesh is already good, no holes and correct number of vertices (', mesh.vertices.shape[0],')')
        mesh.export(output_filename)
        return mesh
    target_faces = 2 * target_vertices - 4
    no_holes = False
    attempts = 0
    aggressivity = 10
    more_faces = 0
    smooth_iteration = 5

    while not no_holes or attempts <= 20:
        print('simplifiying mesh', target_faces, aggressivity, attempts)
        # run quadric decimation
        if attempts>5:
            print(' attempting to smooth more the mesh since now more than 5 attempts')
            # try to smooth more
            mesh = trimesh.smoothing.filter_laplacian(mesh, iterations=smooth_iteration)
            smooth_iteration = smooth_iteration + 5
            aggressivity = 5

        print(' new mesh fast simplify mesh with aggresivity ', aggressivity, ' and more faces ', more_faces)
        resampled_mesh = mesh.simplify_quadric_decimation(face_count=target_faces+more_faces, aggression=aggressivity)
        more_faces = more_faces + int(0.05 * target_faces) # increase target faces by 5% each attempt


        if resampled_mesh.vertices.shape[0] == target_vertices:
            print("Resampled mesh has the correct number of vertices.")
        else:
            print( f"Resampled mesh has {resampled_mesh.vertices.shape[0]} vertices instead of {target_vertices} vertices.")
            print('attempting to fix.. with other library')
            if resampled_mesh.vertices.shape[0] <= 5*target_vertices:
                resampled_mesh = quadric_decimation_garland(resampled_mesh, target_vertices)
                print('number of vertices after garland quadric decimation:', resampled_mesh.vertices.shape[0])
                print('number of broken faces after garland quadric decimation :',
                      trimesh.repair.broken_faces(resampled_mesh))
            else:
                print('the number of vertices is not too far from target, skipping garland quadric decimation')


        # save mesh as a.obj file
        broken_faces = trimesh.repair.broken_faces(resampled_mesh)
        print(f"Found {broken_faces} broken faces in the mesh after quadric decimation.")
        if len(broken_faces) > 0:
            print(f"Fixing them.. filling holes")
            trimesh.repair.fill_holes(resampled_mesh)
            if use_mesh_fix:
                print("Using pymeshfix to further fix the mesh")
                resampled_mesh = mesh_fix(resampled_mesh)
                broken_faces = trimesh.repair.broken_faces(resampled_mesh)
                # apply quadratic decimation again to reach target vertices
                if not within_range(resampled_mesh.vertices.shape[0]) and len(broken_faces) == 0:
                    print('run garland quadric decimation afet mesh fix to reach target vertices, with no broken faces')
                    resampled_mesh = quadric_decimation_garland(resampled_mesh, target_vertices)
                    print('number of vertices after garland quadric from mesh fix:', resampled_mesh.vertices.shape[0])
                    bf = trimesh.repair.broken_faces(resampled_mesh)
                    print('..and number of broken faces after garland quadric decimation from mesh fix :', bf, ' number of broken faces', len(bf))
            # resampled_mesh = trimesh.smoothing.filter_laplacian(resampled_mesh, iterations=10)
            broken_faces = trimesh.repair.broken_faces(resampled_mesh)
            print(f"Number of {broken_faces} broken faces in the mesh after filling holes?")

        broken_faces = trimesh.repair.broken_faces(resampled_mesh)
        if len(broken_faces) == 0 and within_range(resampled_mesh.vertices.shape[0]):
            print('no holes and correct number of vertices (', resampled_mesh.vertices.shape[0],')')
            no_holes = True
            attempts = 21
            resampled_mesh.export(output_filename)
            print('numer of versitce', resampled_mesh.vertices.shape[0])
            print('nmber of faces', resampled_mesh.faces.shape[0])
        else:
            aggressivity = aggressivity - 1
            print(f'current number of vertices: {resampled_mesh.vertices.shape[0]}')
            print(f"After all Found {broken_faces} broken faces in the mesh. Retrying.., with agresivity {aggressivity}")
        attempts = attempts + 1
        if attempts > 20:
            if not no_holes:
                print('To many wholes! attemp is more that 20')
                raise ValueError(f"failed to fix mesh {mesh_filename}")
            break




    return resampled_mesh
