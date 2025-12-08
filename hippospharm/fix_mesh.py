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


def fix_mesh(mesh_filename:str, target_vertices:int=6890, remesh_bin=None, suffix='.obj', tolerance_num_vertices=10, plus=False) -> trimesh.Trimesh:
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
    manifold_output_filename = f"{temp_dir}/manifold_output{suffix}"
    output_filename = f"{temp_dir}/output{suffix}"


    # Resample the mesh to the target number of vertices
    # execute bin and wait for it to finish
    if plus:
        result = subprocess.run([f"{remesh_bin}", "--input", mesh_filename, "--output", manifold_output_filename],
                                check=True, capture_output=True, text=True)
    else:
        result = subprocess.run([f"{remesh_bin}", mesh_filename, manifold_output_filename],
                                check=True, capture_output=True, text=True)
    print("stdout:", result.stdout)
    print("stderr:", result.stderr)
    mesh = trimesh.load_mesh(manifold_output_filename)
    broken_face_num = trimesh.repair.broken_faces(mesh)
    print(f"Number of broken triangles found manifold output {broken_face_num}.")
    # compute the number of faces
    target_faces = 2 * target_vertices - 4
    no_holes = False
    attempts = 0
    aggressivity = 10
    def within_range(curr_num_vertices):
        return np.abs(target_vertices - curr_num_vertices) < tolerance_num_vertices

    while not no_holes or attempts <= 10:
        print('simplifiying mesh', target_faces, aggressivity, attempts)
        # run quadric decimation
        if attempts>0:
            # smooth a bit
            mesh = trimesh.smoothing.filter_laplacian(mesh, iterations=5 + attempts)
        more_faces = 10 * attempts
        resampled_mesh = mesh.simplify_quadric_decimation(face_count=target_faces+more_faces, aggression=aggressivity)
        if resampled_mesh.vertices.shape[0] == target_vertices:
            print("Resampled mesh has the correct number of vertices.")
        else:
            print( f"Resampled mesh has {resampled_mesh.vertices.shape[0]} vertices instead of {target_vertices} vertices.")
            print('attempting to fix.. with other library')
            resampled_mesh = quadric_decimation_garland(resampled_mesh, target_vertices)


        # save mesh as a.obj file
        broken_face_num = trimesh.repair.broken_faces(resampled_mesh)
        print(f"Found {broken_face_num} broken faces in the mesh after quadric decimation.")
        if len(broken_face_num) > 0:
            print(f"Fixing them.. filling holes")
            trimesh.repair.fill_holes(resampled_mesh)
            # resampled_mesh = trimesh.smoothing.filter_laplacian(resampled_mesh, iterations=10)
            print(f"Number of {broken_face_num} broken faces in the mesh after filling holes?")

        broken_face_num = trimesh.repair.broken_faces(resampled_mesh)
        if len(broken_face_num) == 0 and within_range(resampled_mesh.vertices.shape[0]):
            print('no holes and correct number of vertices (', resampled_mesh.vertices.shape[0],')')
            no_holes = True
            attempts = 11
            resampled_mesh.export(output_filename)
            print('numer of versitce', resampled_mesh.vertices.shape[0])
            print('nmber of faces', resampled_mesh.faces.shape[0])
        else:
            aggressivity = aggressivity - 1
            print(f'current number of vertices: {resampled_mesh.vertices.shape[0]}')
            print(f"After all Found {broken_face_num} broken faces in the mesh. Retrying.., with agresivity {aggressivity}")
        attempts = attempts + 1
        if attempts > 11:
            if not no_holes:
                print('To many wholes! attemp is more that 11')
                raise ValueError(f"failed to fix mesh {mesh_filename}")
            break




    return resampled_mesh
