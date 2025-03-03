import os
import subprocess
import trimesh

def fix_mesh(mesh_filename:str, target_vertices:int=6890, remesh_bin=None):
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
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    manifold_output_filename = f"{temp_dir}/manifold_output.obj"
    output_filename = f"{temp_dir}/output.obj"


    # Resample the mesh to the target number of vertices
    # execute bin and wait for it to finish
    subprocess.run([f"{remesh_bin}", mesh_filename, manifold_output_filename], check=True)
    mesh = trimesh.load_mesh(manifold_output_filename)
    broken_face_num = trimesh.repair.broken_faces(mesh)
    print(f"Number of broken triangles found manifold output {broken_face_num}.")
    # compute the number of faces
    target_faces = 2 * target_vertices - 4
    no_holes = False
    attempts = 0
    aggressivity = 10
    while not no_holes or attempts <= 10:
        print('simplifiying mesh', target_faces, aggressivity, attempts)
        resampled_mesh = mesh.simplify_quadric_decimation(face_count=target_faces, aggression=aggressivity)
        if resampled_mesh.vertices.shape[0] == target_vertices:
            print( f"Resampled mesh has {resampled_mesh.vertices.shape[0]} vertices instead of {target_vertices} vertices.")

        # save mesh as a.obj file
        broken_face_num = trimesh.repair.broken_faces(resampled_mesh)
        print(f"Found {broken_face_num} broken faces in the mesh after quadric decimation.")
        if len(broken_face_num) > 0:
            print(f"Fixing them.. filling holes")
            trimesh.repair.fill_holes(resampled_mesh)
            # resampled_mesh = trimesh.smoothing.filter_laplacian(resampled_mesh, iterations=10)
            print(f"Number of {broken_face_num} broken faces in the mesh after filling holes?")

        broken_face_num = trimesh.repair.broken_faces(resampled_mesh)
        if len(broken_face_num) == 0 and resampled_mesh.vertices.shape[0] == target_vertices:
            print('no holes and correct number of vertices')
            no_holes = True
            attempts = 11
            resampled_mesh.export(output_filename)
            print('numer of versitce', resampled_mesh.vertices.shape[0])
            print('nmber of faces', resampled_mesh.faces.shape[0])
        else:
            aggressivity = aggressivity - 1
            print(f"After all Found {broken_face_num} broken faces in the mesh. Retrying.., with agresivity {aggressivity}")
        attempts = attempts + 1
        if attempts > 11:
            if not no_holes:
                raise ValueError(f"failed to fix mesh {mesh_filename}")
            break


    return resampled_mesh
