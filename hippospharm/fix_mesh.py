# python
import os
import subprocess
import logging
import trimesh
import numpy as np
from trimesh.graph import neighbors

logger = logging.getLogger(__name__)


def quadric_decimation_garland(resampled_mesh, target_vertices):
    # use Garland and Heckbert's quadric error metrics for mesh simplification
    from quad_mesh_simplify import simplify_mesh
    positions = np.array(resampled_mesh.vertices).astype(float)
    face = np.array(resampled_mesh.faces).astype("uint32")
    logger.info("Garland quadric decimation: reducing vertices from %d to %d", positions.shape[0], target_vertices)
    new_positions, new_face = simplify_mesh(positions, face, target_vertices)
    resampled_mesh = trimesh.Trimesh(vertices=new_positions, faces=new_face, process=False)
    logger.info("Vertices after Garland quadric decimation: %d", resampled_mesh.vertices.shape[0])
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
    try:
        import pymeshfix
    except Exception as e:
        logger.warning("pymeshfix import failed: %s. Skipping pymeshfix-based fixes.", e)
        return mesh

    logger.info("Running pymeshfix.MeshFix on mesh with %d vertices and %d faces",
                mesh.vertices.shape[0], mesh.faces.shape[0])
    meshfix = pymeshfix.MeshFix(mesh.vertices, mesh.faces)
    meshfix.repair(verbose=False)
    fixed_mesh = trimesh.Trimesh(vertices=meshfix.v, faces=meshfix.f, process=False)
    logger.info("pymeshfix completed: resulting vertices=%d, faces=%d",
                fixed_mesh.vertices.shape[0], fixed_mesh.faces.shape[0])
    return fixed_mesh


def add_vertices_by_edge_split(mesh: trimesh.Trimesh, target_vertices: int) -> trimesh.Trimesh:
    """
    Adds vertices to the mesh by splitting edges until the target number of vertices is reached.
    :param mesh:
    :param target_vertices:
    :return:
    """
    num_v_current = mesh.vertices.shape[0]
    to_add = target_vertices - num_v_current
    if to_add <= 0 or to_add > 50:
        if to_add > 50:
            logger.warning("Requested to add %d vertices (more than 50). Skipping edge-split insertion.", to_add)
        else:
            logger.info("No edge-split required: current vertices=%d target=%d", num_v_current, target_vertices)
        return mesh

    logger.info("Adding %d vertices to mesh by iterative edge split (current=%d, target=%d)",
                to_add, num_v_current, target_vertices)
    mesh = mesh.copy()
    for i in range(to_add):
        edges = mesh.edges_unique
        edge_lengths = mesh.edges_unique_length
        longest_edge_index = np.argmax(edge_lengths)
        longest_edge = edges[longest_edge_index]
        v1_idx, v2_idx = int(longest_edge[0]), int(longest_edge[1])
        v1 = mesh.vertices[v1_idx]
        v2 = mesh.vertices[v2_idx]
        midpoint = (v1 + v2) / 2.0
        new_vertex_index = len(mesh.vertices)
        mesh.vertices = np.vstack([mesh.vertices, midpoint])
        faces_to_update = []
        for face_index, face in enumerate(mesh.faces):
            if v1_idx in face and v2_idx in face:
                faces_to_update.append(face_index)
        for face_index in faces_to_update:
            face = mesh.faces[face_index]
            adjacent = [v for v in face if v not in [v1_idx, v2_idx]]
            if not adjacent:
                continue
            adjacent_vertex = int(adjacent[0])
            face1 = [v1_idx, new_vertex_index, adjacent_vertex]
            face2 = [new_vertex_index, v2_idx, adjacent_vertex]
            mesh.faces[face_index] = face1
            mesh.faces = np.vstack([mesh.faces, face2])
        # update mesh
        mesh.merge_vertices()
        mesh.update_faces(mesh.nondegenerate_faces(height=None))
        logger.debug("Added vertex %d/%d (new total=%d)", i + 1, to_add, mesh.vertices.shape[0])
    logger.info("Finished adding vertices: new vertex count=%d", mesh.vertices.shape[0])
    return mesh


def fix_mesh(mesh_filename: str, target_vertices: int = 6890, remesh_bin=None, suffix='.obj',
             tolerance_num_vertices=10, plus=False, use_mesh_fix=True) -> trimesh.Trimesh:
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
    logger.info("Loaded original mesh: broken_faces=%s, vertices=%d", num_broken, num_vertices)

    if num_broken == 0 and abs(num_vertices - target_vertices) < tolerance_num_vertices:
        logger.info("Mesh has no holes and is within vertex tolerance (%d ± %d). Skipping remeshing.",
                    target_vertices, tolerance_num_vertices)
        original_mesh.export(output_filename)
        return original_mesh

    # filling and fixing before remeshing
    if num_broken > 0:
        logger.info("Filling holes in original mesh before remeshing.")
        trimesh.repair.fill_holes(original_mesh)
        if use_mesh_fix:
            logger.info("Applying mesh_fix (pymeshfix) to original mesh before remeshing.")
            original_mesh = mesh_fix(original_mesh)
        broken_faces = trimesh.repair.broken_faces(original_mesh)
        logger.info("Broken faces after filling/fixing original mesh: %s", broken_faces)
        if len(broken_faces) > 0:
            logger.warning("Original mesh still has broken faces after pre-remeshing fixes.")
        # save the fixed original mesh back to file for remeshing
        mesh_filename = f"{temp_dir}/{fbname}_fixed_mesh{suffix}"
        original_mesh.export(mesh_filename)
        logger.info("Exported pre-fixed mesh for remeshing: %s", mesh_filename)

    # remeshing required, ensure remesh binary is provided
    assert remesh_bin is not None, "manifold_bin_path is not provided but remeshing is required"

    try:
        if plus:
            cmd = [f"{remesh_bin}", "--input", mesh_filename, "--output", manifold_output_filename]
        else:
            cmd = [f"{remesh_bin}", mesh_filename, manifold_output_filename]
        logger.info("Executing remeshing binary: %s", " ".join(cmd))
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.debug("Remeshing stdout: %s", result.stdout)
        logger.debug("Remeshing stderr: %s", result.stderr)
    except subprocess.CalledProcessError as e:
        logger.error("Remeshing binary failed: returncode=%s, stderr=%s", getattr(e, 'returncode', None), getattr(e, 'stderr', None))
        raise

    mesh = trimesh.load_mesh(manifold_output_filename)
    broken_faces = trimesh.repair.broken_faces(mesh)
    logger.info("Manifold output: broken_faces=%s, vertices=%d", broken_faces, mesh.vertices.shape[0])

    if len(broken_faces) > 0:
        logger.info("Filling holes in manifold output.")
        trimesh.repair.fill_holes(mesh)
        if use_mesh_fix:
            logger.info("Applying mesh_fix (pymeshfix) to manifold output.")
            mesh = mesh_fix(mesh)
        broken_faces = trimesh.repair.broken_faces(mesh)
        logger.info("Broken faces after filling/fixing: %s", broken_faces)

    if len(broken_faces) > 0:
        logger.warning("Manifold output still has broken faces. Falling back to original mesh.")
        mesh = trimesh.load_mesh(mesh_filename)
        logger.info("Original mesh broken faces: %s", trimesh.repair.broken_faces(mesh))

    broken_faces = trimesh.repair.broken_faces(mesh)
    if len(broken_faces) > 0:
        logger.warning("Proceeding despite %d broken faces.", len(broken_faces))

    if mesh.vertices.shape[0] > target_vertices and mesh.vertices.shape[0] < 10 * target_vertices:
        logger.info("Trying to reach target vertices with Garland quadric decimation (fast path).")
        resampled_mesh = quadric_decimation_garland(mesh, target_vertices)
        broken_faces = trimesh.repair.broken_faces(resampled_mesh)
        if len(broken_faces) == 0 and np.abs(resampled_mesh.vertices.shape[0] - target_vertices) < tolerance_num_vertices:
            logger.info("Successfully reached target vertices with Garland decimation: vertices=%d", resampled_mesh.vertices.shape[0])
            mesh = resampled_mesh
            mesh.export(output_filename)
            return mesh
        else:
            logger.info("Garland decimation did not reach target or produced broken faces (vertices=%d, broken=%s).",
                        resampled_mesh.vertices.shape[0], broken_faces)

    def within_range(curr_num_vertices):
        return np.abs(target_vertices - curr_num_vertices) < tolerance_num_vertices
    # check before
    if len(broken_faces) == 0 and within_range(mesh.vertices.shape[0]):
        logger.info("Mesh already valid and within target vertex range (%d).", mesh.vertices.shape[0])
        mesh.export(output_filename)
        return mesh
    target_faces = 2 * target_vertices - 4
    no_holes = False
    attempts = 0
    aggressivity = 10
    more_faces = 0
    smooth_iteration = 5

    while not no_holes or attempts <= 20:
        logger.info("Simplifying mesh (target_faces=%d, aggressivity=%d, attempt=%d)", target_faces, aggressivity, attempts)
        if attempts > 5:
            logger.info("More than 5 attempts: applying additional Laplacian smoothing (iterations=%d).", smooth_iteration)
            mesh = trimesh.smoothing.filter_laplacian(mesh, iterations=smooth_iteration)
            smooth_iteration += 5
            aggressivity = 5

        logger.debug("Running trimesh.simplify_quadric_decimation with aggressivity=%d and additional faces=%d", aggressivity, more_faces)
        resampled_mesh = mesh.simplify_quadric_decimation(face_count=target_faces + more_faces, aggression=aggressivity)
        more_faces += int(0.05 * target_faces)

        if resampled_mesh.vertices.shape[0] == target_vertices:
            logger.info("Resampled mesh matches target vertices exactly: %d", target_vertices)
        else:
            logger.info("Resampled mesh vertices=%d (target=%d). Attempting alternative simplification if applicable.",
                        resampled_mesh.vertices.shape[0], target_vertices)
            if resampled_mesh.vertices.shape[0] <= 5 * target_vertices:
                logger.info("Attempting secondary simplification with Garland quadric decimation (quad_mesh_simplify).")
                resampled_mesh = quadric_decimation_garland(resampled_mesh, target_vertices)
                logger.info("Vertices after Garland pass: %d; broken_faces=%s",
                            resampled_mesh.vertices.shape[0], trimesh.repair.broken_faces(resampled_mesh))
            else:
                logger.debug("Skipping Garland decimation because vertex count is far from target.")


        # save mesh as a.obj file
        broken_faces = trimesh.repair.broken_faces(resampled_mesh)
        logger.info("Broken faces after quadric decimation: %s", broken_faces)
        if len(broken_faces) > 0:
            logger.info("Filling holes in resampled mesh.")
            trimesh.repair.fill_holes(resampled_mesh)
            if use_mesh_fix:
                logger.info("Applying mesh_fix (pymeshfix) to resampled mesh.")
                resampled_mesh = mesh_fix(resampled_mesh)
                broken_faces = trimesh.repair.broken_faces(resampled_mesh)
                # apply quadratic decimation again to reach target vertices
                if not within_range(resampled_mesh.vertices.shape[0]) and len(broken_faces) == 0:
                    logger.info("Adding vertices by edge split to reach target after mesh_fix.")
                    resampled_mesh = add_vertices_by_edge_split(resampled_mesh, target_vertices)
                    logger.info("Vertex count after edge split: %d", resampled_mesh.vertices.shape[0])
                    bf = trimesh.repair.broken_faces(resampled_mesh)
                    logger.debug("Broken faces after edge split: %s", bf)
            broken_faces = trimesh.repair.broken_faces(resampled_mesh)
            logger.info("Broken faces after hole filling/fixing: %s", broken_faces)

        broken_faces = trimesh.repair.broken_faces(resampled_mesh)
        if len(broken_faces) == 0 and within_range(resampled_mesh.vertices.shape[0]):
            logger.info("Mesh fixed: no holes and within vertex tolerance (vertices=%d, faces=%d).",
                        resampled_mesh.vertices.shape[0], resampled_mesh.faces.shape[0])
            no_holes = True
            attempts = 21
            resampled_mesh.export(output_filename)
        else:
            aggressivity = max(1, aggressivity - 1)
            logger.info("Retrying: current vertices=%d, broken_faces=%s, next aggressivity=%d",
                        resampled_mesh.vertices.shape[0], broken_faces, aggressivity)
        attempts += 1
        if attempts > 20:
            if not no_holes:
                logger.error("Exceeded maximum attempts (%d) attempting to fix mesh: %s", attempts, mesh_filename)
                raise ValueError(f"failed to fix mesh {mesh_filename}")
            break

    return resampled_mesh
