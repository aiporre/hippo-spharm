import bpy
import bmesh
import sys
import os
import math
# import bpy
# import bmesh
# import sys
#
# def remesh(inputpath, outputpath, n=6890, smooth_iterations=2):
#     """
#     Remesh an .obj file using Blender's bmesh library.
#
#     Parameters
#     ----------
#     inputpath : str
#         Path to input .obj file
#     outputpath : str
#         Path to save the remeshed .obj file
#     n : int, default=6890
#         Target number of vertices
#     smooth_iterations : int, default=2
#         Number of smoothing iterations
#
#     Returns
#     -------
#     None
#     """
#     import bpy
#     import bmesh
#
#     # Clear existing objects
#     bpy.ops.object.select_all(action='SELECT')
#     bpy.ops.object.delete()
#
#     # Import mesh
#     bpy.ops.import_scene.obj(filepath=inputpath)
#
#     # Get the imported object
#     obj = bpy.context.selected_objects[0]
#     bpy.context.view_layer.objects.active = obj
#
#     # Apply scale and rotation
#     bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
#
#     # Switch to edit mode and fix common issues
#     bpy.ops.object.mode_set(mode='EDIT')
#     bpy.ops.mesh.select_all(action='SELECT')
#     bpy.ops.mesh.remove_doubles()
#     bpy.ops.mesh.fill_holes()
#     bpy.ops.mesh.normals_make_consistent(inside=False)
#     bpy.ops.object.mode_set(mode='OBJECT')
#
#     # Apply remeshing
#     bpy.ops.object.modifier_add(type='REMESH')
#     obj.modifiers["Remesh"].mode = 'QUAD'
#     obj.modifiers["Remesh"].use_remove_disconnected = True
#     obj.modifiers["Remesh"].target_vertices = n
#     bpy.ops.object.modifier_apply(modifier="Remesh")
#
#     # Apply smoothing if requested
#     if smooth_iterations > 0:
#         bpy.ops.object.modifier_add(type='SMOOTH')
#         obj.modifiers["Smooth"].iterations = smooth_iterations
#         bpy.ops.object.modifier_apply(modifier="Smooth")
#
#     # Export the remeshed model
#     bpy.ops.export_scene.obj(
#         filepath=outputpath,
#         use_selection=True,
#         use_materials=False,
#         use_triangles=True
#     )
#
#     print(f"Remeshed model saved to {outputpath}")
#     print(f"Vertex count: {len(obj.data.vertices)}")

def remesh(inputpath, outputpath, n=6890, smooth_iterations=2):
    """
    Remesh an .obj file using Blender's bmesh library.

    Parameters
    ----------
    inputpath : str
        Path to input .obj file
    outputpath : str
        Path to save the remeshed .obj file
    n : int, default=6890
        Target number of vertices
    smooth_iterations : int, default=2
        Number of smoothing iterations

    Returns
    -------
    None
    """


    # Path to your OBJ file (change this!)
    input_path = inputpath #"/home/sauron/Documents/Phd/code/hippo-spharm/examples/airplane_0654.obj"
    output_path = outputpath #"/home/sauron/Documents/Phd/code/hippo-spharm/examples/airplane_remesh.obj"
    target_vertex_count = n #3000  # Desired number of vertices (adjust as needed)

    SCALE_TO_UNIT_VOLUME = True  # If True, scale object so bounding-box volume ~= 1.0

    # === CLEAN SCENE ===
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # === IMPORT OBJ ===
    bpy.ops.import_scene.obj(filepath=input_path)
    # assume the imported object is selected
    obj = bpy.context.selected_objects[0]
    bpy.context.view_layer.objects.active = obj

    # Ensure we are dealing with a mesh
    if obj.type != 'MESH':
        raise TypeError("Imported object is not a mesh")

    # Helper to compute bounding-box volume (using object.dimensions which accounts for scale)
    def bbox_volume(obj):
        # Ensure evaluated data updated
        bpy.context.view_layer.update()
        dims = obj.dimensions[:]  # Vector of (x, y, z)
        return abs(dims[0] * dims[1] * dims[2])

    # === COMPUTE ORIGINAL VOLUME ===
    original_volume = bbox_volume(obj)
    print("Original bounding-box volume:", original_volume)

    if original_volume == 0:
        raise ValueError("Object has zero bounding-box volume — check for flat or corrupted geometry.")

    # === SCALE TO UNIT VOLUME (if requested) ===
    if SCALE_TO_UNIT_VOLUME:
        scale_factor = (10.0 / original_volume) ** (1.0 / 3.0)
        # Clamp the scale factor to avoid extreme resizes
        scale_factor = max(1e-6, min(scale_factor, 1e6))
        print(f"Applying uniform scale factor: {scale_factor:.6e} to make volume ~1.0")
        # Apply scale to object
        obj.scale = (scale_factor, scale_factor, scale_factor)
        # Apply the scale to mesh data so obj.scale becomes (1,1,1) and vertices are scaled
        bpy.context.view_layer.update()
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

        # Recompute volume after transform
        new_volume = bbox_volume(obj)
        print("New bounding-box volume after scaling:", new_volume)
    else:
        new_volume = original_volume

    # Safety check
    if new_volume == 0:
        raise ValueError("New volume is zero after scaling — aborting")

    # === ESTIMATE VOXEL SIZE BASED ON DESIRED VERTICES ===
    # Empirical formula: voxel_size ~ cube_root(volume / target_vertex_count)
    voxel_size = 0.27*(new_volume / target_vertex_count) ** (1.0 / 3.0)
    voxel_size = max(0.000001, min(voxel_size, 1.0))  # Clamp to avoid extreme values
    print(f"Estimated voxel size: {voxel_size:.6f} for ~{target_vertex_count} vertices (volume={new_volume:.6f})")

    # === REMESH ===
    # Ensure object is active and selected
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    remesh_mod = obj.modifiers.new(name="Remesh", type='REMESH')
    remesh_mod.mode = 'VOXEL'
    remesh_mod.voxel_size = voxel_size
    remesh_mod.use_smooth_shade = True

    # Apply modifier
    bpy.ops.object.modifier_apply(modifier=remesh_mod.name)

    # === REPORT FINAL VERTEX COUNT ===
    final_vertex_count = len(obj.data.vertices)
    print(f"Remeshed object has {final_vertex_count} vertices.")

    # Apply smoothing if requested
    if smooth_iterations > 0:
        bpy.ops.object.modifier_add(type='SMOOTH')
        obj.modifiers["Smooth"].iterations = smooth_iterations
        bpy.ops.object.modifier_apply(modifier="Smooth")

    # === EXPORT ===
    # Ensure only the object is selected for export
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.ops.export_scene.obj(filepath=output_path, use_selection=True)

    print("Exported remeshed object to:", output_path)

#
if __name__ == "__main__":
    # Get input and output paths from command-line arguments
    argv = sys.argv
    inputpath = argv[argv.index("--") + 1]  # Input path after '--'
    outputpath = argv[argv.index("--") + 2]  # Output path after '--'
    num_vertex = argv[argv.index("--") + 3]
    assert num_vertex.isdigit(), "Number of vertices must be an integer"

    # Call the remesh function
    remesh(inputpath=inputpath, outputpath=outputpath, n=int(num_vertex))
    exit(0)

