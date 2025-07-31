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
#
# if __name__ == "__main__":
#     # Get input and output paths from command-line arguments
#     argv = sys.argv
#     inputpath = argv[argv.index("--") + 1]  # Input path after '--'
#     outputpath = argv[argv.index("--") + 2]  # Output path after '--'
#     num_vertex = argv[argv.index("--") + 3]
#     assert num_vertex.isdigit(), "Number of vertices must be an integer"
#
#     # Call the remesh function
#     remesh(inputpath=inputpath, outputpath=outputpath, n=int(num_vertex))
#     exit(0)

import bpy
import os

# Path to your OBJ file (change this!)
input_path = "/home/sauron/Documents/Phd/code/hippo-spharm/examples/airplane_0654.obj"
output_path = "/home/sauron/Documents/Phd/code/hippo-spharm/examples/airplane_remesh.obj"
target_vertex_count = 5000  # Desired number of vertices (adjust as needed)

# === CLEAN SCENE ===
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# === IMPORT OBJ ===
bpy.ops.import_scene.obj(filepath=input_path)
obj = bpy.context.selected_objects[0]
bpy.context.view_layer.objects.active = obj

# === COMPUTE VOXEL SIZE BASED ON DESIRED VERTICES ===
# Get bounding box volume
bbox = obj.bound_box
dims = [bbox[6][i] - bbox[0][i] for i in range(3)]
volume = abs(dims[0] * dims[1] * dims[2])
if volume == 0:
    raise ValueError("Object has zero volume — check for flat or corrupted geometry.")

# Estimate voxel size to reach target vertices
# Empirical formula: voxel_size ~ cube_root(volume / target_vertex_count)
voxel_size = (volume / target_vertex_count) ** (1/3)
voxel_size = max(0.0001, min(voxel_size, 1.0))  # Clamp to avoid extreme values

print(f"Estimated voxel size: {voxel_size:.6f} for ~{target_vertex_count} vertices")

# === REMESH ===
remesh_mod = obj.modifiers.new(name="Remesh", type='REMESH')
remesh_mod.mode = 'VOXEL'
remesh_mod.voxel_size = voxel_size
remesh_mod.use_smooth_shade = True

# Apply modifier
bpy.ops.object.modifier_apply(modifier=remesh_mod.name)

# === REPORT FINAL VERTEX COUNT ===
final_vertex_count = len(obj.data.vertices)
print(f"Remeshed object has {final_vertex_count} vertices.")

# === EXPORT ===
bpy.ops.export_scene.obj(filepath=output_path, use_selection=True)