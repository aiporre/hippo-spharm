import bpy
import bmesh
import sys

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
    import bpy
    import bmesh

    # Clear existing objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # Import mesh
    bpy.ops.import_scene.obj(filepath=inputpath)

    # Get the imported object
    obj = bpy.context.selected_objects[0]
    bpy.context.view_layer.objects.active = obj

    # Apply scale and rotation
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)

    # Switch to edit mode and fix common issues
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.remove_doubles()
    bpy.ops.mesh.fill_holes()
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')

    # Apply remeshing
    bpy.ops.object.modifier_add(type='REMESH')
    obj.modifiers["Remesh"].mode = 'QUAD'
    obj.modifiers["Remesh"].use_remove_disconnected = True
    obj.modifiers["Remesh"].target_vertices = n
    bpy.ops.object.modifier_apply(modifier="Remesh")

    # Apply smoothing if requested
    if smooth_iterations > 0:
        bpy.ops.object.modifier_add(type='SMOOTH')
        obj.modifiers["Smooth"].iterations = smooth_iterations
        bpy.ops.object.modifier_apply(modifier="Smooth")

    # Export the remeshed model
    bpy.ops.export_scene.obj(
        filepath=outputpath,
        use_selection=True,
        use_materials=False,
        use_triangles=True
    )

    print(f"Remeshed model saved to {outputpath}")
    print(f"Vertex count: {len(obj.data.vertices)}")

if __name__ == "__main__":
    # Get input and output paths from command-line arguments
    argv = sys.argv
    inputpath = argv[argv.index("--") + 1]  # Input path after '--'
    outputpath = argv[argv.index("--") + 2]  # Output path after '--'

    # Call the remesh function
    remesh(inputpath=inputpath, outputpath=outputpath)