import bpy
import numpy as np

# Load the FBX file
fbx_file = "D:/Mocap Data Kinect/Fbx Converter/First.fbx"
bpy.ops.import_scene.fbx(filepath=fbx_file)

# Get the armature (skeleton)
armature = None
for obj in bpy.data.objects:
    if obj.type == 'ARMATURE':
        armature = obj
        break

if armature is None:
    raise Exception("No skeleton found in FBX")

# Extract keyframe motion data
frame_start = bpy.context.scene.frame_start
frame_end = bpy.context.scene.frame_end
motion_data = []

for frame in range(frame_start, frame_end + 1):
    bpy.context.scene.frame_set(frame)
    frame_pose = []
    
    for bone in armature.pose.bones:
        location = armature.matrix_world @ bone.head
        frame_pose.append([location.x, location.y, location.z])
    
    motion_data.append(frame_pose)

motion_data = np.array(motion_data)

# Save as .npy
npy_file = "D:/Mocap Data Kinect/Fbx Converter/First.npy"
np.save(npy_file, motion_data)
print(f"Motion data saved to {npy_file}")
