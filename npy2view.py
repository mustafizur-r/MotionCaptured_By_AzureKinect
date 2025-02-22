import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import keyboard  # Requires `pip install keyboard`

# Load Kinect motion data (update the path)
npy_file = "D:/Mocap Data Kinect/NpyVisual/First.npy"  # Update this path
motion_data = np.load(npy_file)

# Print the shape to verify
print("Loaded motion data with shape:", motion_data.shape)

# Kinect skeleton structure
kinect_skeleton_connections = [
    (0, 1),  # Pelvis to Left_Hip
    (1, 2),  # Left_Hip to Left_Knee
    (2, 3),  # Left_Knee to Left_Ankle
    (0, 5),  # Pelvis to Right_Hip
    (5, 6),  # Right_Hip to Right_Knee
    (6, 7),  # Right_Knee to Right_Ankle
    (0, 9),  # Pelvis to Spine
    (9, 10),  # Spine to Spine1
    (10, 11),  # Spine1 to Spine2
    (10, 12),  # Spine2 to Left_Shoulder
    (12, 13),  # Left_Shoulder to Left_Arm
    (13, 14),  # Left_Arm to Left_Elbow
    (14, 15),  # Left_Elbow to Left_Wrist
    (10, 16),  # Spine2 to Right_Shoulder
    (16, 17),  # Right_Shoulder to Right_Arm
    (17, 18),  # Right_Arm to Right_Elbow
    (18, 19),  # Right_Elbow to Right_Wrist
    (19, 20),  # Right_Wrist to Right_Hand
]

# Get total frames and joints
num_frames, num_joints, _ = motion_data.shape

# Setup Matplotlib 3D plot
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')

# Function to plot a single frame
def plot_frame(frame_idx):
    ax.clear()
    frame = motion_data[frame_idx]

    # Plot joints
    ax.scatter(frame[:, 0], frame[:, 1], frame[:, 2], c='r', marker='o', s=30)

    # Plot bones (connections)
    for joint1, joint2 in kinect_skeleton_connections:
        if joint1 < num_joints and joint2 < num_joints:  # Ensure indices are valid
            j1 = frame[joint1]
            j2 = frame[joint2]
            ax.plot([j1[0], j2[0]], [j1[1], j2[1]], [j1[2], j2[2]], 'b-', linewidth=2)

    # Set labels and limits
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Frame {frame_idx}")

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    plt.draw()
    plt.pause(0.05)

# Animate with 'Q' to exit
print("Press 'Q' to stop visualization...")
frame_idx = 0
while frame_idx < num_frames:
    if keyboard.is_pressed("q"):  # Check if 'Q' is pressed
        print("Stopping visualization...")
        break

    plot_frame(frame_idx)
    time.sleep(0.05)  # Adjust speed
    frame_idx += 1

plt.close()  # Close the plot when exiting
