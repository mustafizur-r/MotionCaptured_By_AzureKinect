import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# Define the connections between joints for the HumanML3D skeleton
HUMANML3D_CONNECTIONS = [
    (0, 1),  # Pelvis to Left_Hip
    (1, 2),  # Left_Hip to Left_Knee
    (2, 3),  # Left_Knee to Left_Ankle
    (0, 4),  # Pelvis to Right_Hip
    (4, 5),  # Right_Hip to Right_Knee
    (5, 6),  # Right_Knee to Right_Ankle
    (0, 7),  # Pelvis to Spine
    (7, 8),  # Spine to Spine1
    (8, 9),  # Spine1 to Spine2
    (9, 10), # Spine2 to Neck
    (10, 11),# Neck to Head
    (9, 12), # Spine2 to Left_Shoulder
    (12, 13),# Left_Shoulder to Left_Arm
    (13, 14),# Left_Arm to Left_Elbow
    (14, 15),# Left_Elbow to Left_Wrist
    (9, 16), # Spine2 to Right_Shoulder
    (16, 17),# Right_Shoulder to Right_Arm
    (17, 18),# Right_Arm to Right_Elbow
    (18, 19),# Right_Elbow to Right_Wrist
    (19, 20), # Right_Wrist to Right_Hand
    (20,21)
]

def plot_skeleton(joint_positions, ax, debug=False):
    """
    Plots the 3D skeleton for a single frame with possible axis corrections.

    Parameters:
        joint_positions (np.ndarray): (22, 3) array of joint positions.
        ax (Axes3D): Matplotlib 3D axis.
        debug (bool): If True, shows the raw positions without corrections.
    """
    if debug:
        positions_to_plot = joint_positions
    else:
        positions_to_plot = joint_positions.copy()
        positions_to_plot[:, [1, 2]] = joint_positions[:, [2, 1]]  # Swap Y and Z
        positions_to_plot[:, 2] *= -1  # Invert Z-axis

    # Plot joints
    ax.scatter(
        positions_to_plot[:, 0],
        positions_to_plot[:, 1],
        positions_to_plot[:, 2],
        color="blue",
        s=20,
    )

    # Plot connections
    for joint1, joint2 in HUMANML3D_CONNECTIONS:
        x = [positions_to_plot[joint1, 0], positions_to_plot[joint2, 0]]
        y = [positions_to_plot[joint1, 1], positions_to_plot[joint2, 1]]
        z = [positions_to_plot[joint1, 2], positions_to_plot[joint2, 2]]
        ax.plot(x, y, z, color="black")

    max_range = np.array([
        positions_to_plot[:, 0].max() - positions_to_plot[:, 0].min(),
        positions_to_plot[:, 1].max() - positions_to_plot[:, 1].min(),
        positions_to_plot[:, 2].max() - positions_to_plot[:, 2].min()
    ]).max() / 2.0

    mid_x = (positions_to_plot[:, 0].max() + positions_to_plot[:, 0].min()) * 0.5
    mid_y = (positions_to_plot[:, 1].max() + positions_to_plot[:, 1].min()) * 0.5
    mid_z = (positions_to_plot[:, 2].max() + positions_to_plot[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

def visualize_npy_data(file_path):
    """
    Visualizes skeleton data stored in .npy file and orients the avatar to face forward.

    Parameters:
        file_path (str): Path to the .npy file.
    """
    def orient_skeleton(joints):
        """
        Adjust the global orientation of the skeleton to face forward (Z-axis).
        """
        # Define the rotation matrix to align the skeleton to face the Z-axis
        forward_rotation = R.from_euler('y', 90, degrees=True).as_matrix()  # Rotate 90° around Y-axis

        # Apply the rotation to all joint positions
        return np.dot(joints, forward_rotation.T)

    # Load the data
    data = np.load(file_path)

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for frame_idx, joint_positions in enumerate(data):
        ax.clear()  # Clear the plot for the next frame

        # Adjust orientation
        oriented_positions = orient_skeleton(joint_positions)

        # Plot the skeleton
        plot_skeleton(oriented_positions, ax)

        # Add title and frame counter
        plt.title(f"Skeleton Visualization - Frame {frame_idx + 1}")

        # Add pause for animation
        plt.pause(0.1)

    # Final show
    plt.show()

if __name__ == "__main__":
    # Specify the file path to the .npy file
    npy_file = "humanml3d_kinect_data.npy"

    # Call the visualization function
    visualize_npy_data(npy_file)
