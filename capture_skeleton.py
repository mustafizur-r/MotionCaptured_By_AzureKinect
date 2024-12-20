import numpy as np
import cv2
import pykinect_azure as pykinect
import time

# Initialize the PyKinect library with body tracking enabled
pykinect.initialize_libraries(track_body=True)

# Start the device and body tracking module
device = pykinect.start_device()
tracker = pykinect.start_body_tracker()

print("Kinect sensor initialized.")

# HumanML3D 22-joint mapping from Azure Kinect 32 joints
HUMANML3D_JOINTS = [
    0,  # Pelvis (Root)
    19,  # Left_Hip
    20,  # Left_Knee
    21,  # Left_Ankle
    23,  # Right_Hip
    24,  # Right_Knee
    25,  # Right_Ankle
    2,   # Spine
    3,   # Spine1
    4,   # Spine2
    27,  # Neck
    28,  # Head
    5,   # Left_Shoulder
    6,   # Left_Arm
    7,   # Left_Elbow
    8,   # Left_Wrist
    12,  # Right_Shoulder
    13,  # Right_Arm
    14,  # Right_Elbow
    15,  # Right_Wrist
    16,  # Left_Hand
    17,  # Right_Hand
]

def capture_kinect_data_and_save(num_frames, output_filename):
    """
    Captures body tracking data from the Kinect sensor and displays the video feed.
    Saves the 22-joint HumanML3D-compatible data in .npy format.
    """
    data = []  # To store joint positions for each frame
    frame_count = 0

    while frame_count < num_frames:
        # Update the device to get the latest capture
        capture = device.update()

        # Get the color image from the capture
        ret, color_image = capture.get_color_image()

        if not ret:
            continue

        # Get body tracking results
        body_frame = tracker.update()

        if body_frame.get_num_bodies() > 0:
            for body_index in range(body_frame.get_num_bodies()):
                # Get the body skeleton
                body = body_frame.get_body(body_index)

                # Extract 22 joints from Azure Kinect's 32 joints
                joint_positions = np.zeros((22, 3))  # Initialize for 22 joints
                for i, azure_joint_id in enumerate(HUMANML3D_JOINTS):
                    joint = body.joints[azure_joint_id]
                    joint_positions[i] = [joint.position.x, joint.position.y, joint.position.z]

                # Normalize the root joint (Pelvis) as origin
                root_position = joint_positions[0].copy()
                joint_positions -= root_position

                data.append(joint_positions)

                # Draw joint positions on the color image
                for joint_id, azure_joint_id in enumerate(HUMANML3D_JOINTS):
                    joint = body.joints[azure_joint_id]
                    joint_position = joint.position
                    # Transform 3D joint positions to 2D image coordinates
                    x = int(joint_position.x * 100 + color_image.shape[1] // 2)
                    y = int(-joint_position.y * 100 + color_image.shape[0] // 2)
                    cv2.circle(color_image, (x, y), 5, (0, 255, 0), -1)

        # Display the video feed
        cv2.imshow("Kinect Video Feed", color_image)

        # Increment frame count
        frame_count += 1

        # Break if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Sleep to maintain roughly 30 FPS capture rate
        time.sleep(0.033)

    # Convert list to numpy array
    data = np.array(data)

    # Save the data as .npy
    np.save(output_filename, data)
    print(f"Body tracking data saved to {output_filename}")

    # Cleanup
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Specify the number of frames to capture and the output file name
    num_frames = 90  # Adjust as needed
    output_file = "humanml3d_kinect_data.npy"
    capture_kinect_data_and_save(num_frames=num_frames, output_filename=output_file)
