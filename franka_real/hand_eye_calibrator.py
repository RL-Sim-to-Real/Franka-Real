import rospy
import numpy as np
import cv2
import pyrealsense2 as rs
from franka_interface import ArmInterface
from scipy.spatial.transform import Rotation as R
import json
import os

from franka_real.FrankaPickCubeCartesian import FrankaPickCubeCartesian

# Chessboard settings
PATTERN_SIZE = (9, 6)   # inner corners
SQUARE_SIZE = 0.025     # meters (25mm)

# Camera intrinsics (use rs.intrinsics or calibration file)
# You can also fetch from the RealSense API directly
def get_camera_intrinsics(profile):
    intr = profile.as_video_stream_profile().intrinsics
    K = np.array([[intr.fx, 0, intr.ppx],
                  [0, intr.fy, intr.ppy],
                  [0, 0, 1]])
    dist = np.array(intr.coeffs)
    return K, dist

def make_homog(Rmat, tvec):
    T = np.eye(4)
    T[:3, :3] = Rmat
    T[:3, 3] = tvec.flatten()
    return T

def detect_chessboard(color_image, camera_matrix, dist_coeffs):
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, PATTERN_SIZE, None)
    if not ret:
        return None

    # Refine corner locations
    corners2 = cv2.cornerSubPix(
        gray, corners, (11,11), (-1,-1),
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    )

    # Prepare 3D points of chessboard in board frame
    objp = np.zeros((np.prod(PATTERN_SIZE), 3), np.float32)
    objp[:,:2] = np.indices(PATTERN_SIZE).T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    # SolvePnP to get pose
    ret, rvec, tvec = cv2.solvePnP(objp, corners2, camera_matrix, dist_coeffs)
    if not ret:
        return None

    Rmat, _ = cv2.Rodrigues(rvec)
    T_camera_board = make_homog(Rmat, tvec)
    return T_camera_board, corners2

def save_data(filename, dataset):
    with open(filename, "w") as f:
        json.dump(dataset, f, indent=2)

def matrix_from_quaternion(q):
    scalar_last = (q[1], q[2], q[3], q[0])
    matrix = R.from_quat(scalar_last).as_matrix()
    return matrix

def collect_calibration_dataset(use_stored_poses=False):
    # rospy.init_node("handeye_data_collector")
    if use_stored_poses:
        saved_T_base_ee, _, saved_joint_positions = load_dataset(os.path.expanduser("handeye_dataset_orig.json"))

    # --- Franka ---
    # arm = ArmInterface()
    env = FrankaPickCubeCartesian()

    # --- RealSense ---
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    K, dist = get_camera_intrinsics(profile.get_stream(rs.stream.color))

    dataset = []  # store dicts with base->ee and camera->board

    # Define some test end-effector poses (relative moves)
    # You can add more poses manually
    # target_positions = [
    #     [0.4, 0.0, 0.3],
    #     [0.35, 0.1, 0.25],
    #     [0.35, -0.1, 0.25],
    #     [0.4, 0.0, 0.2],
    #     [0.45, 0.05, 0.25],
    # ]
    q_down = [1, 0, 0, 0]  # adjust orientation to point camera at board

    rospy.loginfo("Starting calibration capture loop...")
    for i in range(30):
        # Move robot
        if not use_stored_poses:
            input(f'{i}: move the robot to new position and press Enter to capture')
        else:
            # pos = saved_T_base_ee[i][:3, 3]
            # ori = np.array(env.euler_from_quaternion(R.from_matrix(saved_T_base_ee[i][:3, :3]).as_quat(scalar_first=True)))
            # # arm.move_to_cartesian_pose(pos, ori=ori, use_moveit=False)
            # env.move_to_pose_ee(pos, ref_ee_angle=ori, pose_vel_limit=0.05)
            env.move_to_joint_positions(saved_joint_positions[i])

        # Get frames
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        color_image = np.asanyarray(color_frame.get_data())

        # Detect chessboard
        T_camera_board, corners2 = detect_chessboard(color_image, K, dist)
        if T_camera_board is None:
            rospy.logwarn(f"No chessboard found at pose {i}")
            continue

        # Get robot pose
        ee_pose = env.robot.endpoint_pose()  # has .translation and .quaternion
        ee_quaternion = [ee_pose['orientation'].w, ee_pose['orientation'].x,
                         ee_pose['orientation'].y, ee_pose['orientation'].z]
        # R_ee = R.from_quat(ee_pose.quaternion).as_matrix()
        R_ee = matrix_from_quaternion(ee_quaternion)
        T_base_ee = make_homog(R_ee, ee_pose['position'])
        joint_positions = env.get_state()['joints']

        # Store
        dataset.append({
            "T_base_ee": T_base_ee.tolist(),
            "T_camera_board": T_camera_board.tolist(),
            "joint_position": joint_positions.tolist(),
        })

        # Show feedback
        cv2.drawChessboardCorners(color_image, PATTERN_SIZE, corners2, True)
        # cv2.drawChessboardCorners(color_image, PATTERN_SIZE, 
        #                           cv2.cornerSubPix(
        #                               cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY),
        #                               np.array(T_camera_board[:,:2]),
        #                               (11,11), (-1,-1), 
        #                               (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,30,0.001)),
        #                           True)
        cv2.imshow("Chessboard Detection", color_image)
        cv2.waitKey(500)
        rospy.sleep(3.0)

    pipeline.stop()
    cv2.destroyAllWindows()

    # Save dataset
    out_file = os.path.expanduser("handeye_dataset.json")
    save_data(out_file, dataset)
    rospy.loginfo(f"Saved dataset with {len(dataset)} samples to {out_file}")

def load_dataset(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    T_base_ee = [np.array(d["T_base_ee"]) for d in data]
    T_camera_board = [np.array(d["T_camera_board"]) for d in data]
    joint_position = [np.array(d["joint_position"]) for d in data]
    return T_base_ee, T_camera_board, joint_position

def relative_motion(poses):
    """Compute relative motions between consecutive transforms."""
    rel = []
    for i in range(len(poses)-1):
        T1 = poses[i]
        T2 = poses[i+1]
        rel.append(np.linalg.inv(T1) @ T2)
    return rel

def calculate_T_ee_camera():
    dataset_file = os.path.expanduser("handeye_dataset.json")
    T_base_ee, T_camera_board, _ = load_dataset(dataset_file)

    if len(T_base_ee) < 2:
        print("Not enough samples! Collect at least 10–15.")
        return

    # Compute relative motions
    robot_motions = relative_motion(T_base_ee)       # base->ee
    camera_motions = relative_motion(T_camera_board) # camera->board

    # R_gripper2base = [m[:3,:3] for m in robot_motions]
    # t_gripper2base = [m[:3,3] for m in robot_motions]
    # R_target2cam  = [m[:3,:3] for m in camera_motions]
    # t_target2cam  = [m[:3,3] for m in camera_motions]

    R_gripper2base = [m[:3,:3] for m in T_base_ee]
    t_gripper2base = [m[:3,3] for m in T_base_ee]
    R_target2cam  = [m[:3,:3] for m in T_camera_board]
    t_target2cam  = [m[:3,3] for m in T_camera_board]

    # Run OpenCV hand-eye calibration
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base,
        R_target2cam, t_target2cam
    )

    T_ee_camera = np.eye(4)
    T_ee_camera[:3,:3] = R_cam2gripper
    T_ee_camera[:3,3]  = np.squeeze(t_cam2gripper)

    print("Estimated T_ee_camera (homogeneous):")
    print(T_ee_camera)

    # Save to file
    out_file = os.path.expanduser("T_ee_camera.json")
    with open(out_file, "w") as f:
        json.dump(T_ee_camera.tolist(), f, indent=2)
    print(f"Saved calibration to {out_file}")

if __name__ == "__main__":
    # collect_calibration_dataset()
    collect_calibration_dataset(use_stored_poses=True)
    calculate_T_ee_camera()

