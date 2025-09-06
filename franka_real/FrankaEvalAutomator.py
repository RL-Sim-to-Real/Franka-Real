import time
import rospy
import numpy as np
import cv2
import pyrealsense2 as rs

class FrankaEvalAutomator:
    def __init__(self):
        pass
        # self.pipeline = rs.pipeline()
        # config = rs.config()
        # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        # self.pipeline.start(config)
        # self.align = rs.align(rs.stream.color)

    def reset_cube(self, depth_frame, color_frame):
        cube_pos = None
        while cube_pos is None:
            cube_pos = self.get_red_cube_position(depth_frame, color_frame)
        print("Cube position (x, y, z) in robot base frame:", cube_pos)
        return cube_pos

    def get_red_cube_position(self, depth_frame, color_frame):
        # # Get frames
        # frames = self.pipeline.wait_for_frames()
        # frames = self.align.process(frames)
        # color_frame = frames.get_color_frame()
        # depth_frame = frames.get_depth_frame()

        # if not color_frame or not depth_frame:
        #     return None

        color_image = np.asanyarray(color_frame.get_data())

        # Convert to HSV
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

        # Red thresholds (wrap-around in HSV)
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 | mask2
        # cv2.imshow("mask 1", mask1)
        # cv2.imshow("mask 2", mask2)
        # cv2.imshow("combined mask", mask)
        # cv2.waitKey(1)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print('no contours found')
            return None

        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M["m00"] == 0:
            print('m00 is zero')
            return None

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # get cube's yaw angle
        rect = cv2.minAreaRect(c)
        angle = rect[2]
        w, h = rect[1]
        angle_180 = angle + 90 if w < h else angle
        angle_180 = angle_180 % 180
        aspect = min(w, h) / max(w, h)
        is_square = 1.0 if aspect >= 0.8 else 0.0

        # Draw contour, centroid, and rect's angle on the color image for visualization
        cv2.drawContours(color_image, [c], -1, (0, 255, 0), 2)
        cv2.circle(color_image, (cx, cy), 5, (255, 0, 0), -1)
        box = cv2.boxPoints(rect)
        box = np.round(box).astype(np.intp)
        cv2.drawContours(color_image, [box], 0, (255, 0, 255), 2)
        cv2.putText(color_image, f'Angle_180: {angle_180:.2f}', (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("Color Image with Contour", color_image)
        cv2.waitKey(1)

        # Depth intrinsics
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        depth = depth_frame.get_distance(cx, cy)  # in meters
        # print(f"Depth at centroid: {depth}")

        if depth == 0:
            # print('depth is zero')
            return None

        # Pixel → Camera coordinates
        point_cam = rs.rs2_deproject_pixel_to_point(depth_intrin, [cx, cy], depth)
        point_cam = np.array([point_cam[0], point_cam[1], point_cam[2], 1.0, angle, angle_180, is_square], dtype=np.float32)
        # point_cam = np.array([point_cam[1], -point_cam[0], -point_cam[2], 1.0], dtype=np.float32)
        return point_cam

        # # Transform → Robot base frame
        # point_base = T_base_camera @ point_cam
        # return point_base[:3]  # x, y, z in robot base frame
