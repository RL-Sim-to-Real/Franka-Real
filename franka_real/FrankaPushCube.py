from tkinter import S
import numpy as np

import time
from gymnasium import spaces

import gymnasium as gym
from gymnasium.core import ActionWrapper
import numpy as np
from gymnasium import spaces
import os

from numpy.core.defchararray import count
import rospy
from PIL import Image
import math
from collections import deque

from franka_utils import *

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import animation
import time
import logging
import cv2

from franka_interface import ArmInterface, RobotEnable, GripperInterface
from franka_tools import FrankaControllerManagerInterface
from franka_tools import ControllerParamConfigClient
# ids camera lib for use of IDS ueye cameras.
# https://www.ids-imaging.us/files/downloads/ids-peak/readme/ids-peak-linux-readme-1.2_EN.html
#import ids
import time
import signal
import multiprocessing
from gymnasium.spaces import Box as GymBox
from scipy.spatial.transform import Rotation as R
import quaternion
import subprocess

from .franka_analytic_ik import franka_analytic_ik
from .metric_logger import MetricLogger

ARM_VEL_LIMITS = np.array([2.61799, 2.61799, 2.61799, 2.61799, 3.14159, 3.14159, 3.14159, 0])

## Store EE orientation in quaternion format

EE_ORI = [
    0.99741769, -0.07031947,  0.01392346,
   -0.06985664, -0.99705542, -0.03132564,
    0.01608557,  0.03027268, -0.99941224
]

R_matrix = np.array(EE_ORI).reshape((3, 3))
rotation = R.from_matrix(R_matrix)
quat = rotation.as_quat()

EE_ORI = quaternion.quaternion(quat[3], quat[0], quat[1], quat[2])

class FrankaPushCube(gym.Env):
    """
    Gym env for the real franka robot. Set up to perform the placement of a peg that starts in the robots hand into a slot
    """
    def __init__(self, dt=0.04, episode_length=8, camera_index=0, control_mode='joint_velocity', seed=9, size_tol=0.45, render=False, k=None, d=None):
        np.random.seed(seed)
        self.DT= dt
        self.dt = dt
        self.ep_time = 0
        self.control_mode = control_mode
        self.max_episode_duration = episode_length # in seconds
        signal.signal(signal.SIGINT, self.exit_handler)
        # config_file = os.path.join(os.path.dirname(__file__), os.pardir, 'reacher.yaml')
        self.configs = configure('/home/chemist/Desktop/ICRA2026/Franka-Real/franka_real/reacher.yaml')
        self.conf_exp = self.configs['experiment']
        self.conf_env = self.configs['environment']

        rospy.init_node("franka_robot_gym")
        print("stuck")
        self.init_joints_bound = self.conf_env['reset-bound']
        #self.target_joints = self.conf_env['target-bound']
        self.safe_bound_box = np.array(self.conf_env['safe-bound-box'])
        self.target_box = np.array(self.conf_env['target-box'])
        self.joint_angle_bound = np.array(self.conf_env['joint-angle-bound'])
        self.return_point = self.conf_env['return-point']
        self.out_of_boundary_flag = False
        self.joint_names = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']

        self.robot = ArmInterface(True)
        print(self.robot.joint_names())
        self.gripper = GripperInterface()
        # self.gripper.home_joints()
        force = 1e-6
        self.robot.set_collision_threshold(cartesian_forces=[force,force,force,force,force,force])
        self.robot.exit_control_mode(0.1)
        self.robot_status = RobotEnable()
        self.control_frequency = 1/dt
        self.rate = rospy.Rate(self.control_frequency)
        
        self._size_tol = size_tol

        self.ct = dt
        self.tv = time.time()


        self.joint_states_history = deque(np.zeros((5, 21)), maxlen=5)
        self.torque_history = deque(np.zeros((5, 7)), maxlen=5)
        self.last_action_history = deque(np.zeros((5, 7)), maxlen=5)
        self.time_out_reward = False
        action_dim = 7
        self.prev_action = np.zeros(action_dim)

        self.max_time_steps = int(self.max_episode_duration / dt)

        self.previous_place_down = None

        self.joint_action_limit = 0.3

        joint_velocity_limits = np.array([1.0, 1.0, 1.0, 1.0, 1.5, 1.5, 1.5])
        self.action_space = spaces.Box(low=-joint_velocity_limits, high=joint_velocity_limits, dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)

        self.targets = [
            [0.696, -0.129, 0.2], 
            [0.489, -0.123, 0.2],
            [0.730, 0.167, 0.2],
            [0.484, 0.178, 0.2]
        ]

        self.episode_target = self.targets[0]
        self.total_timesteps = 0
        self.reset_ee_quaternion = [0,-1.,0,0]
        self.cmi = FrankaControllerManagerInterface()
        # self.robot_status.enable()
        self.controller_param_config_client = ControllerParamConfigClient(self.cmi.current_controller)
        self.controller_param_config_client.start()
        while self.controller_param_config_client.is_running is False:
            pass
        if k and d:
            self.controller_param_config_client.set_controller_gains(k, d)
            k,d = self.controller_param_config_client.get_controller_gains()
            print("Contrller gains set to  {k}, {d}".format(k=k, d=d))
        


    
        self.stale_counter = 0
        self.max_stale_steps = 100
        self.logger = MetricLogger()


    def _reset_stats(self):
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def monitor(self, reward, done, info):
        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        info['total'] = {'timesteps': self.total_timesteps}

        if done:
            info['episode'] = {}
            info['episode']['return'] = self.reward_sum
            info['episode']['length'] = self.episode_length
            info['episode']['duration'] = time.time() - self.start_time

            if hasattr(self, 'get_normalized_score'):
                info['episode']['return'] = self.get_normalized_score(
                    info['episode']['return']) * 100.0
        return info
        
    def reset(self):
        """
        reset robot to random pose
        Returns
        -------
        object
            Observation of the current state of this env in the format described by this envs observation_space.
        """
        self.episode_target = self.targets[np.random.randint(0, len(self.targets))]
        self.time_steps = 0
        self.ep_time = 0
        self.robot_status.enable()
        # stop the robot
        self.apply_joint_vel(np.zeros((7,)))

 
        self._reset_stats()

        self.reset_ee_quaternion = [0,-1.,0,0]
        
        # self.out_of_boundary_flag = False

        # smoothly_move_to_position_vel(self.robot, self.robot_status, target_pose, MAX_JOINT_VELs=1.3)
        # print("here", self.robot.endpoint_pose()["orientation"])
        # self.move_to_pose_ee(np.array([0.57, 0.0, 0.1]))

        obs = self.get_state()
        new_ee_mat = np.identity(4)
        # new_ee_mat[:3, :3] = self.ee_ori_mat
        new_ee_mat[:3, :3] = self.matrix_from_quaternion(self.ee_orientation)
        new_ee_mat[:3, 3] = np.array([0.57, 0.0, 0.2])
        q7 = obs['joints'][6]
        q_actual = obs['joints']
        q = franka_analytic_ik(new_ee_mat, q7, q_actual) # this is closer to sim
        self.move_to_joint_positions(q)


        # stop the robot
        self.apply_joint_vel(np.zeros((7,)))

        self.time_steps = 0

        self.tv = time.time()
        self.reset_time = time.time()

        self.grasped = False
        self.cur_step = 0 
        self.stale_counter = 0
        self.actuation_steps = 0
        self.prev_joints = None
        self.prev_joint_vels = 0.0
        self.prev_joint_accels = 0.0

        self._reset_stats()

        return self._get_end_effector_pos(), {}


    def get_robot_jacobian(self):
        return self.robot.zero_jacobian()
 
    def euler_from_quaternion(self,q):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        w,x,y,z = q        
        t0 = +2.0 * (w * x + y * z) 
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians

    def matrix_from_quaternion(self,q):
        scalar_last = (q[1], q[2], q[3], q[0])
        matrix = R.from_quat(scalar_last).as_matrix()
        return matrix

    def get_state(self):
        # get object state
        # self.obs_object = self.camera.get_state()
        
        # get robot states
        joint_angles = extract_values(self.robot.joint_angles(), self.joint_names)
        joint_velocitys = extract_values(self.robot.joint_velocities(), self.joint_names)
        # joint_efforts = extract_values(self.robot.joint_efforts(), self.joint_names)
        ee_pose = self.robot.endpoint_pose()
        ee_quaternion = [ee_pose['orientation'].w, ee_pose['orientation'].x,
                         ee_pose['orientation'].y, ee_pose['orientation'].z]

        self.last_action_history.append(self.prev_action)

        ee_height = ee_pose['position'][2]

        observation = {
            
            'last_action': self.prev_action,
            'joints': np.array(joint_angles),
            'joint_vels': np.array(joint_velocitys),
            'height': np.array([ee_height]),
            'ee_position': np.array(ee_pose['position']),
        }
        # print('orientation',ee_pose['orientation'])
        self.ee_position = ee_pose['position']
        # print(self.ee_position)
        self.ee_position_table = np.array([1.07-self.ee_position[0], 0.605-self.ee_position[1], self.ee_position[2]])
        self.ee_orientation = ee_quaternion
        #return observation['joints']
        return observation



    def out_of_boundaries(self):
        x, y, z = self.robot.endpoint_pose()['position']
        
        x_bound = self.safe_bound_box[0,:]
        y_bound= self.safe_bound_box[1,:]
        z_bound = self.safe_bound_box[2,:]
        if scalar_out_of_range(x, x_bound):
            # print('x out of bound, motion will be aborted! x {}'.format(x))
            return True
        if scalar_out_of_range(y, y_bound):
            # print('y out of bound, motion will be aborted! y {}'.format(y))
            return True
        if scalar_out_of_range(z, z_bound):
            # print('z out of bound, motion will be aborted!, z {}'.format(z))
            return True
        return False

    def apply_joint_vel(self, joint_vels):
        joint_vels = dict(zip(self.joint_names, joint_vels))
        self.robot.set_joint_velocities(joint_vels)        
        return True

    def log_metrics(self, dt=None):
        dt = self.dt if dt is None else dt
        obs = self.get_state()
        joint_accels = (obs['joint_vels'] - self.prev_joint_vels) / dt
        joint_jerks = (joint_accels - self.prev_joint_accels) / dt
        self.logger.log({
            't': self.actuation_steps,
            'has_collided': self.robot.has_collided(),
            'joint_jerks': np.linalg.norm(joint_jerks),
            'proprioception': np.concatenate([obs['joints'], obs['joint_vels']], axis=0).astype(np.float32),
        })
        # print(f'jerk norm: {np.linalg.norm(joint_jerks):.4f}')
        self.prev_joint_vels = obs['joint_vels']
        self.prev_joint_accels = joint_accels
        self.actuation_steps += 1

    def step(self, action, control_mode=None):
        control_mode = self.control_mode if control_mode is None else control_mode
        self.ep_time += self.dt
        self.cur_step += 1
        self.robot_status.enable()
        action_scale = 0.05
        if control_mode == 'cartesian_position' or control_mode == 'cartesian_velocity':
            # ee_pos = self._get_end_effector_pos()
            # target_xyz = 0.01 * action[:3] + ee_pos
            # target_xyz = np.array([target_xyz[0], \
            #                         np.clip(target_xyz[1], -0.4, 0.4), \
            #                         np.clip(target_xyz[2], 0.035, 0.2)]) # for safety
            # print(f"Target position: {target_xyz}, Current position: {ee_pos}")
            # self.move_to_target_xyz(target_xyz) # analytic ik
            self.move_tip_orient(action)
            

            # self.move_to_pose_ee(target_xyz) # ReLoD's ik
        elif control_mode == 'joint_position':
            joint_positions = self.get_state()['joints']
            target_joints = joint_positions + action[:7] * 0.03
            # target_joints = dict(zip(self.joint_names, target_joints))
            t0 = time.time()
            control_frequency, motion_duration = 2 / 0.04, 2
            # smoothly_move_to_position(self, self.robot, target_joints, control_frequency=control_frequency, motion_duration=motion_duration)
            # self.robot.set_joint_positions(target_joints)
            self.robot.set_joint_positions_velocities(target_joints, np.zeros((7,)))
            self.log_metrics()
            self.rate.sleep()
            print(f'Time taken to move: {time.time() - t0}')
            new_joints = self.get_state()['joints']
            # print('norm joint diff: ', np.linalg.norm(new_joints - joint_positions))
            if np.linalg.norm(new_joints - joint_positions) < 0.0001:
                self.stale_counter += 1
                if self.stale_counter > self.max_stale_steps:
                    curr_contr = self.cmi.current_controller
                    # print(curr_contr, self.cmi.is_running(curr_contr))
                    print('------------------------------------------------------ stale controller, restarting ', curr_contr)
                    self.cmi.stop_controller(curr_contr)
                    while self.cmi.is_running(curr_contr):
                        print('waiting for controller to stop')
                        time.sleep(1)
                    self.cmi.start_contreller(curr_contr)
                    self.stale_counter = 0
                    self.logger.pop(int(control_frequency * motion_duration))
                    # self.actuation_steps -= int(control_frequency * motion_duration)
                    self.actuation_steps -= self.max_stale_steps
            else:
                self.stale_counter = 0
        elif control_mode == 'joint_velocity':
            new_joints = self.get_state()['joints']
            if self.prev_joints is None:
                self.prev_joints = new_joints
            else:
                if np.linalg.norm(new_joints - self.prev_joints) < 0.0001:
                    self.stale_counter += 1
                    if self.stale_counter > self.max_stale_steps:
                        curr_contr = self.cmi.current_controller
                        # print(curr_contr, self.cmi.is_running(curr_contr))
                        print('------------------------------------------------------ stale controller, restarting ', curr_contr)
                        self.cmi.stop_controller(curr_contr)
                        while self.cmi.is_running(curr_contr):
                            print('waiting for controller to stop')
                            time.sleep(1)
                        self.cmi.start_controller(curr_contr)
                        self.stale_counter = 0
                        self.logger.pop(self.max_stale_steps)
                        self.actuation_steps -= self.max_stale_steps
                else:
                    self.stale_counter = 0
            scaled_action = action[:7] * 0.15
            self.apply_joint_vel(scaled_action)
            self.prev_joints = new_joints
            self.log_metrics()
            self.rate.sleep()
        elif control_mode == 'joint_torque':
            new_joints = self.get_state()['joints']
            if self.prev_joints is None:
                self.prev_joints = new_joints
            else:
                if np.linalg.norm(new_joints - self.prev_joints) < 0.00001:
                    self.stale_counter += 1
                    if self.stale_counter > self.max_stale_steps:
                        curr_contr = self.cmi.current_controller
                        # print(curr_contr, self.cmi.is_running(curr_contr))
                        print('------------------------------------------------------ stale controller, restarting ', curr_contr)
                        self.cmi.stop_controller(curr_contr)
                        while self.cmi.is_running(curr_contr):
                            print('waiting for controller to stop')
                            time.sleep(1)
                        self.cmi.start_controller(curr_contr)
                        self.stale_counter = 0
                        self.logger.pop(self.max_stale_steps)
                        self.actuation_steps -= self.max_stale_steps
                    else:
                        self.stale_counter = 0
            gc_torques = self.robot.gravity_comp()
            c_comp = self.robot.coriolis_comp()
            joint_torques = action[:7] * 1.2 + gc_torques * 0.02 + c_comp
            self.robot.set_joint_torques(dict(zip(self.joint_names, joint_torques)))
            self.prev_joints = new_joints
            self.rate.sleep()            
        return self._get_end_effector_pos()


    def _capture_img(self):
        start = time.time()
        ret, frame = self.cap.read()
        end = time.time()
        print("Capture time:", end - start)
        if not ret:
            raise RuntimeError("Failed to capture image from camera.")

        frame = cv2.resize(frame, (64, 64))
        # print(frame.shape)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        return frame       

    def _get_obs(self, joint_pos, joint_vels):

        robotic_arm_pointer = self._get_end_effector_pos()
        target = self.episode_target
        return np.concatenate([joint_pos, joint_vels, robotic_arm_pointer, target])

    def _compute_reward(self, distance, action):
        return -distance - 0.01 * np.linalg.norm(action)


    def handle_joint_angle_in_bound(self, action):
        current_joint_angle = self.robot.joint_angles()
        in_bound = [False] * 7
        for i, joint_name in enumerate(self.joint_names):
            if current_joint_angle[joint_name] > 0.05 + self.joint_angle_bound[i][1]:
                 
                action[i] = -0.5
            elif current_joint_angle[joint_name] < -0.05+ self.joint_angle_bound[i][0]:
                action[i] = +0.5
        return action

    def get_joint_vel_from_pos_vel(self, pose_vel):
        return np.matmul(np.linalg.pinv( self.get_robot_jacobian() ), pose_vel)

    def safe_actions(self, action):
        out_boundary = self.out_of_boundaries()
        x, y, z = self.robot.endpoint_pose()['position']
        self.box_Normals = np.zeros((6,3))
        self.box_Normals[0,:] = [1,0,0]
        self.box_Normals[1,:] = [-1,0,0]
        self.box_Normals[2,:] = [0,1,0]
        self.box_Normals[3,:] = [0,-1,0]
        self.box_Normals[4,:] = [0,0,1]
        self.box_Normals[5,:] = [0,0,-1]
        self.planes_d = [   self.safe_bound_box[0][0],
                            -self.safe_bound_box[0][1],
                            self.safe_bound_box[1][0],
                            -self.safe_bound_box[1][1],
                            self.safe_bound_box[2][0],
                            -self.safe_bound_box[2][1]]
        if out_boundary:
            action = np.zeros((3,))
            for i in range(6):
                # action += 0.05 * self.box_Normals[i] * ( (self.box_Normals[i].dot(np.array([x,y,z])) - self.planes_d[i]) < 0 ) 
                ####
                action += 0.1 * self.box_Normals[i] * ( (self.box_Normals[i].dot(np.array([x,y,z])) - self.planes_d[i]) < 0 ) 

        return action

    def close(self):
        # stop the robot
        cv2.destroyAllWindows()

        self.apply_joint_vel(np.zeros((7,)))
        self.terminate()
    
    def exit_handler(self,signum):
        exit(signum)

    
    def terminate(self):
        
        self.exit_handler(1)

    def seed(self, seed):
        np.random.seed(seed)

    def open_gripper(self):
        return self.gripper.open()
    def grasp_object(self):
        # self.robot_status.enable()
        return self.gripper.grasp(0, 4, speed=0.1, epsilon_outer=0.1, wait_for_result=True)

    def close_gripper(self):
        return self.gripper.close()

    def get_fingertip_width(self):
        finger_positions = self.gripper.joint_ordered_positions()
        return finger_positions[0] + finger_positions[1]

    def _compute_distance(self, joint_angles):
        robotic_arm_pointer = self._get_end_effector_pos()
        target = np.array([0.7, 0.2, 0.3])
        return np.linalg.norm(target - robotic_arm_pointer)
    
    def _get_end_effector_pos(self):

        ee_pose = self.robot.endpoint_pose()
        return ee_pose['position']

    def move_to_pose_ee(self, ref_ee_pos, ref_ee_angle=None, pose_vel_limit=0.2):
        counter = 0
        # print('11111', rospy.Time.now())
        
        if ref_ee_angle is None:
            ref_ee_angle = np.array(self.euler_from_quaternion(self.reset_ee_quaternion))

        while True:
            self.robot_status.enable()
            
            counter += 1
            
            self.get_state()
            action = np.zeros((4,))
            action[:3] = ref_ee_pos-self.ee_position
            action[-1] = 1
            
            # calculate joint actions
            d_angle =  ref_ee_angle - np.array(self.euler_from_quaternion(self.ee_orientation))
            for i in range(3):
                if d_angle[i] < -np.pi:
                    d_angle[i] += 2*np.pi
                elif d_angle[i] > np.pi:
                    d_angle[i] -= 2*np.pi
            d_angle *= 0.5
            #print('d_angle', d_angle)

            #if max(np.abs(action[:3])) < 0.005 or 
            #print(action)
            if counter > 290 * 0.04 / self.dt:
                curr_contr = self.cmi.current_controller
                # print(curr_contr, self.cmi.is_running(curr_contr))
                print('------------------------------------------------------ stale controller, restarting ', curr_contr)
                self.cmi.stop_controller(curr_contr)
                while self.cmi.is_running(curr_contr):
                    print('waiting for controller to stop')
                    time.sleep(1)
                self.cmi.start_controller(curr_contr)
                counter = 0
            if max(np.abs(action[:3])) < 0.002 and max(np.abs(d_angle)) < 0.1:
                if counter > 300 * 0.04 / self.dt:
                    print('----------------------------------------------------', counter)
                for _ in range(5):
                    self.apply_joint_vel(np.zeros((7,)))
                    self.rate.sleep()
                break

            #self.step(action, ignore_safety=True)
            # limit action
            pose_action = np.clip(action[:3], -pose_vel_limit, pose_vel_limit)

            d_X = np.array([pose_action[0], pose_action[1], pose_action[2], d_angle[0],d_angle[1],d_angle[2]])
            joints_action = self.get_joint_vel_from_pos_vel(d_X)
            # print('joints_action', joints_action)
            self.apply_joint_vel(joints_action)
            
            # action cycle time
            self.rate.sleep()
        for _ in range(5):
            self.apply_joint_vel(np.zeros((7,)))
            self.rate.sleep()

    def move_to_target_xyz(self, target_xyz):
        obs = self.get_state()
        new_ee_mat = np.identity(4)
        # new_ee_mat[:3, :3] = self.ee_ori_mat
        new_ee_mat[:3, :3] = self.matrix_from_quaternion(self.ee_orientation)
        new_ee_mat[:3, 3] = target_xyz
        q7 = obs['joints'][6]
        q_actual = obs['joints']
        q = franka_analytic_ik(new_ee_mat, q7, q_actual)

        # if self.target_q is None:
        #     self.target_q = q_actual.copy()
        #     self.target_q[0] += .2
        joint_positions = dict(zip(self.joint_names, q))
        control_frequency, motion_duration = 1 / 0.04, 2
        # smoothly_move_to_position(self, self.robot, joint_positions, control_frequency=control_frequency, motion_duration=motion_duration)
        # smoothly_move_to_position_vel(self.robot, self.robot_status, joint_positions, control_frequency=40)
        self.robot.set_joint_positions(joint_positions)
        self.log_metrics()
        self.rate.sleep()

        # restart controller if new joint positions are close to old ones
        new_joints = self.get_state()['joints']
        if np.linalg.norm(new_joints - obs['joints']) < 0.0001:
            self.stale_counter += 1
            if self.stale_counter > self.max_stale_steps:
                curr_contr = self.cmi.current_controller
                # print(curr_contr, self.cmi.is_running(curr_contr))
                print('------------------------------------------------------ stale controller, restarting ', curr_contr)
                self.cmi.stop_controller(curr_contr)
                while self.cmi.is_running(curr_contr):
                    print('waiting for controller to stop')
                    time.sleep(1)
                self.cmi.start_controller(curr_contr)

                self.stale_counter = 0
                self.logger.pop(int(control_frequency * motion_duration))
                self.actuation_steps -= int(control_frequency * motion_duration)
        else:
            self.stale_counter = 0

        # q = q_actual.copy()
        # q[0] += .01
        # print('q: ', type(q), q)
        joint_positions = dict(zip(self.joint_names, q))

    def _skew(self, w: np.ndarray) -> np.ndarray:
        wx, wy, wz = float(w[0]), float(w[1]), float(w[2])
        return np.array([
            [0.0, -wz,  wy],
            [wz,  0.0, -wx],
            [-wy, wx,  0.0],
        ], dtype=np.float64)

    def _so3_exp(self, w: np.ndarray) -> np.ndarray:
        """SO(3) exponential map for rotation vector w (axis*angle)."""
        w = np.asarray(w, dtype=np.float64).reshape(3)
        theta2 = float(w @ w)
        theta = float(np.sqrt(theta2 + 1e-12))
        K = self._skew(w)
        I = np.eye(3, dtype=np.float64)

        if theta < 1e-6:
            # series expansion
            A = 1.0 - theta2 / 6.0 + (theta2 * theta2) / 120.0
            B = 0.5 - theta2 / 24.0 + (theta2 * theta2) / 720.0
        else:
            A = np.sin(theta) / theta
            B = (1.0 - np.cos(theta)) / (theta2 + 1e-12)

        return I + A * K + B * (K @ K)

    def _clip_facing_down(self, Rmat: np.ndarray, max_tilt_rad: float = 0.35) -> np.ndarray:
        """
        Clamp tool +Z axis to be within max_tilt of world down [0,0,-1],
        preserving yaw/twist as much as possible (same idea as sim).
        """
        Rmat = np.asarray(Rmat, dtype=np.float64).reshape(3, 3)
        down = np.array([0.0, 0.0, -1.0], dtype=np.float64)

        z = Rmat[:, 2]
        z = z / (np.linalg.norm(z) + 1e-12)
        c = float(np.clip(z @ down, -1.0, 1.0))
        tilt = float(np.arccos(c))

        if tilt <= max_tilt_rad:
            return Rmat

        v = z - (z @ down) * down
        v_norm = np.linalg.norm(v) + 1e-12
        v_hat = v / v_norm

        z_clamped = np.cos(max_tilt_rad) * down + np.sin(max_tilt_rad) * v_hat
        z_clamped = z_clamped / (np.linalg.norm(z_clamped) + 1e-12)

        x = Rmat[:, 0]
        x = x - (x @ z_clamped) * z_clamped
        x_hat = x / (np.linalg.norm(x) + 1e-12)

        y_hat = np.cross(z_clamped, x_hat)
        y_hat = y_hat / (np.linalg.norm(y_hat) + 1e-12)

        # recompute x for orthonormality
        x_hat = np.cross(y_hat, z_clamped)
        return np.stack([x_hat, y_hat, z_clamped], axis=1)

    def move_tip_orient(self, action: np.ndarray):
        """
        Real-robot analog of sim _move_tip_orient.

        action (8,):
          [dx, dy, dz, drot_x, drot_y, drot_z, q7_action, gripper]
        Notes:
          - We ignore gripper here (handle gripper separately as you do now)
          - We apply drot in BODY frame: R_des = R_cur @ exp(drot)
          - We optionally use action[6] to bias q7 (like sim), but analytic IK already
            uses q7 seed; we pass current q7 and current q as seed.
        """
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        assert action.shape[0] >= 7, "Expected at least 7D action for tip+rot (+optional q7)."

        obs = self.get_state()
        ee_pose = self.robot.endpoint_pose()

        # Current EE position/orientation
        p_cur = np.asarray(ee_pose["position"], dtype=np.float64).reshape(3)
        R_cur = self.matrix_from_quaternion(self.ee_orientation).astype(np.float64)

        # --- Position increment (match sim scale idea) ---
        # In sim: scaled_pos = action[:3] * action_scale (default 0.005)
        # pos_scale = 0.01  #  for velocity
        if self.control_mode == 'cartesian_position':
            pos_scale = 0.02  # for position
        elif self.control_mode == 'cartesian_velocity':
            pos_scale = 0.01
        # pos_scale = 0.02  # for position
        
        p_des = p_cur + action[:3] * pos_scale

        # Safety clamps (match your cartesian_position constraints)
        p_des[0] = np.clip(p_des[0], 0.3, 0.77)
        p_des[1] = np.clip(p_des[1], -0.32, 0.32)
        p_des[2] = np.clip(p_des[2], 0.02, 0.50)

        # --- Rotation increment (match sim) ---
        rot_scale = pos_scale
        max_rot_step = 0.10   # cap on ||drot||
        drot = action[3:6] * rot_scale
        n = float(np.linalg.norm(drot) + 1e-12)
        if n > max_rot_step:
            drot = drot * (max_rot_step / n)

        dR = self._so3_exp(drot)
        R_des = R_cur @ dR
        R_des = self._clip_facing_down(R_des, max_tilt_rad=0.35)

        # --- Build target pose matrix for IK ---
        T_des = np.eye(4, dtype=np.float64)
        T_des[:3, :3] = R_des
        T_des[:3, 3] = p_des

        # --- IK ---
        q_actual = obs["joints"].astype(np.float64)
        q7 = float(q_actual[6])
        q7_scale = pos_scale
        q7_min, q7_max =  (-2.9, 2.9)  # set from your model
        q7_des = np.clip(q7 + action[6] * q7_scale, q7_min, q7_max)

        q = franka_analytic_ik(T_des, q7_des, q_actual)

        # If IK fails, it usually returns NaNs; guard for safety
        if q is None or np.any(np.isnan(q)):
            # fall back: do nothing
            self.apply_joint_vel(np.zeros((7,), dtype=np.float64))
            self.rate.sleep()
            return
        if self.control_mode == 'cartesian_velocity':
            # Directly apply joint velocities
            q_current = obs['joints'].astype(np.float64)
            q_vels = (q - q_current) / self.dt
            print("Joint velocities command:", q_vels)
            # max_joint_vel = 1.0  # rad/s
            # q_vels = np.clip(q_vels, -max_joint_vel, max_joint_vel)
            self.apply_joint_vel(q_vels)
            self.log_metrics()
            self.rate.sleep()
        elif self.control_mode == 'cartesian_position':
            # Command joint targets (position servo)
            # print("4th joint target:", q[1])
            # print("Current 4th joint:", obs['joints'][1])
            self.robot.set_joint_positions_velocities(q, np.zeros((7,)))
            # self.robot.set_joint_positions(dict(zip(self.joint_names, q)))
            self.log_metrics()
            self.rate.sleep()

        control_frequency, motion_duration = 1 / 0.04, 2
        new_joints = self.get_state()['joints']
        if np.linalg.norm(new_joints - obs['joints']) < 0.0001:
            self.stale_counter += 1
            if self.stale_counter > self.max_stale_steps:
                curr_contr = self.cmi.current_controller
                # print(curr_contr, self.cmi.is_running(curr_contr))
                print('------------------------------------------------------ stale controller, restarting ', curr_contr)
                self.cmi.stop_controller(curr_contr)
                while self.cmi.is_running(curr_contr):
                    print('waiting for controller to stop')
                    time.sleep(1)
                self.cmi.start_controller(curr_contr)

                self.stale_counter = 0
                self.logger.pop(int(control_frequency * motion_duration))
                self.actuation_steps -= int(control_frequency * motion_duration)
        else:
            self.stale_counter = 0
    
    def move_to_joint_positions(self, target_joints):
        joint_positions = dict(zip(self.joint_names, target_joints))
        smoothly_move_to_position_vel(self.robot, self.robot_status, joint_positions)
    

## TEST REALTIME environment
if __name__ == "__main__":
    env = FrankaPushCube(camera_index=0)
    env.reset()
    while True:
        y = np.random.uniform(-0.2, 0.2)
        z = np.random.uniform(0.06, 0.25)
        action = np.array([0.58, y, z])
        img,_ = env.step(action)
        cv2.imshow("Captured Image", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        cv2.waitKey(1)  # Use 1 instead of 0 to avoid blocking
        # time.sleep(0.02)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    env.close()
    cv2.destroyAllWindows()
        # env.grasp_object()
        # env.step(np.array([0.58, 0., 0.15118339]))
        # env.open_gripper()
        # env.reset()

