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
# ids camera lib for use of IDS ueye cameras.
# https://www.ids-imaging.us/files/downloads/ids-peak/readme/ids-peak-linux-readme-1.2_EN.html
#import ids
import time
import signal
import multiprocessing
from gymnasium.spaces import Box as GymBox
from scipy.spatial.transform import Rotation as R
import quaternion


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

class FrankaPickCubeCartesian(gym.Env):
    """
    Gym env for the real franka robot. Set up to perform the placement of a peg that starts in the robots hand into a slot
    """
    def __init__(self, dt=0.04, episode_length=8, camera_index=0, seed=9, size_tol=0.45, render=False):
        np.random.seed(seed)
        self.DT= dt
        self.dt = dt
        self.ep_time = 0
        self.max_episode_duration = episode_length # in seconds
        signal.signal(signal.SIGINT, self.exit_handler)
        # config_file = os.path.join(os.path.dirname(__file__), os.pardir, 'reacher.yaml')
        self.configs = configure('/home/chemist/Desktop/CoRL-2025/Franka-Real/franka_real/reacher.yaml')
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
        # TODO: Try 40 ms action cycle time with 2 ms actuation cycle time
        # TODO: Get joint velocities and plot them


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
        # close the gripper
        self.open_gripper()
        self._reset_stats()


        target_pose = [0.0, 0.0, 0.0, -1.55, 0.0, 1.88, 0.75, 0.04, 0.04]
        target_pose = dict(zip(self.joint_names, target_pose))

        self.reset_ee_quaternion = [0,-1.,0,0]
        
        obs = self.get_state()


        self.out_of_boundary_flag = False




        smoothly_move_to_position_vel(self.robot, self.robot_status, target_pose, MAX_JOINT_VELs=1.3)
        # print("here", self.robot.endpoint_pose()["orientation"])
        

        # stop the robot
        self.apply_joint_vel(np.zeros((7,)))

        # get the observation
        obs_robot = self.get_state()
        qpos = obs_robot["joints"].copy()
        qvel = obs_robot['joint_vels'].copy()

        obs = self._get_obs(qpos, qvel)

        self.time_steps = 0

        self.tv = time.time()
        self.reset_time = time.time()

        self.cur_step = 0 

        self._reset_stats()

        
        return obs.copy(), {}


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
        
        observation = {
            
            'last_action': self.prev_action,
            'joints': np.array(joint_angles),
            'joint_vels': np.array(joint_velocitys)
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

    def step(self, action):
        self.ep_time += self.dt
        self.cur_step += 1
        self.robot_status.enable()
        ee_pose = self.robot.endpoint_pose()
        print('ee_pose', ee_pose)
        self.move_to_pose_ee(action[:3])
        
        # limit joint action
        action = action.reshape(-1)


    def _get_obs(self, joint_pos, joint_vels):

        robotic_arm_pointer = self._get_end_effector_pos(joint_angles=joint_pos)
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
        self.close()
        self.exit_handler(1)

    def seed(self, seed):
        np.random.seed(seed)

    def open_gripper(self):
        return self.gripper.open()
    def grasp_object(self):
        return self.gripper.grasp(0.03, 5)

    def close_gripper(self):
        return self.gripper.close()

    def _compute_distance(self, joint_angles):
        robotic_arm_pointer = self._get_end_effector_pos(joint_angles)
        target = np.array([0.7, 0.2, 0.3])
        return np.linalg.norm(target - robotic_arm_pointer)
    
    def _get_end_effector_pos(self, joint_angles):

        ee_pose = self.robot.endpoint_pose()
        return ee_pose['position']
    def move_to_pose_ee(self, ref_ee_pos, pose_vel_limit=0.2):
        counter = 0
        # print('11111', rospy.Time.now())
        
        while True:
            self.robot_status.enable()
            
            counter += 1
            
            self.get_state()
            action = np.zeros((4,))
            action[:3] = ref_ee_pos-self.ee_position
            action[-1] = 1
            
            #if max(np.abs(action[:3])) < 0.005 or 
            #print(action)
            if max(np.abs(action[:3])) < 0.005 or counter > 100:
                break

            #self.step(action, ignore_safety=True)
            # limit action
            pose_action = np.clip(action[:3], -pose_vel_limit, pose_vel_limit)

            # calculate joint actions
            d_angle =  np.array(self.euler_from_quaternion(self.reset_ee_quaternion)) - np.array(self.euler_from_quaternion(self.ee_orientation))
            for i in range(3):
                if d_angle[i] < -np.pi:
                    d_angle[i] += 2*np.pi
                elif d_angle[i] > np.pi:
                    d_angle[i] -= 2*np.pi
            d_angle *= 0.5
            #print('d_angle', d_angle)
            d_X = np.array([pose_action[0], pose_action[1], pose_action[2], d_angle[0],d_angle[1],d_angle[2]])
            joints_action = self.get_joint_vel_from_pos_vel(d_X)
            # print('joints_action', joints_action)
            self.apply_joint_vel(joints_action)
            
            # action cycle time
            self.rate.sleep()
        self.apply_joint_vel(np.zeros((7,)))

    

## TEST REALTIME environment
if __name__ == "__main__":
    env = FrankaPickCubeCartesian()
    env.reset()
    time.sleep(1)
    env.step(np.array([0.58, 0., 0.04118339]))
    time.sleep(0.02)
    env.grasp_object()
    env.step(np.array([0.58, 0., 0.15118339]))
    env.open_gripper()
    env.reset()

