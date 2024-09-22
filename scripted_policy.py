import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion

from constants import SIM_TASK_CONFIGS
from ee_sim_env import make_ee_sim_env
from utils import quaternion_to_euler

import IPython
e = IPython.embed

BASE_X = 0.295
BASE_Y = 0.414
# MAX_ANGLE = 90
# MIN_ANGLE = 45.12239478312595

# MAX_DIST = 0.6684396383490685
# MIN_DIST = 0.45241159357381644

# def interpolate_wrist_angle(object_dist):    
#     a = (MIN_ANGLE - MAX_ANGLE)/(MAX_DIST - MIN_DIST)
#     b = MAX_ANGLE - a*MIN_DIST

#     return a*object_dist + b

# def calc_base_to_obj(obj_x, obj_y):
#     dist = np.linalg.norm(np.array([obj_x, obj_y]) - np.array([BASE_X, BASE_Y]))
#     return dist


class BasePolicy:
    def __init__(self, inject_noise=False):
        self.inject_noise = inject_noise
        self.step_count = 0
        self.left_trajectory = None
        self.right_trajectory = None

    def generate_trajectory(self, ts_first):
        raise NotImplementedError

    @staticmethod
    def interpolate(curr_waypoint, next_waypoint, t):
        t_frac = (t - curr_waypoint["t"]) / (next_waypoint["t"] - curr_waypoint["t"])
        curr_xyz = curr_waypoint['xyz']
        curr_quat = curr_waypoint['quat']
        curr_grip = curr_waypoint['gripper']
        next_xyz = next_waypoint['xyz']
        next_quat = next_waypoint['quat']
        next_grip = next_waypoint['gripper']
        xyz = curr_xyz + (next_xyz - curr_xyz) * t_frac
        quat = curr_quat + (next_quat - curr_quat) * t_frac
        gripper = curr_grip + (next_grip - curr_grip) * t_frac
        return xyz, quat, gripper

    def __call__(self, ts):
        # generate trajectory at first timestep, then open-loop execution
        if self.step_count == 0:
            self.generate_trajectory(ts)

        # obtain left and right waypoints
        if self.left_trajectory[0]['t'] == self.step_count:
            self.curr_left_waypoint = self.left_trajectory.pop(0)
        next_left_waypoint = self.left_trajectory[0]

        if self.right_trajectory[0]['t'] == self.step_count:
            self.curr_right_waypoint = self.right_trajectory.pop(0)
        next_right_waypoint = self.right_trajectory[0]

        # interpolate between waypoints to obtain current pose and gripper command
        left_xyz, left_quat, left_gripper = self.interpolate(self.curr_left_waypoint, next_left_waypoint, self.step_count)
        right_xyz, right_quat, right_gripper = self.interpolate(self.curr_right_waypoint, next_right_waypoint, self.step_count)

        # Inject noise
        if self.inject_noise:
            scale = 0.01
            left_xyz = left_xyz + np.random.uniform(-scale, scale, left_xyz.shape)
            right_xyz = right_xyz + np.random.uniform(-scale, scale, right_xyz.shape)

        action_left = np.concatenate([left_xyz, left_quat, [left_gripper]])
        action_right = np.concatenate([right_xyz, right_quat, [right_gripper]])

        self.step_count += 1
        return np.concatenate([action_left, action_right])


class PickAndTransferPolicy(BasePolicy):

    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']

        box_info = np.array(ts_first.observation['env_state'])
        box_xyz = box_info[:3]
        box_quat = box_info[3:]
        # print(f"Generate trajectory for {box_xyz=}")

        gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)

        meet_left_quat = Quaternion(axis=[1.0, 0.0, 0.0], degrees=90)

        meet_xyz = np.array([0, 0.5, 0.25])

        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0}, # sleep
            {"t": 100, "xyz": meet_xyz + np.array([-0.1, 0, -0.02]), "quat": meet_left_quat.elements, "gripper": 1}, # approach meet position
            {"t": 260, "xyz": meet_xyz + np.array([0.02, 0, -0.02]), "quat": meet_left_quat.elements, "gripper": 1}, # move to meet position
            {"t": 310, "xyz": meet_xyz + np.array([0.02, 0, -0.02]), "quat": meet_left_quat.elements, "gripper": 0}, # close gripper
            {"t": 360, "xyz": meet_xyz + np.array([-0.1, 0, -0.02]), "quat": np.array([1, 0, 0, 0]), "gripper": 0}, # move left
            {"t": 400, "xyz": meet_xyz + np.array([-0.1, 0, -0.02]), "quat": np.array([1, 0, 0, 0]), "gripper": 0}, # stay
        ]

        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
            {"t": 90, "xyz": box_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat.elements, "gripper": 1}, # approach the cube
            {"t": 130, "xyz": box_xyz + np.array([0, 0, -0.015]), "quat": gripper_pick_quat.elements, "gripper": 1}, # go down
            {"t": 170, "xyz": box_xyz + np.array([0, 0, -0.015]), "quat": gripper_pick_quat.elements, "gripper": 0}, # close gripper
            {"t": 200, "xyz": meet_xyz + np.array([0.05, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 0}, # approach meet position
            {"t": 220, "xyz": meet_xyz, "quat": gripper_pick_quat.elements, "gripper": 0}, # move to meet position
            {"t": 310, "xyz": meet_xyz, "quat": gripper_pick_quat.elements, "gripper": 1}, # open gripper
            {"t": 360, "xyz": meet_xyz + np.array([0.1, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 1}, # move to right
            {"t": 400, "xyz": meet_xyz + np.array([0.1, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 1}, # stay
        ]


class PickAndPutInPolicy(BasePolicy):
    def __init__(self, inject_noise=False, careful=False, quick=False):
        self.inject_noise = inject_noise
        self.step_count = 0
        self.left_trajectory = None
        self.right_trajectory = None
        self.careful = careful
        self.quick = quick

    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']

        obj_and_dst_info = np.array(ts_first.observation['env_state'])
        obj_xyz = obj_and_dst_info[:3]
        obj_quat = obj_and_dst_info[3:7]
        dst_xyz = obj_and_dst_info[7:10]

        gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat_obj = Quaternion(obj_quat) * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-90) * Quaternion(axis=[1.0, 0.0, 0.0], degrees=-90)

        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0}, # sleep
            {"t": 600, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0}, # sleep
        ]

        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
            {"t": 90, "xyz": obj_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat_obj.elements, "gripper": 1}, # approach the cucumber
            {"t": 130, "xyz": obj_xyz + np.array([0, 0, -0.015]), "quat": gripper_pick_quat_obj.elements, "gripper": 1}, # go down
            {"t": 170, "xyz": obj_xyz + np.array([0, 0, -0.015]), "quat": gripper_pick_quat_obj.elements, "gripper": 0}, # close gripper
            {"t": 300, "xyz": obj_xyz + np.array([0, 0, 0.3]), "quat": gripper_pick_quat_obj.elements, "gripper": 0}, # go up
            {"t": 400, "xyz": dst_xyz + np.array([0, 0, 0.3]), "quat": gripper_pick_quat.elements, "gripper": 0}, # move to the bucket
            {"t": 480, "xyz": dst_xyz + np.array([0, 0, 0.12]), "quat": gripper_pick_quat.elements, "gripper": 0}, # go down
            {"t": 520, "xyz": dst_xyz + np.array([0, 0, 0.12]), "quat": gripper_pick_quat.elements, "gripper": 1}, # open gripper
            {"t": 550, "xyz": dst_xyz + np.array([0, 0, 0.3]), "quat": gripper_pick_quat.elements, "gripper": 1}, # go up
            {"t": 600, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
        ]        

        if self.careful:
            for i, dic in enumerate(self.right_trajectory):
                self.right_trajectory[i]["t"] *= 2
            for i, dic in enumerate(self.left_trajectory):
                self.left_trajectory[i]["t"] *= 2
        if self.quick:
            for i, dic in enumerate(self.right_trajectory):
                self.right_trajectory[i]["t"] //= 2
            for i, dic in enumerate(self.left_trajectory):
                self.left_trajectory[i]["t"] //= 2


class PickAndPutInPolicyMobile(BasePolicy):
    def __init__(self, inject_noise=False, careful=False, quick=False):
        self.inject_noise = inject_noise
        self.step_count = 0
        self.left_trajectory = None
        self.right_trajectory = None
        self.careful = careful
        self.quick = quick

    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']

        obj_and_dst_info = np.array(ts_first.observation['env_state'])
        obj_xyz = obj_and_dst_info[:3]
        # obj_xy = obj_xyz[:2]
        obj_quat = obj_and_dst_info[3:7]
        dst_xyz = obj_and_dst_info[7:10]

        # obj_angle = quaternion_to_euler(obj_quat)[2]
        # base_to_obj_vec = obj_xy - np.array([BASE_X, BASE_Y])
        # base_angle = -np.rad2deg(np.arctan2(*base_to_obj_vec))
        # base_to_obj = calc_base_to_obj(obj_xy[0], obj_xy[1])
        # wrist_angle = interpolate_wrist_angle(base_to_obj)

        gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
        # gripper_pick_quat_obj = Quaternion(axis=[1.0, 0.0, 0.0], degrees=-wrist_angle) * Quaternion(axis=[0.0, -1.0, 0.0], degrees=obj_angle - base_angle)
        gripper_pick_quat_obj = Quaternion(obj_quat) * Quaternion(axis=[1.0, 0.0, 0.0], degrees=-90)
        gripper_pick_quat_obj2 = gripper_pick_quat * Quaternion(axis=[1.0, 0.0, 0.0], degrees=15)
        
        
        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0}, # sleep
            {"t": 720, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0}, # sleep
        ]

        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
            {"t": 150, "xyz": obj_xyz + np.array([0, 0, 0.2]), "quat": gripper_pick_quat_obj.elements, "gripper": 1}, # approach the cucumber
            {"t": 190, "xyz": obj_xyz + np.array([0, 0, -0.01]), "quat": gripper_pick_quat_obj.elements, "gripper": 1}, # go down
            {"t": 230, "xyz": obj_xyz + np.array([0, 0, -0.01]), "quat": gripper_pick_quat_obj.elements, "gripper": 0}, # close gripper
            {"t": 360, "xyz": obj_xyz + np.array([0, 0, 0.4]), "quat": gripper_pick_quat_obj2.elements, "gripper": 0}, # go up
            {"t": 460, "xyz": dst_xyz + np.array([0, 0, 0.4]), "quat": gripper_pick_quat_obj2.elements, "gripper": 0}, # move to the bucket
            {"t": 540, "xyz": dst_xyz + np.array([0, 0, 0.21]), "quat": gripper_pick_quat_obj2.elements, "gripper": 0}, # go down
            {"t": 580, "xyz": dst_xyz + np.array([0, 0, 0.21]), "quat": gripper_pick_quat_obj2.elements, "gripper": 1}, # open gripper
            {"t": 670, "xyz": init_mocap_pose_right[:3], "quat": gripper_pick_quat.elements, "gripper": 0}, # retutrn
            {"t": 720, "xyz": init_mocap_pose_right[:3], "quat": gripper_pick_quat.elements, "gripper": 0}, # sleep
        ]        

        if self.careful:
            for i, dic in enumerate(self.right_trajectory):
                self.right_trajectory[i]["t"] *= 2
            for i, dic in enumerate(self.left_trajectory):
                self.left_trajectory[i]["t"] *= 2
        if self.quick:
            for i, dic in enumerate(self.right_trajectory):
                self.right_trajectory[i]["t"] //= 2
            for i, dic in enumerate(self.left_trajectory):
                self.left_trajectory[i]["t"] //= 2


class PickAndPutInPolicyFranka(BasePolicy):
    def __init__(self, inject_noise=False, careful=False, quick=False):
        self.inject_noise = inject_noise
        self.step_count = 0
        self.left_trajectory = None
        self.right_trajectory = None
        self.careful = careful
        self.quick = quick

    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']

        obj_and_dst_info = np.array(ts_first.observation['env_state'])
        obj_xyz = obj_and_dst_info[:3]
        # obj_xy = obj_xyz[:2]
        obj_quat = obj_and_dst_info[3:7]
        dst_xyz = obj_and_dst_info[7:10]

        obj_angle = quaternion_to_euler(obj_quat)[2]
        obj_quat = Quaternion(axis=[0, 0, 1], degrees=(obj_angle + 180)%180)

        gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat_obj = obj_quat * Quaternion(axis=[1.0, 0.0, 0.0], degrees=-90+360)
        gripper_pick_quat_obj2 = Quaternion(np.array([1, 0, 0, 0])) * Quaternion(axis=[1.0, 0.0, 0.0], degrees=360) * Quaternion(axis=[0.0, 0.0, 1.0], degrees=90)
        
        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0}, # sleep
            {"t": 720, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0}, # sleep
        ]

        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
            {"t": 150, "xyz": obj_xyz + np.array([0, 0, 0.2]), "quat": gripper_pick_quat_obj.elements, "gripper": 1}, # approach the cucumber
            {"t": 190, "xyz": obj_xyz + np.array([0, 0, -0.01]), "quat": gripper_pick_quat_obj.elements, "gripper": 1}, # go down
            {"t": 230, "xyz": obj_xyz + np.array([0, 0, -0.01]), "quat": gripper_pick_quat_obj.elements, "gripper": 0}, # close gripper
            {"t": 360, "xyz": obj_xyz + np.array([0, 0, 0.4]), "quat": gripper_pick_quat_obj2.elements, "gripper": 0}, # go up
            {"t": 460, "xyz": dst_xyz + np.array([0, 0, 0.4]), "quat": gripper_pick_quat_obj2.elements, "gripper": 0}, # move to the bucket
            {"t": 540, "xyz": dst_xyz + np.array([0, 0, 0.21]), "quat": gripper_pick_quat_obj2.elements, "gripper": 0}, # go down
            {"t": 580, "xyz": dst_xyz + np.array([0, 0, 0.21]), "quat": gripper_pick_quat_obj2.elements, "gripper": 1}, # open gripper
            {"t": 670, "xyz": init_mocap_pose_right[:3], "quat": gripper_pick_quat.elements, "gripper": 0}, # retutrn
            {"t": 720, "xyz": init_mocap_pose_right[:3], "quat": gripper_pick_quat.elements, "gripper": 0}, # sleep
        ]        

        if self.careful:
            for i, dic in enumerate(self.right_trajectory):
                self.right_trajectory[i]["t"] *= 2
            for i, dic in enumerate(self.left_trajectory):
                self.left_trajectory[i]["t"] *= 2
        if self.quick:
            for i, dic in enumerate(self.right_trajectory):
                self.right_trajectory[i]["t"] //= 2
            for i, dic in enumerate(self.left_trajectory):
                self.left_trajectory[i]["t"] //= 2


class PickAndPutInCardboardPolicyFranka(BasePolicy):
    def __init__(self, inject_noise=False, careful=False, quick=False):
        self.inject_noise = inject_noise
        self.step_count = 0
        self.left_trajectory = None
        self.right_trajectory = None
        self.careful = careful
        self.quick = quick
        self._x_offset = 0.13

    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']

        obj_and_dst_info = np.array(ts_first.observation['env_state'])
        obj_xyz = obj_and_dst_info[:3]
        obj_quat = obj_and_dst_info[3:7]
        dst_xyz = obj_and_dst_info[7:10]
        dst_quat = obj_and_dst_info[10:]
        dst_angle = quaternion_to_euler(dst_quat, degrees=False)[2]        
        dst_xyz[:2] += np.array([[np.cos(dst_angle), -np.sin(dst_angle)], [np.sin(dst_angle), np.cos(dst_angle)]]) @ np.array([self._x_offset, 0])
        obj_angle = quaternion_to_euler(obj_quat)[2]
        obj_quat = Quaternion(axis=[0, 0, 1], degrees=(obj_angle + 180)%180)

        gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat_obj = obj_quat * Quaternion(axis=[1.0, 0.0, 0.0], degrees=-90+360)
        gripper_pick_quat_obj2 = Quaternion(dst_quat)
        
        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0}, # sleep
            {"t": 720, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0}, # sleep
        ]

        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
            {"t": 150, "xyz": obj_xyz + np.array([0, 0, 0.2]), "quat": gripper_pick_quat_obj.elements, "gripper": 1}, # approach the cucumber
            {"t": 190, "xyz": obj_xyz + np.array([0, 0, -0.01]), "quat": gripper_pick_quat_obj.elements, "gripper": 1}, # go down
            {"t": 230, "xyz": obj_xyz + np.array([0, 0, -0.01]), "quat": gripper_pick_quat_obj.elements, "gripper": 0}, # close gripper
            {"t": 360, "xyz": obj_xyz + np.array([0, 0, 0.4]), "quat": gripper_pick_quat_obj2.elements, "gripper": 0}, # go up
            {"t": 460, "xyz": dst_xyz + np.array([0, 0, 0.4]), "quat": gripper_pick_quat_obj2.elements, "gripper": 0}, # move to the bucket
            {"t": 540, "xyz": dst_xyz + np.array([0, 0, 0.21]), "quat": gripper_pick_quat_obj2.elements, "gripper": 0}, # go down
            {"t": 580, "xyz": dst_xyz + np.array([0, 0, 0.21]), "quat": gripper_pick_quat_obj2.elements, "gripper": 1}, # open gripper
            {"t": 670, "xyz": init_mocap_pose_right[:3], "quat": gripper_pick_quat.elements, "gripper": 0}, # retutrn
            {"t": 720, "xyz": init_mocap_pose_right[:3], "quat": gripper_pick_quat.elements, "gripper": 0}, # sleep
        ]        

        if self.careful:
            for i, dic in enumerate(self.right_trajectory):
                self.right_trajectory[i]["t"] *= 2
            for i, dic in enumerate(self.left_trajectory):
                self.left_trajectory[i]["t"] *= 2
        if self.quick:
            for i, dic in enumerate(self.right_trajectory):
                self.right_trajectory[i]["t"] //= 2
            for i, dic in enumerate(self.left_trajectory):
                self.left_trajectory[i]["t"] //= 2


class PickCoupleAndPutInPolicy(BasePolicy):
    obj = None

    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']

        objs_and_dst_info = np.array(ts_first.observation['env_state'])
        if self.obj == "cucumber":
            obj_xyz = objs_and_dst_info[:3]
            obj_quat = objs_and_dst_info[3:7]
        elif self.obj == "red_box":
            obj_xyz = objs_and_dst_info[7:10]
            obj_quat = objs_and_dst_info[10:14]
        dst_xyz = objs_and_dst_info[14:17]
        dst_quat = objs_and_dst_info[17:]
        # print(f"Generate trajectory for {box_xyz=}")

        gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat_obj = Quaternion(obj_quat) * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-90) * Quaternion(axis=[1.0, 0.0, 0.0], degrees=-90)

        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0}, # sleep
            {"t": 600, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0}, # sleep
        ]

        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
            {"t": 90, "xyz": obj_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat_obj.elements, "gripper": 1}, # approach the cube
            {"t": 130, "xyz": obj_xyz + np.array([0, 0, -0.015]), "quat": gripper_pick_quat_obj.elements, "gripper": 1}, # go down
            {"t": 170, "xyz": obj_xyz + np.array([0, 0, -0.015]), "quat": gripper_pick_quat_obj.elements, "gripper": 0}, # close gripper
            {"t": 300, "xyz": obj_xyz + np.array([0, 0, 0.3]), "quat": gripper_pick_quat_obj.elements, "gripper": 0}, # go up
            # {"t": 360, "xyz": obj_xyz + np.array([0, 0, 0.3]), "quat": gripper_pick_quat.elements, "gripper": 0}, # turn gripper
            {"t": 500, "xyz": dst_xyz + np.array([0, 0, 0.3]), "quat": gripper_pick_quat.elements, "gripper": 0}, # move to destination
            {"t": 540, "xyz": dst_xyz + np.array([0, 0, 0.12]), "quat": gripper_pick_quat.elements, "gripper": 0}, # go down
            {"t": 580, "xyz": dst_xyz + np.array([0, 0, 0.12]), "quat": gripper_pick_quat.elements, "gripper": 1}, # open gripper
            {"t": 600, "xyz": dst_xyz + np.array([0, 0, 0.3]), "quat": gripper_pick_quat.elements, "gripper": 1}, # go up
        ]        


class PickCoupleAndPutInPolicyMobile(BasePolicy):
    obj = None

    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']

        objs_and_dst_info = np.array(ts_first.observation['env_state'])
        if self.obj == "cucumber":
            obj_xyz = objs_and_dst_info[:3]
            obj_quat = objs_and_dst_info[3:7]
        elif self.obj == "red_box":
            obj_xyz = objs_and_dst_info[7:10]
            obj_quat = objs_and_dst_info[10:14]
        dst_xyz = objs_and_dst_info[14:17]
        obj_xy = obj_xyz[:2]
        base_to_obj_vec = obj_xy - np.array([BASE_X, BASE_Y])
        base_angle = -np.rad2deg(np.arctan2(*base_to_obj_vec))

        gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat_obj = Quaternion(obj_quat) * Quaternion(axis=[1.0, 0.0, 0.0], degrees=-90)
        gripper_pick_quat_obj2 = gripper_pick_quat * Quaternion(axis=[0.0, 0.0, 1.0], degrees=base_angle) * Quaternion(axis=[1.0, 0.0, 0.0], degrees=15)

        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0}, # sleep
            {"t": 720, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0}, # sleep
        ]

        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
            {"t": 100, "xyz": obj_xyz + np.array([0, 0, 0.35]), "quat": gripper_pick_quat_obj.elements, "gripper": 1}, # approach the cucumber
            {"t": 190, "xyz": obj_xyz + np.array([0, 0, -0.009]), "quat": gripper_pick_quat_obj.elements, "gripper": 1}, # go down
            {"t": 230, "xyz": obj_xyz + np.array([0, 0, -0.009]), "quat": gripper_pick_quat_obj.elements, "gripper": 0}, # close gripper
            {"t": 360, "xyz": obj_xyz + np.array([0, 0, 0.4]), "quat": gripper_pick_quat_obj2.elements, "gripper": 0}, # go up
            {"t": 460, "xyz": dst_xyz + np.array([0, 0, 0.4]), "quat": gripper_pick_quat_obj2.elements, "gripper": 0}, # move to the bucket
            {"t": 540, "xyz": dst_xyz + np.array([0, 0, 0.21]), "quat": gripper_pick_quat_obj2.elements, "gripper": 0}, # go down
            {"t": 580, "xyz": dst_xyz + np.array([0, 0, 0.21]), "quat": gripper_pick_quat_obj2.elements, "gripper": 1}, # open gripper
            {"t": 670, "xyz": init_mocap_pose_right[:3], "quat": gripper_pick_quat.elements, "gripper": 0}, # retutrn
            {"t": 720, "xyz": init_mocap_pose_right[:3], "quat": gripper_pick_quat.elements, "gripper": 0}, # sleep
        ]        


class PickCoupleAndPutInPolicyFranka(BasePolicy):
    obj = None

    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']

        objs_and_dst_info = np.array(ts_first.observation['env_state'])
        if self.obj == "cucumber":
            obj_xyz = objs_and_dst_info[:3]
            obj_quat = objs_and_dst_info[3:7]
        elif self.obj == "red_box":
            obj_xyz = objs_and_dst_info[7:10]
            obj_quat = objs_and_dst_info[10:14]
        dst_xyz = objs_and_dst_info[14:17]

        obj_angle = quaternion_to_euler(obj_quat)[2]
        obj_quat = Quaternion(axis=[0, 0, 1], degrees=(obj_angle + 180)%180)

        gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat_obj = obj_quat * Quaternion(axis=[1.0, 0.0, 0.0], degrees=-90+360)
        gripper_pick_quat_obj2 = Quaternion(np.array([1, 0, 0, 0])) * Quaternion(axis=[1.0, 0.0, 0.0], degrees=360) * Quaternion(axis=[0.0, 0.0, 1.0], degrees=90)

        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0}, # sleep
            {"t": 720, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0}, # sleep
        ]

        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
            {"t": 150, "xyz": obj_xyz + np.array([0, 0, 0.2]), "quat": gripper_pick_quat_obj.elements, "gripper": 1}, # approach the cucumber
            {"t": 190, "xyz": obj_xyz + np.array([0, 0, -0.01]), "quat": gripper_pick_quat_obj.elements, "gripper": 1}, # go down
            {"t": 230, "xyz": obj_xyz + np.array([0, 0, -0.01]), "quat": gripper_pick_quat_obj.elements, "gripper": 0}, # close gripper
            {"t": 360, "xyz": obj_xyz + np.array([0, 0, 0.4]), "quat": gripper_pick_quat_obj2.elements, "gripper": 0}, # go up
            {"t": 460, "xyz": dst_xyz + np.array([0, 0, 0.4]), "quat": gripper_pick_quat_obj2.elements, "gripper": 0}, # move to the bucket
            {"t": 540, "xyz": dst_xyz + np.array([0, 0, 0.21]), "quat": gripper_pick_quat_obj2.elements, "gripper": 0}, # go down
            {"t": 580, "xyz": dst_xyz + np.array([0, 0, 0.21]), "quat": gripper_pick_quat_obj2.elements, "gripper": 1}, # open gripper
            {"t": 670, "xyz": init_mocap_pose_right[:3], "quat": gripper_pick_quat.elements, "gripper": 0}, # retutrn
            {"t": 720, "xyz": init_mocap_pose_right[:3], "quat": gripper_pick_quat.elements, "gripper": 0}, # sleep
        ]        

class PickCoupleAndPutInPolicyFrankaBimanualLeftFirst(BasePolicy):
    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']

        objs_and_dst_info = np.array(ts_first.observation['env_state'])

        if objs_and_dst_info[0] < objs_and_dst_info[7]:
            # Cucumber is on the left side of Cube
            obj_xyz_left = objs_and_dst_info[:3] # Cucumber Position
            obj_quat_left = objs_and_dst_info[3:7]  # Cucumber Quat
            obj_xyz_right = objs_and_dst_info[7:10] # Cube Position
            obj_quat_right = objs_and_dst_info[10:14] # Cube Position
        else:
            # Cucumber is on the right side of Cube
            obj_xyz_left = objs_and_dst_info[7:10] # Cube Position
            obj_quat_left = objs_and_dst_info[10:14] # Cube Quat
            obj_xyz_right = objs_and_dst_info[0:3] # Cucumber Position
            obj_quat_right = objs_and_dst_info[3:7] # Cube Quat

        dst_xyz = objs_and_dst_info[14:17]
        obj_angle_left = quaternion_to_euler(obj_quat_left)[2]
        obj_quat_left = Quaternion(axis=[0, 0, 1], degrees=(obj_angle_left + 180)%180)
        obj_angle_right = quaternion_to_euler(obj_quat_right)[2]
        obj_quat_right = Quaternion(axis=[0, 0, 1], degrees=(obj_angle_right + 180)%180)

        gripper_pick_quat_left = Quaternion(init_mocap_pose_left[3:])
        gripper_pick_quat_obj_left = obj_quat_left * Quaternion(axis=[1.0, 0.0, 0.0], degrees=-90+360) * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-180) 
        gripper_pick_quat_obj2_left = Quaternion(np.array([1, 0, 0, 0])) * Quaternion(axis=[1.0, 0.0, 0.0], degrees=-360) * Quaternion(axis=[0.0, 0.0, 1.0], degrees=90+180)

        gripper_pick_quat_right = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat_obj_right = obj_quat_right * Quaternion(axis=[1.0, 0.0, 0.0], degrees=-90+360)
        gripper_pick_quat_obj2_right = Quaternion(np.array([1, 0, 0, 0])) * Quaternion(axis=[1.0, 0.0, 0.0], degrees=360) * Quaternion(axis=[0.0, 0.0, 1.0], degrees=90)

        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0}, # sleep
            {"t": 150, "xyz": obj_xyz_left + np.array([0, 0, 0.2]), "quat": gripper_pick_quat_obj_left.elements, "gripper": 1}, # approach the cucumber
            {"t": 190, "xyz": obj_xyz_left + np.array([0, 0, -0.01]), "quat": gripper_pick_quat_obj_left.elements, "gripper": 1}, # go down
            {"t": 230, "xyz": obj_xyz_left + np.array([0, 0, -0.01]), "quat": gripper_pick_quat_obj_left.elements, "gripper": 0}, # close gripper
            {"t": 360, "xyz": obj_xyz_left + np.array([0, 0, 0.35]), "quat": gripper_pick_quat_obj2_left.elements, "gripper": 0}, # go up

            {"t": 460, "xyz": dst_xyz + np.array([0, 0, 0.35]), "quat": gripper_pick_quat_obj2_left.elements, "gripper": 0}, # move to the bucket
            {"t": 540, "xyz": dst_xyz + np.array([0, 0, 0.21]), "quat": gripper_pick_quat_obj2_left.elements, "gripper": 0}, # go down
            {"t": 580, "xyz": dst_xyz + np.array([0, 0, 0.21]), "quat": gripper_pick_quat_obj2_left.elements, "gripper": 1}, # open gripper
            {"t": 670, "xyz": init_mocap_pose_left[:3], "quat": gripper_pick_quat_left.elements, "gripper": 0}, # return
            {"t": 800, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0}, # sleep
        ]

        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
            {"t": 150, "xyz": obj_xyz_right + np.array([0, 0, 0.2]), "quat": gripper_pick_quat_obj_right.elements, "gripper": 1}, # approach the cucumber
            {"t": 190, "xyz": obj_xyz_right + np.array([0, 0, -0.01]), "quat": gripper_pick_quat_obj_right.elements, "gripper": 1}, # go down
            {"t": 230, "xyz": obj_xyz_right + np.array([0, 0, -0.01]), "quat": gripper_pick_quat_obj_right.elements, "gripper": 0}, # close gripper
            {"t": 360, "xyz": obj_xyz_right + np.array([0, 0, 0.4]), "quat": gripper_pick_quat_obj2_right.elements, "gripper": 0}, # go up

            {"t": 600, "xyz": dst_xyz + np.array([0, 0, 0.4]), "quat": gripper_pick_quat_obj2_right.elements, "gripper": 0}, # move to the bucket
            {"t": 680, "xyz": dst_xyz + np.array([0, 0, 0.21]), "quat": gripper_pick_quat_obj2_right.elements, "gripper": 0}, # go down
            {"t": 720, "xyz": dst_xyz + np.array([0, 0, 0.21]), "quat": gripper_pick_quat_obj2_right.elements, "gripper": 1}, # open gripper
            {"t": 790, "xyz": init_mocap_pose_right[:3], "quat": gripper_pick_quat_right.elements, "gripper": 0}, # return
            {"t": 800, "xyz": init_mocap_pose_right[:3], "quat": gripper_pick_quat_right.elements, "gripper": 0}, # sleep
        ]        


class PickCoupleAndPutInPolicyFrankaBimanualRightFirst(BasePolicy):
    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']

        objs_and_dst_info = np.array(ts_first.observation['env_state'])

        if objs_and_dst_info[0] < objs_and_dst_info[7]:
            # Cucumber is on the left side of Cube
            obj_xyz_left = objs_and_dst_info[:3] # Cucumber Position
            obj_quat_left = objs_and_dst_info[3:7]  # Cucumber Quat
            obj_xyz_right = objs_and_dst_info[7:10] # Cube Position
            obj_quat_right = objs_and_dst_info[10:14] # Cube Position
        else:
            # Cucumber is on the right side of Cube
            obj_xyz_left = objs_and_dst_info[7:10] # Cube Position
            obj_quat_left = objs_and_dst_info[10:14] # Cube Quat
            obj_xyz_right = objs_and_dst_info[0:3] # Cucumber Position
            obj_quat_right = objs_and_dst_info[3:7] # Cube Quat

        dst_xyz = objs_and_dst_info[14:17]
        obj_angle_left = quaternion_to_euler(obj_quat_left)[2]
        obj_quat_left = Quaternion(axis=[0, 0, 1], degrees=(obj_angle_left + 180)%180)
        obj_angle_right = quaternion_to_euler(obj_quat_right)[2]
        obj_quat_right = Quaternion(axis=[0, 0, 1], degrees=(obj_angle_right + 180)%180)

        gripper_pick_quat_left = Quaternion(init_mocap_pose_left[3:])
        gripper_pick_quat_obj_left = obj_quat_left * Quaternion(axis=[1.0, 0.0, 0.0], degrees=-90+360) * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-180) 
        gripper_pick_quat_obj2_left = Quaternion(np.array([1, 0, 0, 0])) * Quaternion(axis=[1.0, 0.0, 0.0], degrees=-360) * Quaternion(axis=[0.0, 0.0, 1.0], degrees=90+180)

        gripper_pick_quat_right = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat_obj_right = obj_quat_right * Quaternion(axis=[1.0, 0.0, 0.0], degrees=-90+360)
        gripper_pick_quat_obj2_right = Quaternion(np.array([1, 0, 0, 0])) * Quaternion(axis=[1.0, 0.0, 0.0], degrees=360) * Quaternion(axis=[0.0, 0.0, 1.0], degrees=90)

        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0}, # sleep
            {"t": 150, "xyz": obj_xyz_left + np.array([0, 0, 0.2]), "quat": gripper_pick_quat_obj_left.elements, "gripper": 1}, # approach the cucumber
            {"t": 190, "xyz": obj_xyz_left + np.array([0, 0, -0.01]), "quat": gripper_pick_quat_obj_left.elements, "gripper": 1}, # go down
            {"t": 230, "xyz": obj_xyz_left + np.array([0, 0, -0.01]), "quat": gripper_pick_quat_obj_left.elements, "gripper": 0}, # close gripper
            {"t": 360, "xyz": obj_xyz_left + np.array([0, 0, 0.35]), "quat": gripper_pick_quat_obj2_left.elements, "gripper": 0}, # go up

            {"t": 600, "xyz": dst_xyz + np.array([0, 0, 0.35]), "quat": gripper_pick_quat_obj2_left.elements, "gripper": 0}, # move to the bucket
            {"t": 680, "xyz": dst_xyz + np.array([0, 0, 0.21]), "quat": gripper_pick_quat_obj2_left.elements, "gripper": 0}, # go down
            {"t": 720, "xyz": dst_xyz + np.array([0, 0, 0.21]), "quat": gripper_pick_quat_obj2_left.elements, "gripper": 1}, # open gripper
            {"t": 790, "xyz": init_mocap_pose_left[:3], "quat": gripper_pick_quat_left.elements, "gripper": 0}, # return
            {"t": 800, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0}, # sleep
        ]

        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
            {"t": 150, "xyz": obj_xyz_right + np.array([0, 0, 0.2]), "quat": gripper_pick_quat_obj_right.elements, "gripper": 1}, # approach the cucumber
            {"t": 190, "xyz": obj_xyz_right + np.array([0, 0, -0.01]), "quat": gripper_pick_quat_obj_right.elements, "gripper": 1}, # go down
            {"t": 230, "xyz": obj_xyz_right + np.array([0, 0, -0.01]), "quat": gripper_pick_quat_obj_right.elements, "gripper": 0}, # close gripper
            {"t": 360, "xyz": obj_xyz_right + np.array([0, 0, 0.4]), "quat": gripper_pick_quat_obj2_right.elements, "gripper": 0}, # go up

            {"t": 460, "xyz": dst_xyz + np.array([0, 0, 0.4]), "quat": gripper_pick_quat_obj2_right.elements, "gripper": 0}, # move to the bucket
            {"t": 540, "xyz": dst_xyz + np.array([0, 0, 0.21]), "quat": gripper_pick_quat_obj2_right.elements, "gripper": 0}, # go down
            {"t": 580, "xyz": dst_xyz + np.array([0, 0, 0.21]), "quat": gripper_pick_quat_obj2_right.elements, "gripper": 1}, # open gripper
            {"t": 670, "xyz": init_mocap_pose_right[:3], "quat": gripper_pick_quat_right.elements, "gripper": 0}, # return
            {"t": 800, "xyz": init_mocap_pose_right[:3], "quat": gripper_pick_quat_right.elements, "gripper": 0}, # sleep
        ]        


class PickMultipleAndPutInPolicy(BasePolicy):
    def __init__(self, pick_up_num = 3, obj_num = 4, inject_noise=False):
        self.inject_noise = inject_noise
        self.step_count = 0
        self.left_trajectory = None
        self.right_trajectory = None
        self.pick_up_num = pick_up_num
        self.obj_num = obj_num
        self.num_step = 600

    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']

        objs_and_dst_info = np.array(ts_first.observation['env_state'])


        dst_xyz = objs_and_dst_info[7*self.obj_num:7*self.obj_num+3]
        dst_quat = objs_and_dst_info[7*self.obj_num+3:]

        self.left_trajectory = [
                {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0}, # sleep
                {"t": self.num_step*self.pick_up_num, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0}, # sleep
            ]
        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
        ]

        for i in range(self.pick_up_num):
            xyz_start = i*7
            obj_xyz = objs_and_dst_info[xyz_start:xyz_start+3]
            obj_quat = objs_and_dst_info[xyz_start+3:xyz_start+3+4]

            gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
            gripper_pick_quat_obj = Quaternion(obj_quat) * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-90) * Quaternion(axis=[1.0, 0.0, 0.0], degrees=-90)

            right_trajectory_tmp =[
                {"t": i*self.num_step+90, "xyz": obj_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat_obj.elements, "gripper": 1}, # approach the cube
                {"t": i*self.num_step+130, "xyz": obj_xyz + np.array([0, 0, -0.015]), "quat": gripper_pick_quat_obj.elements, "gripper": 1}, # go down
                {"t": i*self.num_step+170, "xyz": obj_xyz + np.array([0, 0, -0.015]), "quat": gripper_pick_quat_obj.elements, "gripper": 0}, # close gripper
                {"t": i*self.num_step+300, "xyz": obj_xyz + np.array([0, 0, 0.3]), "quat": gripper_pick_quat_obj.elements, "gripper": 0}, # go up
                {"t": i*self.num_step+500, "xyz": dst_xyz + np.array([0, 0, 0.3]), "quat": gripper_pick_quat.elements, "gripper": 0}, # move to destination
                {"t": i*self.num_step+540, "xyz": dst_xyz + np.array([0, 0, 0.12]), "quat": gripper_pick_quat.elements, "gripper": 0}, # go down
                {"t": i*self.num_step+580, "xyz": dst_xyz + np.array([0, 0, 0.12]), "quat": gripper_pick_quat.elements, "gripper": 1}, # open gripper
                {"t": i*self.num_step+600, "xyz": dst_xyz + np.array([0, 0, 0.3]), "quat": gripper_pick_quat.elements, "gripper": 1}, # go up
            ]

            self.right_trajectory += right_trajectory_tmp


class InsertionPolicy(BasePolicy):

    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']

        peg_info = np.array(ts_first.observation['env_state'])[:7]
        peg_xyz = peg_info[:3]
        peg_quat = peg_info[3:]

        socket_info = np.array(ts_first.observation['env_state'])[7:]
        socket_xyz = socket_info[:3]
        socket_quat = socket_info[3:]

        gripper_pick_quat_right = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat_right = gripper_pick_quat_right * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)

        gripper_pick_quat_left = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat_left = gripper_pick_quat_left * Quaternion(axis=[0.0, 1.0, 0.0], degrees=60)

        meet_xyz = np.array([0, 0.5, 0.15])
        lift_right = 0.00715

        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0}, # sleep
            {"t": 120, "xyz": socket_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # approach the cube
            {"t": 170, "xyz": socket_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # go down
            {"t": 220, "xyz": socket_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # close gripper
            {"t": 285, "xyz": meet_xyz + np.array([-0.1, 0, 0]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # approach meet position
            {"t": 340, "xyz": meet_xyz + np.array([-0.05, 0, 0]), "quat": gripper_pick_quat_left.elements,"gripper": 0},  # insertion
            {"t": 400, "xyz": meet_xyz + np.array([-0.05, 0, 0]), "quat": gripper_pick_quat_left.elements, "gripper": 0},  # insertion
        ]

        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
            {"t": 120, "xyz": peg_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, # approach the cube
            {"t": 170, "xyz": peg_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, # go down
            {"t": 220, "xyz": peg_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, # close gripper
            {"t": 285, "xyz": meet_xyz + np.array([0.1, 0, lift_right]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, # approach meet position
            {"t": 340, "xyz": meet_xyz + np.array([0.05, 0, lift_right]), "quat": gripper_pick_quat_right.elements, "gripper": 0},  # insertion
            {"t": 400, "xyz": meet_xyz + np.array([0.05, 0, lift_right]), "quat": gripper_pick_quat_right.elements, "gripper": 0},  # insertion

        ]


def test_policy(task_name):
    # example rolling out pick_and_transfer policy
    onscreen_render = True
    inject_noise = False

    # setup the environment
    episode_len = SIM_TASK_CONFIGS[task_name]['episode_len']
    if 'sim_transfer_cube' in task_name:
        env = make_ee_sim_env('sim_transfer_cube')
    elif 'sim_insertion' in task_name:
        env = make_ee_sim_env('sim_insertion')
    else:
        raise NotImplementedError

    for episode_idx in range(2):
        ts = env.reset()
        episode = [ts]
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation['images']['angle'])
            plt.ion()

        policy = PickAndTransferPolicy(inject_noise)
        for step in range(episode_len):
            action = policy(ts)
            ts = env.step(action)
            episode.append(ts)
            if onscreen_render:
                plt_img.set_data(ts.observation['images']['angle'])
                plt.pause(0.02)
        plt.close()

        episode_return = np.sum([ts.reward for ts in episode[1:]])
        if episode_return > 0:
            print(f"{episode_idx=} Successful, {episode_return=}")
        else:
            print(f"{episode_idx=} Failed")


if __name__ == '__main__':
    test_task_name = 'sim_transfer_cube_scripted'
    test_policy(test_task_name)

