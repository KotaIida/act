import numpy as np
import collections
import os

from constants import DT, XML_DIR_STATIC, XML_DIR_MOBILE, XML_DIR_FRANKA, START_ARM_POSE, START_ARM_POSE_MOBILE, START_ARM_POSE_FRANKA
from constants import PUPPET_GRIPPER_POSITION_CLOSE, PUPPET_GRIPPER_POSITION_CLOSE_MOBILE, PUPPET_GRIPPER_POSITION_CLOSE_FRANKA
from constants import PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN, PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN_MOBILE, PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN_FRANKA
from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN, PUPPET_GRIPPER_POSITION_NORMALIZE_FN_MOBILE, PUPPET_GRIPPER_POSITION_NORMALIZE_FN_FRANKA
from constants import PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN
from utils import quaternion_to_euler

from utils import sample_box_pose, sample_obj_and_dst_pose, sample_insertion_pose, sample_obj_box_dst_pose, sample_objs_dst_pose, sample_2objs_and_dst_pose
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base

import IPython
e = IPython.embed


def make_ee_sim_env(task_name, **kwargs):
    """
    Environment for simulated robot bi-manual manipulation, with end-effector control.
    Action space:      [left_arm_pose (7),             # position and quaternion for end effector
                        left_gripper_positions (1),    # normalized gripper position (0: close, 1: open)
                        right_arm_pose (7),            # position and quaternion for end effector
                        right_gripper_positions (1),]  # normalized gripper position (0: close, 1: open)

    Observation space: {"qpos": Concat[ left_arm_qpos (6),         # absolute joint position
                                        left_gripper_position (1),  # normalized gripper position (0: close, 1: open)
                                        right_arm_qpos (6),         # absolute joint position
                                        right_gripper_qpos (1)]     # normalized gripper position (0: close, 1: open)
                        "qvel": Concat[ left_arm_qvel (6),         # absolute joint velocity (rad)
                                        left_gripper_velocity (1),  # normalized gripper velocity (pos: opening, neg: closing)
                                        right_arm_qvel (6),         # absolute joint velocity (rad)
                                        right_gripper_qvel (1)]     # normalized gripper velocity (pos: opening, neg: closing)
                        "images": {"main": (480x640x3)}        # h, w, c, dtype='uint8'
    """
    if 'sim_transfer_cube_on_static_aloha' == task_name:
        xml_path = os.path.join(XML_DIR_STATIC, f'bimanual_viperx_ee_transfer_cube.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = TransferCubeEETask(random=False)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'sim_put_cucumber_in_bucket_on_static_aloha' == task_name:
        xml_path = os.path.join(XML_DIR_STATIC, f'bimanual_viperx_ee_put_cucumber_in_bucket.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = PickCucumberAndPutInEETask(random=False)
        env = control.Environment(physics, task, time_limit=50, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'sim_put_cube_in_bucket_on_static_aloha' == task_name:
        xml_path = os.path.join(XML_DIR_STATIC, f'bimanual_viperx_ee_put_cube_in_bucket.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = PickCubeAndPutInEETask(random=False)
        env = control.Environment(physics, task, time_limit=50, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'sim_put_couple_in_bucket_on_static_aloha' == task_name:
        xml_path = os.path.join(XML_DIR_STATIC, f'bimanual_viperx_ee_put_couple_in_bucket.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = PickCoupleAndPutInEETask(random=False)
        env = control.Environment(physics, task, time_limit=50, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'sim_put_multiple_in_bucket_on_static_aloha' == task_name:
        xml_path = os.path.join(XML_DIR_STATIC, f'bimanual_viperx_ee_put_multiple_in_bucket.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        pickup_num = kwargs["pickup_num"]
        task = PickMultipleAndPutInEETask(random=False, pickup_num=pickup_num)
        env = control.Environment(physics, task, time_limit=50, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'sim_insertion_scripted' == task_name:
        xml_path = os.path.join(XML_DIR_STATIC, f'bimanual_viperx_ee_insertion.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = InsertionEETask(random=False)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'sim_put_cucumber_in_bucket_on_mobile_aloha' == task_name:
        xml_path = os.path.join(XML_DIR_MOBILE, f'bimanual_viperx_ee_put_cucumber_in_bucket.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = PickCucumberAndPutInEETaskMobile(random=False)
        env = control.Environment(physics, task, time_limit=50, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'sim_put_couple_in_bucket_on_mobile_aloha' == task_name:
        xml_path = os.path.join(XML_DIR_MOBILE, f'bimanual_viperx_ee_put_couple_in_bucket.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = PickCoupleAndPutInEETaskMobile(random=False)
        env = control.Environment(physics, task, time_limit=50, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'sim_put_cucumber_in_bucket_on_franka_dual' == task_name:
        xml_path = os.path.join(XML_DIR_FRANKA, f'franka_dual_ee_put_cucumber_in_bucket.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = PickCucumberAndPutInEETaskFranka(random=False)
        env = control.Environment(physics, task, time_limit=50, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'sim_put_couple_in_bucket_on_franka_dual' == task_name:
        xml_path = os.path.join(XML_DIR_FRANKA, f'franka_dual_ee_put_couple_in_bucket.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = PickCoupleAndPutInEETaskFranka(random=False)
        env = control.Environment(physics, task, time_limit=50, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'sim_put_quadruple_in_bucket_on_franka_dual_bimanual' == task_name:
        xml_path = os.path.join(XML_DIR_FRANKA, f'franka_dual_ee_put_couple_in_bucket.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = PickQuadrupleAndPutInEETaskFrankaBimanual(random=False)
        env = control.Environment(physics, task, time_limit=50, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'sim_put_cushion_in_cardboard_v_on_franka_dual' == task_name:
        xml_path = os.path.join(XML_DIR_FRANKA, f'franka_dual_ee_put_cushion_in_cardboard_v.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = PickCushionAndPutInCardboardEETaskFranka(random=False)
        env = control.Environment(physics, task, time_limit=50, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'sim_put_cushion_in_cardboard_v_recovery_on_franka_dual' == task_name:
        xml_path = os.path.join(XML_DIR_FRANKA, f'franka_dual_ee_put_cushion_in_cardboard_v_recovery.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = PickCushionAndPutInCardboardRecoveryEETaskFranka(random=False)
        env = control.Environment(physics, task, time_limit=50, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'sim_put_cushion_in_cardboard_h_on_franka_dual' == task_name:
        xml_path = os.path.join(XML_DIR_FRANKA, f'franka_dual_ee_put_cushion_in_cardboard_h.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = PickCushionAndPutInCardboardEETaskFranka(random=False)
        env = control.Environment(physics, task, time_limit=50, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'sim_put_cushion_in_cardboard_v_eval_on_franka_dual' == task_name:
        xml_path = os.path.join(XML_DIR_FRANKA, f'franka_dual_ee_put_cushion_in_cardboard_v_eval.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = PickCushionAndPutInCardboardEETaskFranka(random=False)
        env = control.Environment(physics, task, time_limit=50, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'sim_put_cushion_in_cardboard_h_eval_on_franka_dual' == task_name:
        xml_path = os.path.join(XML_DIR_FRANKA, f'franka_dual_ee_put_cushion_in_cardboard_h_eval.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = PickCushionAndPutInCardboardEETaskFranka(random=False)
        env = control.Environment(physics, task, time_limit=50, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    else:
        raise NotImplementedError
    return env

class BimanualViperXEETask(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)

    def before_step(self, action, physics):
        a_len = len(action) // 2
        action_left = action[:a_len]
        action_right = action[a_len:]

        # set mocap position and quat
        # left
        np.copyto(physics.data.mocap_pos[0], action_left[:3])
        np.copyto(physics.data.mocap_quat[0], action_left[3:7])
        # right
        np.copyto(physics.data.mocap_pos[1], action_right[:3])
        np.copyto(physics.data.mocap_quat[1], action_right[3:7])

        # set gripper
        g_left_ctrl = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(action_left[7])
        g_right_ctrl = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(action_right[7])
        np.copyto(physics.data.ctrl, np.array([g_left_ctrl, -g_left_ctrl, g_right_ctrl, -g_right_ctrl]))

    def initialize_robots(self, physics):
        # reset joint position
        physics.named.data.qpos[:16] = START_ARM_POSE

        # reset mocap to align with end effector
        # to obtain these numbers:
        # (1) make an ee_sim env and reset to the same start_pose
        # (2) get env._physics.named.data.xpos['vx300s_left/gripper_link']
        #     get env._physics.named.data.xquat['vx300s_left/gripper_link']
        #     repeat the same for right side
        np.copyto(physics.data.mocap_pos[0], [-0.31718881, 0.5, 0.29525084])
        np.copyto(physics.data.mocap_quat[0], [1, 0, 0, 0])
        # right
        np.copyto(physics.data.mocap_pos[1], np.array([0.31718881, 0.49999888, 0.29525084]))
        np.copyto(physics.data.mocap_quat[1],  [1, 0, 0, 0])

        # reset gripper control
        close_gripper_control = np.array([
            PUPPET_GRIPPER_POSITION_CLOSE,
            -PUPPET_GRIPPER_POSITION_CLOSE,
            PUPPET_GRIPPER_POSITION_CLOSE,
            -PUPPET_GRIPPER_POSITION_CLOSE,
        ])
        np.copyto(physics.data.ctrl, close_gripper_control)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        super().initialize_episode(physics)

    @staticmethod
    def get_qpos(physics):
        qpos_raw = physics.data.qpos.copy()
        left_qpos_raw = qpos_raw[:8]
        right_qpos_raw = qpos_raw[8:16]
        left_arm_qpos = left_qpos_raw[:6]
        right_arm_qpos = right_qpos_raw[:6]
        left_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(left_qpos_raw[6])]
        right_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(right_qpos_raw[6])]
        return np.concatenate([left_arm_qpos, left_gripper_qpos, right_arm_qpos, right_gripper_qpos])

    @staticmethod
    def get_qvel(physics):
        qvel_raw = physics.data.qvel.copy()
        left_qvel_raw = qvel_raw[:8]
        right_qvel_raw = qvel_raw[8:16]
        left_arm_qvel = left_qvel_raw[:6]
        right_arm_qvel = right_qvel_raw[:6]
        left_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(left_qvel_raw[6])]
        right_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(right_qvel_raw[6])]
        return np.concatenate([left_arm_qvel, left_gripper_qvel, right_arm_qvel, right_gripper_qvel])

    @staticmethod
    def get_env_state(physics):
        raise NotImplementedError

    def get_observation(self, physics):
        # note: it is important to do .copy()
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos(physics) # waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate, gripper * left or right = 14
        obs['qvel'] = self.get_qvel(physics)
        obs['env_state'] = self.get_env_state(physics) # xyz and quaternion of red box = 4+3  
        obs['images'] = dict()
        obs['images']['top'] = physics.render(height=480, width=640, camera_id='top')
        obs['images']['angle'] = physics.render(height=480, width=640, camera_id='angle')
        obs['images']['vis'] = physics.render(height=480, width=640, camera_id='front_close')
        obs['images']['left_wrist'] = physics.render(height=480, width=640, camera_id='left_wrist')
        obs['images']['right_wrist'] = physics.render(height=480, width=640, camera_id='right_wrist')
        # used in scripted policy to obtain starting pose
        obs['mocap_pose_left'] = np.concatenate([physics.data.mocap_pos[0], physics.data.mocap_quat[0]]).copy()
        obs['mocap_pose_right'] = np.concatenate([physics.data.mocap_pos[1], physics.data.mocap_quat[1]]).copy()

        # used when replaying joint trajectory
        obs['gripper_ctrl'] = physics.data.ctrl.copy()
        return obs

    def get_reward(self, physics):
        raise NotImplementedError


class TransferCubeEETask(BimanualViperXEETask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self.initialize_robots(physics)
        # randomize box position
        cube_pose = sample_box_pose()
        box_start_idx = physics.model.name2id('red_box_joint', 'joint')
        np.copyto(physics.data.qpos[box_start_idx : box_start_idx + 7], cube_pose)
        # print(f"randomized cube position to {cube_position}")

        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_left_gripper = ("red_box", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
        touch_right_gripper = ("red_box", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        touch_table = ("red_box", "table") in all_contact_pairs

        reward = 0
        if touch_right_gripper:
            reward = 1
        if touch_right_gripper and not touch_table: # lifted
            reward = 2
        if touch_left_gripper: # attempted transfer
            reward = 3
        if touch_left_gripper and not touch_table: # successful transfer
            reward = 4
        return reward


class PickAndPutInEETask(BimanualViperXEETask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 1
        self._bucket_height = 0.11 + 0.002
        self._bucket_radius = 0.055 + 0.002
        self.object_name = "_joint"

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self.initialize_robots(physics)
        # randomize box position
        object_and_bucket_pose = sample_obj_and_dst_pose()
        box_start_idx = physics.model.name2id(self.object_name, 'joint')
        np.copyto(physics.data.qpos[box_start_idx : box_start_idx + 7 + 7], object_and_bucket_pose)
        # print(f"randomized cube position to {cube_position}")

        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        # return whether cucumber is in the bucket
        dist_object_to_bucket_center = np.linalg.norm(physics.named.data.qpos[self.object_name][:2] - physics.named.data.qpos["bucket_joint"][:2])
        object_center_z = physics.named.data.qpos[self.object_name][2]
        in_bucket = (dist_object_to_bucket_center < self._bucket_radius) & (object_center_z < self._bucket_height)
        
        reward = 0
        if in_bucket:
            reward = 1
        return reward


class PickCucumberAndPutInEETask(PickAndPutInEETask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.object_name = "cucumber_joint"


class PickCubeAndPutInEETask(PickAndPutInEETask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.object_name = "red_box_joint"


class PickCoupleAndPutInEETask(PickAndPutInEETask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.cucumber_box_bucket_pose = None
        self.obj = None

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self.initialize_robots(physics)
        # randomize box position
        if self.obj == None:
            self.cucumber_box_bucket_pose = sample_obj_box_dst_pose()
            self.obj = "red_box"
        elif self.obj == "red_box":
            self.obj = "cucumber"
        else:
             self.cucumber_box_bucket_pose = None
             self.obj = None
        box_start_idx = physics.model.name2id(f'cucumber_joint', 'joint')
        np.copyto(physics.data.qpos[box_start_idx : box_start_idx + 7 + 7 + 7], self.cucumber_box_bucket_pose)
        # print(f"randomized cube position to {cube_position}")

        super(PickAndPutInEETask, self).initialize_episode(physics)

    def get_reward(self, physics):
        # return whether cucumbert is in the bucket
        dist_cucumber_to_bucket_center = np.linalg.norm(physics.named.data.qpos[f"{self.obj}_joint"][:2] - physics.named.data.qpos["bucket_joint"][:2])
        cucumber_center_z = physics.named.data.qpos[f"{self.obj}_joint"][2]
        in_bucket = (dist_cucumber_to_bucket_center < self._bucket_radius) & (cucumber_center_z < self._bucket_height)
        
        return int(in_bucket)


class PickMultipleAndPutInEETask(PickAndPutInEETask):
    def __init__(self, random=None, pickup_num=3):
        super().__init__(random=random)
        self.pickup_num = pickup_num
        self.cucumber_types = ["a", "b", "c", "d"]

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self.initialize_robots(physics)
        # randomize box position
        cucumber_and_bucket_pose = sample_objs_dst_pose()
        box_start_idx = physics.model.name2id('cucumber_a_joint', 'joint')
        np.copyto(physics.data.qpos[box_start_idx : box_start_idx + 7*5], cucumber_and_bucket_pose)
        # print(f"randomized cube position to {cube_position}")

        super(PickAndPutInEETask, self).initialize_episode(physics)

    def get_reward(self, physics):
        # return whether cucumbert is in the bucket
        reward = 0
        in_bucket_num = 0
        for cucumber_type in self.cucumber_types:
            dist_cucumber_to_bucket_center = np.linalg.norm(physics.named.data.qpos[f"cucumber_{cucumber_type}_joint"][:2] - physics.named.data.qpos["bucket_joint"][:2])
            cucumber_center_z = physics.named.data.qpos[f"cucumber_{cucumber_type}_joint"][2]
            in_bucket = (dist_cucumber_to_bucket_center < self._bucket_radius) & (cucumber_center_z < self._bucket_height)
        
            if in_bucket:
                in_bucket_num+=1

        if in_bucket_num == self.pickup_num:
            reward+=1

        return reward


class InsertionEETask(BimanualViperXEETask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self.initialize_robots(physics)
        # randomize peg and socket position
        peg_pose, socket_pose = sample_insertion_pose()
        id2index = lambda j_id: 16 + (j_id - 16) * 7 # first 16 is robot qpos, 7 is pose dim # hacky

        peg_start_id = physics.model.name2id('red_peg_joint', 'joint')
        peg_start_idx = id2index(peg_start_id)
        np.copyto(physics.data.qpos[peg_start_idx : peg_start_idx + 7], peg_pose)
        # print(f"randomized cube position to {cube_position}")

        socket_start_id = physics.model.name2id('blue_socket_joint', 'joint')
        socket_start_idx = id2index(socket_start_id)
        np.copyto(physics.data.qpos[socket_start_idx : socket_start_idx + 7], socket_pose)
        # print(f"randomized cube position to {cube_position}")

        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        # return whether peg touches the pin
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_right_gripper = ("red_peg", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        touch_left_gripper = ("socket-1", "vx300s_left/10_left_gripper_finger") in all_contact_pairs or \
                             ("socket-2", "vx300s_left/10_left_gripper_finger") in all_contact_pairs or \
                             ("socket-3", "vx300s_left/10_left_gripper_finger") in all_contact_pairs or \
                             ("socket-4", "vx300s_left/10_left_gripper_finger") in all_contact_pairs

        peg_touch_table = ("red_peg", "table") in all_contact_pairs
        socket_touch_table = ("socket-1", "table") in all_contact_pairs or \
                             ("socket-2", "table") in all_contact_pairs or \
                             ("socket-3", "table") in all_contact_pairs or \
                             ("socket-4", "table") in all_contact_pairs
        peg_touch_socket = ("red_peg", "socket-1") in all_contact_pairs or \
                           ("red_peg", "socket-2") in all_contact_pairs or \
                           ("red_peg", "socket-3") in all_contact_pairs or \
                           ("red_peg", "socket-4") in all_contact_pairs
        pin_touched = ("red_peg", "pin") in all_contact_pairs

        reward = 0
        if touch_left_gripper and touch_right_gripper: # touch both
            reward = 1
        if touch_left_gripper and touch_right_gripper and (not peg_touch_table) and (not socket_touch_table): # grasp both
            reward = 2
        if peg_touch_socket and (not peg_touch_table) and (not socket_touch_table): # peg and socket touching
            reward = 3
        if pin_touched: # successful insertion
            reward = 4
        return reward


class BimanualViperXEETaskMobile(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)

    def before_step(self, action, physics):
        a_len = len(action) // 2
        action_left = action[:a_len]
        action_right = action[a_len:]

        # set mocap position and quat
        # left
        np.copyto(physics.data.mocap_pos[0], action_left[:3])
        np.copyto(physics.data.mocap_quat[0], action_left[3:7])
        # right
        np.copyto(physics.data.mocap_pos[1], action_right[:3])
        np.copyto(physics.data.mocap_quat[1], action_right[3:7])

        # set gripper
        g_left_ctrl = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN_MOBILE(action_left[7])
        g_right_ctrl = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN_MOBILE(action_right[7])
        np.copyto(physics.data.ctrl, np.array([g_left_ctrl, g_right_ctrl]))

    def initialize_robots(self, physics):
        # reset joint position
        physics.named.data.qpos[:16] = START_ARM_POSE_MOBILE

        # reset mocap to align with end effector
        # to obtain these numbers:
        # (1) make an ee_sim env and reset to the same start_pose
        # (2) get env._physics.named.data.xpos['vx300s_left/gripper_link']
        #     get env._physics.named.data.xquat['vx300s_left/gripper_link']
        #     repeat the same for right side
        np.copyto(physics.data.mocap_pos[0], [-0.295, 0.70581119, 1.33725084])
        np.copyto(physics.data.mocap_quat[0], [1, 0, 0, 0])
        # right
        np.copyto(physics.data.mocap_pos[1], np.array([0.295, 0.70581119, 1.33725084]))
        np.copyto(physics.data.mocap_quat[1],  [1, 0, 0, 0])

        # reset gripper control
        close_gripper_control = np.array([
            PUPPET_GRIPPER_POSITION_CLOSE_MOBILE,
            PUPPET_GRIPPER_POSITION_CLOSE_MOBILE,
        ])
        np.copyto(physics.data.ctrl, close_gripper_control)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        super().initialize_episode(physics)

    @staticmethod
    def get_qpos(physics):
        qpos_raw = physics.data.qpos.copy()
        left_qpos_raw = qpos_raw[:8]
        right_qpos_raw = qpos_raw[8:16]
        left_arm_qpos = left_qpos_raw[:6]
        right_arm_qpos = right_qpos_raw[:6]
        left_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN_MOBILE(left_qpos_raw[6])]
        right_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN_MOBILE(right_qpos_raw[6])]
        return np.concatenate([left_arm_qpos, left_gripper_qpos, right_arm_qpos, right_gripper_qpos])

    @staticmethod
    def get_qvel(physics):
        qvel_raw = physics.data.qvel.copy()
        left_qvel_raw = qvel_raw[:8]
        right_qvel_raw = qvel_raw[8:16]
        left_arm_qvel = left_qvel_raw[:6]
        right_arm_qvel = right_qvel_raw[:6]
        left_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(left_qvel_raw[6])]
        right_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(right_qvel_raw[6])]
        return np.concatenate([left_arm_qvel, left_gripper_qvel, right_arm_qvel, right_gripper_qvel])

    @staticmethod
    def get_env_state(physics):
        raise NotImplementedError

    def get_observation(self, physics):
        # note: it is important to do .copy()
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos(physics) # waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate, gripper * left or right = 14
        obs['qvel'] = self.get_qvel(physics)
        obs['env_state'] = self.get_env_state(physics) # xyz and quaternion of red box = 4+3  
        obs['images'] = dict()
        obs['images']['top'] = physics.render(height=480, width=640, camera_id='top')
        obs['images']['angle'] = physics.render(height=480, width=640, camera_id='front_cam')
        obs['images']['vis'] = physics.render(height=480, width=640, camera_id='side')
        obs['images']['left_wrist'] = physics.render(height=480, width=640, camera_id='wrist_cam_left')
        obs['images']['right_wrist'] = physics.render(height=480, width=640, camera_id='wrist_cam_right')
        # used in scripted policy to obtain starting pose
        obs['mocap_pose_left'] = np.concatenate([physics.data.mocap_pos[0], physics.data.mocap_quat[0]]).copy()
        obs['mocap_pose_right'] = np.concatenate([physics.data.mocap_pos[1], physics.data.mocap_quat[1]]).copy()

        # used when replaying joint trajectory
        obs['gripper_ctrl'] = physics.data.ctrl.copy()
        return obs

    def get_reward(self, physics):
        raise NotImplementedError
    


class PickAndPutInEETaskMobile(BimanualViperXEETaskMobile):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 1
        self._table_z = 0.827
        self._bucket_height = 0.11
        self._bucket_radius = 0.055 + 0.002
        self.object_name = "_joint"

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self.initialize_robots(physics)
        # randomize box position
        object_and_bucket_pose = sample_obj_and_dst_pose(mobile=True)
        box_start_idx = physics.model.name2id(self.object_name, 'joint')
        np.copyto(physics.data.qpos[box_start_idx : box_start_idx + 7 + 7], object_and_bucket_pose)
        # print(f"randomized cube position to {cube_position}")

        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        # return whether cucumber is in the bucket
        dist_object_to_bucket_center = np.linalg.norm(physics.named.data.qpos[self.object_name][:2] - physics.named.data.qpos["bucket_joint"][:2])
        object_center_z = physics.named.data.qpos[self.object_name][2]
        in_bucket = (dist_object_to_bucket_center < self._bucket_radius) & (self._table_z < object_center_z < self._table_z+self._bucket_height)
        
        reward = 0
        if in_bucket:
            reward = 1
        return reward


class PickCucumberAndPutInEETaskMobile(PickAndPutInEETaskMobile):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.object_name = "cucumber_joint"


class PickCoupleAndPutInEETaskMobile(PickAndPutInEETaskMobile):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.cucumber_box_bucket_pose = None
        self.obj = None

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self.initialize_robots(physics)
        # randomize box position
        if self.obj == None:
            self.cucumber_box_bucket_pose = sample_obj_box_dst_pose(mobile=True)
            self.obj = "red_box"
        elif self.obj == "red_box":
            self.obj = "cucumber"
        else:
             self.cucumber_box_bucket_pose = None
             self.obj = None
        box_start_idx = physics.model.name2id(f'cucumber_joint', 'joint')
        np.copyto(physics.data.qpos[box_start_idx : box_start_idx + 7 + 7 + 7], self.cucumber_box_bucket_pose)
        # print(f"randomized cube position to {cube_position}")

        super(PickAndPutInEETaskMobile, self).initialize_episode(physics)

    def get_reward(self, physics):
        # return whether cucumbert is in the bucket
        dist_cucumber_to_bucket_center = np.linalg.norm(physics.named.data.qpos[f"{self.obj}_joint"][:2] - physics.named.data.qpos["bucket_joint"][:2])
        object_center_z = physics.named.data.qpos[f"{self.obj}_joint"][2]
        in_bucket = (dist_cucumber_to_bucket_center < self._bucket_radius) & (self._table_z < object_center_z < self._table_z+self._bucket_height)
        
        return int(in_bucket)


class FrankaDualEETask(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)

    def before_step(self, action, physics):
        a_len = len(action) // 2
        action_left = action[:a_len]
        action_right = action[a_len:]

        # set mocap position and quat
        # left
        np.copyto(physics.data.mocap_pos[0], action_left[:3])
        np.copyto(physics.data.mocap_quat[0], action_left[3:7])
        # right
        np.copyto(physics.data.mocap_pos[1], action_right[:3])
        np.copyto(physics.data.mocap_quat[1], action_right[3:7])

        # set gripper
        g_left_ctrl = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN_FRANKA(action_left[7])
        g_right_ctrl = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN_FRANKA(action_right[7])
        np.copyto(physics.data.ctrl, np.array([g_left_ctrl, g_right_ctrl]))

    def initialize_robots(self, physics):
        # reset joint position
        physics.named.data.qpos[:18] = START_ARM_POSE_FRANKA

        # reset mocap to align with end effector
        # to obtain these numbers:
        # (1) make an ee_sim env and reset to the same start_pose
        # (2) get env._physics.named.data.xpos['vx300s_left/gripper_link']
        #     get env._physics.named.data.xquat['vx300s_left/gripper_link']
        #     repeat the same for right side
        np.copyto(physics.data.mocap_pos[0], [-0.46210105, 0.96119948, 1.33010105])
        np.copyto(physics.data.mocap_quat[0], [0.6532681955141636, -0.2705659741758004, 0.27063012531805775, -0.6532947677873284])
        # right
        np.copyto(physics.data.mocap_pos[1], np.array([0.46210105, 0.96119948, 1.33010105]))
        np.copyto(physics.data.mocap_quat[1],  [-0.6532947677873284, 0.2706301253180578, 0.27056597417580075, -0.6532681955141635])

        # reset gripper control
        close_gripper_control = np.array([
            PUPPET_GRIPPER_POSITION_CLOSE_FRANKA,
            PUPPET_GRIPPER_POSITION_CLOSE_FRANKA,
        ])
        np.copyto(physics.data.ctrl, close_gripper_control)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        super().initialize_episode(physics)

    @staticmethod
    def get_qpos(physics):
        qpos_raw = physics.data.qpos.copy()
        left_qpos_raw = qpos_raw[:9]
        right_qpos_raw = qpos_raw[9:18]
        left_arm_qpos = left_qpos_raw[:7]
        right_arm_qpos = right_qpos_raw[:7]
        left_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN_FRANKA(left_qpos_raw[7])]
        right_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN_FRANKA(right_qpos_raw[7])]
        return np.concatenate([left_arm_qpos, left_gripper_qpos, right_arm_qpos, right_gripper_qpos])

    @staticmethod
    def get_qvel(physics):
        qvel_raw = physics.data.qvel.copy()
        left_qvel_raw = qvel_raw[:9]
        right_qvel_raw = qvel_raw[9:18]
        left_arm_qvel = left_qvel_raw[:7]
        right_arm_qvel = right_qvel_raw[:7]
        left_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(left_qvel_raw[7])]
        right_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(right_qvel_raw[7])]
        return np.concatenate([left_arm_qvel, left_gripper_qvel, right_arm_qvel, right_gripper_qvel])

    @staticmethod
    def get_env_state(physics):
        raise NotImplementedError

    def get_observation(self, physics):
        # note: it is important to do .copy()
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos(physics) # waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate, gripper * left or right = 14
        obs['qvel'] = self.get_qvel(physics)
        obs['env_state'] = self.get_env_state(physics) # xyz and quaternion of red box = 4+3  
        obs['images'] = dict()
        obs['images']['top'] = physics.render(height=480, width=640, camera_id='top')
        obs['images']['angle'] = physics.render(height=480, width=640, camera_id='front_cam')
        obs['images']['side'] = physics.render(height=480, width=640, camera_id='side')
        obs['images']['back'] = physics.render(height=480, width=640, camera_id='back')
        obs['images']['left_wrist'] = physics.render(height=480, width=640, camera_id='wrist_cam_left')
        obs['images']['right_wrist'] = physics.render(height=480, width=640, camera_id='wrist_cam_right')
        # used in scripted policy to obtain starting pose
        obs['mocap_pose_left'] = np.concatenate([physics.data.mocap_pos[0], physics.data.mocap_quat[0]]).copy()
        obs['mocap_pose_right'] = np.concatenate([physics.data.mocap_pos[1], physics.data.mocap_quat[1]]).copy()

        # used when replaying joint trajectory
        obs['gripper_ctrl'] = physics.data.ctrl.copy()
        return obs

    def get_reward(self, physics):
        raise NotImplementedError
    

class PickAndPutInEETaskFranka(FrankaDualEETask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 1
        self._table_z = 0.827
        self._bucket_height = 0.11
        self._bucket_radius = 0.055 + 0.002
        self.object_name = "_joint"

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self.initialize_robots(physics)
        # randomize box position
        object_and_bucket_pose = sample_obj_and_dst_pose(mobile=True)
        box_start_idx = physics.model.name2id(self.object_name, 'joint')
        np.copyto(physics.data.qpos[box_start_idx : box_start_idx + 7 + 7], object_and_bucket_pose)
        # print(f"randomized cube position to {cube_position}")

        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[18:]
        return env_state

    def get_reward(self, physics):
        # return whether cucumber is in the bucket
        dist_object_to_bucket_center = np.linalg.norm(physics.named.data.qpos[self.object_name][:2] - physics.named.data.qpos["bucket_joint"][:2])
        object_center_z = physics.named.data.qpos[self.object_name][2]
        in_bucket = (dist_object_to_bucket_center < self._bucket_radius) & (self._table_z < object_center_z < self._table_z+self._bucket_height)
        
        reward = 0
        if in_bucket:
            reward = 1
        return reward

class PickCucumberAndPutInEETaskFranka(PickAndPutInEETaskFranka):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.object_name = "cucumber_joint"


class PickCushionAndPutInCardboardEETaskFranka(PickAndPutInEETaskFranka):
    def __init__(self, random=None):
        super().__init__(random=random)
        self._cushion_joint_name = "cushion_joint"
        self._cardboard_joint_name = "cardboard_joint"
        self._cardboard_btm_geom_name = "sku_cardboard_btm"
        self._cushion_geom_name = "cushion_lower"

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self.initialize_robots(physics)
        self.cardboard_qpos = self.sample_cardboard_quat()
        start_idx = physics.model.name2id(self._cushion_joint_name, 'joint')
        np.copyto(physics.data.qpos[start_idx+7 : start_idx+7+7], self.cardboard_qpos)
        super(PickAndPutInEETaskFranka, self).initialize_episode(physics)

        self._cardboard_btm_size = physics.model.geom(self._cardboard_btm_geom_name).size
        self._cardboard_half_w, self._cardboard_half_h, self._cardboard_half_th = self._cardboard_btm_size
        self._cushion_size = physics.model.geom(self._cushion_geom_name).size
        self._cushion_half_w, self._cushion_half_h, self._cushion_half_th = self._cushion_size

    def get_reward(self, physics):
        cardboard_qpos = physics.data.joint(self._cardboard_joint_name).qpos
        cardboard_xyz = cardboard_qpos[:3]
        cardboard_quat = cardboard_qpos[3:]
        min_x, max_x = cardboard_xyz[0] - self._cardboard_half_w, cardboard_xyz[0] + self._cardboard_half_w
        min_y, max_y = cardboard_xyz[1] - self._cardboard_half_h, cardboard_xyz[1] + self._cardboard_half_h

        cushion_qpos = physics.data.joint(self._cushion_joint_name).qpos
        cushion_xyz = cushion_qpos[:3]
        cushion_xy_rel = cushion_xyz[:2] - cardboard_xyz[:2]
        angle = quaternion_to_euler(cardboard_quat, degrees=False)[2]
        cushion_xy_rel_rot = np.array([[np.cos(-angle), -np.sin(-angle)], [np.sin(-angle), np.cos(-angle)]]) @ cushion_xy_rel
        cushion_xy_rot = cushion_xy_rel_rot + cardboard_xyz[:2]

        in_cardboard = (min_x < cushion_xy_rot[0] < max_x) & (min_y < cushion_xy_rot[1] < max_y) & (cardboard_xyz[2] + self._cardboard_half_th < cushion_xyz[2] < cardboard_xyz[2] + self._cardboard_half_th + self._cushion_half_h*2)
        
        return int(in_cardboard)
    
    def sample_cardboard_quat(self):
        angle_range = [-180, 180] #-180, 180
        x_range = [0.1, 0.3] # 0.1, 0.3
        y_range = [0.88, 0.98] # 0.88, 0.98
        z_range = [0.7, 0.7]

        ranges = np.vstack([x_range, y_range, z_range])
        pos = np.random.uniform(ranges[:, 0], ranges[:, 1])
        angle = np.random.uniform(angle_range[0], angle_range[1])    
        quat = np.array([np.cos(np.deg2rad(angle)/2), 0, 0, np.sin(np.deg2rad(angle)/2)])    

        return np.concatenate([pos, quat])


class PickCushionAndPutInCardboardRecoveryEETaskFranka(PickAndPutInEETaskFranka):
    def __init__(self, random=None):
        super().__init__(random=random)
        self._cushion_joint_name = "cushion_recovery_joint"
        self._cardboard_joint_name = "cardboard_joint"
        self._cardboard_btm_geom_name = "sku_cardboard_btm"
        self._cushion_geom_name = "cushion_recovery_lower"
        self._start_idx = 18

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self.initialize_robots(physics)
        self.cushion_cardboard_pose = self.sample_cushion_cardboard_pose()
        np.copyto(physics.data.qpos[self._start_idx+7 : self._start_idx+7+7+7], self.cushion_cardboard_pose)
        super(PickAndPutInEETaskFranka, self).initialize_episode(physics)

        self._cardboard_btm_size = physics.model.geom(self._cardboard_btm_geom_name).size
        self._cardboard_half_w, self._cardboard_half_h, self._cardboard_half_th = self._cardboard_btm_size
        self._cushion_size = physics.model.geom(self._cushion_geom_name).size
        self._cushion_half_w, self._cushion_half_h, self._cushion_half_th = self._cushion_size

    def get_reward(self, physics):
        cardboard_qpos = physics.data.joint(self._cardboard_joint_name).qpos
        cardboard_xyz = cardboard_qpos[:3]
        cardboard_quat = cardboard_qpos[3:]
        min_x, max_x = cardboard_xyz[0] - self._cardboard_half_w, cardboard_xyz[0] + self._cardboard_half_w
        min_y, max_y = cardboard_xyz[1] - self._cardboard_half_h, cardboard_xyz[1] + self._cardboard_half_h

        cushion_qpos = physics.data.joint(self._cushion_joint_name).qpos
        cushion_xyz = cushion_qpos[:3]
        cushion_xy_rel = cushion_xyz[:2] - cardboard_xyz[:2]
        angle = quaternion_to_euler(cardboard_quat, degrees=False)[2]
        cushion_xy_rel_rot = np.array([[np.cos(-angle), -np.sin(-angle)], [np.sin(-angle), np.cos(-angle)]]) @ cushion_xy_rel
        cushion_xy_rot = cushion_xy_rel_rot + cardboard_xyz[:2]

        in_cardboard = (min_x < cushion_xy_rot[0] < max_x) & (min_y < cushion_xy_rot[1] < max_y) & (cardboard_xyz[2] + self._cardboard_half_th < cushion_xyz[2] < cardboard_xyz[2] + self._cardboard_half_th + self._cushion_half_h*2)
        
        return int(in_cardboard)
    
    def sample_cushion_cardboard_pose(self):
        cardboard_angle_range = [-180, 180] #-180, 180
        cardboard_x_range = [0.1, 0.3] # 0.1, 0.3
        cardboard_y_range = [0.88, 0.98] # 0.88, 0.98
        cardboard_z_range = [0.7, 0.7]

        cushion_angle_range = [-180, 180] #-180, 180
        cushion_x_range = [0.71, 0.81] # 0.1, 0.3
        cushion_y_range = [0.88, 0.98] # 0.88, 0.98
        cushion_z_range = [0.717, 0.717]

        cardboard_ranges = np.vstack([cardboard_x_range, cardboard_y_range, cardboard_z_range])
        cardboard_pos = np.random.uniform(cardboard_ranges[:, 0], cardboard_ranges[:, 1])
        cardboard_angle = np.random.uniform(cardboard_angle_range[0], cardboard_angle_range[1])    
        cardboard_quat = np.array([np.cos(np.deg2rad(cardboard_angle)/2), 0, 0, np.sin(np.deg2rad(cardboard_angle)/2)])    

        cushion_ranges = np.vstack([cushion_x_range, cushion_y_range, cushion_z_range])
        cushion_pos = np.random.uniform(cushion_ranges[:, 0], cushion_ranges[:, 1])
        cushion_angle = np.random.uniform(cushion_angle_range[0], cushion_angle_range[1])    
        cushion_quat = np.array([np.cos(np.deg2rad(cushion_angle)/2), 0, 0, np.sin(np.deg2rad(cushion_angle)/2)])            

        return np.concatenate([cushion_pos, cushion_quat, cardboard_pos, cardboard_quat])


class PickCoupleAndPutInEETaskFranka(PickAndPutInEETaskFranka):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.cucumber_box_bucket_pose = None
        self.obj = None

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self.initialize_robots(physics)
        # randomize box position
        if self.obj == None:
            self.cucumber_box_bucket_pose = sample_obj_box_dst_pose(mobile=True)
            self.obj = "red_box"
        elif self.obj == "red_box":
            self.obj = "cucumber"
        else:
             self.cucumber_box_bucket_pose = None
             self.obj = None
        box_start_idx = physics.model.name2id(f'cucumber_joint', 'joint')
        np.copyto(physics.data.qpos[box_start_idx : box_start_idx + 7 + 7 + 7], self.cucumber_box_bucket_pose)
        # print(f"randomized cube position to {cube_position}")

        super(PickAndPutInEETaskFranka, self).initialize_episode(physics)

    def get_reward(self, physics):
        # return whether cucumbert is in the bucket
        dist_cucumber_to_bucket_center = np.linalg.norm(physics.named.data.qpos[f"{self.obj}_joint"][:2] - physics.named.data.qpos["bucket_joint"][:2])
        object_center_z = physics.named.data.qpos[f"{self.obj}_joint"][2]
        in_bucket = (dist_cucumber_to_bucket_center < self._bucket_radius) & (self._table_z < object_center_z < self._table_z+self._bucket_height)
        
        return int(in_bucket)
    

class PickQuadrupleAndPutInEETaskFrankaBimanual(PickAndPutInEETaskFranka):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.two_objs_bucket_pose = None
        self.target = "left_first_cube"

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self.initialize_robots(physics)
        # randomize box position

        if self.target == "left_first_cube":
            self.two_objs_bucket_pose = sample_2objs_and_dst_pose()
            cube_pose = self.two_objs_bucket_pose[0:7].copy()
            cucumber_pose = self.two_objs_bucket_pose[7:14].copy()
            self.target = "right_first_cube"
        elif self.target == "right_first_cube":
            cube_pose = self.two_objs_bucket_pose[7:14].copy()
            cucumber_pose = self.two_objs_bucket_pose[0:7].copy()
            self.target = "left_first_cucumber"
        elif self.target == "left_first_cucumber":
            cube_pose = self.two_objs_bucket_pose[7:14].copy()
            cucumber_pose = self.two_objs_bucket_pose[0:7].copy()
            self.target = "right_first_cucumber"
        elif self.target == "right_first_cucumber":
            cube_pose = self.two_objs_bucket_pose[0:7].copy()
            cucumber_pose = self.two_objs_bucket_pose[7:14].copy()
            self.target = "left_first_cube"
        bucket_pose = self.two_objs_bucket_pose[14:21].copy()

        self.cube_in_bucket = None
        self.cucumber_in_bucket = None
        self.cube_rewards = 0
        self.cucumber_rewards = 0

        pose = np.concatenate([cucumber_pose, cube_pose, bucket_pose])

        cucumber_start_idx = physics.model.name2id(f'cucumber_joint', 'joint')
        np.copyto(physics.data.qpos[cucumber_start_idx : cucumber_start_idx + 7*3], pose)

        super(PickAndPutInEETaskFranka, self).initialize_episode(physics)

    def get_reward(self, physics):
        dist_cucumber_to_bucket_center = np.linalg.norm(physics.named.data.qpos[f"cucumber_joint"][:2] - physics.named.data.qpos["bucket_joint"][:2])
        dist_cube_to_bucket_center = np.linalg.norm(physics.named.data.qpos[f"red_box_joint"][:2] - physics.named.data.qpos["bucket_joint"][:2])        

        self.cucumber_in_bucket = (dist_cucumber_to_bucket_center < self._bucket_radius)
        self.cube_in_bucket = (dist_cube_to_bucket_center < self._bucket_radius)
        
        if self.cucumber_in_bucket:
            self.cucumber_rewards += 1
        elif self.cube_in_bucket:
            self.cube_rewards += 1
        if self.cucumber_in_bucket & self.cube_in_bucket:
            reward = 1
        else:
            reward = 0
        return reward
