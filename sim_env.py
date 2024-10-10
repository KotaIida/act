import numpy as np
import os
import collections
import matplotlib.pyplot as plt
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from scipy.spatial.transform import Rotation as R

from constants import DT, XML_DIR_STATIC, XML_DIR_MOBILE, XML_DIR_FRANKA, START_ARM_POSE, START_ARM_POSE_MOBILE , START_ARM_POSE_FRANKA
from constants import PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN, PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN_MOBILE, PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN_FRANKA
from constants import MASTER_GRIPPER_POSITION_NORMALIZE_FN
from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN, PUPPET_GRIPPER_POSITION_NORMALIZE_FN_MOBILE, PUPPET_GRIPPER_POSITION_NORMALIZE_FN_FRANKA
from constants import PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN

import IPython
e = IPython.embed

BOX_POSE = [None] # to be changed from outside

def quaternion_to_euler(q, degrees=True):
    r = R.from_quat([q[1], q[2], q[3], q[0]])
    return r.as_euler('xyz', degrees=degrees)

def make_sim_env(task_name, **kwargs):
    """
    Environment for simulated robot bi-manual manipulation, with joint position control
    Action space:      [left_arm_qpos (6),             # absolute joint position
                        left_gripper_positions (1),    # normalized gripper position (0: close, 1: open)
                        right_arm_qpos (6),            # absolute joint position
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
        xml_path = os.path.join(XML_DIR_STATIC, f'bimanual_viperx_transfer_cube.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = TransferCubeTask(random=False)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'sim_put_cucumber_in_bucket_on_static_aloha' == task_name:
        xml_path = os.path.join(XML_DIR_STATIC, f'bimanual_viperx_put_cucumber_in_bucket.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = PickCucumberAndPutInTask(random=False)
        env = control.Environment(physics, task, time_limit=50, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'sim_put_cube_in_bucket_on_static_aloha' == task_name:
        xml_path = os.path.join(XML_DIR_STATIC, f'bimanual_viperx_put_cube_in_bucket.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = PickCubeAndPutInTask(random=False)
        env = control.Environment(physics, task, time_limit=50, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'sim_put_couple_in_bucket_on_static_aloha' == task_name:
        xml_path = os.path.join(XML_DIR_STATIC, f'bimanual_viperx_put_couple_in_bucket.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = PickCoupleAndPutInTask(random=False)
        env = control.Environment(physics, task, time_limit=50, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'sim_put_multiple_in_bucket_on_static_aloha' == task_name:
        xml_path = os.path.join(XML_DIR_STATIC, f'bimanual_viperx_put_multiple_in_bucket.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        pickup_num = kwargs["pickup_num"]
        task = PickMultipleAndPutInTask(random=False, pickup_num=pickup_num)
        env = control.Environment(physics, task, time_limit=50, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'sim_insertion_scripted' in task_name:
        xml_path = os.path.join(XML_DIR_STATIC, f'bimanual_viperx_insertion.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = InsertionTask(random=False)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'sim_put_cucumber_in_bucket_on_mobile_aloha' == task_name:
        xml_path = os.path.join(XML_DIR_MOBILE, f'bimanual_viperx_put_cucumber_in_bucket.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = PickCucumberAndPutInTaskMobile(random=False)
        env = control.Environment(physics, task, time_limit=50, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'sim_put_couple_in_bucket_on_mobile_aloha' == task_name:
        xml_path = os.path.join(XML_DIR_MOBILE, f'bimanual_viperx_put_couple_in_bucket.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = PickCoupleAndPutInTaskMobile(random=False)
        env = control.Environment(physics, task, time_limit=50, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'sim_put_cucumber_in_bucket_on_franka_dual' == task_name:
        xml_path = os.path.join(XML_DIR_FRANKA, f'franka_dual_put_cucumber_in_bucket.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = PickCucumberAndPutInTaskFranka(random=False)
        env = control.Environment(physics, task, time_limit=50, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'sim_put_couple_in_bucket_on_franka_dual' == task_name:
        xml_path = os.path.join(XML_DIR_FRANKA, f'franka_dual_put_couple_in_bucket.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = PickCoupleAndPutInTaskFranka(random=False)
        env = control.Environment(physics, task, time_limit=50, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'sim_put_quadruple_in_bucket_on_franka_dual_bimanual' == task_name:
        xml_path = os.path.join(XML_DIR_FRANKA, f'franka_dual_put_couple_in_bucket.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = PickQuadrupleAndPutInTaskFrankaBimanual(random=False)
        env = control.Environment(physics, task, time_limit=50, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'sim_put_cushion_in_cardboard_v_on_franka_dual' == task_name:
        xml_path = os.path.join(XML_DIR_FRANKA, f'franka_dual_put_cushion_in_cardboard_v.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = PickCushionAndPutInCardboardTaskFranka(random=False)
        env = control.Environment(physics, task, time_limit=50, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'sim_put_cushion_in_cardboard_v_recovery_on_franka_dual' == task_name:
        xml_path = os.path.join(XML_DIR_FRANKA, f'franka_dual_put_cushion_in_cardboard_v_recovery.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = PickCushionAndPutInCardboardRecoveryTaskFranka(random=False)
        env = control.Environment(physics, task, time_limit=50, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'sim_put_cushion_in_cardboard_h_on_franka_dual' == task_name:
        xml_path = os.path.join(XML_DIR_FRANKA, f'franka_dual_put_cushion_in_cardboard_h.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = PickCushionAndPutInCardboardTaskFranka(random=False)
        env = control.Environment(physics, task, time_limit=50, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'sim_put_cushion_in_cardboard_v_eval_on_franka_dual' == task_name:
        xml_path = os.path.join(XML_DIR_FRANKA, f'franka_dual_put_cushion_in_cardboard_v_eval.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = PickCushionAndPutInCardboardTaskFranka(random=False)
        env = control.Environment(physics, task, time_limit=50, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'sim_put_cushion_in_cardboard_h_eval_on_franka_dual' == task_name:
        xml_path = os.path.join(XML_DIR_FRANKA, f'franka_dual_put_cushion_in_cardboard_h_eval.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = PickCushionAndPutInCardboardTaskFranka(random=False)
        env = control.Environment(physics, task, time_limit=50, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    else:
        raise NotImplementedError
    return env

class BimanualViperXTask(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)

    def before_step(self, action, physics):
        left_arm_action = action[:6]
        right_arm_action = action[7:7+6]
        normalized_left_gripper_action = action[6]
        normalized_right_gripper_action = action[7+6]

        left_gripper_action = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(normalized_left_gripper_action)
        right_gripper_action = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(normalized_right_gripper_action)

        full_left_gripper_action = [left_gripper_action, -left_gripper_action]
        full_right_gripper_action = [right_gripper_action, -right_gripper_action]

        env_action = np.concatenate([left_arm_action, full_left_gripper_action, right_arm_action, full_right_gripper_action])
        super().before_step(env_action, physics)
        return

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
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos(physics)
        obs['qvel'] = self.get_qvel(physics)
        obs['env_state'] = self.get_env_state(physics)
        obs['images'] = dict()
        obs['images']['top'] = physics.render(height=480, width=640, camera_id='top')
        obs['images']['angle'] = physics.render(height=480, width=640, camera_id='angle')
        obs['images']['vis'] = physics.render(height=480, width=640, camera_id='front_close')
        obs['images']['left_wrist'] = physics.render(height=480, width=640, camera_id='left_wrist')
        obs['images']['right_wrist'] = physics.render(height=480, width=640, camera_id='right_wrist')

        return obs

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        raise NotImplementedError


class TransferCubeTask(BimanualViperXTask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE
            np.copyto(physics.data.ctrl, START_ARM_POSE)
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-7:] = BOX_POSE[0]
            # print(f"{BOX_POSE=}")
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

class PickAndPutInTask(BimanualViperXTask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 1
        self._bucket_height = 0.11 + 0.002
        self._bucket_radius = 0.055 + 0.002
        self.object_name = "_joint"

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE
            np.copyto(physics.data.ctrl, START_ARM_POSE)
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-14:] = BOX_POSE[0]
            # print(f"{BOX_POSE=}")
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        # return whether cucumber is in the bucket
        dist_cucumber_to_bucket_center = np.linalg.norm(physics.named.data.qpos[self.object_name][:2] - physics.named.data.qpos["bucket_joint"][:2])
        cucumber_center_z = physics.named.data.qpos[self.object_name][2]
        in_bucket = (dist_cucumber_to_bucket_center < self._bucket_radius) & (cucumber_center_z < self._bucket_height)
        
        reward = 0
        if in_bucket:
            reward = 1
        return reward
    

class PickCucumberAndPutInTask(PickAndPutInTask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.object_name = "cucumber_joint"        

    
class PickCubeAndPutInTask(PickAndPutInTask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.object_name = "red_box_joint"


class PickCoupleAndPutInTask(PickAndPutInTask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.obj = None

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE
            np.copyto(physics.data.ctrl, START_ARM_POSE)
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-21:] = BOX_POSE[0]
            # print(f"{BOX_POSE=}")
        if self.obj == None:
            self.obj = "red_box"
        elif self.obj == "red_box":
            self.obj = "cucumber"
        else:
            self.obj = None
        self.cube_in_bucket = None
        self.cucumber_in_bucket = None
        self.cube_rewards = 0
        self.cucumber_rewards = 0
        super(PickAndPutInTask, self).initialize_episode(physics)

    def get_reward(self, physics):
        # return whether cucumber is in the bucket
        dist_cucumber_to_bucket_center = np.linalg.norm(physics.named.data.qpos[f"cucumber_joint"][:2] - physics.named.data.qpos["bucket_joint"][:2])
        dist_cube_to_bucket_center = np.linalg.norm(physics.named.data.qpos[f"red_box_joint"][:2] - physics.named.data.qpos["bucket_joint"][:2])        
        cucumber_center_z = physics.named.data.qpos[f"cucumber_joint"][2]
        cube_center_z = physics.named.data.qpos[f"red_box_joint"][2]

        self.cucumber_in_bucket = (dist_cucumber_to_bucket_center < self._bucket_radius) & (cucumber_center_z < self._bucket_height)
        self.cube_in_bucket = (dist_cube_to_bucket_center < self._bucket_radius) & (cube_center_z < self._bucket_height)
        
        if self.cucumber_in_bucket:
            reward = 1
            self.cucumber_rewards += 1
        elif self.cube_in_bucket:
            reward = 1
            self.cube_rewards += 1
        else:
            reward = 0
        return reward


class PickMultipleAndPutInTask(PickAndPutInTask):
    def __init__(self, random=None, pickup_num=3):
        super().__init__(random=random)
        self.pickup_num = pickup_num
        self.cucumber_types = ["a", "b", "c", "d"]

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE
            np.copyto(physics.data.ctrl, START_ARM_POSE)
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-35:] = BOX_POSE[0]
            # print(f"{BOX_POSE=}")
        super(PickAndPutInTask, self).initialize_episode(physics)
    
    def get_reward(self, physics):
        # return whether cucumber is in the bucket
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


class InsertionTask(BimanualViperXTask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE
            np.copyto(physics.data.ctrl, START_ARM_POSE)
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-7*2:] = BOX_POSE[0] # two objects
            # print(f"{BOX_POSE=}")
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


class BimanualViperXTaskMobile(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)

    def before_step(self, action, physics):
        left_arm_action = action[:6]
        right_arm_action = action[7:7+6]
        normalized_left_gripper_action = action[6]
        normalized_right_gripper_action = action[7+6]

        left_gripper_action = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN_MOBILE(normalized_left_gripper_action)
        right_gripper_action = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN_MOBILE(normalized_right_gripper_action)

        full_left_gripper_action = [left_gripper_action]
        full_right_gripper_action = [right_gripper_action]

        env_action = np.concatenate([left_arm_action, full_left_gripper_action, right_arm_action, full_right_gripper_action])
        super().before_step(env_action, physics)
        return

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
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos(physics)
        obs['qvel'] = self.get_qvel(physics)
        obs['env_state'] = self.get_env_state(physics)
        obs['images'] = dict()
        obs['images']['top'] = physics.render(height=480, width=640, camera_id='top')
        obs['images']['angle'] = physics.render(height=480, width=640, camera_id='front_cam')
        obs['images']['side'] = physics.render(height=480, width=640, camera_id='sie')
        obs['images']['back'] = physics.render(height=480, width=640, camera_id='back')
        obs['images']['vis'] = physics.render(height=480, width=640, camera_id='side')
        obs['images']['left_wrist'] = physics.render(height=480, width=640, camera_id='wrist_cam_left')
        obs['images']['right_wrist'] = physics.render(height=480, width=640, camera_id='wrist_cam_right')

        return obs

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        raise NotImplementedError


class PickAndPutInTaskMobile(BimanualViperXTaskMobile):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 1
        self._table_z = 0.827
        self._bucket_height = 0.11
        self._bucket_radius = 0.055 + 0.002
        self.object_name = "_joint"

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE_MOBILE
            np.copyto(physics.data.ctrl, np.hstack([START_ARM_POSE_MOBILE[:7] , START_ARM_POSE_MOBILE[8:15]]))
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-14:] = BOX_POSE[0]
            # print(f"{BOX_POSE=}")
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        # return whether cucumber is in the bucket
        dist_cucumber_to_bucket_center = np.linalg.norm(physics.named.data.qpos[self.object_name][:2] - physics.named.data.qpos["bucket_joint"][:2])
        cucumber_center_z = physics.named.data.qpos[self.object_name][2]
        in_bucket = (dist_cucumber_to_bucket_center < self._bucket_radius) & (self._table_z < cucumber_center_z < self._table_z+self._bucket_height)
        
        reward = 0
        if in_bucket:
            reward = 1
        return reward
    

class PickCucumberAndPutInTaskMobile(PickAndPutInTaskMobile):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.object_name = "cucumber_joint"        


class PickCoupleAndPutInTaskMobile(PickAndPutInTaskMobile):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.obj = None

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE_MOBILE
            np.copyto(physics.data.ctrl, np.hstack([START_ARM_POSE_MOBILE[:7] , START_ARM_POSE_MOBILE[8:15]]))
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-21:] = BOX_POSE[0]
            # print(f"{BOX_POSE=}")
        if self.obj == None:
            self.obj = "red_box"
        elif self.obj == "red_box":
            self.obj = "cucumber"
        else:
            self.obj = None
        self.cube_in_bucket = None
        self.cucumber_in_bucket = None
        self.cube_rewards = 0
        self.cucumber_rewards = 0
        super(PickAndPutInTaskMobile, self).initialize_episode(physics)

    def get_reward(self, physics):
        # return whether cucumber is in the bucket
        dist_cucumber_to_bucket_center = np.linalg.norm(physics.named.data.qpos[f"cucumber_joint"][:2] - physics.named.data.qpos["bucket_joint"][:2])
        dist_cube_to_bucket_center = np.linalg.norm(physics.named.data.qpos[f"red_box_joint"][:2] - physics.named.data.qpos["bucket_joint"][:2])        
        cucumber_center_z = physics.named.data.qpos[f"cucumber_joint"][2]
        cube_center_z = physics.named.data.qpos[f"red_box_joint"][2]

        self.cucumber_in_bucket = (dist_cucumber_to_bucket_center < self._bucket_radius) & (self._table_z < cucumber_center_z < self._table_z+self._bucket_height)
        self.cube_in_bucket = (dist_cube_to_bucket_center < self._bucket_radius) & (self._table_z < cube_center_z < self._table_z+self._bucket_height)
        
        if self.cucumber_in_bucket:
            reward = 1
            self.cucumber_rewards += 1
        elif self.cube_in_bucket:
            reward = 1
            self.cube_rewards += 1
        else:
            reward = 0
        return reward
    

class FrankaDualTask(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)

    def before_step(self, action, physics):
        left_arm_action = action[:7]
        right_arm_action = action[8:8+7]
        normalized_left_gripper_action = action[7]
        normalized_right_gripper_action = action[8+7]

        left_gripper_action = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN_FRANKA(normalized_left_gripper_action)
        right_gripper_action = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN_FRANKA(normalized_right_gripper_action)

        full_left_gripper_action = [left_gripper_action]
        full_right_gripper_action = [right_gripper_action]

        env_action = np.concatenate([left_arm_action, full_left_gripper_action, right_arm_action, full_right_gripper_action])
        super().before_step(env_action, physics)
        return

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
        left_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN_MOBILE(left_qpos_raw[7])]
        right_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN_MOBILE(right_qpos_raw[7])]
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
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos(physics)
        obs['qvel'] = self.get_qvel(physics)
        obs['env_state'] = self.get_env_state(physics)
        obs['images'] = dict()
        obs['images']['top'] = physics.render(height=480, width=640, camera_id='top')
        obs['images']['angle'] = physics.render(height=480, width=640, camera_id='front_cam')
        obs['images']['side'] = physics.render(height=480, width=640, camera_id='side')
        obs['images']['back'] = physics.render(height=480, width=640, camera_id='back')
        obs['images']['left_wrist'] = physics.render(height=480, width=640, camera_id='wrist_cam_left')
        obs['images']['right_wrist'] = physics.render(height=480, width=640, camera_id='wrist_cam_right')

        return obs

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        raise NotImplementedError


class PickAndPutInTaskFranka(FrankaDualTask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 1
        self._table_z = 0.827
        self._bucket_height = 0.11
        self._bucket_radius = 0.055 + 0.002
        self.object_name = "_joint"

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            physics.named.data.qpos[:18] = START_ARM_POSE_FRANKA
            np.copyto(physics.data.ctrl, np.hstack([START_ARM_POSE_FRANKA[:8] , START_ARM_POSE_FRANKA[9:17]]))
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-14:] = BOX_POSE[0]
            # print(f"{BOX_POSE=}")
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[18:]
        return env_state

    def get_reward(self, physics):
        # return whether cucumber is in the bucket
        dist_cucumber_to_bucket_center = np.linalg.norm(physics.named.data.qpos[self.object_name][:2] - physics.named.data.qpos["bucket_joint"][:2])
        cucumber_center_z = physics.named.data.qpos[self.object_name][2]
        in_bucket = (dist_cucumber_to_bucket_center < self._bucket_radius) & (self._table_z < cucumber_center_z < self._table_z+self._bucket_height)
        
        reward = 0
        if in_bucket:
            reward = 1
        return reward
    

class PickCucumberAndPutInTaskFranka(PickAndPutInTaskFranka):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.object_name = "cucumber_joint"        


class PickCushionAndPutInCardboardTaskFranka(PickAndPutInTaskFranka):
    def __init__(self, random=None):
        super().__init__(random=random)
        self._cushion_joint_name = "cushion_joint"
        self._cardboard_joint_name = "cardboard_joint"
        self._cardboard_btm_geom_name = "sku_cardboard_btm"
        self._cushion_geom_name = "cushion_lower"

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        with physics.reset_context():
            physics.named.data.qpos[:18] = START_ARM_POSE_FRANKA
            np.copyto(physics.data.ctrl, np.hstack([START_ARM_POSE_FRANKA[:8] , START_ARM_POSE_FRANKA[9:17]]))
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-14:] = BOX_POSE[0]
            # print(f"{BOX_POSE=}")
        super(PickAndPutInTaskFranka, self).initialize_episode(physics)

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
    

class PickCushionAndPutInCardboardRecoveryTaskFranka(PickAndPutInTaskFranka):
    def __init__(self, random=None):
        super().__init__(random=random)
        self._cushion_joint_name = "cushion_recovery_joint"
        self._cardboard_joint_name = "cardboard_joint"
        self._cardboard_btm_geom_name = "sku_cardboard_btm"
        self._cushion_geom_name = "cushion_recovery_lower"

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        with physics.reset_context():
            physics.named.data.qpos[:18] = START_ARM_POSE_FRANKA
            np.copyto(physics.data.ctrl, np.hstack([START_ARM_POSE_FRANKA[:8] , START_ARM_POSE_FRANKA[9:17]]))
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-21:] = BOX_POSE[0]
            # print(f"{BOX_POSE=}")
        super(PickAndPutInTaskFranka, self).initialize_episode(physics)

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


class PickCoupleAndPutInTaskFranka(PickAndPutInTaskFranka):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.obj = None

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            physics.named.data.qpos[:18] = START_ARM_POSE_FRANKA
            np.copyto(physics.data.ctrl, np.hstack([START_ARM_POSE_FRANKA[:8] , START_ARM_POSE_FRANKA[9:17]]))
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-21:] = BOX_POSE[0]
            # print(f"{BOX_POSE=}")
        if self.obj == None:
            self.obj = "red_box"
        elif self.obj == "red_box":
            self.obj = "cucumber"
        else:
            self.obj = None
        self.cube_in_bucket = None
        self.cucumber_in_bucket = None
        self.cube_rewards = 0
        self.cucumber_rewards = 0
        super(PickAndPutInTaskFranka, self).initialize_episode(physics)

    def get_reward(self, physics):
        # return whether cucumber is in the bucket
        dist_cucumber_to_bucket_center = np.linalg.norm(physics.named.data.qpos[f"cucumber_joint"][:2] - physics.named.data.qpos["bucket_joint"][:2])
        dist_cube_to_bucket_center = np.linalg.norm(physics.named.data.qpos[f"red_box_joint"][:2] - physics.named.data.qpos["bucket_joint"][:2])        
        cucumber_center_z = physics.named.data.qpos[f"cucumber_joint"][2]
        cube_center_z = physics.named.data.qpos[f"red_box_joint"][2]

        self.cucumber_in_bucket = (dist_cucumber_to_bucket_center < self._bucket_radius) & (self._table_z < cucumber_center_z < self._table_z+self._bucket_height)
        self.cube_in_bucket = (dist_cube_to_bucket_center < self._bucket_radius) & (self._table_z < cube_center_z < self._table_z+self._bucket_height)
        
        if self.cucumber_in_bucket:
            reward = 1
            self.cucumber_rewards += 1
        elif self.cube_in_bucket:
            reward = 1
            self.cube_rewards += 1
        else:
            reward = 0
        return reward
    


class PickCoupleAndPutInTaskFrankaBimanual(PickAndPutInTaskFranka):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.obj = None

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            physics.named.data.qpos[:18] = START_ARM_POSE_FRANKA
            np.copyto(physics.data.ctrl, np.hstack([START_ARM_POSE_FRANKA[:8] , START_ARM_POSE_FRANKA[9:17]]))
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-21:] = BOX_POSE[0]
            # print(f"{BOX_POSE=}")
        if self.obj == None:
            self.obj = "red_box"
        elif self.obj == "red_box":
            self.obj = "cucumber"
        else:
            self.obj = None
        self.cube_in_bucket = None
        self.cucumber_in_bucket = None
        self.cube_rewards = 0
        self.cucumber_rewards = 0
        super(PickAndPutInTaskFranka, self).initialize_episode(physics)

    def get_reward(self, physics):
        # return whether cucumber is in the bucket
        dist_cucumber_to_bucket_center = np.linalg.norm(physics.named.data.qpos[f"cucumber_joint"][:2] - physics.named.data.qpos["bucket_joint"][:2])
        dist_cube_to_bucket_center = np.linalg.norm(physics.named.data.qpos[f"red_box_joint"][:2] - physics.named.data.qpos["bucket_joint"][:2])        
        cucumber_center_z = physics.named.data.qpos[f"cucumber_joint"][2]
        cube_center_z = physics.named.data.qpos[f"red_box_joint"][2]

        self.cucumber_in_bucket = (dist_cucumber_to_bucket_center < self._bucket_radius)
        self.cube_in_bucket = (dist_cube_to_bucket_center < self._bucket_radius)
        
        if self.cucumber_in_bucket:
            reward = 1
            self.cucumber_rewards += 1
        elif self.cube_in_bucket:
            reward = 1
            self.cube_rewards += 1
        if self.cucumber_in_bucket & self.cube_in_bucket:
            reward = 1
        else:
            reward = 0
        return reward


class PickQuadrupleAndPutInTaskFrankaBimanual(PickAndPutInTaskFranka):
    def __init__(self, random=None):
        super().__init__(random=random)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            physics.named.data.qpos[:18] = START_ARM_POSE_FRANKA
            np.copyto(physics.data.ctrl, np.hstack([START_ARM_POSE_FRANKA[:8] , START_ARM_POSE_FRANKA[9:17]]))
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-21:] = BOX_POSE[0]

        self.cube_in_bucket = None
        self.cucumber_in_bucket = None
        self.cube_rewards = 0
        self.cucumber_rewards = 0
        super(PickAndPutInTaskFranka, self).initialize_episode(physics)

    def get_reward(self, physics):
        dist_cucumber_to_bucket_center = np.linalg.norm(physics.named.data.qpos[f"cucumber_joint"][:2] - physics.named.data.qpos["bucket_joint"][:2])
        dist_cube_to_bucket_center = np.linalg.norm(physics.named.data.qpos[f"red_box_joint"][:2] - physics.named.data.qpos["bucket_joint"][:2])        

        self.cucumber_in_bucket = (dist_cucumber_to_bucket_center < self._bucket_radius)
        self.cube_in_bucket = (dist_cube_to_bucket_center < self._bucket_radius)
        
        if self.cucumber_in_bucket:
            reward = 1
            self.cucumber_rewards += 1
        elif self.cube_in_bucket:
            reward = 1
            self.cube_rewards += 1
        if self.cucumber_in_bucket & self.cube_in_bucket:
            reward = 1
        else:
            reward = 0
        return reward

def get_action(master_bot_left, master_bot_right):
    action = np.zeros(14)
    # arm action
    action[:6] = master_bot_left.dxl.joint_states.position[:6]
    action[7:7+6] = master_bot_right.dxl.joint_states.position[:6]
    # gripper action
    left_gripper_pos = master_bot_left.dxl.joint_states.position[7]
    right_gripper_pos = master_bot_right.dxl.joint_states.position[7]
    normalized_left_pos = MASTER_GRIPPER_POSITION_NORMALIZE_FN(left_gripper_pos)
    normalized_right_pos = MASTER_GRIPPER_POSITION_NORMALIZE_FN(right_gripper_pos)
    action[6] = normalized_left_pos
    action[7+6] = normalized_right_pos
    return action

def test_sim_teleop():
    """ Testing teleoperation in sim with ALOHA. Requires hardware and ALOHA repo to work. """
    from interbotix_xs_modules.arm import InterbotixManipulatorXS

    BOX_POSE[0] = [0.2, 0.5, 0.05, 1, 0, 0, 0]

    # source of data
    master_bot_left = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name=f'master_left', init_node=True)
    master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name=f'master_right', init_node=False)

    # setup the environment
    env = make_sim_env('sim_transfer_cube')
    ts = env.reset()
    episode = [ts]
    # setup plotting
    ax = plt.subplot()
    plt_img = ax.imshow(ts.observation['images']['angle'])
    plt.ion()

    for t in range(1000):
        action = get_action(master_bot_left, master_bot_right)
        ts = env.step(action)
        episode.append(ts)

        plt_img.set_data(ts.observation['images']['angle'])
        plt.pause(0.02)


if __name__ == '__main__':
    test_sim_teleop()

