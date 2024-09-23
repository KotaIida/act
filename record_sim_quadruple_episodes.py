import time
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import h5py
import tqdm

from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN, PUPPET_GRIPPER_POSITION_NORMALIZE_FN_MOBILE, PUPPET_GRIPPER_POSITION_NORMALIZE_FN_FRANKA, CAM_NAMES_STATIC, CAM_NAMES_MOBILE, CAM_NAMES_FRANKA
from ee_sim_env import make_ee_sim_env
from sim_env import make_sim_env, BOX_POSE
from scripted_policy import PickCoupleAndPutInPolicy, PickCoupleAndPutInPolicyMobile, PickCoupleAndPutInPolicyFranka, PickCoupleAndPutInPolicyFrankaBimanualLeftFirst, PickCoupleAndPutInPolicyFrankaBimanualRightFirst

import IPython
e = IPython.embed


def main(args):
    """
    Generate demonstration data in simulation.
    First rollout the policy (defined in ee space) in ee_sim_env. Obtain the joint trajectory.
    Replace the gripper joint positions with the commanded joint position.
    Replay this joint trajectory (as action sequence) in sim_env, and record all observations.
    Save this episode of data, and continue to next episode of data collection.
    """

    task_name = args['task_name']
    dataset_dir = args['dataset_dir']
    start_idx = args['start_idx']
    num_episodes = args['num_episodes']
    end_idx = start_idx + num_episodes
    onscreen_render = args['onscreen_render']
    episode_len = args['episode_len']
    if "franka" in task_name:
        camera_names = CAM_NAMES_FRANKA
        render_cam_name = 'side'
        normalize_fn = PUPPET_GRIPPER_POSITION_NORMALIZE_FN_FRANKA
        right_gripper_idx = 1        
        left_ctrl_idx = 7
        right_ctrl_idx = 7+8

        policy_cls_left_first = PickCoupleAndPutInPolicyFrankaBimanualLeftFirst
        policy_cls_right_first = PickCoupleAndPutInPolicyFrankaBimanualRightFirst
    
    inject_noise = False

    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)


    ee_left_first_cube_success = []
    ee_right_first_cube_success = []
    ee_left_first_cucumber_success = []
    ee_right_first_cucumber_success = []

    left_first_cube_success = []
    right_first_cube_success = []
    left_first_cucumber_success = []
    right_first_cucumber_success = []

    episode_idx = start_idx
    with tqdm.tqdm(range(num_episodes)) as pbar:
        while episode_idx < end_idx:
            # setup the environment
            env_ee = make_ee_sim_env(task_name)

            ### Left First Cube ##############################################################################################
            left_first_cube_ts = env_ee.reset()
            left_first_cube_episode = [left_first_cube_ts]
            left_first_cube_policy = policy_cls_left_first(inject_noise)

            # EE rollout
            if onscreen_render:
                ax = plt.subplot()
                plt_img = ax.imshow(left_first_cube_ts.observation['images'][render_cam_name])
                plt.ion()
            for step in range(episode_len):
                action = left_first_cube_policy(left_first_cube_ts) # gripper xyz, quarternion, open close = 8*2
                left_first_cube_ts = env_ee.step(action)
                left_first_cube_episode.append(left_first_cube_ts)
                if onscreen_render:
                    plt_img.set_data(left_first_cube_ts.observation['images'][render_cam_name])
                    plt.pause(0.002)
            episode_return = np.sum([left_first_cube_ts.reward for left_first_cube_ts in left_first_cube_episode[-10:]])
            if episode_return == 10:
                ee_left_first_cube_success.append(1)
                print(f"\nLeft First EE Cube {episode_idx=} Successful, {episode_return=}")
            else:
                ee_left_first_cube_success.append(0)
                print(f"\nLeft First EE Cube {episode_idx=} Failed, {episode_return=}")
                continue
            plt.close()            
            left_first_cube_joint_traj = [left_first_cube_ts.observation['qpos'] for left_first_cube_ts in left_first_cube_episode]

            # Replace gripper pose with gripper control
            left_first_cube_gripper_ctrl_traj = [left_first_cube_ts.observation['gripper_ctrl'] for left_first_cube_ts in left_first_cube_episode]
            for joint, ctrl in zip(left_first_cube_joint_traj, left_first_cube_gripper_ctrl_traj):
                left_ctrl = normalize_fn(ctrl[0])
                right_ctrl = normalize_fn(ctrl[right_gripper_idx])
                joint[left_ctrl_idx] = left_ctrl
                joint[right_ctrl_idx] = right_ctrl
            left_first_cube_subtask_info = left_first_cube_episode[0].observation['env_state'].copy() # box pose at step 0
            del left_first_cube_policy, left_first_cube_episode

            # Setup the environment
            env = make_sim_env(task_name)

            # Joint Rollout
            BOX_POSE[0] = left_first_cube_subtask_info # make sure the sim_env has the same object configurations as ee_sim_env
            left_first_cube_ts = env.reset()
            left_first_cube_episode_replay = [left_first_cube_ts]
            
            # Setup plotting
            if onscreen_render:
                ax = plt.subplot()
                plt_img = ax.imshow(left_first_cube_ts.observation['images'][render_cam_name])
                plt.ion()
            for t in range(len(left_first_cube_joint_traj)): # note: this will increase episode length by 1
                action = left_first_cube_joint_traj[t]
                left_first_cube_ts = env.step(action)
                left_first_cube_episode_replay.append(left_first_cube_ts)
                if onscreen_render:
                    plt_img.set_data(left_first_cube_ts.observation['images'][render_cam_name])
                    plt.pause(0.02)
            episode_return = np.sum([left_first_cube_ts.reward for left_first_cube_ts in left_first_cube_episode_replay[-10:]])
            if episode_return == 10:
                left_first_cube_success.append(1)
                print(f"\nLeft First Cube {episode_idx=} Successful, {episode_return=}")
            else:
                left_first_cube_success.append(0)
                print(f"\nLeft First Cube {episode_idx=} Failed, {episode_return=}")
                continue
            plt.close()
            #######################################################################################################


            ### Right First Cube ##############################################################################################
            right_first_cube_ts = env_ee.reset()
            right_first_cube_episode = [right_first_cube_ts]
            right_first_cube_policy = policy_cls_right_first(inject_noise)

            # EE rollout
            if onscreen_render:
                ax = plt.subplot()
                plt_img = ax.imshow(right_first_cube_ts.observation['images'][render_cam_name])
                plt.ion()
            for step in range(episode_len):
                action = right_first_cube_policy(right_first_cube_ts) # gripper xyz, quarternion, open close = 8*2
                right_first_cube_ts = env_ee.step(action)
                right_first_cube_episode.append(right_first_cube_ts)
                if onscreen_render:
                    plt_img.set_data(right_first_cube_ts.observation['images'][render_cam_name])
                    plt.pause(0.002)
            episode_return = np.sum([right_first_cube_ts.reward for right_first_cube_ts in right_first_cube_episode[-10:]])
            if episode_return == 10:
                ee_right_first_cube_success.append(1)
                print(f"\nRight First EE Cube {episode_idx=} Successful, {episode_return=}")
            else:
                ee_right_first_cube_success.append(0)
                print(f"\nRight First EE Cube {episode_idx=} Failed, {episode_return=}")
                continue
            plt.close()            

            right_first_cube_joint_traj = [right_first_cube_ts.observation['qpos'] for right_first_cube_ts in right_first_cube_episode]

            # Replace gripper pose with gripper control
            right_first_cube_gripper_ctrl_traj = [right_first_cube_ts.observation['gripper_ctrl'] for right_first_cube_ts in right_first_cube_episode]
            for joint, ctrl in zip(right_first_cube_joint_traj, right_first_cube_gripper_ctrl_traj):
                left_ctrl = normalize_fn(ctrl[0])
                right_ctrl = normalize_fn(ctrl[right_gripper_idx])
                joint[left_ctrl_idx] = left_ctrl
                joint[right_ctrl_idx] = right_ctrl
            right_first_cube_subtask_info = right_first_cube_episode[0].observation['env_state'].copy() # box pose at step 0
            del right_first_cube_policy, right_first_cube_episode

            # Setup the environment
            env = make_sim_env(task_name)

            # Joint Rollout
            BOX_POSE[0] = right_first_cube_subtask_info # make sure the sim_env has the same object configurations as ee_sim_env
            right_first_cube_ts = env.reset()
            right_first_cube_episode_replay = [right_first_cube_ts]
            
            # Setup plotting
            if onscreen_render:
                ax = plt.subplot()
                plt_img = ax.imshow(right_first_cube_ts.observation['images'][render_cam_name])
                plt.ion()
            for t in range(len(right_first_cube_joint_traj)): # note: this will increase episode length by 1
                action = right_first_cube_joint_traj[t]
                right_first_cube_ts = env.step(action)
                right_first_cube_episode_replay.append(right_first_cube_ts)
                if onscreen_render:
                    plt_img.set_data(right_first_cube_ts.observation['images'][render_cam_name])
                    plt.pause(0.02)
            episode_return = np.sum([right_first_cube_ts.reward for right_first_cube_ts in right_first_cube_episode_replay[-10:]])
            if episode_return == 10:
                right_first_cube_success.append(1)
                print(f"\nRight First Cube {episode_idx=} Successful, {episode_return=}")
            else:
                right_first_cube_success.append(0)
                print(f"\nRight First Cube {episode_idx=} Failed, {episode_return=}")
                continue
            plt.close()
            #######################################################################################################



            ### Left First Cucumber ##########################################################################################
            left_first_cucumber_ts = env_ee.reset()
            left_first_cucumber_episode = [left_first_cucumber_ts]
            left_first_cucumber_policy = policy_cls_left_first(inject_noise)
            left_first_cucumber_policy.obj = "cucumber"
            
            # EE Rollout
            if onscreen_render:
                ax = plt.subplot()
                plt_img = ax.imshow(left_first_cucumber_ts.observation['images'][render_cam_name])
                plt.ion()
            for step in range(episode_len):
                action = left_first_cucumber_policy(left_first_cucumber_ts) # gripper xyz, quarternion, open close = 8*2
                left_first_cucumber_ts = env_ee.step(action)
                left_first_cucumber_episode.append(left_first_cucumber_ts)
                if onscreen_render:
                    plt_img.set_data(left_first_cucumber_ts.observation['images'][render_cam_name])
                    plt.pause(0.002)
            episode_return = np.sum([left_first_cucumber_ts.reward for left_first_cucumber_ts in left_first_cucumber_episode[-10:]])
            if episode_return == 10:
                ee_left_first_cucumber_success.append(1)
                print(f"\nLeft First EE Cucumber {episode_idx=} Successful, {episode_return=}")
            else:
                ee_left_first_cucumber_success.append(0)
                print(f"\nLeft First EE Cucumber {episode_idx=} Failed, {episode_return=}")
                continue
            plt.close()
            left_first_cucumber_joint_traj = [left_first_cucumber_ts.observation['qpos'] for left_first_cucumber_ts in left_first_cucumber_episode]

            # Replace gripper pose with gripper control
            left_first_cucumber_gripper_ctrl_traj = [left_first_cucumber_ts.observation['gripper_ctrl'] for left_first_cucumber_ts in left_first_cucumber_episode]
            for joint, ctrl in zip(left_first_cucumber_joint_traj, left_first_cucumber_gripper_ctrl_traj):
                left_ctrl = normalize_fn(ctrl[0])
                right_ctrl = normalize_fn(ctrl[right_gripper_idx])
                joint[left_ctrl_idx] = left_ctrl
                joint[right_ctrl_idx] = right_ctrl
            left_first_cucumber_subtask_info = left_first_cucumber_episode[0].observation['env_state'].copy() # box pose at step 0
            del left_first_cucumber_policy, left_first_cucumber_episode
            
            # Joint Rollout
            BOX_POSE[0] = left_first_cucumber_subtask_info # make sure the sim_env has the same object configurations as ee_sim_env
            left_first_cucumber_ts = env.reset()
            left_first_cucumber_episode_replay = [left_first_cucumber_ts]
            # Setup plotting
            if onscreen_render:
                ax = plt.subplot()
                plt_img = ax.imshow(left_first_cucumber_ts.observation['images'][render_cam_name])
                plt.ion()
            for t in range(len(left_first_cucumber_joint_traj)): # note: this will increase episode length by 1
                action = left_first_cucumber_joint_traj[t]
                left_first_cucumber_ts = env.step(action)
                left_first_cucumber_episode_replay.append(left_first_cucumber_ts)
                if onscreen_render:
                    plt_img.set_data(left_first_cucumber_ts.observation['images'][render_cam_name])
                    plt.pause(0.02)
            episode_return = np.sum([left_first_cucumber_ts.reward for left_first_cucumber_ts in left_first_cucumber_episode_replay[-10:]])
            if episode_return == 10:
                left_first_cucumber_success.append(1)
                print(f"\nLeft First Cucumber {episode_idx=} Successful, {episode_return=}")
            else:
                left_first_cucumber_success.append(0)
                print(f"\nLeft First Cucumber {episode_idx=} Failed, {episode_return=}")
                continue
            plt.close()
            #######################################################################################################


            ### Right First Cucumber ##########################################################################################
            right_first_cucumber_ts = env_ee.reset()
            right_first_cucumber_episode = [right_first_cucumber_ts]
            right_first_cucumber_policy = policy_cls_right_first(inject_noise)
            right_first_cucumber_policy.obj = "cucumber"
            
            # EE Rollout
            if onscreen_render:
                ax = plt.subplot()
                plt_img = ax.imshow(right_first_cucumber_ts.observation['images'][render_cam_name])
                plt.ion()
            for step in range(episode_len):
                action = right_first_cucumber_policy(right_first_cucumber_ts) # gripper xyz, quarternion, open close = 8*2
                right_first_cucumber_ts = env_ee.step(action)
                right_first_cucumber_episode.append(right_first_cucumber_ts)
                if onscreen_render:
                    plt_img.set_data(right_first_cucumber_ts.observation['images'][render_cam_name])
                    plt.pause(0.002)
            episode_return = np.sum([right_first_cucumber_ts.reward for right_first_cucumber_ts in right_first_cucumber_episode[-10:]])
            if episode_return == 10:
                ee_right_first_cucumber_success.append(1)
                print(f"\nRight First EE Cucumber {episode_idx=} Successful, {episode_return=}")
            else:
                ee_right_first_cucumber_success.append(0)
                print(f"\nRight First EE Cucumber {episode_idx=} Failed, {episode_return=}")
                continue
            plt.close()
            right_first_cucumber_joint_traj = [right_first_cucumber_ts.observation['qpos'] for right_first_cucumber_ts in right_first_cucumber_episode]

            # Replace gripper pose with gripper control
            right_first_cucumber_gripper_ctrl_traj = [right_first_cucumber_ts.observation['gripper_ctrl'] for right_first_cucumber_ts in right_first_cucumber_episode]
            for joint, ctrl in zip(right_first_cucumber_joint_traj, right_first_cucumber_gripper_ctrl_traj):
                left_ctrl = normalize_fn(ctrl[0])
                right_ctrl = normalize_fn(ctrl[right_gripper_idx])
                joint[left_ctrl_idx] = left_ctrl
                joint[right_ctrl_idx] = right_ctrl
            right_first_cucumber_subtask_info = right_first_cucumber_episode[0].observation['env_state'].copy() # box pose at step 0
            del right_first_cucumber_policy, right_first_cucumber_episode

            del env_ee 
            
            # Joint Rollout
            BOX_POSE[0] = right_first_cucumber_subtask_info # make sure the sim_env has the same object configurations as ee_sim_env
            right_first_cucumber_ts = env.reset()
            right_first_cucumber_episode_replay = [right_first_cucumber_ts]
            # Setup plotting
            if onscreen_render:
                ax = plt.subplot()
                plt_img = ax.imshow(right_first_cucumber_ts.observation['images'][render_cam_name])
                plt.ion()
            for t in range(len(right_first_cucumber_joint_traj)): # note: this will increase episode length by 1
                action = right_first_cucumber_joint_traj[t]
                right_first_cucumber_ts = env.step(action)
                right_first_cucumber_episode_replay.append(right_first_cucumber_ts)
                if onscreen_render:
                    plt_img.set_data(right_first_cucumber_ts.observation['images'][render_cam_name])
                    plt.pause(0.02)
            episode_return = np.sum([right_first_cucumber_ts.reward for right_first_cucumber_ts in right_first_cucumber_episode_replay[-10:]])
            if episode_return == 10:
                right_first_cucumber_success.append(1)
                print(f"\nRight First Cucumber {episode_idx=} Successful, {episode_return=}")
            else:
                right_first_cucumber_success.append(0)
                print(f"\nRight First Cucumber {episode_idx=} Failed, {episode_return=}")
                continue
            plt.close()
            #######################################################################################################







            """
            For each timestep:
            observations
            - images
                - each_cam_name     (480, 640, 3) 'uint8'
            - qpos                  (14,)         'float64'
            - qvel                  (14,)         'float64'

            action                  (14,)         'float64'
            """

            ### Left First Cube ##############################################################################################
            left_first_cube_data_dict = {
                '/observations/qpos': [],
                '/observations/qvel': [],
                '/action': [],
            }
            for cam_name in camera_names:
                left_first_cube_data_dict[f'/observations/images/{cam_name}'] = []

            # because the replaying, there will be eps_len + 1 actions and eps_len + 2 timesteps
            # truncate here to be consistent
            left_first_cube_joint_traj = left_first_cube_joint_traj[:-1]
            left_first_cube_episode_replay = left_first_cube_episode_replay[:-1]

            # len(joint_traj) i.e. actions: max_timesteps
            # len(episode_replay) i.e. time steps: max_timesteps + 1
            max_timesteps = len(left_first_cube_joint_traj)
            while left_first_cube_joint_traj:
                action = left_first_cube_joint_traj.pop(0)
                left_first_cube_ts = left_first_cube_episode_replay.pop(0)
                left_first_cube_data_dict['/observations/qpos'].append(left_first_cube_ts.observation['qpos'])
                left_first_cube_data_dict['/observations/qvel'].append(left_first_cube_ts.observation['qvel'])
                left_first_cube_data_dict['/action'].append(action)
                for cam_name in camera_names:
                    left_first_cube_data_dict[f'/observations/images/{cam_name}'].append(left_first_cube_ts.observation['images'][cam_name])


            ### Right First Cube ##############################################################################################
            right_first_cube_data_dict = {
                '/observations/qpos': [],
                '/observations/qvel': [],
                '/action': [],
            }
            for cam_name in camera_names:
                right_first_cube_data_dict[f'/observations/images/{cam_name}'] = []

            # because the replaying, there will be eps_len + 1 actions and eps_len + 2 timesteps
            # truncate here to be consistent
            right_first_cube_joint_traj = right_first_cube_joint_traj[:-1]
            right_first_cube_episode_replay = right_first_cube_episode_replay[:-1]

            # len(joint_traj) i.e. actions: max_timesteps
            # len(episode_replay) i.e. time steps: max_timesteps + 1
            max_timesteps = len(right_first_cube_joint_traj)
            while right_first_cube_joint_traj:
                action = right_first_cube_joint_traj.pop(0)
                right_first_cube_ts = right_first_cube_episode_replay.pop(0)
                right_first_cube_data_dict['/observations/qpos'].append(right_first_cube_ts.observation['qpos'])
                right_first_cube_data_dict['/observations/qvel'].append(right_first_cube_ts.observation['qvel'])
                right_first_cube_data_dict['/action'].append(action)
                for cam_name in camera_names:
                    right_first_cube_data_dict[f'/observations/images/{cam_name}'].append(right_first_cube_ts.observation['images'][cam_name])


            ### Left First Cucumber ##########################################################################################
            left_first_cucumber_data_dict = {
                '/observations/qpos': [],
                '/observations/qvel': [],
                '/action': [],
            }
            for cam_name in camera_names:
                left_first_cucumber_data_dict[f'/observations/images/{cam_name}'] = []

            # because the replaying, there will be eps_len + 1 actions and eps_len + 2 timesteps
            # truncate here to be consistent
            left_first_cucumber_joint_traj = left_first_cucumber_joint_traj[:-1]
            left_first_cucumber_episode_replay = left_first_cucumber_episode_replay[:-1]

            # len(joint_traj) i.e. actions: max_timesteps
            # len(episode_replay) i.e. time steps: max_timesteps + 1
            max_timesteps = len(left_first_cucumber_joint_traj)
            while left_first_cucumber_joint_traj:
                action = left_first_cucumber_joint_traj.pop(0)
                left_first_cucumber_ts = left_first_cucumber_episode_replay.pop(0)
                left_first_cucumber_data_dict['/observations/qpos'].append(left_first_cucumber_ts.observation['qpos'])
                left_first_cucumber_data_dict['/observations/qvel'].append(left_first_cucumber_ts.observation['qvel'])
                left_first_cucumber_data_dict['/action'].append(action)
                for cam_name in camera_names:
                    left_first_cucumber_data_dict[f'/observations/images/{cam_name}'].append(left_first_cucumber_ts.observation['images'][cam_name])


            ### Right First Cucumber ##########################################################################################
            right_first_cucumber_data_dict = {
                '/observations/qpos': [],
                '/observations/qvel': [],
                '/action': [],
            }
            for cam_name in camera_names:
                right_first_cucumber_data_dict[f'/observations/images/{cam_name}'] = []

            # because the replaying, there will be eps_len + 1 actions and eps_len + 2 timesteps
            # truncate here to be consistent
            right_first_cucumber_joint_traj = right_first_cucumber_joint_traj[:-1]
            right_first_cucumber_episode_replay = right_first_cucumber_episode_replay[:-1]

            # len(joint_traj) i.e. actions: max_timesteps
            # len(episode_replay) i.e. time steps: max_timesteps + 1
            max_timesteps = len(right_first_cucumber_joint_traj)
            while right_first_cucumber_joint_traj:
                action = right_first_cucumber_joint_traj.pop(0)
                right_first_cucumber_ts = right_first_cucumber_episode_replay.pop(0)
                right_first_cucumber_data_dict['/observations/qpos'].append(right_first_cucumber_ts.observation['qpos'])
                right_first_cucumber_data_dict['/observations/qvel'].append(right_first_cucumber_ts.observation['qvel'])
                right_first_cucumber_data_dict['/action'].append(action)
                for cam_name in camera_names:
                    right_first_cucumber_data_dict[f'/observations/images/{cam_name}'].append(right_first_cucumber_ts.observation['images'][cam_name])


            action_dim = len(action)
            
            # HDF5
            ### Left First Cube ##########################################################################################
            dataset_path = os.path.join(dataset_dir, f'episode_{4*episode_idx+2}')
            with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
                root.attrs['sim'] = True
                obs = root.create_group('observations')
                image = obs.create_group('images')
                for cam_name in camera_names:
                    _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                            chunks=(1, 480, 640, 3), )
                # compression='gzip',compression_opts=2,)
                # compression=32001, compression_opts=(0, 0, 0, 0, 9, 1, 1), shuffle=False)
                qpos = obs.create_dataset('qpos', (max_timesteps, action_dim))
                qvel = obs.create_dataset('qvel', (max_timesteps, action_dim))
                action = root.create_dataset('action', (max_timesteps, action_dim))

                for name, array in left_first_cube_data_dict.items():
                    root[name][...] = array

            ### Right First Cube ##########################################################################################
            dataset_path = os.path.join(dataset_dir, f'episode_{4*episode_idx+3}')
            with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
                root.attrs['sim'] = True
                obs = root.create_group('observations')
                image = obs.create_group('images')
                for cam_name in camera_names:
                    _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                            chunks=(1, 480, 640, 3), )
                # compression='gzip',compression_opts=2,)
                # compression=32001, compression_opts=(0, 0, 0, 0, 9, 1, 1), shuffle=False)
                qpos = obs.create_dataset('qpos', (max_timesteps, action_dim))
                qvel = obs.create_dataset('qvel', (max_timesteps, action_dim))
                action = root.create_dataset('action', (max_timesteps, action_dim))

                for name, array in right_first_cube_data_dict.items():
                    root[name][...] = array

            ### Left First Cucumber ##########################################################################################
            dataset_path = os.path.join(dataset_dir, f'episode_{4*episode_idx+0}')
            with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
                root.attrs['sim'] = True
                obs = root.create_group('observations')
                image = obs.create_group('images')
                for cam_name in camera_names:
                    _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                            chunks=(1, 480, 640, 3), )
                # compression='gzip',compression_opts=2,)
                # compression=32001, compression_opts=(0, 0, 0, 0, 9, 1, 1), shuffle=False)
                qpos = obs.create_dataset('qpos', (max_timesteps, action_dim))
                qvel = obs.create_dataset('qvel', (max_timesteps, action_dim))
                action = root.create_dataset('action', (max_timesteps, action_dim))

                for name, array in left_first_cucumber_data_dict.items():
                    root[name][...] = array

            ### Right First Cucumber ##########################################################################################
            dataset_path = os.path.join(dataset_dir, f'episode_{4*episode_idx+1}')
            with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
                root.attrs['sim'] = True
                obs = root.create_group('observations')
                image = obs.create_group('images')
                for cam_name in camera_names:
                    _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                            chunks=(1, 480, 640, 3), )
                # compression='gzip',compression_opts=2,)
                # compression=32001, compression_opts=(0, 0, 0, 0, 9, 1, 1), shuffle=False)
                qpos = obs.create_dataset('qpos', (max_timesteps, action_dim))
                qvel = obs.create_dataset('qvel', (max_timesteps, action_dim))
                action = root.create_dataset('action', (max_timesteps, action_dim))

                for name, array in right_first_cucumber_data_dict.items():
                    root[name][...] = array


            episode_idx += 1
            pbar.update()

    print(f'Saved to {dataset_dir}')
    left_first_cube_success_rate = sum(left_first_cube_success)/len(left_first_cube_success)
    right_first_cube_success_rate = sum(right_first_cube_success)/len(right_first_cube_success)
    left_first_cucumber_success_rate = sum(left_first_cucumber_success)/len(left_first_cucumber_success)
    right_first_cucumber_success_rate = sum(right_first_cucumber_success)/len(right_first_cucumber_success)
    with open(os.path.join(dataset_dir, f"record_result_{start_idx}_to_{end_idx}.txt"), "w") as f:
        f.write(f"### Trial Num ###\n")
        f.write(f"Left First Cube: {len(left_first_cube_success)}\n")
        f.write(f"Right First Cube: {len(right_first_cube_success)}\n")
        f.write(f"Left First Cucumber: {len(left_first_cucumber_success)}\n")
        f.write(f"Right First Cucumber: {len(right_first_cucumber_success)}\n")

        f.write("\n")
        f.write(f"### Success Rate ###\n")        
        f.write(f"Left First Cube: {left_first_cube_success_rate}\n")
        f.write(f"Right First Cube: {right_first_cube_success_rate}\n")
        f.write(f"Left First Cucumber: {left_first_cucumber_success_rate}\n")
        f.write(f"Right First Cucumber: {right_first_cucumber_success_rate}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--dataset_dir', action='store', type=str, help='dataset saving dir', required=True)
    parser.add_argument('--num_episodes', action='store', type=int, help='num_episodes', required=False)
    parser.add_argument('--start_idx', action='store', default=0, type=int, help='num_episodes', required=False)
    parser.add_argument('--episode_len', action='store', type=int, default=600, help='length of one episode', required=False)
    parser.add_argument('--onscreen_render', action='store_true')
    
    main(vars(parser.parse_args()))