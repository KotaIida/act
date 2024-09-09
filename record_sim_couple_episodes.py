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
from scripted_policy import PickCoupleAndPutInPolicy, PickCoupleAndPutInPolicyMobile, PickCoupleAndPutInPolicyFranka

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
    if "static" in task_name:
        camera_names = CAM_NAMES_STATIC
        render_cam_name = 'angle'
        normalize_fn = PUPPET_GRIPPER_POSITION_NORMALIZE_FN
        right_gripper_idx = 2
        policy_cls = PickCoupleAndPutInPolicy
        left_ctrl_idx = 6
        right_ctrl_idx = 6+7
    elif "mobile" in task_name:
        camera_names = CAM_NAMES_MOBILE
        render_cam_name = 'vis'
        normalize_fn = PUPPET_GRIPPER_POSITION_NORMALIZE_FN_MOBILE
        right_gripper_idx = 1        
        policy_cls = PickCoupleAndPutInPolicyMobile
        left_ctrl_idx = 6
        right_ctrl_idx = 6+7
    elif "franka" in task_name:
        camera_names = CAM_NAMES_FRANKA
        render_cam_name = 'vis'
        normalize_fn = PUPPET_GRIPPER_POSITION_NORMALIZE_FN_FRANKA
        right_gripper_idx = 1        
        policy_cls = PickCoupleAndPutInPolicyFranka
        left_ctrl_idx = 7
        right_ctrl_idx = 7+8
    inject_noise = False

    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)


    cucumber_success = []
    cube_success = []
    episode_idx = start_idx
    with tqdm.tqdm(range(num_episodes)) as pbar:
        while episode_idx < end_idx:
            # setup the environment
            env_ee = make_ee_sim_env(task_name)

            ### Cube ##############################################################################################
            cube_ts = env_ee.reset()
            cube_episode = [cube_ts]
            cube_policy = policy_cls(inject_noise)
            cube_policy.obj = "red_box"

            # EE rollout
            if onscreen_render:
                ax = plt.subplot()
                plt_img = ax.imshow(cube_ts.observation['images'][render_cam_name])
                plt.ion()
            for step in range(episode_len):
                action = cube_policy(cube_ts) # gripper xyz, quarternion, open close = 8*2
                cube_ts = env_ee.step(action)
                cube_episode.append(cube_ts)
                if onscreen_render:
                    plt_img.set_data(cube_ts.observation['images'][render_cam_name])
                    plt.pause(0.002)
            plt.close()            
            cube_joint_traj = [cube_ts.observation['qpos'] for cube_ts in cube_episode]

            # Replace gripper pose with gripper control
            cube_gripper_ctrl_traj = [cube_ts.observation['gripper_ctrl'] for cube_ts in cube_episode]
            for joint, ctrl in zip(cube_joint_traj, cube_gripper_ctrl_traj):
                left_ctrl = normalize_fn(ctrl[0])
                right_ctrl = normalize_fn(ctrl[right_gripper_idx])
                joint[left_ctrl_idx] = left_ctrl
                joint[right_ctrl_idx] = right_ctrl
            cube_subtask_info = cube_episode[0].observation['env_state'].copy() # box pose at step 0
            del cube_policy, cube_episode

            # setup the environment
            env = make_sim_env(task_name)

            # Joint Rollout
            BOX_POSE[0] = cube_subtask_info # make sure the sim_env has the same object configurations as ee_sim_env
            cube_ts = env.reset()
            cube_episode_replay = [cube_ts]
            # setup plotting
            if onscreen_render:
                ax = plt.subplot()
                plt_img = ax.imshow(cube_ts.observation['images'][render_cam_name])
                plt.ion()
            for t in range(len(cube_joint_traj)): # note: this will increase episode length by 1
                action = cube_joint_traj[t]
                cube_ts = env.step(action)
                cube_episode_replay.append(cube_ts)
                if onscreen_render:
                    plt_img.set_data(cube_ts.observation['images'][render_cam_name])
                    plt.pause(0.02)
            episode_return = np.sum([cube_ts.reward for cube_ts in cube_episode_replay[-10:]])
            if episode_return == 10:
                cube_success.append(1)
                print(f"\nCube {episode_idx=} Successful, {episode_return=}")
            else:
                cube_success.append(0)
                print(f"\nCube {episode_idx=} Failed, {episode_return=}")
                continue
            plt.close()
            #######################################################################################################

            ### Cucumber ##########################################################################################
            cucumber_ts = env_ee.reset()
            cucumber_episode = [cucumber_ts]
            cucumber_policy = policy_cls(inject_noise)
            cucumber_policy.obj = "cucumber"
            
            # EE Rollout
            if onscreen_render:
                ax = plt.subplot()
                plt_img = ax.imshow(cucumber_ts.observation['images'][render_cam_name])
                plt.ion()
            for step in range(episode_len):
                action = cucumber_policy(cucumber_ts) # gripper xyz, quarternion, open close = 8*2
                cucumber_ts = env_ee.step(action)
                cucumber_episode.append(cucumber_ts)
                if onscreen_render:
                    plt_img.set_data(cucumber_ts.observation['images'][render_cam_name])
                    plt.pause(0.002)
            plt.close()
            cucumber_joint_traj = [cucumber_ts.observation['qpos'] for cucumber_ts in cucumber_episode]

            # Replace gripper pose with gripper control
            cucumber_gripper_ctrl_traj = [cucumber_ts.observation['gripper_ctrl'] for cucumber_ts in cucumber_episode]
            for joint, ctrl in zip(cucumber_joint_traj, cucumber_gripper_ctrl_traj):
                left_ctrl = normalize_fn(ctrl[0])
                right_ctrl = normalize_fn(ctrl[right_gripper_idx])
                joint[left_ctrl_idx] = left_ctrl
                joint[right_ctrl_idx] = right_ctrl
            cucumber_subtask_info = cucumber_episode[0].observation['env_state'].copy() # box pose at step 0
            del cucumber_policy, cucumber_episode

            del env_ee 
            
            # Joint Rollout
            BOX_POSE[0] = cucumber_subtask_info # make sure the sim_env has the same object configurations as ee_sim_env
            cucumber_ts = env.reset()
            cucumber_episode_replay = [cucumber_ts]
            # setup plotting
            if onscreen_render:
                ax = plt.subplot()
                plt_img = ax.imshow(cucumber_ts.observation['images'][render_cam_name])
                plt.ion()
            for t in range(len(cucumber_joint_traj)): # note: this will increase episode length by 1
                action = cucumber_joint_traj[t]
                cucumber_ts = env.step(action)
                cucumber_episode_replay.append(cucumber_ts)
                if onscreen_render:
                    plt_img.set_data(cucumber_ts.observation['images'][render_cam_name])
                    plt.pause(0.02)
            episode_return = np.sum([cucumber_ts.reward for cucumber_ts in cucumber_episode_replay[-10:]])
            if episode_return == 10:
                cucumber_success.append(1)
                print(f"\nCucumber {episode_idx=} Successful, {episode_return=}")
            else:
                cucumber_success.append(0)
                print(f"\nCucumber {episode_idx=} Failed, {episode_return=}")
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

            ### Cube ##############################################################################################
            cube_data_dict = {
                '/observations/qpos': [],
                '/observations/qvel': [],
                '/action': [],
            }
            for cam_name in camera_names:
                cube_data_dict[f'/observations/images/{cam_name}'] = []

            # because the replaying, there will be eps_len + 1 actions and eps_len + 2 timesteps
            # truncate here to be consistent
            cube_joint_traj = cube_joint_traj[:-1]
            cube_episode_replay = cube_episode_replay[:-1]

            # len(joint_traj) i.e. actions: max_timesteps
            # len(episode_replay) i.e. time steps: max_timesteps + 1
            max_timesteps = len(cube_joint_traj)
            while cube_joint_traj:
                action = cube_joint_traj.pop(0)
                cube_ts = cube_episode_replay.pop(0)
                cube_data_dict['/observations/qpos'].append(cube_ts.observation['qpos'])
                cube_data_dict['/observations/qvel'].append(cube_ts.observation['qvel'])
                cube_data_dict['/action'].append(action)
                for cam_name in camera_names:
                    cube_data_dict[f'/observations/images/{cam_name}'].append(cube_ts.observation['images'][cam_name])

            ### Cucumber ##########################################################################################
            cucumber_data_dict = {
                '/observations/qpos': [],
                '/observations/qvel': [],
                '/action': [],
            }
            for cam_name in camera_names:
                cucumber_data_dict[f'/observations/images/{cam_name}'] = []

            # because the replaying, there will be eps_len + 1 actions and eps_len + 2 timesteps
            # truncate here to be consistent
            cucumber_joint_traj = cucumber_joint_traj[:-1]
            cucumber_episode_replay = cucumber_episode_replay[:-1]

            # len(joint_traj) i.e. actions: max_timesteps
            # len(episode_replay) i.e. time steps: max_timesteps + 1
            max_timesteps = len(cucumber_joint_traj)
            while cucumber_joint_traj:
                action = cucumber_joint_traj.pop(0)
                cucumber_ts = cucumber_episode_replay.pop(0)
                cucumber_data_dict['/observations/qpos'].append(cucumber_ts.observation['qpos'])
                cucumber_data_dict['/observations/qvel'].append(cucumber_ts.observation['qvel'])
                cucumber_data_dict['/action'].append(action)
                for cam_name in camera_names:
                    cucumber_data_dict[f'/observations/images/{cam_name}'].append(cucumber_ts.observation['images'][cam_name])


            action_dim = len(action)
            # HDF5
            ### Cube ##########################################################################################
            dataset_path = os.path.join(dataset_dir, f'episode_{2*episode_idx+1}')
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

                for name, array in cube_data_dict.items():
                    root[name][...] = array

            ### Cucumber ##########################################################################################
            dataset_path = os.path.join(dataset_dir, f'episode_{2*episode_idx}')
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

                for name, array in cucumber_data_dict.items():
                    root[name][...] = array

            episode_idx += 1
            pbar.update()

    print(f'Saved to {dataset_dir}')
    cube_success_rate = sum(cube_success)/len(cube_success)
    cucumber_success_rate = sum(cucumber_success)/len(cucumber_success)
    with open(os.path.join(dataset_dir, f"record_result_{start_idx}_to_{end_idx}.txt"), "w") as f:
        f.write(f"Trial Num -> Cube: {len(cube_success)} Cucumber: {len(cucumber_success)}\n")
        f.write("\n")
        f.write(f"Success Rate -> Cube: {cube_success_rate} Cucumber: {cucumber_success_rate}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--dataset_dir', action='store', type=str, help='dataset saving dir', required=True)
    parser.add_argument('--num_episodes', action='store', type=int, help='num_episodes', required=False)
    parser.add_argument('--start_idx', action='store', default=0, type=int, help='num_episodes', required=False)
    parser.add_argument('--episode_len', action='store', type=int, default=600, help='length of one episode', required=False)
    parser.add_argument('--onscreen_render', action='store_true')
    
    main(vars(parser.parse_args()))