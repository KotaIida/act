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
from scripted_policy import PickAndTransferPolicy, PickAndPutInPolicy, InsertionPolicy, PickMultipleAndPutInPolicy, PickAndPutInPolicyMobile, PickAndPutInPolicyFranka, PickAndPutInCardboardVPolicyFranka, PickAndPutInCardboardHPolicyFranka, PickAndPutInCardboardVRecoveryPolicyFranka

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
    careful = args['careful']
    quick = args['quick']
    pickup_num = args['pickup_num']
    episode_len = args['episode_len']
    if "static" in task_name:
        camera_names = CAM_NAMES_STATIC
        render_cam_name = 'angle'
    elif "mobile" in task_name:
        camera_names = CAM_NAMES_MOBILE
        render_cam_name = 'vis'
    elif "franka" in task_name:
        camera_names = CAM_NAMES_FRANKA
        render_cam_name = 'angle'
    inject_noise = False

    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)


    if careful:
        episode_len *= 2
    if quick:
        episode_len //= 2

    if task_name == 'sim_transfer_cube_on_static_aloha':
        policy_cls = PickAndTransferPolicy
    elif task_name == 'sim_put_multiple_in_bucket_on_static_aloha':
        policy_cls = PickMultipleAndPutInPolicy
        episode_len *= pickup_num
    elif task_name == 'sim_insertion_scripted':
        policy_cls = InsertionPolicy
    else:
        if "static" in task_name:   
            policy_cls = PickAndPutInPolicy
        elif "mobile" in task_name:
            policy_cls = PickAndPutInPolicyMobile
        elif "franka" in task_name:
            if "cardboard_v" in task_name:
                if "recovery" in task_name:
                    policy_cls = PickAndPutInCardboardVRecoveryPolicyFranka
                else:
                    policy_cls = PickAndPutInCardboardVPolicyFranka
            elif "cardboard_h" in task_name:
                policy_cls = PickAndPutInCardboardHPolicyFranka
            else:
                policy_cls = PickAndPutInPolicyFranka

    success = []
    episode_idx = start_idx
    with tqdm.tqdm(range(num_episodes)) as pbar:
        while episode_idx < end_idx:
            # setup the environment
            env = make_ee_sim_env(task_name, pickup_num=pickup_num)
            
            ts = env.reset()
            episode = [ts]
            if task_name == 'sim_transfer_cube_on_static_aloha':
                policy = policy_cls(inject_noise)
            elif task_name == 'sim_put_multiple_in_bucket_on_static_aloha':
                policy = policy_cls(pickup_num)
            elif task_name == 'sim_insertion_scripted':
                policy = policy_cls(inject_noise)
            else:
                policy = policy_cls(inject_noise, careful, quick)
            # setup plotting
            if onscreen_render:
                ax = plt.subplot()
                plt_img = ax.imshow(ts.observation['images'][render_cam_name])
                plt.ion()
            for step in range(episode_len):
                action = policy(ts) # gripper xyz, quarternion, open close = 8*2
                ts = env.step(action)
                episode.append(ts)
                if onscreen_render:
                    plt_img.set_data(ts.observation['images'][render_cam_name])
                    plt.pause(0.002)
            plt.close()


            joint_traj = [ts.observation['qpos'] for ts in episode]
            # replace gripper pose with gripper control
            # gripperの開閉状態を観測値で制御してしまうと、力が弱すぎてつかめないため置換
            gripper_ctrl_traj = [ts.observation['gripper_ctrl'] for ts in episode]
            for joint, ctrl in zip(joint_traj, gripper_ctrl_traj):
                if "static" in task_name:   
                    left_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[0])
                    right_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[2])
                    joint[6] = left_ctrl
                    joint[6+7] = right_ctrl
                elif "mobile" in task_name:
                    left_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN_MOBILE(ctrl[0])
                    right_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN_MOBILE(ctrl[1])
                    joint[6] = left_ctrl
                    joint[6+7] = right_ctrl
                elif "franka" in task_name:
                    left_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN_FRANKA(ctrl[0])
                    right_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN_FRANKA(ctrl[1])
                    joint[7] = left_ctrl
                    joint[7+8] = right_ctrl

            subtask_info = episode[0].observation['env_state'].copy() # box pose at step 0

            # clear unused variables
            del env
            del episode
            del policy

            # setup the environment
            # print('Replaying joint commands')
            env = make_sim_env(task_name, pickup_num=pickup_num)
            BOX_POSE[0] = subtask_info # make sure the sim_env has the same object configurations as ee_sim_env
            ts = env.reset()

            episode_replay = [ts]
            # setup plotting
            if onscreen_render:
                ax = plt.subplot()
                plt_img = ax.imshow(ts.observation['images'][render_cam_name])
                plt.ion()
            for t in range(len(joint_traj)): # note: this will increase episode length by 1
                action = joint_traj[t]
                ts = env.step(action)
                episode_replay.append(ts)
                if onscreen_render:
                    plt_img.set_data(ts.observation['images'][render_cam_name])
                    plt.pause(0.02)


            if task_name == "sim_transfer_cube_on_static_aloha":
                episode_return = np.sum([ts.reward for ts in episode_replay[1:]])
                episode_max_reward = np.max([ts.reward for ts in episode_replay[1:]])
                if episode_max_reward == env.task.max_reward:
                    success.append(1)
                    # print(f"{episode_idx=} Successful, {episode_return=}")
                else:
                    success.append(0)
                    continue
                    # print(f"{episode_idx=} Failed")
            else:
                episode_return = np.sum([ts.reward for ts in episode_replay[-10:]])
                if episode_return == 10:
                    success.append(1)
                    print(f"{episode_idx=} Successful, {episode_return=}")
                else:
                    success.append(0)
                    print(f"{episode_idx=} Failed, {episode_return=}")
                    continue


            plt.close()

            """
            For each timestep:
            observations
            - images
                - each_cam_name     (480, 640, 3) 'uint8'
            - qpos                  (14,)         'float64'
            - qvel                  (14,)         'float64'

            action                  (14,)         'float64'
            """

            data_dict = {
                '/observations/qpos': [],
                '/observations/qvel': [],
                '/action': [],
            }
            for cam_name in camera_names:
                data_dict[f'/observations/images/{cam_name}'] = []

            # because the replaying, there will be eps_len + 1 actions and eps_len + 2 timesteps
            # truncate here to be consistent
            joint_traj = joint_traj[:-1]
            episode_replay = episode_replay[:-1]

            # len(joint_traj) i.e. actions: max_timesteps
            # len(episode_replay) i.e. time steps: max_timesteps + 1
            max_timesteps = len(joint_traj)
            while joint_traj:
                action = joint_traj.pop(0)
                ts = episode_replay.pop(0)
                data_dict['/observations/qpos'].append(ts.observation['qpos'])
                data_dict['/observations/qvel'].append(ts.observation['qvel'])
                data_dict['/action'].append(action)
                for cam_name in camera_names:
                    data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])

            # HDF5
            t0 = time.time()
            dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}')
            with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
                root.attrs['sim'] = True
                obs = root.create_group('observations')
                image = obs.create_group('images')
                for cam_name in camera_names:
                    _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                            chunks=(1, 480, 640, 3), )
                # compression='gzip',compression_opts=2,)
                # compression=32001, compression_opts=(0, 0, 0, 0, 9, 1, 1), shuffle=False)
                qpos = obs.create_dataset('qpos', (max_timesteps, len(action)))
                qvel = obs.create_dataset('qvel', (max_timesteps, len(action)))
                action = root.create_dataset('action', (max_timesteps, len(action)))

                for name, array in data_dict.items():
                    root[name][...] = array
            episode_idx += 1
            pbar.update()

    print(f'Saved to {dataset_dir}')
    success_indices = np.where(success)[0].tolist()
    success_rate = sum(success)/len(success)
    with open(os.path.join(dataset_dir, f"record_result_{start_idx}_to_{end_idx}.txt"), "w") as f:
        f.write(f"Trial Num: {len(success)}\n")
        f.write(repr(success_indices))
        f.write("\n")
        f.write(f"Success Rate: {success_rate}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--dataset_dir', action='store', type=str, help='dataset saving dir', required=True)
    parser.add_argument('--num_episodes', action='store', type=int, help='num_episodes', required=False)
    parser.add_argument('--start_idx', action='store', default=0, type=int, help='num_episodes', required=False)
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--episode_len', action='store', type=int, default=600, help='length of one episode', required=False)
    parser.add_argument('--careful', action='store_true')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--pickup_num', action='store', type=int, help='number of targets to pick up', required=False)
    
    main(vars(parser.parse_args()))

