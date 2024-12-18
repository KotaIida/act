import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader
from scipy.spatial.transform import Rotation as R

import IPython
e = IPython.embed

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            original_action_shape = root['/action'].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            qpos = root['/observations/qpos'][start_ts]
            qvel = root['/observations/qvel'][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
            # get all actions after and including start_ts
            if is_sim:
                action = root['/action'][start_ts:]
                action_len = episode_len - start_ts
            else:
                action = root['/action'][max(0, start_ts - 1):] # hack, to make timesteps more aligned
                action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned

        self.is_sim = is_sim
        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_dir, num_episodes):
    all_qpos_data = []
    all_action_data = []
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]
            qvel = root['/observations/qvel'][()]
            action = root['/action'][()]
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
    all_qpos_data = torch.stack(all_qpos_data)
    all_action_data = torch.stack(all_action_data)
    all_action_data = all_action_data

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos}

    return stats


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_obj_and_dst_pose(mobile=False):
    if not mobile:
        obj_x_range = [-0.1, 0.3]
        dst_x_range = [-0.1, 0.3]
        y_range = [0.3, 0.8]
        obj_z_range = [0.015, 0.015]
        dst_z_range = [0.015, 0.015]
        obj_angle_range = [0, 180]
        obj_dst_interval = 0.1    
        obj_ranges = np.vstack([obj_x_range, y_range, obj_z_range])
        obj_position = np.random.uniform(obj_ranges[:, 0], obj_ranges[:, 1])
        obj_angle = np.random.uniform(obj_angle_range[0], obj_angle_range[1])    
        obj_quat = np.array([np.cos(np.deg2rad(obj_angle)/2), 0, 0, np.sin(np.deg2rad(obj_angle)/2)])    
        dst_ranges = np.vstack([dst_x_range, y_range, dst_z_range])
        dst_position = np.random.uniform(dst_ranges[:, 0], dst_ranges[:, 1])

        while np.linalg.norm(dst_position[:2] - obj_position[:2]) < 0.055+0.002+obj_dst_interval:
            dst_position = np.random.uniform(dst_ranges[:, 0], dst_ranges[:, 1])
    
    else:
        circle_x = 0.295
        circle_y = 0.414
        obj_circle_r = 0.564095655480825
        dst_circle_r = 0.7767498780487515

        rect_center_x = -0.0235
        rect_center_y = 0.9845
        rect_offset = 0.05
        rect_w = 0.272*2 - rect_offset*2
        rect_h = 0.1785*2 - rect_offset*2

        obj_z = 0.842
        dst_z = 0.827
        obj_angle_range = [-90, 90] # -90, 90
        obj_dst_interval = 0.1            
        obj_xy = hit_or_miss_sample(obj_circle_r, circle_x, circle_y, rect_center_x, rect_center_y, rect_w, rect_h)
        dst_xy = hit_or_miss_sample(dst_circle_r, circle_x, circle_y, rect_center_x, rect_center_y, rect_w, rect_h)

        obj_angle = np.random.uniform(obj_angle_range[0], obj_angle_range[1])    
        obj_quat = np.array([np.cos(np.deg2rad(obj_angle)/2), 0, 0, np.sin(np.deg2rad(obj_angle)/2)])    

        while np.linalg.norm(dst_xy - obj_xy) < 0.055+0.002+obj_dst_interval:
            dst_xy = hit_or_miss_sample(dst_circle_r, circle_x, circle_y, rect_center_x, rect_center_y, rect_w, rect_h)
        
        obj_position = np.hstack([obj_xy, obj_z])
        dst_position = np.hstack([dst_xy, dst_z])

    dst_quat = np.array([1, 0, 0, 0])
    return np.concatenate([obj_position, obj_quat, dst_position, dst_quat])

def hit_or_miss_sample(circle_r, circle_x, circle_y, rect_center_x, rect_center_y, rect_w, rect_h):
    rect_min_x = rect_center_x-rect_w/2
    rect_max_x = rect_center_x+rect_w/2
    rect_min_y = rect_center_y-rect_h/2
    rect_max_y = rect_center_y+rect_h/2
    
    min_x = min(circle_x-circle_r, rect_min_x)
    max_x = max(circle_x+circle_r, rect_max_x)    
    min_y = min(circle_y-circle_r, rect_min_y)
    max_y = max(circle_y+circle_r, rect_max_y)        

    in_rect, in_circle = False, False

    while not (in_rect and in_circle):    
        sample = np.random.uniform(low=[min_x, min_y], high=[max_x, max_y])
        sample_x, sample_y = sample
        
        in_rect = (rect_min_x < sample_x) & (sample_x < rect_max_x) & (rect_min_y < sample_y) & (sample_y < rect_max_y)
        in_circle = np.linalg.norm(sample - np.stack([circle_x, circle_y])) < circle_r

    return sample

def sample_obj_box_dst_pose(mobile=False):
    if not mobile:
        obj_x_range = [-0.1, 0.3]
        dst_x_range = [-0.1, 0.3]
        y_range = [0.3, 0.8]
        z_range = [0.015, 0.015]
        obj_angle_range = [0, 180]
        obj_dst_interval = 0.1    
        obj_ranges = np.vstack([obj_x_range, y_range, z_range])
        obj_position = np.random.uniform(obj_ranges[:, 0], obj_ranges[:, 1])
        obj_angle = np.random.uniform(obj_angle_range[0], obj_angle_range[1])    
        obj_quat = np.array([np.cos(np.deg2rad(obj_angle)/2), 0, 0, np.sin(np.deg2rad(obj_angle)/2)])    

        box_position = np.random.uniform(obj_ranges[:, 0], obj_ranges[:, 1])
        while np.linalg.norm(box_position[:2] - obj_position[:2]) < 0.055+0.002+obj_dst_interval:
            box_position = np.random.uniform(obj_ranges[:, 0], obj_ranges[:, 1])
        box_angle = np.random.uniform(obj_angle_range[0], obj_angle_range[1])
        box_quat = np.array([np.cos(np.deg2rad(box_angle)/2), 0, 0, np.sin(np.deg2rad(box_angle)/2)]) 

        dst_ranges = np.vstack([dst_x_range, y_range, z_range])
        dst_position = np.random.uniform(dst_ranges[:, 0], dst_ranges[:, 1])
        while np.linalg.norm(dst_position[:2] - obj_position[:2]) < 0.055+0.002+obj_dst_interval or np.linalg.norm(dst_position[:2] - box_position[:2]) < 0.055+0.002+obj_dst_interval:
            dst_position = np.random.uniform(dst_ranges[:, 0], dst_ranges[:, 1])
    else:
        circle_x = 0.295
        circle_y = 0.414
        obj_circle_r = 0.564095655480825
        dst_circle_r = 0.7767498780487515

        rect_center_x = -0.0235
        rect_center_y = 0.9845
        rect_offset = 0.05
        rect_w = 0.272*2 - rect_offset*2
        rect_h = 0.1785*2 - rect_offset*2

        obj_z = 0.842
        dst_z = 0.827
        box_z = 0.842
        obj_angle_range = [-90, 90] # -90, 90
        obj_dst_interval = 0.055 + 0.002  + 0.1            
        box_dst_interval = 0.055 + 0.002 + 0.02*np.sqrt(2)
        obj_box_interval = 0.1 + 0.02*np.sqrt(2)

        obj_xy = hit_or_miss_sample(obj_circle_r, circle_x, circle_y, rect_center_x, rect_center_y, rect_w, rect_h)
        box_xy = hit_or_miss_sample(obj_circle_r, circle_x, circle_y, rect_center_x, rect_center_y, rect_w, rect_h)
        dst_xy = hit_or_miss_sample(dst_circle_r, circle_x, circle_y, rect_center_x, rect_center_y, rect_w, rect_h)
        while np.linalg.norm(dst_xy - obj_xy) < obj_dst_interval or np.linalg.norm(dst_xy - box_xy) < box_dst_interval or np.linalg.norm(obj_xy - box_xy) < obj_box_interval:
            obj_xy = hit_or_miss_sample(obj_circle_r, circle_x, circle_y, rect_center_x, rect_center_y, rect_w, rect_h)
            box_xy = hit_or_miss_sample(obj_circle_r, circle_x, circle_y, rect_center_x, rect_center_y, rect_w, rect_h)
            dst_xy = hit_or_miss_sample(dst_circle_r, circle_x, circle_y, rect_center_x, rect_center_y, rect_w, rect_h)

        obj_angle = np.random.uniform(obj_angle_range[0], obj_angle_range[1])    
        obj_quat = np.array([np.cos(np.deg2rad(obj_angle)/2), 0, 0, np.sin(np.deg2rad(obj_angle)/2)])    

        box_angle = np.random.uniform(obj_angle_range[0], obj_angle_range[1])    
        box_quat = np.array([np.cos(np.deg2rad(box_angle)/2), 0, 0, np.sin(np.deg2rad(box_angle)/2)])            
        
        obj_position = np.hstack([obj_xy, obj_z])
        box_position = np.hstack([box_xy, box_z])
        dst_position = np.hstack([dst_xy, dst_z])
    dst_quat = np.array([1, 0, 0, 0])

    return np.concatenate([obj_position, obj_quat, box_position, box_quat, dst_position, dst_quat])

def sample_objs_dst_pose(num_obj = 4):
    BUCKET_THICK = 0.002
    BUCKET_RADIUS = 0.055
    OBJ_DST_INTERVAL = 0.1 + BUCKET_RADIUS + BUCKET_THICK
    OBJ_OBJ_INTERVAL = 0.1 + 0.015

    obj_x_range = [-0.1, 0.3]
    dst_x_range = [-0.1, 0.3]
    y_range = [0.3, 0.8]
    z_range = [0.015, 0.015]
    dst_ranges = np.vstack([dst_x_range, y_range, z_range])
    obj_angle_range = [0, 180]
    obj_ranges = np.vstack([obj_x_range, y_range, z_range])

    indices = []
    for i in range(num_obj):
        for j in range(i):
            indices.append([i, j])
    y_indices = np.array(indices)[:, 0]
    x_indices = np.array(indices)[:, 1]

    dst_position = np.random.uniform(dst_ranges[:, 0], dst_ranges[:, 1])
    obj_positions = np.random.uniform(obj_ranges[:, 0], obj_ranges[:, 1], (num_obj, 3))
    obj_obj_dist_table = np.linalg.norm(obj_positions[:, None, :2] - obj_positions[:, :2], axis=2) # (num_obj, num_obj)
    obj_dst_dist_table = np.linalg.norm(obj_positions[:, :2] - dst_position[:2], axis=1) # (num_obj)

    obj_obj_mask = obj_obj_dist_table < OBJ_OBJ_INTERVAL
    obj_dst_mask = obj_dst_dist_table < OBJ_DST_INTERVAL

    while np.any(obj_obj_mask[y_indices, x_indices]) or np.any(obj_dst_mask):
        dst_position = np.random.uniform(dst_ranges[:, 0], dst_ranges[:, 1])
        obj_positions = np.random.uniform(obj_ranges[:, 0], obj_ranges[:, 1], (num_obj, 3))
        obj_obj_dist_table = np.linalg.norm(obj_positions[:, None, :2] - obj_positions[:, :2], axis=2) # (num_obj, num_obj)
        obj_dst_dist_table = np.linalg.norm(obj_positions[:, :2] - dst_position[:2], axis=1) # (num_obj)

        obj_obj_mask = obj_obj_dist_table < OBJ_OBJ_INTERVAL
        obj_dst_mask = obj_dst_dist_table < OBJ_DST_INTERVAL

    obj_angles = np.random.uniform(obj_angle_range[0], obj_angle_range[1], num_obj)    
    obj_quats = np.array([np.cos(np.deg2rad(obj_angles)/2), np.zeros(num_obj), np.zeros(num_obj), np.sin(np.deg2rad(obj_angles)/2)]).T

    obj_poses = []
    for i in range(num_obj):
        obj_poses.append(obj_positions[i])
        obj_poses.append(obj_quats[i])

    dst_quat = np.array([1, 0, 0, 0])

    obj_poses.append(dst_position)
    obj_poses.append(dst_quat)

    return np.concatenate(obj_poses)


def sample_2objs_and_dst_pose():
    rect_center_x = -0.0235
    rect_center_y = 0.9845
    rect_half_w = 0.272
    rect_half_h = 0.1785    
    rect_offset = 0.05
    valid_w = (rect_half_w - rect_offset) * 2 
    valid_h = (rect_half_h - rect_offset) * 2 
    valid_x_start = rect_offset + rect_center_x - rect_half_w
    valid_x_end = rect_center_x + rect_half_w - rect_offset
    valid_y_start = rect_offset + rect_center_y - rect_half_h
    valid_y_end = rect_center_y + rect_half_h - rect_offset

    obj_x_range_left = [valid_x_start, valid_x_start+valid_w/3]
    obj_x_range_right = [valid_x_start+valid_w*2/3, valid_x_end]    
    dst_x_range = [valid_x_start+valid_w/3, valid_x_start+valid_w*2/3]

    obj_y_range_left = [valid_y_start, valid_y_end]
    obj_y_range_right = [valid_y_start, valid_y_end]
    dst_y_range = [valid_y_start, valid_y_end]

    obj_z_left = 0.842
    obj_z_right = 0.842
    dst_z = 0.827

    obj_angle_range = [-90, 90] # -90, 90
    obj_dst_interval = 0.055 + 0.002  + 0.1            

    obj_xy_ranges_left = np.vstack([obj_x_range_left, obj_y_range_left])
    obj_xy_ranges_right = np.vstack([obj_x_range_right, obj_y_range_right])
    dst_xy_ranges = np.vstack([dst_x_range, dst_y_range])

    dst_xy = np.random.uniform(dst_xy_ranges[:, 0], dst_xy_ranges[:, 1])    
    obj_xy_left = np.random.uniform(obj_xy_ranges_left[:, 0], obj_xy_ranges_left[:, 1])
    obj_xy_right = np.random.uniform(obj_xy_ranges_right[:, 0], obj_xy_ranges_right[:, 1])

    while np.linalg.norm(dst_xy - obj_xy_left) < obj_dst_interval:
        obj_xy_left = np.random.uniform(obj_xy_ranges_left[:, 0], obj_xy_ranges_left[:, 1])
    while np.linalg.norm(dst_xy - obj_xy_right) < obj_dst_interval:
        obj_xy_right = np.random.uniform(obj_xy_ranges_right[:, 0], obj_xy_ranges_right[:, 1])

    obj_angle_left = np.random.uniform(obj_angle_range[0], obj_angle_range[1])    
    obj_quat_left = np.array([np.cos(np.deg2rad(obj_angle_left)/2), 0, 0, np.sin(np.deg2rad(obj_angle_left)/2)])    
    obj_angle_right = np.random.uniform(obj_angle_range[0], obj_angle_range[1])    
    obj_quat_right = np.array([np.cos(np.deg2rad(obj_angle_right)/2), 0, 0, np.sin(np.deg2rad(obj_angle_right)/2)])    
    dst_quat = np.array([1, 0, 0, 0])

    obj_position_left = np.hstack([obj_xy_left, obj_z_left])
    obj_position_right = np.hstack([obj_xy_right, obj_z_right])
    dst_position = np.hstack([dst_xy, dst_z])    

    return np.concatenate([obj_position_left, obj_quat_left, obj_position_right, obj_quat_right, dst_position, dst_quat])


def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def quaternion_to_euler(q, degrees=True):
    r = R.from_quat([q[1], q[2], q[3], q[0]])
    return r.as_euler('xyz', degrees=degrees)
