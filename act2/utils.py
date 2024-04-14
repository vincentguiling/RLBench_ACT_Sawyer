import numpy as np
import random
import torch
import os
import h5py
import json
from torch.utils.data import TensorDataset, DataLoader

import IPython
e = IPython.embed

CROP_TOP = True  # hardcode
FILTER_MISTAKES = True  # Filter out mistakes from the dataset even if not use_language

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, max_len=None, command_list=None, use_language=False, language_encoder=None):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids if len(episode_ids) > 0 else [0] #################
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        #################
        self.max_len = max_len
        self.command_list = [cmd.strip("'\"") for cmd in command_list]
        self.use_language = use_language
        self.language_encoder = language_encoder
        self.transformations = None
        #################
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        max_len = self.max_len

        sample_full_episode = False # hardcode ### 没有用了

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        
        ######################################################################################################
        if self.use_language or FILTER_MISTAKES:
            
            json_name = f"episode_{episode_id}_encoded_{self.language_encoder}.json"
            encoded_json_path = os.path.join(self.dataset_dir, json_name)
            
            with open(encoded_json_path, "r") as f:
                episode_data = json.load(f)
        
        # print(f"{len(self.command_list)=}")
        # if len(self.command_list) > 0: # 为什么训练的时候读取数据集要直接给指令呢？还要给这种格式的[{"command": "grasp the red target", "start_timestep": 0, "end_timestep": 31, "type": "instruction"}]
        #     # If command_list is provided, use the JSON file to determine the relevant timesteps
        #     matching_segments = []

        #     for segment in episode_data:
        #         if segment["command"] in self.command_list: # 筛选和输入commands一样的内容
        #             current_idx = episode_data.index(segment)
        #             if (current_idx + 1 < len(episode_data)and episode_data[current_idx + 1]["type"] == "correction"):
        #                 continue # 如果随机产生的 current_idx则取消是纠正的指令
        #             else: 
        #                 matching_segments.append(segment)        
        #     # Choose a segment randomly among the matching segments
        #     chosen_segment = random.choice(matching_segments) # 然后在随机选，从match匹配的里面选

        #     segment_start, segment_end = (
        #         chosen_segment["start_timestep"],
        #         chosen_segment["end_timestep"],
        #     )
        #     if self.use_language:
        #         command_embedding = torch.tensor(chosen_segment["embedding"]).squeeze()

        #     if segment_start is None or segment_end is None:
        #         raise ValueError(f"Command segment not found for episode {episode_id}")   
             
        if self.use_language or FILTER_MISTAKES:
            while True:
                # Randomly sample a segment
                
                segment = np.random.choice(episode_data) # 从所有数据里面随机选，没有筛选
                current_idx = episode_data.index(segment)
                if (current_idx + 1 < len(episode_data) and episode_data[current_idx + 1]["type"] == "correction"):
                    continue
                segment_start, segment_end = (segment["start_timestep"],segment["end_timestep"],)
                # if end and start are too close, skip
                if segment_end - segment_start + 1 < 20:
                    continue
                command_embedding = torch.tensor(segment["embedding"]).squeeze()
                break    
        ######################################################################################################
        
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            self.is_sim = is_sim
            original_action_shape = root['/action'].shape
            
            ######################################################################################################
            if len(self.command_list) > 0 or self.use_language:
                # Sample within the segment boundaries
                start_ts = np.random.randint(segment_start, segment_end) # 每个指令有固定的步数？
                end_ts = min(segment_end, start_ts + max_len - 2)
            else:
                start_ts = np.random.choice(original_action_shape[0])
                end_ts = original_action_shape[0] - 1
                
            # episode_len = original_action_shape[0] # episode_len 不是固定了的，用end_ts代替episode_len
            
            # if sample_full_episode:
            #     start_ts = 0
            # else:
            #     start_ts = np.random.choice(episode_len)
            ######################################################################################################
            
            # get observation at start_ts only
            gpos = root['/observations/gpos'][start_ts]
            qpos = root['/observations/qpos'][start_ts]
            
            qpos = np.append(qpos,gpos)###### boxjod
            
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
                
            # get all actions after and including start_ts
            if is_sim:
                action = root['/action'][start_ts : end_ts + 1]
                action_len = end_ts - start_ts + 1 # episode_len 不是固定了的
            else:
                action = root['/action'][max(0, start_ts - 1) : end_ts + 1] # hack, to make timesteps more aligned
                action_len = end_ts - max(0, start_ts - 1) + 1 # hack, to make timesteps more aligned

        # print(f"{action_len=}")
        
        padded_action = np.zeros((max_len,) + original_action_shape[1:], dtype=np.float32)
        padded_action[:action_len] = action
        
        is_pad = np.zeros(max_len)
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # Constructing the observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # Adjusting channel
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
        
        if self.use_language:
            return image_data, qpos_data, action_data, is_pad, command_embedding
        else:
            return image_data, qpos_data, action_data, is_pad
        # return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_dir, num_episodes):
    all_qpos_data = []
    all_action_data = []
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]
            gpos = root['/observations/gpos'][()]
            action = root['/action'][()]
        qpos = np.append(qpos,gpos)
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

    stats = {"action_mean": action_mean.numpy().squeeze(), 
             "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), 
             "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos
             } # example_qpos就像是在作弊一样，应该可以大大提高成功率

    return stats


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val, max_len=None, command_list=None, use_language=False, language_encoder=None):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]
    
    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats, max_len, command_list, use_language, language_encoder)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats, max_len, command_list, use_language, language_encoder)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim

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
    ################################################
    random.seed(seed)
    if torch.backends.cudnn.enabled:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
def number_to_one_hot(number, size=501):
    one_hot_array = np.zeros(size)
    one_hot_array[number] = 1
    return one_hot_array

    ################################################
