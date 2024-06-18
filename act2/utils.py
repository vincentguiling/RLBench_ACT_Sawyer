import numpy as np
import random
import torch
import os
import h5py
import json
from torch.utils.data import TensorDataset, DataLoader

import IPython
e = IPython.embed
import cv2 as cv

CROP_TOP = True  # hardcode
FILTER_MISTAKES = False  # Filter out mistakes from the dataset even if not use_language

class  EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, max_len=None, num_queries=None, 
                 command_list=None, use_language=False, language_encoder=None, use_gpos=True, use_diff=True, action_is_qpos=False):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids if len(episode_ids) > 0 else [0] 
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        
        self.max_len = max_len
        self.num_queries = num_queries if (num_queries != None) and (num_queries<max_len) else max_len
        
        self.command_list = [cmd.strip("'\"") for cmd in command_list]
        self.use_language = use_language
        self.language_encoder = language_encoder
        self.transformations = None
        
        self.use_diff = use_diff
        self.action_is_qpos = action_is_qpos

        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        max_len = self.max_len

        sample_full_episode = False # hardcode ### 没有用了

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        
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
        
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            self.is_sim = is_sim
            original_action_shape = root['/action'].shape
            
            if len(self.command_list) > 0 or self.use_language:
                # Sample within the segment boundaries
                start_ts = np.random.randint(segment_start, segment_end) # 每个指令有固定的步数
                end_ts = min(segment_end, start_ts + max_len - 2)
            else:
                start_ts = np.random.choice(original_action_shape[0]) ######################### (30) #
                end_ts = original_action_shape[0] - 1
                
            # episode_len = original_action_shape[0] # episode_len 不是固定了的，用 end_ts 代替episode_len
            # if sample_full_episode:
            #     start_ts = 0
            # else:
            #     start_ts = np.random.choice(episode_len)
            ######################################################################################################
            
            # get observation at start_ts only
            qpos = root['/observations/qpos'][start_ts]
            
            gpos = root['/observations/gpos'][start_ts]
            if self.use_diff:
                qpos_diff = [a-b for a,b in zip(qpos, root['/observations/qpos'][0])]
                qpos = np.append(qpos, qpos_diff[:7])
                gpos_diff = [a-b for a,b in zip(gpos, root['/observations/gpos'][0])]
                gpos = np.append(gpos, gpos_diff[:7])
            
            # gpos = gpos[:7] # 兼顾实物机器人和仿真机器人
            # gpos =  np.append(gpos,qpos[7])
            # qpos = np.append(qpos,gpos)###### boxjod
            
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
                
            # get all actions after and including start_ts
    
            action_len = end_ts - start_ts + 1 
            if action_len <= self.num_queries:

                if self.action_is_qpos:
                    action = root['/action_qpos'][start_ts : end_ts + 1]
                else:
                    action = root['/action'][start_ts : end_ts + 1]
                    
            else:
                if self.action_is_qpos:
                    action = root['/action_qpos'][start_ts : start_ts + self.num_queries]
                else:
                    action = root['/action'][start_ts : start_ts + self.num_queries]
            action_len = min(action_len, self.num_queries)    
            
            # if is_sim:
            # else: # 其实不需要固定最大长度帧的
            #     action = root['/action'][max(0, start_ts - 1) : end_ts + 1] # hack, to make timesteps more aligned
            #     action_len = end_ts - max(0, start_ts - 1) + 1 # hack, to make timesteps more aligned
            
            # 加入历史图像和历史action
            history_images = []
            history_image_dict = dict()

            if 'sorting_program5' in self.dataset_dir or 'close_jar' in self.dataset_dir:# 用夹爪状态分割历史
                qpos_his = root['/observations/qpos'][0:start_ts + 1] # 只需要检测之前状态的改变
                qpos_his_len = len(qpos_his) # 它会等于0

                gripper_state = qpos_his[qpos_his_len-1][7] # 读取了最后的状态
                gripper_change_point = []

                for idx in range(qpos_his_len-1, -1, -1): # 倒着来检测夹爪状态变化
                    if gripper_state != qpos_his[idx][7]:
                        gripper_state = qpos_his[idx][7]
                        gripper_change_point.append(idx+1)
                        break
                gripper_change_point.append(0)
                # print(f"#####\n当前start_ts = {start_ts}, history_len = {qpos_his_len}, now_state = {gripper_state}, history_change = {gripper_change_point}")
                
                change_point = gripper_change_point[0] # 向前推最近的一个夹爪改变点
                history_action_len = start_ts - change_point + 1 # 如果是20-10 = 10，如果是10，那么+1 =11个动作历史因为有0 
                # print(f"now_frame = {start_ts}, last_point = {change_point}, history_action_len = {history_action_len}")
                # now_frame = 34, last_point = 31, history_action_len = 4 这种状态下就可以产生pad的开启新现阶段的效果
                
                if start_ts<=(change_point+self.num_queries): 
                    
                    history_action = root['/action'][change_point : start_ts + 1]
                    for history_idx in range(change_point, start_ts + 1):
                        for cam_name in self.camera_names:
                            history_image_dict[cam_name] = root[f'/observations/images/{cam_name}'][history_idx]
                        history_images.append(history_image_dict.copy())
                
                else: # 距离上一个夹爪分界点，之间的历史长度大于chunking数量，那就取前面这么chunking个
                    history_action = root['/action'][start_ts - self.num_queries : start_ts]
                    for history_idx in range(start_ts - self.num_queries, start_ts + 1):
                        for cam_name in self.camera_names:
                            history_image_dict[cam_name] = root[f'/observations/images/{cam_name}'][history_idx]
                        history_images.append(history_image_dict.copy())
                history_action_len = min(history_action_len, self.num_queries)
                
                # 更新diff
                if self.use_diff:
                    qpos = root['/observations/qpos'][start_ts]
                    
                    # print(f"{qpos=} \n {root['/observations/qpos'][change_point]=}")
                    qpos_diff = [a-b for a,b in zip(qpos, root['/observations/qpos'][change_point])]
                    # print(f"{qpos_diff=}")
                    qpos = np.append(qpos, qpos_diff[:7])
                    # print(f"{qpos=}\n")
                    
                    gpos = root['/observations/gpos'][start_ts]
                    gpos_diff = [a-b for a,b in zip(gpos, root['/observations/gpos'][change_point])]
                    gpos = np.append(gpos, gpos_diff[:7])
                         
            else: # 不需要分段任务
                history_action_len = start_ts + 1
                if history_action_len <= self.num_queries: # 存在的历史长度小于chunking数量
                    history_action = root['/action'][0 : start_ts + 1]
                    for history_idx in range(start_ts + 1):
                        for cam_name in self.camera_names:
                            history_image_dict[cam_name] = root[f'/observations/images/{cam_name}'][history_idx]
                        history_images.append(history_image_dict.copy())
                else:
                    history_action = root['/action'][start_ts - self.num_queries : start_ts]
                    for history_idx in range(start_ts - self.num_queries, start_ts + 1): # 存在的历史长度大于chunking数量，但是也只读这么多个，为了节省空间
                        for cam_name in self.camera_names:
                            history_image_dict[cam_name] = root[f'/observations/images/{cam_name}'][history_idx]
                        history_images.append(history_image_dict.copy())
                history_action_len = min(history_action_len, self.num_queries)
            
            
        padded_action = np.zeros((self.num_queries,) + original_action_shape[1:], dtype=np.float32) 
        # 随机抽样的结果当中只剩余17步了之后，也会推广到32，方便后面做action chunking 的截取，也就是说最大的quary就是max_len
        padded_action[:action_len] = action[:action_len] # 如果这里报错，去看看command json是不是多给了一个action步骤
        padded_history_action = np.zeros((self.num_queries,) + original_action_shape[1:], dtype=np.float32)
        is_pad_action = np.zeros(self.num_queries)
        is_pad_action[action_len:] = 1
        
        history_action = history_action[::-1] # 翻转 history_action
        padded_history_action[:history_action_len] = history_action[:history_action_len] # 往前面添加historyaction
        is_pad_history = np.zeros(self.num_queries)
        is_pad_history[history_action_len:] = 1 # 前面是空白，后面是最近的action_pos
        
        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names: ###############################################################
            # if 'sawyer' in self.dataset_dir: # 已经在数据集中调整过了
            #     image_dict[cam_name] = cv.resize(image_dict[cam_name], (0, 0), fx=0.25, fy=0.25, interpolation=cv.INTER_LINEAR)
                
            # if mamba
            # image_dict[cam_name] = image_dict[cam_name][0:120, 20:140, :] # Slicing to crop the image
            # image_dict[cam_name] = cv.resize(image_dict[cam_name], (224, 224))
            # print(image_dict[cam_name].shape)
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)
        
        # print(f"{all_cam_images.shape=}")
        
        # history_images 加入图像
        history_all_cam_images = []
        for history_idx in range(history_action_len):
            all_history_cam_images = []
            for cam_name in self.camera_names: 
                # if 'sawyer' in self.dataset_dir:
                #     history_images[history_idx][cam_name] = cv.resize(history_images[history_idx][cam_name], (0, 0), fx=0.25, fy=0.25, interpolation=cv.INTER_LINEAR)
                    
                all_history_cam_images.append(history_images[history_idx][cam_name])
            all_history_cam_images = np.stack(all_cam_images, axis=0)
            history_all_cam_images.append(all_history_cam_images)
            
        original_images_shape = np.shape(history_all_cam_images)
        padded_history_images = np.zeros((self.num_queries,) + original_images_shape[1:], dtype=np.float32)
        history_all_cam_images = history_all_cam_images[::-1] # 以第一个维度倒序, 翻转图像
        padded_history_images[:history_action_len] = history_all_cam_images[:history_action_len] # 往前面添加 padded_history_images
        
        history_all_cam_images = np.array(padded_history_images)
        history_images_data = torch.from_numpy(history_all_cam_images)
        
        # Constructing the observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        gpos_data = torch.from_numpy(gpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad_action = torch.from_numpy(is_pad_action).bool()
        history_action_data = torch.from_numpy(padded_history_action).float()
        is_pad_history = torch.from_numpy(is_pad_history).bool()
        
        # Adjusting channel
        image_data = torch.einsum('k h w c -> k c h w', image_data)
        history_images_data = torch.einsum('q k h w c -> q k c h w', history_images_data) # 多一个querry，chunking的队列长度
        
        # normalize image and change dtype to float
        image_data = image_data / 255.0
        history_images_data = history_images_data / 255.0
        
        # 标准化过程
        # np.set_printoptions(linewidth=300)
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"] 
        # print(self.norm_stats["qpos_mean"])
        # print(f"标准化之前{qpos_data=}")
        # print(qpos_data - self.norm_stats["qpos_mean"])
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
        # print(f"标准化之后：{qpos_data=}")
        gpos_data = (gpos_data - self.norm_stats["gpos_mean"]) / self.norm_stats["gpos_std"]
        history_action_data = (history_action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        
        # return
        
        if self.use_language:
            return image_data, qpos_data, gpos_data, history_images_data, history_action_data, is_pad_history, action_data, is_pad_action, command_embedding
        else:
            return image_data, qpos_data, gpos_data, history_images_data,  history_action_data, is_pad_history, action_data, is_pad_action

def get_norm_stats(dataset_dir, num_episodes, use_gpos=True, use_diff=True, action_is_qpos=False):
    all_qpos_data = []
    all_gpos_data = []
    all_action_data = []
    max_episode_len = 0    
            
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            episode_len = root['/action'].shape[0]
            qpos = root['/observations/qpos'][()]
            # print(f"{qpos.shape=}")
            gpos = root['/observations/gpos'][()]
            if action_is_qpos:
                action = root['/action_qpos'][()]
            else:
                action = root['/action'][()]
                
        #################
        # 更正，反而成功率下降
        if use_diff: 
            # 还要查找 change point
            qpos_len = len(qpos) # 它会等于0
            gripper_state = qpos[qpos_len-1][7] # 读取了最后的状态
            gripper_change_point = []

            for idx in range(qpos_len-1, -1, -1): # 倒着来检测夹爪状态变化
                if gripper_state != qpos[idx][7]:
                    gripper_state = qpos[idx][7]
                    gripper_change_point.append(idx+1)
                    # break
            gripper_change_point.append(0)
            
            # print(f"history_change = {gripper_change_point}")
            
            for idx in range(qpos.shape[0]):
                for c_idx in range(len(gripper_change_point)):
                    if gripper_change_point[c_idx] <= idx: # 0 <80,<30,<0正确
                        last_change_point = gripper_change_point[c_idx]
                        break
                # print(f"{idx=}, {last_change_point=}")
                qpos_diff = [a-b for a,b in zip(qpos[idx], qpos[last_change_point])]  # 就是因为精细化之后这里没改，导致错误
                gpos_diff = [a-b for a,b in zip(gpos[idx], gpos[last_change_point])]
                
                qpos_temp = np.concatenate((qpos[idx],qpos_diff[:7]), axis=0)[np.newaxis,:]
                gpos_temp = np.concatenate((gpos[idx],gpos_diff[:7]), axis=0)[np.newaxis,:]
                
                if idx == 0:
                    qpos_data = qpos_temp
                    gpos_data = gpos_temp
                else:
                    qpos_data = np.concatenate((qpos_data, qpos_temp), axis=0)     
                    gpos_data = np.concatenate((gpos_data, gpos_temp), axis=0)                     
            # print(f"{qpos_data.shape=}")
       
            if episode_idx == 0:
                all_qpos_data = qpos_data
                all_gpos_data = gpos_data
                all_action_data = action
            else:
                all_qpos_data = np.concatenate((all_qpos_data, qpos_data), axis=0)
                all_gpos_data = np.concatenate((all_gpos_data, gpos_data), axis=0)
                all_action_data = np.concatenate((all_action_data, action), axis=0)
        else: # 原始的样子
            if episode_idx == 0:
                all_qpos_data = qpos
                all_gpos_data = gpos
                all_action_data = action
            else:
                all_qpos_data = np.concatenate((all_qpos_data, qpos), axis=0)
                all_gpos_data = np.concatenate((all_gpos_data, gpos), axis=0)
                all_action_data = np.concatenate((all_action_data, action), axis=0)
                
        #################

    #     # 错误但是正确率挺高的
    #     if use_diff:
    #         qpos_diff = [a-b for a,b in zip(qpos, qpos[0])] # 这不是该是change_point吗 # (90,8) - (1,8) => (8,8)
    #         # print(f"{qpos.shape=}")
    #         # print(f"{qpos[0].shape=}")
    #         # print(f"{qpos_diff.shape=}") # (90,8)才是对的
    #         # print(f"{len(qpos)=}") # (90,8)才是对的
    #         # print(f"{len(qpos)=}") # (90,8)才是对的
    #         qpos = np.append(qpos, qpos_diff)                       # 出问题了 # (90,8) + (7,8) = 776  # [:7]
    #         # print(f"{len(qpos)=}") # (90,8)才是对的
    #         gpos_diff = [a-b for a,b in zip(gpos, gpos[0])]
    #         gpos = np.append(gpos, gpos_diff) # [:7]
    #     all_qpos_data.append(torch.from_numpy(qpos))  # (50, 776)
    #     all_gpos_data.append(torch.from_numpy(gpos))
    #     all_action_data.append(torch.from_numpy(action))
        
        
    # # print(f"{all_qpos_data[0].shape=}")
    
    # # 错误但是正确率挺高的 (标准化过程涉及到所有关节和位移矢量牵引，计算出来的值，更加泛化？) 【更加泛性的标准差？】
    # all_qpos_data = torch.stack(all_qpos_data)  # (50, 776)
    # print(f"{all_qpos_data.shape=}") 
    
    # all_gpos_data = torch.stack(all_gpos_data)  # (50, 776)
    # print(f"{all_gpos_data.shape=}") 
    
    # all_action_data = torch.stack(all_action_data) # (50, 90, 8)
    # print(f"{all_action_data.shape=}") 
    
    # # normalize action data
    # action_mean = all_action_data.mean(dim=[0, 1], keepdim=True) # (1, 1, 8)
    # print(f"{action_mean.shape=}") 
    
    # action_std = all_action_data.std(dim=[0, 1], keepdim=True)  
    # action_std = torch.clip(action_std, 1e-2, np.inf) # clipping
    
    # # normalize qpos data
    # qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True) # (1, 1)
    # print(f"{qpos_mean.shape=}") 
    # qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True) 
    # qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping
    
    # gpos_mean = all_gpos_data.mean(dim=[0, 1], keepdim=True) # (1, 1)
    # print(f"{gpos_mean.shape=}") 
    # gpos_std = all_gpos_data.std(dim=[0, 1], keepdim=True)
    # gpos_std = torch.clip(gpos_std, 1e-2, np.inf) # clipping
    
    
    # 更正，反而成功率下降
    all_qpos_data = np.array(all_qpos_data)
    all_gpos_data = np.array(all_gpos_data)
    all_action_data = np.array(all_action_data)
    
    all_qpos_data = torch.from_numpy(all_qpos_data) # 姿态要求不高
    print(f"{all_qpos_data.shape=}") 
    all_gpos_data = torch.from_numpy(all_gpos_data) # 姿态要求不高
    print(f"{all_gpos_data.shape=}") 
    all_action_data = torch.from_numpy(all_action_data) # 动作精度要求高
    print(f"{all_action_data.shape=}") 

    # normalize action data
    action_mean = all_action_data.mean(dim=0, keepdim=True).unsqueeze(0) # [1, 1, 8]
    print(f"{action_mean.shape=}") 
    action_std = all_action_data.std(dim=0, keepdim=True).unsqueeze(0)
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping
    
    # normalize qpos data
    if use_diff: 
        qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True).unsqueeze(0) # [1, 1, 1]
        print(f"{qpos_mean.shape=}")    
        qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True).unsqueeze(0)
        qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping
        
        gpos_mean = all_gpos_data.mean(dim=[0, 1], keepdim=True).unsqueeze(0) # [1, 1, 1]
        print(f"{gpos_mean.shape=}") 
        gpos_std = all_gpos_data.std(dim=[0, 1], keepdim=True).unsqueeze(0)
        gpos_std = torch.clip(gpos_std, 1e-2, np.inf) # clipping
        # print(f"{gpos_mean=}")
    else:
        qpos_mean = all_qpos_data.mean(dim=0, keepdim=True).unsqueeze(0) # [1, 1, 8]
        print(f"{qpos_mean.shape=}")    
        qpos_std = all_qpos_data.std(dim=0, keepdim=True).unsqueeze(0)
        qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping
        
        gpos_mean = all_gpos_data.mean(dim=0, keepdim=True).unsqueeze(0) # [1, 1, 8]
        print(f"{gpos_mean.shape=}") 
        gpos_std = all_gpos_data.std(dim=0, keepdim=True).unsqueeze(0)
        gpos_std = torch.clip(gpos_std, 1e-2, np.inf) # clipping
        
    
    stats = {"action_mean": action_mean.numpy().squeeze(), 
             "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), 
             "qpos_std": qpos_std.numpy().squeeze(),
             "gpos_mean": gpos_mean.numpy().squeeze(), 
             "gpos_std": gpos_std.numpy().squeeze(),
             "example_qpos": qpos,
             "example_gpos": gpos
             } # example_qpos就像是在作弊一样，应该可以大大提高成功率

    return stats


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val, max_len=None, num_queries=None, command_list=None, use_language=False, language_encoder=None, use_gpos=True, use_diff=True, action_is_qpos=False):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]
    
    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes, use_gpos=use_gpos, use_diff=use_diff, action_is_qpos=action_is_qpos)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats, max_len, num_queries, command_list, use_language, language_encoder,use_gpos=use_gpos, use_diff=use_diff, action_is_qpos=action_is_qpos)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats, max_len, num_queries, command_list, use_language, language_encoder, use_gpos=use_gpos, use_diff=use_diff, action_is_qpos=action_is_qpos)
    
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
    random.seed(seed)
    if torch.backends.cudnn.enabled:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
def number_to_one_hot(number, size=501):
    one_hot_array = np.zeros(size)
    one_hot_array[number] = 1
    return one_hot_array

# audio2text parrot
class HParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()


def get_gpu_mem_info(gpu_id=0):
    """
    根据显卡 id 获取显存使用信息, 单位 MB
    :param gpu_id: 显卡 ID
    :return: total 所有的显存，used 当前使用的显存, free 可使用的显存
    """
    import pynvml
    pynvml.nvmlInit()
    if gpu_id < 0 or gpu_id >= pynvml.nvmlDeviceGetCount():
        print(r'gpu_id {} 对应的显卡不存在!'.format(gpu_id))
        return 0, 0, 0

    handler = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)
    total = round(meminfo.total / 1024 / 1024, 2)
    used = round(meminfo.used / 1024 / 1024, 2)
    free = round(meminfo.free / 1024 / 1024, 2)
    return total, used, free

def get_cpu_mem_info():
    import psutil
    """
    获取当前机器的内存信息, 单位 MB
    :return: mem_total 当前机器所有的内存 mem_free 当前机器可用的内存 mem_process_used 当前进程使用的内存
    """
    mem_total = round(psutil.virtual_memory().total / 1024 / 1024, 2)
    mem_free = round(psutil.virtual_memory().available / 1024 / 1024, 2)
    mem_process_used = round(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024, 2)
    return mem_total, mem_free, mem_process_used
