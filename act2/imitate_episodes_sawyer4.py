#######################################
# # 加入FiLM的单步训练和推理
# python act2/imitate_episodes_sawyer4.py \
#     --task_name sorting_program21 \
#     --ckpt_dir Trainings \
#     --policy_class ACT --kl_weight 10 --chunk_size 10 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
#     --num_epochs 5000  --lr 1e-5 --seed 0 --backbone efficientnet_b0film \
#     --use_language --language_encoder distilbert \
#     ; \
# python act2/imitate_episodes_sawyer4.py \
#     --task_name sorting_program21 \
#     --ckpt_dir Trainings \
#     --policy_class ACT --kl_weight 10 --chunk_size 10 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
#     --num_epochs 5000  --lr 1e-5 --seed 0 --backbone efficientnet_b0film \
#     --use_language --language_encoder distilbert \
#     --eval --temporal_agg 
#######################################

import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange

from constants import DT
from constants import PUPPET_GRIPPER_JOINT_OPEN
from utils import load_data # data functions
from utils import compute_dict_mean, set_seed, detach_dict # helper functions
from policy import ACTPolicy, CNNMLPPolicy
from visualize_episodes import save_videos

from sim_env import BOX_POSE
from pyrep.errors import ConfigurationPathError
from command_script.command_utils import initialize_model_and_tokenizer, encode_text
import IPython
e = IPython.embed

def main(args):
    
    np.set_printoptions(linewidth=300)
    set_seed(1)
    # command line parameters
    is_eval = args['eval']
    policy_class = args['policy_class']
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']
    batch_size_train = args['batch_size'] # train 和 eval 使用相同的 batch_size
    batch_size_val = args['batch_size']
    num_epochs = args['num_epochs']
    num_verification = args['num_verification']
    variation = args['variation']
    
    is_sim = True 
    if is_sim:
        from constants import SIM_TASK_CONFIGS
        task_config = SIM_TASK_CONFIGS[task_name]

    dataset_dir = task_config['dataset_dir']
    num_episodes = task_config['num_episodes'] * task_config['num_variation'] 
    print(f"{task_config['num_episodes']=}, {task_config['num_variation']=}, {num_episodes=}, ")
    
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']
    
    commands = args["command"].split(",") if args["command"] else [] # task_config['commands'] # 
    use_language = args["use_language"]
    language_encoder = args["language_encoder"]
    
    # max_skill_len = (args["max_skill_len"] if args["max_skill_len"] is not None else episode_len)
    max_skill_len = episode_len
    
    
    ####################
    # 1.训练和推理文件中，只处理【模型结构的差异】，涉及编码器的都保持一致的输入，在模型实际推理过程中才做筛选： action_is_qpos， use_gpos， state_dim
    # 2.在数据集读取中会判断，是否使用文本指令（use_language），是否使用位移矢量牵引（gpos_diff）
    # 3.在模型推理过程中才会考虑使用那个编码器（Z， history_action, history_action_images）
    ####################
    
    # 初始参数 ACT3
    action_dim = 8 # 输出的姿态 7+1
    
    if 'ACT0' in policy_class: # 最原始的参数 对应的数据集不同
        action_is_qpos = True
        use_gpos = False
        state_dim =  8 
        
    elif 'ACT1' in policy_class: # qpos => gpos 1
        action_is_qpos = False
        use_gpos = False 
        state_dim =  8
        
    elif 'ACT2' in policy_class: # (qpos)+(gpos)=> gpos 2
        action_is_qpos = False
        use_gpos = True # True
        state_dim =  8 
        
    else: # 'ACT3' in policy_class: # (qpos+qdiff)+(gpos+gdiff)=> gpos 3 (默认)
        action_is_qpos = False 
        use_gpos = True 
        state_dim =  15 
    
    lr_backbone = 1e-5
    backbone = args['backbone']
    print("backbone:",backbone)
    # backbone = 'resnet18' # 图像基础处理网络是ResNet18
    if 'ACT' in policy_class: # policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8 # 8头注意力机制
        policy_config = {'policy_class': policy_class,
                         'camera_names': camera_names,
                         'backbone': backbone,
                         'state_dim': state_dim,
                         'use_gpos': use_gpos,

                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         
                         'lr': args['lr'],
                         'kl_weight': args['kl_weight'],
                         'lr_backbone': lr_backbone,
                         
                         'action_dim': action_dim,
                         'num_queries': args['chunk_size'],
                         }
    elif policy_class == 'CNNMLP':
        policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': 1,
                         'camera_names': camera_names,}
    else:
        raise NotImplementedError
    
    # 增加参数保存
    chunk_size = args['chunk_size']
    batch_size = args['batch_size']
    ckpt_dir = args['ckpt_dir'] + f'/{task_name}/{policy_class}_{num_episodes}demo_{episode_len}step_{chunk_size}chunk_{batch_size}batch_{backbone}'
    # print(f"train on {camera_names}")
    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'episode_len': episode_len,
        'lr': args['lr'],
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
        'real_robot': not is_sim,
        "use_language": use_language,
        "language_encoder": language_encoder,
        "max_skill_len": max_skill_len,
    }
    
    use_diff = False if state_dim == 8 else True
    print(f"{action_is_qpos=}, {use_gpos=}, {state_dim=}, {use_diff=}")
    if is_eval: # 如果是验证的话
        ckpt_names = [f'policy_best_epoch{num_epochs}.pth']
        results = []
        for ckpt_name in ckpt_names:
            success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=True, num_verification=num_verification,variation=variation, use_diff=use_diff, action_is_qpos=action_is_qpos) # 调用 eval_bc() 直接验证
            results.append([ckpt_name, success_rate, avg_return])

        for ckpt_name, success_rate, avg_return in results:
            print(f'{ckpt_name}: {success_rate=} {avg_return=}')
        print()
        exit() # eval 结束后退出程序
      
    # 如果不是evaluation
    train_dataloader, val_dataloader, stats, _ = load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val, 
                                                           max_len=max_skill_len, num_queries=chunk_size, command_list=commands, use_language=use_language, language_encoder=language_encoder,
                                                           use_gpos=use_gpos, use_diff=use_diff, action_is_qpos=action_is_qpos)

    # save dataset stats
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
        
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl') # pkl 是 pickle 打包的文件
    with open(stats_path, 'wb') as f: 
        pickle.dump(stats, f)
        
    train_bc(train_dataloader, val_dataloader, config) # 调用 train_bc() 训练，保存最新的为 best_ckpt_info 文件
    

def make_policy(policy_class, policy_config):
    if 'ACT' in policy_class: # policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy

def make_optimizer(policy_class, policy):
    if 'ACT' in policy_class: #policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers() 
    else:
        raise NotImplementedError
    return optimizer

from rlbench.backend.utils import float_array_to_rgb_image
from rlbench.backend.const import DEPTH_SCALE
def get_image(ts, camera_names): # 推理的时候采用到
    curr_images = []
    
    curr_image = rearrange(ts.wrist_rgb, 'h w c -> c h w')
    
    curr_images.append(curr_image)    
    # if mamba
    # image_dict[cam_name] = image_dict[cam_name][0:120, 20:140, :] # Slicing to crop the image
    # image_dict[cam_name] = cv.resize(image_dict[cam_name], (224, 224))
    # print(image_dict[cam_name].shape)
    if "wrist_depth" in camera_names:
        # wrist_depth = float_array_to_rgb_image(ts.wrist_depth, scale_factor=DEPTH_SCALE)
        # wrist_depth = np.clip(np.array(wrist_depth), 0, 255).astype(np.uint8)
        
        import cv2
        wrist_depth0 = ts.wrist_depth*255.0*5
        wrist_depth = cv2.applyColorMap(cv2.convertScaleAbs(wrist_depth0, alpha=1), cv2.COLORMAP_JET)
        
        curr_image = rearrange(wrist_depth, 'h w c -> c h w')
        curr_images.append(curr_image)
        
    if "head" in camera_names:    
        curr_image = rearrange(ts.head_rgb, 'h w c -> c h w')
        curr_images.append(curr_image)
        
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image

def generate_command_embedding(command, t, language_encoder, tokenizer, model, instructor=None):

    command_embedding = encode_text(command, language_encoder, tokenizer, model)
    command_embedding = torch.tensor(command_embedding).cuda()
    if instructor is not None:
        command_embedding = instructor.get_nearest_embedding(command_embedding)[0]
    return command_embedding
    
    
def eval_bc(config, ckpt_name, save_episode=True, num_verification=50, variation=0, use_gpos=True, use_diff=True, action_is_qpos=False):
    seed = 10
    set_seed(seed)
    ckpt_dir = config['ckpt_dir']
    state_dim = config['policy_config']['state_dim']
    action_dim = config['policy_config']['action_dim']
    
    real_robot = config['real_robot']
    policy_class = config['policy_class']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    temporal_agg = config['temporal_agg']
    
    hidden_dim = policy_config['hidden_dim']
    ##################################################
    use_language = config["use_language"]
    language_encoder = config["language_encoder"]
    max_skill_len = config["max_skill_len"]
    if use_language:
        tokenizer, model = initialize_model_and_tokenizer(language_encoder)
        assert tokenizer is not None and model is not None
    ######################################################
    
    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    ckpt_name0 =ckpt_name.split('.')[0]
    
    print(loading_status)
    policy.cuda()
    policy.eval() # 将policy配置为eval模式
    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    
    pre_process_qpos = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    pre_process_gpos = lambda s_gpos: (s_gpos - stats['gpos_mean']) / stats['gpos_std']
    pre_process_action_history = lambda s_action: (s_action - stats['action_mean']) / stats['action_std']
    
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']
    
    # load environment
    if not real_robot:
        from sim_env import make_sim_env
        env = make_sim_env(task_name)
        print("random seed = ", seed)
        env_max_reward = 1 # env.task.max_reward
    # chunk_size = num_queries
    query_frequency = policy_config['num_queries']
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']
        
    ##########################################################################################################
    max_timesteps = int(max_timesteps * 1.0) # may increase for real-world tasks
    # max_timesteps = 45
    ##########################################################################################################
    
    num_rollouts = num_verification # 验证 50 次
    
    command_list = []
    command_embedding = None
    
    episode_returns = []
    highest_rewards = []
    for rollout_id in range(num_rollouts):
        gripper_flag = 0
        
        if variation >= 0:
            env.set_variation(variation) 
        else:
            random_variation = np.random.randint(3)
            env.set_variation(random_variation) 
            
        descriptions, ts_obs = env.reset() # 重置帧数
        # print(f"command is {descriptions[0]}")
        
        ### evaluation loop
        if temporal_agg: # 是否使用GPU提前读取数据？？应该可以提高 eval 速度
            all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, action_dim]).cuda() ## 输出8维，但是输入时15维度

        qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        gpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        
        image_list = [] # for visualization
        qpos_list = []
        gpos_list = []
        target_gpos_list = []
        rewards = []
        
        history_action = np.zeros((num_queries,) + (action_dim,), dtype=np.float32)
        history_image_feature = np.zeros((2,num_queries,) + (hidden_dim,), dtype=np.float32)
        
        with torch.inference_mode():
            path = []
            t = 0
            for timestep in range(max_timesteps): # 最大帧数
                obs = ts_obs

                if use_diff and timestep == 0:
                    qpos_initial = obs.joint_positions
                    gpos_initial = obs.gripper_pose
                
                is_pad_history = np.zeros(max_timesteps)
                is_pad_history[timestep:] = 1
                is_pad_history = torch.from_numpy(is_pad_history).bool().cuda()
                
                if(rollout_id%5 == 0): # 限制保存数量，增快速度
                    # image_list.append({'wrist':obs.wrist_rgb, 'head':obs.head_rgb, })
                    image_list.append({'front':obs.front_rgb, 'head':obs.head_rgb, 'wrist':obs.wrist_rgb})
                
                qpos_numpy = np.array(np.append(obs.joint_positions, obs.gripper_open)) # 7 + 1 = 8
                gpos_numpy = np.array(np.append(obs.gripper_pose, obs.gripper_open)) # 7 + 1 = 8
                
                if use_diff:
                    qpos_diff = [a - b for a,b in zip(obs.joint_positions, qpos_initial)]
                    qpos_numpy = np.array(np.append(qpos_numpy, qpos_diff)) # 7 + 1 + 7 = 15
                    # print(f"{qpos_numpy=}")
                    gpos_diff = [a - b for a,b in zip(obs.gripper_pose, gpos_initial)] ########## 卧槽历史遗留问题 joint_positions gripper_pose
                    gpos_numpy = np.array(np.append(gpos_numpy, gpos_diff)) # 7 + 1 + 7 = 15
                    # print(gpos_numpy)
                
                # print(f"{qpos_numpy=}")
                qpos = pre_process_qpos(qpos_numpy)
                # print(f"{qpos=}")
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                qpos_history[:, t] = qpos

                gpos = pre_process_gpos(gpos_numpy)
                gpos = torch.from_numpy(gpos).float().cuda().unsqueeze(0)
                gpos_history[:, t] = gpos
                
                
                history_action_numpy = np.array(history_action)
                history_action_numpy = pre_process_action_history(history_action_numpy)
                history_action_numpy = torch.from_numpy(history_action_numpy).float().cuda()
                
                curr_image = get_image(obs, camera_names) # 获取帧数据的图像

                ### query policy
                if 'ACT' in config['policy_class']:  # config['policy_class'] == "ACT":
                    if t % query_frequency == 0:
                        
                        if use_language and (t % max_skill_len == 0) :
                            # Check if an intervention is needed; if so, language correction
                            command = descriptions[0] 
                            # command =  "grasp the blue target" #"aaaaaaaaaaaaaaaaaa"# commands[0] # "reach to the red target"  # "pick up the plate" "grasp the blue target"
                            command_embedding = generate_command_embedding(command, t, language_encoder, tokenizer, model)
                            # print(command_embedding)
                        
                        if not use_gpos:
                            gpos = None    
                        all_actions, image_feature = policy(qpos, gpos, curr_image, 
                                            history_image_feature, history_action_numpy, is_pad_history=is_pad_history, 
                                            actions=None, is_pad_action=None, command_embedding=command_embedding) # 100帧才预测一次，# 没有提供 action 数据，是验证模式
                        
                        language_correction = False
                        if use_language:
                            prefix = "user" if language_correction else "prediction"
                            command_list.append(f"{prefix}: {command}")
                        
                    if temporal_agg: # 做了一个 Action Chunking
                        all_time_actions[[t], t:t+num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        
                        # 在生成的多个序列中不是简单的平均，又做了一个运算（时间集成？？）
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum() # 做了一个归一化
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1) # 压缩维度
                        
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                        
                    else: # 如果没用的话，等于就是 100帧才预测一次，然后挨着执行
                        raw_action = all_actions[:, t % query_frequency]
                        
                elif config['policy_class'] == "CNNMLP":
                    raw_action = policy(qpos, gpos, curr_image) 
                else:
                    raise NotImplementedError
                
                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)  # 就是因为这个的保护和限制，所以初始化位置不能随意改变
                # target_qpos = action

                history_action = np.insert(history_action, 0, action, axis=0)[:num_queries]
                
                history_image_feature[0] = np.insert(history_image_feature[0], 0, image_feature[0], axis=0)[:num_queries]
                history_image_feature[1] = np.insert(history_image_feature[1], 0, image_feature[1], axis=0)[:num_queries]
                
                ###################################################
                # 将action_diff作为action
                # gpos_diff = action[:7]
                # # gpos_diff = [elem / 10 for elem in gpos_diff]
                # target_gpos= [obs.gripper_pose + gpos_diff]
                # action = np.append(target_gpos, action[7])
                # print(f"{action=}")
                
                if action_is_qpos:# 将qpos作为action
                    ts_obs, reward, terminate = env.step(action)
                    qpos_list.append(qpos_numpy)
                    target_gpos_list.append(action)
                    rewards.append(reward) # 由仿真环境 step 产生 reward：0，1，2，3，4，4代表全部成功
                else:# 将gpos作为action
                    try:
                        next_gripper_position = action[0:3] # next 
                        next_gripper_quaternion = action[3:7]
                        path.append(env._robot.arm.get_linear_path(position=next_gripper_position, quaternion=next_gripper_quaternion, steps=10, relative_to=env._robot.arm, ignore_collisions=True))
                        gripper_state = action[7]
                        # print(f"{timestep}:{gripper_state=}")
                        # 夹爪控制###############################################################################################
                        done = False
                        if task_name =="sorting_program5" or task_name =="close_jar":
                            if gripper_state < 0.60 and gripper_flag < 2 : # 适合步骤1 夹取
                                print(timestep,": close_gripper: ", gripper_state)
                                gripper_flag = gripper_flag + 2 # 留出一帧错误
                                # while done != True:
                                done = env._robot.gripper.actuate(0, 1.0)
                                    # env._scene.step() # Scene 步进
                                
                                # 清空历史信息
                                history_action = np.zeros((num_queries,) + (action_dim,), dtype=np.float32)
                                history_image_feature = np.zeros((2,num_queries,) + (hidden_dim,), dtype=np.float32)
                                qpos_initial = obs.joint_positions
                                gpos_initial = obs.gripper_pose

                            elif gripper_state > 0.6 and gripper_flag == 2 :# 适合步骤1 夹取
                                print(timestep, ": open_gripper: ", gripper_state)
                                gripper_flag = gripper_flag + 1
                                while done != True:
                                    done = env._robot.gripper.actuate(1, 0.4)
                                    env._scene.step() # Scene 步进
                                    
                                # 清空历史信息
                                history_action = np.zeros((num_queries,) + (action_dim,), dtype=np.float32)
                                history_image_feature = np.zeros((2,num_queries,) + (hidden_dim,), dtype=np.float32)
                                qpos_initial = obs.joint_positions
                                gpos_initial = obs.gripper_pose
                                
                        else:
                            if gripper_state < 0.90 and gripper_flag < 2 : # 适合步骤1 夹取
                                print(timestep,": close_gripper: ", gripper_state)
                                gripper_flag = gripper_flag + 2 # 留出一帧错误
                                # while done != 1:
                                done = env._robot.gripper.actuate(0, 1.0)
                                    # env._scene.step() # Scene 步进
                            elif gripper_state > 0.7 and gripper_flag == 2 :# 适合步骤1 夹取
                                print(timestep, ": open_gripper: ", gripper_state)
                                gripper_flag = gripper_flag + 1
                                while done != True:
                                    done = env._robot.gripper.actuate(1, 0.4)
                                    env._scene.step() # Scene 步进
                        
                        path[t].visualize() # 在仿真环境中画出轨迹
                        
                        done = False # 当done 置为 True 的时候，说明预测的轨迹执行完毕了
                        while done != 1: # 如果 done 是 False 则执行
                            done = path[t].step() # ArmConfigurationPath类型的step运行载入下一帧动作
                            env._scene.step() # Scene 步进
                            
                        ts_obs = env._scene.get_observation()
                        reward, _ = env._task.success() # 任务是否完成状态读取
                        qpos_list.append(qpos_numpy)
                        gpos_list.append(gpos_numpy)
                        target_gpos_list.append(action)
                        rewards.append(reward) # 由仿真环境 step 产生 reward：0，1，2，3，4，4代表全部成功
                        
                        # if gripper_flag == 0 and reward: # 目标是松爪，开始开爪子（对于复杂的任务来说，有点简单了，还是用目标性图像来表征吧）
                        #     break
                    except ConfigurationPathError: 
                        print("ConfigurationPathError ", t) # , "path lens: ",len(path))
                        break # 跳出推理循环

                        # 不跳出，而是前面的2/3步数。在相同的观测下面有相同的推理？？？，不是，是反复迭代退回了
                        # np.random.seed(0)
                        t_back = (t*8)//10
                        
                        back_gripper_pose = [elem * (1 + (np.random.randint(100) - 50)/50) for elem in target_gpos_list[t_back][:7]]
                        for i in range(t - t_back):
                            path.pop()
                            qpos_list.pop()
                            target_gpos_list.pop()
                        t = t_back
                            
                        # back_gripper_pose = target_gpos_list[(t*9)//10][:7]
                        next_gripper_position = back_gripper_pose[0:3] # next 
                        next_gripper_quaternion = back_gripper_pose[3:7]
                        try:
                            # path.pop()
                            path.append(env._robot.arm.get_linear_path(position=next_gripper_position, quaternion=next_gripper_quaternion, steps=10, relative_to=env._robot.arm, ignore_collisions=True))
                            path[t].visualize() # 在仿真环境中画出轨迹
                            
                            done = False # 当done 置为 True 的时候，说明预测的轨迹执行完毕了
                            while done != 1: # 如果 done 是 False 则执行
                                done = path[t].step() # ArmConfigurationPath类型的step运行载入下一帧动作
                                env._scene.step() # Scene 步进
                                
                            ts_obs = env._scene.get_observation()
                            qpos_list.append(qpos_numpy)
                            target_gpos_list.append(target_qpos)    
                            
                        except ConfigurationPathError:
                            print("ConfigurationPathError ConfigurationPathError")
                            break # 跳出推理循环
                    
                t = t + 1
                
            plt.close()
            
        # for i in range(t+1): # clear the path history
        #     path[i].clear_visualization()
        rewards = np.array(rewards) # 
        episode_return = np.sum(rewards[rewards!=None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)  # 这附近报错的话，可能是运行错误，仿真器崩了
        highest_rewards.append(episode_highest_reward)
        # print(f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}')
        print(f'{rollout_id} Rollout with {t} steps for [{descriptions[0]}]: {episode_highest_reward==env_max_reward}')
        if(rollout_id%5 == 0): # 限制保存数量，增快速度
            if save_episode:
                save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f'video_{ckpt_name0}_{rollout_id}_{episode_highest_reward==env_max_reward}.mp4'))

    success_rate = np.mean(np.array(highest_rewards) == env_max_reward) # 计算占比
    avg_return = np.mean(episode_returns) # 计算平均数
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    for r in range(env_max_reward+1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n'

    print(summary_str)

    # save success rate to txt
    result_file_name = 'result_' + ckpt_name0 + f'({more_or_equal_r_rate*100}%).txt'
    with open(os.path.join(ckpt_dir, result_file_name), 'w') as f:
        f.write(summary_str)
        # f.write(repr(episode_returns))
        f.write('\n\n')
        f.write(repr(highest_rewards)) # 输出所有验证的最好奖励分数

    return success_rate, avg_return

# from utils import get_gpu_mem_info

# def print_gpu_mem():
#     gpu_mem_total, gpu_mem_used, gpu_mem_free = get_gpu_mem_info()
#     return (gpu_mem_used/gpu_mem_total)*100

def forward_pass(data, policy, use_gpos=True, use_language=False):
    
    # print("########\n前向传播：", print_gpu_mem())
    if use_language:  # use_language
        image_data, qpos_data, gpos_data, history_images_data, history_action_data, is_pad_history, action_data, is_pad_action, command_embedding = data
        command_embedding = command_embedding.cuda()
    else: # len = 8
        image_data, qpos_data, gpos_data, history_images_data, history_action_data, is_pad_history, action_data, is_pad_action = data
        command_embedding = None
        
    image_data, qpos_data, gpos_data, = image_data.cuda(), qpos_data.cuda(), gpos_data.cuda()
    history_images_data, history_action_data, is_pad_history = history_images_data.cuda(),history_action_data.cuda(), is_pad_history.cuda()
    action_data, is_pad_action = action_data.cuda(), is_pad_action.cuda()
    
    if use_gpos:
        return policy(qpos_data, gpos_data, image_data, history_images_data, history_action_data, is_pad_history, action_data, is_pad_action, command_embedding) # TODO remove None # 提供了action data 不是训练模式
    else:
        gpos_data = None
        return policy(qpos_data, gpos_data, image_data, history_images_data, history_action_data, is_pad_history, action_data, is_pad_action, command_embedding) # TODO remove None # 提供了action data 不是训练模式


def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']
    
    use_gpos = config['policy_config']['use_gpos']
    use_language = config["use_language"]
    set_seed(seed)
    
# 1. make policy
    policy = make_policy(policy_class, policy_config)
    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)
    
    # print("创建模型：", print_gpu_mem())
    
    # 加载已有的权重
    start_epoch = 0
    min_val_loss = np.inf
    train_history = []
    validation_history = []
    for last_history_epoch in range(num_epochs-2,-1,-1):
        ckpt_path = os.path.join(ckpt_dir, f'policy_epoch{last_history_epoch + 1}_seed{seed}.ckpt')
         
        if os.path.exists(ckpt_path): # Load the history trained weights of epoch
            print(f'Load the history trained weights of epoch={last_history_epoch+1}')
            checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
            policy.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
            optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
            start_epoch = checkpoint['epoch']  # 设置开始的epoch
            min_val_loss = checkpoint['min_val_loss']
            train_history = checkpoint['train_history']
            validation_history = checkpoint['validation_history']
            start_epoch = start_epoch + 1
            save_1000 = 0
            break 
    
    best_ckpt_info = None # 准备返回的是数据
    # 2. do train epoch
    epoch = start_epoch
    for epoch in tqdm(range(start_epoch, num_epochs)): # for 循环训练 epoch
        
        # 2.1 validation and summary the last epoch：验证出 best policy
        with torch.inference_mode():
            policy.eval() # 将 policy 配置为 eval 模式
            # print("########\n模型验证 ：", print_gpu_mem())
            epoch_dicts = []
            
            # 将验证集的数据都跑一下
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy, use_gpos=use_gpos, use_language=use_language) # 前向传播！！
                # 为什么验证的时候要做前向传播呢？？  ===》在policy在eval模式下，权重不会做更新，在eval的时候做前向传播是为了计算loss
                epoch_dicts.append(forward_dict)
                
            epoch_summary = compute_dict_mean(epoch_dicts) # 计算 epoch 的 eval 平均数
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary['loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss # 更新 最低的 loss of epochs
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
    
        # print(f'Val loss:   {epoch_val_loss:.5f}')
        # summary_string = ''
        # for k, v in epoch_summary.items():
        #     summary_string += f'{k}: {v.item():.3f} '
        # print(summary_string)
        
        # 2.2. training epoch 训练只出 last policy
        policy.train() # 将policy配置为 train 模式（可以更新其中的参数）
        # print("########\n模型训练 ：", print_gpu_mem())
        optimizer.zero_grad() # 重置优化器梯度参数
        # print("加载优化器 ：", print_gpu_mem())
        
        for batch_idx, data in enumerate(train_dataloader): # 迭代循环训练集
            forward_dict = forward_pass(data, policy, use_gpos=use_gpos, use_language=use_language) # 前向传播！！
            # backward
            loss = forward_dict['loss'] # 没有用训练的loss，而是用eval的loss做输出
            loss.backward() # 损失反向传播
            
            # 优化器前向传播
            optimizer.step() # 主要核心是靠这个训练的？
            optimizer.zero_grad() # 重置优化器梯度参数
            
            train_history.append(detach_dict(forward_dict)) #记录训练历史
    
    # save the policy weight for continue train
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch{epoch + 1}_seed{seed}.ckpt')
    checkpoint = {
        "net": policy.state_dict(),
        'optimizer':optimizer.state_dict(),
        # "z_info":z_info,
        "epoch": epoch,
        "min_val_loss": min_val_loss,
        "train_history": train_history,
        "validation_history": validation_history
    }
    torch.save(checkpoint, ckpt_path)
    
    # save best checkpoint
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info  # 如果这里报错，说明就是没有更好的了
    pth_path = os.path.join(ckpt_dir, f'policy_best_epoch{epoch + 1}.pth') # 用来推理用的ckpt
    torch.save(best_state_dict, pth_path) 
    
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')
    # save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)
    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'trainval_{key}_epoch{num_epochs}_seed{seed}.png')
        
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
        
    print(f'Saved plots to {ckpt_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true') # 是训练还是验证评估
    parser.add_argument('--num_verification',  default=50, type=int, help='number of verification') # 验证次数
    parser.add_argument('--onscreen_render', action='store_true') # 是否在屏幕上实时渲染？（只在eval时才有用）
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True) # 权重文件保存地址
    
    # 模型策略类型，本文提出的ACT，用来对比的是CNNMLP
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True) 
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    
    # 训练参数
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True) # 每一批次大小
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True) # 随机种子
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True) # 训练多少个epochs
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True) # learning rate 学习率是多大

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False) # 
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False) # 批量众多预测的步数【经过实验验证最优的是100】
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False) # 隐藏层层数
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False) # 前馈层层数
    parser.add_argument('--temporal_agg', action='store_true')
    parser.add_argument('--backbone', default='resnet18', type=str, help="Name of the convolutional backbone to use")
    
    # for LLMs
    parser.add_argument('--command', action='store', type=str, help='comma-separated list of commands', default='', required=False)
    parser.add_argument('--use_language', action='store_true')
    parser.add_argument('--language_encoder', action='store', type=str, choices=['distilbert', 'clip'], default='distilbert', help='Type of language encoder to use: distilbert or clip', required=False)
    
    # variation
    parser.add_argument('--variation', action='store', type=int, default=0, help='the variations of the task', required=False)
    
    main(vars(parser.parse_args()))
