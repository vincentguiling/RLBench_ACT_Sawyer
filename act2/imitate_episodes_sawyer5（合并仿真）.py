#######################################
# # 加入FiLM的多步骤组合
# python act2/imitate_episodes_sawyer5.py \
#     --task_name sorting_program5 \
#     --ckpt_dir Trainings --policy_class ACT \
#     --kl_weight 10 --chunk_size 10 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
#     --num_epochs 2000 --backbone efficientnet_b3film --lr 1e-5 --seed 0  \
#     --use_language --language_encoder distilbert \
#     --eval --temporal_agg --variation -1 
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
    
    np.set_printoptions(linewidth=200)
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
    # print(f"{task_config['num_episodes']=}, {task_config['num_variation']=}, {num_episodes=}, ")
    
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']
    task_steps = task_config['task_steps'] if task_config['task_steps'] is not None else None
    steps_backbones = task_config['steps_backbones'] if task_config['steps_backbones'] is not None else None
    
    ####################################################################################
    commands = args["command"].split(",") if args["command"] else [] # task_config['commands'] # 
    use_language = args["use_language"]
    language_encoder = args["language_encoder"]
    ####################################################################################
    
    # max_skill_len = (args["max_skill_len"] if args["max_skill_len"] is not None else episode_len)
    max_skill_len = episode_len
    
    # fixed parameters
    state_dim = 15 # 左右机械臂，一共7*2 = 14,7+1
    lr_backbone = 1e-5
    backbone = args['backbone']

    # 增加参数保存
    chunk_size = args['chunk_size']
    batch_size = args['batch_size']
    if task_steps == None:
        ckpt_dir = args['ckpt_dir'] + f'/{task_name}/{num_episodes}demo_{episode_len}step_{chunk_size}chunk_{batch_size}batch_{backbone}'
    else:
        ckpt_dir = args['ckpt_dir'] + f'/{task_name}'
        backbone = steps_backbones
        
    # policy_class == 'ACT'
    enc_layers = 4
    dec_layers = 7
    nheads = 8 # 8头注意力机制
    policy_config = {'lr': args['lr'],
                    'num_queries': args['chunk_size'],
                    'kl_weight': args['kl_weight'],
                    'hidden_dim': args['hidden_dim'],
                    'dim_feedforward': args['dim_feedforward'],
                    'lr_backbone': lr_backbone,
                    'backbone': backbone,
                    'enc_layers': enc_layers,
                    'dec_layers': dec_layers,
                    'nheads': nheads,
                    'camera_names': camera_names,
                    }
    
    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'episode_len': episode_len,
        'state_dim': state_dim,
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
        'stats_dir': '<steps>_dataset_stats.pkl'
        }

    if is_eval: # 如果是验证的话
        results = []
        from sim_env import make_sim_env
        env = make_sim_env(task_name)
        env_max_reward = 1
        num_rollouts = num_verification
        set_seed(100)
        episode_returns = []
        highest_rewards = []
        command_embedding = []
        
        use_language = config["use_language"]
        language_encoder = config["language_encoder"]
        max_skill_len = config["max_skill_len"]
        
        if use_language:
            tokenizer, model = initialize_model_and_tokenizer(language_encoder)
            assert tokenizer is not None and model is not None

        for rollout_id in range(num_rollouts):
            # 验证循环初始化
            max_timesteps = episode_len[0] + episode_len[1]
            qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
            target_qpos_list = []
            image_list = [] # for visualization
            qpos_list = []
            gripper_state = 1
            if variation >= 0:
                env.set_variation(variation) 
            else:
                random_variation = np.random.randint(3)
                env.set_variation(random_variation) 
            descriptions, ts_obs = env.reset() # 重置帧数
            print(f"{descriptions[0]}, and {descriptions[1]}")
            
            # 命令编码
            if use_language: # and (t % max_skill_len == 0) :
                # Check if an intervention is needed; if so, language correction
                command_embedding.append(generate_command_embedding(descriptions[0], language_encoder, tokenizer, model))
                command_embedding.append(generate_command_embedding(descriptions[1], language_encoder, tokenizer, model))
            # 调用任务步骤完成
            for task_step_index in range(len(task_steps)):
                
                ckpt_name = f'{task_steps[task_step_index]}_policy_best.pth'  # 直接把地址发给他
                
                config['policy_config']['backbone'] = backbone[task_step_index]
                config['camera_names'] = camera_names[task_step_index]
                config['episode_len'] = episode_len[task_step_index] * 1.0
                config['policy_config']['camera_names'] = camera_names[task_step_index]
                config['stats_dir'] = f'{task_steps[task_step_index]}_dataset_stats.pkl'
                
                rewards, gripper_state, ts_obs = eval_bc(config, ckpt_name, env, ts_obs, qpos_history, target_qpos_list, image_list, qpos_list, gripper_state, command_embedding[task_step_index]) # 调用 eval_bc() 直接验证
            
            # for i in range(t+1): # clear the path history
            #     path[i].clear_visualization()
            rewards = np.array(rewards) # 
            episode_return = np.sum(rewards[rewards!=None])
            episode_returns.append(episode_return)
            episode_highest_reward = np.max(rewards)
            highest_rewards.append(episode_highest_reward)
            # print(f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}')
            print(f'{rollout_id} Rollout with {len(target_qpos_list)} steps : {episode_highest_reward==env_max_reward}')
            save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f'video_{rollout_id}.mp4'))

        success_rate = np.mean(np.array(highest_rewards) == env_max_reward) # 计算占比
        avg_return = np.mean(episode_returns) # 计算平均数
        summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
        for r in range(env_max_reward+1):
            more_or_equal_r = (np.array(highest_rewards) >= r).sum()
            more_or_equal_r_rate = more_or_equal_r / num_rollouts
            summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n'

        print(summary_str)
        
        results.append([ckpt_name, success_rate, avg_return])
        # save success rate to txt
        result_file_name = 'result' +  f'({more_or_equal_r_rate*100}%).txt'
        with open(os.path.join(ckpt_dir, result_file_name), 'w') as f:
            f.write(summary_str)
            # f.write(repr(episode_returns))
            f.write('\n\n')
            f.write(repr(highest_rewards)) # 输出所有验证的最好奖励分数
        for ckpt_name, success_rate, avg_return in results:
            print(f'{ckpt_name}: {success_rate=} {avg_return=}')
        print()
        exit() # eval 结束后退出程序
    
    # 如果不是evaluation
    train_dataloader, val_dataloader, stats, _ = load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val, 
                                                           max_len=max_skill_len, command_list=commands, use_language=use_language, language_encoder=language_encoder)
    # save dataset stats
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
        
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl') # pkl 是 pickle 打包的文件
    with open(stats_path, 'wb') as f: 
        pickle.dump(stats, f)
        
    train_bc(train_dataloader, val_dataloader, config) # 调用 train_bc() 训练，保存最新的为 best_ckpt_info 文件
    

def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy

def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
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

def generate_command_embedding(command, language_encoder, tokenizer, model, instructor=None):

    command_embedding = encode_text(command, language_encoder, tokenizer, model)
    command_embedding = torch.tensor(command_embedding).cuda()
    if instructor is not None:
        command_embedding = instructor.get_nearest_embedding(command_embedding)[0]
    return command_embedding
    
    
def eval_bc(config, ckpt_name, env, ts_obs, qpos_history, target_qpos_list, image_list, qpos_list, gripper_flag, command_embedding):
    ckpt_dir = config['ckpt_dir']
    policy_class = config['policy_class']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = int(config['episode_len'])
    temporal_agg = config['temporal_agg']
    stats_dir = config['stats_dir']
    
    gripper_flag = 1
    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    policy.load_state_dict(torch.load(ckpt_path))
    
    policy.cuda()
    policy.eval() # 将policy配置为eval模式
    
    # print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, stats_dir)
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    
    pre_process_qpos = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']
    
    query_frequency = policy_config['num_queries']
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']

    ### evaluation loop
    if temporal_agg: # 是否使用GPU提前读取数据？？应该可以提高 eval 速度
        all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, 8]).cuda() ## 输出8维，但是输入时15维度
        
    rewards = []
    with torch.inference_mode():
        path = []
        t = 0
        for timestep in range(max_timesteps): # 最大帧数
            obs = ts_obs
            image_list.append({'front':obs.front_rgb, 'head':obs.head_rgb, 'wrist':obs.wrist_rgb})
            
            qpos_numpy = np.array(np.append(obs.joint_positions, obs.gripper_open)) # 7 + 1 + 7 = 15
            qpos_numpy = np.array(np.append(qpos_numpy, obs.gripper_pose)) # 7 + 1 + 7 = 15
            qpos = pre_process_qpos(qpos_numpy)
            qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
            qpos_history[:, t] = qpos
            
            curr_image = get_image(obs, camera_names) # 获取帧数据的图像

            ### query policy
            if config['policy_class'] == "ACT":
                if t % query_frequency == 0:
                    
                    all_actions = policy(qpos, curr_image, command_embedding=command_embedding) # 100帧才预测一次
                    
                if temporal_agg: # 做了一个 Action Chunking
                    all_time_actions[[t], t:t+num_queries] = all_actions
                    actions_for_curr_step = all_time_actions[:, t]
                    
                    # 在生成的多个序列中不是简单的平均，又做了一个运算（时间集成？？）
                    actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                    k = 0.25
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum() # 做了一个归一化
                    exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1) # 压缩维度
                    
                    raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    
                else: # 如果没用的话，等于就是 100帧才预测一次，然后挨着执行
                    raw_action = all_actions[:, t % query_frequency]
                    
            elif config['policy_class'] == "CNNMLP":
                raw_action = policy(qpos, curr_image) 
            else:
                raise NotImplementedError
            
            ### post-process actions
            raw_action = raw_action.squeeze(0).cpu().numpy()
            action = post_process(raw_action)  # 就是因为这个的保护和限制，所以初始化位置不能随意改变
            target_qpos = action
            
            # 将gpos作为action
            try:
                next_gripper_position = action[0:3] # next 
                next_gripper_quaternion = action[3:7]
                path.append(env._robot.arm.get_linear_path(position=next_gripper_position, quaternion=next_gripper_quaternion, steps=10, relative_to=env._robot.arm, ignore_collisions=True))
                gripper_state = action[7]
                # print( f"\r {gripper_flag}", end='')
                
                # print(gripper_state)
                # 夹爪控制###############################################################################################
                if gripper_state < 0.9 and gripper_flag == 1 : # 适合步骤1 夹取
                    # print("close_gripper")
                    gripper_flag = 0
                    env._robot.gripper.actuate(0, 0.8)
                        
                # elif gripper_state > 0.6 and gripper_flag == 0 :# 适合步骤2 放置
                elif gripper_state > 0.5 and gripper_flag == 0 :# 适合步骤1 夹取
                    # print("open_gripper")
                    gripper_flag = 1
                    env._robot.gripper.actuate(1.0,0.04)
                
                # print("steps: ", t, end=' ')
                path[t].visualize() # 在仿真环境中画出轨迹
                
                done = False # 当done 置为 True 的时候，说明预测的轨迹执行完毕了
                while done != 1: # 如果 done 是 False 则执行
                    done = path[t].step() # ArmConfigurationPath类型的step运行载入下一帧动作
                    env._scene.step() # Scene 步进
                    
                ts_obs = env._scene.get_observation()
                reward, _ = env._task.success() # 任务是否完成状态读取
                qpos_list.append(qpos_numpy)
                target_qpos_list.append(target_qpos)
                rewards.append(reward) # 由仿真环境 step 产生 reward：0，1，2，3，4，4代表全部成功
                
                if gripper_flag == 0 and reward: # 目标是松爪，开始开爪子（对于复杂的任务来说，有点简单了，还是用目标性图像来表征吧）
                    break
            except ConfigurationPathError: 
                print("ConfigurationPathError ", t) # , "path lens: ",len(path))
                break # 跳出推理循环

                # 不跳出，而是前面的2/3步数。在相同的观测下面有相同的推理？？？，不是，是反复迭代退回了
                # np.random.seed(0)
                t_back = (t*8)//10
                
                back_gripper_pose = [elem * (1 + (np.random.randint(100) - 50)/50) for elem in target_qpos_list[t_back][:7]]
                for i in range(t - t_back):
                    path.pop()
                    qpos_list.pop()
                    target_qpos_list.pop()
                t = t_back
                    
                # back_gripper_pose = target_qpos_list[(t*9)//10][:7]
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
                    target_qpos_list.append(target_qpos)    
                    
                except ConfigurationPathError:
                    print("ConfigurationPathError ConfigurationPathError")
                    break # 跳出推理循环
            t = t + 1

    return rewards, gripper_state, ts_obs


def forward_pass(data, policy):
    if len(data) == 5:  # use_language
        image_data, qpos_data, action_data, is_pad, command_embedding = data
        command_embedding = command_embedding.cuda()
    else:
        image_data, qpos_data, action_data, is_pad = data
        command_embedding = None
        
    image_data, qpos_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
    return policy(qpos_data, image_data, action_data, is_pad, command_embedding) # TODO remove None # 提供了action data 不是训练模式


def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']

    set_seed(seed)
    
# 1. make policy
    policy = make_policy(policy_class, policy_config)
    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)
    
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
            epoch_dicts = []
            
            # 将验证集的数据都跑一下
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy) # 前向传播！！
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
        optimizer.zero_grad() # 重置优化器梯度参数
        
        for batch_idx, data in enumerate(train_dataloader): # 迭代循环训练集
            forward_dict = forward_pass(data, policy) # 前向传播！！
            
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
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info 
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
