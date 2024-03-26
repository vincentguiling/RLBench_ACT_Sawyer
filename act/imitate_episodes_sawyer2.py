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
from utils import sample_box_pose, sample_insertion_pose # robot functions
from utils import compute_dict_mean, set_seed, detach_dict # helper functions
from policy import ACTPolicy, CNNMLPPolicy
from visualize_episodes import save_videos

from sim_env import BOX_POSE
from pyrep.errors import ConfigurationPathError

import IPython
e = IPython.embed

def main(args):
    set_seed(1)
    # command line parameters
    is_eval = args['eval']
    policy_class = args['policy_class']
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']
    batch_size_train = args['batch_size'] # train 和 eval 使用相同的 batch_size
    batch_size_val = args['batch_size']
    num_epochs = args['num_epochs']

    is_sim = True 
    if is_sim:
        from constants import SIM_TASK_CONFIGS
        task_config = SIM_TASK_CONFIGS[task_name]

    dataset_dir = task_config['dataset_dir']
    num_episodes = task_config['num_episodes']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']

    # fixed parameters
    state_dim = 8 # 左右机械臂，一共7*2 = 14,7+1
    lr_backbone = 1e-5
    backbone = 'resnet18' # 图像基础处理网络是ResNet18
    if policy_class == 'ACT':
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
    elif policy_class == 'CNNMLP':
        policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': 1,
                         'camera_names': camera_names,}
    else:
        raise NotImplementedError
    
    # 增加参数保存
    chunk_size = args['chunk_size']
    batch_size = args['batch_size']
    ckpt_dir = args['ckpt_dir'] + f'_{num_episodes}demo_{episode_len}step_{chunk_size}chunk_{num_epochs}epoch_{batch_size}batch'
    
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
        'real_robot': not is_sim
    }

    if is_eval: # 如果是验证的话
        ckpt_names = [f'policy_best.ckpt']
        results = []
        for ckpt_name in ckpt_names:
            success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=True) # 调用 eval_bc() 直接验证
            results.append([ckpt_name, success_rate, avg_return])

        for ckpt_name, success_rate, avg_return in results:
            print(f'{ckpt_name}: {success_rate=} {avg_return=}')
        print()
        exit() # eval 结束后退出程序
    
    # 如果不是evaluation
    train_dataloader, val_dataloader, stats, _ = load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val)

    # save dataset stats
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
        
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl') # pkl 是 pickle 打包的文件
    with open(stats_path, 'wb') as f: # 以读写方式是打开
        pickle.dump(stats, f)

    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config) # 调用 train_bc() 训练，保存最新的为 best_ckpt_info 文件
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info # min_val_loss 和 best_epoch 保存在

    # save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path) 
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')


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

def get_image(ts, camera_names):
    curr_images = []
    # curr_image = rearrange(ts.head_rgb, 'h w c -> c h w')
    # curr_images.append(curr_image)
    curr_image = rearrange(ts.wrist_rgb, 'h w c -> c h w')
    curr_images.append(curr_image)    
        
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image

def eval_bc(config, ckpt_name, save_episode=True):
    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    real_robot = config['real_robot']
    policy_class = config['policy_class']
    onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    temporal_agg = config['temporal_agg']
    onscreen_cam = 'angle'

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval() # 将policy配置为eval模式
    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    # load environment
    if not real_robot:
        from sim_env import make_sim_env
        env = make_sim_env(task_name)
        env_max_reward = 1 # env.task.max_reward
    # chunk_size = num_queries
    query_frequency = policy_config['num_queries']
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']
        
    ##########################################################################################################
    max_timesteps = int(max_timesteps * 1.3) # may increase for real-world tasks
    ##########################################################################################################
    
    num_rollouts = 50 # 验证 50 次
    episode_returns = []
    highest_rewards = []
    
    # 验证次数
    for rollout_id in range(num_rollouts):
        rollout_id += 0
        ### set task
        if 'sim_transfer_cube' in task_name:
            BOX_POSE[0] = sample_box_pose() # used in sim reset # 在一定范围内随机生成采样一个 cube 的位置
        elif 'sim_insertion' in task_name:
            BOX_POSE[0] = np.concatenate(sample_insertion_pose()) # used in sim reset
        
        # 重置帧数
        _, ts_obs = env.reset()

        ### onscreen render
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id=onscreen_cam))
            plt.ion()

        ### evaluation loop
        if temporal_agg: # 是否使用GPU提前读取数据？？应该可以提高 eval 速度
            all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim]).cuda()

        qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        image_list = [] # for visualization
        qpos_list = []
        target_qpos_list = []
        rewards = []
        with torch.inference_mode():
            path = []
            for t in range(max_timesteps): # 最大帧数
                ### update onscreen render and wait for DT
                if onscreen_render:
                    image = env._physics.render(height=480, width=640, camera_id=onscreen_cam)
                    plt_img.set_data(image)
                    plt.pause(DT)

                obs = ts_obs
                image_list.append({'wrist':obs.wrist_rgb})
                # image_list.append({'front':obs.front_rgb, 'head':obs.head_rgb, 'wrist':obs.wrist_rgb})
                    
                qpos_numpy = np.array(np.append(obs.joint_positions, obs.gripper_open)) # 7 + 1 = 8
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                qpos_history[:, t] = qpos
                curr_image = get_image(ts_obs, camera_names) # 获取帧数据的图像

                ### query policy
                if config['policy_class'] == "ACT":
                    if t % query_frequency == 0:
                        all_actions = policy(qpos, curr_image) # 100帧才预测一次，# 没有提供 action 数据，是验证模式
                        
                        # 核心重点！！！
                    if temporal_agg: # 做了一个 Action Chunking
                        all_time_actions[[t], t:t+num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        
                        # 在生成的多个序列中不是简单的平均，又做了一个运算（时间集成？？）
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        ############################################################################################################################################
                        k = 0.25
                        ############################################################################################################################################
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
                action = post_process(raw_action) # 又对预测出来的动作做了一不处理
                target_qpos = action
                
                next_gripper_position = action[0:3]
                next_gripper_quaternion = action[3:7]
                ### step the environment
                
                # ts_obs, reward, _ = env.step(target_qpos) # 原关节轨迹
                try:
                    path.append(env._robot.arm.get_linear_path(position=next_gripper_position, quaternion=next_gripper_quaternion, steps=2, relative_to=env._robot.arm))
                    path[t].visualize() # 在仿真环境中画出轨迹
                
                    done = False # 当done 置为 True 的时候，说明预测的轨迹执行完毕了
                    while done != 1: # 如果 done 是 False 则执行
                        done = path[t].step() # ArmConfigurationPath类型的step运行载入下一帧动作
                        env._scene.step() # Scene 步进
                        
                    ts_obs = env._scene.get_observation()
                    reward, _ = env._task.success() # 任务是否完成状态读取

                    ### for visualization
                    qpos_list.append(qpos_numpy)
                    target_qpos_list.append(target_qpos)
                    rewards.append(reward) # 由仿真环境 step 产生 reward：0，1，2，3，4，4代表全部成功
                    if reward >= 1 :
                        print("reward >= 1")
                        break
                except ConfigurationPathError:
                    print("ConfigurationPathError")
                    break # 跳出推理循环
                
            plt.close()
            
        # for i in range(t+1): # clear the path history
        #     path[i].clear_visualization()
        print(t,"steps has inference")
        rewards = np.array(rewards) # 
        episode_return = np.sum(rewards[rewards!=None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        print(f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}')

        if save_episode:
            save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f'video{rollout_id}.mp4'))

    success_rate = np.mean(np.array(highest_rewards) == env_max_reward) # 计算占比
    avg_return = np.mean(episode_returns) # 计算平均数
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    for r in range(env_max_reward+1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n'

    print(summary_str)

    # save success rate to txt
    result_file_name = 'result_' + ckpt_name.split('.')[0] + '.txt'
    with open(os.path.join(ckpt_dir, result_file_name), 'w') as f:
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write('\n\n')
        f.write(repr(highest_rewards)) # 输出所有验证的最好奖励分数

    return success_rate, avg_return


def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    image_data, qpos_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
    return policy(qpos_data, image_data, action_data, is_pad) # TODO remove None # 提供了action data 不是训练模式


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

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None # 准备返回的是数据
    
# 2. do train epoch
    for epoch in tqdm(range(num_epochs)): # for 循环训练 epoch
        print(f'\nEpoch {epoch}')
        
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
    
        print(f'Val loss:   {epoch_val_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)
        
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
    
    # 2.3. summary the train
        epoch_summary = compute_dict_mean(train_history[(batch_idx+1)*epoch:(batch_idx+1)*(epoch+1)])
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

    # 2.4. save the weight file
        if epoch % 100 == 0: # 100个epoch保存一个权重文件
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)
    
    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    # save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        
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
    
    main(vars(parser.parse_args()))