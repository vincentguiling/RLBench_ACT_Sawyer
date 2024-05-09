import rospy
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge
from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,)

import cv2 as cv
import torch # 如果pytorch安装成功即可导入
import numpy as np
import os
from policy import ACTPolicy
import pickle
from einops import rearrange

# sawyer
import intera_interface
import intera_external_devices

from intera_interface import CHECK_VERSION
from intera_core_msgs.msg import (
    IODeviceStatus,
    EndpointState,)

########################### 修改的参数 ########################### 
task_name = 'sorting_program_sawyer21'
ckpt_dir = '/home/boxjod/RLBench_ACT_Sawyer/Trainings/sorting_program_sawyer21/50demo_36step_10chunk_8batch_efficientnet_b3'
ckpt_name = 'policy_best_epoch4000.pth'

# 上方
# 1，22，3

# 中方
# 44，5，6

# 下方
# 77，88，99

# task_name = 'sorting_program_sawyer22'
# ckpt_dir = '/home/boxjod/RLBench_ACT_Sawyer/Trainings/sorting_program_sawyer22/50demo_77step_10chunk_8batch_efficientnet_b0'
# ckpt_name = 'policy_best_epoch3000.pth'


########################### 修改的参数 ###########################

def observation_to_action(policy, max_timesteps, ckpt_dir):
  print("start get the robot observation and do the model... ")
  bridge = CvBridge()
  
  gpos = []
  qpos = []
  gripper_state = 1
  stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
  with open(stats_path, 'rb') as f:
    stats = pickle.load(f)
  pre_process_qpos = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
  post_process = lambda a: a * stats['action_std'] + stats['action_mean']
  
  query_frequency = 1
  num_queries = 10 # chunking_size
  if task_name == 'sorting_program_sawyer21':
    max_timesteps = int(max_timesteps * 2.8)  # 做一个scale ##############################################################
  elif task_name == 'sorting_program_sawyer22':
    max_timesteps = int(max_timesteps * 3.5)  # 做一个scale ##############################################################
  all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, 8]).cuda() ## 输出8维，但是输入时15维度
  image_list = [] # for visualization
  
  tray_defaul_waypoint = {'right_j0': 0.0568505859375, 'right_j1': -0.1841328125, 'right_j2': -0.0026962890625, 'right_j3': 0.8043798828125, 'right_j4': -0.019640625, 'right_j5': 0.950361328125, 'right_j6': -1.4002099609375}
  
  # intera structure
  side = 'right'
  limb = intera_interface.Limb(side)
  head = intera_interface.Head()
  gripper = intera_interface.Gripper(side + '_gripper')
  joints_name = limb.joint_names()
  
  timestep = 0
  image_get_count = 0
  clac_rate = 30 # hz #####################################################
  
  def go_to_next_gpos(target_gpos):
    nonlocal timestep, limb, all_time_actions, gripper_state, max_timesteps
    global task_name, ckpt_name, ckpt_dir
    
    hover_distance = -0.15 ##########可能有点问题0.01的问题
    tip_name = 'right_gripper_tip'
    
    approach = Pose()
    approach.position.x = target_gpos[0]
    approach.position.y = target_gpos[1]
    approach.position.z = target_gpos[2]
    approach.orientation.x = target_gpos[3]
    approach.orientation.y = target_gpos[4]
    approach.orientation.z = target_gpos[5]
    approach.orientation.w = target_gpos[6]
    approach.position.z = approach.position.z + hover_distance # 数据集制作和训练的时候都有用
    timestep = timestep + 1
    print(timestep,end=' ')

    joint_angles = limb.ik_request(approach, tip_name) # 逆向运动学
    
    limb.set_joint_position_speed(0.06) ##############################################################################
    done = limb.move_to_joint_positions(joint_angles, timeout = 1/clac_rate) # 运动到目标位置
    
    # 处理夹爪：
    print(f"{target_gpos[7]=}")
    
    if gripper_state == 1 and target_gpos[7] <= 0.9:
      print("close the gripper")
      gripper.close()
      
    elif timestep > 60 and gripper_state == 0 and target_gpos[7] >= 0.1 :
      print("open the gripper")
      gripper.open()
      
      
    if timestep >= max_timesteps:
      subscriber_control(0)
      if task_name == 'sorting_program_sawyer21':
        print("you can try agqin input '1' and '2' to next setp")
      elif   task_name == 'sorting_program_sawyer22':
        print("you can try agqin input '1'")
        
      done = False
      while not done and not rospy.is_shutdown():
        c = intera_external_devices.getch()
        if c in ['\x1b', '\x03']:
          done = True
          rospy.signal_shutdown("Example finished.")
        elif c == '1':
          if task_name == 'sorting_program_sawyer21':
            go_to_initial_position()
            all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, 8]).cuda() ## 输出8维，但是输入时15维度
            print("retest the step 1")
            timestep = 0
            subscriber_control(1)
            done = True
          elif task_name == 'sorting_program_sawyer22':
            print("restart the task")
            task_name = 'sorting_program_sawyer21'
            ckpt_dir = '/home/boxjod/RLBench_ACT_Sawyer/Trainings/sorting_program_sawyer21/50demo_36step_10chunk_8batch_efficientnet_b3'
            ckpt_name = 'policy_best_epoch3000.pth'
            subscriber_control(0)
          
            policy, max_timesteps = buil_model(ckpt_dir, ckpt_name)
            observation_to_action(policy, max_timesteps, ckpt_dir)
          
        elif task_name == 'sorting_program_sawyer21' and c =='2':
          print("continue to the  the step 2")
          task_name = 'sorting_program_sawyer22'
          ckpt_dir = '/home/boxjod/RLBench_ACT_Sawyer/Trainings/sorting_program_sawyer22/50demo_77step_10chunk_8batch_efficientnet_b0'
          ckpt_name = 'policy_best_epoch3000.pth'
          subscriber_control(0)
          
          policy, max_timesteps = buil_model(ckpt_dir, ckpt_name)
          observation_to_action(policy, max_timesteps, ckpt_dir)
          
      
  def policy_model_calc(policy, qpos_current, curr_image):
    with torch.inference_mode(): # 模型推理
      image_list.append({'wrist':curr_image})

      qpos_numpy = np.array(qpos_current) # 7 + 1 + 7 = 15
      qpos = pre_process_qpos(qpos_numpy)
      qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)

      ### query policy
      if timestep % query_frequency == 0:
        command_embedding = None
        all_actions = policy(qpos, curr_image, command_embedding=command_embedding) # 100帧才预测一次，# 没有提供 action 数据，是验证模式

      all_time_actions[[timestep], timestep: timestep+num_queries] = all_actions
      actions_for_curr_step = all_time_actions[:, timestep]
      
      actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
      actions_for_curr_step = actions_for_curr_step[actions_populated]
      k = 0.25
      exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
      exp_weights = exp_weights / exp_weights.sum() # 做了一个归一化
      exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1) # 压缩维度
      raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
          
      ### post-process actions
      raw_action = raw_action.squeeze(0).cpu().numpy()
      action = post_process(raw_action)  # 就是因为这个的保护和限制，所以初始化位置不能随意改变
      go_to_next_gpos(action)     
      # print(f"{action}") 

  def get_observations(data): # 30Hz
    nonlocal image_get_count, image_list, qpos, gripper_state, gpos, timestep, timestep
    
    image_get_count = image_get_count + 1
    if (image_get_count >= 30 / clac_rate) and (timestep < max_timesteps): # 相机是30Hz的 
      image_get_count = 0
      
      image = bridge.imgmsg_to_cv2(data,  desired_encoding='rgb8')
      wrist_image_current = image
      # if task_name == 'sorting_program_sawyer21':
      wrist_image_current = cv.resize(image, (0, 0), fx=0.25, fy=0.25, interpolation=cv.INTER_LINEAR) # wrist_rgb
      
      # 用小的训练，然后用高清的推理
      
      curr_images = []
      curr_image = rearrange(wrist_image_current, 'h w c -> c h w')
      curr_images.append(curr_image)    
      curr_image = np.stack(curr_images, axis=0)
      curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
      qpos_current = qpos + [gripper_state] + gpos
      
      policy_model_calc(policy, qpos_current, curr_image)


  def get_gpos(data): # gpos # 100Hz
    nonlocal gpos
    position = data.pose.position
    orientation = data.pose.orientation
    gpos = [position.x, position.y, position.z, orientation.x, orientation.y, orientation.z, orientation.w] 
    
  def get_qpos(data): # gpos  # 100Hz
    nonlocal qpos, timestep
    if len(data.position) == 9:
      qpos = [data.position[1], data.position[2], data.position[3], data.position[4], data.position[5], data.position[6], data.position[7]]
      effort = sum(data.effort)
      # print(f"{effort=}")
      if effort > -20:
        
        if gripper_state == 0:
          gripper.open()
        else:
          print("检测到碰撞") # 能不能立马停掉啊
          subscriber_control(0)
        timestep = max_timesteps
        
        
  def get_gripper_state(data): # gpos  # 100Hz
    nonlocal gripper_state
    gripper_state = float(data.signals[10].data[1:-1]) # 10表示的是第11个信号是 position_response_m
    if gripper_state >= 0.03:
      gripper_state = 1
    else:
      gripper_state = 0
  
  def subscriber_control(ctrl):
    nonlocal wrist_cam_Subscriber, endpoint_state_Subscriber, joint_states_Subscriber, right_gripper_Subscriber
    if ctrl == 1:
      wrist_cam_Subscriber = rospy.Subscriber("/camera/color/image_raw", Image, get_observations) # wrist rgb
      endpoint_state_Subscriber = rospy.Subscriber("/robot/limb/right/endpoint_state", EndpointState, get_gpos) # wrist rgb
      joint_states_Subscriber = rospy.Subscriber("/robot/joint_states", JointState, get_qpos) # wrist rgb
      right_gripper_Subscriber = rospy.Subscriber("/io/end_effector/right_gripper/state", IODeviceStatus, get_gripper_state) # wrist rgb
    else:
      wrist_cam_Subscriber.unregister()
      endpoint_state_Subscriber.unregister()
      joint_states_Subscriber.unregister()
      right_gripper_Subscriber.unregister()
      
      
  def go_to_initial_position():
    print("going to the initial position...")
    
    gripper.open()
    init_positions=[0, -1.046, 0, 1.046, 0, 1.57, -1.48]
    start_angles = dict(zip(joints_name, init_positions))
    
    limb.set_joint_position_speed(0.2)
    limb.move_to_joint_positions(start_angles,timeout=6.0)
    limb.set_joint_position_speed(0.1)
    head.set_pan(0.0)
  
  def go_to_waypoint0_position():
      nonlocal all_time_actions, timestep, tray_defaul_waypoint
      go_to_initial_position() # 先回去一下，避免碰撞
      
      limb.move_to_joint_positions(tray_defaul_waypoint,timeout=8.0)# 去托盘左上角
      
      print("you can use ' joint_position_keyboard.py ' random the waypoint0, \nand put the block in the gripper \nif ok then input '1'")
      done = False
      while not done and not rospy.is_shutdown(): # 然后随机移动，即可开始推理
        c = intera_external_devices.getch()
        if c in ['\x1b', '\x03']:
          done = True
          rospy.signal_shutdown("Example finished.")
        elif c == '1':
          tray_defaul_waypoint = limb.joint_angles()
          return
      
  #  初始化
  if task_name == 'sorting_program_sawyer21':
    go_to_initial_position()
  elif task_name == 'sorting_program_sawyer22':
    # go_to_waypoint0_position()
    x = 1
  
  wrist_cam_Subscriber = rospy.Subscriber("/camera/color/image_raw", Image, get_observations) # wrist rgb
  endpoint_state_Subscriber = rospy.Subscriber("/robot/limb/right/endpoint_state", EndpointState, get_gpos) # wrist rgb
  joint_states_Subscriber = rospy.Subscriber("/robot/joint_states", JointState, get_qpos) # wrist rgb
  right_gripper_Subscriber = rospy.Subscriber("/io/end_effector/right_gripper/state", IODeviceStatus, get_gripper_state) # wrist rgb
  
  rospy.spin()  


def buil_model(ckpt_dir, ckpt_name):
  print("buil the model structure")
  from constants import SIM_TASK_CONFIGS
  task_config = SIM_TASK_CONFIGS[task_name]

  max_timesteps = task_config['episode_len']
  camera_names = task_config['camera_names']
  
  enc_layers = 4
  dec_layers = 7
  nheads = 8 # 8头注意力机制
  policy_config = {'lr': 1e-5,
                    'num_queries': 10,
                    'kl_weight': 10,
                    'hidden_dim': 512,
                    'dim_feedforward': 3200,
                    'lr_backbone': 1e-5,
                    'backbone': 'efficientnet_b3' if task_name == 'sorting_program_sawyer21' else 'efficientnet_b0',
                    'enc_layers': enc_layers,
                    'dec_layers': dec_layers,
                    'nheads': nheads,
                    'camera_names': camera_names,
                    }
  
  ckpt_path = os.path.join(ckpt_dir, ckpt_name)
  policy = ACTPolicy(policy_config)
  loading_status = policy.load_state_dict(torch.load(ckpt_path))
  print(loading_status)
  policy.cuda()
  policy.eval() # 将policy配置为eval模式
  print(f'Loaded: {ckpt_path}')
  
  return policy, max_timesteps


def main():
  
  print("Initializing node... ")
  rospy.init_node('verification_sawyer_node')
       
  print("Getting robot state... ", end=' ')
  rs = intera_interface.RobotEnable(CHECK_VERSION)
  init_state = rs.state().enabled
  print(f"{init_state}")
  
  def clean_shutdown():
    print("\nExiting example.")
  rospy.on_shutdown(clean_shutdown)
  
  if init_state == False:
    rospy.loginfo("Enabling robot...")
    rs.enable()
  
  policy, max_timesteps = buil_model(ckpt_dir, ckpt_name)
  observation_to_action(policy, max_timesteps, ckpt_dir)
  
  print("Done.")
  
if __name__ == '__main__':
    main()
