# rlbench
from rlbench.observation_config import ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointPosition, JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend.utils import task_file_to_task_class

from rlbench.environment import Environment
from rlbench.backend.exceptions import *
import os, socket
BOX_POSE = [None] # to be changed from outside # 工作区间里面随机小球出现的boundary

def make_sim_env(task_name):
    """
    Environment for simulated robot bi-manual manipulation, with joint position control
    Action space:      [left_arm_qpos (6),             # absolute joint position
                        left_gripper_positions (1),    # normalized gripper position (0: close, 1: open)
                        right_arm_qpos (6),            # absolute joint position
                        right_gripper_positions (1),]  # normalized gripper position (0: close, 1: open)

    Observation space: {"qpos": Concat[ left_arm_qpos (6),         # absolute joint position
                                        left_gripper_position (1),  # normalized gripper position (0: close, 1: open)
                                        right_arm_qpos (6),         # absolute joint position
                                        right_gripper_qpos (1)]     # normalized gripper position (0: close, 1: open)
                        "qvel": Concat[ left_arm_qvel (6),         # absolute joint velocity (rad)
                                        left_gripper_velocity (1),  # normalized gripper velocity (pos: opening, neg: closing)
                                        right_arm_qvel (6),         # absolute joint velocity (rad)
                                        right_gripper_qvel (1)]     # normalized gripper velocity (pos: opening, neg: closing)
                        "images": {"main": (480x640x3)}        # h, w, c, dtype='uint8'
    """
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = "/home/boxjod/Gym/RLBench/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
    if socket.gethostname() != 'XJ':
        os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = "/home/ubuntu/data0/liangws/peract_root/CoppeliaSim"
    
    # 1. 设置仿真环境的配置文件
    img_size = [160, 120] # 图片格式大小
    
    obs_config = ObservationConfig()
    
    obs_config.set_all(False)
    obs_config.wrist_camera.set_all(True)
    obs_config.head_camera.set_all(True)
    obs_config.front_camera.set_all(True)
    obs_config.set_all_low_dim(True)
    
    obs_config.wrist_camera.image_size = img_size
    obs_config.head_camera.image_size = img_size
    obs_config.wrist_camera.depth_in_meters = False
    obs_config.front_camera.image_size = img_size
    
    headless_val = True # False 
    if socket.gethostname() != 'XJ':
        headless_val = True
        
    # 2. 启动环境
    rlbench_env = Environment(
        action_mode=MoveArmThenGripper(JointPosition(), Discrete()),
        obs_config=obs_config,
        headless=headless_val,
        robot_setup='sawyer')
    
    rlbench_env.launch()
    rlbench_env._pyrep.step_ui()

    # 3. 挂载任务
    task_class = task_file_to_task_class(task_name)
    task_env = rlbench_env.get_task(task_class) # Type[Task]) -> TaskEnvironment（include scene）
    var_target = task_env.variation_count()
    # descriptions, _ = task_env.reset()
    
    return task_env