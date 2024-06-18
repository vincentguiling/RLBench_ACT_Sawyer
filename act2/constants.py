import pathlib

### Task parameters 

DATA_DIR = 'Datasets'
SIM_TASK_CONFIGS = {
    'sorting_program2':{  # 一次性完成抓取和放置任务
        'dataset_dir': DATA_DIR + '/sorting_program2/variation0',
        'num_episodes': 50,
        'episode_len': 88,
        'camera_names': ['wrist'], # , 'wrist_depth', 'head'
    },
    'sorting_program21':{ # 抓取 用imitate_episodes_sawyer4
        'dataset_dir': DATA_DIR + '/sorting_program21/variation0',# 单个颜色
        # 'dataset_dir': DATA_DIR + '/sorting_program21/variation12',# 红色和蓝色，2个颜色
        # 'dataset_dir': DATA_DIR + '/sorting_program21/variation123',# 红色、蓝色和绿色3个颜色
        'episode_len': 32,
        'num_episodes': 50,
        'num_variation': 1, # 这个问题，他应该是0？
        'camera_names': ['wrist'],# , 'wrist_depth', 'head'
    },
    # 'sorting_program211':{ # 测试不同初始位置的额效果
    #     # 'dataset_dir': DATA_DIR + '/sorting_program21/variation0',# 单个颜色
    #     # 'dataset_dir': DATA_DIR + '/sorting_program211/variation12',# 红色和蓝色，2个颜色
    #     'dataset_dir': DATA_DIR + '/sorting_program211/variation123',# 红色、蓝色和绿色3个颜色
    #     'episode_len': 32,
    #     'num_episodes': 50,
    #     'num_variation': 3,
    #     'camera_names': ['wrist'],# , 'wrist_depth', 'head'
    # },
    'sorting_program212':{ # 木块有旋转
        'dataset_dir': DATA_DIR + '/sorting_program21/variation0',# 单个颜色
        'episode_len': 32,
        'num_episodes': 50,
        'num_variation': 1,
        'camera_names': ['wrist'],# , 'wrist_depth', 'head'
    },
    'sorting_program22':{ # 放置
        'dataset_dir': DATA_DIR + '/sorting_program22/variation0',
        # 'dataset_dir': DATA_DIR + '/sorting_program22/variation123', # 三个颜色，开头带旋转
        'num_episodes': 50,
        'episode_len': 63,
        'num_variation': 1,
        'camera_names': ['wrist'], # , 'wrist_depth'
    },
    # 'sorting_program4':{ # 用imitate_episodes_sawyer4 不同语言指令控制不同阶段
    #     'dataset_dir': DATA_DIR + '/sorting_program4/variation0',# 红色、蓝色和绿色3个颜色
    #     'episode_len': 49,
    #     'num_episodes': 50,
    #     'num_variation': 1,
    #     'camera_names': ['wrist'],
    # },
    'sorting_program3':{ # 用imitate_episodes_sawyer5 分段权重读取
        'dataset_dir': DATA_DIR + '/sorting_program3/variation0',
        'num_episodes': 50,
        'episode_len': [32,63],
        'num_variation': 1,
        # 'steps_backbones':['efficientnet_b5film', 'efficientnet_b3film'],
        'task_steps':94, # 大概是两个步骤相加再多加几步
        'camera_names': [['wrist'], ['wrist']],
        'task_steps':['sorting_program21', 'sorting_program22'],
    },
    # 'sorting_program5':{ # 用imitate_episodes_sawyer5 分段权重读取
    #     'dataset_dir': DATA_DIR + '/sorting_program5/variation0',
    #     'num_episodes': 50,
    #     'episode_len': [32,63],
    #     'num_variation': 3,
    #     'steps_backbones':['efficientnet_b5film', 'efficientnet_b3film'],
    #     'task_steps':94, # 大概是两个步骤相加再多加几步
    #     'camera_names': [['wrist'], ['wrist']],
    #     'task_steps':['sorting_program21', 'sorting_program22'],
    # },
    'sorting_program5':{ #  用imitate_episodes_sawyer4 完整任务学习
        'dataset_dir': DATA_DIR + '/sorting_program5/variation0',# 单个颜色
        'episode_len': 90,
        'num_episodes': 50,
        'num_variation': 1,
        'camera_names': ['wrist'],# , 'wrist_depth', 'head'
    },
    'sorting_program_sawyer21':{ # 抓取 用imitate_episodes_sawyer4 , 1红色
        'dataset_dir': '/home/boxjod/sawyer_ws/Datasets/sorting_program_sawyer21',# 单个颜色
        'episode_len': 36,
        'num_episodes': 50,
        'num_variation': 1,
        'camera_names': ['wrist'],# , 'wrist_depth', 'head'
    },
    'sorting_program_sawyer211':{ # 抓取 用imitate_episodes_sawyer4 ， 1红色
        'dataset_dir': '/home/boxjod/sawyer_ws/Datasets/sorting_program_sawyer211',# 单个颜色
        'episode_len': 36,
        'num_episodes': 50,
        'num_variation': 1,
    },
    'sorting_program_sawyer212':{ # 抓取 用imitate_episodes_sawyer4 ， 2蓝色
        'dataset_dir': '/home/boxjod/sawyer_ws/Datasets/sorting_program_sawyer212',# 单个颜色
        'episode_len': 36,
        'num_episodes': 50,
        'num_variation': 1,
        'camera_names': ['wrist'],# , 'wrist_depth', 'head'
    },
    'sorting_program_sawyer213':{ # 抓取 用imitate_episodes_sawyer4 ， 3绿色
        'dataset_dir': '/home/boxjod/sawyer_ws/Datasets/sorting_program_sawyer213',# 单个颜色
        'episode_len': 36,
        'num_episodes': 50,
        'num_variation': 1,
        'camera_names': ['wrist'],# , 'wrist_depth', 'head'
    },
    'sorting_program_sawyer22':{ # 抓取 用imitate_episodes_sawyer4
        'dataset_dir': '/home/boxjod/sawyer_ws/Datasets/sorting_program_sawyer22',# 单个颜色
        'episode_len': 77,
        'num_episodes': 50,
        'num_variation': 1,
        'camera_names': ['wrist'],# , 'wrist_depth', 'head'
    },
    'sorting_program_sawyer221':{ # 抓取 用imitate_episodes_sawyer4， 1红色-》红色
        'dataset_dir': '/home/boxjod/sawyer_ws/Datasets/sorting_program_sawyer221',# 单个颜色
        'episode_len': 77,
        'num_episodes': 50,
        'num_variation': 1,
        'camera_names': ['wrist'],# , 'wrist_depth', 'head'
    },
    'sorting_program_sawyer222':{ # 抓取 用imitate_episodes_sawyer4， 2蓝色-》绿色
        'dataset_dir': '/home/boxjod/sawyer_ws/Datasets/sorting_program_sawyer222',# 单个颜色
        'episode_len': 77,
        'num_episodes': 50,
        'num_variation': 1,
        'camera_names': ['wrist'],# , 'wrist_depth', 'head'
    },
    'sorting_program_sawyer223':{ # 抓取 用imitate_episodes_sawyer4， 3绿色-》红色
        'dataset_dir': '/home/boxjod/sawyer_ws/Datasets/sorting_program_sawyer223',# 单个颜色
        'episode_len': 77,
        'num_episodes': 50,
        'num_variation': 1,
        'camera_names': ['wrist'],# , 'wrist_depth', 'head'
    },
    
    'close_jar':{ # 抓取 用imitate_episodes_sawyer4， 3绿色-》红色
        'dataset_dir': DATA_DIR + '/close_jar/variation0', # 单个颜色
        'episode_len': 93, # 是一个平均值
        'num_episodes': 50,
        'num_variation': 1,
        # 'camera_names': ['wrist', 'head'],# , 'wrist_depth', 'head'
        'camera_names': ['wrist']
    },
    'slide_cabinet_open_and_place_cups':{ # 抓取 用imitate_episodes_sawyer4， 3绿色-》红色
        'dataset_dir': DATA_DIR + '/slide_cabinet_open_and_place_cups/variation0', # 单个颜色
        'episode_len': 100, # 是一个平均值
        'num_episodes': 50,
        'num_variation': 1,
        # 'camera_names': ['wrist', 'head'],# , 'wrist_depth', 'head'
        'camera_names': ['wrist']
    },
    
}

### Simulation envs fixed constants
DT = 0.05
JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
START_ARM_POSE = [0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239,  0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239]

XML_DIR = str(pathlib.Path(__file__).parent.resolve()) + '/assets/' # note: absolute path

# Left finger position limits (qpos[7]), right_finger = -1 * left_finger
MASTER_GRIPPER_POSITION_OPEN = 0.02417
MASTER_GRIPPER_POSITION_CLOSE = 0.01244
PUPPET_GRIPPER_POSITION_OPEN = 0.05800
PUPPET_GRIPPER_POSITION_CLOSE = 0.01844

# Gripper joint limits (qpos[6])
MASTER_GRIPPER_JOINT_OPEN = 0.3083
MASTER_GRIPPER_JOINT_CLOSE = -0.6842
PUPPET_GRIPPER_JOINT_OPEN = 1.4910
PUPPET_GRIPPER_JOINT_CLOSE = -0.6213

############################ Helper functions ############################

MASTER_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_POSITION_CLOSE) / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_POSITION_CLOSE) / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)
MASTER_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE) + MASTER_GRIPPER_POSITION_CLOSE
PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE) + PUPPET_GRIPPER_POSITION_CLOSE
MASTER2PUPPET_POSITION_FN = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(MASTER_GRIPPER_POSITION_NORMALIZE_FN(x))

MASTER_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)
PUPPET_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)
MASTER_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
MASTER2PUPPET_JOINT_FN = lambda x: PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(MASTER_GRIPPER_JOINT_NORMALIZE_FN(x))

MASTER_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)

MASTER_POS2JOINT = lambda x: MASTER_GRIPPER_POSITION_NORMALIZE_FN(x) * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
MASTER_JOINT2POS = lambda x: MASTER_GRIPPER_POSITION_UNNORMALIZE_FN((x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE))
PUPPET_POS2JOINT = lambda x: PUPPET_GRIPPER_POSITION_NORMALIZE_FN(x) * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
PUPPET_JOINT2POS = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN((x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE))

MASTER_GRIPPER_JOINT_MID = (MASTER_GRIPPER_JOINT_OPEN + MASTER_GRIPPER_JOINT_CLOSE)/2
