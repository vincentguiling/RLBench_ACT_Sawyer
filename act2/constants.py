import pathlib

### Task parameters 

DATA_DIR = 'Datasets'
SIM_TASK_CONFIGS = {
    'reach_target_sawyer':{
        'dataset_dir': DATA_DIR + '/reach_target_sawyer/variation0',
        'num_episodes': 50,
        'episode_len': 51,
        'camera_names': ['wrist']
    },
    'reach_target_sawyer2':{
        'dataset_dir': DATA_DIR + '/reach_target_sawyer2/variation0',
        'num_episodes': 50,
        'episode_len': 51,
        'camera_names': ['wrist','wrist_depth']
    },
    'reach_target_sawyer3':{ 
        'dataset_dir': DATA_DIR + '/reach_target_sawyer3/variation0', ## 有一次名字忘记改了
        'num_episodes': 50,
        'episode_len': 37,
        'camera_names': ['wrist']
    },
    'reach_target_sawyer4':{ 
        'dataset_dir': DATA_DIR + '/reach_target_sawyer4/variation0',
        'num_episodes': 50,
        'episode_len': 51,
        'camera_names': ['wrist','wrist_depth']
    },
    'sorting_program':{ 
        'dataset_dir': DATA_DIR + '/sorting_program/variation0',
        'num_episodes': 50,
        'episode_len': 31,
        'camera_names': ['wrist','wrist_depth']
    },
    'sorting_program2':{ 
        'dataset_dir': DATA_DIR + '/sorting_program2/variation0',
        'num_episodes': 50,
        'episode_len': 88,
        'camera_names': ['wrist'] # , 'wrist_depth', 'head'
    },
    'sorting_program21':{ 
        'dataset_dir': DATA_DIR + '/sorting_program21/variation0',
        'num_episodes': 50,
        'episode_len': 32,
        'camera_names': ['wrist'] # , 'wrist_depth', 'head'
    },
    'sorting_program22':{ 
        'dataset_dir': DATA_DIR + '/sorting_program22/variation0',
        'num_episodes': 50,
        'episode_len': 63,
        'camera_names': ['wrist', 'head'] # , 'wrist_depth'
    },
    'sorting_program3':{ 
        'dataset_dir': DATA_DIR + '/sorting_program3/variation0',
        'num_episodes': 50,
        'episode_len': [32,63],
        'camera_names': [['wrist'], ['wrist', 'head']],
        'task_steps':['sorting_program21', 'sorting_program22']
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
