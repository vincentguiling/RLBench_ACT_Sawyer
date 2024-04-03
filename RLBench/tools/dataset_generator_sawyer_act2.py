from multiprocessing import Process, Manager

from pyrep.const import RenderMode

from rlbench.observation_config import ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity, JointPosition
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend.utils import task_file_to_task_class
from rlbench.environment import Environment
import rlbench.backend.task as task

import os, socket
import pickle
from PIL import Image
from rlbench.backend import utils
from rlbench.backend.const import *
import numpy as np

from absl import app
from absl import flags

import time
import h5py

FLAGS = flags.FLAGS

# 使用还是 --tasks reach_target
flags.DEFINE_string('save_path',
                    '/tmp/rlbench_data/', # 默认位置
                    'Where to save the demos.')
flags.DEFINE_list('tasks', [],
                  'The tasks to collect. If empty, all tasks are collected.')
flags.DEFINE_list('image_size', [640, 480],# [128, 128], ACT是读取的 [height x width]，coppliasim是 [width x height]
                  'The size of the images tp save.')
flags.DEFINE_enum('renderer',  'opengl3', ['opengl', 'opengl3'],
                  'The renderer to use. opengl does not include shadows, '
                  'but is faster.')
flags.DEFINE_integer('processes', 1,
                     'The number of parallel processes during collection.')
flags.DEFINE_integer('episodes_per_task', 10,
                     'The number of episodes to collect per task.')
flags.DEFINE_integer('variations', -1,
                     'Number of variations to collect per task. -1 for all.')
flags.DEFINE_integer('episode_len', 0,
                     'the lenght of one episode, means how many steps of one episode. if not assign, not strict for the steps of generate')

np.set_printoptions(linewidth=200)

def check_and_make(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def matrix_to_pose(matrix): # 定义一个matrix。现在要做的事，判断一下relate to robot.ram的 pose 和 matrix 里面的(x,y,z)是否相同，如果相同，则不需要我来变换了 【一样的】   
    pose = matrix
    # obs.gripper_matrix: [[-9.40002918e-01  2.54769984e-04  3.41166764e-01  4.29283112e-01]
    #                      [ 2.09459744e-04  9.99999940e-01 -1.69645005e-04  1.60404623e-01]
    #                      [-3.41166824e-01 -8.80060616e-05 -9.40002918e-01  7.26700842e-01]
    #                      [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]
                        #  [ 4.29312766e-01  1.60412401e-01  7.26682484e-01  1.11043439e-04  9.84890819e-01 -4.92185354e-05  1.73176572e-01]
    return pose
    # 什么事robot.arm?在哪里？

def save_demo(demo, example_path, ex_idx):
    data_dict = {
        '/action': [], 
        '/observations/images/wrist': [],
        '/observations/images/wrist_depth': [],
        '/observations/images/head': [],
        '/observations/gpos': [],
        '/observations/qpos': [],
    }
    max_timesteps = len(demo)
    
    for i, obs in enumerate(demo): 
        if i != 0: # action是下一步的姿态
            data_dict['/action'].append(np.append(obs.gripper_pose, obs.gripper_open))
        
        # data_dict['/observations/images/wrist'].append(obs.wrist_rgb*255) # 480， 640， 3 # 从最开始就错了，为什么要*255！
           
        data_dict['/observations/images/wrist'].append(obs.wrist_rgb) # 480， 640， 3
        wrist_depth = utils.float_array_to_rgb_image(obs.wrist_depth, scale_factor=DEPTH_SCALE)
        wrist_depth = np.clip(np.array(wrist_depth), 0, 255).astype(np.uint8)
        data_dict['/observations/images/wrist_depth'].append(wrist_depth)
        data_dict['/observations/images/head'].append(obs.head_rgb)
        data_dict['/observations/gpos'].append(np.append(obs.gripper_pose, obs.gripper_open))
        data_dict['/observations/qpos'].append(np.append(obs.joint_positions, obs.gripper_open))
    
    data_dict['/action'].append(np.append(obs.gripper_pose,obs.gripper_open))
    dataset_path = os.path.join(example_path, f'episode_{ex_idx}') # save path
    check_and_make(example_path)
    
    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root: 
        root.attrs['sim'] = True # 根目录 
        action = root.create_dataset('action', (max_timesteps, 8))
        obs = root.create_group('observations')
        image = obs.create_group('images')
        image.create_dataset('wrist', (max_timesteps, 480, 640, 3), dtype='uint8',chunks=(1, 480, 640, 3), ) 
        image.create_dataset('wrist_depth', (max_timesteps, 480, 640, 3), dtype='uint8',chunks=(1, 480, 640, 3), ) 
        image.create_dataset('head', (max_timesteps, 480, 640, 3), dtype='uint8',chunks=(1, 480, 640, 3), ) 
        gpos = obs.create_dataset('gpos', (max_timesteps, 8))
        qpos = obs.create_dataset('qpos', (max_timesteps, 8))
        
        for name, array in data_dict.items():
            root[name][...] = array
        print("demo save successfully")
                     
def run(i, lock, task_index, variation_count, results, file_lock, tasks):
    """Each thread will choose one task and variation, and then gather
    all the episodes_per_task for that variation."""

    # Initialise each thread with random seed
    np.random.seed(None)
    num_tasks = len(tasks)
    img_size = list(map(int, FLAGS.image_size))
    obs_config = ObservationConfig()
    
    obs_config.set_all(False)
    obs_config.wrist_camera.set_all(True)
    obs_config.head_camera.set_all(True)
    obs_config.set_all_low_dim(True)
    
    obs_config.wrist_camera.image_size = img_size
    obs_config.head_camera.image_size = img_size
    obs_config.wrist_camera.depth_in_meters = False

    if FLAGS.renderer == 'opengl':
        obs_config.wrist_camera.render_mode = RenderMode.OPENGL
    elif FLAGS.renderer == 'opengl3':
        obs_config.wrist_camera.render_mode = RenderMode.OPENGL3
    
    headless_val = False
    if socket.gethostname() != 'XJ':
        headless_val = True
    
    ##############################################################################################################
    rlbench_env = Environment( # 训练数据生成是使用的构建的场景
        action_mode=MoveArmThenGripper(JointVelocity(), Discrete()),
        obs_config=obs_config,
        headless=headless_val,
        robot_setup='sawyer'
        )
    ##############################################################################################################
    
    rlbench_env.launch()
    task_env = None
    tasks_with_problems = results[i] = ''
    while True:
        with lock: # Figure out what task/variation this thread is going to do
            if task_index.value >= num_tasks:
                print('Process', i, 'finished')
                break

            my_variation_count = variation_count.value
            t = tasks[task_index.value]
            task_env = rlbench_env.get_task(t)
            var_target = task_env.variation_count()
            
            if FLAGS.variations >= 0:
                var_target = np.minimum(FLAGS.variations, var_target)
            if my_variation_count >= var_target:
                variation_count.value = my_variation_count = 0
                task_index.value += 1

            variation_count.value += 1
            if task_index.value >= num_tasks:
                print('Process', i, 'finished')
                break
            t = tasks[task_index.value]

        task_env = rlbench_env.get_task(t)
        task_env.set_variation(my_variation_count)
        descriptions, _ = task_env.reset()

        variation_path = os.path.join(
            FLAGS.save_path, task_env.get_name(),
            VARIATIONS_FOLDER % my_variation_count)

        check_and_make(variation_path)
        with open(os.path.join(
                variation_path, VARIATION_DESCRIPTIONS), 'wb') as f:
            pickle.dump(descriptions, f)

        abort_variation = False
        for ex_idx in range(FLAGS.episodes_per_task):
            print('Process', i, '// Task:', task_env.get_name(),
                  '// Variation:', my_variation_count, '// Demo:', ex_idx)
            
            attempts = 10 # 每次episode给10次机会，但每一个demo默认又有10次机会，一共一个episode由100次机会
            while attempts > 0:
                try:
                    # TODO: for now we do the explicit looping.
                    #############################################################################################################################################
                    demo, = task_env.get_demos(amount=1, live_demos=True, episode_len=FLAGS.episode_len)
                    #############################################################################################################################################
                except Exception as e: 
                    attempts -= 1
                    if attempts > 0:
                        continue      
                    problem = (
                        'Process %d failed collecting task %s (variation: %d, '
                        'example: %d). Skipping this task/variation.\n%s\n' % (
                            i, task_env.get_name(), my_variation_count, ex_idx,
                            str(e))
                    )
                    print(problem)
                    tasks_with_problems += problem
                    abort_variation = True
                    break
                with file_lock:
                    save_demo(demo, variation_path, ex_idx)
                break
            if abort_variation:
                break

    results[i] = tasks_with_problems
    rlbench_env.shutdown()

def main(argv):
    task_files = [t.replace('.py', '') for t in os.listdir(task.TASKS_PATH)
                  if t != '__init__.py' and t.endswith('.py')]
    if len(FLAGS.tasks) > 0:
        for t in FLAGS.tasks:
            if t not in task_files:
                raise ValueError('Task %s not recognised!.' % t)
        task_files = FLAGS.tasks

    tasks = [task_file_to_task_class(t) for t in task_files]
    manager = Manager()
    result_dict = manager.dict()
    file_lock = manager.Lock()
    task_index = manager.Value('i', 0)
    variation_count = manager.Value('i', 0)
    lock = manager.Lock()

    check_and_make(FLAGS.save_path)
    processes = [Process(
        target=run, args=(
            i, lock, task_index, variation_count, result_dict, file_lock,
            tasks))
        for i in range(FLAGS.processes)]
    [t.start() for t in processes]
    [t.join() for t in processes]

    print('Data collection done!')
    for i in range(FLAGS.processes):
        print(result_dict[i])
        
if __name__ == '__main__':
  app.run(main)
