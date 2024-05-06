from typing import List, Tuple
from rlbench.backend.task import Task
from pyrep.objects import ProximitySensor, Shape

from rlbench.backend.task import Task
from pyrep.objects.dummy import Dummy

from rlbench.backend.conditions import DetectedCondition, GraspedCondition

from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.const import colors
import numpy as np

class SortingProgram22(Task):

    def init_task(self) -> None:
        # 添加目标：
        self.target_block = Shape('target_block')
        
        # 添加干扰项的和随机范围边缘
        self.target_container0 = Shape('small_container0')
        self.target_container1 = Shape('small_container1')
        self.boundary = Shape('boundary')
        self.distractor_block0 = Shape('distractor_block0')
        self.distractor_block1 = Shape('distractor_block1')
        self.box_boundary = Shape('box_boundary')
        
        # 注册成功条件
        self.register_graspable_objects([self.target_block])
        success_sensor = ProximitySensor('success')
        self.success_detector0 = ProximitySensor('success0')
        self.success_detector1 = ProximitySensor('success1')
        self.register_success_conditions([
            # DetectedCondition(self.robot.arm.get_tip(), success_sensor),
            # GraspedCondition(self.robot.gripper, self.target_block),
            DetectedCondition(self.target_block, self.success_detector0)
        ])
        

    def init_episode(self, index: int) -> List[str]: 
        # index来自cariation
        color_name, color_rgb = colors[index] 
        # # 产生2个在index 前面和后面的 随机数（不跟index相同）
        # color_choices = np.random.choice(list(range(index)) 
        # + list(range(index +1, len(colors))),size=2,replace=True)
        
        # for ob, i in zip([self.distractor_block0, self.distractor_block1],
        #                  color_choices):
        #     ob.set_color(colors[i][1])
        # self.target_container1.set_color(colors[i][1])
            
        # self.target_block.set_color(color_rgb)
        # self.target_container0.set_color(color_rgb)
        
        boundary_spawn = SpawnBoundary([self.boundary])
        try:
            for ob in [self.target_block, self.distractor_block0, self.distractor_block1]:
                boundary_spawn.sample(ob, min_distance=0.1, min_rotation=(0, 0, 0), max_rotation=(0, 0, 0)) # 0.395
        except:
            for ob in [self.target_block, self.distractor_block0, self.distractor_block1]:
                boundary_spawn.sample(ob, min_distance=0.1, min_rotation=(0, 0, 0), max_rotation=(0, 0, 0)) # 0.395
        
        box_boundary_spawn = SpawnBoundary([self.box_boundary])
        try:
            for ob in [self.target_container0, self.target_container1]:
                box_boundary_spawn.sample(ob, min_distance=0, min_rotation=(0, 0, 0), max_rotation=(0.395, 0, 0)) # 0.395
        except:
            for ob in [self.target_container0, self.target_container1]:
                box_boundary_spawn.sample(ob, min_distance=0, min_rotation=(0, 0, 0), max_rotation=(0.395, 0, 0)) # 0.395
            
        return ['put the %s target to the %s box' % (color_name, color_name),
                'put the %s thing to the %s box' % (color_name, color_name)]  # 可以用nlp来处理
    
    # 颜色变化
    def variation_count(self) -> int:
        return len(colors)
    
    def base_rotation_bounds(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        return [0.0, 0.0, 0.0],[0.0, 0.0, 0.0]
    
    def is_static_workspace(self) -> bool:
        return True