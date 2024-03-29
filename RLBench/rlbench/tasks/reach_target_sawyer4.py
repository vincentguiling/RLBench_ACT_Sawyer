from typing import List, Tuple
from rlbench.backend.task import Task

from pyrep.objects import ProximitySensor, Shape, Dummy
from rlbench.backend.conditions import DetectedCondition, GraspedCondition

from rlbench.const import colors
import numpy as np
from rlbench.backend.spawn_boundary import SpawnBoundary


class ReachTargetSawyer3(Task):

    def init_task(self) -> None:
        # 手臂随机初始化点
        self.arm_pose = Dummy('waypoint0')
        self.arm_boundary = Shape('arm_boundary')
        
        # 添加目标：
        self.target_block = Shape('target_block')
        
        # 添加干扰项的和随机范围边缘
        self.boundary = Shape('boundary')
        self.distractor_block0 = Shape('distractor_block0')
        self.distractor_block1 = Shape('distractor_block1')
        
        # 注册成功条件
        success_sensor = ProximitySensor('success')
        self.register_success_conditions([
            DetectedCondition(self.robot.arm.get_tip(), success_sensor)
        ])
        

    def init_episode(self, index: int) -> List[str]: 
        # index来自cariation
        color_name, color_rgb = colors[index] 
        # 产生2个在index 前面和后面的 随机数（不跟index相同）
        color_choices = np.random.choice(list(range(index)) 
        + list(range(index +1, len(colors))),size=2,replace=True)
        
        for ob, i in zip([self.distractor_block0, self.distractor_block1],
                         color_choices):
            ob.set_color(colors[i][1])
            
        self.target_block.set_color(color_rgb)
        
        boundary_spawn = SpawnBoundary([self.boundary])
        for ob in [self.target_block, self.distractor_block0, self.distractor_block1]:
            boundary_spawn.sample(ob, min_distance=0.2, min_rotation=(0, 0, 0), max_rotation=(0, 0, 0))
        
        arm_boundary_spawn = SpawnBoundary([self.arm_boundary])
        arm_boundary_spawn.sample(self.arm_pose, min_rotation=(0, 0, 0), max_rotation=(0, 0, 0))
        
        return ['grasp the %s target' % color_name,
                'grasp the %s thing' % color_name]  # 可以用nlp来处理
    
    # 颜色变化
    def variation_count(self) -> int:
        return len(colors)
    
    def base_rotation_bounds(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        return [0.0, 0.0, 0.0],[0.0, 0.0, 0.0]
    
    def is_static_workspace(self) -> bool:
        return True