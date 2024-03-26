from typing import List, Tuple
from rlbench.backend.task import Task

from pyrep.objects import ProximitySensor, Shape
from rlbench.backend.conditions import DetectedCondition

from rlbench.const import colors
import numpy as np
from rlbench.backend.spawn_boundary import SpawnBoundary

class ReachTargetSawyer(Task):

    def init_task(self) -> None:
        self.target = Shape('target0')
        # 添加干扰项
        self.distractor0 = Shape('distractor0')
        self.distractor1 = Shape('distractor1')
        # 添加/绑定触碰传感器
        success_sensor = ProximitySensor('success')
        
        # 添加干扰项的边缘
        self.boundary = Shape('boundary')
        
        # 注册成功条件
        self.register_success_conditions([
            DetectedCondition(self.robot.arm.get_tip(), success_sensor)
        ])
        
    # 可以用nlp来处理
    def init_episode(self, index: int) -> List[str]: 
        # index来自cariation
        color_name, color_rgb = colors[index] 
        # 产生2个在index 前面和后面的 随机数（不跟index相同）
        color_choices = np.random.choice(list(range(index)) 
        + list(range(index +1, len(colors))),size=2,replace=True)
        
        for ob, i in zip([self.distractor0, self.distractor1],
                         color_choices):
            ob.set_color(colors[i][1])
            
        self.target.set_color(color_rgb)
        
        b = SpawnBoundary([self.boundary])
        for ob in [self.target, self.distractor0, self.distractor1]:
            b.sample(ob, min_distance=0.2, min_rotation=(0, 0, 0),
                     max_rotation=(0, 0, 0))
        
        return ['reach the %s target' % color_name,
                'reach the %s thing' % color_name]
    
    # 颜色变化
    def variation_count(self) -> int:
        return len(colors)

    def base_rotation_bounds(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        return [0.0, 0.0, 0.0],[0.0, 0.0, 0.0]
    
    def is_static_workspace(self) -> bool:
        return True