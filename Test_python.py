
#  显示模型结构

# 针对有网络模型，但还没有训练保存 .pth 文件的情况
# import netron
# import torch.onnx
# from torch.autograd import Variable
# from torchvision.models import resnet18  # 以 resnet18 为例

# myNet = resnet18()  # 实例化 resnet18
# x = torch.randn(16, 3, 40, 40)  # 随机生成一个输入
# modelData = "./demo.pth"  # 定义模型数据保存的路径
# # modelData = "./demo.onnx"  # 有人说应该是 onnx 文件，但我尝试 pth 是可以的 
# torch.onnx.export(myNet, x, modelData)  # 将 pytorch 模型以 onnx 格式导出并保存
# netron.start(modelData)  # 输出网络结构

#  针对已经存在网络模型 .pth 文件的情况
# import netron

# modelData = "/home/boxjod/RLBench_ACT_Sawyer/Trainings/sorting_program21/50demo_32step_20chunk_8batch_efficientnet_b0/policy_best_epoch3000.pth"  # 定义模型数据保存的路径
# netron.start(modelData)  # 输出网络结构


# 文本编码差异
import json

json_name_path = "/home/boxjod/RLBench_ACT_Sawyer/Datasets/sorting_program21/test/grasp the green target.json"
with open(json_name_path, "r") as f:
    grasp_green = json.load(f)[0]['embedding'][0]

json_name_path = "/home/boxjod/RLBench_ACT_Sawyer/Datasets/sorting_program21/test/grasp the red target.json"
with open(json_name_path, "r") as f:
    grasp_red = json.load(f)[0]['embedding'][0]

json_name_path = "/home/boxjod/RLBench_ACT_Sawyer/Datasets/sorting_program21/test/grasp the blue target.json"
with open(json_name_path, "r") as f:
    grasp_blue = json.load(f)[0]['embedding'][0]
    
################

json_name_path = "/home/boxjod/RLBench_ACT_Sawyer/Datasets/sorting_program21/test/blue.json"
with open(json_name_path, "r") as f:
    blue = json.load(f)[0]['embedding'][0]  

json_name_path = "/home/boxjod/RLBench_ACT_Sawyer/Datasets/sorting_program21/test/red.json"
with open(json_name_path, "r") as f:
    red = json.load(f)[0]['embedding'][0]  
    
json_name_path = "/home/boxjod/RLBench_ACT_Sawyer/Datasets/sorting_program21/test/green.json"
with open(json_name_path, "r") as f:
    green = json.load(f)[0]['embedding'][0]   
        
################
        
json_name_path = "/home/boxjod/RLBench_ACT_Sawyer/Datasets/sorting_program21/test/put the green target to the green box.json"
with open(json_name_path, "r") as f:
  put_green_to_green_box = json.load(f)[0]['embedding'][0]

json_name_path = "/home/boxjod/RLBench_ACT_Sawyer/Datasets/sorting_program21/test/red to red.json"
with open(json_name_path, "r") as f:
  red_to_red = json.load(f)[0]['embedding'][0]    
    
json_name_path = "/home/boxjod/RLBench_ACT_Sawyer/Datasets/sorting_program21/test/red to green.json"
with open(json_name_path, "r") as f:
  red_to_green = json.load(f)[0]['embedding'][0]    
    
json_name_path = "/home/boxjod/RLBench_ACT_Sawyer/Datasets/sorting_program21/test/red to blue.json"
with open(json_name_path, "r") as f:
  red_to_blue = json.load(f)[0]['embedding'][0]      
  
json_name_path = "/home/boxjod/RLBench_ACT_Sawyer/Datasets/sorting_program21/test/blue to green.json"
with open(json_name_path, "r") as f:
  blue_to_green = json.load(f)[0]['embedding'][0]      
      
x = red_to_green
y = blue_to_green
filter = 0.1
diff = [abs(l1-l2) for (l1,l2) in zip(x,y)]

big = [l3>filter for l3 in diff].count(True)/len(x)
big1 = [l3>filter for l3 in x].count(True)/len(x)
big2 = [l3>filter for l3 in y].count(True)/len(y)

print("相似程度:",1 - big)

