import os
import h5py
import numpy as np
import cv2 as cv

# 1.检测有多少个文件
# save_dir = "/home/boxjod/sawyer_ws/Datasets/sorting_program_sawyer223/"
save_dir = "/home/boxjod/RLBench_ACT_Sawyer/Datasets/sorting_program21/variation0"
demo_len = 0
for ex_idx in range(150): # 检测有多少个
  dataset_path = os.path.join(save_dir, f'episode_{ex_idx}.hdf5') # save path
  if os.path.exists(dataset_path) == True:
    demo_len = demo_len + 1
print("检测到存在的演示为:",demo_len,"个")

# 2.修改
for idx_demo in range(demo_len):
  dataset_path = os.path.join(save_dir, f'episode_{idx_demo}.hdf5') # save path
  
  data_dict2 = {
        '/action': [], 
        '/observations/images/wrist': [],
        '/observations/qpos': [],
        '/observations/gpos': [],}

  with h5py.File(dataset_path, 'r') as data_dict:
    demo_frame = data_dict['/action'].shape[0]
    # print(demo_frame)
    for idx in range(demo_frame):
      if idx > 0:
        data_dict2['/action'].append(data_dict['/observations/qpos'][idx]) # 预测 qpos
      # print(data_dict['/observations/qpos'][idx])
      data_dict2['/observations/images/wrist'].append(data_dict['/observations/images/wrist'][idx])
      # data_dict2['/observations/images/wrist'][idx] = cv.resize(data_dict2['/observations/images/wrist'][idx], (0, 0), fx=0.25, fy=0.25, interpolation=cv.INTER_AREA)
      
      data_dict2['/observations/qpos'].append(data_dict['/observations/qpos'][idx])
      data_dict2['/observations/gpos'].append(data_dict['/observations/gpos'][idx])
      
    data_dict2['/action'].append(data_dict['/observations/qpos'][idx]) # 预测 qpos
    
  # 3.重新写入
  with h5py.File(dataset_path, 'w', rdcc_nbytes=1024 ** 2 * 2) as root: 
    root.attrs['sim'] = True # 根目录 
    action = root.create_dataset('action', (demo_frame, 8))
    obs = root.create_group('observations')
    image = obs.create_group('images')
    image.create_dataset('wrist', (demo_frame, 120, 160, 3), dtype='uint8',chunks=(1, 120, 160, 3), ) # 480, 640 # 120, 160
    qpos = obs.create_dataset('qpos', (demo_frame, 8))
    gpos = obs.create_dataset('gpos', (demo_frame, 8))

    for name, array in data_dict2.items():
        root[name][...] = array
    print(f"demo [{dataset_path}] successfully",idx_demo)
