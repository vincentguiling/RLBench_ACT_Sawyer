# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
from torch import nn
from torch.autograd import Variable
from .backbone import build_backbone
from .transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer

import numpy as np

import IPython
e = IPython.embed


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)

# from utils import get_gpu_mem_info

# def print_gpu_mem():
#     gpu_mem_total, gpu_mem_used, gpu_mem_free = get_gpu_mem_info()
#     return (gpu_mem_used/gpu_mem_total)*100

class DETRVAE(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbones, transformer, encoder, state_dim, action_dim, num_queries, camera_names, use_language=False, use_film=False, num_command=2, use_gpos=True, policy_class='ACT'):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            action_dim: robot output dimension of action
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            use_film: Whether to use FiLM language encoding.
        """
        super().__init__()
        self.num_queries = num_queries
        self.camera_names = camera_names
        self.transformer = transformer
        self.encoder = encoder
        self.hidden_dim = hidden_dim = transformer.d_model
        
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        self.use_language = use_language
        self.use_film = use_film
        if use_language:
            self.lang_embed_proj = nn.Linear(768, hidden_dim)  # 512 / 768 for clip / distilbert
        
        self.use_diff = False if state_dim == 8 else True # use_diff
        self.use_gpos = use_gpos # use_gpos
        
        if self.use_diff:
            # input_dim pose to hidden_dim = 7 + 1 + 7
            self.input_proj_robot_state_qpos = nn.Linear(15, hidden_dim)
            self.encoder_qpos_proj = nn.Linear(15, hidden_dim)  # project qpos to embedding
            if use_gpos:
                self.input_proj_robot_state_gpos = nn.Linear(15, hidden_dim)
                self.encoder_gpos_proj = nn.Linear(15, hidden_dim)  # project gpos to embedding ###############
        else:
            self.input_proj_robot_state_qpos = nn.Linear(8, hidden_dim)
            self.encoder_qpos_proj = nn.Linear(8, hidden_dim)  # project qpos to embedding
            if use_gpos:
                self.input_proj_robot_state_gpos = nn.Linear(8, hidden_dim)
                self.encoder_gpos_proj = nn.Linear(8, hidden_dim)  # project gpos to embedding ###############
        
        # backbones
        if backbones is not None:
            self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1) # 卷积层
            self.backbones = nn.ModuleList(backbones) 

        else:
            self.input_proj_env_state = nn.Linear(7, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbones = None

        # encoder extra parameters
        self.latent_dim = 32 # final size of latent z # TODO tune
        self.cls_embed = nn.Embedding(1, hidden_dim) # extra cls token embedding
        self.encoder_action_proj = nn.Linear(8, hidden_dim) # project action to embedding

        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim*2) # project hidden state to latent std, var

        # 编码器类型
        print(policy_class)
        if 'E0' in policy_class: # 最原始的 Z
            self.use_Z = True
            self.use_history = False
            self.use_history_images =  False 
            
        elif 'E1' in policy_class: # 没有编码器的
            self.use_Z = False
            self.use_history = False
            self.use_history_images =  False 
        
        elif 'E2' in policy_class: # 历史动作编码器
            self.use_Z = False
            self.use_history = True
            self.use_history_images =  False 
        
        else: # 'E3' in policy_class: # 历史动作与图像编码器 （默认）
            self.use_Z = False
            self.use_history = True
            self.use_history_images =  True
        
        print(f"{self.use_Z=}, {self.use_history=}, {self.use_history_images=}")
        
        if self.use_Z:
            self.register_buffer('pos_table', get_sinusoid_encoding_table(1+1+num_queries, hidden_dim)) # [CLS], qpos，gpos, a_seq
        else:
            self.register_buffer('pos_table', get_sinusoid_encoding_table(1+num_queries, hidden_dim)) # [CLS], qpos，gpos, a_seq
        
        # decoder extra parameters
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim) # project latent sample to embedding
        
        if self.use_gpos and self.use_language:
            pos_embed_dim = 4
        elif self.use_gpos or self.use_language:
            pos_embed_dim = 3
        else:
            pos_embed_dim = 2
        # pos_embed_dim = 4 if self.use_language else 3 # 因为command_embedding 加了1，因为gpos分开了 加了1
        print(f"Transformer input block number = {pos_embed_dim + num_queries}")
        self.additional_pos_embed = nn.Embedding(pos_embed_dim, hidden_dim) # learned position embedding for proprio and latent

    def forward(self, qpos, gpos, image, env_state, 
                history_images=None, history_action=None, is_pad_history=None, 
                actions=None, is_pad_action=None, 
                command_embedding=None):
        """
        qpos: batch, qpos_dim
        gpos: batch, gpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        command_embedding: batch, command_embedding_dim
        """
        is_training = actions is not None # train or val
        bs, _ = qpos.shape # bs = Batch_Size
        
        # print("模型计算开始 ：", print_gpu_mem())
        
        # Project the command embedding to the required dimension
        if command_embedding is not None:
            if self.use_language:
                command_embedding_proj = self.lang_embed_proj(command_embedding)
            else:
                raise NotImplementedError
        
        # print(f"{self.use_history=}, {self.use_Z=}, {self.use_history_images=}")
        
        # return
        ### Obtain latent z from action sequence
        if self.use_history: # 历史 action 和 images 用 encoder 编码之后输出
            history_action_embed = self.encoder_action_proj(history_action) # (bs, seq, hidden_dim), (8,10) x (8, 512)  = (8, 10, 512)
            cls_embed = self.cls_embed.weight # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1) # (bs, 1, hidden_dim)
            
            if self.use_history_images:
                if not is_training:
                    history_action_embed = torch.unsqueeze(history_action_embed, axis=0).repeat(bs, 1, 1)
                    is_pad_history = torch.unsqueeze(is_pad_history, axis=0).repeat(bs, 1) # (1, 90) 8*10
                    
                    # 利用历史编码的数据，直接输入
                    history_all_cam_src = torch.from_numpy(history_images[0]).unsqueeze(0).cuda() # (1,10,512) bs = 1
                    history_all_cam_pos = torch.from_numpy(history_images[1]).unsqueeze(0).cuda() # (1,10,512) bs = 1 # 每次用的都是一模一样的，只要形状一样
                    
                else:
                    # 使用backbone 对 history_images 图像处理【只有在训练的时候才会现场编译】【直接把 num_queries 叠加到 batch_size 上 】
                    # [batch_size, history_idx, cam_id, chanel, width, height] -> [batch_size * history_idx, cam_id, chanel, width, height]
                    target_shape = np.append(-1, history_images.shape[2:])  # [ -1,   1,   3, 120, 160]，其中-1参数可以使torch.view自动拼接1、2维度的数据
                    history_images = history_images.view(target_shape[0], target_shape[1], target_shape[2], target_shape[3], target_shape[4])
                    
                    history_all_cam_src = []
                    history_all_cam_pos = []
                    for cam_id, cam_name in enumerate(self.camera_names): # 0 ‘wrist’
                        # features, pos = self.backbones[cam_id](history_images[:, cam_id]) # [batch_size * history_idx, , cam_id, chanel, width, height]
                        
                        if self.use_film:
                            # command_embedding 需要重复batch_size * history_idx次，原本是需要重复batch_size
                            command_embedding_history = command_embedding.repeat(self.num_queries,1)
                            # total_bytes = command_embedding_history.numel() * command_embedding_history.element_size()  # 393216，占用内存0.375MB
                            # print(f"{command_embedding_history.shape=}, {total_bytes/1024=}")
                            features, pos = self.backbones[cam_id](history_images[:, cam_id], command_embedding_history) # add command_embedding
                        else:
                            features, pos = self.backbones[cam_id](history_images[:, cam_id]) # image[:,id]前面的冒号就是表示的batch_size
                        
                        features = features[0] # take the last layer feature [80, 1536, 4, 5]
                        pos = pos[0] # take the last layer pos [10, 512, 4, 5]
                        
                        history_all_cam_src.append(self.input_proj(features)) # 训练的时候[80, 1536, 4, 5] -> [80, 512, 4, 5] -> [1, 80, 512, 4, 5]
                        history_all_cam_pos.append(pos)  # pos的 shape 本来就是 512，有多少层呢？
                        
                    # fold camera dimension into width dimension [1, 80, 512, 4*n, 5] -> [80, 512, 4*n, 5], 把多张图片折叠到了 width 维度
                    history_all_cam_src = torch.cat(history_all_cam_src, axis=3)  
                    history_all_cam_pos = torch.cat(history_all_cam_pos, axis=3)  # [1, 10, 512, 4*n, 5] -> [10, 512, 4*n, 5]
                    
                    # 如何处理成Transformer Encoder的输入格式 (bs, 512, 4, 5) -> 4x5=20 (bs, 512, 20) -> (bs,1, 512)
                    history_all_cam_src = history_all_cam_src.flatten(2) 
                    history_all_cam_src = torch.mean(history_all_cam_src, dim=2).unsqueeze(1) 
                    
                    # (1, 512, 4, 5) -> (num_queries, 512,20）->  (num_queries, 512) -> (1, num_queries, 512) pos本身不需要batch_size，是共享的
                    history_all_cam_pos = history_all_cam_pos.flatten(2).repeat(self.num_queries, 1, 1) 
                    history_all_cam_pos = torch.mean(history_all_cam_pos, dim=2).unsqueeze(0)  
                    
                    # src = (80,1,512) -> (8,10,512)； 
                    history_all_cam_src = history_all_cam_src.view(bs, self.num_queries, self.hidden_dim) # [bs, seq, 512]
                
                # (bs, 1 + seq + seq, hidden_dim) -> (1 + seq + seq, bs, hidden_dim)
                encoder_input = torch.cat([cls_embed, history_action_embed, history_all_cam_src], axis=1) 
                encoder_input = encoder_input.permute(1, 0, 2) 
                
                # (bs, 1 + seq + seq)
                cls_joint_is_pad = torch.full((bs, 1), False).to(qpos.device) 
                is_pad_history = torch.cat([cls_joint_is_pad, is_pad_history, is_pad_history], axis=1)
                
                # (1, seq + seq + 1, hidden_dim) -> ((seq + 1) + seq, 1, hidden_dim)
                pos_embed = self.pos_table.clone().detach()
                pos_embed = torch.cat([pos_embed, history_all_cam_pos], axis=1)
                pos_embed = pos_embed.permute(1, 0, 2)  
                # print(f"{encoder_input.shape=}")
                # return
                encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad_history)     
            else:
                if not is_training: # 需要伪装一个 batch_size 出来
                    history_action_embed = torch.unsqueeze(history_action_embed, axis=0).repeat(bs, 1, 1)
                    is_pad_history = torch.unsqueeze(is_pad_history, axis=0).repeat(bs, 1) # (1, 90) 8*10
                    
                encoder_input = torch.cat([cls_embed, history_action_embed], axis=1) # (bs, 1 + seq, hidden_dim) -> (1 + seq, bs, hidden_dim)
                encoder_input = encoder_input.permute(1, 0, 2) 
                
                cls_joint_is_pad = torch.full((bs, 1), False).to(qpos.device) # (bs, 1 + seq)
                is_pad_history = torch.cat([cls_joint_is_pad, is_pad_history], axis=1)

                pos_embed = self.pos_table.clone().detach()# (1, seq + 1, hidden_dim) -> (seq + 1, 1, hidden_dim)
                pos_embed = pos_embed.permute(1, 0, 2)  
                
                encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad_history)
                
            encoder_output = encoder_output[0] # take cls output only
            
            latent_info = self.latent_proj(encoder_output) # 映射到 (hidden_dim, self.latent_dim*2)
            mu = latent_info[:, :self.latent_dim] # mu取前面latent_dim
            logvar = latent_info[:, self.latent_dim:] # logvar取后面latent_dim
            latent_sample = reparametrize(mu, logvar) # mu和logvar都在这里用了
            latent_input = self.latent_out_proj(latent_sample)
            
        elif self.use_Z: # 使用风格变量 Z 用 encoder 编码之后输出
            if is_training:
                # project action sequence to embedding dim, and concat with a CLS token
                action_embed = self.encoder_action_proj(actions) # (bs, seq, hidden_dim), (8,10) x (8, 10, 512) 
                qpos_embed = self.encoder_qpos_proj(qpos)  # (bs, hidden_dim), (8, 512)
                qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim), (8, 1, 512)

                cls_embed = self.cls_embed.weight # (1, hidden_dim)
                cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1) # (bs, 1, hidden_dim)
                
                # if self.use_gpos:
                #     gpos_embed = self.encoder_gpos_proj(gpos)  # (bs, hidden_dim)
                #     gpos_embed = torch.unsqueeze(gpos_embed, axis=1)  # (bs, 1, hidden_dim)
                # encoder_input = torch.cat([cls_embed, qpos_embed, gpos_embed, action_embed], axis=1) # (bs, seq+3, hidden_dim), (8, 13, 512)
                encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1)# (bs, seq+2, hidden_dim), (8, 13, 512)
                encoder_input = encoder_input.permute(1, 0, 2) # (seq+1, bs, hidden_dim)
                
                # do not mask cls token
                cls_joint_is_pad = torch.full((bs, 2), False).to(qpos.device) # cls+qpos=(8,2) ; cls+qpos+gpos=(8, 3)
                is_pad_action = torch.cat([cls_joint_is_pad, is_pad_action], axis=1)  # (bs, 2+10) -> (8, 12) 
                # 其实transformer还是固定长度输入，只是使用了cls填充了不是固定长度的值
                
                # obtain position embedding
                pos_embed = self.pos_table.clone().detach()
                pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
                
                # query model
                encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad_action)
                encoder_output = encoder_output[0] # take cls output only
                
                latent_info = self.latent_proj(encoder_output)
                mu = latent_info[:, :self.latent_dim] # mu 是做隐层的映射，用来干什么了？
                logvar = latent_info[:, self.latent_dim:] # logvar是隐层的另一个维度的映射提取
                latent_sample = reparametrize(mu, logvar) # mu和logvar都在这里用了
                latent_input = self.latent_out_proj(latent_sample)
                
            else: 
                mu = logvar = None
                latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
                latent_input = self.latent_out_proj(latent_sample)
                
        else: # 什么也没有，没有history，没有Z
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
            latent_input = self.latent_out_proj(latent_sample)
        # print("历史编码结束 ：", print_gpu_mem())

        if self.backbones is not None: # 用骨干网络做图像预处理
            # Image observation features and position embeddings
            all_cam_features = []
            all_cam_pos = []
            for cam_id, cam_name in enumerate(self.camera_names):
                if self.use_film:
                    features, pos = self.backbones[cam_id](image[:, cam_id], command_embedding) # add command_embedding
                else:
                    features, pos = self.backbones[cam_id](image[:, cam_id]) # image[:,id]前面的冒号就是表示的batch_size
            
                features = features[0] # take the last layer feature # (bs,1536,4,5)
                pos = pos[0] # (1,1536,4,5)
                
                all_cam_features.append(self.input_proj(features)) # (bs,1536,4,5) x (1536, 512) = (1,512,4,5),(4,5)是卷积核？
                all_cam_pos.append(pos)
            
            # fold camera dimension into width dimension
            src = torch.cat(all_cam_features, axis=3)
            pos = torch.cat(all_cam_pos, axis=3)
            
            # Only append the command embedding if we are using one-hot
            command_embedding_to_append = (command_embedding_proj if self.use_language else None)
            
            # proprioception features
            proprio_input_qpos = self.input_proj_robot_state_qpos(qpos)
            if self.use_gpos:
                proprio_input_gpos = self.input_proj_robot_state_gpos(gpos)# 将gpos单独编码了
            else:
                proprio_input_gpos = None

            hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input, 
                                proprio_input_qpos, proprio_input_gpos, 
                                self.additional_pos_embed.weight, 
                                command_embedding=command_embedding_to_append)[0]

            image_feature = None
            if not is_training:
               image_feature = src.flatten(2) # (1, 512, 4, 5) -> 4x5=20 (1, 512, 20)
               image_feature = torch.mean(image_feature, dim=2).unsqueeze(1)# (1, 512, 20) -> (1, 512)
               image_pos = pos.flatten(2)# (1, 512, 4, 5) -> 4x5=20 (1, 512, 20)
               image_pos = torch.mean(image_pos, dim=2).unsqueeze(1)# (1, 512, 20) -> (1, 512)，但是每一个 image_pos不一样吧
               
               image_feature = image_feature.cpu()
               image_pos = image_pos.cpu()
               encode_history_image_pos = [image_feature, image_pos]
               
        else: # 不使用backbone
            proprio_input_qpos = self.input_proj_robot_state_qpos(qpos)
            env_state = self.input_proj_env_state(env_state)
            
            if self.use_gpos:
                proprio_input_gpos = self.input_proj_robot_state_gpos(gpos)
                proprio_input_gpos = None
                
            transformer_input = torch.cat([proprio_input_qpos, proprio_input_gpos, env_state], axis=1) # seq length = 2
                
            hs = self.transformer(transformer_input, None, self.query_embed.weight, self.pos.weight)[0]
            
        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)
        if is_training:
            return a_hat, is_pad_hat, [mu, logvar]
        else:
            return a_hat, is_pad_hat, [mu, logvar], encode_history_image_pos

class CNNMLP(nn.Module):
    def __init__(self, backbones, state_dim, camera_names):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.camera_names = camera_names
        self.action_head = nn.Linear(1000, state_dim) # TODO add more
        if backbones is not None:
            self.backbones = nn.ModuleList(backbones)
            backbone_down_projs = []
            for backbone in backbones:
                down_proj = nn.Sequential(
                    nn.Conv2d(backbone.num_channels, 128, kernel_size=5),
                    nn.Conv2d(128, 64, kernel_size=5),
                    nn.Conv2d(64, 32, kernel_size=5)
                )
                backbone_down_projs.append(down_proj)
            self.backbone_down_projs = nn.ModuleList(backbone_down_projs)

            mlp_in_dim = 768 * len(backbones) + 8
            self.mlp = mlp(input_dim=mlp_in_dim, hidden_dim=1024, output_dim=8, hidden_depth=2)
        else:
            raise NotImplementedError

    def forward(self, qpos, image, env_state, actions=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        is_training = actions is not None # train or val
        bs, _ = qpos.shape
        # Image observation features and position embeddings
        all_cam_features = []
        for cam_id, cam_name in enumerate(self.camera_names):
            features, pos = self.backbones[cam_id](image[:, cam_id])
            features = features[0] # take the last layer feature
            pos = pos[0] # not used
            all_cam_features.append(self.backbone_down_projs[cam_id](features))
        # flatten everything
        flattened_features = []
        for cam_feature in all_cam_features:
            flattened_features.append(cam_feature.reshape([bs, -1]))
        flattened_features = torch.cat(flattened_features, axis=1) # 768 each
        features = torch.cat([flattened_features, qpos], axis=1) # qpos: 14
        a_hat = self.mlp(features)
        return a_hat


def mlp(input_dim, hidden_dim, output_dim, hidden_depth):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    trunk = nn.Sequential(*mods)
    return trunk


def build_encoder(args):
    d_model = args.hidden_dim # 256
    dropout = args.dropout # 0.1
    nhead = args.nheads # 8
    dim_feedforward = args.dim_feedforward # 2048
    num_encoder_layers = args.enc_layers # 4 # TODO shared with VAE decoder
    normalize_before = args.pre_norm # False
    activation = "relu"

    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    return encoder


def build(args): # 核心模型部分 称为类VAE模型
    action_dim = args.action_dim #14 # TODO hardcode
    state_dim = args.state_dim
    use_gpos = args.use_gpos
    policy_class = args.policy_class
    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    # print(f"{args.camera_names=}")
    for _ in args.camera_names: # 不同视角用不同的backbone
        backbone = build_backbone(args)
        backbones.append(backbone)
        
    # backbone = build_backbone(args)
    # backbones.append(backbone)

    transformer = build_transformer(args)
        
    encoder = build_encoder(args)

    model = DETRVAE(
        backbones, # resnet18
        transformer, # 用来预测动作的
        encoder, # 用来计算 z 的
        state_dim=state_dim,
        # 输出层
        action_dim = action_dim,
        num_queries=args.num_queries, # 每个演示的步数
        camera_names=args.camera_names, # 相机名字
        use_language=args.use_language,
        use_film="film" in args.backbone,
        use_gpos=use_gpos,
        policy_class=policy_class,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model

def build_cnnmlp(args):
    state_dim = 8 #14 # TODO hardcode

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    for _ in args.camera_names:
        backbone = build_backbone(args)
        backbones.append(backbone)

    model = CNNMLP(
        backbones,
        state_dim=state_dim,
        camera_names=args.camera_names,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model

