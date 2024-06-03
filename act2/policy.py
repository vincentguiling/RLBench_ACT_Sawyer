import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

from detr.models.backbone import FilMedBackbone
from detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer
import IPython
e = IPython.embed

# from utils import get_gpu_mem_info

# def print_gpu_mem():
#     gpu_mem_total, gpu_mem_used, gpu_mem_free = get_gpu_mem_info()
#     return (gpu_mem_used/gpu_mem_total)*100

class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        # print(f'KL Weight {self.kl_weight}')

    def __call__(self, qpos, gpos, image, history_images=None, history_action=None, is_pad_history=None, actions=None, is_pad_action=None, command_embedding=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            history_images = history_images[:, :self.model.num_queries]
            history_action = history_action[:, :self.model.num_queries]
            is_pad_history = is_pad_history[:, :self.model.num_queries]
            
            actions = actions[:, :self.model.num_queries] # 如果输入的actions queries没有10个怎么办[有pad补充了]
            is_pad_action = is_pad_action[:, :self.model.num_queries]
            
            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, gpos, image, env_state, history_images,
                                                         history_action, is_pad_history=is_pad_history, 
                                                         actions=actions, is_pad_action=is_pad_action, 
                                                         command_embedding=command_embedding) 
                
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            l1 = (all_l1 * ~is_pad_action.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1

            if mu != None:
                total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar) # 如果没有编码器，注释掉
                loss_dict['kl'] = total_kld[0] # 如果没有编码器，注释掉
                loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight #   * self.kl_weight 
            else:
                loss_dict['loss'] = loss_dict['l1']
            
            return loss_dict
        else: # inference time
            
            is_pad_history = is_pad_history[:self.model.num_queries]
            history_action = history_action[:self.model.num_queries]
                     
            a_hat, _, (_, _) , image_feature = self.model(qpos, gpos, image, env_state, history_images, 
                                          history_action, is_pad_history=is_pad_history, 
                                          actions=None, is_pad_action=is_pad_action, 
                                          command_embedding=command_embedding) # no action, sample from prior  
            return a_hat, image_feature

    def configure_optimizers(self):
        return self.optimizer


class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model # decoder
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None # TODO
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict['mse'] = mse
            loss_dict['loss'] = loss_dict['mse']
            return loss_dict
        else: # inference time
            a_hat = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
