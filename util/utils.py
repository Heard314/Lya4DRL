import os
import numpy as np
import torch
import torch.nn as nn

'''calculate mean and std dynamically'''
class RunningMeanStd():
    def __init__(self, shape):
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)
            
    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)
            
class ObsScaling():
    def __init__(self, gen_params, alg_params, max_task_num, max_data_size, max_comp_dens, std_comp_freq):
        self.max_task_num = max_task_num
        # unit: Mb
        self.max_data_size = max_data_size
        # unit: Gcycles/Mb
        self.max_comp_dens = max_comp_dens
        # unit: Gcycles/s 
        self.std_comp_freq = std_comp_freq
        self.gen_params = gen_params
        self.alg_params = alg_params
        # unit: s
        # self.max_dly_cons = self.max_data_size * self.max_comp_dens \
        #                     / self.std_comp_freq
        #! 最大延时设置为1s
        self.max_dly_cons = 1
        self.device_type_num = gen_params.device_type_num
        self.device_num = gen_params.device_num
        
    def __call__(self, edge_obs, device_obss):
        for i in range(len(edge_obs)):
            edge_obs[i] = np.clip(edge_obs[i], 0, 20) / 10
        for i in range(len(device_obss)):
            device_obss[i][self.device_type_num] = np.clip(device_obss[i][0], 0, 20) / 10
            device_obss[i][self.device_type_num+1] = np.clip(device_obss[i][1], 0, 20) / 10
            for j in range(self.max_task_num):
                device_obss[i][self.device_type_num+2 + j * 3] /= self.max_data_size
                device_obss[i][self.device_type_num+2 + j * 3 + 1] /= self.max_comp_dens
                device_obss[i][self.device_type_num+2 + j * 3 + 2] /= self.max_dly_cons
                
class RewardScaling():
    def __init__(self, gamma):
        # discount factor
        self.gamma = gamma
        self.R = 0
        self.r_running_ms = RunningMeanStd(1)

    def __call__(self, reward):
        self.R = self.gamma * self.R + reward
        self.r_running_ms.update(self.R)
        reward = float(reward / (self.r_running_ms.std + 1e-8))
        
        return reward
        
    # reset 'R' when an episode is done 
    def reset(self):
        self.R = 0
        
class GaussianNoise():
    def __init__(self, action_dim, mu = 0.25, sigma = 0.5):
        self.action_dim = action_dim
        self.mu = mu
        self.sigma = sigma
        
    def sample(self):
        x = np.random.normal(self.mu, self.sigma, self.action_dim)
                
        return x

def OrthogonalInit(layer, gain = 1.0):
    for name, params in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(params, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(params, gain = gain)
            
def GetPolicyInputs(obs):
    inputs = torch.tensor(obs, dtype = torch.float).reshape([1, -1])
    
    return inputs

def GetValueInputs(edge_obs, device_obss):
    inputs = []
    
    inputs += edge_obs
    for i in range(len(device_obss)):
        inputs += device_obss[i]
    inputs = torch.tensor(inputs, dtype = torch.float).reshape([1, -1])
    
    return inputs

import matplotlib
matplotlib.use("Agg")  # use non-GUI backend
import matplotlib.pyplot as plt

def save_device_hist_plots(device_overtime_nums,
                           device_comp_dlys,
                           e_id,
                           out_dir="runs/plot/"):
    """
    Draw bar-like histograms for all devices and save to disk.

    Args:
        device_overtime_nums: list or 1D array, overtime metric per device
        device_comp_dlys:     list or 1D array, delay metric per device
        e_id:                 current episode id (used in filename)
        out_dir:              directory to save figures
    """
    os.makedirs(out_dir, exist_ok=True)

    # Convert to numpy array for convenience
    overtime = np.asarray(device_overtime_nums, dtype=float)
    delays   = np.asarray(device_comp_dlys, dtype=float)

    num_devices = len(overtime)
    x_idx = np.arange(num_devices)  # 0,1,2,... as device index

    # --------- Plot 1: overtime per device ---------
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    ax1.bar(x_idx, overtime)
    ax1.set_xlabel("Device index")
    ax1.set_ylabel("Timeout task count")
    ax1.set_title(f"Timeout task count of all devices (episode {e_id})")
    ax1.set_xticks(x_idx)  # show all device indices

    save_path1 = os.path.join(out_dir, f"timeout_all_device_ep_{e_id}.png")
    fig1.tight_layout()
    fig1.savefig(save_path1)
    plt.close(fig1)

    # --------- Plot 2: computation delay per device ---------
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.bar(x_idx, delays)
    ax2.set_xlabel("Device index")
    ax2.set_ylabel("Computation delay(s)")
    ax2.set_title(f"Computation delay of all devices (episode {e_id})")
    ax2.set_xticks(x_idx)

    save_path2 = os.path.join(out_dir, f"comp_dly_all_device_ep_{e_id}.png")
    fig2.tight_layout()
    fig2.savefig(save_path2)
    plt.close(fig2)
