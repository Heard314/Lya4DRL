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
            self.std = np.zeros_like(x)
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)
            
class ObsScaling():
    def __init__(self, gen_params, alg_params, clip_obs=5.0, eps=1e-8, vq_clip=1e6):
        self.gen_params = gen_params
        self.alg_params = alg_params
        self.device_num = gen_params.device_num
        self.device_obs_dim = alg_params.device_obs_dim
        self.device_type_num = gen_params.device_type_num
        self.edge_queue_obs_dim = alg_params.edge_queue_obs_dim
        self.edge_rms = RunningMeanStd(self.edge_queue_obs_dim)

        self.clip_obs = clip_obs
        self.eps = eps
        self.vq_clip = vq_clip

    # device_obss: S x trans_rates, local_queue, local_vir_queue, task_data_size, task_comp_dens, task_dly_cons
    # edge_obs: edge_queue, vir_edge_queue for all queues
    def __call__(self, edge_obs, device_obss):
        S = self.gen_params.edge_server_num
        for i in range(self.device_num):
            for j in range(S):
                device_obss[i][j] /= 10
            device_obss[i][S] = np.log1p(np.clip(device_obss[i][S], 0.0, self.vq_clip))
            device_obss[i][S + 1] = np.log1p(np.clip(device_obss[i][S + 1], 0.0, self.vq_clip))

        return edge_obs, device_obss

                
class RewardScaling():
    def __init__(self, gen_params, alg_params, dim=3):
        # discount factor
        self.gamma = alg_params.gamma
        self.train_mode = gen_params.train_mode
        self.dim = dim
        self.R = [0.0 for _ in range(dim)]
        self.r_running_ms = [RunningMeanStd(1) for _ in range(dim)]

    def __call__(self, rewards):
        if self.train_mode == "mappo":
            if isinstance(rewards, (list, tuple, np.ndarray, torch.Tensor)):
                for i in range(len(self.R)):
                    self.R[i] = self.gamma * self.R[i] + rewards[i]
                    self.r_running_ms[i].update(self.R[i])
                rewards = [float(rewards[i] / (self.r_running_ms[i].std + 1e-8)) for i in range(len(rewards))]
            else:
                self.R[0] = self.gamma * self.R[0] + float(rewards)
                self.r_running_ms[0].update(self.R[0])
                rewards = float(rewards / (self.r_running_ms[0].std + 1e-8))
        elif self.train_mode == "maddpg":
            reward_scale = 1e-4
            if isinstance(rewards, (list, tuple, np.ndarray)):
                rewards = [ri * reward_scale for ri in rewards]
            else:
                rewards = float(rewards) * reward_scale
        return rewards
        
    # reset 'R' when an episode is done
    def reset(self):
        self.R  = [0.0 for _ in range(self.dim)]

class GaussianNoise:
    def __init__(self, action_dim, sigma=0.2, device="cpu"):
        self.action_dim = action_dim
        self.sigma = sigma
        self.device = device

    def sample(self, sigma=None):
        s = self.sigma if sigma is None else sigma
        return torch.randn(self.action_dim, device=self.device) * s


def OrthogonalInit(layer, gain = 1.0):
    for name, params in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(params, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(params, gain = gain)
            
def to_lstm_hidden(h, batch_size, hid_dim):
    """Normalize LSTM hidden state to shape [1, batch_size, hid_dim]."""
    if h is None:
        return torch.zeros(1, batch_size, hid_dim)
    if isinstance(h, list):
        h = torch.tensor(h, dtype=torch.float32)
    if h.dim() == 1:
        h = h.view(1, 1, -1)
    elif h.dim() == 2:
        h = h.unsqueeze(1)
    return h


def GetPolicyInputs(obs):
    inputs = torch.tensor(obs, dtype = torch.float).reshape([1, -1])

    return inputs

# concatenate 1D array for 'numpy array' or 'list'
def concatenate(a, b, dtype=np.float32):
    """
    Concatenate two 1D arrays/lists into one 1D numpy array.

    Supports:
      - Python list/tuple
      - numpy.ndarray
    Returns:
      - Python list (1D)
    """
    a = np.asarray(a, dtype=dtype).reshape(-1)
    b = np.asarray(b, dtype=dtype).reshape(-1)
    return np.concatenate([a, b], axis=0).tolist()

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
