import torch
import torch.nn as nn
import torch.nn.functional as F
from util.utils import OrthogonalInit
import math
class MappoPolicyNet(nn.Module):
    def __init__(self, alg_params):
        super(MappoPolicyNet, self).__init__()
        
        self.fc1 = nn.Linear(alg_params.policy_input_dim, alg_params.p_hid_dims[0])
        self.fc2 = nn.Linear(alg_params.p_hid_dims[0], alg_params.p_hid_dims[1])
        self.mu_head = nn.Linear(alg_params.p_hid_dims[1], alg_params.action_dim)
        self.log_std = nn.Parameter(torch.tensor([0] * alg_params.action_dim,
                                    dtype = torch.float))
        self.tanh = nn.Tanh()

        # orthogonal initialization
        if alg_params.use_orthogonal_init:
            nn.init.orthogonal_(self.fc1.weight, gain=nn.init.calculate_gain('tanh'))
            nn.init.zeros_(self.fc1.bias)
            nn.init.orthogonal_(self.fc2.weight, gain=nn.init.calculate_gain('tanh'))
            nn.init.zeros_(self.fc2.bias)
            nn.init.orthogonal_(self.mu_head.weight, gain=0.01)
            nn.init.zeros_(self.mu_head.bias)

        # -------- per-dimension action range --------
        # S server logits in [-1,1] + 3*ae_dim continuous (offl, trpw, comp)
        ae_dim = alg_params.action_encode_dim
        S = alg_params.action_dim - 3 * ae_dim
        low  = torch.tensor([-1.] * S + [0.6] * ae_dim + [0.6] * ae_dim + [0.6] * ae_dim)
        high = torch.tensor([1.] * S + [1.0] * ae_dim + [1.0] * ae_dim + [1.0] * ae_dim)
        self.register_buffer("act_low",  low)
        self.register_buffer("act_high", high)
        self.register_buffer("act_scale", (high - low) / 2.0)
        self.register_buffer("act_bias",  (high + low) / 2.0)

        self.LOG_STD_MIN = -5.0

        # Std annealing config
        self.LOG_STD_MAX_INIT = 0.5     # initial max log_std
        self.LOG_STD_MAX_FINAL = -1.5   # final max log_std

        self.total_episodes = alg_params.train_episodes
        self.std_anneal_tau = int((self.total_episodes) / 3)
        self.cur_episode = 0
        self.EPS = 1e-6

    def set_episode(self, episode_idx: int):
        """
        Call this once at the beginning of each episode
        """
        self.cur_episode = episode_idx

    def _cur_log_std_max(self, device):
        t = torch.tensor(float(self.cur_episode), device=device)
        tau = torch.tensor(float(self.std_anneal_tau), device=device)
        cur = self.LOG_STD_MAX_FINAL + (self.LOG_STD_MAX_INIT - self.LOG_STD_MAX_FINAL) * torch.exp(-t / tau)
        return cur

    def forward(self, obs):
        x = self.tanh(self.fc1(obs))
        x = self.tanh(self.fc2(x))
        # Generate the mean value
        mean = self.mu_head(x)
        cur_log_std_max = self._cur_log_std_max(device=mean.device)
        log_std = torch.clamp(self.log_std, min=self.LOG_STD_MIN, max=cur_log_std_max)
        # Generate the variance
        std = torch.exp(log_std).expand_as(mean)
        return mean, std

class MappoPolicyNetLSTM(nn.Module): 
    def __init__(self, alg_params):
        super(MappoPolicyNetLSTM, self).__init__()

        obs_dim = alg_params.policy_input_dim
        hid1, hid2 = alg_params.p_hid_dims
        act_dim = alg_params.action_dim
        # ---------- feature extraction ----------
        self.fc1 = nn.Linear(obs_dim, hid1)
        self.lstm = nn.LSTM(hid1, hid2, batch_first=True)
        self.mu_head = nn.Linear(hid2, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float))

        self.tanh = nn.Tanh()

        # ---------- init ----------
        if alg_params.use_orthogonal_init:
            nn.init.orthogonal_(self.fc1.weight, gain=nn.init.calculate_gain('tanh'))
            nn.init.zeros_(self.fc1.bias)
            for name, param in self.lstm.named_parameters():
                if "weight" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)
            nn.init.orthogonal_(self.mu_head.weight, gain=0.01)
            nn.init.zeros_(self.mu_head.bias)

        # ---------- action range ----------
        # S server logits in [-1,1] + 3*ae_dim continuous (offl, trpw, comp)
        ae_dim = alg_params.action_encode_dim
        S = alg_params.action_dim - 3 * ae_dim
        low  = torch.tensor([-1.] * S + [0.6] * ae_dim + [0.6] * ae_dim + [0.6] * ae_dim)
        high = torch.tensor([1.] * S + [1.0] * ae_dim + [1.0] * ae_dim + [1.0] * ae_dim)
        self.register_buffer("act_low",  low)
        self.register_buffer("act_high", high)
        self.register_buffer("act_scale", (high - low) / 2.0)
        self.register_buffer("act_bias",  (high + low) / 2.0)

        self.LOG_STD_MIN = -5.0

        # Std annealing config
        self.LOG_STD_MAX_INIT = 0.5     # initial max log_std
        self.LOG_STD_MAX_FINAL = -1.5   # final max log_std
        # One episode has about 600 env steps
        self.total_episodes = alg_params.train_episodes
        self.std_anneal_tau = int((self.total_episodes) / 3)
        self.cur_episode = 0
        self.EPS = 1e-6

    def set_episode(self, episode_idx: int):
        """
        Call this once at the beginning of each episode
        """
        self.cur_episode = episode_idx
        # print(f"[DEBUG] set_episode: {episode_idx}")

    def _cur_log_std_max(self, device):
        t = torch.tensor(float(self.cur_episode), device=device)
        tau = torch.tensor(float(self.std_anneal_tau), device=device)
        cur = self.LOG_STD_MAX_FINAL + (self.LOG_STD_MAX_INIT - self.LOG_STD_MAX_FINAL) * torch.exp(-t / tau)
        return cur

    def forward(self, obs, h_in=None):
        """
        obs: [B, obs_dim] or [B, T, obs_dim]
        h_in: (h0, c0)
              h0,c0: [1, B, hidden_dim]
        return:
            mean: [B, T, act_dim] or [B, act_dim]
            std:  same shape as mean
            h_out: (h1, c1)
        """

        single_step = False
        if obs.dim() == 2:  # [B, obs_dim]
            single_step = True
            obs = obs.unsqueeze(1)  # [B,1,D]

        x = self.tanh(self.fc1(obs))           # [B,T,hid1]
        out, (h_n, c_n) = self.lstm(x, h_in)   # [B,T,hid2], ([1,B,hid2],[1,B,hid2])
        x = self.tanh(out)
        # Generate the mean value
        mean = self.mu_head(x)                 # [B,T,act_dim]
        # Generate the variance
        cur_log_std_max = self._cur_log_std_max(device=mean.device)
        log_std = torch.clamp(self.log_std, self.LOG_STD_MIN, cur_log_std_max)
        std = log_std.exp().expand_as(mean)
        if single_step:
            mean = mean.squeeze(1)
            std = std.squeeze(1)
        
        return mean, std, (h_n, c_n)

class MaddpgPolicyNet(nn.Module):
    def __init__(self, alg_params):
        super(MaddpgPolicyNet, self).__init__()
        
        self.fc1 = nn.Linear(alg_params.obs_dim, alg_params.p_hid_dims[0])
        self.fc2 = nn.Linear(alg_params.p_hid_dims[0], alg_params.p_hid_dims[1])
        self.fc3 = nn.Linear(alg_params.p_hid_dims[1], alg_params.action_dim)
        self.tanh = nn.Tanh()
        
        # orthogonal initialization
        if alg_params.use_orthogonal_init:
            OrthogonalInit(self.fc1)
            OrthogonalInit(self.fc2)
            OrthogonalInit(self.fc3, gain = 0.01)
    
    def forward(self, obs):
        x = self.tanh(self.fc1(obs))
        x = self.tanh(self.fc2(x))
        act = self.tanh(self.fc3(x)) + 1
        
        return act


class MaddpgPolicyNetLSTM(nn.Module):
    def __init__(self, alg_params):
        super(MaddpgPolicyNetLSTM, self).__init__()
        
        obs_dim = alg_params.policy_input_dim
        hid1, hid2 = alg_params.p_hid_dims
        act_dim = alg_params.action_dim
        self.fc1 = nn.Linear(obs_dim, hid1)
        self.lstm = nn.LSTM(hid1, hid2, batch_first=True)
        self.fc3 = nn.Linear(hid2, act_dim)
        self.tanh = nn.Tanh()
        
        # orthogonal initialization
        if alg_params.use_orthogonal_init:
            nn.init.orthogonal_(self.fc1.weight, gain=nn.init.calculate_gain('tanh'))
            nn.init.zeros_(self.fc1.bias)
            for name, param in self.lstm.named_parameters():
                if "weight" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)
            nn.init.orthogonal_(self.fc3.weight, gain=0.01)
            nn.init.zeros_(self.fc3.bias)

        # ---------- action range ----------
        # S server logits in [-1,1] + 3*ae_dim continuous (offl, trpw, comp)
        ae_dim = alg_params.action_encode_dim
        S = alg_params.action_dim - 3 * ae_dim
        low  = torch.tensor([-1.] * S + [0.6] * ae_dim + [0.6] * ae_dim + [0.6] * ae_dim)
        high = torch.tensor([1.] * S + [1.0] * ae_dim + [1.0] * ae_dim + [1.0] * ae_dim)
        self.register_buffer("act_low",  low)
        self.register_buffer("act_high", high)
        self.register_buffer("act_scale", (high - low) / 2.0)
        self.register_buffer("act_bias",  (high + low) / 2.0)


    def forward(self, obs, h_in=None):

        single_step = False
        if obs.dim() == 2:  # [B, obs_dim]
            single_step = True
            obs = obs.unsqueeze(1)  # [B,1,D]
        x = self.tanh(self.fc1(obs))
        out, (h_n, c_n) = self.lstm(x, h_in)
        x = self.tanh(out)
        act = self.tanh(self.fc3(x))
        return act, (h_n, c_n)