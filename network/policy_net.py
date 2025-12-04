import torch
import torch.nn as nn
import torch.nn.functional as F
from util.utils import OrthogonalInit
import math
class MappoPolicyNet(nn.Module):
    # def __init__(self, alg_params):
    #     super(MappoPolicyNet, self).__init__()
        
    #     self.fc1 = nn.Linear(alg_params.obs_dim, alg_params.p_hid_dims[0])
    #     self.fc2 = nn.Linear(alg_params.p_hid_dims[0], alg_params.p_hid_dims[1])
    #     self.fc3 = nn.Linear(alg_params.p_hid_dims[1], alg_params.action_dim)
    #     self.tanh = nn.Tanh()
            
    #     self.log_std = nn.Parameter(torch.tensor([0] * alg_params.action_dim,
    #                                 dtype = torch.float))
        
    #     # orthogonal initialization
    #     if alg_params.use_orthogonal_init:
    #         OrthogonalInit(self.fc1)
    #         OrthogonalInit(self.fc2)
    #         OrthogonalInit(self.fc3, gain = 0.01)
    def __init__(self, alg_params):
        super(MappoPolicyNet, self).__init__()
        
        self.fc1 = nn.Linear(alg_params.obs_dim, alg_params.p_hid_dims[0])
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
        # 维度0: [0,10]  维度1: [6,10]  维度2: [6,10]
        # low = torch.zeros(alg_params.action_dim)
        # high = torch.full((alg_params.action_dim,), 10.) # 10
        low  = torch.tensor([0., 6., 6.])
        high = torch.tensor([10., 10., 10.])
        # assert low.numel() == alg_params.action_dim and high.numel() == alg_params.action_dim, \
        #     "low/high 的长度应与 action_dim 一致"
        self.register_buffer("act_low",  low)
        self.register_buffer("act_high", high)
        self.register_buffer("act_scale", (high - low) / 2.0)
        self.register_buffer("act_bias",  (high + low) / 2.0)

        self.LOG_STD_MIN, self.LOG_STD_MAX = -20.0, 2.0
        self.EPS = 1e-6
    
    # def forward(self, obs):
    #     x = self.tanh(self.fc1(obs))
    #     x = self.tanh(self.fc2(x))
    #     # 平均值生成
    #     mean = self.tanh(self.fc3(x)) * 5 + 5
    #     # 方差生成
    #     log_std = self.log_std.expand_as(mean)
    #     std = torch.exp(log_std)
        
    #     return mean, std

    def forward(self, obs):
        assert(False) #正在测试LSTM网络
        x = self.tanh(self.fc1(obs))
        x = self.tanh(self.fc2(x))
        # 平均值生成
        mean = self.mu_head(x) # 暂不扩展到动作空间维度
        log_std = torch.clamp(self.log_std, min=self.LOG_STD_MIN, max=self.LOG_STD_MAX)       
        # 方差生成
        std = torch.exp(log_std).expand_as(mean)
        return mean, std

class MappoPolicyNetLSTM(nn.Module): 
    def __init__(self, alg_params):
        super(MappoPolicyNetLSTM, self).__init__()

        obs_dim = alg_params.obs_dim
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
        low  = torch.tensor([0., 6., 6.])
        high = torch.tensor([10., 10., 10.])
        self.register_buffer("act_low",  low)
        self.register_buffer("act_high", high)
        self.register_buffer("act_scale", (high - low) / 2.0)
        self.register_buffer("act_bias",  (high + low) / 2.0)

        self.LOG_STD_MIN, self.LOG_STD_MAX = -20.0, 2.0
        self.EPS = 1e-6

        # ---------- debug flag ----------
        # Set this to False if you want to disable NaN/Inf checks
        self.debug_nan = False
        # print(f"The debug_nan is {self.debug_nan}")

    def _check_tensor(self, name, t):
        """Check whether a tensor contains NaN or Inf values."""
        if t is None:
            return
        if not torch.is_tensor(t):
            return
        has_nan = torch.isnan(t).any()
        has_inf = torch.isinf(t).any()
        if has_nan or has_inf:
            print(f"[NaN/Inf Detect] {name} has invalid values.")
            print(f"  shape: {tuple(t.shape)}")
            print(f"  has_nan: {bool(has_nan)}, has_inf: {bool(has_inf)}")
            # print a small sample for inspection
            flat = t.detach().reshape(-1).cpu()
            print("  sample values:", flat[:10])
            raise ValueError(f"NaN/Inf detected in {name}")

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
        if self.debug_nan:
            self._check_tensor("obs(input)", obs)
            if h_in is not None:
                h0, c0 = h_in
                self._check_tensor("h_in[0] (h0)", h0)
                self._check_tensor("h_in[1] (c0)", c0)

        single_step = False
        if obs.dim() == 2:  # [B, obs_dim]
            single_step = True
            obs = obs.unsqueeze(1)  # [B,1,D]

        if self.debug_nan:
            self._check_tensor("obs(after unsqueeze)", obs)

        x = self.tanh(self.fc1(obs))           # [B,T,hid1]

        if self.debug_nan:
            self._check_tensor("x(after fc1+tanh)", x)

        out, (h_n, c_n) = self.lstm(x, h_in)   # [B,T,hid2], ([1,B,hid2],[1,B,hid2])

        if self.debug_nan:
            self._check_tensor("out(after lstm)", out)
            self._check_tensor("h_n(output h)", h_n)
            self._check_tensor("c_n(output c)", c_n)

        x = self.tanh(out)
        if self.debug_nan:
            self._check_tensor("x(after lstm+tanh)", x)

        mean = self.mu_head(x)                 # [B,T,act_dim]
        if self.debug_nan:
            self._check_tensor("mean(before squeeze)", mean)

        log_std = torch.clamp(self.log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp().expand_as(mean)
        if self.debug_nan:
            self._check_tensor("log_std(clamped)", log_std)
            self._check_tensor("std(expanded)", std)

        if single_step:
            mean = mean.squeeze(1)
            std = std.squeeze(1)
            if self.debug_nan:
                self._check_tensor("mean(single_step)", mean)
                self._check_tensor("std(single_step)", std)

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