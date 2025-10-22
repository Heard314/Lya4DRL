import torch
import torch.nn as nn
from util.utils import OrthogonalInit
import math
class MappoPolicyNet(nn.Module):
    def __init__(self, alg_params):
        super(MappoPolicyNet, self).__init__()
        
        self.fc1 = nn.Linear(alg_params.obs_dim, alg_params.p_hid_dims[0])
        self.fc2 = nn.Linear(alg_params.p_hid_dims[0], alg_params.p_hid_dims[1])
        self.fc3 = nn.Linear(alg_params.p_hid_dims[1], alg_params.action_dim)
        self.tanh = nn.Tanh()
            
        self.log_std = nn.Parameter(torch.tensor([0] * alg_params.action_dim,
                                    dtype = torch.float))
        
        # orthogonal initialization
        if alg_params.use_orthogonal_init:
            OrthogonalInit(self.fc1)
            OrthogonalInit(self.fc2)
            OrthogonalInit(self.fc3, gain = 0.01)
    # def __init__(self, alg_params):
    #     super(MappoPolicyNet, self).__init__()
        
    #     self.fc1 = nn.Linear(alg_params.obs_dim, alg_params.p_hid_dims[0])
    #     self.fc2 = nn.Linear(alg_params.p_hid_dims[0], alg_params.p_hid_dims[1])
    #     self.mu_head = nn.Linear(alg_params.p_hid_dims[1], alg_params.action_dim)
    #     self.log_std = nn.Parameter(torch.tensor([0] * alg_params.action_dim,
    #                                 dtype = torch.float))
    #     self.tanh = nn.Tanh()

    #     # orthogonal initialization
    #     if alg_params.use_orthogonal_init:
    #         nn.init.orthogonal_(self.fc1.weight, gain=nn.init.calculate_gain('tanh'))
    #         nn.init.zeros_(self.fc1.bias)
    #         nn.init.orthogonal_(self.fc2.weight, gain=nn.init.calculate_gain('tanh'))
    #         nn.init.zeros_(self.fc2.bias)
    #         nn.init.orthogonal_(self.mu_head.weight, gain=0.01)
    #         nn.init.zeros_(self.mu_head.bias)

    #     # action range
    #     low = torch.zeros(alg_params.action_dim)
    #     high = torch.full((alg_params.action_dim,), 10.) # 10
    #     self.register_buffer("act_low",  low)
    #     self.register_buffer("act_high", high)
    #     self.register_buffer("act_scale", (high - low) / 2.0)
    #     self.register_buffer("act_bias",  (high + low) / 2.0)

    #     self.LOG_STD_MIN, self.LOG_STD_MAX = -20.0, 2
    #     self.EPS = 1e-6
    
    def forward(self, obs):
        x = self.tanh(self.fc1(obs))
        x = self.tanh(self.fc2(x))
        # 平均值生成
        mean = self.tanh(self.fc3(x)) * 5 + 5
        # 方差生成
        log_std = self.log_std.expand_as(mean)
        std = torch.exp(log_std)
        
        return mean, std

    # def forward(self, obs):
    #     x = self.tanh(self.fc1(obs))
    #     x = self.tanh(self.fc2(x))
    #     # 平均值生成
    #     mean = self.mu_head(x) # 暂不扩展到动作空间维度
    #     log_std = self.clamp(self.log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)        
    #     # 方差生成
    #     std = torch.exp(log_std).expand_as(mean)
    #     return mean, std
    
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