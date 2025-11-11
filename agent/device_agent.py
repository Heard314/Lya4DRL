from abc import abstractmethod
import numpy as np
import torch
from torch.distributions import Normal
from network.policy_net import MappoPolicyNet, MaddpgPolicyNet
from util.utils import GetPolicyInputs, GaussianNoise
import math
class MappoDeviceAgent():
    def __init__(self, agent_id, gen_params, alg_params):
        # agent id
        self.agent_id = agent_id
        
        # policy network
        self.p_net = MappoPolicyNet(alg_params)
        
        self.evaluate = gen_params.evaluate

        self.action_dim = alg_params.action_dim
        
    def choose_action(self, obs, active: bool = True):
        if not active:
            return [0.0] * self.action_dim, 0.0

        p_inputs = GetPolicyInputs(obs)
        with torch.no_grad():
            mean, std = self.p_net(p_inputs)
        if self.evaluate:
            act = mean.squeeze(dim = 0).tolist()
            act_logprob = None
        else:
            dist = Normal(mean, std)
            act = dist.sample()
            act = torch.clamp(act, 0, 10)
            act_logprob = dist.log_prob(act).sum(-1)
            act = act.squeeze(dim = 0).tolist()
            act_logprob = float(act_logprob)
            
        return act, act_logprob
    
    # def choose_action(self, obs):
    #     p_inputs = GetPolicyInputs(obs)
    #     with torch.no_grad():
    #         mean, std = self.p_net(p_inputs)
    #     if self.evaluate:
    #         # 确定性：用均值 -> tanh -> 映射到 [0,10]
    #         u = mean
    #         a = torch.tanh(u)
    #         action = a * 5.0 + 5.0    # [0,10]
    #         act = action.squeeze(0).tolist()
    #         act_logprob = None
    #     else:
    #         # 2) 采样未压缩动作
    #         dist = Normal(mean, std)
    #         u = dist.rsample()        # 重参数化

    #         # 3) tanh 压缩 & 线性映射到 [0,10]
    #         a = torch.tanh(u)
    #         action = a * 5.0 + 5.0

    #         # 4) 正确的 log_prob（高斯 + tanh 的雅可比 + 线性缩放的雅可比）
    #         normal_logp = (-0.5 * (
    #             ((u - mean) / (std + 1e-6))**2 +
    #             2*torch.log(std + 1e-6) +
    #             math.log(2*math.pi)
    #         )).sum(-1)
    #         squash = torch.log(1 - a.pow(2) + 1e-6).sum(-1)
    #         scale  = math.log(5.0) * mean.shape[-1]  # 每维 scale=5，求和后是常数
    #         logp = normal_logp - squash - scale

    #         act = action.squeeze(0).tolist()
    #         act_logprob = float(logp)

    #     return act, act_logprob

    def update_net(self, params):
        self.p_net.load_state_dict(params)
        
    def load_net(self, path):
        self.update_net(torch.load(path))
        
class MaddpgDeviceAgent():
    def __init__(self, agent_id, gen_params, alg_params):
        # agent id
        self.agent_id = agent_id
        
        # policy network
        self.p_net = MaddpgPolicyNet(alg_params)
        
        # action noise
        self.use_action_noise = alg_params.use_action_noise
        if self.use_action_noise:
            self.action_noise = GaussianNoise(alg_params.action_dim)
            
        self.evaluate = gen_params.evaluate
        
    def choose_action(self, obs):
        p_inputs = GetPolicyInputs(obs)
        with torch.no_grad():
            act = self.p_net(p_inputs).squeeze(0).tolist()
        if not self.evaluate:
            act = np.clip((act + self.action_noise.sample()), 0, 2).tolist()
        
        return act
        
    def update_net(self, params):
        self.p_net.load_state_dict(params)
        
    def load_net(self, path):
        self.update_net(torch.load(path))

class StaticDeviceAgent():
    def __init__(self, agent_id, gen_params):
        # agent id
        self.agent_id = agent_id
        self.max_task_num = gen_params.max_task_num
    
    @abstractmethod
    def choose_action(self):
        pass
    
class LocalComputingDeviceAgent(StaticDeviceAgent):
    def __init__(self, agent_id, gen_params):
        super().__init__(agent_id, gen_params)
        
    def choose_action(self):
        act = [0 for i in range(self.max_task_num + 1)]
        
        return act

class EdgeComputingDeviceAgent(StaticDeviceAgent):
    def __init__(self, agent_id, gen_params):
        super().__init__(agent_id, gen_params)
        
    def choose_action(self):
        act = [1 for i in range(self.max_task_num + 1)]
        
        return act

class RandomComputingDeviceAgent(StaticDeviceAgent):
    def __init__(self, agent_id, gen_params):
        super().__init__(agent_id, gen_params)
        
    def choose_action(self):
        act = [np.random.uniform(0, 1) for i in range(self.max_task_num + 1)]
        
        return act