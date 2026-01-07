from abc import abstractmethod
import numpy as np
import torch
from torch.distributions import Normal
from network.policy_net import MappoPolicyNet, MaddpgPolicyNet, MappoPolicyNetLSTM
from util.utils import GetPolicyInputs, GaussianNoise
import math
import config.global_params as gp

class MappoDeviceAgent():
    def __init__(self, agent_id, gen_params, alg_params):
        # agent id
        self.agent_id = agent_id
        
        # policy network
        # self.p_net = MappoPolicyNet(alg_params)
        self.p_net = MappoPolicyNetLSTM(alg_params)
        
        self.evaluate = gen_params.evaluate
        self.action_dim = alg_params.action_dim
        self.lstm_hidden_dim = alg_params.p_hid_dims[1]

        #OU exploration params
        self.use_ou_noise = gen_params.use_ou_noise
        self.ou_theta = gen_params.ou_theta  # 越大回归越快，相关性越弱
        self.ou_sigma = gen_params.ou_sigma   # 越大噪声越强
        self.ou_dt    = gen_params.ou_dt
        # 额外缩放
        self.ou_scale = gen_params.ou_scale

        # OU state: 存在 eps 空间，shape = [B, action_dim]
        self._ou_state = None

        # 连续 OU 的稳态方差：sigma^2 / (2*theta)
        # 用它做标准化，让 eps 边际更接近 N(0,1)，再乘 ou_scale 控制强度
        self._ou_stationary_std = math.sqrt((self.ou_sigma ** 2) / (2.0 * self.ou_theta + 1e-8) + 1e-8)
    
    def reset_ou(self):
        # 每个episode开始时调用一次，避免跨 episode 噪声漂移。
        self._ou_state = None

    def _ou_eps(self, device, batch_size, dim):
        # 生成 OU 噪声 eps（时间相关），并标准化到近似 N(0,1)。
        if (self._ou_state is None) or (self._ou_state.shape[0] != batch_size) or (self._ou_state.shape[1] != dim):
            self._ou_state = torch.zeros(batch_size, dim, device=device)

        z = torch.randn(batch_size, dim, device=device)  # OU 的驱动噪声
        self._ou_state = (
            self._ou_state
            + self.ou_theta * (0.0 - self._ou_state) * self.ou_dt
            + self.ou_sigma * math.sqrt(self.ou_dt) * z
        )

        # 标准化：让边际更接近标准正态，再用 ou_scale 控制探索强度
        eps = (self._ou_state / (self._ou_stationary_std + 1e-8)) * self.ou_scale
        return eps

    def choose_action(self, obs, lstm_hidden_h, lstm_hidden_c, active: bool = True):

        enable_print = gp.settings.enable_print

        p_inputs = GetPolicyInputs(obs)

        # process the lstm hidden state
        hid_dim = self.p_net.lstm.hidden_size
        batch_size = p_inputs.size(0)  
        def to_hidden(h):
            # 如果是 None，就初始化为 0
            if h is None:
                return torch.zeros(1, batch_size, hid_dim)
            # 如果是 list -> 转成 tensor
            if isinstance(h, list):
                h = torch.tensor(h, dtype=torch.float32)
            # 如果是一维 [hid_dim] -> [1,1,hid_dim]
            if h.dim() == 1:
                h = h.view(1, 1, -1)
            # 如果是二维 [1,hid_dim] -> [1,1,hid_dim]
            elif h.dim() == 2:
                h = h.unsqueeze(1)
            # 如果已经是 [1,B,hid_dim] -> 不动
            return h

        lstm_hidden_h = to_hidden(lstm_hidden_h)
        lstm_hidden_c = to_hidden(lstm_hidden_c)

        with torch.no_grad():
            mean, std, (next_lstm_hidden_h, next_lstm_hidden_c) = self.p_net(p_inputs, (lstm_hidden_h, lstm_hidden_c))
        
        # if enable_print: print(f"[DEBUG] the p_net output: mean({mean}), std({std})")

        if not active:
            return [0.0] * self.action_dim, 0.0, next_lstm_hidden_h, next_lstm_hidden_c

        # 数值稳定：避免 std 太小
        std = torch.clamp(std, min=1e-6)

        # ===== 逐维映射参数 =====
        # dim0 -> [0,10]: scale=5, loc=5
        # dim1 -> [6,10]: scale=2, loc=8
        # dim2 -> [6,10]: scale=2, loc=8
        scale = self.p_net.act_scale.expand_as(mean)
        loc   = self.p_net.act_bias.expand_as(mean)

        if self.evaluate:
            # 确定性：用均值 -> tanh -> 分维线性映射
            u = mean
            a = torch.tanh(u)
            action = a * scale + loc
            act = action.squeeze(0).tolist()
            act_logprob = None
        else:
            # OU exploration: 用 OU 生成时间相关 eps，替代 i.i.d. eps
            if self.use_ou_noise:
                eps = self._ou_eps(device=mean.device, batch_size=batch_size, dim=mean.size(-1))
            else:
                eps = torch.randn_like(mean)

            # reparameterized sample in eps-space (更平滑)
            u = mean + std * eps

            a = torch.tanh(u)
            action = a * scale + loc

            # log_prob：仍按 Normal(mean,std) + tanh + scale 的雅可比计算
            # 用 dist.log_prob(u) 更干净
            dist = Normal(mean, std)
            normal_logp = dist.log_prob(u).sum(-1)
            squash = torch.log(1 - a.pow(2) + 1e-6).sum(-1)
            scale_logsum = torch.log(scale).sum(-1)
            logp = normal_logp - squash - scale_logsum

            act = action.squeeze(0).tolist()
            act_logprob = float(logp)

        return act, act_logprob, next_lstm_hidden_h, next_lstm_hidden_c

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