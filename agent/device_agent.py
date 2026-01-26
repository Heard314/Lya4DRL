from abc import abstractmethod
import numpy as np
import torch
from torch.distributions import Normal
from network.policy_net import MaddpgPolicyNetLSTM, MappoPolicyNet, MaddpgPolicyNet, MappoPolicyNetLSTM
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
        # ou noise
        self.ou_scale = gen_params.ou_scale
        # OU state, shape = [B, action_dim]
        self._ou_state = None
        self._ou_stationary_std = math.sqrt((self.ou_sigma ** 2) / (2.0 * self.ou_theta + 1e-8) + 1e-8)
    
    def reset_ou(self):
        # Called once at the beginning of each episode to avoid noise drift across episodes
        self._ou_state = None

    def _ou_eps(self, device, batch_size, dim):
        # Generate OU noise eps (time-correlated) and normalize it to approximately N(0, 1)
        if (self._ou_state is None) or (self._ou_state.shape[0] != batch_size) or (self._ou_state.shape[1] != dim):
            self._ou_state = torch.zeros(batch_size, dim, device=device)

        z = torch.randn(batch_size, dim, device=device)  # OU noise
        self._ou_state = (
            self._ou_state
            + self.ou_theta * (0.0 - self._ou_state) * self.ou_dt
            + self.ou_sigma * math.sqrt(self.ou_dt) * z
        )

        # Normalization makes the marginal distribution closer to standard normal, then ou_scale controls exploration strength
        eps = (self._ou_state / (self._ou_stationary_std + 1e-8)) * self.ou_scale
        return eps

    def choose_action(self, obs, lstm_hidden_h, lstm_hidden_c, active: bool = True):

        enable_print = gp.settings.enable_print

        p_inputs = GetPolicyInputs(obs)

        # process the lstm hidden state
        hid_dim = self.p_net.lstm.hidden_size
        batch_size = p_inputs.size(0)
        def to_hidden(h):
            # if None: initilize as 0
            if h is None:
                return torch.zeros(1, batch_size, hid_dim)
            # if list: cast as tensor
            if isinstance(h, list):
                h = torch.tensor(h, dtype=torch.float32)
            # if dim == 1: [hid_dim] -> [1,1,hid_dim]
            if h.dim() == 1:
                h = h.view(1, 1, -1)
            # if dim == 2: [1,hid_dim] -> [1,1,hid_dim]
            elif h.dim() == 2:
                h = h.unsqueeze(1)
            # if h == [1,B,hid_dim] -> no change
            return h

        lstm_hidden_h = to_hidden(lstm_hidden_h)
        lstm_hidden_c = to_hidden(lstm_hidden_c)

        with torch.no_grad():
            mean, std, (next_lstm_hidden_h, next_lstm_hidden_c) = self.p_net(p_inputs, (lstm_hidden_h, lstm_hidden_c))
        
        # if enable_print: print(f"[DEBUG] the p_net output: mean({mean}), std({std})")

        if not active:
            return [0.0] * self.action_dim, 0.0, next_lstm_hidden_h, next_lstm_hidden_c

        # dim0 -> [0,10]: scale=5, loc=5
        # dim1 -> [6,10]: scale=2, loc=8
        # dim2 -> [6,10]: scale=2, loc=8
        scale = self.p_net.act_scale.expand_as(mean)
        loc   = self.p_net.act_bias.expand_as(mean)

        if self.evaluate or gp.settings.is_evaluate:
            u = mean
            a = torch.tanh(u)
            action = a * scale + loc
            act = action.squeeze(0).tolist()
            act_logprob = None
        else:
            if self.use_ou_noise:
                # OU exploration:
                eps = self._ou_eps(device=mean.device, batch_size=batch_size, dim=mean.size(-1))
            else:
                eps = torch.randn_like(mean).clamp(-3.0, 3.0)

            noise_scale = torch.tensor([0.75, 1.0, 1.0], device=mean.device).view(1, -1)
            # reparameterized sample in eps-space
            u = mean + (std * noise_scale) * eps
            # print(f"[DEBUG] mean: {mean}, std: {std}, eps: {eps}, u: {u}")
            a = torch.tanh(u)
            # print(f"[DEBUG] train_a: {a}")
            # print(f"[DEBUG] eval_a: {torch.tanh(mean)}")
            action = a * scale + loc
            # print(f"[DEBUG] train_action: {action}")
            # print(f"[DEBUG] eval_action: {torch.tanh(mean) * scale + loc}")
            dist = Normal(mean, std)
            normal_logp = dist.log_prob(u).sum(-1)
            squash = torch.log(1 - a.pow(2) + 1e-6).sum(-1)
            scale_logsum = torch.log(scale).sum(-1)
            logp = normal_logp - squash - scale_logsum

            act = action.squeeze(0).tolist()
            act_logprob = float(logp)
            # act_logprob = None

        return act, act_logprob, next_lstm_hidden_h, next_lstm_hidden_c

    def update_net(self, params):
        self.p_net.load_state_dict(params)

    def load_net(self, path):
        self.update_net(torch.load(path))

class MaddpgDeviceAgent():
    def __init__(self, agent_id, gen_params, alg_params):
        # general
        self.device_type_num = gen_params.device_type_num
        self.device_in_types = gen_params.device_in_types
        self.device = gp.settings.device

        # agent id
        self.agent_id = agent_id

        # policy network
        # self.p_net = MaddpgPolicyNet(alg_params)
        self.p_net = MaddpgPolicyNetLSTM(alg_params)

        # action noise
        self.use_action_noise = alg_params.use_action_noise
        # train noise
        self.train_update_cnt = 0
        self.noise_sigma_start = alg_params.noise_sigma_start
        self.noise_sigma_end = alg_params.noise_sigma_end
        self.noise_decay_updates = alg_params.noise_decay_updates

        if self.use_action_noise:
            self.action_noise = GaussianNoise(alg_params.action_dim, sigma=self.noise_sigma_start, device=self.device)
            
        self.evaluate = gen_params.evaluate
        self.lstm_hidden_dim = alg_params.p_hid_dims[1]

    def increment_update_cnt(self):
        self.train_update_cnt += 1
        # print(f"[DEBUG] train_update_cnt: {self.train_update_cnt}")

    def get_noise_sigma(self):
        t = min(self.train_update_cnt / max(self.noise_decay_updates, 1), 1.0)
        return self.noise_sigma_start + t * (self.noise_sigma_end - self.noise_sigma_start)

    def choose_action(self, obs, lstm_hidden_h, lstm_hidden_c):
        p_inputs = GetPolicyInputs(obs)

        # process the lstm hidden state
        hid_dim = self.p_net.lstm.hidden_size
        batch_size = p_inputs.size(0)  
        def to_hidden(h):
            # if None: initilize as 0
            if h is None:
                return torch.zeros(1, batch_size, hid_dim)
            # if list: cast as tensor
            if isinstance(h, list):
                h = torch.tensor(h, dtype=torch.float32)
            # if dim == 1: [hid_dim] -> [1,1,hid_dim]
            if h.dim() == 1:
                h = h.view(1, 1, -1)
            # if dim == 2: [1,hid_dim] -> [1,1,hid_dim]
            elif h.dim() == 2:
                h = h.unsqueeze(1)
            # if h == [1,B,hid_dim] -> no change
            return h
        lstm_hidden_h = to_hidden(lstm_hidden_h)
        lstm_hidden_c = to_hidden(lstm_hidden_c)
        
        with torch.no_grad():
            act, (next_lstm_hidden_h, next_lstm_hidden_c) = self.p_net(p_inputs, (lstm_hidden_h, lstm_hidden_c))
        if not (self.evaluate or gp.settings.is_evaluate):
            sigma = self.get_noise_sigma()
            noise = self.action_noise.sample(sigma).view_as(act)
            act = torch.clamp(act + noise, -1.0, 1.0)
    
        # dim0 -> [0,10]: scale=1, loc=1
        # dim1 -> [6,10]: scale=0.4, loc=1.6
        # dim2 -> [6,10]: scale=0.4, loc=1.6
        # act = torch.tensor(act, dtype=torch.float32)
        scale = self.p_net.act_scale
        loc   = self.p_net.act_bias
        # every action indicate with 10 dim
        scale = scale.repeat_interleave(10)
        loc = loc.repeat_interleave(10)
        action = act * scale + loc
        act = action.view(-1).tolist()
        return act, next_lstm_hidden_h, next_lstm_hidden_c
        
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
        # config 
        act_dim = 3
        act = [0 for i in range(act_dim)]
        act[0] = 0
        act[1] = 1
        act[2] = 1
        return act

class EdgeComputingDeviceAgent(StaticDeviceAgent):
    def __init__(self, agent_id, gen_params):
        super().__init__(agent_id, gen_params)
        
    def choose_action(self):
        act_dim = 3
        act = [0 for i in range(act_dim)]
        act[0] = 1
        act[1] = 1
        act[2] = 1
        return act

class RandomComputingDeviceAgent(StaticDeviceAgent):
    def __init__(self, agent_id, gen_params):
        super().__init__(agent_id, gen_params)
        
    def choose_action(self):
        act_dim = 3
        act = [0 for i in range(act_dim)]
        act[0] = np.random.uniform(0, 1)
        act[1] = np.random.uniform(0.6, 1)
        act[2] = np.random.uniform(0.6, 1)
        return act