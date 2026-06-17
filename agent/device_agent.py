from abc import abstractmethod
import numpy as np
import torch
from torch.distributions import Normal
from network.policy_net import MaddpgPolicyNet, MappoPolicyNet
from util.utils import GetPolicyInputs, GaussianNoise
import math
import config.global_params as gp

class MappoDeviceAgent():
    def __init__(self, agent_id, gen_params, alg_params):
        # agent id
        self.agent_id = agent_id

        # policy network
        self.p_net = MappoPolicyNet(alg_params)

        self.evaluate = gen_params.evaluate
        self.action_dim = alg_params.action_dim
        self.action_encode_dim = alg_params.action_encode_dim
        self.edge_server_num = gen_params.edge_server_num

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

    def choose_action(self, obs, active: bool = True):

        enable_print = gp.settings.enable_print

        p_inputs = GetPolicyInputs(obs)

        with torch.no_grad():
            mean, std = self.p_net(p_inputs)

        S = self.edge_server_num
        if not active:
            self.last_full_act = [-1.0] * self.action_dim
            return [-1] * (S + 1), None

        # dim 0~S-1: server logits → argmax
        # dim S~S+3*ae_dim-1: 3 continuous actions (offl/trpw/comp), each ae_dim dims
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
                eps = self._ou_eps(device=mean.device, batch_size=batch_size, dim=mean.size(-1))
            else:
                eps = torch.randn_like(mean).clamp(-3.0, 3.0)

            ae_dim = self.action_encode_dim
            noise_scale = torch.tensor(
                [1.0] * S + [0.75] * ae_dim + [1.0] * ae_dim + [1.0] * ae_dim,
                device=mean.device
            ).view(1, -1)
            u = mean + (std * noise_scale) * eps
            a = torch.tanh(u)
            action = a * scale + loc
            act = action.squeeze(0).tolist()
            dist = Normal(mean, std)
            normal_logp = dist.log_prob(u).sum(-1)
            squash = torch.log(1 - a.pow(2) + 1e-6).sum(-1)
            scale_logsum = torch.log(scale).sum(-1)
            logp = normal_logp - squash - scale_logsum
            act_logprob = float(logp)

        # extract server_id and build env action (average ae_dim blocks)
        ae_dim = self.action_encode_dim
        server_id = int(np.argmax(act[:S]))
        if gp.settings.enable_print:
            print(f"[DEBUG] Device {self.agent_id} offloads to server {server_id}")
        cont_vals = []
        for j in range(3):
            block = act[S + j * ae_dim : S + (j + 1) * ae_dim]
            cont_vals.append(sum(block) / ae_dim)
        env_act = [float(server_id)] + cont_vals
        self.last_full_act = act  # for replay buffer

        return env_act, act_logprob

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

        self.edge_server_num = gen_params.edge_server_num

        # policy network
        self.p_net = MaddpgPolicyNet(alg_params)

        # action noise
        self.use_action_noise = alg_params.use_action_noise
        # train noise
        self.train_update_cnt = 0
        self.noise_sigma_start = alg_params.noise_sigma_start
        self.noise_sigma_end = alg_params.noise_sigma_end
        self.noise_decay_num = alg_params.noise_decay_num

        if self.use_action_noise:
            # self.action_noise = GaussianNoise(alg_params.action_dim)
            self.action_noise = GaussianNoise(alg_params.action_dim, sigma=self.noise_sigma_start, device=self.device)
            
        self.evaluate = gen_params.evaluate
        self.action_encode_dim = alg_params.action_encode_dim

    def increment_update_cnt(self):
        self.train_update_cnt += 1
        # print(f"[DEBUG] train_update_cnt: {self.train_update_cnt}")

    def get_noise_sigma(self):
        t = min(self.train_update_cnt / max(self.noise_decay_num, 1), 1.0)
        return self.noise_sigma_start + t * (self.noise_sigma_end - self.noise_sigma_start)

    def choose_action(self, obs):
        p_inputs = GetPolicyInputs(obs)

        with torch.no_grad():
            act = self.p_net(p_inputs)
        if not (self.evaluate or gp.settings.is_evaluate):
            sigma = self.get_noise_sigma()
            noise = self.action_noise.sample(sigma).view_as(act).to(act.device)
            act = torch.clamp(act + noise, -1.0, 1.0)

        S = self.edge_server_num
        ae_dim = self.action_encode_dim
        # First S dims: server logits → argmax
        server_logits = act[:, :S]
        server_id = int(torch.argmax(server_logits, dim=-1).item())
        if gp.settings.enable_print:
            print(f"[DEBUG] Device {self.agent_id} offloads to server {server_id}")

        # 3*ae_dim dims: continuous actions (ae_dim each)
        cont_act = act[:, S:S + 3 * ae_dim]
        scale = self.p_net.act_scale[S:S + 3 * ae_dim]
        loc = self.p_net.act_bias[S:S + 3 * ae_dim]
        action = cont_act * scale + loc
        act_list = action.view(-1).tolist()

        # Average per ae_dim to get 3 continuous values
        env_cont = []
        for j in range(3):
            env_cont.append(sum(act_list[j * ae_dim : (j + 1) * ae_dim]) / ae_dim)

        env_act = [float(server_id)] + env_cont
        self.last_full_act = act.view(-1).tolist()
        return env_act
        
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
        return [0, 0, 1.0, 1.0]

class EdgeComputingDeviceAgent(StaticDeviceAgent):
    def __init__(self, agent_id, gen_params):
        super().__init__(agent_id, gen_params)
        self.edge_server_num = gen_params.edge_server_num

    def choose_action(self):
        return [0, 1.0, 1.0, 1.0]

class RandomComputingDeviceAgent(StaticDeviceAgent):
    def __init__(self, agent_id, gen_params):
        super().__init__(agent_id, gen_params)
        self.edge_server_num = gen_params.edge_server_num
        base_seed = gp.settings.seed
        self._np_rnd = np.random.RandomState(base_seed + agent_id)

    def choose_action(self):
        server_id = self._np_rnd.randint(0, self.edge_server_num)
        offl = self._np_rnd.uniform(0.6, 1)
        trpw = self._np_rnd.uniform(0.6, 1)
        comp = self._np_rnd.uniform(0.6, 1)
        return [float(server_id), offl, trpw, comp]