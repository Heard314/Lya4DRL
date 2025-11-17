import copy
import torch
from util.utils import GetPolicyInputs, GetValueInputs

class MappoReplayBuffer():
    def __init__(self, gen_params, alg_params):
        self.device_num = gen_params.device_num
        self.train_freq = alg_params.train_freq
        self.train_time_slots = alg_params.train_time_slots
        self.obs_dim = alg_params.obs_dim
        self.state_dim = alg_params.state_dim
        self.action_dim = alg_params.action_dim
        self.lstm_hidden_dim = alg_params.p_hid_dims[1]
        self.gamma = alg_params.gamma
        self.lamda = alg_params.lamda
        
        self.ps = [0, 0]
        self.edge_obs = [[None for j in range(self.train_time_slots + 1)]
                                for i in range(self.train_freq)]
        self.device_obss = [[None for j in range(self.train_time_slots + 1)]
                                    for i in range(self.train_freq)]
        self.lstm_hidden_hs = [[None for j in range(self.train_time_slots + 1)]
                                    for i in range(self.train_freq)]
        self.lstm_hidden_cs = [[None for j in range(self.train_time_slots + 1)]
                                    for i in range(self.train_freq)]
        self.device_acts = [[None for j in range(self.train_time_slots + 1)]
                                    for i in range(self.train_freq)]
        self.device_act_logprobs = [[None for j in range(self.train_time_slots + 1)]
                                            for i in range(self.train_freq)]
        self.joint_reward = [[None for j in range(self.train_time_slots + 1)]
                                    for i in range(self.train_freq)]
        self.device_active = [[None for j in range(self.train_time_slots + 1)]
                                    for i in range(self.train_freq)]
       
    def store(self, edge_obs, device_obss, lstm_hidden_hs, lstm_hidden_cs, device_acts, device_act_logprobs, joint_reward, device_active):
        self.edge_obs[self.ps[0]][self.ps[1]] = copy.copy(edge_obs)
        self.device_obss[self.ps[0]][self.ps[1]] = copy.copy(device_obss)
        self.lstm_hidden_hs[self.ps[0]][self.ps[1]] = copy.copy(lstm_hidden_hs)
        self.lstm_hidden_cs[self.ps[0]][self.ps[1]] = copy.copy(lstm_hidden_cs)
        self.device_acts[self.ps[0]][self.ps[1]] = copy.copy(device_acts)
        self.device_act_logprobs[self.ps[0]][self.ps[1]] = copy.copy(device_act_logprobs)
        self.joint_reward[self.ps[0]][self.ps[1]] = copy.copy(joint_reward)
        self.device_active[self.ps[0]][self.ps[1]] = copy.copy(device_active)

        # update positions
        if self.ps[1] == self.train_time_slots:
            self.ps[0] = (self.ps[0] + 1) % (self.train_freq)
        self.ps[1] = (self.ps[1] + 1) % (self.train_time_slots + 1)
        
    def get_training_data(self, value_net):
        '''GAE'''
        v_inputs = torch.zeros([self.train_freq, self.train_time_slots + 1, self.state_dim])
        for i in range(self.train_freq):
            for j in range(self.train_time_slots + 1):
                edge_obs = copy.copy(self.edge_obs[i][j])
                device_obss = copy.copy(self.device_obss[i][j])
                inputs = GetValueInputs(edge_obs, device_obss)
                v_inputs[i, j] = inputs
        
        with torch.no_grad():
            vs = value_net(v_inputs.reshape([-1, self.state_dim]))
        # 使用价值网络计算状态价值
        vs = vs.reshape([self.train_freq, self.train_time_slots + 1, 1])
        
        # [train_freq, train_time_slots, 1]
        rewards = torch.tensor(self.joint_reward, dtype = torch.float) \
                    [:, 0: self.train_time_slots].unsqueeze(-1)
        # 计算优势函数
        # [train_episodes, train_time_slots, 1]
        deltas = rewards + self.gamma * vs[:, 1: self.train_time_slots + 1] - \
                    vs[:, 0: self.train_time_slots]
        # 计算GAE优势函数
        gae = 0
        advs = torch.zeros([self.train_freq, self.train_time_slots, 1])
        for t in reversed(range(self.train_time_slots)):
            gae = deltas[:, t] + self.lamda * self.gamma * gae
            advs[:, t] = gae
        
        # 计算价值网络目标值
        # [train_episodes, train_time_slots, 1]
        v_tags = advs + vs[:, 0: self.train_time_slots]
        # normalization
        advs = (advs - advs.mean()) / (advs.std() + 1e-5)
        
        '''training data - value network'''
        # [train_freq x train_time_slots, state_dim]
        v_inputs = v_inputs[:, 0: self.train_time_slots].reshape([-1, self.state_dim])
        # [train_freq x train_time_slots, 1]
        v_tags = v_tags.reshape([-1, 1])
        
        '''training data - policy networks'''
        p_inputs = torch.zeros([self.train_freq, self.train_time_slots, 
                                self.device_num, self.obs_dim])
        acts = torch.zeros([self.train_freq, self.train_time_slots, 
                            self.device_num, self.action_dim])
        act_logprobs = torch.zeros([self.train_freq, self.train_time_slots, 
                                    self.device_num, 1])
        device_active = torch.zeros([self.train_freq, self.train_time_slots, 
                                    self.device_num, 1])
        lstm_hidden_hs = torch.zeros([self.train_freq, self.train_time_slots, 
                                    self.device_num, self.lstm_hidden_dim])
        lstm_hidden_cs = torch.zeros([self.train_freq, self.train_time_slots, 
                                    self.device_num, self.lstm_hidden_dim])
        for i in range(self.train_freq):
            for j in range(self.train_time_slots):
                for k in range(self.device_num):
                    obs = copy.copy(self.device_obss[i][j][k])
                    inputs = GetPolicyInputs(obs)
                    p_inputs[i, j, k] = inputs
                    acts[i, j, k] = torch.tensor(self.device_acts[i][j][k],
                                                    dtype = torch.float)
                    act_logprobs[i, j, k] = torch.tensor(self.device_act_logprobs[i][j][k],
                                                            dtype = torch.float)
                    device_active[i, j, k] = torch.tensor(self.device_active[i][j][k],
                                                            dtype = torch.float)
                    raw_h = self.lstm_hidden_hs[i][j][k]    # 可能是 list 或 tensor
                    raw_c = self.lstm_hidden_cs[i][j][k]

                    # 1. 安全转成 tensor（不管是 list 还是 tensor）
                    h = torch.as_tensor(raw_h, dtype=torch.float32)
                    c = torch.as_tensor(raw_c, dtype=torch.float32)

                    # 2. 拉平成一维，并强制长度为 lstm_hidden_dim
                    h = h.view(-1)[:self.lstm_hidden_dim]
                    c = c.view(-1)[:self.lstm_hidden_dim]

                    # 3. 塞进预分配好的大 tensor
                    lstm_hidden_hs[i, j, k] = h
                    lstm_hidden_cs[i, j, k] = c

        # [train_freq x train_time_slots, device_num, obs_dim]
        p_inputs = p_inputs.reshape([-1, self.device_num, self.obs_dim])
        # [train_freq x train_time_slots, device_num, action_dim]
        acts = acts.reshape([-1, self.device_num, self.action_dim])
        # [train_freq x train_time_slots, device_num, 1]
        act_logprobs = act_logprobs.reshape([-1, self.device_num, 1])
        device_active = device_active.reshape([-1, self.device_num, 1])
        lstm_hidden_hs = lstm_hidden_hs.reshape([-1, self.device_num, self.lstm_hidden_dim])
        lstm_hidden_cs = lstm_hidden_cs.reshape([-1, self.device_num, self.lstm_hidden_dim])
        advs = advs.reshape([-1, 1])
        
        return v_inputs, v_tags, p_inputs, lstm_hidden_hs, lstm_hidden_cs, acts, act_logprobs, advs, device_active
   
class MaddpgReplayBuffer():
    def __init__(self, gen_params, alg_params):
        self.device_num = gen_params.device_num
        self.buffer_size = alg_params.buffer_size
        
        self.edge_obss = [None for i in range(self.buffer_size)]
        self.device_obss = [None for i in range(self.buffer_size)]
        self.device_acts = [None for i in range(self.buffer_size)]
        self.joint_rewards = [None for i in range(self.buffer_size)]
        self.next_edge_obss = [None for i in range(self.buffer_size)]
        self.next_device_obss = [None for i in range(self.buffer_size)]
        
        # position
        self.ps = 0
        
    def store(self, edge_obs, device_obss, 
                device_acts, joint_reward, next_edge_obs, next_device_obss):
        self.edge_obss[self.ps] = copy.copy(edge_obs)
        self.device_obss[self.ps] = copy.copy(device_obss)
        self.device_acts[self.ps] = copy.copy(device_acts)
        self.joint_rewards[self.ps] = copy.copy(joint_reward)
        self.next_edge_obss[self.ps] = copy.copy(next_edge_obs)
        self.next_device_obss[self.ps] = copy.copy(next_device_obss)
        
        # update position
        self.ps = (self.ps + 1) % self.buffer_size
        
    def sample(self, batch_ids):
        # states and device obss
        batch_states = []
        batch_device_obss = []
        for id_ in batch_ids:
            state = [] 
            state += self.edge_obss[id_]
            for i in range(self.device_num):
                state += self.device_obss[id_][i]
            batch_states.append(state)
            batch_device_obss.append(self.device_obss[id_])
        # [batch_size, state_dim]
        batch_states = torch.tensor(batch_states, dtype = torch.float)
        # [batch_size, device_num, obs_dim]
        batch_device_obss = torch.tensor(batch_device_obss, dtype = torch.float)
        
        # joint actions
        batch_joint_acts = []
        for id_ in batch_ids:
            joint_act = []
            for i in range(self.device_num):
                joint_act += self.device_acts[id_][i]
            batch_joint_acts.append(joint_act)
        # [batch_size, joint_act_dim]
        batch_joint_acts = torch.tensor(batch_joint_acts, dtype = torch.float)
        
        # joint rewards
        batch_joint_rewards = []
        for id_ in batch_ids:
            batch_joint_rewards.append(self.joint_rewards[id_])
        # [batch_size, 1]
        batch_joint_rewards = torch.tensor(batch_joint_rewards, dtype = torch.float) \
                                .reshape([-1, 1])
        
        # next states and device obss
        batch_next_states = []
        batch_next_device_obss = []
        for id_ in batch_ids:
            next_state = []
            next_state += self.next_edge_obss[id_]
            for i in range(self.device_num):
                next_state += self.next_device_obss[id_][i]
            batch_next_states.append(next_state)
            batch_next_device_obss.append(self.next_device_obss[id_])
        # [batch_size, state_dim]
        batch_next_states = torch.tensor(batch_next_states, dtype = torch.float)
        # [batch_size, device_num, obs_dim]
        batch_next_device_obss = torch.tensor(batch_next_device_obss, dtype = torch.float)
        
        return batch_states, batch_device_obss, \
                batch_joint_acts, batch_joint_rewards, \
                batch_next_states, batch_next_device_obss