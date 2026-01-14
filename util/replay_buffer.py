import copy
import torch
import numpy as np

# Store the transition data of devices that process a specific task type
class MappoReplayBuffer():
    def __init__(self, gen_params, alg_params):
        self.device_num = gen_params.device_num
        self.device_types = gen_params.device_types
        self.device_in_types = gen_params.device_in_types
        self.train_freq = alg_params.train_freq
        self.buffer_train_time_slots = alg_params.buffer_train_time_slots
        self.value_input_dims = alg_params.value_input_dims
        self.policy_input_dim = alg_params.policy_input_dim
        self.edge_queue_obs_dim = alg_params.edge_queue_obs_dim
        self.action_dim = alg_params.action_dim
        self.lstm_hidden_dim = alg_params.p_hid_dims[1]
        self.gamma = alg_params.gamma
        self.lamda = alg_params.lamda
        
        self.ps = [0, 0]
        self.edge_obs = [[None for j in range(self.buffer_train_time_slots + 1)]
                                for i in range(self.train_freq)]
        self.device_obss = [[None for j in range(self.buffer_train_time_slots + 1)]
                                    for i in range(self.train_freq)]
        self.lstm_hidden_hs = [[None for j in range(self.buffer_train_time_slots + 1)]
                                    for i in range(self.train_freq)]
        self.lstm_hidden_cs = [[None for j in range(self.buffer_train_time_slots + 1)]
                                    for i in range(self.train_freq)]
        self.device_acts = [[None for j in range(self.buffer_train_time_slots + 1)]
                                    for i in range(self.train_freq)]
        self.device_act_logprobs = [[None for j in range(self.buffer_train_time_slots + 1)]
                                            for i in range(self.train_freq)]
        self.joint_rewards = [[None for j in range(self.buffer_train_time_slots + 1)]
                                    for i in range(self.train_freq)]
        self.device_active = [[None for j in range(self.buffer_train_time_slots + 1)]
                                    for i in range(self.train_freq)]
        
    def store(self, edge_obs, device_obss, lstm_hidden_hs, lstm_hidden_cs, device_acts, device_act_logprobs, joint_rewards, device_active):
        self.edge_obs[self.ps[0]][self.ps[1]] = copy.copy(edge_obs)
        self.device_obss[self.ps[0]][self.ps[1]] = copy.copy(device_obss)
        self.lstm_hidden_hs[self.ps[0]][self.ps[1]] = copy.copy(lstm_hidden_hs)
        self.lstm_hidden_cs[self.ps[0]][self.ps[1]] = copy.copy(lstm_hidden_cs)
        self.device_acts[self.ps[0]][self.ps[1]] = copy.copy(device_acts)
        self.device_act_logprobs[self.ps[0]][self.ps[1]] = copy.copy(device_act_logprobs)
        self.joint_rewards[self.ps[0]][self.ps[1]] = copy.copy(joint_rewards)
        self.device_active[self.ps[0]][self.ps[1]] = copy.copy(device_active)

        # update positions
        if self.ps[1] == self.buffer_train_time_slots:
            self.ps[0] = (self.ps[0] + 1) % (self.train_freq)
        self.ps[1] = (self.ps[1] + 1) % (self.buffer_train_time_slots + 1)

    def package_value_inputs(self, train_episode, train_time_slot, queue_id):
        v_input = []

        def add(x):
            if isinstance(x, torch.Tensor):
                v_input.extend(x.detach().cpu().reshape(-1).tolist())
            elif isinstance(x, np.ndarray):
                v_input.extend(x.reshape(-1).tolist())
            elif isinstance(x, (list, tuple)):
                v_input.extend(np.asarray(x, dtype=np.float32).reshape(-1).tolist())
            else:
                v_input.append(float(x))

        start = queue_id * self.edge_queue_obs_dim
        end = (queue_id + 1) * self.edge_queue_obs_dim
        for i in range(start, end):
            add(self.edge_obs[train_episode][train_time_slot][i])

        for dev_id in self.device_in_types[queue_id]:
            add(self.device_obss[train_episode][train_time_slot][dev_id])

        return torch.tensor(v_input, dtype=torch.float32).reshape(1, -1)


    def package_policy_input(self, train_episode, train_time_slot, device_id):
        p_input = []

        def add(x):
            if isinstance(x, torch.Tensor):
                p_input.extend(x.detach().cpu().reshape(-1).tolist())
            elif isinstance(x, np.ndarray):
                p_input.extend(x.reshape(-1).tolist())
            elif isinstance(x, (list, tuple)):
                p_input.extend(np.asarray(x, dtype=np.float32).reshape(-1).tolist())
            else:
                p_input.append(float(x))

        queue_id = self.device_types[device_id]
        start = queue_id * self.edge_queue_obs_dim
        end = (queue_id + 1) * self.edge_queue_obs_dim
        for i in range(start, end):
            add(self.edge_obs[train_episode][train_time_slot][i])

        add(self.device_obss[train_episode][train_time_slot][device_id])

        return torch.tensor(p_input, dtype=torch.float32).reshape(1, -1)

    def get_policy_net_training_data(self):
        # build policy training data (still on CPU) -----
        p_inputs = torch.zeros(
            [self.train_freq, self.buffer_train_time_slots, self.device_num, self.policy_input_dim],
            dtype=torch.float32
        )
        acts = torch.zeros(
            [self.train_freq, self.buffer_train_time_slots, self.device_num, self.action_dim],
            dtype=torch.float32
        )
        act_logprobs = torch.zeros(
            [self.train_freq, self.buffer_train_time_slots, self.device_num, 1],
            dtype=torch.float32
        )
        device_active = torch.zeros(
            [self.train_freq, self.buffer_train_time_slots, self.device_num, 1],
            dtype=torch.float32
        )
        lstm_hidden_hs = torch.zeros(
            [self.train_freq, self.buffer_train_time_slots, self.device_num, self.lstm_hidden_dim],
            dtype=torch.float32
        )
        lstm_hidden_cs = torch.zeros(
            [self.train_freq, self.buffer_train_time_slots, self.device_num, self.lstm_hidden_dim],
            dtype=torch.float32
        )

        for i in range(self.train_freq):
            for j in range(self.buffer_train_time_slots):
                for k in range(self.device_num):
                    
                    obs = copy.copy(self.device_obss[i][j][k])
                    inputs = self.package_policy_input(i,j,k)
                    if not torch.is_tensor(inputs):
                        inputs = torch.as_tensor(inputs, dtype=torch.float32)
                    p_inputs[i, j, k] = inputs

                    acts[i, j, k] = torch.tensor(
                        self.device_acts[i][j][k], dtype=torch.float32
                    )
                    act_logprobs[i, j, k] = torch.tensor(
                        self.device_act_logprobs[i][j][k], dtype=torch.float32
                    )
                    device_active[i, j, k] = torch.tensor(
                        self.device_active[i][j][k], dtype=torch.float32
                    )

                    raw_h = self.lstm_hidden_hs[i][j][k]
                    raw_c = self.lstm_hidden_cs[i][j][k]

                    h = torch.as_tensor(raw_h, dtype=torch.float32).view(-1)[:self.lstm_hidden_dim]
                    c = torch.as_tensor(raw_c, dtype=torch.float32).view(-1)[:self.lstm_hidden_dim]

                    lstm_hidden_hs[i, j, k] = h
                    lstm_hidden_cs[i, j, k] = c

        # reshape to [train_freq * buffer_train_time_slots, ...]
        p_inputs = p_inputs.reshape([-1, self.device_num, self.policy_input_dim])
        acts = acts.reshape([-1, self.device_num, self.action_dim])
        act_logprobs = act_logprobs.reshape([-1, self.device_num, 1])
        device_active = device_active.reshape([-1, self.device_num, 1])
        lstm_hidden_hs = lstm_hidden_hs.reshape([-1, self.device_num, self.lstm_hidden_dim])
        lstm_hidden_cs = lstm_hidden_cs.reshape([-1, self.device_num, self.lstm_hidden_dim])

        return p_inputs, lstm_hidden_hs, lstm_hidden_cs, \
               acts, act_logprobs, device_active

    def get_value_net_training_data(self, queue_id, value_net):
        """
        Build training data and compute GAE.
        value_net is already on some device (CPU or GPU).
        """
        # print(f"get_value_net_training_data:value_input_dims {self.value_input_dims}")
        v_inputs = torch.zeros(
            [self.train_freq, self.buffer_train_time_slots + 1, self.value_input_dims[queue_id]],
            dtype=torch.float32
        )
        # print(f"v_inputs {v_inputs.shape}")
        for i in range(self.train_freq):
            for j in range(self.buffer_train_time_slots + 1):
                inputs = self.package_value_inputs(i,j,queue_id)
                if not torch.is_tensor(inputs):
                    inputs = torch.as_tensor(inputs, dtype=torch.float32)
                v_inputs[i, j] = inputs

        # compute state values with value_net on its device -----
        dev_v = next(value_net.parameters()).device
        with torch.no_grad():
            # move inputs to value_net device for forward
            v_inputs_flat = v_inputs.reshape([-1, self.value_input_dims[queue_id]]).to(dev_v)
            vs = value_net(v_inputs_flat)  # [train_freq*(T+1), 1] on dev_v
            # move back to CPU for further processing
            vs = vs.cpu().reshape([self.train_freq, self.buffer_train_time_slots + 1, 1])

        # compute GAE etc. on CPU -----
        jr = torch.as_tensor(self.joint_rewards, dtype=torch.float32)
        rewards = jr[:, :self.buffer_train_time_slots, queue_id:queue_id+1]

        # deltas: [train_freq, buffer_train_time_slots, 1]
        deltas = rewards + self.gamma * vs[:, 1: self.buffer_train_time_slots + 1] - \
                 vs[:, 0: self.buffer_train_time_slots]

        gae = 0.0
        advs = torch.zeros([self.train_freq, self.buffer_train_time_slots, 1],
                           dtype=torch.float32)
        for t in reversed(range(self.buffer_train_time_slots)):
            gae = deltas[:, t] + self.lamda * self.gamma * gae
            advs[:, t] = gae

        # value targets
        v_tags = advs + vs[:, 0: self.buffer_train_time_slots]

        # normalize advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-5)

        # flatten value training data -----
        # [train_freq * buffer_train_time_slots, state_dim]
        v_inputs = v_inputs[:, 0: self.buffer_train_time_slots].reshape([-1, self.value_input_dims[queue_id]])
        # [train_freq * buffer_train_time_slots, 1]
        v_tags = v_tags.reshape([-1, 1])

        advs = advs.reshape([-1, 1])

        return v_inputs, v_tags, advs

class MaddpgReplayBuffer():
    def __init__(self, gen_params, alg_params):
        self.device_num = gen_params.device_num
        self.buffer_size = alg_params.buffer_size
        self.device_type_num = gen_params.device_type_num
        self.device_in_types = gen_params.device_in_types
        self.edge_queue_obs_dim = alg_params.edge_queue_obs_dim
        self.device_types = gen_params.device_types
        self.policy_input_dim = alg_params.policy_input_dim
        self.value_input_dims = alg_params.value_input_dims
        self.value_input_obs_dims = alg_params.value_input_obs_dims
        self.value_input_act_dims = alg_params.value_input_act_dims
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

    def package_value_inputs(self, batch_id, queue_id):
        v_input = []
        next_v_input = []
        def add(x):
            if isinstance(x, torch.Tensor):
                v_input.extend(x.detach().cpu().reshape(-1).tolist())
            elif isinstance(x, np.ndarray):
                v_input.extend(x.reshape(-1).tolist())
            elif isinstance(x, (list, tuple)):
                v_input.extend(np.asarray(x, dtype=np.float32).reshape(-1).tolist())
            else:
                v_input.append(float(x))

        def add_next(x):
            if isinstance(x, torch.Tensor):
                next_v_input.extend(x.detach().cpu().reshape(-1).tolist())
            elif isinstance(x, np.ndarray):
                next_v_input.extend(x.reshape(-1).tolist())
            elif isinstance(x, (list, tuple)):
                next_v_input.extend(np.asarray(x, dtype=np.float32).reshape(-1).tolist())
            else:
                next_v_input.append(float(x))

        start = queue_id * self.edge_queue_obs_dim
        end = (queue_id + 1) * self.edge_queue_obs_dim
        for i in range(start, end):
            add(self.edge_obss[batch_id][i])
            add_next(self.next_edge_obss[batch_id][i])

        for dev_id in self.device_in_types[queue_id]:
            add(self.device_obss[batch_id][dev_id])
            add_next(self.next_device_obss[batch_id][dev_id])

        return torch.tensor(v_input, dtype=torch.float32).reshape(1, -1), torch.tensor(next_v_input, dtype=torch.float32).reshape(1, -1)


    def package_policy_input(self, batch_id, device_id):
        p_input = []
        next_p_input = []

        def add(x):
            if isinstance(x, torch.Tensor):
                p_input.extend(x.detach().cpu().reshape(-1).tolist())
            elif isinstance(x, np.ndarray):
                p_input.extend(x.reshape(-1).tolist())
            elif isinstance(x, (list, tuple)):
                p_input.extend(np.asarray(x, dtype=np.float32).reshape(-1).tolist())
            else:
                p_input.append(float(x))

        def add_next(x):
            if isinstance(x, torch.Tensor):
                next_p_input.extend(x.detach().cpu().reshape(-1).tolist())
            elif isinstance(x, np.ndarray):
                next_p_input.extend(x.reshape(-1).tolist())
            elif isinstance(x, (list, tuple)):
                next_p_input.extend(np.asarray(x, dtype=np.float32).reshape(-1).tolist())
            else:
                next_p_input.append(float(x))

        queue_id = self.device_types[device_id]
        start = queue_id * self.edge_queue_obs_dim
        end = (queue_id + 1) * self.edge_queue_obs_dim
        for i in range(start, end):
            add(self.edge_obss[batch_id][i])
            add_next(self.next_edge_obss[batch_id][i])

        add(self.device_obss[batch_id][device_id])
        add_next(self.next_device_obss[batch_id][device_id])

        return torch.tensor(p_input, dtype=torch.float32).reshape(1, -1), torch.tensor(next_p_input, dtype=torch.float32).reshape(1, -1)

    '''
    batch_states:             v_net输入,按queue_id划分 [device_type_num, batch_size, state_dim]
    batch_device_obss:        p_net输入,按device_id划分 [batch_size, device_num, obs_dim]
    batch_joint_acts:         v_net输出,按queue_id划分 [device_type_num, batch_size, joint_act_dim]
    batch_joint_rewards:      v_net输入,按queue_id划分 [device_type_num, batch_size, device_type_num]
    batch_next_states:        v_net输入,按queue_id划分 [device_type_num, batch_size, state_dim]
    batch_next_device_obss:   p_net输入,按device_id划分 [batch_size, device_num, obs_dim]
    '''
    def sample(self, batch_ids):
        # states and device obss
        # next states and device obss
        batch_states = [
            torch.zeros((len(batch_ids), self.value_input_obs_dims[i]), dtype=torch.float32)
            for i in range(self.device_type_num)
        ]
        batch_next_states = [
            torch.zeros((len(batch_ids), self.value_input_obs_dims[i]), dtype=torch.float32)
            for i in range(self.device_type_num)
        ]
        batch_device_obss = torch.zeros(
            [len(batch_ids), self.device_num,  self.policy_input_dim],
            dtype=torch.float32
        )
        batch_next_device_obss = torch.zeros(
            [len(batch_ids), self.device_num,  self.policy_input_dim],
            dtype=torch.float32
        )
        j = 0
        for id_ in batch_ids:
            for i in range(self.device_type_num):
                state, next_state = self.package_value_inputs(id_,i)
                state = state.reshape(-1)
                next_state = next_state.reshape(-1)
                batch_states[i][j] = state
                batch_next_states[i][j] = next_state
            j += 1
        
        j = 0
        for id_ in batch_ids:
            for i in range(self.device_num):
                device_obs, next_device_obs = self.package_policy_input(id_, i)
                device_obs = device_obs.reshape(-1)
                next_device_obs = next_device_obs.reshape(-1)
                batch_device_obss[j][i] = device_obs
                batch_next_device_obss[j][i] = next_device_obs
            j += 1

        # joint actions
        batch_joint_acts = [
            torch.zeros((len(batch_ids), self.value_input_act_dims[i]), dtype=torch.float32)
            for i in range(self.device_type_num)
        ]
        j = 0
        for id_ in batch_ids:
            for k in range(self.device_type_num):
                joint_act = []
                for i in self.device_in_types[k]:
                    joint_act += self.device_acts[id_][i]
                joint_act = torch.tensor(joint_act, dtype=torch.float32).reshape(-1)
                batch_joint_acts[k][j] = joint_act
            j += 1

        # joint rewards
        batch_joint_rewards = [
            torch.zeros((len(batch_ids), 1), dtype=torch.float32)
            for i in range(self.device_type_num)
        ]
        j = 0
        for id_ in batch_ids:
            for k in range(self.device_type_num):
                batch_joint_rewards[k][j] = self.joint_rewards[id_][k]
            j += 1
        return batch_states, batch_device_obss, \
                batch_joint_acts, batch_joint_rewards, \
                batch_next_states, batch_next_device_obss
