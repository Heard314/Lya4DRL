import copy
import torch
import numpy as np


def _append_flat(target_list, x):
    """Append scalar/tensor/array elements into target_list in-place."""
    if isinstance(x, torch.Tensor):
        target_list.extend(x.detach().cpu().reshape(-1).tolist())
    elif isinstance(x, np.ndarray):
        target_list.extend(x.reshape(-1).tolist())
    elif isinstance(x, (list, tuple)):
        target_list.extend(np.asarray(x, dtype=np.float32).reshape(-1).tolist())
    else:
        target_list.append(float(x))


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
        self.action_encode_dim = alg_params.action_encode_dim
        self.gamma = alg_params.gamma
        self.lamda = alg_params.lamda
        
        self.ps = [0, 0]
        self.edge_obs = [[None for j in range(self.buffer_train_time_slots + 1)]
                                for i in range(self.train_freq)]
        self.device_obss = [[None for j in range(self.buffer_train_time_slots + 1)]
                                    for i in range(self.train_freq)]
        self.device_acts = [[None for j in range(self.buffer_train_time_slots + 1)]
                                    for i in range(self.train_freq)]
        self.device_act_logprobs = [[None for j in range(self.buffer_train_time_slots + 1)]
                                            for i in range(self.train_freq)]
        self.joint_rewards = [[None for j in range(self.buffer_train_time_slots + 1)]
                                    for i in range(self.train_freq)]
        self.device_active = [[None for j in range(self.buffer_train_time_slots + 1)]
                                    for i in range(self.train_freq)]
        
    def store(self, edge_obs, device_obss, device_acts, device_act_logprobs, joint_rewards, device_active):
        self.edge_obs[self.ps[0]][self.ps[1]] = copy.copy(edge_obs)
        self.device_obss[self.ps[0]][self.ps[1]] = copy.copy(device_obss)
        self.device_acts[self.ps[0]][self.ps[1]] = copy.copy(device_acts)
        self.device_act_logprobs[self.ps[0]][self.ps[1]] = copy.copy(device_act_logprobs)
        self.joint_rewards[self.ps[0]][self.ps[1]] = copy.copy(joint_rewards)
        self.device_active[self.ps[0]][self.ps[1]] = copy.copy(device_active)

        # update positions
        if self.ps[1] == self.buffer_train_time_slots:
            self.ps[0] = (self.ps[0] + 1) % (self.train_freq)
        self.ps[1] = (self.ps[1] + 1) % (self.buffer_train_time_slots + 1)

    def package_value_inputs(self, train_episode, train_time_slot):
        v_input = []
        for dev_id in range(self.device_num):
            for i in range(self.edge_queue_obs_dim):
                _append_flat(v_input, self.edge_obs[train_episode][train_time_slot][i])
            _append_flat(v_input, self.device_obss[train_episode][train_time_slot][dev_id])

        return torch.tensor(v_input, dtype=torch.float32).reshape(1, -1)


    def package_policy_input(self, train_episode, train_time_slot, device_id):
        p_input = []

        # Global edge_obs (shared across all devices)
        for i in range(self.edge_queue_obs_dim):
            _append_flat(p_input, self.edge_obs[train_episode][train_time_slot][i])

        _append_flat(p_input, self.device_obss[train_episode][train_time_slot][device_id])

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

        # reshape to [train_freq * buffer_train_time_slots, ...]
        p_inputs = p_inputs.reshape([-1, self.device_num, self.policy_input_dim])
        acts = acts.reshape([-1, self.device_num, self.action_dim])
        act_logprobs = act_logprobs.reshape([-1, self.device_num, 1])
        device_active = device_active.reshape([-1, self.device_num, 1])

        return p_inputs, \
               acts, act_logprobs, device_active

    def get_value_net_training_data(self, value_net):
        """
        Build training data and compute GAE with global reward.
        value_net is already on some device (CPU or GPU).
        """
        v_inputs = torch.zeros(
            [self.train_freq, self.buffer_train_time_slots + 1, self.value_input_dims],
            dtype=torch.float32
        )
        for i in range(self.train_freq):
            for j in range(self.buffer_train_time_slots + 1):
                inputs = self.package_value_inputs(i, j)
                if not torch.is_tensor(inputs):
                    inputs = torch.as_tensor(inputs, dtype=torch.float32)
                v_inputs[i, j] = inputs

        # compute state values with value_net on its device -----
        dev_v = next(value_net.parameters()).device
        with torch.no_grad():
            v_inputs_flat = v_inputs.reshape([-1, self.value_input_dims]).to(dev_v)
            vs = value_net(v_inputs_flat)  # [train_freq*(T+1), 1] on dev_v
            vs = vs.cpu().reshape([self.train_freq, self.buffer_train_time_slots + 1, 1])

        # compute GAE etc. on CPU -----
        jr = torch.as_tensor(self.joint_rewards, dtype=torch.float32)
        # joint_reward is already a single global scalar per slot
        rewards = jr[:, :self.buffer_train_time_slots].unsqueeze(-1)

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
        v_inputs = v_inputs[:, 0: self.buffer_train_time_slots].reshape([-1, self.value_input_dims])
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
        self.action_dim = alg_params.action_dim
        self.action_encode_dim = alg_params.action_encode_dim
        self.value_input_dims = alg_params.value_input_dims
        self.value_input_obs_dim = alg_params.value_input_obs_dim
        self.value_input_act_dim = alg_params.value_input_act_dim
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

    def package_value_inputs(self, batch_id):
        v_input = []
        next_v_input = []

        for dev_id in range(self.device_num):
            for i in range(self.edge_queue_obs_dim):
                _append_flat(v_input, self.edge_obss[batch_id][i])
                _append_flat(next_v_input, self.next_edge_obss[batch_id][i])
            _append_flat(v_input, self.device_obss[batch_id][dev_id])
            _append_flat(next_v_input, self.next_device_obss[batch_id][dev_id])

        return torch.tensor(v_input, dtype=torch.float32).reshape(1, -1), torch.tensor(next_v_input, dtype=torch.float32).reshape(1, -1)


    def package_policy_input(self, batch_id, device_id):
        p_input = []
        next_p_input = []

        # Global edge_obs (shared across all devices)
        for i in range(self.edge_queue_obs_dim):
            _append_flat(p_input, self.edge_obss[batch_id][i])
            _append_flat(next_p_input, self.next_edge_obss[batch_id][i])

        _append_flat(p_input, self.device_obss[batch_id][device_id])
        _append_flat(next_p_input, self.next_device_obss[batch_id][device_id])

        return torch.tensor(p_input, dtype=torch.float32).reshape(1, -1), torch.tensor(next_p_input, dtype=torch.float32).reshape(1, -1)

    '''
    batch_states:             v_net输入 [batch_size, state_dim] (all devices)
    batch_device_obss:        p_net输入 [batch_size, device_num, obs_dim]
    batch_joint_acts:         v_net输入 [batch_size, joint_act_dim] (all devices)
    batch_joint_rewards:      v_net输入 [batch_size, 1] (sum of all types)
    batch_next_states:        v_net输入 [batch_size, state_dim]
    batch_next_device_obss:   p_net输入 [batch_size, device_num, obs_dim]
    '''
    def sample(self, batch_ids):
        S = self.edge_queue_obs_dim // 2
        ae_dim = self.action_encode_dim

        batch_states = torch.zeros((len(batch_ids), self.value_input_obs_dim), dtype=torch.float32)
        batch_next_states = torch.zeros((len(batch_ids), self.value_input_obs_dim), dtype=torch.float32)
        batch_device_obss = torch.zeros([len(batch_ids), self.device_num, self.policy_input_dim], dtype=torch.float32)
        batch_next_device_obss = torch.zeros([len(batch_ids), self.device_num, self.policy_input_dim], dtype=torch.float32)
        batch_joint_acts = torch.zeros((len(batch_ids), self.value_input_act_dim), dtype=torch.float32)
        batch_joint_rewards = torch.zeros((len(batch_ids), 1), dtype=torch.float32)

        j = 0
        for id_ in batch_ids:
            state, next_state = self.package_value_inputs(id_)
            batch_states[j] = state.reshape(-1)
            batch_next_states[j] = next_state.reshape(-1)

            for i in range(self.device_num):
                device_obs, next_device_obs = self.package_policy_input(id_, i)
                batch_device_obss[j][i] = device_obs.reshape(-1)
                batch_next_device_obss[j][i] = next_device_obs.reshape(-1)

            joint_act = []
            for i in range(self.device_num):
                full_act = self.device_acts[id_][i]  # [S + 3*ae_dim]
                server_logits = full_act[:S]
                cont_all = full_act[S:S + 3 * ae_dim]
                cont_3 = [sum(cont_all[c * ae_dim : (c + 1) * ae_dim]) / ae_dim for c in range(3)]
                joint_act.extend(server_logits + cont_3)
            batch_joint_acts[j] = torch.tensor(joint_act, dtype=torch.float32).reshape(-1)

            # global reward: single scalar per slot
            batch_joint_rewards[j] = self.joint_rewards[id_]

            j += 1

        return batch_states, batch_device_obss, \
                batch_joint_acts, batch_joint_rewards, \
                batch_next_states, batch_next_device_obss
