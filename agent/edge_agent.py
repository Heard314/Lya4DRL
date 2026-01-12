import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import Normal
from network.value_net import MappoValueNet, MaddpgValueNet
from network.policy_net import MaddpgPolicyNetLSTM, MappoPolicyNet, MaddpgPolicyNet, MappoPolicyNetLSTM
import config.global_params as gp

class MappoEdgeAgent():
    def __init__(self, gen_params, alg_params):
        self.device_num = gen_params.device_num
        self.device_in_types = gen_params.device_in_types
        # training
        self.train_time_slots = alg_params.train_time_slots
        self.train_freq = alg_params.train_freq
        self.train_batch_size = alg_params.train_batch_size
        self.v_epochs = alg_params.v_epochs
        self.p_epochs = alg_params.p_epochs
        self.p_clip = alg_params.p_clip
        self.enty_coef = alg_params.enty_coef
        self.v_lr = alg_params.v_lr
        self.p_lr = alg_params.p_lr
        self.gamma = alg_params.gamma
        # gradient clip
        self.use_grad_clip = alg_params.use_grad_clip
        self.v_grad_clip = alg_params.v_grad_clip
        self.p_grad_clip = alg_params.p_grad_clip
        # learning-rate decay
        self.use_lr_decay = alg_params.use_lr_decay
        self.min_v_lr = alg_params.min_v_lr
        self.min_p_lr = alg_params.min_p_lr
        self.decay_fac = alg_params.decay_fac
        root_path = gp.settings.exp_result_dir
        run_dir = gp.settings.run_dir
        self.weights_dir = gp.settings.weight_dir
        
        self.device = gp.settings.device
        print("The device for training is: ", self.device)

        self.device_type_num = gen_params.device_type_num

        # value network
        self.v_nets = []
        self.v_optimizers = []
        for i in range(self.device_type_num):
            v_net = MappoValueNet(alg_params, i).to(self.device)
            self.v_nets.append(v_net)
            v_optimizer = torch.optim.Adam(v_net.parameters(),
                                            lr = self.v_lr)
            self.v_optimizers.append(v_optimizer)
        # LSTM hidden dim
        self.lstm_hidden_dim = alg_params.p_hid_dims[1]
        # policy networks
        self.p_nets = []
        self.p_optimizers = []
        for i in range(self.device_num):
            # p_net = MappoPolicyNet(alg_params)
            p_net = MappoPolicyNetLSTM(alg_params).to(self.device)
            self.p_nets.append(p_net)
                
            p_optimizer = torch.optim.Adam(p_net.parameters(),
                                           lr = self.p_lr)
            self.p_optimizers.append(p_optimizer)
            
        # load networks' weights 

        if gen_params.load_weights:
            print(f"Loading value network from: {self.weights_dir}v_net_params.pkl")
            for i in range(self.device_type_num):
                v_path = self.weights_dir + "v_net_params_" + str(i) + f"{gen_params.resume_episode}.pkl"
                self.v_nets[i].load_state_dict(torch.load(v_path, map_location=self.device))
            print(f"Loading policy networks from: {self.weights_dir}p_net_params.pkl")
            for i in range(self.device_num):
                p_path = self.weights_dir + "p_net_params_" + str(i) + f"_{gen_params.resume_episode}.pkl"
                self.p_nets[i].load_state_dict(torch.load(p_path, map_location=self.device))

    def train_nets(self, replay_buffers):
        '''training data'''
        # v_inputs: [train_freq x train_time_slots, state_dim]
        # v_tags: [train_freq x train_time_slots, 1] 价值网络的目标值
        # p_inputs: [train_freq x train_time_slots, device_num, obs_dim]
        # acts: [train_freq x train_time_slots, device_num, action_dim]
        # act_logprobs: [train_freq x train_time_slots, device_num, 1]
        # advs: [train_freq x train_time_slots, 1]
        # active_masks: [train_freq x train_time_slots, device_num, 1]
        p_inputs, lstm_hidden_hs, lstm_hidden_cs, \
        acts, act_logprobs, active_masks = replay_buffers.get_policy_net_training_data()
        p_inputs       = p_inputs.to(self.device)
        acts           = acts.to(self.device)
        act_logprobs   = act_logprobs.to(self.device)
        active_masks   = active_masks.to(self.device)
        if lstm_hidden_hs is not None:
            lstm_hidden_hs = lstm_hidden_hs.to(self.device)
        if lstm_hidden_cs is not None:
            lstm_hidden_cs = lstm_hidden_cs.to(self.device)
        for k in range(self.device_type_num):
            v_inputs_, v_tags_, advs_ = replay_buffers.get_value_net_training_data(k, self.v_nets[k])
            v_inputs_       = v_inputs_.to(self.device)
            v_tags_         = v_tags_.to(self.device)
            advs_           = advs_.to(self.device)
            self.train_value_net(k, v_inputs_, v_tags_)

            for i in self.device_in_types[k]:
                # 训练策略网络依然只用局部信息
                self.train_policy_net(i, p_inputs[:, i], acts[:, i], act_logprobs[:, i], advs_, active_masks=active_masks[:, i],lstm_hidden_hs=lstm_hidden_hs[:, i], lstm_hidden_cs=lstm_hidden_cs[:, i])
            
        if self.use_lr_decay:
            self.decay_lr()
    
    def train_value_net(self, queue_id, v_inputs, v_tags):
        
        total_size = self.train_freq * self.train_time_slots
        for e in range(self.v_epochs):
            for ids in BatchSampler(SubsetRandomSampler(range(total_size)),
                                    self.train_batch_size, False):
                vs = self.v_nets[queue_id](v_inputs[ids])
                
                loss = F.mse_loss(v_tags[ids], vs)
                
                self.v_optimizers[queue_id].zero_grad()
                loss.backward()
                
                # gradient clip
                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.v_nets[queue_id].parameters(), 
                                                   self.v_grad_clip)
                self.v_optimizers[queue_id].step()

    def train_policy_net(self, agent_id, p_inputs, acts, act_logprobs, advs,
                          active_masks,lstm_hidden_hs=None, lstm_hidden_cs=None):
        total_size = self.train_freq * self.train_time_slots
        for e in range(self.p_epochs):
            for ids in BatchSampler(SubsetRandomSampler(range(total_size)),
                                        self.train_batch_size, False):
                # mean: [train_batch_size, p_out_dim]
                # std: [train_batch_size, p_out_dim]
                p_in = p_inputs[ids] 
                if lstm_hidden_hs is not None:
                    h0 = lstm_hidden_hs[ids]      # [B, hid_dim]
                    h0 = h0.unsqueeze(0)          # [1, B, hid_dim]
                else:
                    h0 = None

                if lstm_hidden_cs is not None:
                    c0 = lstm_hidden_cs[ids]      # [B, hid_dim]
                    c0 = c0.unsqueeze(0)          # [1, B, hid_dim]
                else:
                    c0 = None
                mean, std, _ = self.p_nets[agent_id](p_in, (h0, c0))
                dist = Normal(mean, std)
                # [train_batch_size]
                enty = dist.entropy().sum(-1)
                # [train_batch_size]
                new_act_logprobs = dist.log_prob(acts[ids]).sum(-1)
                # [train_batch_size]
                old_act_logprobs = act_logprobs[ids].reshape([-1])
                ratios = torch.exp(new_act_logprobs - old_act_logprobs)
                
                mask_b = active_masks[ids].reshape([-1]).to(acts.device, dtype=acts.dtype)
                adv_b = advs[ids].reshape([-1])
                m = (mask_b > 0)
                if m.any():
                    adv_sel = adv_b[m]
                    adv_sel = (adv_sel - adv_sel.mean()).div(adv_sel.std().clamp_min(1e-8))
                    adv_b = adv_b.clone()
                    adv_b[m] = adv_sel
                
                # PPO-clip
                surr1 = ratios * adv_b
                surr2 = torch.clamp(ratios, 1 - self.p_clip, 1 + self.p_clip) * \
                        adv_b
                
                # 只用有效样本算平均值
                denom = mask_b.sum().clamp_min(1.0)
                policy_loss = -(torch.min(surr1, surr2) * mask_b).sum() / denom
                ent_loss    = -(enty * self.enty_coef * mask_b).sum() / denom

                loss = policy_loss + ent_loss
                
                self.p_optimizers[agent_id].zero_grad()
                loss.backward()
                
                # gradient clip
                if self.use_grad_clip:  
                    torch.nn.utils.clip_grad_norm_(self.p_nets[agent_id].parameters(),
                                                   self.p_grad_clip)
                self.p_optimizers[agent_id].step()

    def decay_lr(self):
        if self.v_lr > self.min_v_lr:  
            self.v_lr *= self.decay_fac
            for i in range(self.device_type_num):
                for params in self.v_optimizers[i].param_groups:
                    params['lr'] = self.v_lr
       
        if self.p_lr > self.min_p_lr:  
            self.p_lr *= self.decay_fac
            for i in range(self.device_num):
                for params in self.p_optimizers[i].param_groups:
                    params['lr'] = self.p_lr
        
    def save_nets(self, e_id, seed):
        if not os.path.exists(self.weights_dir):
            os.makedirs(self.weights_dir)
        
        for i in range(self.device_type_num):
            torch.save(self.v_nets[i].state_dict(), 
                    self.weights_dir + "v_net_params_" + str(i) + "_" + str(e_id) + ".pkl")
        
        for i in range(self.device_num):
            torch.save(self.p_nets[i].state_dict(),
                       self.weights_dir + "p_net_params_" + str(i) + "_" + str(e_id) + ".pkl")        

        train_info = {
            "resume_episode": e_id,
            "train_seed": seed,
            "run_dir": gp.settings.run_dir
        }
        with open(self.weights_dir + f"train_info_{e_id}.pkl", "wb") as f:
            # save training meta info (overwritten every time)
            pickle.dump(train_info, f)
            
class MaddpgEdgeAgent():
    def __init__(self, gen_params, alg_params):
        # general 
        self.device_type_num = gen_params.device_type_num
        self.device_in_types = gen_params.device_in_types
        self.device = gp.settings.device

        self.device_num = gen_params.device_num
        self.action_dim = alg_params.action_dim
        
        # training
        self.warm_time_slots = alg_params.warm_time_slots
        self.train_batch_size = alg_params.train_batch_size
        self.v_epochs = alg_params.v_epochs
        self.p_epochs = alg_params.p_epochs
        self.buffer_size = alg_params.buffer_size
        self.gamma = alg_params.gamma
        self.v_lr = alg_params.v_lr
        self.p_lr = alg_params.p_lr
        # gradient clip
        self.use_grad_clip = alg_params.use_grad_clip
        self.v_grad_clip = alg_params.v_grad_clip
        self.p_grad_clip = alg_params.p_grad_clip
        root_path = gp.settings.exp_result_dir
        run_dir = gp.settings.run_dir
        self.weights_dir = gp.settings.weight_dir
        # learning-rate decay
        self.use_lr_decay = alg_params.use_lr_decay
        self.min_v_lr = alg_params.min_v_lr
        self.min_p_lr = alg_params.min_p_lr
        self.decay_intl = alg_params.decay_intl
        self.decay_fac = alg_params.decay_fac
        
        self.v_nets = []
        self.target_v_nets = []
        self.v_optimizers = []
        for i in range(self.device_type_num):
            # value network
            v_net = MaddpgValueNet(alg_params, i).to(self.device)
            self.v_nets.append(v_net)
            # target value network
            target_v_net = MaddpgValueNet(alg_params, i).to(self.device)
            target_v_net.load_state_dict(v_net.state_dict())
            self.target_v_nets.append(target_v_net)
            # optimizer
            v_optimizer = torch.optim.Adam(v_net.parameters(),
                                            lr = self.v_lr)
            self.v_optimizers.append(v_optimizer)

        self.p_nets = []
        self.target_p_nets = []
        self.p_optimizers = []
        for i in range(self.device_num):
            # policy network
            p_net = MaddpgPolicyNetLSTM(alg_params).to(self.device)
            self.p_nets.append(p_net)
            # target policy network
            target_p_net = MaddpgPolicyNetLSTM(alg_params).to(self.device)
            target_p_net.load_state_dict(p_net.state_dict())
            self.target_p_nets.append(target_p_net)
            # optimizer
            p_optimizer = torch.optim.Adam(p_net.parameters(),
                                            lr = self.p_lr)
            self.p_optimizers.append(p_optimizer)
            
        # load networks' weights
        if gen_params.load_weights:
            for i in range(self.device_type_num):
                v_path = self.weights_dir + "v_net_params_" + str(i) + f"{gen_params.resume_episode}.pkl"
                self.v_nets[i].load_state_dict(torch.load(v_path, map_location=self.device))
                target_v_path = self.weights_dir + "target_v_net_params_" + str(i) + f"{gen_params.resume_episode}.pkl"
                self.target_v_nets[i].load_state_dict(torch.load(target_v_path, map_location=self.device))
        
            for i in range(self.device_num):
                p_path = self.weights_dir + "p_net_params_" + str(i) + f"_{gen_params.resume_episode}.pkl"
                self.p_nets[i].load_state_dict(torch.load(p_path))
                target_p_path = self.weights_dir + "target_p_net_params_" + str(i) + f"_{gen_params.resume_episode}.pkl"
                self.target_p_nets[i].load_state_dict(torch.load(target_p_path))
        
    def train_nets(self, total_time_slots, replay_buffer):
        if total_time_slots >= self.warm_time_slots:
            if total_time_slots < self.buffer_size:
                batch_ids = np.random.choice(range(total_time_slots),
                                             self.train_batch_size, replace = False)
            else:
                batch_ids = np.random.choice(range(self.buffer_size),
                                             self.train_batch_size, replace = False)
            
            '''training data'''
            # batch_states: [device_type_num, batch_size, state_dim]
            # batch_device_obss: [batch_size, device_num, obs_dim]
            # batch_joint_acts: [device_type_num, batch_size, joint_act_dim]
            # batch_joint_rewards: [device_type_num, batch_size, device_type_num]
            # batch_next_states: [device_type_num, batch_size, state_dim]
            # batch_next_device_obss: [batch_size, device_num, obs_dim]
            batch_states, batch_device_obss, \
            batch_joint_acts, batch_joint_rewards, \
            batch_next_states, batch_next_device_obss = replay_buffer.sample(batch_ids)

            batch_states = batch_states
            batch_device_obss = batch_device_obss
            batch_joint_acts = batch_joint_acts
            batch_joint_rewards = batch_joint_rewards
            batch_next_states = batch_next_states
            batch_next_device_obss = batch_next_device_obss

            for i in range(self.device_type_num):
                self.train_value_net(i, batch_states[i], batch_joint_acts[i], 
                                    batch_joint_rewards[i],
                                    batch_next_states[i], batch_next_device_obss)
            
            for i in range(self.device_type_num):
                for j in self.device_in_types[i]:
                    self.train_policy_net(j, i, batch_states[i], batch_device_obss[:, j], 
                                          batch_joint_acts[i])

            if self.use_lr_decay:
                self.decay_lr(total_time_slots)

    def train_value_net(self, queue_id, batch_states, batch_joint_acts, 
                              batch_joint_rewards, 
                              batch_next_states, batch_next_device_obss):
        batch_states = batch_states.to(self.device)
        batch_joint_acts = batch_joint_acts.to(self.device)
        batch_joint_rewards = batch_joint_rewards.to(self.device)
        batch_next_states = batch_next_states.to(self.device)
        batch_next_device_obss = batch_next_device_obss.to(self.device)
        with torch.no_grad():
            batch_next_joint_acts = []
            for i in self.device_in_types[queue_id]:
                batch_next_acts, _ = self.target_p_nets[i](batch_next_device_obss[:, i])
                batch_next_joint_acts.append(batch_next_acts)
                # print(f"batch_next_acts shape: {len(batch_next_acts)}")
                # print(f"batch_next_acts : {batch_next_acts}")
            # [batch_size, joint_act_dim]
            batch_next_joint_acts = torch.concat(batch_next_joint_acts, dim = -1).squeeze(1)
            # [batch_size, 1]
            next_qs = self.target_v_nets[queue_id](batch_next_states, batch_next_joint_acts)
            target_qs = batch_joint_rewards + self.gamma * next_qs
            # normalization
            target_qs = (target_qs - target_qs.mean()) / (target_qs.std() + 1e-5)
            
        for i in range(self.v_epochs):
            # [batch_size, 1]
            qs = self.v_nets[queue_id](batch_states, batch_joint_acts)
            
            v_loss = F.mse_loss(target_qs, qs)
            
            self.v_optimizers[queue_id].zero_grad()
            v_loss.backward()
            # gradient clip
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.v_nets[queue_id].parameters(), 
                                               self.v_grad_clip)
            self.v_optimizers[queue_id].step()
            
    def train_policy_net(self, agent_id, queue_id, batch_states, batch_device_obss, batch_joint_acts):
        batch_states = batch_states.to(self.device)
        batch_device_obss = batch_device_obss.to(self.device)
        batch_joint_acts = batch_joint_acts.to(self.device)
        agent_id_in_type = agent_id - self.device_in_types[queue_id][0]
        for i in range(self.p_epochs):
            batch_joint_acts_ = batch_joint_acts.clone()
            batch_acts, _ = self.p_nets[agent_id](batch_device_obss)
            batch_acts = batch_acts.squeeze(1)
            batch_joint_acts_[:, agent_id_in_type * self.action_dim:
                                (agent_id_in_type + 1) * self.action_dim] = batch_acts

            p_loss = (-self.v_nets[queue_id](batch_states, batch_joint_acts_)).mean()

            self.p_optimizers[agent_id].zero_grad()
            p_loss.backward()
            # gradient clip
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.p_nets[agent_id].parameters(), 
                                               self.p_grad_clip)
            self.p_optimizers[agent_id].step()
            
    def update_target_nets(self, total_time_slots):
        if total_time_slots >= self.warm_time_slots:
            for i in range(self.device_type_num):
                self.target_v_nets[i].load_state_dict(self.v_nets[i].state_dict())
            
            for i in range(self.device_num):
                self.target_p_nets[i].load_state_dict(self.p_nets[i].state_dict())
                
    def decay_lr(self, total_time_slots):
        if total_time_slots % self.decay_intl == 0:
            if self.v_lr > self.min_v_lr:
                self.v_lr -= self.decay_fac
                for i in range(self.device_type_num):
                    for params in self.v_optimizers[i].param_groups:
                        params['lr'] = self.v_lr
            
            if self.p_lr > self.min_p_lr:
                self.p_lr -= self.decay_fac
                for i in range(self.device_num):
                    for params in self.p_optimizers[i].param_groups:
                        params['lr'] = self.p_lr
        
    def save_nets(self, total_time_slots):
        if not os.path.exists(self.weights_dir):
            os.makedirs(self.weights_dir)
        for i in range(self.device_type_num):
            torch.save(self.v_nets[i].state_dict(),
                    self.weights_dir + "v_net_params_" + 
                    str(i) + "_" + str(total_time_slots) + ".pkl")
            torch.save(self.target_v_nets[i].state_dict(),
                    self.weights_dir + "target_v_net_params_" + 
                    str(i) + "_" + str(total_time_slots) + ".pkl")
        
        for i in range(self.device_num):
            torch.save(self.p_nets[i].state_dict(),
                       self.weights_dir + "p_net_params_" + 
                       str(i) + "_" + str(total_time_slots) + ".pkl")
            torch.save(self.target_p_nets[i].state_dict(),
                       self.weights_dir + "target_p_net_params_" + 
                       str(i) + "_" + str(total_time_slots) + ".pkl")