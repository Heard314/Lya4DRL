import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import Normal
from network.value_net import MappoValueNet, MaddpgValueNet
from network.policy_net import MappoPolicyNet, MaddpgPolicyNet, MappoPolicyNetLSTM
import config.global_params as gp

class MappoEdgeAgent():
    def __init__(self, gen_params, alg_params):
        self.device_num = gen_params.device_num
        
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
        root_path = gp.settings.project_dir
        run_dir = gp.settings.run_dir
        self.weights_dir = gp.settings.weight_dir
        
        self.device = gp.settings.device
        print("The device for training is: ", self.device)
        # value network
        self.v_net = MappoValueNet(alg_params).to(self.device)
        self.v_optimizer = torch.optim.Adam(self.v_net.parameters(),
                                            lr = self.v_lr)
        
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
            v_path = self.weights_dir + f"v_net_params_{gen_params.resume_episode}.pkl"
            print("Loading value network from: ", v_path)
            self.v_net.load_state_dict(torch.load(v_path, map_location=self.device))
            print(f"Loading policy networks from: {self.weights_dir + "p_net_params.pkl"}")
            for i in range(self.device_num):
                p_path = self.weights_dir + "p_net_params_" + str(i) + f"_{gen_params.resume_episode}.pkl"
                self.p_nets[i].load_state_dict(torch.load(p_path, map_location=self.device))

    def train_nets(self, replay_buffer):
        '''training data'''
        # v_inputs: [train_freq x train_time_slots, state_dim]
        # v_tags: [train_freq x train_time_slots, 1] 价值网络的目标值
        # p_inputs: [train_freq x train_time_slots, device_num, obs_dim]
        # acts: [train_freq x train_time_slots, device_num, action_dim]
        # act_logprobs: [train_freq x train_time_slots, device_num, 1]
        # advs: [train_freq x train_time_slots, 1]
        # active_masks: [train_freq x train_time_slots, device_num, 1] 当前智能体是否需要处理任务
        v_inputs, v_tags, p_inputs, lstm_hidden_hs, lstm_hidden_cs, \
        acts, act_logprobs, advs, active_masks = replay_buffer.get_training_data(self.v_net)

        v_inputs       = v_inputs.to(self.device)
        v_tags         = v_tags.to(self.device)
        p_inputs       = p_inputs.to(self.device)
        acts           = acts.to(self.device)
        act_logprobs   = act_logprobs.to(self.device)
        advs           = advs.to(self.device)
        active_masks   = active_masks.to(self.device)
        if lstm_hidden_hs is not None:
            lstm_hidden_hs = lstm_hidden_hs.to(self.device)
        if lstm_hidden_cs is not None:
            lstm_hidden_cs = lstm_hidden_cs.to(self.device)

        self.train_value_net(v_inputs, v_tags)
        
        for i in range(self.device_num):
            # 训练策略网络依然只用局部信息
            self.train_policy_net(i, p_inputs[:, i], acts[:, i], act_logprobs[:, i], advs, active_masks=active_masks[:, i],lstm_hidden_hs=lstm_hidden_hs[:, i], lstm_hidden_cs=lstm_hidden_cs[:, i])
        
        if self.use_lr_decay:
            self.decay_lr()
    
    def train_value_net(self, v_inputs, v_tags):
        
        # print("cuda available:", torch.cuda.is_available())
        # print("v_net device:", next(self.v_net.parameters()).device)
        # print("sample batch device:", v_inputs.device)
        total_size = self.train_freq * self.train_time_slots
        for e in range(self.v_epochs):
            for ids in BatchSampler(SubsetRandomSampler(range(total_size)),
                                    self.train_batch_size, False):
                vs = self.v_net(v_inputs[ids])
                
                loss = F.mse_loss(v_tags[ids], vs)
                
                self.v_optimizer.zero_grad()
                loss.backward()
                
                # gradient clip
                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.v_net.parameters(), 
                                                   self.v_grad_clip)
                self.v_optimizer.step()

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
                
                #! 取出子批次的mask
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
           for params in self.v_optimizer.param_groups:
               params['lr'] = self.v_lr
       
       if self.p_lr > self.min_p_lr:  
           self.p_lr *= self.decay_fac
           for i in range(self.device_num):
               for params in self.p_optimizers[i].param_groups:
                   params['lr'] = self.p_lr
        
    def save_nets(self, e_id, seed):
        if not os.path.exists(self.weights_dir):
            os.makedirs(self.weights_dir)
            
        torch.save(self.v_net.state_dict(), 
                   self.weights_dir + "v_net_params_" + str(e_id) + ".pkl")
        
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

    # def train_nets(self, replay_buffer):
    #     '''training data'''
    #     # v_inputs: [train_freq x train_time_slots, state_dim]
    #     # v_tags: [train_freq x train_time_slots, 1] 价值网络的目标值
    #     # p_inputs: [train_freq x train_time_slots, device_num, obs_dim]
    #     # acts: [train_freq x train_time_slots, device_num, action_dim]
    #     # act_logprobs: [train_freq x train_time_slots, device_num, 1]
    #     # advs: [train_freq x train_time_slots, 1]
    #     # active_masks: [train_freq x train_time_slots, device_num, 1] 当前智能体是否需要处理任务
    #     v_inputs, v_tags, p_inputs, \
    #     acts, act_logprobs, advs, active_masks = replay_buffer.get_training_data(self.v_net)
                                    
    #     self.train_value_net(v_inputs, v_tags)
        
    #     for i in range(self.device_num):
    #         # 训练策略网络依然只用局部信息
    #         self.train_policy_net(i, p_inputs[:, i], acts[:, i], act_logprobs[:, i], advs, active_masks=active_masks[:, i])
        
    #     if self.use_lr_decay:
    #         self.decay_lr()

    # 按比例分配样本
    # def train_policy_net(self, agent_id, p_inputs, acts, act_logprobs, advs,
    #                     active_masks, lstm_hidden_hs=None, lstm_hidden_cs=None):
    #     total_size = self.train_freq * self.train_time_slots

    #     # ========= 1. 预先根据 active_masks 做一次划分 =========  #
    #     # 展平成 [total_size]
    #     mask_flat = active_masks.reshape(-1)
    #     # 索引 [0, 1, ..., total_size-1]
    #     all_ids = torch.arange(total_size, device=mask_flat.device)

    #     active_ids = all_ids[mask_flat > 0]      # 有任务样本
    #     inactive_ids = all_ids[mask_flat <= 0]   # 无任务样本

    #     # 如果 inactive 太少/没有，避免后面出错
    #     if inactive_ids.numel() == 0:
    #         inactive_ids = active_ids  # 退化成只用 active，但逻辑仍然成立

    #     # 样本比值 active : inactive ≈ 8 : 2
    #     active_ratio = 0.8
    #     batch_size = self.train_batch_size

    #     for e in range(self.p_epochs):
    #         # ========= 2. 每个 epoch 重新打乱 active / inactive =========  #
    #         perm_active = active_ids[torch.randperm(active_ids.numel(), device=active_ids.device)]
    #         perm_inactive = inactive_ids[torch.randperm(inactive_ids.numel(), device=inactive_ids.device)]

    #         print(f"[DEBUG] For the agent {agent_id}, the total active samples number is {perm_active.numel()}, the total inactive samples number is {perm_inactive.numel()}.")
    #         pa = 0
    #         pi = 0

    #         # ========= 3. 自己“手动”组成一堆 mini-batch =========  #
    #         while pa < perm_active.numel() or pi < perm_inactive.numel():
    #             # 理想中：每个 batch 希望拿多少 active / inactive
    #             ideal_active = int(batch_size * active_ratio)
    #             ideal_inactive = batch_size - ideal_active

    #             take_active = min(ideal_active, perm_active.numel() - pa)
    #             take_inactive = min(ideal_inactive, perm_inactive.numel() - pi)

    #             if take_active + take_inactive == 0:
    #                 break

    #             remaining = batch_size - (take_active + take_inactive)
    #             if remaining > 0:
    #                 extra_a = min(remaining, perm_active.numel() - (pa + take_active))
    #                 take_active += extra_a
    #                 remaining -= extra_a

    #             if remaining > 0:
    #                 # 再尝试多拿 inactive
    #                 extra_i = min(remaining, perm_inactive.numel() - (pi + take_inactive))
    #                 take_inactive += extra_i
    #                 remaining -= extra_i

    #             if take_active + take_inactive == 0:
    #                 break
    #             print(f"[DEBUG] For the agent {agent_id}, the active samples number is {take_active}, the inactive samples number is {take_inactive}.")
    #             batch_ids = torch.cat([
    #                 perm_active[pa:pa + take_active],
    #                 perm_inactive[pi:pi + take_inactive]
    #             ], dim=0)

    #             pa += take_active
    #             pi += take_inactive
    #             ids = batch_ids

    #             # mean: [train_batch_size, p_out_dim]
    #             # std:  [train_batch_size, p_out_dim]
    #             p_in = p_inputs[ids]

    #             if lstm_hidden_hs is not None:
    #                 h0 = lstm_hidden_hs[ids]      # [B, hid_dim]
    #                 h0 = h0.unsqueeze(0)          # [1, B, hid_dim]
    #             else:
    #                 h0 = None

    #             if lstm_hidden_cs is not None:
    #                 c0 = lstm_hidden_cs[ids]      # [B, hid_dim]
    #                 c0 = c0.unsqueeze(0)          # [1, B, hid_dim]
    #             else:
    #                 c0 = None

    #             mean, std, _ = self.p_nets[agent_id](p_in, (h0, c0))
    #             dist = Normal(mean, std)

    #             # [batch]
    #             enty = dist.entropy().sum(-1)
    #             # [batch]
    #             new_act_logprobs = dist.log_prob(acts[ids]).sum(-1)
    #             # [batch]
    #             old_act_logprobs = act_logprobs[ids].reshape([-1])
    #             ratios = torch.exp(new_act_logprobs - old_act_logprobs)

    #             #! 取出子批次的mask
    #             mask_b = active_masks[ids].reshape([-1]).to(acts.device, dtype=acts.dtype)
    #             adv_b = advs[ids].reshape([-1])

    #             # m = (mask_b > 0)
    #             # if m.any():
    #             #     adv_sel = adv_b[m]
    #             #     adv_sel = (adv_sel - adv_sel.mean()).div(adv_sel.std().clamp_min(1e-8))
    #             #     adv_b = adv_b.clone()
    #             #     adv_b[m] = adv_sel

    #             # PPO-clip
    #             surr1 = ratios * adv_b
    #             surr2 = torch.clamp(ratios, 1 - self.p_clip, 1 + self.p_clip) * adv_b

    #             # 只用有效样本算平均值
    #             denom = mask_b.sum().clamp_min(1.0)
    #             policy_loss = -(torch.min(surr1, surr2) * mask_b).sum() / denom
    #             ent_loss    = -(enty * self.enty_coef * mask_b).sum() / denom

    #             loss = policy_loss + ent_loss

    #             self.p_optimizers[agent_id].zero_grad()
    #             loss.backward()

    #             if self.use_grad_clip:
    #                 torch.nn.utils.clip_grad_norm_(self.p_nets[agent_id].parameters(),
    #                                             self.p_grad_clip)
    #             self.p_optimizers[agent_id].step()

    # 原始版本（样本均衡）
    # def train_policy_net(self, agent_id, p_inputs, acts, act_logprobs, advs, active_masks):
    #     total_size = self.train_freq * self.train_time_slots
    #     for e in range(self.p_epochs):
    #         for ids in BatchSampler(SubsetRandomSampler(range(total_size)),
    #                                     self.train_batch_size, False):
    #             # mean: [train_batch_size, p_out_dim]
    #             # std: [train_batch_size, p_out_dim]
    #             p_in = p_inputs[ids] 
    #             h0 = torch.zeros(1, len(ids), self.lstm_hidden_dim, device=p_in.device)
    #             c0 = torch.zeros(1, len(ids), self.lstm_hidden_dim, device=p_in.device)
    #             mean, std, _ = self.p_nets[agent_id](p_in, (h0, c0))
    #             dist = Normal(mean, std)
    #             # [train_batch_size]
    #             enty = dist.entropy().sum(-1)
    #             # [train_batch_size]
    #             new_act_logprobs = dist.log_prob(acts[ids]).sum(-1)
    #             # [train_batch_size]
    #             old_act_logprobs = act_logprobs[ids].reshape([-1])
    #             ratios = torch.exp(new_act_logprobs - old_act_logprobs)
                
    #             #! 取出子批次的mask
    #             mask_b = active_masks[ids].reshape([-1]).to(acts.device, dtype=acts.dtype)
    #             adv_b = advs[ids].reshape([-1])
    #             m = (mask_b > 0)
    #             if m.any():
    #                 adv_sel = adv_b[m]
    #                 adv_sel = (adv_sel - adv_sel.mean()).div(adv_sel.std().clamp_min(1e-8))
    #                 adv_b = adv_b.clone()
    #                 adv_b[m] = adv_sel
    #             # PPO-clip
    #             surr1 = ratios * adv_b
    #             surr2 = torch.clamp(ratios, 1 - self.p_clip, 1 + self.p_clip) * \
    #                     adv_b
                
    #             # 只用有效样本算平均值
    #             denom = mask_b.sum().clamp_min(1.0)
    #             policy_loss = -(torch.min(surr1, surr2) * mask_b).sum() / denom
    #             ent_loss    = -(enty * self.enty_coef * mask_b).sum() / denom

    #             loss = policy_loss + ent_loss
                
    #             self.p_optimizers[agent_id].zero_grad()
    #             loss.backward()
                
    #             # gradient clip
    #             if self.use_grad_clip:  
    #                 torch.nn.utils.clip_grad_norm_(self.p_nets[agent_id].parameters(),
    #                                                self.p_grad_clip)
    #             self.p_optimizers[agent_id].step()

    # 只使用active的样本
    # def train_policy_net(self, agent_id, p_inputs, acts, act_logprobs, advs, active_masks, lstm_hidden_hs=None, lstm_hidden_cs=None):
    #     total_size = self.train_freq * self.train_time_slots
    #     for e in range(self.p_epochs):
    #         for ids in BatchSampler(SubsetRandomSampler(range(total_size)),
    #                                     self.train_batch_size, False):
    #             # mean: [train_batch_size, p_out_dim]
    #             # std: [train_batch_size, p_out_dim]
    #             p_in = p_inputs[ids]
    #             if lstm_hidden_hs != None:
    #                 h0 = lstm_hidden_hs[ids]
    #                 h0 = h0.unsqueeze(0) 
    #             if lstm_hidden_cs != None:
    #                 c0 = lstm_hidden_cs[ids]
    #                 c0 = c0.unsqueeze(0)

    #             mean, std, _ = self.p_nets[agent_id](p_in, (h0, c0))
    #             dist = Normal(mean, std)
    #             # [train_batch_size]
    #             enty = dist.entropy().sum(-1)
    #             # [train_batch_size]
    #             new_act_logprobs = dist.log_prob(acts[ids]).sum(-1)
    #             # [train_batch_size]
    #             old_act_logprobs = act_logprobs[ids].reshape([-1])
    #             ratios = torch.exp(new_act_logprobs - old_act_logprobs)
                
    #             #! 取出子批次的mask
    #             mask_b = active_masks[ids].reshape([-1]).to(acts.device, dtype=acts.dtype)
    #             adv_b = advs[ids].reshape([-1])
    #             m = (mask_b > 0)
    #             if m.any():
    #                 adv_sel = adv_b[m]
    #                 adv_sel = (adv_sel - adv_sel.mean()).div(adv_sel.std().clamp_min(1e-8))
    #                 adv_b = adv_b.clone()
    #                 adv_b[m] = adv_sel
    #             # PPO-clip
    #             surr1 = ratios * adv_b
    #             surr2 = torch.clamp(ratios, 1 - self.p_clip, 1 + self.p_clip) * \
    #                     adv_b
                
    #             # 只用有效样本算平均值
    #             denom = mask_b.sum().clamp_min(1.0)
    #             policy_loss = -(torch.min(surr1, surr2) * mask_b).sum() / denom
    #             ent_loss    = -(enty * self.enty_coef * mask_b).sum() / denom

    #             loss = policy_loss + ent_loss
                
    #             self.p_optimizers[agent_id].zero_grad()
    #             loss.backward()
                
    #             # gradient clip
    #             if self.use_grad_clip:  
    #                 torch.nn.utils.clip_grad_norm_(self.p_nets[agent_id].parameters(),
    #                                                self.p_grad_clip)
    #             self.p_optimizers[agent_id].step()
            
class MaddpgEdgeAgent():
    def __init__(self, gen_params, alg_params):
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
        root_path = gp.settings.project_dir
        run_dir = gp.settings.run_dir
        self.weights_dir = gp.settings.weight_dir
        # learning-rate decay
        self.use_lr_decay = alg_params.use_lr_decay
        self.min_v_lr = alg_params.min_v_lr
        self.min_p_lr = alg_params.min_p_lr
        self.decay_intl = alg_params.decay_intl
        self.decay_fac = alg_params.decay_fac
        
        # value network
        self.v_net = MaddpgValueNet(alg_params)
        # target value network 
        self.target_v_net = MaddpgValueNet(alg_params) 
        self.target_v_net.load_state_dict(self.v_net.state_dict())
        # optimizer
        self.v_optimizer = torch.optim.Adam(self.v_net.parameters(),
                                            lr = self.v_lr)
        
        self.p_nets = []
        self.target_p_nets = []
        self.p_optimizers = []
        for i in range(self.device_num):
            # policy network
            p_net = MaddpgPolicyNet(alg_params)
            self.p_nets.append(p_net)
            # target policy network
            target_p_net = MaddpgPolicyNet(alg_params)
            target_p_net.load_state_dict(p_net.state_dict())
            self.target_p_nets.append(target_p_net)
            # optimizer
            p_optimizer = torch.optim.Adam(p_net.parameters(),
                                            lr = self.p_lr)
            self.p_optimizers.append(p_optimizer)
            
        # load networks' weights
        if gen_params.load_weights:
            v_path = self.weights_dir + "v_net_params.pkl"
            self.v_net.load_state_dict(torch.load(v_path))
            target_v_path = self.weights_dir + "target_v_net_params.pkl"
            self.target_v_net.load_state_dict(torch.load(target_v_path))
            
            for i in range(self.device_num):
                p_path = self.weights_dir + "p_net_params_" + str(i) + ".pkl"
                self.p_nets[i].load_state_dict(torch.load(p_path))
                target_p_path = self.weights_dir + "target_p_net_params_" + str(i) + ".pkl"
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
            # batch_states: [batch_size, state_dim]
            # batch_device_obss: [batch_size, device_num, obs_dim]
            # batch_joint_acts: [batch_size, joint_act_dim]
            # batch_joint_rewards: [batch_size, 1]
            # batch_next_states: [batch_size, state_dim]
            # batch_next_device_obss: [batch_size, device_num, obs_dim]
            batch_states, batch_device_obss, \
            batch_joint_acts, batch_joint_rewards, \
            batch_next_states, batch_next_device_obss = replay_buffer.sample(batch_ids)
            
            self.train_value_net(batch_states, batch_joint_acts, 
                                 batch_joint_rewards, 
                                 batch_next_states, batch_next_device_obss)
            
            for i in range(self.device_num):
                self.train_policy_net(i, batch_states, batch_device_obss[:, i], 
                                      batch_joint_acts)
            
            if self.use_lr_decay:
                self.decay_lr(total_time_slots)
    
    def train_value_net(self, batch_states, batch_joint_acts, 
                              batch_joint_rewards, 
                              batch_next_states, batch_next_device_obss):
        with torch.no_grad():
            batch_next_joint_acts = []
            for i in range(self.device_num):
                batch_next_acts = self.target_p_nets[i](batch_next_device_obss[:, i])
                batch_next_joint_acts.append(batch_next_acts)
            # [batch_size, joint_act_dim]
            batch_next_joint_acts = torch.concat(batch_next_joint_acts, dim = -1)
            # [batch_size, 1]
            next_qs = self.target_v_net(batch_next_states, batch_next_joint_acts)
            target_qs = batch_joint_rewards + self.gamma * next_qs
            # normalization
            target_qs = (target_qs - target_qs.mean()) / (target_qs.std() + 1e-5)
            
        for i in range(self.v_epochs):
            # [batch_size, 1]
            qs = self.v_net(batch_states, batch_joint_acts)
            
            v_loss = F.mse_loss(target_qs, qs)
            
            self.v_optimizer.zero_grad()
            v_loss.backward()
            # gradient clip
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.v_net.parameters(), 
                                               self.v_grad_clip)
            self.v_optimizer.step()
            
    def train_policy_net(self, agent_id, batch_states, batch_device_obss, batch_joint_acts):
        for i in range(self.p_epochs):
            batch_joint_acts_ = batch_joint_acts.clone()
            batch_acts = self.p_nets[agent_id](batch_device_obss)
            batch_joint_acts_[:, agent_id * self.action_dim:
                                (agent_id + 1) * self.action_dim] = batch_acts
            
            p_loss = (-self.v_net(batch_states, batch_joint_acts_)).mean()
            
            self.p_optimizers[agent_id].zero_grad()
            p_loss.backward()
            # gradient clip
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.p_nets[agent_id].parameters(), 
                                               self.p_grad_clip)
            self.p_optimizers[agent_id].step()
            
    def update_target_nets(self, total_time_slots):
        if total_time_slots >= self.warm_time_slots:
            self.target_v_net.load_state_dict(self.v_net.state_dict())
            
            for i in range(self.device_num):
                self.target_p_nets[i].load_state_dict(self.p_nets[i].state_dict())
                
    def decay_lr(self, total_time_slots):
        if total_time_slots % self.decay_intl == 0:
            if self.v_lr > self.min_v_lr:
                self.v_lr -= self.decay_fac
                for params in self.v_optimizer.param_groups:
                    params['lr'] = self.v_lr
            
            if self.p_lr > self.min_p_lr:
                self.p_lr -= self.decay_fac
                for i in range(self.device_num):
                    for params in self.p_optimizers[i].param_groups:
                        params['lr'] = self.p_lr
        
    def save_nets(self, total_time_slots):
        if not os.path.exists(self.weights_dir):
            os.makedirs(self.weights_dir)
            
        torch.save(self.v_net.state_dict(),
                   self.weights_dir + "v_net_params_" + 
                   str(total_time_slots) + ".pkl")
        torch.save(self.target_v_net.state_dict(),
                   self.weights_dir + "target_v_net_params_" + 
                   str(total_time_slots) + ".pkl")
        
        for i in range(self.device_num):
            torch.save(self.p_nets[i].state_dict(),
                       self.weights_dir + "p_net_params_" + 
                       str(i) + "_" + str(total_time_slots) + ".pkl")
            torch.save(self.target_p_nets[i].state_dict(),
                       self.weights_dir + "target_p_net_params_" + 
                       str(i) + "_" + str(total_time_slots) + ".pkl")