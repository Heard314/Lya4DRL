import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import Normal
from network.value_net import MappoValueNet, MaddpgValueNet
from network.policy_net import MaddpgPolicyNet, MappoPolicyNet
import config.global_params as gp

class MappoEdgeAgent():
    def __init__(self, gen_params, alg_params):
        self.device_num = gen_params.device_num
        self.device_in_types = gen_params.device_in_types
        # training
        self.train_time_slots = alg_params.train_time_slots
        self.buffer_train_time_slots = alg_params.buffer_train_time_slots
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

        # value network (single global critic)
        self.v_net = MappoValueNet(alg_params).to(self.device)
        self.v_optimizer = torch.optim.Adam(self.v_net.parameters(), lr=self.v_lr)
        # policy networks
        self.p_nets = []
        self.p_optimizers = []
        for i in range(self.device_num):
            p_net = MappoPolicyNet(alg_params).to(self.device)
            self.p_nets.append(p_net)

            p_optimizer = torch.optim.Adam(p_net.parameters(),
                                           lr = self.p_lr)
            self.p_optimizers.append(p_optimizer)
            
        # load networks' weights
        if gen_params.load_weights:
            v_path = self.weights_dir + "v_net_params_" + str(gen_params.resume_episode) + ".pkl"
            print(f"Loading value network from: {v_path}")
            self.v_net.load_state_dict(torch.load(v_path, map_location=self.device))
            print(f"Loading policy networks from: {self.weights_dir}p_net_params.pkl")
            for i in range(self.device_num):
                p_path = self.weights_dir + "p_net_params_" + str(i) + f"_{gen_params.resume_episode}.pkl"
                print(f"[DEBUG] Loading policy network {i} from: ", p_path)
                self.p_nets[i].load_state_dict(torch.load(p_path, map_location=self.device))

    def train_nets(self, replay_buffers):
        '''training data'''
        # v_inputs: [train_freq x buffer_train_time_slots, state_dim]
        # v_tags: [train_freq x buffer_train_time_slots, 1]
        # p_inputs: [train_freq x buffer_train_time_slots, device_num, obs_dim]
        # acts: [train_freq x buffer_train_time_slots, device_num, action_dim]
        # act_logprobs: [train_freq x buffer_train_time_slots, device_num, 1]
        # advs: [train_freq x buffer_train_time_slots, 1]
        # active_masks: [train_freq x buffer_train_time_slots, device_num, 1]
        p_inputs, \
        acts, act_logprobs, active_masks = replay_buffers.get_policy_net_training_data()
        p_inputs       = p_inputs.to(self.device)
        acts           = acts.to(self.device)
        act_logprobs   = act_logprobs.to(self.device)
        active_masks   = active_masks.to(self.device)

        v_inputs_, v_tags_, advs_ = replay_buffers.get_value_net_training_data(self.v_net)
        v_inputs_       = v_inputs_.to(self.device)
        v_tags_         = v_tags_.to(self.device)
        advs_           = advs_.to(self.device)
        self.train_value_net(v_inputs_, v_tags_)

        for i in range(self.device_num):
            self.train_policy_net(i, p_inputs[:, i], acts[:, i], act_logprobs[:, i], advs_, active_masks=active_masks[:, i])

        if self.use_lr_decay:
            self.decay_lr()

    def train_value_net(self, v_inputs, v_tags):

        total_size = self.train_freq * self.buffer_train_time_slots
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
                          active_masks):
        total_size = self.train_freq * self.buffer_train_time_slots
        for e in range(self.p_epochs):
            for ids in BatchSampler(SubsetRandomSampler(range(total_size)),
                                        self.train_batch_size, False):
                p_in = p_inputs[ids]
                mean, std = self.p_nets[agent_id](p_in)
                dist = Normal(mean, std)

                # Calcuate new act logprobs
                # Reconstruct u from action (inverse tanh + rescale)
                scale = self.p_nets[agent_id].act_scale
                loc   = self.p_nets[agent_id].act_bias

                # acts: env action -> [-1,1]
                a = (acts[ids] - loc) / scale
                a = torch.clamp(a, -1 + 1e-6, 1 - 1e-6)

                # inverse tanh
                u = 0.5 * (torch.log1p(a) - torch.log1p(-a))

                normal_logp = dist.log_prob(u).sum(-1)
                squash = torch.log(1 - a.pow(2) + 1e-6).sum(-1)
                scale_logsum = torch.log(scale).sum(-1)
                # [train_batch_size]
                new_act_logprobs = normal_logp - squash - scale_logsum

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
                # [train_batch_size]
                enty = dist.entropy().sum(-1)
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
            
class MaddpgEdgeAgent():
    def __init__(self, gen_params, alg_params):
        # general 
        self.device_type_num = gen_params.device_type_num
        self.device_in_types = gen_params.device_in_types
        self.device = gp.settings.device

        self.device_num = gen_params.device_num
        self.action_dim = alg_params.action_dim
        self.action_encode_dim = alg_params.action_encode_dim
        self.edge_server_num = gen_params.edge_server_num
        
        # training
        self.gen_task_cycle = gen_params.gen_task_cycle
        self.warm_time_slots = alg_params.warm_time_slots
        self.train_batch_size = alg_params.train_batch_size
        self.v_epochs = alg_params.v_epochs
        self.p_epochs = alg_params.p_epochs
        self.buffer_size = alg_params.buffer_size
        self.gamma = alg_params.gamma
        self.v_lr = alg_params.v_lr
        self.p_lr = alg_params.p_lr

        self.train_update_cnt = 0
        self.critic_updates_round = alg_params.critic_updates_round
        self.policy_delay_round = alg_params.policy_delay_round

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
        self.p_decay_fac = alg_params.p_decay_fac
        self.v_decay_fac = alg_params.v_decay_fac
        self.tau = alg_params.tau

        # single global value network
        self.v_net = MaddpgValueNet(alg_params).to(self.device)
        self.target_v_net = MaddpgValueNet(alg_params).to(self.device)
        self.target_v_net.load_state_dict(self.v_net.state_dict())
        self.v_optimizer = torch.optim.Adam(self.v_net.parameters(), lr=self.v_lr)

        self.p_nets = []
        self.target_p_nets = []
        self.p_optimizers = []
        for i in range(self.device_num):
            # policy network
            p_net = MaddpgPolicyNet(alg_params).to(self.device)
            self.p_nets.append(p_net)
            # target policy network
            target_p_net = MaddpgPolicyNet(alg_params).to(self.device)
            target_p_net.load_state_dict(p_net.state_dict())
            self.target_p_nets.append(target_p_net)
            # optimizer
            p_optimizer = torch.optim.Adam(p_net.parameters(),
                                            lr = self.p_lr)
            self.p_optimizers.append(p_optimizer)

        # load networks' weights
        if gen_params.load_weights:
            v_path = self.weights_dir + "v_net_params_" + f"{gen_params.resume_episode}.pkl"
            self.v_net.load_state_dict(torch.load(v_path, map_location=self.device))
            target_v_path = self.weights_dir + "target_v_net_params_" + f"{gen_params.resume_episode}.pkl"
            self.target_v_net.load_state_dict(torch.load(target_v_path, map_location=self.device))

            for i in range(self.device_num):
                p_path = self.weights_dir + "p_net_params_" + str(i) + f"_{gen_params.resume_episode}.pkl"
                self.p_nets[i].load_state_dict(torch.load(p_path))
                target_p_path = self.weights_dir + "target_p_net_params_" + str(i) + f"_{gen_params.resume_episode}.pkl"
                self.target_p_nets[i].load_state_dict(torch.load(target_p_path))
        
    def train_nets(self, total_time_slots, replay_buffer):
        if total_time_slots >= self.warm_time_slots:
            batch_slots = (total_time_slots + self.gen_task_cycle - 1)//self.gen_task_cycle
            if batch_slots < self.buffer_size:
                # Buffer not full yet — sample only from valid range
                if batch_slots < self.train_batch_size:
                    batch_ids = np.random.choice(range(batch_slots),
                                                    self.train_batch_size, replace=True)
                else:
                    batch_ids = np.random.choice(range(batch_slots),
                                                    self.train_batch_size, replace=False)
            else:
                batch_ids = np.random.choice(range(self.buffer_size),
                                                self.train_batch_size, replace = False)

            '''training data'''
            # batch_states: [batch_size, value_input_obs_dim]
            # batch_device_obss: [batch_size, device_num, policy_input_dim]
            # batch_joint_acts: [batch_size, value_input_act_dim]
            # batch_joint_rewards: [batch_size, 1]
            # batch_next_states: [batch_size, value_input_obs_dim]
            # batch_next_device_obss: [batch_size, device_num, policy_input_dim]
            self.train_update_cnt += 1
            batch_states, batch_device_obss, \
            batch_joint_acts, batch_joint_rewards, \
            batch_next_states, batch_next_device_obss = replay_buffer.sample(batch_ids)

            for _ in range(self.critic_updates_round):
                self.train_value_net(batch_states, batch_joint_acts,
                                    batch_joint_rewards,
                                    batch_next_states, batch_next_device_obss)
            if self.train_update_cnt % self.policy_delay_round == 0:
                for i in range(self.device_num):
                    self.train_policy_net(i, batch_states, batch_device_obss[:, i],
                                        batch_joint_acts)

            if self.use_lr_decay:
                self.decay_lr(total_time_slots)

    def train_value_net(self, batch_states, batch_joint_acts,
                              batch_joint_rewards,
                              batch_next_states, batch_next_device_obss):
        batch_states = batch_states.to(self.device)
        batch_joint_acts = batch_joint_acts.to(self.device)
        batch_joint_rewards = batch_joint_rewards.to(self.device)
        batch_next_states = batch_next_states.to(self.device)
        batch_next_device_obss = batch_next_device_obss.to(self.device)
        with torch.no_grad():
            S = self.edge_server_num
            ae_dim = self.action_encode_dim
            batch_next_joint_acts = []
            for i in range(self.device_num):
                batch_next_acts = self.target_p_nets[i](batch_next_device_obss[:, i])
                # Compress: S logits + 3 averaged continuous per device
                server_logits = batch_next_acts[:, :S]
                cont_all = batch_next_acts[:, S:S + 3 * ae_dim]
                cont_blocks = cont_all.reshape(-1, 3, ae_dim)
                cont_compressed = cont_blocks.mean(dim=-1)
                compressed = torch.cat([server_logits, cont_compressed], dim=-1)  # [B, S+3]
                batch_next_joint_acts.append(compressed)
            # [batch_size, device_num * (S+3)]
            batch_next_joint_acts = torch.concat(batch_next_joint_acts, dim=-1)
            # [batch_size, 1]
            next_qs = self.target_v_net(batch_next_states, batch_next_joint_acts)
            target_qs = batch_joint_rewards + self.gamma * next_qs
            target_qs = target_qs.detach()

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
            
    def set_requires_grad(self, net, flag):
        for p in net.parameters():
            p.requires_grad_(flag)

    def train_policy_net(self, agent_id, batch_states, batch_device_obss, batch_joint_acts):
        batch_states = batch_states.to(self.device)
        batch_device_obss = batch_device_obss.to(self.device)
        batch_joint_acts = batch_joint_acts.to(self.device)
        S = self.edge_server_num
        ae_dim = self.action_encode_dim
        compressed_dim = S + 3  # per-device compressed action for Critic
        self.set_requires_grad(self.v_net, False)
        for i in range(self.p_epochs):
            batch_joint_acts_ = batch_joint_acts.clone()
            batch_acts = self.p_nets[agent_id](batch_device_obss)

            # Compress policy output from S+3*ae_dim to S+3 for Critic
            server_logits = batch_acts[:, :S]  # [B, S]
            cont_all = batch_acts[:, S:S + 3 * ae_dim]  # [B, 3*ae_dim]
            cont_blocks = cont_all.reshape(-1, 3, ae_dim)  # [B, 3, ae_dim]
            cont_compressed = cont_blocks.mean(dim=-1)  # [B, 3]
            compressed_act = torch.cat([server_logits, cont_compressed], dim=-1)  # [B, S+3]

            s = agent_id * compressed_dim
            e = (agent_id + 1) * compressed_dim
            batch_joint_acts_[:, s:e] = compressed_act

            p_loss = (-self.v_net(batch_states, batch_joint_acts_)).mean()

            self.p_optimizers[agent_id].zero_grad()
            p_loss.backward()

            # gradient clip
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.p_nets[agent_id].parameters(),
                                               self.p_grad_clip)
            self.p_optimizers[agent_id].step()
        self.set_requires_grad(self.v_net, True)

    @torch.no_grad()
    def soft_update(self, target_net, online_net, tau):
        # Polyak averaging: target = (1 - tau) * target + tau * online
        for t_param, o_param in zip(target_net.parameters(), online_net.parameters()):
            t_param.data.mul_(1.0 - tau)
            t_param.data.add_(tau * o_param.data)

    def update_target_nets(self, total_time_slots):
        if total_time_slots >= self.warm_time_slots:
            self.soft_update(self.target_v_net, self.v_net, self.tau)

            for i in range(self.device_num):
                self.soft_update(self.target_p_nets[i], self.p_nets[i], self.tau)

    def decay_lr(self, total_time_slots):
        if total_time_slots % self.decay_intl == 0:
            if self.v_lr > self.min_v_lr:
                self.v_lr -= self.v_decay_fac
                self.v_lr = max(self.v_lr, self.min_v_lr)
                for params in self.v_optimizer.param_groups:
                    params['lr'] = self.v_lr

            if self.p_lr > self.min_p_lr:
                self.p_lr -= self.p_decay_fac
                self.p_lr = max(self.p_lr, self.min_p_lr)
                for i in range(self.device_num):
                    for params in self.p_optimizers[i].param_groups:
                        params['lr'] = self.p_lr

    def save_nets(self, total_time_slots):
        if not os.path.exists(self.weights_dir):
            os.makedirs(self.weights_dir)
        torch.save(self.v_net.state_dict(),
                self.weights_dir + "v_net_params_" + str(total_time_slots) + ".pkl")
        torch.save(self.target_v_net.state_dict(),
                self.weights_dir + "target_v_net_params_" + str(total_time_slots) + ".pkl")

        for i in range(self.device_num):
            torch.save(self.p_nets[i].state_dict(),
                       self.weights_dir + "p_net_params_" +
                       str(i) + "_" + str(total_time_slots) + ".pkl")
            torch.save(self.target_p_nets[i].state_dict(),
                       self.weights_dir + "target_p_net_params_" +
                       str(i) + "_" + str(total_time_slots) + ".pkl")