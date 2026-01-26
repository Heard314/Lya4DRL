import copy
from operator import xor
from operator import xor
import pickle
import numpy as np
import torch
from env.mec_env import MECEnv
from agent.device_agent import MappoDeviceAgent, MaddpgDeviceAgent, \
LocalComputingDeviceAgent, EdgeComputingDeviceAgent, RandomComputingDeviceAgent
from agent.edge_agent import MappoEdgeAgent, MaddpgEdgeAgent
from util.replay_buffer import MappoReplayBuffer, MaddpgReplayBuffer
from util.utils import ObsScaling, RewardScaling, concatenate
from torch.utils.tensorboard import SummaryWriter
import sys, atexit, os
import config.global_params as gp
class Rollout:
    def __init__(self, gen_params, alg_params):
        self.device_num = gen_params.device_num
        self.task_arrival_prob = gen_params.task_arrival_prob
        self.evaluate = gen_params.evaluate
        self.train_mode = gen_params.train_mode
        self.eval_mode = gen_params.eval_mode
        self.resume_episode = 0
        self.device_type_num = gen_params.device_type_num
        # task generation cycle
        self.gen_task_cycle = gen_params.gen_task_cycle
        self.start_slot = gen_params.start_slot
        # resume
        self.load_weights = gen_params.load_weights
        # project storage path
        root_path = gp.settings.exp_result_dir
        run_dir = gp.settings.run_dir

        # edge agent and replay buffer
        if (not self.evaluate and self.train_mode == "mappo") or \
            (self.evaluate and self.eval_mode == "mappo"):
            print("The training mode is in rollout: mappo")
            self.edge_agent = MappoEdgeAgent(gen_params, alg_params)
            self.replay_buffer = MappoReplayBuffer(gen_params, alg_params)
        if (not self.evaluate and self.train_mode == "maddpg") or \
            (self.evaluate and self.eval_mode == "maddpg"):
            print("The training mode is in rollout: maddpg")
            self.edge_agent = MaddpgEdgeAgent(gen_params, alg_params)
            self.replay_buffer = MaddpgReplayBuffer(gen_params, alg_params)
        
        # obs scaling
        if not self.evaluate or (self.evaluate and self.eval_mode[0] == "m"):
            if alg_params.use_obs_scaling:
                self.obs_scaling = ObsScaling(gen_params, alg_params)

        # device agents
        self.device_agents = []
        for i in range(self.device_num):
            if (not self.evaluate and self.train_mode == "mappo") or \
               (self.evaluate and self.eval_mode == "mappo"):
                self.device_agents.append(MappoDeviceAgent(i, gen_params, alg_params))
            if (not self.evaluate and self.train_mode == "maddpg") or \
                (self.evaluate and self.eval_mode == "maddpg"):
                self.device_agents.append(MaddpgDeviceAgent(i, gen_params, alg_params))
            if self.evaluate and self.eval_mode == "local_comp":
                self.device_agents.append(LocalComputingDeviceAgent(i, gen_params))
            if self.evaluate and self.eval_mode == "edge_comp":
                self.device_agents.append(EdgeComputingDeviceAgent(i, gen_params))
            if self.evaluate and self.eval_mode == "random_comp":
                self.device_agents.append(RandomComputingDeviceAgent(i, gen_params))

        if not self.evaluate or (self.evaluate and self.eval_mode[0] == "m"):
            self.action_dim = alg_params.action_dim
            self.edge_queue_obs_dim = alg_params.edge_queue_obs_dim
            self.lstm_hidden_dim = alg_params.p_hid_dims[1]
        else:
            self.action_dim = -1
            self.edge_queue_obs_dim = -1
            self.lstm_hidden_dim = -1

        # training
        if not self.evaluate:
            # seed 
            self.seed = gp.settings.seed
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            self.train_mode = gen_params.train_mode
            self.train_time_slots = alg_params.train_time_slots
            self.train_freq = alg_params.train_freq
            if self.train_mode == "maddpg":
                self.target_update_freq = alg_params.target_update_freq
            self.save_freq = alg_params.save_freq
            
            # reward scaling
            if alg_params.use_reward_scaling:
                self.reward_scaling = RewardScaling(gen_params, alg_params, dim = self.device_type_num)
            
            # initialize agents' policy networks
            for i in range(self.device_num):
                self.device_agents[i].update_net(self.edge_agent.p_nets[i].state_dict())
        # evaluation
        else:
            # fix random seed
            self.seed = gp.settings.seed
            torch.manual_seed(gen_params.eval_seed)
            np.random.seed(gen_params.eval_seed)
            
            self.eval_time_slots = gen_params.eval_time_slots

            # reward scaling
            if alg_params.use_reward_scaling:
                self.reward_scaling = RewardScaling(gen_params, alg_params, dim = self.device_type_num)

            # initialize agents' policy networks
            if self.eval_mode[0] == "m":
                for i in range(self.device_num):
                    self.device_agents[i].update_net(self.edge_agent.p_nets[i].state_dict())
            #     for i in range(self.device_num):
            #         path = gen_params.weights_dir + "p_net_params_" + str(i) + ".pkl"
            #         self.device_agents[i].load_net(path)

        self.tb_log_dir = (
                root_path
                + "runs/"
                + run_dir
            )
        self.writer = SummaryWriter(log_dir=f"{self.tb_log_dir}/")
        print(f"The tensorboard file path is {self.tb_log_dir}")
        self.log_txt_dir_name = (
                root_path
                + "log/"
                + run_dir[:-1]
            )
        log_path = self.log_txt_dir_name + ".log"
        print(f"The log file path is {log_path}")
        
        # print hyperparameters info
        print(f"[DEBUG] device_act_queue_reward_weight: {gen_params.device_act_queue_reward_weight}")
        print(f"[DEBUG] device_vir_queue_reward_weight: {gen_params.device_vir_queue_reward_weight}")
        print(f"[DEBUG] device_act_queue_reward_max_bound: {gen_params.device_act_queue_reward_max_bound}")
        print(f"[DEBUG] device_act_queue_reward_min_bound: {gen_params.device_act_queue_reward_min_bound}")
        print(f"[DEBUG] device_vir_queue_reward_max_bound: {gen_params.device_vir_queue_reward_max_bound}")
        print(f"[DEBUG] device_vir_queue_reward_min_bound: {gen_params.device_vir_queue_reward_min_bound}")

        print(f"[DEBUG] timeout_reward_penalty: {gen_params.timeout_reward_penalty}")
        print(f"[DEBUG] target_reward_penalty: {gen_params.target_reward_penalty}")

        # device_env info print
        print(f"[DEBUG] device_act_queue_growth_rate: {gen_params.device_act_queue_growth_rate}")
        print(f"[DEBUG] device_vir_queue_growth_rate: {gen_params.device_vir_queue_growth_rate}")

        # edge_env info print
        print(f"[DEBUG] edge_act_queue_growth_rate: {gen_params.edge_act_queue_growth_rate}")
        print(f"[DEBUG] edge_vir_queue_growth_rate: {gen_params.edge_vir_queue_growth_rate}")

        # enable queue reward info
        print(f"[DEBUG] enable_actual_queue_reward: {gen_params.enable_actual_queue_reward}")
        print(f"[DEBUG] enable_virtual_queue_reward: {gen_params.enable_virtual_queue_reward}")

        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        log_txt_file = open(log_path, "a", encoding = "utf-8")
        sys.stdout = log_txt_file
        sys.stderr = log_txt_file
        atexit.register(log_txt_file.close)

        self.time_slots = self.train_time_slots + 1 if not self.evaluate else self.eval_time_slots

        # MEC env
        self.mec_env = MECEnv(gen_params, self.time_slots, self.writer)
        self.device_type_num = gen_params.device_type_num
        self.device_num_per_type = gen_params.device_num_per_type
        self.device_types = gen_params.device_types

        self.joint_rewards = None
        self.device_rewards = None
        self.joint_cost = None
        self.device_costs = None
        self.edge_comp_qls = None
        self.device_comp_qls = None
        self.comp_dlys = None
        self.edge_comp_dlys = None
        self.comp_dlys = None
        self.device_csum_engys = None
        self.device_esum_engys = None
        self.device_overtime_nums = None
        self.device_task_avail_nums = None
    
        # enable queue reward info
        print(f"[DEBUG] enable_actual_queue_reward: {gen_params.enable_actual_queue_reward}")
        print(f"[DEBUG] enable_virtual_queue_reward: {gen_params.enable_virtual_queue_reward}")


    def reset(self):
        if hasattr(self, "reward_scaling"):
            self.reward_scaling.reset()
        
        self.joint_rewards = np.zeros([self.device_type_num], dtype = np.float32)
        self.device_rewards = np.zeros([self.device_num], dtype = np.float32)
        self.joint_cost = 0
        self.device_costs = np.zeros([self.device_num], dtype = np.float32)
        self.edge_comp_qls = np.zeros([self.device_type_num], dtype = np.float32)
        self.device_comp_qls = np.zeros([self.device_num], dtype = np.float32)
        self.comp_dlys = np.zeros([self.device_num], dtype = np.float32)
        self.comp_dlys = np.zeros([self.device_num], dtype = np.float32)
        self.edge_comp_dlys = np.zeros([self.device_num], dtype = np.float32) #the edge computing delay for each task.
        self.device_csum_engys = np.zeros([self.device_num], dtype = np.float32)
        self.device_esum_engys = np.zeros([self.device_num], dtype = np.float32)
        self.device_overtime_nums = np.zeros([self.device_num], dtype = np.float32)
        self.device_task_avail_nums = np.zeros([self.device_num], dtype = np.float32)
        
    def run(self, e_id, visualize=False):

        writer = self.writer
        # reset
        self.reset()
        
        self.mec_env.reset()
        edge_obs = self.mec_env.edge_env.get_obs()
        device_obss = [None for i in range(self.device_num)]
        for i in range(self.device_num):
            device_obss[i] = self.mec_env.device_envs[i].get_obs()

        lstm_hidden_hs = [[0.0 for _ in range(self.lstm_hidden_dim)] for _ in range(self.device_num)]
        lstm_hidden_cs = [[0.0 for _ in range(self.lstm_hidden_dim)] for _ in range(self.device_num)]
        next_lstm_hidden_hs = [[0.0 for _ in range(self.lstm_hidden_dim)] for _ in range(self.device_num)]
        next_lstm_hidden_cs = [[0.0 for _ in range(self.lstm_hidden_dim)] for _ in range(self.device_num)]
        import numpy as np
        edge_comp_qls = [edge_obs[i * self.edge_queue_obs_dim] for i in range(self.device_type_num)]
        device_comp_qls = [obs[1] for obs in device_obss]
        # obs scaling
        if hasattr(self, "obs_scaling"):
            edge_obs, device_obss = self.obs_scaling(edge_obs, device_obss)
        # rollout
        time_slots = self.time_slots
        visualize_this_episode = visualize
        gen_task_cycle = self.gen_task_cycle

        start_t_id = self.start_slot
        end_t_id = time_slots + start_t_id + 1

        gen_t_id = -1 

        # update current episode for std annealing
        if self.train_mode == "mappo":
            for i in range(self.device_num):
                self.device_agents[i].p_net.set_episode(e_id + gp.settings.resume_episode)
                self.edge_agent.p_nets[i].set_episode(e_id + gp.settings.resume_episode)

        for t_id in range(start_t_id, end_t_id):
            if t_id % gen_task_cycle == start_t_id:
                gen_t_id += 1
                visualize = visualize_this_episode
            else:
                visualize = False

            # print("-------------time slot: " + str(t_id) + "-------------")
            # choose action (use deterministic strategy during evaluation)
            device_acts = [None for i in range(self.device_num)]
            device_active = [None for i in range(self.device_num)]
            if "Mappo" in type(self.device_agents[0]).__name__:
                # store actions used for interacting with the MEC env 
                device_acts_ = [[] for i in range(self.device_num)]
                if not (self.evaluate or gp.settings.is_evaluate):
                    device_act_logprobs = [None for i in range(self.device_num)]
                for i in range(self.device_num):
                    task_num = self.mec_env.device_envs[i].task_num
                    device_active[i] = task_num >= 1
                    device_type = self.device_types[i]
                    # print(f"[DEBUG] device {i}, t_id {t_id}")
                    # print(f"[DEBUG] task_num >= 1: {task_num >= 1}, t_id % gen_task_cycle == start_slot: {t_id % gen_task_cycle == start_t_id}")
                    assert ((task_num >= 1) == (t_id % gen_task_cycle == start_t_id))
                    if task_num >= 1:
                        device_value_obs = concatenate(device_obss[i], edge_obs[device_type * self.edge_queue_obs_dim : (device_type + 1) * self.edge_queue_obs_dim]) 
                        act, act_logprob, next_lstm_hidden_hs[i], next_lstm_hidden_cs[i] = self.device_agents[i].choose_action(device_value_obs, lstm_hidden_hs[i], lstm_hidden_cs[i], active=device_active[i])
                        device_acts[i] = act
                        for j in range(self.action_dim):
                            device_acts_[i].append(act[j] / 10)
                    else:
                        device_acts[i] = [-1.0 for _ in range(self.action_dim)]
                        device_acts_[i] = [-1.0 for _ in range(self.action_dim)]
                    if not (act_logprob == None):
                        device_act_logprobs[i] = act_logprob
            if "Maddpg" in type(self.device_agents[0]).__name__:
                # store actions used for interacting with the MEC env
                device_acts_ = [[] for i in range(self.device_num)]
                for i in range(self.device_num):
                    task_num = self.mec_env.device_envs[i].task_num
                    device_type = self.device_types[i]
                    assert ((task_num >= 1) == (t_id % gen_task_cycle == start_t_id))
                    if task_num >= 1:
                        device_value_obs = concatenate(device_obss[i], edge_obs[device_type * self.edge_queue_obs_dim : (device_type + 1) * self.edge_queue_obs_dim])
                        act, next_lstm_hidden_hs[i], next_lstm_hidden_cs[i] = self.device_agents[i].choose_action(device_value_obs, lstm_hidden_hs[i], lstm_hidden_cs[i])
                        device_acts[i] = act
                        for j in range(self.action_dim // 10):
                            device_acts_[i].append((act[j * 10] + act[j * 10 + 1] + act[j * 10 + 2] + 
                                                    act[j * 10 + 3] + act[j * 10 + 4] + act[j * 10 + 5] +
                                                    act[j * 10 + 6] + act[j * 10 + 7] + act[j * 10 + 8] +
                                                    act[j * 10 + 9]) / 20)
                    else:
                        device_acts[i] = [-1.0 for _ in range(self.action_dim)]
                        device_acts_[i] = [-1.0 for _ in range(self.action_dim // 10)]
            if "Computing" in type(self.device_agents[0]).__name__:
                for i in range(self.device_num):
                    task_num = self.mec_env.device_envs[i].task_num
                    device_type = self.device_types[i]
                    assert ((task_num >= 1) == (t_id % gen_task_cycle == start_t_id))
                    if task_num >= 1:
                        act = self.device_agents[i].choose_action()
                        device_acts[i] = act
                    else:
                        device_acts[i] = [-1.0 for _ in range(self.action_dim)]
                device_acts_ = device_acts


            # step
            joint_rewards, device_rewards, \
            joint_cost, device_costs, \
            comp_dlys, device_csum_engys, \
            device_esum_engys, device_overtime_nums, \
            next_edge_obs, next_device_obss, device_task_is_available = self.mec_env.step(device_acts_, e_id = e_id, t_id = t_id, visualize = visualize)
            
            # update computing-queue lengths
            edge_comp_qls = [next_edge_obs[i * self.edge_queue_obs_dim] for i in range(self.device_type_num)]
            device_comp_qls = [obs[1] for obs in next_device_obss]

            if t_id % gen_task_cycle == start_t_id:
                self.average(gen_t_id, joint_rewards, device_rewards,
                                joint_cost, device_costs,
                                comp_dlys, device_csum_engys,
                                device_esum_engys, device_overtime_nums,
                                device_task_is_available, edge_comp_qls, device_comp_qls)
            # self.average_always(t_id, edge_comp_qls, device_comp_qls)
            
            # reward scaling
            if hasattr(self, "reward_scaling"):
                joint_rewards = self.reward_scaling(joint_rewards)
            # print(f"[DEBUG] joint_rewards: {joint_rewards}")
            # obs scaling
            if hasattr(self, "obs_scaling"):
                next_edge_obs, next_device_obss = self.obs_scaling(next_edge_obs, next_device_obss)

            if t_id % gen_task_cycle == start_t_id:
                if not (self.evaluate or gp.settings.is_evaluate) and self.train_mode == "mappo":
                    # print(f"[DEBUG] edge_obs: {edge_obs}")
                    # print(f"[DEBUG] device_obss: {device_obss}")
                    # print(f"[DEBUG] lstm_hidden_hs: {lstm_hidden_hs}")
                    # print(f"[DEBUG] lstm_hidden_cs: {lstm_hidden_cs}")
                    # print(f"[DEBUG] device_acts: {device_acts}")
                    # print(f"[DEBUG] device_act_logprobs: {device_act_logprobs}")
                    # print(f"[DEBUG] joint_rewards: {joint_rewards}")
                    self.replay_buffer.store(edge_obs, device_obss, lstm_hidden_hs, lstm_hidden_cs,
                                            device_acts, device_act_logprobs,
                                            joint_rewards, device_active)
                if not (self.evaluate or gp.settings.is_evaluate) and self.train_mode == "maddpg":
                    # print(f"[DEBUG] edge_obs: {edge_obs}")
                    # print(f"[DEBUG] device_obss: {device_obss}")
                    # print(f"[DEBUG] device_acts: {device_acts}")
                    # print(f"[DEBUG] joint_rewards: {joint_rewards}")
                    # print(f"[DEBUG] next_edge_obs: {next_edge_obs}")
                    # print(f"[DEBUG] next_device_obss: {next_device_obss}")
                    self.replay_buffer.store(edge_obs, device_obss, 
                                            device_acts, joint_rewards,
                                            next_edge_obs, next_device_obss)
            
            # update obs
            edge_obs = next_edge_obs
            device_obss = next_device_obss
            if t_id % gen_task_cycle == start_t_id:
                lstm_hidden_hs = next_lstm_hidden_hs
                lstm_hidden_cs = next_lstm_hidden_cs

            if not (self.evaluate or gp.settings.is_evaluate) and self.train_mode == "maddpg":
                total_time_slots = e_id * self.train_time_slots + t_id + 1
                
                # train networks
                if total_time_slots % self.train_freq == 0:
                    for i in range(self.device_num):
                        self.device_agents[i].increment_update_cnt()
                    self.edge_agent.train_nets(total_time_slots, self.replay_buffer)
                    # update agents' policy networks
                    for i in range(self.device_num):
                        self.device_agents[i].update_net(self.edge_agent.p_nets[i].state_dict())

                # update target networks
                if total_time_slots % self.target_update_freq == 0:
                    self.edge_agent.update_target_nets(total_time_slots)
                
                # save networks
                if total_time_slots % self.save_freq == 0:
                    self.edge_agent.save_nets(total_time_slots)
                    
        if not (self.evaluate or gp.settings.is_evaluate) and self.train_mode == "mappo":
            # train networks
            if (e_id + 1) % self.train_freq == 0:
                self.edge_agent.train_nets(self.replay_buffer)
                # update agents' policy networks
                for i in range(self.device_num):
                    self.device_agents[i].update_net(self.edge_agent.p_nets[i].state_dict())
                    
            if (e_id + 1) % self.save_freq == 0:
                self.edge_agent.save_nets(e_id, self.seed)

        joint_rewards = copy.copy(self.joint_rewards)
        device_rewards = copy.copy(self.device_rewards)
        joint_cost = copy.copy(self.joint_cost)
        device_costs = copy.copy(self.device_costs)
        edge_comp_qls =  copy.copy(self.edge_comp_qls)
        device_comp_qls = copy.copy(self.device_comp_qls)
        comp_dlys = copy.copy(self.comp_dlys)
        device_csum_engys = copy.copy(self.device_csum_engys)
        device_esum_engys = copy.copy(self.device_esum_engys)
        device_overtime_nums = copy.copy(self.device_overtime_nums)

        # Periodically visualize the operation status of all devices in this round using plot
        from util.utils import save_device_hist_plots
        if visualize:
            save_device_hist_plots(
                device_overtime_nums=device_overtime_nums,
                device_comp_dlys=comp_dlys,
                e_id=e_id,
                out_dir=gp.settings.plot_dir
            )

        # tensorboard log
        for i in range(self.device_type_num):
            writer.add_scalar(f"joint_reward_{i}{'_eval' if gp.settings.is_evaluate else ''}", joint_rewards[i], e_id)
            print(f"joint_reward_{i}: {joint_rewards[i]}")
        writer.add_scalar(f"joint_cost{'_eval' if gp.settings.is_evaluate else ''}", joint_cost, e_id)
        print(f"joint_cost: {joint_cost}")
        for i in range(self.device_type_num):
            writer.add_scalar(f"edge_comp_ql_{i}{'_eval' if gp.settings.is_evaluate else ''}", edge_comp_qls[i], e_id)
            print(f"edge_comp_ql_{i}: {edge_comp_qls[i]}")
        for i in range(self.device_num):
            writer.add_scalar(f"device_reward_{i}{'_eval' if gp.settings.is_evaluate else ''}", device_rewards[i], e_id)
            print(f"device_reward_{i}: {device_rewards[i]}")
            writer.add_scalar(f"device_cost_{i}{'_eval' if gp.settings.is_evaluate else ''}", device_costs[i], e_id)
            print(f"device_cost_{i}: {device_costs[i]}")
            writer.add_scalar(f"device_comp_ql_{i}{'_eval' if gp.settings.is_evaluate else ''}", device_comp_qls[i], e_id)
            print(f"device_comp_ql_{i}: {device_comp_qls[i]}")
            writer.add_scalar(f"device_virtual_time_ql_{i}{'_eval' if gp.settings.is_evaluate else ''}", self.mec_env.device_envs[i].virtual_time_ql, e_id)
            print(f"device_virtual_time_ql_{i}: {self.mec_env.device_envs[i].virtual_time_ql}")
            writer.add_scalar(f"comp_dlys_{i}{'_eval' if gp.settings.is_evaluate else ''}", comp_dlys[i], e_id)
            print(f"comp_dlys_{i}: {comp_dlys[i]}")
            writer.add_scalar(f"device_csum_engys_{i}{'_eval' if gp.settings.is_evaluate else ''}", device_csum_engys[i], e_id)
            print(f"device_csum_engys_{i}: {device_csum_engys[i]}")
            # writer.add_scalar("device_comp_expns_"+str(i), device_esum_engys[i], e_id)
            # print(f"device_comp_expns_{i}: {device_esum_engys[i]}")
            writer.add_scalar(f"device_overtime_nums_{i}{'_eval' if gp.settings.is_evaluate else ''}", device_overtime_nums[i], e_id)
            print(f"device_overtime_nums_{i}: {device_overtime_nums[i]}")
        
        if e_id % 50 == 0:
            self.writer.flush()

        return joint_rewards, device_rewards, \
               joint_cost, device_costs, \
               edge_comp_qls, device_comp_qls, \
               comp_dlys, device_csum_engys, \
               device_esum_engys, device_overtime_nums
    
    # Update only when new tasks arrive in the time slot
    def average(self, gen_t_id, joint_rewards, device_rewards, 
                            joint_cost, device_costs, 
                            comp_dlys, device_csum_engys, 
                            device_esum_engys, device_overtime_nums,
                            device_task_is_available, edge_comp_qls, device_comp_qls):
        gen_t_id_ = gen_t_id + 1
        self.joint_rewards += 1 / gen_t_id_ * (joint_rewards - self.joint_rewards)
        self.device_rewards += 1 / gen_t_id_ * (device_rewards - self.device_rewards)
        self.joint_cost += 1 / gen_t_id_ * (joint_cost - self.joint_cost)
        self.device_costs += 1 / gen_t_id_ * (device_costs - self.device_costs)
        self.comp_dlys += 1 / gen_t_id_ * (comp_dlys - self.comp_dlys)
        self.device_csum_engys += (device_csum_engys)
        self.device_esum_engys += (device_esum_engys)
        self.device_overtime_nums += device_overtime_nums

        self.edge_comp_qls += 1 / gen_t_id_ * (edge_comp_qls - self.edge_comp_qls)
        self.device_comp_qls += 1 / gen_t_id_ * (device_comp_qls - self.device_comp_qls)

        for i in range(self.device_num):
            if device_task_is_available[i]:
                self.device_task_avail_nums[i]+=1

        # for i in range(self.device_num):
        #     self.device_task_avail_nums[i] = max(1.0, self.device_task_avail_nums[i])
        #     self.comp_dlys[i] /= self.device_task_avail_nums[i]
        #     self.device_csum_engys[i] /= self.device_task_avail_nums[i]
        #     self.device_esum_engys[i] /= self.device_task_avail_nums[i]
    
    # # Update queue information at every time slot
    # def average_always(self, t_id, edge_comp_qls, device_comp_qls):
    #     t_id_ = t_id + 1
    #     self.edge_comp_qls += 1 / t_id_ * (edge_comp_qls - self.edge_comp_qls)
    #     self.device_comp_qls += 1 / t_id_ * (device_comp_qls - self.device_comp_qls)