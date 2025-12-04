import copy
import numpy as np
import torch
from env.mec_env import MECEnv
from agent.device_agent import MappoDeviceAgent, MaddpgDeviceAgent, \
LocalComputingDeviceAgent, EdgeComputingDeviceAgent, RandomComputingDeviceAgent
from agent.edge_agent import MappoEdgeAgent, MaddpgEdgeAgent
from util.replay_buffer import MappoReplayBuffer, MaddpgReplayBuffer
from util.utils import ObsScaling, RewardScaling
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
        
        # edge agent and replay buffer
        if not self.evaluate and self.train_mode == "mappo":
            print("The training mode is in rollout: mappo")
            self.edge_agent = MappoEdgeAgent(gen_params, alg_params)
            self.replay_buffer = MappoReplayBuffer(gen_params, alg_params)
        if not self.evaluate and self.train_mode == "maddpg":
            print("The training mode is in rollout: maddpg")
            self.edge_agent = MaddpgEdgeAgent(gen_params, alg_params)
            self.replay_buffer = MaddpgReplayBuffer(gen_params, alg_params)
        
        # obs scaling
        if not self.evaluate or (self.evaluate and self.eval_mode[0] == "m"):
            if alg_params.use_obs_scaling:
                self.obs_scaling = ObsScaling(gen_params.max_task_num,
                                              gen_params.max_data_size,
                                              gen_params.max_comp_dens,
                                              gen_params.std_comp_freq)
        
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

        self.action_dim = alg_params.action_dim
        self.lstm_hidden_dim = alg_params.p_hid_dims[1]

        # training
        if not self.evaluate:
            # fix random seed
            self.seed = alg_params.train_seed
            torch.manual_seed(alg_params.train_seed)
            np.random.seed(alg_params.train_seed)
            
            self.train_mode = gen_params.train_mode
            self.train_time_slots = alg_params.train_time_slots
            self.train_freq = alg_params.train_freq
            if self.train_mode == "maddpg":
                self.target_update_freq = alg_params.target_update_freq
            self.save_freq = alg_params.save_freq
            
            # reward scaling
            if alg_params.use_reward_scaling:
                self.reward_scaling = RewardScaling(alg_params.gamma)
            
            # initialize agents' policy networks
            for i in range(self.device_num):
                self.device_agents[i].update_net(self.edge_agent.p_nets[i].state_dict())
        # evaluation
        else:
            # fix random seed
            self.seed = gen_params.eval_seed
            torch.manual_seed(gen_params.eval_seed)
            np.random.seed(gen_params.eval_seed)
            
            self.eval_time_slots = gen_params.eval_time_slots
            
            # initialize agents' policy networks
            if self.eval_mode[0] == "m":
                for i in range(self.device_num):
                    path = alg_params.weights_dir + "p_net_params_" + str(i) + ".pkl"
                    self.device_agents[i].load_net(path)
        root_path = gp.settings.project_root
        import datetime
        file_subpath =  (
                (self.evaluate and "evaluate" or "train")
                + "/"
                + (self.evaluate and self.eval_mode or self.train_mode)
                + "_s_"
                + str(self.seed)
                + "_t_"
                + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
                + "_d_"
                + gen_params.run_desc
        )

        self.log_dir_name = (
                root_path
                + "runs/"
                + file_subpath
            )
        self.writer = SummaryWriter(log_dir=f"{self.log_dir_name}/")
        self.log_txt_dir_name = (
                root_path
                + "log/"
                + file_subpath
            )
        log_path = self.log_txt_dir_name + ".log"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        log_txt_file = open(log_path, "w", encoding = "utf-8")
        sys.stdout = log_txt_file
        sys.stderr = log_txt_file
        atexit.register(log_txt_file.close)

        self.time_slots = self.train_time_slots + 1 if not self.evaluate else self.eval_time_slots

        # MEC env
        self.mec_env = MECEnv(gen_params, self.time_slots, self.writer)
        self.device_type_num = gen_params.device_type_num


        self.joint_reward = None
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
        
    def reset(self):
        if hasattr(self, "reward_scaling"):
            self.reward_scaling.reset()
        
        self.joint_reward = 0
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
        
        edge_obs, device_obss = self.mec_env.reset()
        lstm_hidden_hs = [[0.0 for _ in range(self.lstm_hidden_dim)] for _ in range(self.device_num)]
        lstm_hidden_cs = [[0.0 for _ in range(self.lstm_hidden_dim)] for _ in range(self.device_num)]
        next_lstm_hidden_hs = [[0.0 for _ in range(self.lstm_hidden_dim)] for _ in range(self.device_num)]
        next_lstm_hidden_cs = [[0.0 for _ in range(self.lstm_hidden_dim)] for _ in range(self.device_num)]
        edge_comp_qls = [edge_obs[i] for i in range(self.device_type_num,2*self.device_type_num)]

        device_comp_qls = [obs[7] for obs in device_obss]
        # obs scaling
        if hasattr(self, "obs_scaling"):
            self.obs_scaling(edge_obs, device_obss)
        
        # rollout
        time_slots = self.time_slots
        for t_id in range(1, time_slots + 1):
            print("-------------time slot: " + str(t_id) + "-------------")
            
            # choose action (use deterministic strategy during evaluation)
            device_acts = [None for i in range(self.device_num)]
            device_active = [None for i in range(self.device_num)]
            if "Mappo" in type(self.device_agents[0]).__name__:
                # store actions used for interacting with the MEC env 
                device_acts_ = [[] for i in range(self.device_num)]
                if not self.evaluate:
                    device_act_logprobs = [None for i in range(self.device_num)]
                for i in range(self.device_num):
                    task_num = self.mec_env.device_envs[i].task_num
                    device_active[i] = task_num >= 1    
                    act, act_logprob, next_lstm_hidden_hs[i], next_lstm_hidden_cs[i] = self.device_agents[i].choose_action(device_obss[i], lstm_hidden_hs[i], lstm_hidden_cs[i], active=device_active[i])
                    device_acts[i] = act
                    # 当不需要处理其他任务时设为0
                    if task_num >= 1:
                        for j in range(self.action_dim):
                            device_acts_[i].append(act[j] / 10)
                    if not (act_logprob == None):
                        device_act_logprobs[i] = act_logprob
            # if "Maddpg" in type(self.device_agents[0]).__name__:
            #     # store actions used for interacting with the MEC env
            #     device_acts_ = [[] for i in range(self.device_num)]
            #     for i in range(self.device_num):
            #         act = self.device_agents[i].choose_action(device_obss[i])
            #         device_acts[i] = act
            #         for j in range(self.task_num):
            #             device_acts_[i].append((act[j * 10] + act[j * 10 + 1] + act[j * 10 + 2] + 
            #                                     act[j * 10 + 3] + act[j * 10 + 4] + act[j * 10 + 5] +
            #                                     act[j * 10 + 6] + act[j * 10 + 7] + act[j * 10 + 8] +
            #                                     act[j * 10 + 9]) / 20)
            # if "Computing" in type(self.device_agents[0]).__name__:
            #     for i in range(self.device_num):
            #         act = self.device_agents[i].choose_action()
            #         device_acts[i] = act
            #     device_acts_ = device_acts
            
            # step
            # device_rewards为能耗奖励
            # act_queue_rewards为实际队列奖励
            # vir_queue_rewards为虚拟队列奖励
            # total_rewards为综合奖励
            # device_engys和edge_engys分别为任务在设备端和边缘端的能耗
            joint_reward, device_rewards, \
            joint_cost, device_costs, \
            comp_dlys, device_csum_engys, \
            device_esum_engys, device_overtime_nums, \
            next_edge_obs, next_device_obss, device_task_is_available = self.mec_env.step(device_acts_, e_id = e_id, t_id = t_id, visualize = visualize)
            
            self.average(t_id, joint_reward, device_rewards,
                               joint_cost, device_costs,
                               edge_comp_qls, device_comp_qls,
                               comp_dlys, device_csum_engys,
                               device_esum_engys, device_overtime_nums,
                               device_task_is_available)
            
            if hasattr(self, "reward_scaling"):
                joint_reward = self.reward_scaling(joint_reward)

            device_type_num = self.device_type_num
            # update computing-queue lengths
            edge_comp_qls = [next_edge_obs[i] for i in range(device_type_num,2*device_type_num)]
            device_comp_qls = [obs[7] for obs in next_device_obss]
            # obs scaling
            if hasattr(self, "obs_scaling"):
                self.obs_scaling(next_edge_obs, next_device_obss)
            
            if not self.evaluate and self.train_mode == "mappo":
                self.replay_buffer.store(edge_obs, device_obss,lstm_hidden_hs,lstm_hidden_cs,
                                         device_acts, device_act_logprobs,
                                         joint_reward, device_active)
            if not self.evaluate and self.train_mode == "maddpg":
                self.replay_buffer.store(edge_obs, device_obss, 
                                         device_acts, joint_reward,
                                         next_edge_obs, next_device_obss)
            
            # update obs
            edge_obs = next_edge_obs
            device_obss = next_device_obss
            lstm_hidden_hs = next_lstm_hidden_hs
            lstm_hidden_cs = next_lstm_hidden_cs

            if not self.evaluate and self.train_mode == "maddpg":
                total_time_slots = (e_id - 1) * self.train_time_slots + t_id
                
                # train networks
                if total_time_slots % self.train_freq == 0:
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
                    
        if not self.evaluate and self.train_mode == "mappo":
            # train networks
            if e_id % self.train_freq == 0:
                self.edge_agent.train_nets(self.replay_buffer)
                # update agents' policy networks
                for i in range(self.device_num):
                    self.device_agents[i].update_net(self.edge_agent.p_nets[i].state_dict())
                    
            if e_id % self.save_freq == 0:
                self.edge_agent.save_nets(e_id)
        
        self.average_available()

        joint_reward = copy.copy(self.joint_reward)
        device_rewards = copy.copy(self.device_rewards)
        joint_cost = copy.copy(self.joint_cost)
        device_costs = copy.copy(self.device_costs)
        edge_comp_qls =  copy.copy(self.edge_comp_qls)
        device_comp_qls = copy.copy(self.device_comp_qls)
        comp_dlys = copy.copy(self.comp_dlys)
        # device_comp_dlys = copy.copy(self.device_comp_dlys)
        # edge_comp_dlys = copy.copy(self.edge_comp_dlys)
        device_csum_engys = copy.copy(self.device_csum_engys)
        device_esum_engys = copy.copy(self.device_esum_engys)
        device_overtime_nums = copy.copy(self.device_overtime_nums)

        # tensorboard日志保存
        # if visualize:
        #     self.visualize_for_one_episode()
        writer.add_scalar("joint_reward", joint_reward, e_id)
        print(f"joint_reward: {joint_reward}")
        writer.add_scalar("joint_cost", joint_cost, e_id)
        print(f"joint_cost: {joint_cost}")
        for i in range(self.device_type_num):
            writer.add_scalar("edge_comp_ql_"+str(i), edge_comp_qls[i], e_id)
            print(f"edge_comp_ql_{i}: {edge_comp_qls[i]}")
        for i in range(self.device_num):
            
            writer.add_scalar("device_reward_"+str(i), device_rewards[i], e_id)
            print(f"device_reward_{i}: {device_rewards[i]}")
            writer.add_scalar("device_cost_"+str(i), device_costs[i], e_id)
            print(f"device_cost_{i}: {device_costs[i]}")
            writer.add_scalar("device_comp_ql_"+str(i), device_comp_qls[i], e_id)
            print(f"device_comp_ql_{i}: {device_comp_qls[i]}")
            writer.add_scalar("device_virtual_time_ql_"+str(i), self.mec_env.device_envs[i].virtual_time_ql, e_id)
            print(f"device_virtual_time_ql_{i}: {self.mec_env.device_envs[i].virtual_time_ql}")
            writer.add_scalar("comp_dlys_"+str(i), comp_dlys[i], e_id)
            print(f"comp_dlys_{i}: {comp_dlys[i]}")
            writer.add_scalar("device_csum_engys_"+str(i), device_csum_engys[i], e_id)
            print(f"device_csum_engys_{i}: {device_csum_engys[i]}")
            writer.add_scalar("device_comp_expns_"+str(i), device_esum_engys[i], e_id)
            print(f"device_comp_expns_{i}: {device_esum_engys[i]}")
            writer.add_scalar("device_overtime_nums_"+str(i), device_overtime_nums[i], e_id)
            print(f"device_overtime_nums_{i}: {device_overtime_nums[i]}")

            # writer.add_scalars(
            #     "comp_dlys_" + str(i),
            #     {
            #         "total":  comp_dlys[i],
            #         "device": device_comp_dlys[i],
            #         "edge":   edge_comp_dlys[i],
            #     },
            #     e_id
            # )

            # writer.add_scalars(
            #     "device_time_ql_" + str(i),
            #     {
            #         "act":  self.mec_env.device_envs[i].time_ql,
            #         "vir": self.mec_env.device_envs[i].virtual_time_ql,
            #         "act_chg":   edge_comp_dlys[i],
            #         "vir_chg":   edge_comp_dlys[i],
            #         "act_reward": device_rewards[i],
            #         "vir_reward": device_rewards[i],
            #     },
            #     e_id
            # )

            # writer.add_scalars(
            #     "edge_time_ql_" + str(i),
            #     {
            #         "act":  comp_dlys[i],
            #         "vir": device_comp_dlys[i],
            #         "act_chg":   edge_comp_dlys[i],
            #         "vir_chg":   edge_comp_dlys[i],
            #         "act_reward": device_rewards[i],
            #         "vir_reward": device_rewards[i],
            #     },
            #     e_id
            # )

            # writer.add_scalars(
            #     "reward_" + str(i),
            #     {
            #         "navie":
            #         "act_queue": device_rewards[i],
            #         "vir_queue": device_rewards[i],
            #         "total": device_rewards[i]
            #     },
            #     e_id
            # )
        
        if e_id % 50 == 0:
            self.writer.flush()

        return joint_reward, device_rewards, \
               joint_cost, device_costs, \
               edge_comp_qls, device_comp_qls, \
               comp_dlys, device_csum_engys, \
               device_esum_engys, device_overtime_nums
    
    def average(self, t_id, joint_reward, device_rewards, 
                            joint_cost, device_costs, 
                            edge_comp_qls, device_comp_qls,
                            comp_dlys, device_csum_engys, 
                            device_esum_engys, device_overtime_nums,
                            device_task_is_available):
        self.joint_reward += 1 / t_id * (joint_reward - self.joint_reward)
        self.device_rewards += 1 / t_id * (device_rewards - self.device_rewards)
        self.joint_cost += 1 / t_id * (joint_cost - self.joint_cost)
        self.device_costs += 1 / t_id * (device_costs - self.device_costs)
        self.edge_comp_qls += 1 / t_id * (edge_comp_qls - self.edge_comp_qls)
        self.device_comp_qls += 1 / t_id * (device_comp_qls - self.device_comp_qls)
        # self.device_comp_dlys += 1 / t_id * (device_comp_dlys - self.device_comp_dlys)
        # self.device_csum_engys += 1 / t_id * (device_csum_engys - self.device_csum_engys)
        # self.device_esum_engys += 1 / t_id * (device_esum_engys - self.device_esum_engys)
        self.comp_dlys += (comp_dlys)
        self.device_csum_engys += (device_csum_engys)
        self.device_esum_engys += (device_esum_engys)
        self.device_overtime_nums += device_overtime_nums
        for i in range(self.device_num):
            if device_task_is_available[i]:
                self.device_task_avail_nums[i]+=1
    
    # 任务延迟、能耗、费用都是按照可用任务数量来平均的
    def average_available(self):
        for i in range(self.device_num):
            self.device_task_avail_nums[i] = max(1.0, self.device_task_avail_nums[i])
            self.comp_dlys[i] /= self.device_task_avail_nums[i]
            self.device_csum_engys[i] /= self.device_task_avail_nums[i]
            self.device_esum_engys[i] /= self.device_task_avail_nums[i]