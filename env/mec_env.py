from env import device_env
from env.device_env import DeviceEnv
from env.edge_env import EdgeEnv
import torch
import math
import config.global_params as gp
class MECEnv():
    def __init__(self, gen_params,time_slots, writer = None):
        # Summary Writer
        self.writer = writer
        self.device_num = gen_params.device_num
        self.edge_energy_weights = gen_params.edge_energy_weights
        self.device_energy_weights = gen_params.device_energy_weights
        self.time_slots = time_slots
        self.device_type_num = gen_params.device_type_num
        self.device_num_per_type = gen_params.device_num_per_type
        # edge env
        self.edge_env = EdgeEnv(gen_params, writer)
        # device envs
        self.device_envs = []
        for i in range(self.device_num):
            self.device_envs.append(DeviceEnv(i, gen_params, self.edge_env, writer))
        # self.lyaV = gen_params.lyaV
        self.local_reward_weight = gen_params.local_reward_weight
        self.edge_reward_weight = gen_params.edge_reward_weight

        # self.device_local_reward_bound = gen_params.local_reward_bound
        # self.edge_reward_bound = gen_params.edge_reward_bound

        self.enable_actual_queue_reward = gen_params.enable_actual_queue_reward
        self.enable_virtual_queue_reward = gen_params.enable_virtual_queue_reward
    
    def reset(self):
        edge_obs = self.edge_env.reset()
        
        device_obss = [None for i in range(self.device_num)]
        for i in range(self.device_num):
            device_obss[i] = self.device_envs[i].reset()
        
        return edge_obs, device_obss
    
    def step(self, device_acts, e_id, t_id, visualize=False):

        writer = self.writer
        if e_id % 50 == 1:
            gp.settings.enable_print = True
        else:
            gp.settings.enable_print = False
        enable_print = gp.settings.enable_print
        # 首先每个设备对待执行任务做出卸载决策，然后执行任务的本地计算部分，返回远程卸载部分（以下代码中的sched_tasks）
        device_sched_tasks = [None for i in range(self.device_num)]
        for i in range(self.device_num):
            sched_tasks = self.device_envs[i].compute(device_acts[i], e_id = e_id, t_id = t_id, visualize = visualize)
            device_sched_tasks[i] = sched_tasks
        # 边缘服务器执行任务的远程卸载部分
        self.edge_env.compute(device_sched_tasks, e_id = e_id, t_id = t_id, visualize = visualize)
        
        # reward
        device_rewards = [0 for i in range(self.device_num)]
        edge_queue_num = self.device_type_num
        device_queue_actual_rewards = [0 for i in range(self.device_num)]
        device_queue_virtual_rewards = [0 for i in range(self.device_num)]
        edge_queue_actual_rewards = [0 for i in range(edge_queue_num)]
        edge_queue_virtual_rewards = [0 for i in range(edge_queue_num)]
        device_costs = [0 for i in range(self.device_num)]


        device_comp_dlys = [0 for i in range(self.device_num)]

        device_csum_engys = [0 for i in range(self.device_num)]
        device_esum_engys = [0 for i in range(self.device_num)]
        device_overtime_nums = [0 for i in range(self.device_num)]
        # device_delay_adjust_coefs = [1.0 for i in range(self.device_num)]
        device_task_is_available = [False for i in range(self.device_num)] #在该时间间隙下是否有任务到达
        for i in range(self.device_num):
            sched_tasks = device_sched_tasks[i]
            task_num = len(sched_tasks)
            # if(enable_print): print(f"[DEBUG] In device {i}, the task number is {task_num}")
            device_type = self.device_envs[i].device_type
            device_task_is_available[i] = task_num>=1
            for j in range(task_num):
                task = sched_tasks[j]
                comp_dly = max(task.l_comp_dly, task.e_comp_dly)
                if visualize:
                    writer.add_scalars(
                        f"detail/comp_dly_{i}",
                        {f"ep_{e_id}": comp_dly},
                        t_id
                    )
                    writer.add_scalars(
                        f"detail/l_comp_dly_{i}",
                        {f"ep_{e_id}": task.l_comp_dly},
                        t_id
                    )
                    writer.add_scalars(
                        f"detail/e_comp_dly_{i}",
                        {f"ep_{e_id}": task.e_comp_dly},
                        t_id
                    )
                if(enable_print): print(f"[DEBUG] the comp_dly in device {i} is {comp_dly}")
                device_comp_dlys[i] += 1 / (j + 1) * (comp_dly - device_comp_dlys[i])
                
                csum_engy = task.local_comp_engy + task.tran_engy
                if(enable_print): print(f"[DEBUG] the local csum_engy in device {i} is {task.local_comp_engy}")
                if(enable_print): print(f"[DEBUG] the edge csum_engy in device {i} is {task.tran_engy}")
                if visualize:
                    writer.add_scalars(
                        f"detail/engy_{i}",
                        {f"ep_{e_id}_local": task.local_comp_engy + task.tran_engy},
                        t_id
                    )
                    writer.add_scalars(
                        f"detail/engy_{i}",
                        {f"ep_{e_id}_edge": task.edge_comp_engy},
                        t_id
                    )


                device_csum_engys[i] += 1 / (j + 1) * (csum_engy - device_csum_engys[i])
                edge_comp_engy = task.edge_comp_engy
                device_esum_engys[i] += 1 / (j + 1) * (edge_comp_engy - device_esum_engys[i])
                
                device_costs[i] += self.device_energy_weights[device_type] * csum_engy \
                                   + self.edge_energy_weights[device_type] * edge_comp_engy
                
                # if t_id % 20 == 0 and e_id % 20 == 0 and j == 0:
                #     print("[DEBUG] The device index is: ", i)
                    # print("[DEBUG] The task", j ,"'s dly_cons is: ", task.dly_cons, " The comp_dly is: ", comp_dly)
                    # print("[DEBUG] The task", j ,"'s norm_csum_engy is: ", task.norm_csum_engy, " The csum_engy is: ", csum_engy)
                    # print("[DEBUG] The task", j ,"'s norm_esum_engy is: ", task.norm_esum_engy, " The edge_comp_engy is: ", edge_comp_engy)
                    # print("[DEBUG] The device", i, "'s virtual comp ql is: ", self.device_envs[i].virtual_comp_ql)
                    # print("[DEBUG] The device", i, "'s completed comp is: ", self.device_envs[i].completed_comp)
                # 计算超时惩罚，其中task.dly_cons是按照设备计算能力为2Gcycles/s计算的，实际的设备计算能力在2.1~2.4Gcycles/s之间
                if comp_dly > task.dly_cons:
                    #! 考虑到每个任务的超时程度会影响到任务的执行效果，在原有惩罚的基础上多乘一个log函数（表示超时程度）
                    # device_rewards[i] += -5000 * torch.log(torch.exp(torch.tensor(1.0)) -1.0 + comp_dly / task.dly_cons)
                    device_rewards[i] += -5000
                    device_overtime_nums[i] += 1
                    #! 当设备i超时严重时，其他设备的动态时间阈值调整系数应适当增大
                    # if t_id % 20 == 0 and e_id % 20 == 0 and j == 0:
                    #     print("[DEBUG] The ratio of comp_dly to task.dly_cons in device", i, "is: ", comp_dly / task.dly_cons)
                    # if comp_dly > 1.5 * task.dly_cons:
                    #     for k in range(self.device_num):
                    #         if k == i:
                    #             continue
                    #         device_delay_adjust_coefs[k] = max(device_delay_adjust_coefs[k], comp_dly / task.dly_cons)
                else:
                    norm_csum_engy = task.norm_csum_engy
                    norm_esum_engy = task.norm_esum_engy
                    # 奖励函数 能耗比重 *实际总能耗 / 标准化能耗 + 成本比重 *实际总成本 / 标准化成本
                    device_rewards[i] += -1000 * (self.device_energy_weights[device_type] * 
                                                  csum_engy / norm_csum_engy
                                                  + self.edge_energy_weights[device_type] * 
                                                  edge_comp_engy / norm_esum_engy
                                                  )
                if(enable_print): print(f"[DEBUG] The device", i, "'s navie reward is: ", device_rewards[i])
                device_queue_actual_rewards[i] = 0.0
                device_queue_virtual_rewards[i] = 0.0
                if(self.enable_actual_queue_reward):
                    device_queue_actual_rewards[i] = self.local_reward_weight * \
                            self.device_envs[i].time_ql * (self.device_envs[i].new_ql_change)
                    device_queue_actual_rewards[i] = min(max(-2000, device_queue_actual_rewards[i]), 2000)
                if(self.enable_virtual_queue_reward):
                    device_queue_virtual_rewards[i] = self.local_reward_weight * \
                            self.device_envs[i].virtual_time_ql * (self.device_envs[i].new_vir_ql_change)
                    device_queue_virtual_rewards[i] = min(max(-6000, device_queue_virtual_rewards[i]), 6000)
                if(enable_print): print(f"[DEBUG] The device", i, "'s device_queue_actual_rewards is: ", device_queue_actual_rewards[i])
                if(enable_print): print(f"[DEBUG] The device", i, "'s device_queue_virtual_rewards is: ", device_queue_virtual_rewards[i])
                if visualize:
                    writer.add_scalars(
                        f"detail/dev_reward_{i}",
                        {f"ep_{e_id}_act": device_queue_actual_rewards[i]},
                        t_id
                    )
                    writer.add_scalars(
                        f"detail/dev_reward_{i}",
                        {f"ep_{e_id}_vir": device_queue_virtual_rewards[i]},
                        t_id
                    )
                    writer.add_scalars(
                        f"detail/dev_reward_{i}",
                        {f"ep_{e_id}_navie": device_rewards[i]},
                        t_id
                    )

        actual_queue_type_scale_fac = self.device_num / self.device_type_num * 2000
        virtual_queue_type_scale_fac = self.device_num / self.device_type_num * 6000
        for i in range(edge_queue_num):
            edge_queue_actual_rewards[i] = 0.0
            edge_queue_virtual_rewards[i] = 0.0
            if(self.enable_actual_queue_reward):
                edge_queue_actual_rewards[i] = self.edge_reward_weight * \
                            self.edge_env.edge_queue_time_ql[i] * (self.edge_env.new_edge_ql_change[i])
                edge_queue_actual_rewards[i] = min(max(-actual_queue_type_scale_fac, edge_queue_actual_rewards[i]), actual_queue_type_scale_fac)
            if(self.enable_virtual_queue_reward):
                edge_queue_virtual_rewards[i] = self.edge_reward_weight * \
                    self.edge_env.virtual_edge_queue_time_ql[i] * (self.edge_env.new_vir_edge_ql_change[i])
                edge_queue_virtual_rewards[i] = min(max(-virtual_queue_type_scale_fac, edge_queue_virtual_rewards[i]), virtual_queue_type_scale_fac)
            if(enable_print): print(f"[DEBUG] The edge_queue", i, "'s edge_queue_actual_rewards is: ", edge_queue_actual_rewards[i])
            if(enable_print): print(f"[DEBUG] The edge_queue", i, "'s edge_queue_virtual_rewards is: ", edge_queue_virtual_rewards[i])
            if visualize:
                writer.add_scalars(
                    f"detail/edge_reward_{i}",
                    {f"ep_{e_id}_act": edge_queue_actual_rewards[i]},
                    t_id
                )
                writer.add_scalars(
                    f"detail/edge_reward_{i}",
                    {f"ep_{e_id}_vir": edge_queue_virtual_rewards[i]},
                    t_id
                )

        for i in range(self.device_num):
            device_rewards[i] += device_queue_actual_rewards[i] + device_queue_virtual_rewards[i] + \
                edge_queue_actual_rewards[self.device_envs[i].device_type] / self.device_num_per_type[self.device_envs[i].device_type] + \
                edge_queue_virtual_rewards[self.device_envs[i].device_type] / self.device_num_per_type[self.device_envs[i].device_type]
            if(enable_print): print(f"[DEBUG] The device", i, "'s final reward is: ", device_rewards[i])
            if visualize:
                writer.add_scalars(
                    f"detail/dev_reward_{i}",
                    {f"ep_{e_id}_final": device_rewards[i]},
                    t_id
                )
                writer.add_scalars(
                    f"detail/dev_timeout_num_{i}",
                    {f"ep_{e_id}": device_overtime_nums[i]},
                    t_id
                )

        joint_reward = sum(device_rewards)
        joint_cost = sum(device_costs)
        
        if visualize:
            writer.add_scalars(
                f"detail/joint_reward",
                {f"ep_{e_id}": joint_reward},
                t_id
            )
            writer.add_scalars(
                f"detail/joint_cost",
                {f"ep_{e_id}": joint_cost},
                t_id
            )

        # next obs
        next_edge_obs = self.edge_env.get_obs()
        next_device_obss = [None for i in range(self.device_num)]
        for i in range(self.device_num):
            next_device_obss[i] = self.device_envs[i].get_obs()
        
        # #! 更新其他设备的动态时间阈值调整系数
        # for i in range(self.device_num):
        #     self.device_envs[i].adjust_delay_threshold_coef(device_delay_adjust_coefs[i])

        return joint_reward, device_rewards, \
               joint_cost, device_costs, \
               device_comp_dlys, device_csum_engys, \
               device_esum_engys, device_overtime_nums, \
               next_edge_obs, next_device_obss,device_task_is_available