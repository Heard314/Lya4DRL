from env import device_env
from env.device_env import DeviceEnv
from env.edge_env import EdgeEnv
import torch
import math
import config.global_params as gp

from concurrent.futures import ThreadPoolExecutor
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

        self.device_freqs = gen_params.device_comp_freqs

        # device envs
        self.device_envs = []
        for i in range(self.device_num):
            self.device_envs.append(DeviceEnv(i, gen_params, self.edge_env, writer))
        # self.lyaV = gen_params.lyaV
        self.device_queue_reward_weight = gen_params.device_queue_reward_weight
        self.edge_queue_reward_weight = gen_params.edge_queue_reward_weight

        print(f"[DEBUG] device_queue_reward_weight: {self.device_queue_reward_weight}")
        print(f"[DEBUG] edge_queue_reward_weight: {self.edge_queue_reward_weight}")

        # self.device_local_reward_bound = gen_params.local_reward_bound
        # self.edge_reward_bound = gen_params.edge_reward_bound

        self.enable_actual_queue_reward = gen_params.enable_actual_queue_reward
        self.enable_virtual_queue_reward = gen_params.enable_virtual_queue_reward

        self.device_act_queue_reward_max_bound = gen_params.device_act_queue_reward_max_bound
        self.device_act_queue_reward_min_bound = gen_params.device_act_queue_reward_min_bound
        self.device_vir_queue_reward_max_bound = gen_params.device_vir_queue_reward_max_bound
        self.device_vir_queue_reward_min_bound = gen_params.device_vir_queue_reward_min_bound
    
        print(f"[DEBUG] device_act_queue_reward_max_bound: {self.device_act_queue_reward_max_bound}")
        print(f"[DEBUG] device_act_queue_reward_min_bound: {self.device_act_queue_reward_min_bound}")
        print(f"[DEBUG] device_vir_queue_reward_max_bound: {self.device_vir_queue_reward_max_bound}")
        print(f"[DEBUG] device_vir_queue_reward_min_bound: {self.device_vir_queue_reward_min_bound}")

        self.timeout_reward_penalty = gen_params.timeout_reward_penalty
        self.target_reward_penalty = gen_params.target_reward_penalty

        print(f"[DEBUG] timeout_reward_penalty: {self.timeout_reward_penalty}")
        print(f"[DEBUG] target_reward_penalty: {self.target_reward_penalty}")

        # device_env info print
        print(f"[DEBUG] device_act_queue_growth_rate: {gen_params.device_act_queue_growth_rate}")
        print(f"[DEBUG] device_vir_queue_growth_rate: {gen_params.device_vir_queue_growth_rate}")

        # edge_env info print
        print(f"[DEBUG] edge_act_queue_growth_rate: {gen_params.edge_act_queue_growth_rate}")
        print(f"[DEBUG] edge_vir_queue_growth_rate: {gen_params.edge_vir_queue_growth_rate}")

        # create thread pool for parallel device_env.compute
        # each worker thread handles one DeviceEnv
        self._device_executor = ThreadPoolExecutor(max_workers=self.device_num)

    def reset(self):
        edge_obs = self.edge_env.reset()
        
        device_obss = [None for i in range(self.device_num)]
        for i in range(self.device_num):
            device_obss[i] = self.device_envs[i].reset()
        
        return edge_obs, device_obss
    
    def step(self, device_acts, e_id, t_id, visualize=False):

        writer = self.writer
        if e_id % 50 == 1:
            # gp.settings.enable_print = True
            gp.settings.enable_print = False
        else:
            gp.settings.enable_print = False
        enable_print = gp.settings.enable_print
        # 首先每个设备对待执行任务做出卸载决策，然后执行任务的本地计算部分，返回远程卸载部分（以下代码中的sched_tasks）
        device_sched_tasks = [None for i in range(self.device_num)]

        # Serial execution of device computations
        for i in range(self.device_num):
            sched_tasks = self.device_envs[i].compute(device_acts[i], e_id = e_id, t_id = t_id, visualize = visualize)
            device_sched_tasks[i] = sched_tasks

        # Parallel execution of device computations
        # submit compute tasks to thread pool
        # futures = []
        # for i in range(self.device_num):
        #     env = self.device_envs[i]
        #     act = device_acts[i]
        #     future = self._device_executor.submit(
        #         env.compute, act, e_id, t_id, visualize
        #     )
        #     futures.append((i, future))

        # # collect results
        # for i, future in futures:
        #     device_sched_tasks[i] = future.result()

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
                # print(f"[GDEBUG] the comp_dly in device {i} is {comp_dly}")
                # print(f"[GDEBUG] the local comp_dly in device {i} is {task.l_comp_dly}")
                # print(f"[GDEBUG] the edge comp_dly in device {i} is {task.e_comp_dly}")
                local_engy = task.local_comp_engy + task.tran_engy
                if(enable_print): print(f"[DEBUG] the local local_engy in device {i} is {task.local_comp_engy}")
                if(enable_print): print(f"[DEBUG] the tran local_engy in device {i} is {task.tran_engy}")
                if visualize:
                    writer.add_scalars(
                        f"detail/engy_{i}",
                        {f"ep_{e_id}_local": task.local_comp_engy + task.tran_engy},
                        t_id
                    )
                    # writer.add_scalars(
                    #     f"detail/engy_{i}",
                    #     {f"ep_{e_id}_edge": task.edge_comp_engy},
                    #     t_id
                    # )


                device_csum_engys[i] += 1 / (j + 1) * (local_engy - device_csum_engys[i])
                edge_comp_engy = task.edge_comp_engy
                # device_esum_engys[i] += 1 / (j + 1) * (edge_comp_engy - device_esum_engys[i])
                
                device_costs[i] += self.device_energy_weights[device_type] * local_engy 
                                #    + self.edge_energy_weights[device_type] * edge_comp_engy
                
                # if t_id % 20 == 0 and e_id % 20 == 0 and j == 0:
                #     print("[DEBUG] The device index is: ", i)
                    # print("[DEBUG] The task", j ,"'s dly_cons is: ", task.dly_cons, " The comp_dly is: ", comp_dly)
                    # print("[DEBUG] The task", j ,"'s norm_csum_engy is: ", task.norm_csum_engy, " The local_engy is: ", local_engy)
                    # print("[DEBUG] The task", j ,"'s norm_esum_engy is: ", task.norm_esum_engy, " The edge_comp_engy is: ", edge_comp_engy)
                    # print("[DEBUG] The device", i, "'s virtual comp ql is: ", self.device_envs[i].virtual_comp_ql)
                    # print("[DEBUG] The device", i, "'s completed comp is: ", self.device_envs[i].completed_comp)
                # 计算超时惩罚，其中task.dly_cons是按照设备计算能力为2Gcycles/s计算的，实际的设备计算能力在2.1~2.4Gcycles/s之间
                if comp_dly > task.dly_cons:
                    #! 考虑到每个任务的超时程度会影响到任务的执行效果，在原有惩罚的基础上多乘一个log函数（表示超时程度）
                    # device_rewards[i] += -5000 * torch.log(torch.exp(torch.tensor(1.0)) -1.0 + comp_dly / task.dly_cons)
                    device_rewards[i] += self.timeout_reward_penalty
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
                    device_rewards[i] += self.target_reward_penalty * (self.device_energy_weights[device_type] * 
                                                  local_engy / norm_csum_engy
                                                #   + self.edge_energy_weights[device_type] * 
                                                #   edge_comp_engy / norm_esum_engy
                                                  )
                if(enable_print): print(f"[DEBUG] The device", i, "'s navie reward is: ", device_rewards[i])
                device_queue_actual_rewards[i] = 0.0
                device_queue_virtual_rewards[i] = 0.0

                device_act_queue_reward_max_bound = self.device_act_queue_reward_max_bound #200
                device_act_queue_reward_min_bound = self.device_act_queue_reward_min_bound #-300
                device_vir_queue_reward_max_bound = self.device_vir_queue_reward_max_bound #800
                device_vir_queue_reward_min_bound = self.device_vir_queue_reward_min_bound #-1200

                if(self.enable_actual_queue_reward and self.enable_virtual_queue_reward):
                    device_queue_actual_rewards[i] = self.device_queue_reward_weight * \
                            self.device_envs[i].time_ql * (self.device_envs[i].new_ql_change)
                    device_queue_actual_rewards[i] = min(max(device_act_queue_reward_min_bound, device_queue_actual_rewards[i]), device_act_queue_reward_max_bound)
                    device_queue_virtual_rewards[i] = self.device_queue_reward_weight * \
                            self.device_envs[i].virtual_time_ql * (self.device_envs[i].new_vir_ql_change)
                    device_queue_virtual_rewards[i] = min(max(device_vir_queue_reward_min_bound, device_queue_virtual_rewards[i]), device_vir_queue_reward_max_bound)
                
                elif(self.enable_virtual_queue_reward):
                    device_queue_virtual_rewards[i] = self.device_queue_reward_weight * \
                            self.device_envs[i].virtual_time_ql * (self.device_envs[i].new_vir_ql_change)
                    device_queue_virtual_rewards[i] = min(max(device_act_queue_reward_min_bound + device_vir_queue_reward_min_bound, device_queue_virtual_rewards[i]), device_act_queue_reward_max_bound + device_vir_queue_reward_max_bound)
                
                elif(self.enable_actual_queue_reward):
                    device_queue_actual_rewards[i] = self.device_queue_reward_weight * \
                            self.device_envs[i].time_ql * (self.device_envs[i].new_ql_change)
                    device_queue_actual_rewards[i] = min(max(device_act_queue_reward_min_bound + device_vir_queue_reward_min_bound, device_queue_actual_rewards[i]), device_act_queue_reward_max_bound + device_vir_queue_reward_max_bound)

                # print(f"[DEBUG] The device", i, "'s device_queue_actual_rewards is: ", device_queue_actual_rewards[i])
                # print(f"[DEBUG] The device", i, "'s device_queue_virtual_rewards is: ", device_queue_virtual_rewards[i])

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

        for i in range(edge_queue_num):
            edge_queue_actual_rewards[i] = 0.0
            edge_queue_virtual_rewards[i] = 0.0
            # actual_queue_type_scale_posfac = self.device_num_per_type[i] * device_act_queue_reward_max_bound
            # virtual_queue_type_scale_posfac = self.device_num_per_type[i] * device_vir_queue_reward_max_bound
            # actual_queue_type_scale_negfac = self.device_num_per_type[i] * device_act_queue_reward_min_bound
            # virtual_queue_type_scale_negfac = self.device_num_per_type[i] * device_vir_queue_reward_min_bound
            actual_queue_type_scale_posfac = device_act_queue_reward_max_bound
            virtual_queue_type_scale_posfac = device_vir_queue_reward_max_bound
            actual_queue_type_scale_negfac = device_act_queue_reward_min_bound
            virtual_queue_type_scale_negfac = device_vir_queue_reward_min_bound
            if(self.enable_actual_queue_reward and self.enable_virtual_queue_reward):
                edge_queue_actual_rewards[i] = self.edge_queue_reward_weight * \
                    torch.max(1.0, torch.pow(self.edge_env.alloc_edge_freq[i]/self.device_freqs[i]/self.device_num_per_type[i],2)) * \
                    self.edge_env.edge_queue_time_ql[i] * (self.edge_env.new_edge_ql_change[i])
                edge_queue_actual_rewards[i] = min(max(actual_queue_type_scale_negfac, edge_queue_actual_rewards[i]), actual_queue_type_scale_posfac)
                edge_queue_virtual_rewards[i] = self.edge_queue_reward_weight * \
                    self.edge_env.virtual_edge_queue_time_ql[i] * (self.edge_env.new_vir_edge_ql_change[i])
                edge_queue_virtual_rewards[i] = min(max(virtual_queue_type_scale_negfac, edge_queue_virtual_rewards[i]), virtual_queue_type_scale_posfac)
            
            elif(self.enable_virtual_queue_reward):
                edge_queue_virtual_rewards[i] = self.edge_queue_reward_weight * \
                    self.edge_env.virtual_edge_queue_time_ql[i] * (self.edge_env.new_vir_edge_ql_change[i])
                edge_queue_virtual_rewards[i] = min(max(actual_queue_type_scale_negfac + virtual_queue_type_scale_negfac, edge_queue_virtual_rewards[i]), actual_queue_type_scale_posfac + virtual_queue_type_scale_posfac)
            
            elif(self.enable_actual_queue_reward):
                edge_queue_actual_rewards[i] = self.edge_queue_reward_weight * \
                    torch.max(1.0, torch.pow(self.edge_env.alloc_edge_freq[i]/self.device_freqs[i]/self.device_num_per_type[i],2)) * \
                    self.edge_env.edge_queue_time_ql[i] * (self.edge_env.new_edge_ql_change[i])
                edge_queue_actual_rewards[i] = min(max(actual_queue_type_scale_negfac + virtual_queue_type_scale_negfac, edge_queue_actual_rewards[i]), actual_queue_type_scale_posfac + virtual_queue_type_scale_posfac)

            # print(f"[DEBUG] The edge_queue", i, "'s edge_queue_actual_rewards is: ", edge_queue_actual_rewards[i])
            # print(f"[DEBUG] The edge_queue", i, "'s edge_queue_virtual_rewards is: ", edge_queue_virtual_rewards[i])

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
                edge_queue_actual_rewards[self.device_envs[i].device_type] + \
                edge_queue_virtual_rewards[self.device_envs[i].device_type]
                # edge_queue_actual_rewards[self.device_envs[i].device_type] / self.device_num_per_type[self.device_envs[i].device_type] + \
                # edge_queue_virtual_rewards[self.device_envs[i].device_type] / self.device_num_per_type[self.device_envs[i].device_type]

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
                # writer.add_scalars(
                #     f"overall/timeout_alldev_ep_{e_id}",
                #     {"device": device_overtime_nums[i]},
                #     i
                # )
                # writer.add_scalars(
                #     f"overall/comp_dly_alldev_ep_{e_id}",
                #     {"device": device_comp_dlys[i]},
                #     i
                # )

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