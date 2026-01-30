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
        self.device_in_types = gen_params.device_in_types
        self.comp_dly_thre = gen_params.comp_dly_thre
        self.delta = gen_params.delta
        # task generation cycle
        self.gen_task_cycle = gen_params.gen_task_cycle
        self.start_slot = gen_params.start_slot

        # edge env
        self.edge_env = EdgeEnv(gen_params, writer)

        self.device_freqs = gen_params.device_comp_freqs

        # device envs
        self.device_envs = []
        for i in range(self.device_num):
            self.device_envs.append(DeviceEnv(i, gen_params, self.edge_env, writer))

        # reward parameters
        self.device_act_queue_reward_weight = gen_params.device_act_queue_reward_weight
        self.device_vir_queue_reward_weight = gen_params.device_vir_queue_reward_weight

        print(f"[DEBUG] device_act_queue_reward_weight: {self.device_act_queue_reward_weight}")
        print(f"[DEBUG] device_vir_queue_reward_weight: {self.device_vir_queue_reward_weight}")

        self.edge_queue_reward_bound_fac = gen_params.edge_queue_reward_bound_fac

        self.enable_actual_queue_reward = gen_params.enable_actual_queue_reward
        self.enable_virtual_queue_reward = gen_params.enable_virtual_queue_reward

        self.device_act_queue_reward_max_bound = gen_params.device_act_queue_reward_max_bound
        self.device_act_queue_reward_min_bound = gen_params.device_act_queue_reward_min_bound
        self.device_vir_queue_reward_max_bound = gen_params.device_vir_queue_reward_max_bound
        self.device_vir_queue_reward_min_bound = gen_params.device_vir_queue_reward_min_bound
        self.base_reward_penalty = gen_params.base_reward_penalty

        print(f"[DEBUG] device_act_queue_reward_max_bound: {self.device_act_queue_reward_max_bound}")
        print(f"[DEBUG] device_act_queue_reward_min_bound: {self.device_act_queue_reward_min_bound}")
        print(f"[DEBUG] device_vir_queue_reward_max_bound: {self.device_vir_queue_reward_max_bound}")
        print(f"[DEBUG] device_vir_queue_reward_min_bound: {self.device_vir_queue_reward_min_bound}")

        self.timeout_reward_penalty = gen_params.timeout_reward_penalty
        self.lya_timeout_reward_penalty = gen_params.lya_timeout_reward_penalty
        self.target_reward_penalty = gen_params.target_reward_penalty

        print(f"[DEBUG] timeout_reward_penalty: {self.timeout_reward_penalty}")
        print(f"[DEBUG] lya_timeout_reward_penalty: {self.lya_timeout_reward_penalty}")
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
        self.edge_env.reset()
        
        for i in range(self.device_num):
            self.device_envs[i].reset()

    def step(self, device_acts, e_id, t_id, visualize=False):
        
        gen_task_cycle = self.gen_task_cycle
        start_slot = self.start_slot

        writer = self.writer
        if e_id % 50 == 1:
            # gp.settings.enable_print = True
            gp.settings.enable_print = False
        else:
            gp.settings.enable_print = False
        enable_print = gp.settings.enable_print
        # First, each device makes an offloading decision for the pending task
        # Then, execute the local computation part and return the remote offloading part (sched_tasks in the code below)
        device_sched_tasks = [None for i in range(self.device_num)]

        # Serial execution of device computations
        for i in range(self.device_num):
            sched_tasks = self.device_envs[i].compute(device_acts[i], e_id = e_id, t_id = t_id, visualize = visualize)
            device_sched_tasks[i] = sched_tasks

        # 边缘服务器执行任务的远程卸载部分
        self.edge_env.compute(device_sched_tasks, e_id = e_id, t_id = t_id, visualize = visualize)
        
        # reward
        device_rewards = [self.base_reward_penalty for i in range(self.device_num)]
        device_queue_actual_rewards = [0 for i in range(self.device_num)]
        device_queue_virtual_rewards = [0 for i in range(self.device_num)]
        edge_queue_num = self.device_type_num
        edge_queue_rewards = [0 for i in range(edge_queue_num)]
        edge_queue_actual_rewards = [0 for i in range(edge_queue_num)]
        edge_queue_virtual_rewards = [0 for i in range(edge_queue_num)]
        joint_rewards = [0 for i in range(edge_queue_num)]
        joint_cost = 0
        device_costs = [0 for i in range(self.device_num)]
        device_comp_dlys = [0 for i in range(self.device_num)]
        device_csum_engys = [0 for i in range(self.device_num)]
        device_esum_engys = [0 for i in range(self.device_num)]
        device_overtime_nums = [0 for i in range(self.device_num)]
        device_task_is_available = [False for i in range(self.device_num)] #在该时间间隙下是否有任务到达

        task_type_in_edge_is_overtime = [False for i in range(self.device_type_num)] # 在该时间间隙下该类型任务是否有超时

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
                        f"detail{'_eval' if gp.settings.is_evaluate else ''}/comp_dly_{i}",
                        {f"ep_{e_id}": comp_dly},
                        t_id
                    )
                    writer.add_scalars(
                        f"detail{'_eval' if gp.settings.is_evaluate else ''}/l_comp_dly_{i}",
                        {f"ep_{e_id}_total": task.l_comp_dly},
                        t_id
                    )
                    writer.add_scalars(
                        f"detail{'_eval' if gp.settings.is_evaluate else ''}/l_comp_dly_{i}",
                        {f"ep_{e_id}_queue": task.l_queue_dly},
                        t_id
                    )
                    writer.add_scalars(
                        f"detail{'_eval' if gp.settings.is_evaluate else ''}/l_comp_dly_{i}",
                        {f"ep_{e_id}_proc": task.l_proc_dly},
                        t_id
                    )
                    writer.add_scalars(
                        f"detail{'_eval' if gp.settings.is_evaluate else ''}/e_comp_dly_{i}",
                        {f"ep_{e_id}_total": task.e_comp_dly},
                        t_id
                    )
                    # writer.add_scalars(
                    #     f"detail/e_comp_dly_{i}",
                    #     {f"ep_{e_id}_tran": task.trans_time},
                    #     t_id
                    # )
                    writer.add_scalars(
                        f"detail{'_eval' if gp.settings.is_evaluate else ''}/e_comp_dly_{i}",
                        {f"ep_{e_id}_queue": task.e_queue_dly},
                        t_id
                    )
                    writer.add_scalars(
                        f"detail{'_eval' if gp.settings.is_evaluate else ''}/e_comp_dly_{i}",
                        {f"ep_{e_id}_proc": task.e_proc_dly},
                        t_id
                    )

                if(enable_print): print(f"[DEBUG] the comp_dly in device {i} is {comp_dly}")
                device_comp_dlys[i] += 1 / (j + 1) * (comp_dly - device_comp_dlys[i])
                local_engy = task.local_comp_engy + task.tran_engy
                if(enable_print): print(f"[DEBUG] the local local_engy in device {i} is {task.local_comp_engy}")
                if(enable_print): print(f"[DEBUG] the tran local_engy in device {i} is {task.tran_engy}")
                if visualize:
                    writer.add_scalars(
                        f"detail{'_eval' if gp.settings.is_evaluate else ''}/engy_{i}",
                        {f"ep_{e_id}_local": local_engy},
                        t_id
                    )

                device_csum_engys[i] += 1 / (j + 1) * (local_engy - device_csum_engys[i])
                edge_comp_engy = task.edge_comp_engy
                
                device_costs[i] += self.device_energy_weights[device_type] * local_engy 
                                #    + self.edge_energy_weights[device_type] * edge_comp_engy
                
                if comp_dly > task.dly_cons:
                    device_overtime_nums[i] += 1

                if task.e_comp_dly > task.dly_cons:
                    task_type_in_edge_is_overtime[device_type] = True

                # print(f"[DEBUG] the comp_dly is {comp_dly}, the task.dly_cons is {task.dly_cons}")
                if comp_dly > task.dly_cons and not (self.enable_virtual_queue_reward or self.enable_actual_queue_reward):
                    device_rewards[i] += self.timeout_reward_penalty
                elif comp_dly > task.dly_cons and self.enable_virtual_queue_reward:
                    device_rewards[i] += self.lya_timeout_reward_penalty
                else:
                    norm_csum_engy = task.norm_csum_engy
                    norm_esum_engy = task.norm_esum_engy
                    device_rewards[i] += self.target_reward_penalty * (self.device_energy_weights[device_type] * 
                                                  local_engy
                                                #   + self.edge_energy_weights[device_type] * 
                                                #   edge_comp_engy / norm_esum_engy
                                                  )
                # print(f"[DEBUG] the part of engy reward in device {i} is {self.target_reward_penalty * (self.device_energy_weights[device_type] * local_engy)}")
                if(enable_print): print(f"[DEBUG] The device", i, "'s navie reward is: ", device_rewards[i])
                device_queue_actual_rewards[i] = 0.0
                device_queue_virtual_rewards[i] = 0.0

                device_act_queue_reward_max_bound = self.device_act_queue_reward_max_bound #200
                device_act_queue_reward_min_bound = self.device_act_queue_reward_min_bound #-300
                device_vir_queue_reward_max_bound = self.device_vir_queue_reward_max_bound #800
                device_vir_queue_reward_min_bound = self.device_vir_queue_reward_min_bound #-1200

                device_act_reward_fac = self.device_envs[i].device_act_reward_fac                

                if(self.enable_virtual_queue_reward):
                    device_queue_virtual_rewards[i] = self.device_vir_queue_reward_weight * \
                            self.device_envs[i].virtual_time_ql * (self.device_envs[i].new_vir_ql_change)
                    device_queue_virtual_rewards[i] = min(max(device_vir_queue_reward_min_bound, device_queue_virtual_rewards[i]), device_vir_queue_reward_max_bound)
                    if (task.l_comp_dly > task.dly_cons):
                        device_queue_actual_rewards[i] = device_act_reward_fac * self.device_act_queue_reward_weight * \
                                self.device_envs[i].time_ql * (self.device_envs[i].new_ql_change)
                        device_queue_actual_rewards[i] = min(max(device_act_queue_reward_min_bound, device_queue_actual_rewards[i]), device_act_queue_reward_max_bound)
                    # if device_queue_actual_rewards[i] < 0:
                    #     device_queue_actual_rewards[i]  = min(-400, device_queue_actual_rewards[i])
                if(self.enable_actual_queue_reward):
                    device_queue_actual_rewards[i] = device_act_reward_fac * self.device_act_queue_reward_weight * 50 * \
                            self.device_envs[i].time_ql * (self.device_envs[i].new_ql_change)
                    device_queue_actual_rewards[i] = min(max(device_act_queue_reward_min_bound * 50, device_queue_actual_rewards[i]), device_act_queue_reward_max_bound * 50)

                if(enable_print): print(f"[DEBUG] The device", i, "'s device_queue_actual_rewards is: ", device_queue_actual_rewards[i])
                if(enable_print): print(f"[DEBUG] The device", i, "'s device_queue_virtual_rewards is: ", device_queue_virtual_rewards[i])
                if visualize:
                    writer.add_scalars(
                        f"detail{'_eval' if gp.settings.is_evaluate else ''}/dev_reward_{i}",
                        {f"ep_{e_id}_act": device_queue_actual_rewards[i]},
                        t_id
                    )
                    writer.add_scalars(
                        f"detail{'_eval' if gp.settings.is_evaluate else ''}/dev_reward_{i}",
                        {f"ep_{e_id}_vir": device_queue_virtual_rewards[i]},
                        t_id
                    )
                    writer.add_scalars(
                        f"detail{'_eval' if gp.settings.is_evaluate else ''}/dev_reward_{i}",
                        {f"ep_{e_id}_navie": device_rewards[i]},
                        t_id
                    )
                device_rewards[i] += device_queue_actual_rewards[i] + device_queue_virtual_rewards[i] 
                if(enable_print): print(f"[DEBUG] The device", i, "'s final reward is: ", device_rewards[i])
                if visualize:
                    writer.add_scalars(
                        f"detail{'_eval' if gp.settings.is_evaluate else ''}/dev_reward_{i}",
                        {f"ep_{e_id}_final": device_rewards[i]},
                        t_id
                    )
                    writer.add_scalars(
                        f"detail{'_eval' if gp.settings.is_evaluate else ''}/dev_timeout_num_{i}",
                        {f"ep_{e_id}": device_overtime_nums[i]},
                        t_id
                    )
        
        if t_id % gen_task_cycle == start_slot:
            for i in range(edge_queue_num):
                edge_queue_actual_rewards[i] = 0.0
                edge_queue_virtual_rewards[i] = 0.0
                actual_queue_type_scale_posfac = self.device_num_per_type[i]*device_act_queue_reward_max_bound*self.edge_queue_reward_bound_fac
                virtual_queue_type_scale_posfac = self.device_num_per_type[i]*device_vir_queue_reward_max_bound*self.edge_queue_reward_bound_fac
                actual_queue_type_scale_negfac = self.device_num_per_type[i]*device_act_queue_reward_min_bound*self.edge_queue_reward_bound_fac
                virtual_queue_type_scale_negfac = self.device_num_per_type[i]*device_vir_queue_reward_min_bound*self.edge_queue_reward_bound_fac
                edge_act_reward_fac = self.edge_env.edge_act_reward_fac[i]

                edge_act_queue_reward_weight = 0.85 * self.device_act_queue_reward_weight * self.device_num_per_type[i] * 1.0 / self.delta / self.comp_dly_thre[i]
                edge_vir_queue_reward_weight = 0.85 * self.device_vir_queue_reward_weight * self.device_num_per_type[i]

                if(self.enable_virtual_queue_reward):
                    edge_queue_virtual_rewards[i] = edge_vir_queue_reward_weight * \
                        self.edge_env.virtual_edge_queue_time_ql[i] * (self.edge_env.new_vir_edge_ql_change[i])
                    edge_queue_virtual_rewards[i] = min(max(virtual_queue_type_scale_negfac, edge_queue_virtual_rewards[i]), virtual_queue_type_scale_posfac)
                    if task_type_in_edge_is_overtime[i]:
                        edge_queue_actual_rewards[i] = edge_act_queue_reward_weight * \
                            self.edge_env.edge_queue_time_ql[i] * (self.edge_env.new_edge_ql_change[i])
                        # if edge_queue_actual_rewards[i] < 0:
                        #     edge_queue_actual_rewards[i] = min(-200 * self.device_num_per_type[i], edge_queue_actual_rewards[i])
                        edge_queue_actual_rewards[i] = min(max(actual_queue_type_scale_negfac, edge_queue_actual_rewards[i]), actual_queue_type_scale_posfac)

                if(self.enable_actual_queue_reward):
                    edge_queue_actual_rewards[i] = edge_act_queue_reward_weight * 50 * \
                        self.edge_env.edge_queue_time_ql[i] * (self.edge_env.new_edge_ql_change[i])
                    edge_queue_actual_rewards[i] = min(max(actual_queue_type_scale_negfac * 50, edge_queue_actual_rewards[i]), actual_queue_type_scale_posfac * 50)
    

                if(enable_print): print(f"[DEBUG] The edge_queue", i, "'s edge_queue_actual_rewards is: ", edge_queue_actual_rewards[i])
                if(enable_print): print(f"[DEBUG] The edge_queue", i, "'s edge_queue_virtual_rewards is: ", edge_queue_virtual_rewards[i])
                if visualize:
                    for j in self.device_in_types[i]:
                        writer.add_scalars(
                            f"detail{'_eval' if gp.settings.is_evaluate else ''}/dev_reward_{j}",
                            {f"ep_{e_id}_edge_act": edge_queue_actual_rewards[i]},
                            t_id
                        )
                        writer.add_scalars(
                            f"detail{'_eval' if gp.settings.is_evaluate else ''}/dev_reward_{j}",
                            {f"ep_{e_id}_edge_vir": edge_queue_virtual_rewards[i]},
                            t_id
                        )
                edge_queue_rewards[i] = edge_queue_actual_rewards[i] + edge_queue_virtual_rewards[i]
                joint_rewards[i] = edge_queue_rewards[i]
                joint_cost_per_type = 0.0
                for j in self.device_in_types[i]:
                    joint_rewards[i] += device_rewards[j]
                    joint_cost_per_type += device_costs[j]
                if visualize:
                    writer.add_scalars(
                        f"detail{'_eval' if gp.settings.is_evaluate else ''}/edge_joint_reward_{i}",
                        {f"ep_{e_id}": joint_rewards[i]},
                        t_id
                    )
                    writer.add_scalars(
                        f"detail{'_eval' if gp.settings.is_evaluate else ''}/edge_joint_cost_{i}",
                        {f"ep_{e_id}": joint_cost_per_type},
                        t_id
                    )
            joint_cost = sum(device_costs)
            
            if visualize:
                writer.add_scalars(
                    f"detail{'_eval' if gp.settings.is_evaluate else ''}/joint_cost",
                    {f"ep_{e_id}": joint_cost},
                    t_id
                )

        # next obs
        next_edge_obs = self.edge_env.get_obs()
        # print(f"next_edge_obs: {next_edge_obs}")
        next_device_obss = [None for i in range(self.device_num)]
        for i in range(self.device_num):
            next_device_obss[i] = self.device_envs[i].get_obs()
            # print(f"device_id {i}, next_device_obss: {next_device_obss[i]}")

        return joint_rewards, device_rewards, \
               joint_cost, device_costs, \
               device_comp_dlys, device_csum_engys, \
               device_esum_engys, device_overtime_nums, \
               next_edge_obs, next_device_obss,device_task_is_available