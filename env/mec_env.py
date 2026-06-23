from env.device_env import DeviceEnv
from env.edge_env import EdgeEnv
import config.global_params as gp

class MECEnv():
    def __init__(self, gen_params, time_slots, writer=None):
        # Summary Writer
        self.writer = writer
        self.device_num = gen_params.device_num
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
        self.edge_server_num = gen_params.edge_server_num
        self.edge_envs = [EdgeEnv(gen_params, writer) for _ in range(self.edge_server_num)]

        self.device_freqs = gen_params.device_comp_freqs

        # device envs
        self.device_envs = []
        for i in range(self.device_num):
            self.device_envs.append(DeviceEnv(i, gen_params, self.edge_envs[0], writer))

        # reward parameters
        self.device_act_queue_reward_weight = gen_params.device_act_queue_reward_weight
        self.device_vir_queue_reward_weight = gen_params.device_vir_queue_reward_weight

        self.edge_queue_reward_bound_fac = gen_params.edge_queue_reward_bound_fac

        self.enable_actual_queue_reward = gen_params.enable_actual_queue_reward
        self.enable_virtual_queue_reward = gen_params.enable_virtual_queue_reward

        self.device_act_queue_reward_max_bound = gen_params.device_act_queue_reward_max_bound
        self.device_act_queue_reward_min_bound = gen_params.device_act_queue_reward_min_bound
        self.device_vir_queue_reward_max_bound = gen_params.device_vir_queue_reward_max_bound
        self.device_vir_queue_reward_min_bound = gen_params.device_vir_queue_reward_min_bound
        self.base_reward_penalty = gen_params.base_reward_penalty

        self.timeout_reward_penalty = gen_params.timeout_reward_penalty
        self.target_reward_penalty = gen_params.target_reward_penalty


    def reset(self):
        for edge_env in self.edge_envs:
            edge_env.reset()

        for i in range(self.device_num):
            self.device_envs[i].reset()

    def step(self, device_acts, e_id, t_id, visualize=False):
        
        gen_task_cycle = self.gen_task_cycle
        start_slot = self.start_slot

        writer = self.writer
        enable_print = gp.settings.enable_print
        # First, each device makes an offloading decision for the pending task
        # Then, execute the local computation part and return the remote offloading part (sched_tasks in the code below)
        device_sched_tasks = [None for i in range(self.device_num)]

        # Serial execution of device computations
        for i in range(self.device_num):
            sched_tasks = self.device_envs[i].compute(device_acts[i], e_id = e_id, t_id = t_id, visualize = visualize)
            device_sched_tasks[i] = sched_tasks

        # Route tasks to servers (no r_e softmax in v3)
        for s in range(self.edge_server_num):
            server_tasks = []
            for i in range(self.device_num):
                tasks = device_sched_tasks[i]
                if tasks:
                    server_tasks.extend([t for t in tasks if t.target_server == s])
            self.edge_envs[s].compute(server_tasks, e_id=e_id, t_id=t_id, s_id=s, visualize=visualize)
        
        # reward
        device_rewards = [self.base_reward_penalty for i in range(self.device_num)]
        device_queue_actual_rewards = [0 for i in range(self.device_num)]
        device_queue_virtual_rewards = [0 for i in range(self.device_num)]
        edge_queue_num = self.edge_server_num  # v3: per-server FIFO queues
        edge_queue_rewards = [0.0 for _ in range(edge_queue_num)]
        edge_queue_actual_rewards = [0.0 for _ in range(edge_queue_num)]
        edge_queue_virtual_rewards = [0.0 for _ in range(edge_queue_num)]
        joint_rewards = [0.0 for _ in range(self.device_type_num)]  # per-type for value nets
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
                
                if comp_dly > task.dly_cons:
                    device_overtime_nums[i] += 1

                if task.e_comp_dly > task.dly_cons:
                    task_type_in_edge_is_overtime[device_type] = True

                # print(f"[DEBUG] the comp_dly is {comp_dly}, the task.dly_cons is {task.dly_cons}")
                if comp_dly > task.dly_cons and not (self.enable_virtual_queue_reward or self.enable_actual_queue_reward):
                    device_rewards[i] += self.timeout_reward_penalty
                else:
                    norm_csum_engy_fac = task.norm_csum_engy_fac
                    norm_esum_engy_fac = task.norm_esum_engy_fac
                    device_rewards[i] += self.target_reward_penalty * (self.device_energy_weights[device_type] *
                                                  local_engy/norm_csum_engy_fac)
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
                        # if device_queue_actual_rewards[i] < 0:
                        #     device_queue_actual_rewards[i]  = min(-400, device_queue_actual_rewards[i])
                        device_queue_actual_rewards[i] = min(max(device_act_queue_reward_min_bound, device_queue_actual_rewards[i]), device_act_queue_reward_max_bound)
                if(self.enable_actual_queue_reward):
                    device_queue_actual_rewards[i] = device_act_reward_fac * self.device_act_queue_reward_weight * 3 * \
                            self.device_envs[i].time_ql * (self.device_envs[i].new_ql_change)
                    device_queue_actual_rewards[i] = min(max(device_act_queue_reward_min_bound * 3, device_queue_actual_rewards[i]), device_act_queue_reward_max_bound * 3)

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
            # v3: per-server edge rewards (1 FIFO queue per server)
            tightest_dly = min(self.comp_dly_thre)
            actual_scale_posfac = self.device_num * device_act_queue_reward_max_bound * self.edge_queue_reward_bound_fac
            virtual_scale_posfac = self.device_num * device_vir_queue_reward_max_bound * self.edge_queue_reward_bound_fac
            actual_scale_negfac = self.device_num * device_act_queue_reward_min_bound * self.edge_queue_reward_bound_fac
            virtual_scale_negfac = self.device_num * device_vir_queue_reward_min_bound * self.edge_queue_reward_bound_fac

            edge_act_queue_reward_weight = self.device_act_queue_reward_weight * self.device_num * 1.0 / self.delta / tightest_dly
            edge_vir_queue_reward_weight = self.device_vir_queue_reward_weight * self.device_num

            edge_queue_rewards = [0.0] * edge_queue_num
            edge_queue_actual_rewards = [0.0] * edge_queue_num
            edge_queue_virtual_rewards = [0.0] * edge_queue_num

            any_overtime = any(task_type_in_edge_is_overtime)

            for s in range(self.edge_server_num):
                edge_env = self.edge_envs[s]

                if self.enable_virtual_queue_reward:
                    edge_queue_virtual_rewards[s] += edge_vir_queue_reward_weight * \
                        edge_env.virtual_edge_queue_time_ql * edge_env.new_vir_edge_ql_change
                    if any_overtime:
                        edge_queue_actual_rewards[s] += edge_act_queue_reward_weight * \
                            edge_env.edge_queue_time_ql * edge_env.new_edge_ql_change

                if self.enable_actual_queue_reward:
                    edge_queue_actual_rewards[s] += edge_act_queue_reward_weight * 3 * \
                        edge_env.edge_queue_time_ql * edge_env.new_edge_ql_change

                edge_queue_virtual_rewards[s] = min(max(virtual_scale_negfac, edge_queue_virtual_rewards[s]), virtual_scale_posfac)
                edge_queue_actual_rewards[s] = min(max(actual_scale_negfac, edge_queue_actual_rewards[s]), actual_scale_posfac)

                edge_queue_rewards[s] = edge_queue_actual_rewards[s] + edge_queue_virtual_rewards[s]

                if enable_print:
                    print(f"[DEBUG] The edge server {s}'s edge_queue_actual_rewards is: {edge_queue_actual_rewards[s]}")
                    print(f"[DEBUG] The edge server {s}'s edge_queue_virtual_rewards is: {edge_queue_virtual_rewards[s]}")
                if visualize:
                    writer.add_scalars(
                        f"detail{'_eval' if gp.settings.is_evaluate else ''}/edge_reward_s{s}",
                        {f"ep_{e_id}_act": edge_queue_actual_rewards[s]},
                        t_id
                    )
                    writer.add_scalars(
                        f"detail{'_eval' if gp.settings.is_evaluate else ''}/edge_reward_s{s}",
                        {f"ep_{e_id}_vir": edge_queue_virtual_rewards[s]},
                        t_id
                    )

            # Assemble per-type joint rewards (all servers' edge rewards shared across types)
            for i in range(self.device_type_num):
                joint_rewards[i] = sum(edge_queue_rewards)
                joint_cost_per_type = 0.0
                for j in self.device_in_types[i]:
                    joint_rewards[i] += device_rewards[j]
                    joint_cost_per_type += device_costs[j]
                if visualize:
                    writer.add_scalars(
                        f"detail{'_eval' if gp.settings.is_evaluate else ''}/joint_reward_{i}",
                        {f"ep_{e_id}": joint_rewards[i]},
                        t_id
                    )
                    writer.add_scalars(
                        f"detail{'_eval' if gp.settings.is_evaluate else ''}/joint_cost_{i}",
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

        # next obs — by server (shared across all devices): [s0_act, s0_vir, s1_act, s1_vir, s2_act, s2_vir]
        next_edge_obs = []
        for edge_env in self.edge_envs:
            next_edge_obs.extend(edge_env.get_obs())
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