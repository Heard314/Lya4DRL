import config.global_params as gp
class EdgeEnv():
    def __init__(self, general_params, writer = None):
        
        # Summary Writer
        self.writer = writer

        # task generation cycle
        self.gen_task_cycle = general_params.gen_task_cycle
        self.start_slot = general_params.start_slot

        # unit: s
        self.delta = general_params.delta
        
        self.edge_queue_num = general_params.device_type_num
        self.device_types_ref = general_params.device_types
        # unit: Gcycles/s
        self.edge_comp_freq = general_params.edge_comp_freq
        
        self.device_num = general_params.device_num
        self.device_num_per_type = general_params.device_num_per_type

        # Init: split server freq across queues proportionally
        self.alloc_edge_freq = [self.edge_comp_freq * self.device_num_per_type[i] / self.device_num for i in range(self.edge_queue_num)]

        self.edge_weight_w = general_params.edge_weight_w
        self.edge_weight_b = general_params.edge_weight_b

        # unit: s
        self.edge_queue_time_ql = [ 0 for _ in range(self.edge_queue_num)]
        self.old_edge_queue_time_ql = [ 0 for _ in range(self.edge_queue_num)]
        self.virtual_edge_queue_time_ql = [ 0 for _ in range(self.edge_queue_num)]
        self.old_virtual_edge_queue_time_ql = [ 0 for _ in range(self.edge_queue_num)]
        self.new_edge_ql_change = [ 0 for _ in range(self.edge_queue_num)]
        self.new_vir_edge_ql_change = [ 0 for _ in range(self.edge_queue_num)]
        self.edge_dly_adj_fac = general_params.edge_dly_adj_fac
        self.edge_dly_adj_val = [self.edge_dly_adj_fac[i] * general_params.comp_dly_thre[i] * self.delta for i in range(self.edge_queue_num)]
        self.avg_edge_time = [ 0 for _ in range(self.edge_queue_num)]
        self.old_comp_times =[ [] for _ in range(self.edge_queue_num)]
        self.statSlotNum = general_params.statSlotNum
        self.delta_num = 0 # elapsed slots in this episode

        self.total_comp_time = [0 for _ in range(self.edge_queue_num)]
        self.total_comp_amount = [0 for _ in range(self.edge_queue_num)]

        self.edge_act_reward_fac = [1.0 for _ in range(self.edge_queue_num)]
        self.avg_data_sizes = [(general_params.data_size_inls[i][0]+general_params.data_size_inls[i][1])/2 for i in range(self.edge_queue_num)]
        self.avg_comp_denss = [(general_params.comp_dens_inls[i][0]+general_params.comp_dens_inls[i][1])/2 for i in range(self.edge_queue_num)]
        # To balance reward_act and reward_vir, add a scaling factor to the growth rate (inversely proportional to task computation time)
        self.edge_act_reward_fac = [self.edge_act_reward_fac[i] / (0.8 * self.device_num_per_type[i] * self.avg_data_sizes[i] * self.avg_comp_denss[i] / self.alloc_edge_freq[i]) for i in range(self.edge_queue_num)]
        self.edge_act_reward_fac = [self.edge_act_reward_fac[i] * self.edge_act_reward_fac[i] for i in range(self.edge_queue_num)]
        print(f"[DEBUG] The edge's edge_act_reward_fac are {self.edge_act_reward_fac}")
        
        self.edge_act_queue_growth_rate = general_params.edge_act_queue_growth_rate
        self.edge_vir_queue_growth_rate = general_params.edge_vir_queue_growth_rate

    def reset(self):
        # reset computation-queue length
        self.alloc_edge_freq = [self.edge_comp_freq / self.edge_queue_num for _ in range(self.edge_queue_num)]
        self.total_comp_time = [0 for _ in range(self.edge_queue_num)]
        self.total_comp_amount = [0 for _ in range(self.edge_queue_num)]
        # reset computation time queue length
        self.edge_queue_time_ql = [ 0 for _ in range(self.edge_queue_num)]
        self.old_edge_queue_time_ql = [ 0 for _ in range(self.edge_queue_num)]
        self.virtual_edge_queue_time_ql = [ 0 for _ in range(self.edge_queue_num)]
        self.old_virtual_edge_queue_time_ql = [ 0 for _ in range(self.edge_queue_num)]
        self.avg_edge_time = [ 0 for _ in range(self.edge_queue_num)]
        self.old_comp_times =[ [] for _ in range(self.edge_queue_num)]
        self.new_edge_ql_change = [ 0 for _ in range(self.edge_queue_num)]
        self.new_vir_edge_ql_change = [ 0 for _ in range(self.edge_queue_num)]
        self.delta_num = 0
        
    def get_obs(self):

        obs = []
        # print(f"[DEBUG] the edge_queue_time_ql is {self.edge_queue_time_ql}")
        # print(f"[DEBUG] the virtual_edge_queue_time_ql is {self.virtual_edge_queue_time_ql}")
        for i in range(self.edge_queue_num):
            obs.append(self.edge_queue_time_ql[i])
            obs.append(self.virtual_edge_queue_time_ql[i])
        return obs
    

    def compute(self, device_sched_tasks, e_id, t_id, visualize=False):
        writer = self.writer
        enable_print = gp.settings.enable_print
        device_type_sched_tasks = [[] for _ in range(self.edge_queue_num)]
        for device_id, sched_tasks in enumerate(device_sched_tasks):
            device_type = self.device_types_ref[device_id]
            device_type_sched_tasks[device_type] += sched_tasks

        # Process per type, FIFO by trans_time
        device_type_sched_tasks = [
            sorted(sub_list, key=lambda x: x.trans_time)
            for sub_list in device_type_sched_tasks
        ]

        for i in range(self.edge_queue_num):
            self.old_edge_queue_time_ql[i] = self.edge_queue_time_ql[i]
            # if(enable_print): print(f"[DEBUG] Before compute, edge time_ql of device_type: {i} is {self.edge_queue_time_ql[i]}")
        alloc_edge_freq = self.alloc_edge_freq
        total_comp_used_time_this_epi = [min(self.delta, self.edge_queue_time_ql[i]) for i in range(self.edge_queue_num)]
        comp_dlys = [min(self.delta, self.edge_queue_time_ql[i]) for i in range(self.edge_queue_num)]
        self.delta_num += 1
        for device_type in range(self.edge_queue_num):
            if(enable_print): print(f"[DEBUG] We are processing the tasks of type {device_type}")
            # new_edge_comp = 0
            total_comp_need_time_this_epi = 0
            
            for i,task in enumerate(device_type_sched_tasks[device_type]):
                if task.trans_time == 0:
                    task.e_comp_dly = 0
                    task.e_queue_dly = 0
                    task.e_proc_dly = 0
                    task.edge_comp_engy = 0
                    if(enable_print): print(f"[DEBUG] the edge comp_dly in device {task.device_id} is {task.e_comp_dly}")
                else:
                    # new_edge_comp += task.offl_dz * task.comp_dens
                    if(enable_print): print(f"[DEBUG] The edge comp of task in device {task.device_id} is {task.offl_dz * task.comp_dens}")
                    if(enable_print): print(f"[DEBUG] The edge calculated amount in device {task.device_id} is {min(task.offl_dz * task.comp_dens, alloc_edge_freq[device_type] * max(0, self.delta - max(comp_dlys[device_type], task.trans_time)))}")
                    
                    # config: energy coef = 1 J/GFlops
                    task.edge_comp_engy = task.offl_dz * task.comp_dens * pow(alloc_edge_freq[device_type],2)
                    # if(enable_print): print(f"[DEBUG] The edge freq pow2 is {pow(alloc_edge_freq[device_type],2)}")
                    task.e_queue_dly = max(self.total_comp_time[device_type] - t_id * self.delta, 0)
                    task.e_proc_dly = task.offl_dz * task.comp_dens / alloc_edge_freq[device_type]
                    task.e_comp_dly = max(task.e_queue_dly, task.trans_time) + task.e_proc_dly
                    total_comp_need_time_this_epi += task.e_proc_dly
                    total_comp_used_time_this_epi[device_type] += min(task.offl_dz * task.comp_dens/alloc_edge_freq[device_type], max(0, self.delta-max(task.trans_time, comp_dlys[device_type])))
                    if(enable_print): print(f"[DEBUG] the edge comp_dly in device {task.device_id} is {task.e_comp_dly}")
                    comp_dlys[device_type] = max(comp_dlys[device_type], task.trans_time) + task.e_proc_dly
                    self.total_comp_time[device_type] += task.e_proc_dly
                    self.total_comp_amount[device_type] += task.offl_dz * task.comp_dens
        
                # avg_edge_time: mean over last statSlotNum slots
                self.old_comp_times[device_type].append(total_comp_need_time_this_epi)
                tail = self.old_comp_times[device_type][-self.statSlotNum:]
                self.avg_edge_time[device_type] = sum(tail) / len(tail) if tail else 0
            
            edge_act_queue_growth_rate = self.edge_act_queue_growth_rate
            self.edge_queue_time_ql[device_type] = max(self.edge_queue_time_ql[device_type] + edge_act_queue_growth_rate * (total_comp_need_time_this_epi - total_comp_used_time_this_epi[device_type]), 0)
            
            old_edge_queue_time_ql_ = self.edge_queue_time_ql[device_type]
            self.new_edge_ql_change[device_type] = self.edge_queue_time_ql[device_type] - old_edge_queue_time_ql_
            self.act_backlog = total_comp_need_time_this_epi
            self.old_virtual_edge_queue_time_ql[device_type] = self.virtual_edge_queue_time_ql[device_type]
            edge_vir_queue_growth_rate = self.edge_vir_queue_growth_rate
            old_virtual_edge_queue_time_ql_ = self.virtual_edge_queue_time_ql[device_type]
            EPS = 1e-6
            if self.avg_edge_time[device_type] > EPS:
                self.virtual_edge_queue_time_ql[device_type] = max(self.virtual_edge_queue_time_ql[device_type] + edge_vir_queue_growth_rate*(self.edge_queue_time_ql[device_type]/self.avg_edge_time[device_type]*self.delta*self.gen_task_cycle - self.edge_dly_adj_val[device_type]), 0)
                self.new_vir_edge_ql_change[device_type] = self.virtual_edge_queue_time_ql[device_type] - old_virtual_edge_queue_time_ql_
                self.vir_backlog = self.edge_queue_time_ql[device_type]/self.avg_edge_time[device_type]

            if(enable_print): print(f"[DEBUG] The edge_queue", device_type, "'s old_edge_queue_time_ql is: ", self.old_edge_queue_time_ql[device_type])
            if(enable_print): print(f"[DEBUG] The edge_queue", device_type, "'s old_virtual_edge_queue_time_ql is: ", self.old_virtual_edge_queue_time_ql[device_type])
            if(enable_print): print(f"[DEBUG] The edge_queue", device_type, "'s new_edge_ql_change is: ", self.new_edge_ql_change[device_type])
            if(enable_print): print(f"[DEBUG] The edge_queue", device_type, "'s new_vir_edge_ql_change is: ", self.new_vir_edge_ql_change[device_type])
            if(enable_print): print(f"[DEBUG] The edge_queue", device_type, "'s edge_queue_time_ql is: ", self.edge_queue_time_ql[device_type])
            if(enable_print): print(f"[DEBUG] The edge_queue", device_type, "'s virtual_edge_queue_time_ql is: ", self.virtual_edge_queue_time_ql[device_type])
            
            if visualize:
                writer.add_scalars(
                    f"detail/avg_edge_time_{device_type}",
                    {f"ep_{e_id}": self.avg_edge_time[device_type]},
                    t_id
                )
                writer.add_scalars(
                    f"detail/edge_time_ql_{device_type}",
                    {f"ep_{e_id}_act": self.edge_queue_time_ql[device_type]},
                    t_id
                )
                writer.add_scalars(
                    f"detail/edge_time_ql_{device_type}",
                    {f"ep_{e_id}_act_chg": self.new_edge_ql_change[device_type]},
                    t_id
                )
                writer.add_scalars(
                    f"detail/edge_time_ql_{device_type}",
                    {f"ep_{e_id}_vir": self.virtual_edge_queue_time_ql[device_type]},
                    t_id
                )
                writer.add_scalars(
                    f"detail/edge_time_ql_{device_type}",
                    {f"ep_{e_id}_vir_chg": self.new_vir_edge_ql_change[device_type]},
                    t_id
                )