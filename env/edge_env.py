import config.global_params as gp
class EdgeEnv():
    def __init__(self, general_params, writer = None):
        
        # Summary Writer
        self.writer = writer
        # unit: s
        self.delta = general_params.delta
        
        self.edge_queue_num = general_params.device_type_num
        self.device_types_ref = general_params.device_types
        # unit: Gcycles/s
        self.edge_comp_freq = general_params.edge_comp_freq
        
        # prev: 根据各个任务类型的计算量与到达频率分配服务器的计算频率
        # edge_freq_weights = []
        # data_size_inls = general_params.data_size_inls
        # comp_dens_inls = general_params.comp_dens_inls
        # task_arrival_prob = general_params.task_arrival_prob
        # device_num_per_type = general_params.device_num_per_type
        
        # for i in range(self.edge_queue_num):
        #     dz_mean = (data_size_inls[i][0]+data_size_inls[i][1])/2
        #     # dz_mean = data_size_inls[i][1]
        #     dens_mean = (comp_dens_inls[i][0]+comp_dens_inls[i][1])/2
        #     # dens_mean = comp_dens_inls[i][1]
        #     device_num = device_num_per_type[i]
        #     edge_freq_weights.append((dz_mean*dens_mean*task_arrival_prob[i]-self.delta*general_params.device_comp_freqs[i])*device_num)

        # total_weight = sum(edge_freq_weights)
        # self.alloc_edge_freq = [w/total_weight*self.edge_comp_freq for w in edge_freq_weights]
        # 服务器计算频率初始时，每个计算队列分配均等的计算频率
        self.alloc_edge_freq = [self.edge_comp_freq / self.edge_queue_num for _ in range(self.edge_queue_num)]
        # print(f"[DEBUG] alloc_edge_freq: {self.alloc_edge_freq}")
        # unit: Gcycles
        # self.edge_queue_comp_ql = [ 0 for _ in range(self.edge_queue_num)]
        # self.virtual_edge_queue_comp_ql = [ 0 for _ in range(self.edge_queue_num)]
        # self.old_edge_queue_comp_ql = [ 0 for _ in range(self.edge_queue_num)]
        # self.old_virtual_edge_queue_comp_ql = [ 0 for _ in range(self.edge_queue_num)]

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
        self.delta_num = 0 # 本回合已经度过的时隙数目

        # self.completed_comp = [ 0 for _ in range(self.edge_queue_num)]
        # self.new_edge_comp = [ 0 for _ in range(self.edge_queue_num)]
        # self.virtual_edge_comp_ql_growth = [0 for i in range(self.edge_queue_num)]
        # device_comp_freqs = general_params.device_comp_freqs
        # comp_dly_cons = general_params.comp_dly_cons
        # for i in range(self.edge_queue_num):
        #     dz_mean = (data_size_inls[i][0]+data_size_inls[i][1])/2
        #     dens_mean = (comp_dens_inls[i][0]+comp_dens_inls[i][1])/2
        #     device_num = device_num_per_type[i]
        #     for j in general_params.device_in_types[i]:
        #         ideal_offl_rto = (dz_mean * dens_mean * task_arrival_prob[i]-self.delta * general_params.device_comp_freqs[i]) / (dz_mean * dens_mean * task_arrival_prob[i])
        #         self.virtual_edge_comp_ql_growth[i] += general_params.vir_edge_ql_growth_rate * \
        #                                     ideal_offl_rto * (dz_mean * dens_mean * task_arrival_prob[i] - self.delta*device_comp_freqs[i])
        #                                     self.alloc_edge_freq[i]/(self.alloc_edge_freq[i]+device_comp_freqs[i]*device_num) * \
        #                                     dz_mean * dens_mean * task_arrival_prob[i]
        self.total_comp_time = [0 for _ in range(self.edge_queue_num)]
        self.total_comp_amount = [0 for _ in range(self.edge_queue_num)]

        self.edge_act_queue_growth_rate = general_params.edge_act_queue_growth_rate
        self.edge_vir_queue_growth_rate = general_params.edge_vir_queue_growth_rate

    # 当一个时隙结束时更新各个计算队列的服务器计算频率
    def update_freq(self):
        edge_freq_weights = [(self.edge_weight_w * max(self.virtual_edge_queue_time_ql[i]-self.avg_edge_time[i] * self.edge_dly_adj_val[i], 0) + self.edge_weight_b) for i in range(self.edge_queue_num)]
        total_weight = sum(edge_freq_weights)

        self.alloc_edge_freq = [(self.edge_comp_freq / (2.0*self.edge_queue_num) + self.edge_comp_freq / 2.0 * edge_freq_weights[i]/total_weight) for i in range(self.edge_queue_num)]
        # print(f"[DEBUG] alloc_edge_freq: {self.alloc_edge_freq}")

    def reset(self):
        # reset computation-queue length
        # self.edge_queue_comp_ql = [ 0 for _ in range(self.edge_queue_num)]
        # self.old_edge_queue_comp_ql = [ 0 for _ in range(self.edge_queue_num)]
        # self.virtual_edge_queue_comp_ql = [ 0 for _ in range(self.edge_queue_num)]
        # self.old_virtual_edge_queue_comp_ql = [ 0 for _ in range(self.edge_queue_num)]
        # self.completed_comp = [ 0 for _ in range(self.edge_queue_num)]
        # self.new_edge_comp = [ 0 for _ in range(self.edge_queue_num)]
        self.alloc_edge_freq = [self.edge_comp_freq / self.edge_queue_num for _ in range(self.edge_queue_num)]
        # print(f"[DEBUG] alloc_edge_freq: {self.alloc_edge_freq}")
        self.total_comp_time = [0 for _ in range(self.edge_queue_num)]
        self.total_comp_amount = [0 for _ in range(self.edge_queue_num)]
        # reset computation time queue length
        self.edge_queue_time_ql = [ 0 for _ in range(self.edge_queue_num)]
        self.old_edge_queue_time_ql = [ 0 for _ in range(self.edge_queue_num)]
        self.virtual_edge_queue_time_ql = [ 0 for _ in range(self.edge_queue_num)]
        self.old_virtual_edge_queue_time_ql = [ 0 for _ in range(self.edge_queue_num)]
        self.avg_edge_time = [ 0 for _ in range(self.edge_queue_num)]
        self.new_edge_ql_change = [ 0 for _ in range(self.edge_queue_num)]
        self.new_vir_edge_ql_change = [ 0 for _ in range(self.edge_queue_num)]
        self.delta_num = 0
        # obs
        obs = self.get_obs()
        return obs
        
    def get_obs(self):

        alloc_edge_freq = self.alloc_edge_freq
        # edge_queue_comp_ql = self.edge_queue_comp_ql
        edge_queue_time_ql = self.edge_queue_time_ql
        obs = []
        obs += alloc_edge_freq
        obs += edge_queue_time_ql
        # for i in range(self.edge_queue_num):
        #     print(f"[DEBUG] The time queue length of edge {i} is {edge_queue_time_ql[i]}")
        # print(f"[DEBUG] The observation of edge is {obs}")
        return obs
    
    def compute(self, device_sched_tasks, e_id, t_id, visualize=False):
        writer = self.writer
        enable_print = gp.settings.enable_print
        device_type_sched_tasks = [[] for _ in range(self.edge_queue_num)]
        for device_id, sched_tasks in enumerate(device_sched_tasks):
            device_type = self.device_types_ref[device_id]
            device_type_sched_tasks[device_type] += sched_tasks
        # self.completed_comp = [ 0 for _ in range(self.edge_queue_num)]
        # self.new_edge_comp = [ 0 for _ in range(self.edge_queue_num)]

        #! 处理任务的顺序为FIFO，并分任务类型进行处理
        device_type_sched_tasks = [
            sorted(sub_list, key=lambda x: x.trans_time)
            for sub_list in device_type_sched_tasks
        ]

        # for type_idx, sub_list in enumerate(device_type_sched_tasks):
        #     print(f"Task type {type_idx}:")
        #     for task in sub_list:
        #         # assume each task has attributes 'device_id' and 'trans_time'
        #         print(f"  device_id={task.device_id}, trans_time={task.trans_time}")
        #     print("-" * 40)

        # comp_dly = self.comp_ql / self.edge_comp_freq
        # self.comp_ql = max(0, self.comp_ql - self.edge_comp_freq * self.delta)

        for i in range(self.edge_queue_num):
            # self.old_edge_queue_comp_ql[i] = self.edge_queue_comp_ql[i]
            self.old_edge_queue_time_ql[i] = self.edge_queue_time_ql[i]
            # if(enable_print): print(f"[DEBUG] Before compute, edge time_ql of device_type: {i} is {self.edge_queue_time_ql[i]}")
        alloc_edge_freq = self.alloc_edge_freq
        total_comp_used_time_this_epi = [min(self.delta, self.edge_queue_time_ql[i]) for i in range(self.edge_queue_num)]
        comp_dlys = [min(self.delta, self.edge_queue_time_ql[i]) for i in range(self.edge_queue_num)]
        # for i in range(self.edge_queue_num):
        #     if(enable_print): print(f"[DEBUG] The edge calculated comp for old tasks of device_type: {i} is {min(self.edge_queue_comp_ql[i], alloc_edge_freq[i] * self.delta)}")
        # self.completed_comp = [
        #         min(self.edge_queue_comp_ql[i], alloc_edge_freq[i] * self.delta)
        #         for i in range(self.edge_queue_num)
        #     ]
        # self.edge_queue_comp_ql = [
        #         max(0, self.edge_queue_comp_ql[i] - alloc_edge_freq[i] * self.delta)
        #         for i in range(self.edge_queue_num)
        #     ]
        self.delta_num += 1
        for device_type in range(self.edge_queue_num):
            if(enable_print): print(f"[DEBUG] We are processing the tasks of type {device_type}")
            # new_edge_comp = 0
            total_comp_need_time_this_epi = 0
            
            for i,task in enumerate(device_type_sched_tasks[device_type]):
                if task.trans_time == 0:
                    # if(enable_print): print(f"[DEBUG] The edge comp of task of type {device_type} in device {task.device_id} is 0")
                    # if(enable_print): print(f"[DEBUG] The edge calculated amount of type {device_type} in this episode is {alloc_edge_freq[device_type] * self.delta}")
                    task.e_comp_dly = 0
                    task.e_queue_dly = 0
                    task.e_proc_dly = 0
                    task.edge_comp_engy = 0
                    if(enable_print): print(f"[DEBUG] the edge comp_dly in device {task.device_id} is {task.e_comp_dly}")
                else:
                    # new_edge_comp += task.offl_dz * task.comp_dens
                    if(enable_print): print(f"[DEBUG] The edge comp of task in device {task.device_id} is {task.offl_dz * task.comp_dens}")
                    if(enable_print): print(f"[DEBUG] The edge calculated amount in device {task.device_id} is {min(task.offl_dz * task.comp_dens, alloc_edge_freq[device_type] * max(0, self.delta - max(comp_dlys[device_type], task.trans_time)))}")
                    
                    #config
                    # 能耗系数为1 J/GFlops
                    task.edge_comp_engy = task.offl_dz * task.comp_dens * pow(alloc_edge_freq[device_type],2)
                    # if(enable_print): print(f"[DEBUG] The edge freq pow2 is {pow(alloc_edge_freq[device_type],2)}")
                    task.e_comp_dly = max(max(self.total_comp_time[device_type] - (t_id-1)*self.delta, 0), task.trans_time) + task.offl_dz * \
                                    task.comp_dens / alloc_edge_freq[device_type]
                    task.e_queue_dly = max(self.total_comp_time[device_type] - (t_id-1)*self.delta, 0)
                    task.e_proc_dly = task.offl_dz * task.comp_dens / alloc_edge_freq[device_type]
                    total_comp_need_time_this_epi += task.offl_dz * task.comp_dens / alloc_edge_freq[device_type]
                    total_comp_used_time_this_epi[device_type] += min(task.offl_dz * task.comp_dens/alloc_edge_freq[device_type], max(0, self.delta-max(task.trans_time, comp_dlys[device_type])))
                    if(enable_print): print(f"[DEBUG] the edge comp_dly in device {task.device_id} is {task.e_comp_dly}")
                    # self.edge_queue_comp_ql[device_type] += max(0, task.offl_dz * task.comp_dens - 
                    #                     alloc_edge_freq[device_type] * max(0, self.delta -
                    #                                             max(comp_dlys[device_type], task.trans_time)))

                    # self.completed_comp[device_type] += min(task.offl_dz * task.comp_dens, alloc_edge_freq[device_type] * max(0, self.delta -
                    #                                             max(comp_dlys[device_type], task.trans_time)))
                    comp_dlys[device_type] = max(comp_dlys[device_type], task.trans_time) + task.offl_dz * task.comp_dens / alloc_edge_freq[device_type]
                    self.total_comp_time[device_type] += task.offl_dz * task.comp_dens / alloc_edge_freq[device_type]
                    self.total_comp_amount[device_type] += task.offl_dz * task.comp_dens
                    
            old_edge_queue_time_ql_ = self.avg_edge_time[device_type]
            self.avg_edge_time[device_type] = (self.avg_edge_time[device_type] * (self.delta_num-1) + total_comp_need_time_this_epi) / self.delta_num
            edge_act_queue_growth_rate = self.edge_act_queue_growth_rate
            self.edge_queue_time_ql[device_type] = max(self.edge_queue_time_ql[device_type] + edge_act_queue_growth_rate * (total_comp_need_time_this_epi - total_comp_used_time_this_epi[device_type]), 0)
            
            self.new_edge_ql_change[device_type] = self.edge_queue_time_ql[device_type] - old_edge_queue_time_ql_
            self.act_backlog = self.edge_queue_time_ql[device_type] - old_edge_queue_time_ql_
            # self.edge_queue_time_ql[device_type] = max(self.edge_queue_time_ql[device_type] + total_comp_need_time_this_epi - total_comp_used_time_this_epi[device_type], 0)
            # self.new_edge_comp[device_type] = new_edge_comp
            # self.old_virtual_edge_queue_comp_ql = self.virtual_edge_queue_comp_ql
            # self.virtual_edge_queue_comp_ql[device_type] = max(0, self.virtual_edge_queue_comp_ql[device_type] + self.virtual_edge_comp_ql_growth[device_type] - self.completed_comp[device_type])
            self.old_virtual_edge_queue_time_ql[device_type] = self.virtual_edge_queue_time_ql[device_type]
            
            edge_vir_queue_growth_rate = self.edge_vir_queue_growth_rate
            old_virtual_edge_queue_time_ql_ = self.virtual_edge_queue_time_ql[device_type]
            self.virtual_edge_queue_time_ql[device_type] = max(self.virtual_edge_queue_time_ql[device_type] + edge_vir_queue_growth_rate*(self.edge_queue_time_ql[device_type]/self.avg_edge_time[device_type] - self.edge_dly_adj_val[device_type]), 0)
            self.new_vir_edge_ql_change[device_type] = self.virtual_edge_queue_time_ql[device_type] - old_virtual_edge_queue_time_ql_
            self.vir_backlog = self.edge_queue_time_ql[device_type]/self.avg_edge_time[device_type]
            
            # print(f"[DEBUG] The edge_queue", device_type, "'s old_virtual_edge_queue_time_ql is: ", self.old_virtual_edge_queue_time_ql[device_type])
            # print(f"[DEBUG] The edge_queue", device_type, "'s new_vir_edge_ql_change is: ", self.new_vir_edge_ql_change[device_type])
            # print(f"[DEBUG] The edge_queue", device_type, "'s virtual_edge_queue_time_ql is: ", self.virtual_edge_queue_time_ql[device_type])
            
            # print(f"[DEBUG] The edge_queue", device_type, "'s avg_edge_time is: ", self.avg_edge_time[device_type])
            # print(f"[DEBUG] The edge_queue", device_type, "'s actual task dwell time is: ", self.edge_queue_time_ql[device_type]/self.avg_edge_time[device_type])
            # print(f"[DEBUG] The edge_queue", device_type, "'s edge_dly_adj_val is: ", self.edge_dly_adj_val[device_type])

            if(enable_print): print(f"[DEBUG] The edge_queue", device_type, "'s old_edge_queue_time_ql is: ", self.old_edge_queue_time_ql[device_type])
            if(enable_print): print(f"[DEBUG] The edge_queue", device_type, "'s old_virtual_edge_queue_time_ql is: ", self.old_virtual_edge_queue_time_ql[device_type])
            if(enable_print): print(f"[DEBUG] The edge_queue", device_type, "'s new_edge_ql_change is: ", self.new_edge_ql_change[device_type])
            if(enable_print): print(f"[DEBUG] The edge_queue", device_type, "'s new_vir_edge_ql_change is: ", self.new_vir_edge_ql_change[device_type])
            if(enable_print): print(f"[DEBUG] The edge_queue", device_type, "'s edge_queue_time_ql is: ", self.edge_queue_time_ql[device_type])
            if(enable_print): print(f"[DEBUG] The edge_queue", device_type, "'s virtual_edge_queue_time_ql is: ", self.virtual_edge_queue_time_ql[device_type])
            
            if visualize:
                writer.add_scalars(
                    f"detail/edge_avg_local_time_{device_type}",
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
                    {f"ep_{e_id}_act_backlog": self.act_backlog},
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
                writer.add_scalars(
                    f"detail/edge_time_ql_{device_type}",
                    {f"ep_{e_id}_vir_backlog": self.vir_backlog},
                    t_id
                )

                writer.add_scalars(
                    f"detail/total_comp_{device_type}",
                    {f"ep_{e_id}_time": self.total_comp_time[device_type]},
                    t_id
                )

                writer.add_scalars(
                    f"detail/total_comp_{device_type}",
                    {f"ep_{e_id}_amount": self.total_comp_amount[device_type]},
                    t_id
                )

                writer.add_scalars(
                    f"detail/alloc_edge_freq_{device_type}",
                    {f"ep_{e_id}_freq": alloc_edge_freq[device_type]},
                    t_id
                )

        # 不启用服务器动态变化算法了
        # self.update_freq()
        # for i in range(self.edge_queue_num):
        #     if(enable_print): print(f"[DEBUG] After compute, edge time_ql of device_type: {i} is {self.edge_queue_time_ql[i]}")