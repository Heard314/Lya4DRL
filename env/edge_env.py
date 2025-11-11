import config.global_params as gp
class EdgeEnv():
    def __init__(self, general_params):
        # unit: s
        self.delta = general_params.delta
        
        self.device_type_num = general_params.device_type_num
        self.device_types_ref = general_params.device_types
        # unit: Gcycles/s
        self.edge_comp_freq = general_params.edge_comp_freq
        
        # 根据各个任务类型的计算量与到达频率分配服务器的计算频率
        edge_freq_weights = []
        data_size_inls = general_params.data_size_inls
        comp_dens_inls = general_params.comp_dens_inls
        task_arrival_prob = general_params.task_arrival_prob
        device_num_per_type = general_params.device_num_per_type
        for i in range(general_params.device_type_num):
            dz_mean = (data_size_inls[i][0]+data_size_inls[i][1])/2
            dens_mean = (comp_dens_inls[i][0]+comp_dens_inls[i][1])/2
            device_num = device_num_per_type[i]
            edge_freq_weights.append(dz_mean*dens_mean*task_arrival_prob[i]*device_num)

        total_weight = sum(edge_freq_weights)
        self.alloc_edge_freq = [w/total_weight*self.edge_comp_freq for w in edge_freq_weights]
        # if(enable_print): print(f"[DEBUG] alloc_edge_freq: {self.alloc_edge_freq}")
        # unit: Gcycles
        self.device_type_comp_ql = [ 0 for _ in range(self.device_type_num)]
        
        
    def reset(self):
        # reset computation-queue length
        self.device_type_comp_ql = [ 0 for _ in range(self.device_type_num)]
        # obs
        obs = self.get_obs()
        return obs
        
    def get_obs(self):

        alloc_edge_freq = self.alloc_edge_freq
        device_type_comp_ql = self.device_type_comp_ql
        obs = []
        obs += alloc_edge_freq
        obs += device_type_comp_ql

        return obs
    
    def compute(self, device_sched_tasks):
        enable_print = gp.settings.enable_print
        device_type_sched_tasks = [[] for _ in range(self.device_type_num)]
        for device_id, sched_tasks in enumerate(device_sched_tasks):
            device_type = self.device_types_ref[device_id]
            device_type_sched_tasks[device_type] += sched_tasks

        #! 处理任务的顺序为FIFO，并分任务类型进行处理
        device_type_sched_tasks = [
            sorted(sub_list, key=lambda x: x.trans_time)
            for sub_list in device_type_sched_tasks
        ]
        # comp_dly = self.comp_ql / self.edge_comp_freq
        # self.comp_ql = max(0, self.comp_ql - self.edge_comp_freq * self.delta)
        alloc_edge_freq = self.alloc_edge_freq
        comp_dlys = [self.device_type_comp_ql[i]/alloc_edge_freq[i] for i in range(self.device_type_num)]
        self.device_type_comp_ql = [
                max(0, self.device_type_comp_ql[i] - alloc_edge_freq[i] * self.delta)
                for i in range(self.device_type_num)
            ]
        for device_type in range(self.device_type_num):
            if(enable_print): print(f"[DEBUG] Before compute, edge comp_ql of device_type: {device_type} is {self.device_type_comp_ql[device_type]}")

            for i,task in enumerate(device_type_sched_tasks[device_type]):
                if task.trans_time == 0:
                    # if(enable_print): print(f"[DEBUG] The edge comp of task of type {device_type} in device {task.device_id} is 0")
                    # if(enable_print): print(f"[DEBUG] The edge calculated amount of type {device_type} in this episode is {alloc_edge_freq[device_type] * self.delta}")
                    task.e_comp_dly = 0
                    if(enable_print): print(f"[DEBUG] the edge comp_dly in device {task.device_id} is {task.e_comp_dly}")
                    self.device_type_comp_ql[device_type] = max(0, self.device_type_comp_ql[device_type] - alloc_edge_freq[device_type] * self.delta)
                else:
                    # if(enable_print): print(f"[DEBUG] The edge comp of task of type {device_type} in device {task.device_id} is {task.offl_dz * task.comp_dens}")
                    # if(enable_print): print(f"[DEBUG] The edge calculated amount of type {device_type} in this episode is {min(task.offl_dz * task.comp_dens, alloc_edge_freq[device_type] * max(0, self.delta - max(comp_dlys[device_type], task.trans_time)))}")
                    task.e_comp_dly = max(comp_dlys[device_type], task.trans_time) + task.offl_dz * \
                                    task.comp_dens / alloc_edge_freq[device_type]
                    if(enable_print): print(f"[DEBUG] the edge comp_dly in device {task.device_id} is {task.e_comp_dly}")
                    self.device_type_comp_ql[device_type] += max(0, task.offl_dz * task.comp_dens - 
                                        alloc_edge_freq[device_type] * max(0, self.delta -
                                                                max(comp_dlys[device_type], task.trans_time)))
                    comp_dlys[device_type] = task.e_comp_dly
            if(enable_print): print(f"[DEBUG] After compute, edge comp_ql of device_type: {device_type} is {self.device_type_comp_ql[device_type]}")