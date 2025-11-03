class EdgeEnv():
    def __init__(self, general_params):
        # unit: s
        self.delta = general_params.delta
        # unit: Gcycles/s
        self.edge_comp_freq = general_params.edge_comp_freq
        
        # unit: Gcycles
        self.comp_ql = None

        self.device_num = general_params.device_num
        self.device_comp_qls = []
        self.device_sched_tasks = []
        self.device_sched_
        for i in range(self.device_num):
            self.device_comp_qls.append(0)
            self.device_sched_tasks.append([])

    def reset(self):
        # reset computation-queue length
        self.comp_ql = 0
        
        # obs
        obs = self.get_obs()
        
        return obs
        
    def get_obs(self):
        obs = [self.comp_ql]
        
        return obs
    
    def compute(self, new_sched_tasks, isPrint):
        available_ql_num = 0
        device_enable = [0 for i in range(self.device_num)]
        for i in range(self.device_num):
            for task in self.device_sched_tasks[i]:
                if task.offl_dz > 0:
                    device_enable[i] = 1
                    break
            for task in new_sched_tasks[i]:
                if task.offl_dz > 0:
                    self.device_sched_tasks[i].append(task)
                    device_enable[i] = 1
                    break
            available_ql_num += device_enable[i]
            
        if available_ql_num == 0:
            return
        
        comp_freq_mean = self.edge_comp_freq / available_ql_num
        #每个设备各自有一个计算队列，按照FIFO顺序计算
        for i in range(self.device_num):
            if device_enable[i] == 0:
                continue
            comp_dly = 0
            if len(self.device_sched_tasks[i]) > 0:
                suf_process = False
                for task in self.device_sched_tasks[i][:]:
                    if task.offl_dz == 0:
                        task.e_comp_dly = 0
                        continue
                    task_comp = task.offl_dz * task.comp_dens
                    if ((self.delay - max(comp_dly, task.edge_trans_time) * comp_freq_mean)) >= task_comp:
                        task.e_comp_dly += max(task.edge_trans_time, comp_dly) + \
                            task.offl_dz * task.comp_dens / comp_freq_mean
                        task.edge_trans_time = 0
                        task.offl_dz = 0
                        self.device_sched_tasks[i].pop(0)
                        
                    elif suf_process == False:
                        task.e_comp_dly += self.delay
                        task.edge_trans_time = 0
                        task.offl_dz -= (self.delay - max(comp_dly, task.edge_trans_time) * comp_freq_mean) / task.comp_dens
                        suf_process = True
                    else:
                        task.e_comp_dly += self.delay
                        task.edge_trans_time = 0
                        break
            self.device_comp_qls[i] = max(0, self.device_comp_qls[i] - comp_freq_mean * self.delta)