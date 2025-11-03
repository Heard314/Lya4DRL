import copy
import math
import numpy as np

class Task():
    def __init__(self, data_size, comp_dens, device_id):
        '''attributes'''
        # The device id for publishing the task
        self.device_id = device_id
        # unit: Mb
        self.data_size = data_size
        # unit: Gcycles/Mb
        self.comp_dens = comp_dens
        # unit: s
        self.dly_cons = None
        '''local subtask'''
        # computing delay
        self.l_comp_dly = None
        # energy consumption
        self.l_csum_engy = None
        '''offloading subtask'''
        # offloading data-size
        self.offl_dz = None
        # transmission time
        self.trans_time = None
        # computing delay 
        self.e_comp_dly = None
        # consumed energy
        self.e_csum_engy = None
        # service expense
        self.comp_expn = None
        '''normalization'''
        self.norm_csum_engy = None
        self.norm_comp_expn = None

        # 为了方便处理任务等待延时中的传输时间部分
        self.edge_trans_time = None

    def __str__(self):
        return "data_size: " + str(self.data_size) + \
               "\ncomp_dens: " + str(self.comp_dens) + \
               "\ndly_cons: " + str(self.dly_cons) + \
               "\nl_comp_dly: " + str(self.l_comp_dly) + \
               "\nl_csum_engy: " + str(self.l_csum_engy) + \
               "\noffl_data_size: " + str(self.offl_dz) + \
               "\ntrans_time: " + str(self.trans_time) + \
               "\ne_comp_dly: " + str(self.e_comp_dly) + \
               "\ne_csum_engy: " + str(self.e_csum_engy) + \
               "\ncomp_expn: " + str(self.comp_expn) + \
               "\nnorm_csum_engy: " + str(self.norm_csum_engy) + \
               "\nnorm_comp_expn: " + str(self.norm_comp_expn)

class DeviceEnv():
    def __init__(self, env_id, gen_params):
        # env id
        self.env_id = env_id
        self.device_type = gen_params.device_types[env_id]
        # unit: s
        self.delta = gen_params.delta
        self.task_arrival_prob = gen_params.task_arrival_prob[self.device_type]
        # unit: Hz
        self.bandwidth = gen_params.total_bandwidth / gen_params.device_num
        # unit: mW
        self.trans_power = gen_params.device_trans_powers[self.device_type]
        self.path_loss = gen_params.device_path_loss[self.device_type]
        self.channel_gain = None
        # unit: mW
        self.noise_power = gen_params.spec_dens * self.bandwidth
        # unit: Mb/s
        self.trans_rate = None
        # unit: Gcycles/s
        self.device_comp_freq = gen_params.device_comp_freqs[self.device_type]
        # unit: Gcycles/s
        self.std_comp_freq = gen_params.std_comp_freq
        # unit: J/Gcycles
        self.engy_fac = gen_params.device_engy_facs[self.device_type]
        # unit: Mb
        self.data_size_inl = gen_params.data_size_inls[self.device_type]
        # unit: cycles/Mb
        self.comp_dens_inl = gen_params.comp_dens_inls[self.device_type]
        # unit: $/Gcycles
        self.service_price = gen_params.service_price
        
        self.max_task_num = gen_params.max_task_num

        self.trans_ql = []

        # unit: Gcycles
        self.comp_ql = 0
        self.virtual_comp_ql = 0
        # unit: Mb
        data_size_mean = (self.data_size_inl[0] + self.data_size_inl[1]) / 2 
        # unit: Gcycles/Mb
        comp_dens_mean = (self.comp_dens_inl[0] + self.comp_dens_inl[1]) / 2 
        self.virtual_comp_ql_growth = gen_params.vir_comp_ql_growth_rate * data_size_mean * comp_dens_mean
        self.sched_tasks = []
        #! 动态时间阈值调整
        self.dynamic_delay_adjust_coef = 1.0
    
    def reset(self):
        # reset computation-queue length
        self.comp_ql = 0
        self.virtual_comp_ql = 0

        # reset channel gain
        self.channel_gain = self.path_loss * np.random.exponential(1)
        
        # reset scheduling tasks
        self.sched_tasks.clear()

        # reset transmission-queue
        self.trans_ql.clear()

        #! 重置动态时间阈值
        self.dynamic_delay_adjust_coef = 1.0
        
        self.task_num = np.random.binomial(1, self.task_arrival_prob)
        for i in range(self.task_num):
            # unit: Mb
            data_size = np.random.uniform(self.data_size_inl[0],
                                          self.data_size_inl[1])
            # unit: Gcycles/Mb
            comp_dens = np.random.uniform(self.comp_dens_inl[0],
                                          self.comp_dens_inl[1])
            comp_dens = comp_dens
            
            task = Task(data_size, comp_dens, self.env_id)
            
            comp = data_size * comp_dens
            task.dly_cons = comp / self.std_comp_freq * self.dynamic_delay_adjust_coef
            task.norm_csum_engy = comp * self.engy_fac
            task.norm_comp_expn = comp * self.service_price
            # print("[DEBUG] The norm_csum_engy is: ", task.norm_csum_engy)
            # print("[DEBUG] The norm_comp_expn is: ", task.norm_comp_expn)
            self.sched_tasks.append(task)
        
        # obs
        obs = self.get_obs()
        
        return obs
    
    def get_obs(self):
        comp_ql = self.comp_ql
        cgnp_rto = self.channel_gain / self.noise_power
        task_msgs = []
        for i in range(self.task_num):
            data_size = self.sched_tasks[i].data_size
            comp_dens = self.sched_tasks[i].comp_dens
            dly_cons = self.sched_tasks[i].dly_cons
            task_msgs += [data_size, comp_dens, dly_cons]
        obs = [comp_ql, cgnp_rto] + task_msgs
        
        return obs

    # act: [offl_rto], offl_rto is in [0, 1] is in [0, 1]
    def compute(self, act, isPrint):
        '''offloading'''
        # offloading data-size
        offl_dzs = {}
        local_comps = {}
        for i in range(self.max_task_num):
            # offloading decision 0 or 1
            is_offl = act[i]
            if is_offl == 0:
                offl_dz = 0
                local_dz = self.sched_tasks[i].data_size
            else:
                offl_dz = self.sched_tasks[i].data_size
                local_dz = 0
                self.trans_ql.append(self.sched_tasks[i])
            offl_dzs[i] = offl_dz
            local_comps[i] = local_dz * self.sched_tasks[i].comp_dens
        
        #! 处理的任务的顺序为FIFO
        trans_power = self.trans_power
        # unit: Mb/s
        trans_rate = self.bandwidth * math.log(1 + trans_power * self.channel_gain / 
                                               self.noise_power, 2) * pow(10, -6)
        if isPrint:
            print("[DEBUG] The trans_rate is: ", trans_rate)
        total_trans_dz = trans_rate * self.delta
        total_offl_dz = 0
        total_offl_comp = 0

        prepared_trans_tasks = []
        # process the transmission-queue
        for i in len(self.trans_ql):
            task = self.trans_ql[i]
            if total_trans_dz >= task.offl_dz:
                total_trans_dz -= task.offl_dz
                total_offl_dz += task.offl_dz
                task.trans_time += total_offl_dz / trans_rate
                task.e_csum_engy += trans_power * pow(10, -3) * \
                    task.offl_dz / trans_rate
                task.comp_expn += task.offl_dz * task.comp_dens * \
                                 self.service_price
                total_offl_comp += task.comp_dens * task.offl_dz
                task.edge_trans_time = task.offl_dz / trans_rate
                task.offl_dz = 0
                self.trans_ql.pop(i)
                prepared_trans_tasks.append(task)
            else:
                total_offl_dz += total_trans_dz
                task.trans_time += total_offl_dz / trans_rate
                task.e_csum_engy += trans_power * pow(10, -3) * \
                    total_trans_dz / trans_rate
                task.comp_expn += total_trans_dz * task.comp_dens * \
                                 self.service_price
                task.offl_dz -= total_trans_dz
                total_offl_comp += task.comp_dens * total_trans_dz
                total_trans_dz = 0

        '''local computing'''
        # computation-frequency ratio
        total_local_comp = self.comp_ql
        for task_id, local_comp in local_comps:
            if local_comp == 0:
                continue
            task = self.sched_tasks[task_id]
            total_local_comp += local_comp
            
            task.l_comp_dly = total_local_comp / self.device_comp_freq
            task.l_csum_engy = self.engy_fac * local_comp
            if isPrint:
                print("[DEBUG] The action of ", self.env_id, " is:", act)
                print("[DEBUG] the actual previous local compute amount of ", self.env_id," is", total_local_comp)
                print("[DEBUG] the actual local compute freq of ", self.env_id," is", self.device_comp_freq)
                print("[DEBUG] the actual local compute delay of ", self.env_id," is", task.l_comp_dly)
        
        #! Lya相关代码，修改完再启用
        # update computation-queue length
        # self.completed_comp = total_offl_comp + self.device_comp_freq * self.delta
        self.comp_ql = max(0, total_local_comp - self.device_comp_freq * self.delta)
        # print("[DEBUG] The virtual comp qs growth is: ", self.virtual_comp_ql_growth)
        # print("[DEBUG] The completed comp is: ", self.completed_comp)
        # self.virtual_comp_ql = max(0, self.virtual_comp_ql - self.completed_comp + self.virtual_comp_ql_growth)

        # update channel gain
        self.channel_gain = self.path_loss * np.random.exponential(1)
        
        # 生成新一轮的任务
        # update scheduling tasks
        sched_tasks = copy.copy(self.sched_tasks)
        self.sched_tasks.clear()
        self.task_num = np.random.binomial(1, self.task_arrival_prob)
        for i in range(self.task_num):
            # unit: Mb
            data_size = np.random.uniform(self.data_size_inl[0],
                                          self.data_size_inl[1])
            data_size = data_size
            # unit: Gcycles/Mb
            comp_dens = np.random.uniform(self.comp_dens_inl[0],
                                          self.comp_dens_inl[1])
            comp_dens = comp_dens
            
            task = Task(data_size, comp_dens, self.env_id)
            
            comp = data_size * comp_dens
            task.dly_cons = comp / self.std_comp_freq * self.dynamic_delay_adjust_coef
            task.norm_csum_engy = comp * self.engy_fac
            task.norm_comp_expn = comp * self.service_price
            
            self.sched_tasks.append(task)
        
        # 暂时先返回已准备好传输完成的任务
        return prepared_trans_tasks

    def adjust_delay_threshold_coef(self, overtime_coef):
        exp_arg = min(overtime_coef - 1.5, 10.0)# 限制指数参数范围
        self.dynamic_delay_adjust_coef = max(4.0, self.dynamic_delay_adjust_coef + 0.01 * exp_arg * exp_arg)