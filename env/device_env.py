import copy
import math
import numpy as np
import config.global_params as gp
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
    def __init__(self, env_id, gen_params, edge_env):

        self.edge_env = edge_env
        # env id
        self.env_id = env_id
        self.device_type = gen_params.device_types[env_id]
        self.device_type_num = gen_params.device_type_num
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
        # unit: Gcycles/Mb
        self.comp_dens_inl = gen_params.comp_dens_inls[self.device_type]
        # unit: $/Gcycles
        self.service_price = gen_params.service_price
        
        self.max_task_num = gen_params.max_task_num

        self.task_timeout_thre = gen_params.comp_dly_cons[self.device_type]

        self.task_num = 0
        # unit: Gcycles
        self.comp_ql = 0
        self.old_comp_ql = 0
        self.virtual_comp_ql = 0
        self.old_virtual_comp_ql = 0
        self.completed_comp = 0
        self.new_local_comp = 0
        # unit: Mb
        self.trans_ql = 0
        data_size_mean = (self.data_size_inl[0] + self.data_size_inl[1]) / 2
        comp_dens_mean = (self.comp_dens_inl[0] + self.comp_dens_inl[1]) / 2
        task_arrival_prob = gen_params.task_arrival_prob[self.device_type]
        comp_dly_cons = gen_params.comp_dly_cons[self.device_type]
        device_num_this_type = gen_params.device_num_per_type[self.device_type]
        ideal_offl_rto = (data_size_mean * comp_dens_mean * task_arrival_prob-self.delta * self.device_comp_freq) / (data_size_mean * comp_dens_mean * task_arrival_prob)
        self.virtual_comp_ql_growth = gen_params.vir_local_ql_growth_rate * \
                                    (1-ideal_offl_rto) * min(data_size_mean * comp_dens_mean * task_arrival_prob, self.delta * self.device_comp_freq)
                                    # self.device_comp_freq*device_num_this_type/(self.edge_env.alloc_edge_freq[self.device_type]+self.device_comp_freq*device_num_this_type) * \
                                    # data_size_mean * comp_dens_mean * task_arrival_prob
        self.sched_tasks = []
        #! 动态时间阈值调整
        self.dynamic_delay_adjust_coef = 1.0
        #TODO 后续可以考虑对动态时间阈值对超时程度的影响做进一步研究
    
    def reset(self):
        # reset computation-queue length
        self.comp_ql = 0
        self.old_comp_ql = 0
        self.virtual_comp_ql = 0

        # reset transmission-queue length
        self.trans_ql = 0

        # reset channel gain
        self.channel_gain = self.path_loss * np.random.exponential(1)
        
        # reset scheduling tasks
        self.sched_tasks.clear()

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
            
            task = Task(data_size, comp_dens, self.env_id)
            
            comp = data_size * comp_dens
            # task.dly_cons = comp / self.std_comp_freq * self.dynamic_delay_adjust_coef
            # 超时阈值先设置为10个episode
            task.dly_cons = self.task_timeout_thre
            task.norm_csum_engy = comp * self.engy_fac
            task.norm_comp_expn = comp * self.service_price
            # print("[DEBUG] The norm_csum_engy is: ", task.norm_csum_engy)
            # print("[DEBUG] The norm_comp_expn is: ", task.norm_comp_expn)
            self.sched_tasks.append(task)
        
        # obs
        obs = self.get_obs()
        
        return obs

    # 只考虑有一个任务的情况
    def get_obs(self):
        device_type = [1 if i == self.device_type else 0 for i in range(self.device_type_num)] #onehot编码
        comp_ql = self.comp_ql
        comp_freq = self.device_comp_freq
        task_arrival_prob = self.task_arrival_prob
        trans_rate = self.trans_rate
        trans_ql = self.trans_ql
        task_msgs = []
        has_task = 1
        if self.task_num == 0:
            has_task = 0
            task_msgs = [0,0,0]
        else: 
            for i in range(self.task_num):
                data_size = self.sched_tasks[i].data_size
                comp_dens = self.sched_tasks[i].comp_dens
                dly_cons = self.sched_tasks[i].dly_cons
                task_msgs += [data_size, comp_dens, dly_cons]
        obs = []
        obs += device_type
        obs += [comp_freq, trans_rate, comp_ql, trans_ql, task_arrival_prob, has_task] + task_msgs
        
        return obs

    #! 目前每个时隙至多只传输一个任务的数据
    # act: [offl_rto, trans_rto, local_comp_rto], all in [0, 1]
    def compute(self, act):
        enable_print = gp.settings.enable_print
        '''offloading'''
        # offloading data-size
        offl_dzs = []
        trans_power = self.trans_power
        device_comp_freq = self.device_comp_freq
        if(enable_print): print(f"[DEBUG] The action of device {self.env_id} is {act}")
        if self.task_num>=1:
            # offloading ratio
            offl_rto = act[0]
            offl_dz = self.sched_tasks[0].data_size * offl_rto
            offl_dzs.append(offl_dz)
            # transmission-power ratio
            trpw_rto = act[1]
            trans_power = self.trans_power * trpw_rto
            # local compute ratio
            device_comp_rto = act[2]
            device_comp_freq = self.device_comp_freq * device_comp_rto
        #! 处理的任务的顺序为FIFO
        # ascending order 
        # offl_dzs = sorted(offl_dzs.items(), key = lambda x: x[1])
        # trans_power = self.trans_power # 传输功耗是满的
        # unit: Mb/s
        # print(f"[DEBUG] the full ratio trans_rate is {self.bandwidth * math.log(1 + self.trans_power * self.channel_gain / self.noise_power, 2) * pow(10, -6)}")
        # print(f"[DEBUG] bandwidth: {self.bandwidth}, trans_power: {self.trans_power}, channel_gain: {self.channel_gain}, noise_power: {self.noise_power}")
        trans_rate = self.bandwidth * math.log(1 + trans_power * self.channel_gain / 
                                               self.noise_power, 2) * pow(10, -6)
        delta_trans_dz = trans_rate * self.delta
        total_offl_dz = 0
        total_offl_comp = 0
        total_trans_dz = self.trans_ql
        # 卸载的部分
        local_comps = []
        for task_id, offl_dz in enumerate(offl_dzs):
            # offl_dz = min(offl_dz, total_trans_dz)
            # total_trans_dz -= offl_dz
            total_offl_dz += offl_dz
            trans_before_delay = total_trans_dz / trans_rate
            total_trans_dz += offl_dz
            task = self.sched_tasks[task_id]
            task.offl_dz = offl_dz
            total_offl_comp += offl_dz * task.comp_dens
            # if offl_dz = 0, there is no need to queue
            if task.offl_dz == 0:
                task.trans_time = 0
                task.e_csum_engy = 0
                task.comp_expn = 0
            else:
                task.trans_time = trans_before_delay + total_offl_dz / trans_rate
                task.e_csum_engy = trans_power * pow(10, -3) * \
                                   task.offl_dz / trans_rate
                task.comp_expn = task.offl_dz * task.comp_dens * \
                                 self.service_price
            local_comps.append((task.data_size - task.offl_dz) * \
                                    task.comp_dens)
            if(enable_print): print(f"[DEBUG] the local_comp in device {self.env_id} is {(task.data_size - task.offl_dz) * task.comp_dens}")
            if(enable_print): print(f"[DEBUG] the offl_comp in device {self.env_id} is {task.offl_dz * task.comp_dens}")

        # 每个时隙，无论有没有新任务到达，都改变传输队列长度
        self.trans_ql = max(0, total_trans_dz - delta_trans_dz)
        '''local computing'''
        # 以FIFO的顺序进行
        if(enable_print): print(f"[DEBUG] Before compute, the comp_ql in device {self.env_id} is {self.comp_ql}")
        self.old_comp_ql = self.comp_ql
        total_local_comp = self.comp_ql #计算队列剩余的计算量
        new_local_comp = 0
        for task_id, local_comp in enumerate(local_comps):
            task = self.sched_tasks[task_id]
            total_local_comp += local_comp
            new_local_comp += local_comp
            # if(enable_print): print(f"[DEBUG] The local comp of task {task_id} in device {self.env_id} is {local_comp}")
            # if local_comp = 0, there is no need to queue
            if local_comp == 0:
                task.l_comp_dly = 0
                task.l_csum_engy = 0
            else:
                task.l_comp_dly = total_local_comp / device_comp_freq
                task.l_csum_engy = self.engy_fac * local_comp
            if(enable_print): print(f"[DEBUG] the local comp_dly in device {self.env_id} is {task.l_comp_dly}")
            # if isPrint:
            #     print("[DEBUG] The action of ", self.env_id, " is:", act)
            #     print("[DEBUG] the actual previous local compute amount of ", self.env_id," is", total_local_comp)
            #     print("[DEBUG] the actual local compute freq of ", self.env_id," is", device_comp_freq)
            #     print("[DEBUG] the actual local compute delay of ", self.env_id," is", task.l_comp_dly)
        
        #! 后面再改队列相关的部分
        # update computation-queue length
        self.completed_comp = min(device_comp_freq * self.delta, total_local_comp)
        self.new_local_comp = new_local_comp
        self.comp_ql = max(0, total_local_comp - device_comp_freq * self.delta)
        # if(enable_print): print(f"[DEBUG] The calculated amount in device {self.env_id} in this episode is {device_comp_freq * self.delta}")
        if(enable_print): print(f"[DEBUG] After compute, the comp_ql in device {self.env_id} is {self.comp_ql}")
        # print("[DEBUG] The virtual comp qs growth is: ", self.virtual_comp_ql_growth)
        # print("[DEBUG] The completed comp is: ", self.completed_comp)
        self.old_virtual_comp_ql = self.virtual_comp_ql
        self.virtual_comp_ql = max(0, self.virtual_comp_ql - self.completed_comp + self.virtual_comp_ql_growth)

        # update channel gain
        self.channel_gain = self.path_loss * np.random.exponential(1)
        
        # 随机生成新任务
        # update scheduling tasks
        sched_tasks = copy.copy(self.sched_tasks)
        self.sched_tasks.clear()
        self.task_num = np.random.binomial(1, self.task_arrival_prob)
        for i in range(self.task_num):
            # unit: Mb
            data_size = np.random.uniform(self.data_size_inl[0],
                                          self.data_size_inl[1])
            # unit: Gcycles/Mb
            comp_dens = np.random.uniform(self.comp_dens_inl[0],
                                          self.comp_dens_inl[1])
            
            task = Task(data_size, comp_dens, self.env_id)
            
            comp = data_size * comp_dens
            # task.dly_cons = comp / self.std_comp_freq * self.dynamic_delay_adjust_coef
            task.dly_cons = self.task_timeout_thre
            task.norm_csum_engy = comp * self.engy_fac
            task.norm_comp_expn = comp * self.service_price
            
            self.sched_tasks.append(task)
        
        return sched_tasks

    # def adjust_delay_threshold_coef(self, overtime_coef):
    #     exp_arg = min(overtime_coef - 1.5, 10.0)# 限制指数参数范围
    #     self.dynamic_delay_adjust_coef = max(4.0, self.dynamic_delay_adjust_coef + 0.01 * exp_arg * exp_arg)