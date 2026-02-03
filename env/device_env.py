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
        '''offloading subtask'''
        # offloading data-size
        self.offl_dz = None
        # transmission time
        self.trans_time = None
        # computing delay 
        self.e_comp_dly = None
        # energy consumption
        self.local_comp_engy = None
        # device consumed energy
        self.tran_engy = None
        # edge compute consumed energy
        self.edge_comp_engy = None
        '''normalization'''
        self.norm_csum_engy = None
        self.norm_esum_engy = None

        # Debug
        self.l_queue_dly = None
        self.e_queue_dly = None
        self.l_proc_dly = None
        self.e_proc_dly = None
    def __str__(self):
        return "data_size: " + str(self.data_size) + \
               "\ncomp_dens: " + str(self.comp_dens) + \
               "\ndly_cons: " + str(self.dly_cons) + \
               "\nl_comp_dly: " + str(self.l_comp_dly) + \
               "\nlocal_comp_engy: " + str(self.local_comp_engy) + \
               "\noffl_data_size: " + str(self.offl_dz) + \
               "\ntrans_time: " + str(self.trans_time) + \
               "\ne_comp_dly: " + str(self.e_comp_dly) + \
               "\ntran_engy: " + str(self.tran_engy) + \
               "\ncomp_expn: " + str(self.edge_comp_engy) + \
               "\nnorm_csum_engy: " + str(self.norm_csum_engy) + \
               "\nnorm_comp_expn: " + str(self.norm_esum_engy)

class DeviceEnv():
    
    def gen_position(self, min_dis, max_dis):
        while True:
            x = float(np.random.uniform(-max_dis, max_dis))
            y = float(np.random.uniform(-max_dis, max_dis))
            distance = math.sqrt(x**2 + y**2)
            
            if min_dis <= distance <= max_dis:
                return x, y

    def __init__(self, env_id, gen_params, edge_env, writer = None):

        # Summary Writer
        self.writer = writer
        # task generation cycle
        self.gen_task_cycle = gen_params.gen_task_cycle
        self.start_slot = gen_params.start_slot
        self.edge_env = edge_env
        # env id
        self.env_id = env_id

        self.device_type = gen_params.device_types[env_id]
        self.device_type_num = gen_params.device_type_num

        self.device_type = gen_params.device_types[env_id]
        self.device_type_num = gen_params.device_type_num

        self.enable_virtual_queue_reward = gen_params.enable_virtual_queue_reward
        self.enable_actual_queue_reward = gen_params.enable_actual_queue_reward
        # unit: s
        self.delta = gen_params.delta
        self.task_arrival_prob = gen_params.task_arrival_prob[self.device_type]
        # unit: Hz
        self.bandwidth = gen_params.total_bandwidth / gen_params.device_num
        # unit: mW
        self.trans_power = gen_params.device_trans_powers[self.device_type]
        # self.path_loss = gen_params.device_path_loss[self.device_type]
        self.channel_gain = None
        # unit: mW
        self.noise_power = gen_params.spec_dens * self.bandwidth
        # unit: Mb/s
        self.trans_rate = None
        self.max_trans_rate = None
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
        # # unit: $/Gcycles
        # self.service_price = gen_params.service_price
        
        self.max_task_num = gen_params.max_task_num

        self.task_timeout_thre = gen_params.comp_dly_thre[self.device_type] * self.delta

        self.task_num = 0

        self.speed_max = gen_params.speed_max
        self.speed_min = gen_params.speed_min

        self.direction_max = gen_params.direction_max
        self.direction_min = gen_params.direction_min

        self.accelerate_speed_max = gen_params.accelerate_speed_max
        self.accelerate_speed_min = gen_params.accelerate_speed_min

        self.accelerate_direction_max = gen_params.accelerate_direction_max
        self.accelerate_direction_min = gen_params.accelerate_direction_min

        self.max_distance_from_edge = gen_params.max_distance_from_edge
        self.min_distance_from_edge = gen_params.min_distance_from_edge

        # edge server position is (0,0)
        self.position_x, self.position_y = self.gen_position(self.min_distance_from_edge, self.max_distance_from_edge)
        self.position_z = 1.8

        self.speed_x = 0.0
        self.speed_y = 0.0

        #unit: s
        self.time_ql = 0
        self.old_time_ql = 0
        self.virtual_time_ql = 0
        self.old_virtual_time_ql = 0
        self.device_dly_adj_fac = gen_params.device_dly_adj_fac[self.device_type]
        self.device_dly_adj_val = self.device_dly_adj_fac * gen_params.comp_dly_thre[self.device_type] * self.delta
        self.avg_local_time = 0
        self.old_comp_times = []
        self.statSlotNum = gen_params.statSlotNum
        self.new_ql_change = 0
        self.new_vir_ql_change = 0

        # unit: Mb
        self.trans_ql = 0
        data_size_mean = (self.data_size_inl[0] + self.data_size_inl[1]) / 2
        comp_dens_mean = (self.comp_dens_inl[0] + self.comp_dens_inl[1]) / 2
        task_arrival_prob = gen_params.task_arrival_prob[self.device_type]
        self.avail_task_num = 0
        self.sched_tasks = []

        self.old_comp_queue_delay = 0.0
        self.old_tran_queue_delay = 0.0

        # To balance the comp_dly_thre role of different task type.
        self.device_act_reward_fac = 1.0 / self.delta / gen_params.comp_dly_thre[self.device_type]

        print(f"[DEBUG] The device {self.env_id} 's device_act_reward_fac is {self.device_act_reward_fac}")
        self.device_act_queue_growth_rate = gen_params.device_act_queue_growth_rate
        self.device_vir_queue_growth_rate = gen_params.device_vir_queue_growth_rate
    
    def reset(self):
        # reset computation-queue length
        self.avail_task_num = 0
        # reset computation time queue length
        self.time_ql = 0
        self.old_time_ql = 0
        self.avg_local_time = 0
        self.virtual_time_ql = 0
        self.old_virtual_time_ql = 0
        self.new_ql_change = 0
        self.new_vir_ql_change = 0

        self.old_comp_queue_delay = 0.0
        self.old_tran_queue_delay = 0.0

        self.act_backlog = 0.0
        self.vir_backlog = 0.0

        # reset transmission-queue length
        self.trans_ql = 0

        # reset channel gain     
        self.channel_gain = 0.0
        self.max_trans_rate = 0.0
        self.old_comp_times = []
        # reset scheduling tasks
        self.sched_tasks.clear()

        self.position_x, self.position_y = self.gen_position(self.min_distance_from_edge, self.max_distance_from_edge)
        self.position_z = 1.8

        self.speed_x = 0.0
        self.speed_y = 0.0

        # self.task_num = np.random.binomial(1, self.task_arrival_prob)
        self.task_num = 1
        for i in range(self.task_num):
            # unit: Mb
            data_size = np.random.uniform(self.data_size_inl[0],
                                          self.data_size_inl[1])
            # unit: Gcycles/Mb
            comp_dens = np.random.uniform(self.comp_dens_inl[0],
                                          self.comp_dens_inl[1])
            
            task = Task(data_size, comp_dens, self.env_id)
            
            comp = data_size * comp_dens
            task.dly_cons = self.task_timeout_thre
            task.norm_csum_engy = comp * self.engy_fac * 6.25
            task.norm_esum_engy = comp * self.engy_fac * 1600
            self.sched_tasks.append(task)

    # Channel gain, local queue information, and task information
    def get_obs(self):
        max_trans_rate = self.max_trans_rate
        local_queue = self.time_ql
        if self.enable_virtual_queue_reward:
            local_vir_queue = self.virtual_time_ql
        else:
            local_vir_queue = -1.0
        task_msgs = []
        for i in range(self.task_num):
            data_size = self.sched_tasks[i].data_size
            comp_dens = self.sched_tasks[i].comp_dens
            dly_cons = self.sched_tasks[i].dly_cons
            task_msgs += [data_size, comp_dens, dly_cons]
        if self.task_num == 0:
            task_msgs = [0.0,0.0,0.0]
        obs = []
        obs += [max_trans_rate, local_queue, local_vir_queue] + task_msgs

        return obs

    # the end device moves when a time slot ends
    def move(self):
        enable_print = gp.settings.enable_print

        # sample acceleration size
        a_size = float(np.random.uniform(self.accelerate_speed_min, self.accelerate_speed_max))

        # sample acceleration direction in degrees
        a_dir_deg = float(np.random.uniform(self.accelerate_direction_min, self.accelerate_direction_max))  # unit: degree

        # convert degree to radian for trig functions
        a_dir_rad = math.radians(a_dir_deg)

        # decompose acceleration to x and y
        a_x = a_size * math.cos(a_dir_rad)
        a_y = a_size * math.sin(a_dir_rad)

        self.speed_x += a_x * self.delta
        self.speed_y += a_y * self.delta

        speed = math.sqrt(self.speed_x**2 + self.speed_y**2)
        if speed > self.speed_max:
            scale = self.speed_max / speed
            self.speed_x *= scale
            self.speed_y *= scale
        elif speed < self.speed_min:
            scale = self.speed_min / (speed + 1e-6)
            self.speed_x *= scale
            self.speed_y *= scale

        new_x = self.position_x + self.speed_x * self.delta
        new_y = self.position_y + self.speed_y * self.delta
        new_dist = math.sqrt(new_x**2 + new_y**2)

        if self.min_distance_from_edge <= new_dist <= self.max_distance_from_edge:
            # update position if it is valid
            self.position_x = new_x
            self.position_y = new_y

        if enable_print:
            print(f"[DEBUG] The position x of device {self.env_id} is {self.position_x}")
            print(f"[DEBUG] The position y of device {self.env_id} is {self.position_y}")
            print(f"[DEBUG] The speed x of device {self.env_id} is {self.speed_x}")
            print(f"[DEBUG] The speed y of device {self.env_id} is {self.speed_y}")

    # act: [offl_rto, trans_rto, local_comp_rto], all in [0, 1]
    # t_id: the start t_id is 1
    def compute(self, act, e_id, t_id, visualize=False):
        writer = self.writer
        enable_print = gp.settings.enable_print
        gen_task_cycle = self.gen_task_cycle
        start_slot = self.start_slot
    
        '''offloading'''
        # offloading data-size
        offl_dzs = []
        trans_power = self.trans_power
        device_comp_freq = self.device_comp_freq
        if(enable_print): print(f"[DEBUG] The action of device {self.env_id} is {act}")
        if self.task_num>=1:
            gap = gen_task_cycle * self.delta
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
            if visualize:
                writer.add_scalars(
                    f"detail{'_eval' if gp.settings.is_evaluate else ''}/offl_rto_{self.env_id}",
                    {f"ep_{e_id}": offl_rto},
                    t_id
                )
                writer.add_scalars(
                    f"detail{'_eval' if gp.settings.is_evaluate else ''}/trpw_rto_{self.env_id}",
                    {f"ep_{e_id}": trpw_rto},
                    t_id
                )
                writer.add_scalars(
                    f"detail{'_eval' if gp.settings.is_evaluate else ''}/comp_rto_{self.env_id}",
                    {f"ep_{e_id}": device_comp_rto},
                    t_id
                )
            # Config
            self.distance_from_edge = math.sqrt(self.position_x**2 + self.position_y**2)
            self.channel_gain = 0.1 * self.distance_from_edge ** (-3.5)
            # unit: Mb/s
            self.max_trans_rate = self.bandwidth * math.log(1 + self.trans_power * self.channel_gain / 
                                                self.noise_power, 2) * pow(10, -6)
            trans_rate = self.bandwidth * math.log(1 + trans_power * self.channel_gain / 
                                                self.noise_power, 2) * pow(10, -6)
            self.trans_rate = trans_rate
            delta_trans_dz = self.trans_rate * gap
            total_offl_dz = 0
            total_offl_comp = 0
            total_trans_dz = self.trans_ql
            # compute the queue time of transfering data.
            new_tran_queue_delay = 0
            '''offload computing part'''
            local_comps = []
            for task_id, offl_dz in enumerate(offl_dzs):
                
                total_offl_dz += offl_dz
                tran_queue_delay = new_tran_queue_delay + self.old_tran_queue_delay
                total_trans_dz += offl_dz
                task = self.sched_tasks[task_id]
                task.offl_dz = offl_dz
                total_offl_comp += offl_dz * task.comp_dens
                if task.offl_dz == 0:
                    task.trans_time = 0
                    task.tran_engy = 0
                    task.edge_comp_engy = 0
                else:
                    task.trans_time = tran_queue_delay + task.offl_dz / trans_rate
                    # print(f"[DEBUG] The device {self.env_id} 's offl_dz is {task.offl_dz}")
                    # print(f"[DEBUG] The device {self.env_id} 's trans_rate is {trans_rate}")
                    # print(f"[DEBUG] The device {self.env_id} 's old_tran_queue_delay before is {self.old_tran_queue_delay}")
                    # print(f"[DEBUG] The device {self.env_id} 's new_tran_queue_delay before is {new_tran_queue_delay}")
                    # print(f"[DEBUG] The device {self.env_id} 's tran_queue_delay is {tran_queue_delay}")
                    # print(f"[DEBUG] The device {self.env_id} 's trans_time is {task.trans_time}")
                    new_tran_queue_delay += task.offl_dz / trans_rate
                    task.tran_engy = trans_power * pow(10, -3) * \
                                    task.offl_dz / trans_rate
                local_comps.append((task.data_size - task.offl_dz) * \
                                        task.comp_dens)

                if(enable_print): print(f"[DEBUG] the local_comp in device {self.env_id} is {(task.data_size - task.offl_dz) * task.comp_dens}")
                if(enable_print): print(f"[DEBUG] the offl_comp in device {self.env_id} is {task.offl_dz * task.comp_dens}")

            self.old_tran_queue_delay = max(self.old_tran_queue_delay + new_tran_queue_delay - gap, 0)
            # Update the transmission queue length at every time slot, regardless of whether new tasks arrive
            self.trans_ql = max(0, total_trans_dz - delta_trans_dz)

            '''local computing part'''
            self.old_time_ql = self.time_ql
            device_act_queue_growth_rate = self.device_act_queue_growth_rate
            new_comp_queue_delay = 0
            for task_id, local_comp in enumerate(local_comps):
                task = self.sched_tasks[task_id]
                if local_comp == 0:
                    task.l_comp_dly = 0
                    task.l_proc_dly = 0
                    task.l_queue_dly = 0
                    task.local_comp_engy = 0
                else:
                    task.l_queue_dly = self.old_comp_queue_delay + new_comp_queue_delay
                    task.l_proc_dly = local_comp / device_comp_freq
                    task.l_comp_dly = task.l_queue_dly + task.l_proc_dly
                    new_comp_queue_delay += task.l_proc_dly

                task.local_comp_engy = self.engy_fac * pow(device_comp_freq,2) * local_comp

                # print(f"[DEBUG] The device", self.env_id, "'s old_comp_queue_delay is: ", self.old_comp_queue_delay)
                # print(f"[DEBUG] The device", self.env_id, "'s new_comp_queue_delay is: ", new_comp_queue_delay-task.l_proc_dly)
                # print(f"[DEBUG] The device", self.env_id, "'s l_queue_dly is: ", task.l_queue_dly)
                # print(f"[DEBUG] The device", self.env_id, "'s offl_rto is: ", offl_rto)
                # print(f"[DEBUG] The device", self.env_id, "'s engy_fac is: ", self.engy_fac)
                # print(f"[DEBUG] The device", self.env_id, "'s local_comp is: ", local_comp)
                # print(f"[DEBUG] The device", self.env_id, "'s data_size is: ", task.data_size)
                # print(f"[DEBUG] The device", self.env_id, "'s comp_dens is: ", task.comp_dens)
                # print(f"[DEBUG] The device", self.env_id, "'s local_comp_engy is: ", task.local_comp_engy)
                # print(f"[DEBUG] The device", self.env_id, "'s tran_engy is: ", task.tran_engy)

                # if(enable_print): print(f"[DEBUG] The device freq pow2 is {pow(device_comp_freq,2)}")
                old_time_ql_ = self.time_ql
                self.act_backlog = local_comp / device_comp_freq
                self.new_ql_change = device_act_queue_growth_rate * (self.act_backlog - gap)
                self.time_ql = max(0, old_time_ql_ + self.new_ql_change)

                # print(f"[DEBUG] The device", self.env_id, "'s time_ql is: ", self.time_ql)
                # print(f"[DEBUG] The device", self.env_id, "'s old_time_ql is: ", self.old_time_ql)
                # print(f"[DEBUG] The device", self.env_id, "'s new_ql_change is: ", self.new_ql_change)
                # print(f"[DEBUG] The device", self.env_id, "'s local_comp is: ", local_comp)
                # print(f"[DEBUG] The device", self.env_id, "'s device_comp_freq is: ", device_comp_freq)
                # print(f"[DEBUG] The device", self.env_id, "'s delta is: ", self.delta)

                self.avail_task_num += 1
                # avg_local_time: use only the average computation time of the most recent time slots
                self.old_comp_times.append(local_comp / device_comp_freq)
                tail = self.old_comp_times[-self.statSlotNum:]
                self.avg_local_time = sum(tail) / len(tail) if tail else 0
                if(enable_print): print(f"[DEBUG] the local comp_dly in device {self.env_id} is {task.l_comp_dly}")
            
            self.old_comp_queue_delay = max(self.old_comp_queue_delay + new_comp_queue_delay - gap, 0)
            self.old_virtual_time_ql = self.virtual_time_ql
            EPS = 1e-8
            device_vir_queue_growth_rate = self.device_vir_queue_growth_rate
            if self.avg_local_time > EPS:
                old_vir_time_ql_ = self.virtual_time_ql
                self.vir_backlog = self.time_ql/self.avg_local_time*self.delta*self.gen_task_cycle
                self.new_vir_ql_change = device_vir_queue_growth_rate*(self.vir_backlog - self.device_dly_adj_val)
                self.virtual_time_ql = max(0, self.virtual_time_ql + self.new_vir_ql_change)
    
        # print(f"[DEBUG] The device", self.env_id, "'s avg_local_time is: ", self.avg_local_time)
        # print(f"[DEBUG] The device", self.env_id, "'s device_dly_adj_val is: ", self.device_dly_adj_val)

        # print(f"[DEBUG] The device", self.env_id, "'s virtual_time_ql is: ", self.virtual_time_ql)
        # print(f"[DEBUG] The device", self.env_id, "'s old_virtual_time_ql is: ", self.old_virtual_time_ql)
        # print(f"[DEBUG] The device", self.env_id, "'s new_vir_ql_change is: ", self.new_vir_ql_change)
        # print(f"[DEBUG] The device", self.env_id, "'s fine_vir_ql_change is: ", device_vir_queue_growth_rate*(self.time_ql/self.avg_local_time - self.device_dly_adj_val))
        

        if(enable_print): print(f"[DEBUG] The device", self.env_id, "'s avg_local_time is: ", self.avg_local_time)
        if(enable_print): print(f"[DEBUG] The device", self.env_id, "'s old_time_ql is: ", self.old_time_ql)
        if(enable_print): print(f"[DEBUG] The device", self.env_id, "'s new_ql_change is: ", self.new_ql_change)
        if(enable_print): print(f"[DEBUG] The device", self.env_id, "'s time_ql is: ", self.time_ql)
        if(enable_print): print(f"[DEBUG] The device", self.env_id, "'s old_virtual_time_ql is: ", self.old_virtual_time_ql)
        if(enable_print): print(f"[DEBUG] The device", self.env_id, "'s new_vir_ql_change is: ", self.new_vir_ql_change)
        if(enable_print): print(f"[DEBUG] The device", self.env_id, "'s virtual_time_ql is: ", self.virtual_time_ql)

        if visualize:
            # writer.add_scalars(
            #     f"detail{'_eval' if gp.settings.is_evaluate else ''}/device_avg_local_time_{self.env_id}",
            #     {f"ep_{e_id}": self.avg_local_time},
            #     t_id
            # )
            writer.add_scalars(
                f"detail{'_eval' if gp.settings.is_evaluate else ''}/device_time_ql_{self.env_id}",
                {f"ep_{e_id}_act": self.time_ql},
                t_id
            )
            # writer.add_scalars(
            #     f"detail{'_eval' if gp.settings.is_evaluate else ''}/device_time_ql_{self.env_id}",
            #     {f"ep_{e_id}_act_chg": self.new_ql_change},
            #     t_id
            # )
            writer.add_scalars(
                f"detail{'_eval' if gp.settings.is_evaluate else ''}/device_time_ql_{self.env_id}",
                {f"ep_{e_id}_vir": self.virtual_time_ql},
                t_id
            )
            # writer.add_scalars(
            #     f"detail{'_eval' if gp.settings.is_evaluate else ''}/device_time_ql_{self.env_id}",
            #     {f"ep_{e_id}_vir_chg": self.new_vir_ql_change},
            #     t_id
            # )

        # update scheduling tasks
        sched_tasks = copy.copy(self.sched_tasks)
        self.sched_tasks.clear()
        if (t_id+1) % gen_task_cycle == start_slot:
            self.task_num = 1
            for i in range(self.task_num):
                # unit: Mb
                data_size = np.random.uniform(self.data_size_inl[0],
                                            self.data_size_inl[1])
                # unit: Gcycles/Mb
                comp_dens = np.random.uniform(self.comp_dens_inl[0],
                                            self.comp_dens_inl[1])
                
                task = Task(data_size, comp_dens, self.env_id)
                
                comp = data_size * comp_dens
                task.dly_cons = self.task_timeout_thre
                task.norm_csum_engy = comp * self.engy_fac * 9
                task.norm_esum_engy = comp * self.engy_fac * 1600
                
                self.sched_tasks.append(task)
        else:
            self.task_num = 0
        # end device moves when the time slot ends
        self.move()

        return sched_tasks