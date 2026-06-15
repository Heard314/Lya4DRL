import copy
import math
import random
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
        self.norm_csum_engy_fac = None
        self.norm_esum_engy_fac = None

        # Debug
        self.l_queue_dly = None
        self.e_queue_dly = None
        self.l_proc_dly = None
        self.e_proc_dly = None
        self.target_server = None  # which edge server the task is offloaded to
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
               "\nnorm_csum_engy_fac: " + str(self.norm_csum_engy_fac) + \
               "\nnorm_comp_expn: " + str(self.norm_esum_engy_fac)

class DeviceEnv():
    
    def gen_position(self, min_dis, max_dis):
        while True:
            x = self._rnd.uniform(-max_dis, max_dis)
            y = self._rnd.uniform(-max_dis, max_dis)
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

        # create per-env random generators
        base_seed = gp.settings.seed
        self._rnd = random.Random(base_seed + env_id)
        self._np_rnd = np.random.RandomState(base_seed + env_id)

        self.device_type = gen_params.device_types[env_id]
        self.device_type_num = gen_params.device_type_num

        self.edge_server_num = gen_params.edge_server_num
        self.server_positions = gen_params.edge_server_positions
        self.max_trans_rates = [0.0] * self.edge_server_num

        self.enable_virtual_queue_reward = gen_params.enable_virtual_queue_reward
        self.enable_actual_queue_reward = gen_params.enable_actual_queue_reward
        # unit: s
        self.delta = gen_params.delta
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
        self.edge_comp_freq = gen_params.edge_comp_freq
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
        self.avail_task_num = 0
        self.sched_tasks = []

        self.total_comp_time = 0.0
        self.total_tran_time = 0.0

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

        self.total_comp_time = 0.0
        self.total_tran_time = 0.0

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
            data_size = self._np_rnd.uniform(self.data_size_inl[0],
                                             self.data_size_inl[1])
            # unit: Gcycles/Mb
            comp_dens = self._np_rnd.uniform(self.comp_dens_inl[0],
                                             self.comp_dens_inl[1])

            task = Task(data_size, comp_dens, self.env_id)

            comp = data_size * comp_dens
            task.dly_cons = self.task_timeout_thre
            task.norm_csum_engy_fac = self.engy_fac * self.device_comp_freq**2
            task.norm_esum_engy_fac = self.engy_fac * self.edge_comp_freq**2
            self.sched_tasks.append(task)

    # Channel gains to all servers, local queue information, and task information
    def get_obs(self):
        obs = list(self.max_trans_rates)
        obs.append(self.time_ql)
        if self.enable_virtual_queue_reward:
            obs.append(self.virtual_time_ql)
        else:
            obs.append(-1.0)
        task_msgs = []
        for i in range(self.task_num):
            data_size = self.sched_tasks[i].data_size
            comp_dens = self.sched_tasks[i].comp_dens
            dly_cons = self.sched_tasks[i].dly_cons
            task_msgs += [data_size, comp_dens, dly_cons]
        if self.task_num == 0:
            task_msgs = [0.0, 0.0, 0.0]
        obs += task_msgs

        return obs

    def _update_trans_rates(self):
        """Calculate max transmission rate to each edge server."""
        for s in range(self.edge_server_num):
            sx, sy = self.server_positions[s]
            dist = math.sqrt((self.position_x - sx)**2 + (self.position_y - sy)**2 + self.position_z**2)
            ch_gain = 0.1 * max(dist, 1.0) ** (-3.5)
            self.max_trans_rates[s] = (
                self.bandwidth * math.log(1 + self.trans_power * ch_gain / self.noise_power, 2) * 1e-6
            )

    # the end device moves when a time slot ends
    def move(self):
        enable_print = gp.settings.enable_print

        # sample acceleration size
        a_size = self._rnd.uniform(self.accelerate_speed_min, self.accelerate_speed_max)

        # sample acceleration direction in degrees
        a_dir_deg = self._rnd.uniform(self.accelerate_direction_min, self.accelerate_direction_max)  # unit: degree

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

        self._update_trans_rates()

    # act: [server_id, offl_rto, trpw_rto, comp_rto]
    def compute(self, act, e_id, t_id, visualize=False):
        writer = self.writer
        enable_print = gp.settings.enable_print
        gen_task_cycle = self.gen_task_cycle
        start_slot = self.start_slot

        '''offloading'''
        offl_dzs = []
        trans_power = self.trans_power
        device_comp_freq = self.device_comp_freq
        if(enable_print): print(f"[DEBUG] The action of device {self.env_id} is {act}")
        if self.task_num>=1:
            gap = gen_task_cycle * self.delta
            # server selection
            server_id = int(act[0])
            # offloading ratio (clamped to valid range)
            offl_rto = np.clip(act[1], 0.6, 1.0)
            offl_dz = self.sched_tasks[0].data_size * offl_rto
            offl_dzs.append(offl_dz)
            # transmission-power ratio (clamped to valid range)
            trpw_rto = np.clip(act[2], 0.6, 1.0)
            trans_power = self.trans_power * trpw_rto
            # local compute ratio (clamped to valid range)
            device_comp_rto = np.clip(act[3], 0.6, 1.0)
            device_comp_freq = self.device_comp_freq * device_comp_rto

            if gp.settings.enable_trace:
                gp.trace_print(f"\n{'='*60}")
                gp.trace_print(f"[TRACE] t={t_id}, Device {self.env_id}, Task arrival (type={self.device_type})")
                gp.trace_print(f"  Action raw: server_id={server_id}, offl_rto={act[1]:.4f}, trpw_rto={act[2]:.4f}, comp_rto={act[3]:.4f}")
                gp.trace_print(f"  After clip: offl_rto={offl_rto:.4f}, trpw_rto={trpw_rto:.4f}, comp_rto={device_comp_rto:.4f}")
                gp.trace_print(f"  offl_dz = data_size * offl_rto = {self.sched_tasks[0].data_size:.4f} * {offl_rto:.4f} = {offl_dz:.4f}")
                gp.trace_print(f"  trans_power = base_trans_power * trpw_rto = {self.trans_power:.4f} * {trpw_rto:.4f} = {trans_power:.4f}")
                gp.trace_print(f"  device_comp_freq = base_freq * comp_rto = {self.device_comp_freq:.4f} * {device_comp_rto:.4f} = {device_comp_freq:.4f}")
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
                writer.add_scalars(
                    f"detail{'_eval' if gp.settings.is_evaluate else ''}/server_id_{self.env_id}",
                    {f"ep_{e_id}": server_id},
                    t_id
                )
            # Config
            sx, sy = self.server_positions[server_id]
            self.distance_from_edge = math.sqrt((self.position_x - sx)**2 + (self.position_y - sy)**2 + self.position_z**2)
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

            if gp.settings.enable_trace:
                gp.trace_print(f"  distance = sqrt(({self.position_x:.2f}-{sx:.2f})^2 + ({self.position_y:.2f}-{sy:.2f})^2 + {self.position_z}^2) = {self.distance_from_edge:.2f}")
                gp.trace_print(f"  channel_gain = 0.1 * distance^(-3.5) = 0.1 * {self.distance_from_edge:.2f}^(-3.5) = {self.channel_gain:.6e}")
                gp.trace_print(f"  trans_rate = BW * log2(1 + P_tx * ch_gain / N0) * 1e-6")
                gp.trace_print(f"             = {self.bandwidth:.2e} * log2(1 + {trans_power:.2e} * {self.channel_gain:.6e} / {self.noise_power:.2e}) * 1e-6")
                gp.trace_print(f"             = {trans_rate:.6f} Mb/s")
                gp.trace_print(f"  max_trans_rate = {self.max_trans_rate:.6f} Mb/s")
                gp.trace_print(f"  delta_trans_dz = trans_rate * gap = {trans_rate:.6f} * {gap:.4f} = {delta_trans_dz:.6f}")
                gp.trace_print(f"  total_trans_dz (init) = trans_ql = {self.trans_ql:.6f}")
            '''offload computing part'''
            local_comps = []
            for task_id, offl_dz in enumerate(offl_dzs):
                total_offl_dz += offl_dz
                tran_queue_delay = max(self.total_tran_time - t_id * self.delta,0)
                total_trans_dz += offl_dz
                task = self.sched_tasks[task_id]
                task.offl_dz = offl_dz
                task.target_server = server_id
                total_offl_comp += offl_dz * task.comp_dens
                if task.offl_dz == 0:
                    task.trans_time = 0
                    task.tran_engy = 0
                    task.edge_comp_engy = 0
                else:
                    task.trans_time = tran_queue_delay + task.offl_dz / trans_rate
                    self.total_tran_time += task.offl_dz / trans_rate
                    task.tran_engy = trans_power * pow(10, -3) * \
                                    task.offl_dz / trans_rate
                local_comps.append((task.data_size - task.offl_dz) * \
                                        task.comp_dens)

                if(enable_print): print(f"[DEBUG] the local_comp in device {self.env_id} is {(task.data_size - task.offl_dz) * task.comp_dens}")
                if(enable_print): print(f"[DEBUG] the offl_comp in device {self.env_id} is {task.offl_dz * task.comp_dens}")

                if gp.settings.enable_trace:
                    gp.trace_print(f"\n  --- Task {task_id} offloading ---")
                    gp.trace_print(f"  task.data_size={task.data_size:.4f}, task.comp_dens={task.comp_dens:.4f}, task.dly_cons={task.dly_cons:.4f}")
                    gp.trace_print(f"  offl_dz = {offl_dz:.4f}, local_data = {task.data_size - offl_dz:.4f}")
                    gp.trace_print(f"  local_comp = (data_size - offl_dz) * comp_dens = {task.data_size - offl_dz:.4f} * {task.comp_dens:.4f} = {local_comps[-1]:.4f}")
                    gp.trace_print(f"  offl_comp = offl_dz * comp_dens = {offl_dz:.4f} * {task.comp_dens:.4f} = {offl_dz * task.comp_dens:.4f}")
                    if task.offl_dz > 0:
                        gp.trace_print(f"  tran_queue_delay = max(total_tran_time - t*delta, 0) = max({self.total_tran_time:.6f} - {t_id * self.delta:.4f}, 0) = {max(self.total_tran_time - t_id * self.delta, 0):.6f}")
                        gp.trace_print(f"  trans_time = tran_queue_delay + offl_dz / trans_rate = {tran_queue_delay:.6f} + {offl_dz:.4f}/{trans_rate:.6f} = {task.trans_time:.6f}")
                        gp.trace_print(f"  total_tran_time (after) = {self.total_tran_time:.6f}")
                        gp.trace_print(f"  tran_engy = P_tx * 1e-3 * offl_dz / trans_rate = {trans_power:.2f} * 1e-3 * {offl_dz:.4f}/{trans_rate:.6f} = {task.tran_engy:.6f}")
            # Update the transmission queue length at every time slot, regardless of whether new tasks arrive
            old_trans_ql = self.trans_ql
            self.trans_ql = max(0, total_trans_dz - delta_trans_dz)

            if gp.settings.enable_trace:
                gp.trace_print(f"\n  --- Transmission queue update ---")
                gp.trace_print(f"  trans_ql = max(0, total_trans_dz - delta_trans_dz)")
                gp.trace_print(f"           = max(0, {total_trans_dz:.6f} - {delta_trans_dz:.6f})")
                gp.trace_print(f"           = max(0, {total_trans_dz - delta_trans_dz:.6f}) = {self.trans_ql:.6f} (was {old_trans_ql:.6f})")
            '''local computing part'''
            self.old_time_ql = self.time_ql
            device_act_queue_growth_rate = self.device_act_queue_growth_rate
            for task_id, local_comp in enumerate(local_comps):
                task = self.sched_tasks[task_id]
                if local_comp == 0:
                    task.l_comp_dly = 0
                    task.l_proc_dly = 0
                    task.l_queue_dly = 0
                    task.local_comp_engy = 0
                else:
                    task.l_queue_dly = max(self.total_comp_time - t_id * self.delta, 0)
                    task.l_proc_dly = local_comp / device_comp_freq
                    task.l_comp_dly = task.l_queue_dly + task.l_proc_dly
                    
                self.total_comp_time += task.l_proc_dly
                task.local_comp_engy = self.engy_fac * pow(device_comp_freq,2) * local_comp
                old_time_ql_ = self.time_ql
                self.time_ql = max(0, old_time_ql_ + device_act_queue_growth_rate * (local_comp / device_comp_freq - gap))
                self.new_ql_change = device_act_queue_growth_rate * (local_comp / device_comp_freq - gap)
                self.act_backlog = local_comp / device_comp_freq
                self.avail_task_num += 1
                # avg_local_time: use only the average computation time of the most recent time slots
                self.old_comp_times.append(local_comp / device_comp_freq)
                tail = self.old_comp_times[-self.statSlotNum:]
                self.avg_local_time = sum(tail) / len(tail) if tail else 0
                if(enable_print): print(f"[DEBUG] the local comp_dly in device {self.env_id} is {task.l_comp_dly}")

                if gp.settings.enable_trace:
                    gp.trace_print(f"\n  --- Task {task_id} local computation ---")
                    gp.trace_print(f"  l_queue_dly = max(total_comp_time - t*delta, 0) = max({self.total_comp_time - task.l_proc_dly:.6f} - {t_id * self.delta:.4f}, 0) = {task.l_queue_dly:.6f}")
                    gp.trace_print(f"  l_proc_dly = local_comp / dev_comp_freq = {local_comp:.4f} / {device_comp_freq:.4f} = {task.l_proc_dly:.6f}")
                    gp.trace_print(f"  l_comp_dly = l_queue_dly + l_proc_dly = {task.l_queue_dly:.6f} + {task.l_proc_dly:.6f} = {task.l_comp_dly:.6f}")
                    gp.trace_print(f"  total_comp_time (after) = {self.total_comp_time:.6f}")
                    gp.trace_print(f"  local_comp_engy = engy_fac * freq^2 * local_comp = {self.engy_fac:.4f} * {device_comp_freq:.4f}^2 * {local_comp:.4f} = {task.local_comp_engy:.6f}")
                    gp.trace_print(f"  time_ql = max(0, old_time_ql + growth_rate * (comp_time - gap))")
                    gp.trace_print(f"          = max(0, {old_time_ql_:.6f} + {device_act_queue_growth_rate} * ({local_comp/device_comp_freq:.6f} - {gap:.4f}))")
                    gp.trace_print(f"          = max(0, {old_time_ql_ + device_act_queue_growth_rate * (local_comp / device_comp_freq - gap):.6f}) = {self.time_ql:.6f}")
                    gp.trace_print(f"  new_ql_change = {self.new_ql_change:.6f}")
                    gp.trace_print(f"  old_comp_times tail (last {len(tail)}): avg_local_time = {self.avg_local_time:.6f}")

            self.old_virtual_time_ql = self.virtual_time_ql
            EPS = 1e-8
            device_vir_queue_growth_rate = self.device_vir_queue_growth_rate
            if self.avg_local_time > EPS:
                old_vir_time_ql_ = self.virtual_time_ql
                vir_backlog = self.time_ql / self.avg_local_time * self.delta * self.gen_task_cycle
                self.virtual_time_ql = max(0, self.virtual_time_ql + device_vir_queue_growth_rate*(vir_backlog - self.device_dly_adj_val))
                self.new_vir_ql_change = device_vir_queue_growth_rate*(vir_backlog - self.device_dly_adj_val)
                self.vir_backlog = vir_backlog

                if gp.settings.enable_trace:
                    gp.trace_print(f"\n  --- Virtual queue update ---")
                    gp.trace_print(f"  vir_backlog = time_ql / avg_local_time * delta * gen_cycle")
                    gp.trace_print(f"              = {self.time_ql:.6f} / {self.avg_local_time:.6f} * {self.delta} * {self.gen_task_cycle} = {vir_backlog:.6f}")
                    gp.trace_print(f"  device_dly_adj_val = {self.device_dly_adj_val:.6f}")
                    gp.trace_print(f"  virtual_time_ql = max(0, old_vir_ql + vir_growth * (vir_backlog - dly_adj))")
                    gp.trace_print(f"                  = max(0, {old_vir_time_ql_:.6f} + {device_vir_queue_growth_rate} * ({vir_backlog:.6f} - {self.device_dly_adj_val:.6f}))")
                    gp.trace_print(f"                  = max(0, {old_vir_time_ql_ + device_vir_queue_growth_rate * (vir_backlog - self.device_dly_adj_val):.6f}) = {self.virtual_time_ql:.6f}")
        else:
            # Idle slot: no task, no computation, no transmission.
            # The arrival-slot formula already used gap = gen_task_cycle * delta (0.5s)
            # to account for the full 5-slot cycle. No further queue changes here.
            self.old_time_ql = self.time_ql
            self.old_virtual_time_ql = self.virtual_time_ql
            self.act_backlog = 0.0
            self.new_ql_change = 0.0
            self.new_vir_ql_change = 0.0

            if gp.settings.enable_trace:
                gp.trace_print(f"\n{'='*60}")
                gp.trace_print(f"[TRACE] t={t_id}, Device {self.env_id}, IDLE slot (no task, queues unchanged)")
                gp.trace_print(f"  time_ql = {self.time_ql:.6f}, trans_ql = {self.trans_ql:.6f}, virtual_time_ql = {self.virtual_time_ql:.6f}")

    
        if(enable_print): print(f"[DEBUG] The device", self.env_id, "'s avg_local_time is: ", self.avg_local_time)
        if(enable_print): print(f"[DEBUG] The device", self.env_id, "'s old_time_ql is: ", self.old_time_ql)
        if(enable_print): print(f"[DEBUG] The device", self.env_id, "'s new_ql_change is: ", self.new_ql_change)
        if(enable_print): print(f"[DEBUG] The device", self.env_id, "'s time_ql is: ", self.time_ql)
        if(enable_print): print(f"[DEBUG] The device", self.env_id, "'s old_virtual_time_ql is: ", self.old_virtual_time_ql)
        if(enable_print): print(f"[DEBUG] The device", self.env_id, "'s new_vir_ql_change is: ", self.new_vir_ql_change)
        if(enable_print): print(f"[DEBUG] The device", self.env_id, "'s virtual_time_ql is: ", self.virtual_time_ql)

        if visualize:
            writer.add_scalars(
                f"detail{'_eval' if gp.settings.is_evaluate else ''}/device_time_ql_{self.env_id}",
                {f"ep_{e_id}_act": self.time_ql},
                t_id
            )
            writer.add_scalars(
                f"detail{'_eval' if gp.settings.is_evaluate else ''}/device_time_ql_{self.env_id}",
                {f"ep_{e_id}_vir": self.virtual_time_ql},
                t_id
            )

        # update scheduling tasks
        sched_tasks = copy.copy(self.sched_tasks)
        self.sched_tasks.clear()
        if (t_id+1) % gen_task_cycle == start_slot:
            self.task_num = 1
            for i in range(self.task_num):
                # unit: Mb
                data_size = self._np_rnd.uniform(self.data_size_inl[0],
                                            self.data_size_inl[1])
                # unit: Gcycles/Mb
                comp_dens = self._np_rnd.uniform(self.comp_dens_inl[0],
                                            self.comp_dens_inl[1])
                
                task = Task(data_size, comp_dens, self.env_id)
                
                comp = data_size * comp_dens
                task.dly_cons = self.task_timeout_thre
                task.norm_csum_engy_fac = self.engy_fac * self.device_comp_freq**2
                task.norm_esum_engy_fac = self.engy_fac * self.edge_comp_freq**2
                
                self.sched_tasks.append(task)
        else:
            self.task_num = 0
        # end device moves when the time slot ends
        self.move()

        return sched_tasks