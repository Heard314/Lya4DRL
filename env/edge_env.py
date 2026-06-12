import config.global_params as gp
class EdgeEnv():
    def __init__(self, general_params, writer = None):

        # Summary Writer
        self.writer = writer

        # task generation cycle
        self.gen_task_cycle = general_params.gen_task_cycle
        self.start_slot = general_params.start_slot

        self.enable_virtual_queue_reward = general_params.enable_virtual_queue_reward
        self.enable_actual_queue_reward = general_params.enable_actual_queue_reward

        # unit: s
        self.delta = general_params.delta

        self.device_num = general_params.device_num
        self.device_type_num = general_params.device_type_num
        self.device_types_ref = general_params.device_types
        # unit: Gcycles/s
        self.edge_comp_freq = general_params.edge_comp_freq

        self.device_num_per_type = general_params.device_num_per_type

        self.edge_weight_w = general_params.edge_weight_w
        self.edge_weight_b = general_params.edge_weight_b

        # v3: single FIFO queue per server
        self.queue_num = 1

        # Scalar queues
        self.edge_queue_time_ql = 0.0
        self.old_edge_queue_time_ql = 0.0
        self.virtual_edge_queue_time_ql = 0.0
        self.old_virtual_edge_queue_time_ql = 0.0
        self.new_edge_ql_change = 0.0
        self.new_vir_edge_ql_change = 0.0
        self.total_comp_time = 0.0
        self.total_comp_amount = 0.0
        self.avg_edge_time = 0.0
        self.old_comp_times = []

        self.edge_dly_adj_fac = general_params.edge_dly_adj_fac
        self.edge_dly_adj_val = min(general_params.comp_dly_thre) * general_params.edge_dly_adj_fac[0]
        self.statSlotNum = general_params.statSlotNum
        self.delta_num = 0  # elapsed slots in this episode

        self.edge_act_queue_growth_rate = general_params.edge_act_queue_growth_rate
        self.edge_vir_queue_growth_rate = general_params.edge_vir_queue_growth_rate

    def reset(self):
        self.total_comp_time = 0.0
        self.total_comp_amount = 0.0
        self.edge_queue_time_ql = 0.0
        self.old_edge_queue_time_ql = 0.0
        self.virtual_edge_queue_time_ql = 0.0
        self.old_virtual_edge_queue_time_ql = 0.0
        self.avg_edge_time = 0.0
        self.old_comp_times = []
        self.new_edge_ql_change = 0.0
        self.new_vir_edge_ql_change = 0.0
        self.delta_num = 0

    def get_obs(self):
        return [
            self.edge_queue_time_ql,
            self.virtual_edge_queue_time_ql if self.enable_virtual_queue_reward else -1.0
        ]

    def compute(self, tasks, e_id, t_id, visualize=False):
        writer = self.writer
        enable_print = gp.settings.enable_print

        gap = self.gen_task_cycle * self.delta

        self.old_edge_queue_time_ql = self.edge_queue_time_ql

        self.delta_num += 1

        if not tasks:
            if t_id % self.gen_task_cycle == self.start_slot:
                self.edge_queue_time_ql = max(self.edge_queue_time_ql -
                    self.edge_act_queue_growth_rate * gap, 0)
                self.new_edge_ql_change = -self.edge_act_queue_growth_rate * gap

                self.old_virtual_edge_queue_time_ql = self.virtual_edge_queue_time_ql
                EPS = 1e-8
                if self.avg_edge_time > EPS:
                    self.virtual_edge_queue_time_ql = max(
                        self.virtual_edge_queue_time_ql + self.edge_vir_queue_growth_rate * (
                            self.edge_queue_time_ql / self.avg_edge_time * self.delta * self.gen_task_cycle -
                            self.edge_dly_adj_val
                        ), 0
                    )
                    self.new_vir_edge_ql_change = self.edge_vir_queue_growth_rate * (
                        self.edge_queue_time_ql / self.avg_edge_time * self.delta * self.gen_task_cycle -
                        self.edge_dly_adj_val
                    )
            return

        # Sort by trans_time for FIFO processing
        tasks = sorted(tasks, key=lambda t: t.trans_time)

        total_comp_need = 0.0
        total_comp_used = 0.0

        for task in tasks:
            if task.trans_time == 0:
                task.e_comp_dly = 0
                task.e_queue_dly = 0
                task.e_proc_dly = 0
                task.edge_comp_engy = 0
            else:
                task.edge_comp_engy = task.offl_dz * task.comp_dens * pow(self.edge_comp_freq, 2)
                task.e_queue_dly = max(self.total_comp_time - t_id * self.delta, 0)
                task.e_proc_dly = task.offl_dz * task.comp_dens / self.edge_comp_freq
                task.e_comp_dly = max(task.e_queue_dly, task.trans_time) + task.e_proc_dly
                total_comp_need += task.e_proc_dly
                comp_used = min(task.e_proc_dly, max(0, gap - task.trans_time))
                total_comp_used += comp_used
                self.total_comp_time += task.e_proc_dly
                self.total_comp_amount += task.offl_dz * task.comp_dens

            self.old_comp_times.append(task.e_proc_dly)
            tail = self.old_comp_times[-self.statSlotNum:]
            self.avg_edge_time = sum(tail) / len(tail) if tail else 0

        # Single FIFO queue evolution (scalar)
        if t_id % self.gen_task_cycle == self.start_slot:
            self.edge_queue_time_ql = max(
                self.edge_queue_time_ql +
                self.edge_act_queue_growth_rate * (total_comp_need - total_comp_used), 0
            )
            self.new_edge_ql_change = self.edge_act_queue_growth_rate * (total_comp_need - total_comp_used)

            self.old_virtual_edge_queue_time_ql = self.virtual_edge_queue_time_ql
            EPS = 1e-8
            if self.avg_edge_time > EPS:
                self.virtual_edge_queue_time_ql = max(
                    self.virtual_edge_queue_time_ql + self.edge_vir_queue_growth_rate * (
                        self.edge_queue_time_ql / self.avg_edge_time * self.delta * self.gen_task_cycle -
                        self.edge_dly_adj_val
                    ), 0
                )
                self.new_vir_edge_ql_change = self.edge_vir_queue_growth_rate * (
                    self.edge_queue_time_ql / self.avg_edge_time * self.delta * self.gen_task_cycle -
                    self.edge_dly_adj_val
                )

        if visualize:
            writer.add_scalars(
                f"detail{'_eval' if gp.settings.is_evaluate else ''}/edge_time_ql_{e_id}",
                {f"ep_{e_id}_act": self.edge_queue_time_ql},
                t_id
            )
            writer.add_scalars(
                f"detail{'_eval' if gp.settings.is_evaluate else ''}/edge_time_ql_{e_id}",
                {f"ep_{e_id}_vir": self.virtual_edge_queue_time_ql},
                t_id
            )
