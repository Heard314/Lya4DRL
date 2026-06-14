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
        self.edge_dly_adj_val = min(general_params.comp_dly_thre) * general_params.edge_dly_adj_fac[0] * self.delta
        self.statSlotNum = general_params.statSlotNum
        self.delta_num = 0  # elapsed slots in this episode

        self.edge_act_queue_growth_rate = general_params.edge_act_queue_growth_rate
        self.edge_vir_queue_growth_rate = general_params.edge_vir_queue_growth_rate

        self.avg_device_num = 5

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
            # Idle: no tasks arrived this slot. The queue state from the last
            # arrival already accounts for the full gen_task_cycle gap. No change.
            if gp.settings.enable_trace and t_id % self.gen_task_cycle == self.start_slot:
                gp.trace_print(f"\n{'='*60}")
                gp.trace_print(f"[TRACE] t={t_id}, Edge (idle) | No tasks, queues unchanged")
                gp.trace_print(f"  edge_queue_time_ql = {self.edge_queue_time_ql:.6f}, virtual_edge_queue_time_ql = {self.virtual_edge_queue_time_ql:.6f}")
            return

        # Sort by trans_time for FIFO processing
        tasks = sorted(tasks, key=lambda t: t.trans_time)

        total_comp_need = 0.0
        total_comp_used = min(self.edge_queue_time_ql, gap)  # can only use up to the current queue time or the gap
        comp_dlys = min(gap, self.edge_queue_time_ql)  # actual comp delay that can be processed in this slot
        old_total_comp_time = self.total_comp_time
        new_queue_time = 0.0
        if gp.settings.enable_trace:
            gp.trace_print(f"\n{'='*60}")
            gp.trace_print(f"[TRACE] t={t_id}, Edge (tasks={len(tasks)}) | Processing {len(tasks)} offloaded tasks")
            gp.trace_print(f"  Initial: edge_queue_time_ql(old)={self.old_edge_queue_time_ql:.6f}, total_comp_time={self.total_comp_time:.6f}, total_comp_used(init)=min(ql,gap)={total_comp_used:.6f}, comp_dlys(init)=min(gap,ql)={comp_dlys:.6f}")
            gp.trace_print(f"  gap = gen_cycle * delta = {self.gen_task_cycle} * {self.delta} = {gap:.4f}")

        for task in tasks:
            if task.trans_time == 0:
                task.e_comp_dly = 0
                task.e_queue_dly = 0
                task.e_proc_dly = 0
                task.edge_comp_engy = 0
            else:
                task.edge_comp_engy = task.offl_dz * task.comp_dens * pow(self.edge_comp_freq, 2)
                task.e_queue_dly = max(old_total_comp_time - t_id * self.delta, 0) + new_queue_time
                task.e_proc_dly = task.offl_dz * task.comp_dens / self.edge_comp_freq
                task.e_comp_dly = max(task.e_queue_dly, task.trans_time) + task.e_proc_dly
                total_comp_need += task.e_proc_dly
                old_comp_dlys = comp_dlys
                old_new_queue_time = new_queue_time
                comp_used = min(task.e_proc_dly, max(0, gap - max(task.trans_time, comp_dlys)))
                total_comp_used += comp_used
                comp_dlys = max(comp_dlys, task.trans_time) + task.e_proc_dly
                new_queue_time += task.e_proc_dly
                self.total_comp_time += task.e_proc_dly
                self.total_comp_amount += task.offl_dz * task.comp_dens

                if gp.settings.enable_trace:
                    qdelay_base = max(old_total_comp_time - t_id * self.delta, 0)
                    gp.trace_print(f"\n  --- Task (device={task.device_id}, offl_dz={task.offl_dz:.4f}, comp_dens={task.comp_dens:.4f}) ---")
                    gp.trace_print(f"  trans_time={task.trans_time:.6f}, comp_dlys(old)={old_comp_dlys:.6f}, new_queue_time(old)={old_new_queue_time:.6f}, global_total_comp(old)={old_total_comp_time:.6f}")
                    gp.trace_print(f"  edge_comp_engy = offl_dz * comp_dens * freq^2 = {task.offl_dz:.4f} * {task.comp_dens:.4f} * {self.edge_comp_freq}^2 = {task.edge_comp_engy:.6f}")
                    gp.trace_print(f"  e_queue_dly = max(old_global_total - t*delta, 0) + new_queue_time(old)")
                    gp.trace_print(f"              = max({old_total_comp_time:.6f} - {t_id * self.delta:.4f}, 0) + {old_new_queue_time:.6f}")
                    gp.trace_print(f"              = {qdelay_base:.6f} + {old_new_queue_time:.6f} = {task.e_queue_dly:.6f}")
                    gp.trace_print(f"  e_proc_dly = offl_dz * comp_dens / edge_freq = {task.offl_dz:.4f} * {task.comp_dens:.4f} / {self.edge_comp_freq} = {task.e_proc_dly:.6f}")
                    gp.trace_print(f"  e_comp_dly = max(e_queue_dly, trans_time) + e_proc_dly = max({task.e_queue_dly:.6f}, {task.trans_time:.6f}) + {task.e_proc_dly:.6f} = {task.e_comp_dly:.6f}")
                    gp.trace_print(f"  comp_used = min(e_proc_dly, max(0, gap - max(trans_time, comp_dlys)))")
                    gp.trace_print(f"           = min({task.e_proc_dly:.6f}, max(0, {gap:.4f} - max({task.trans_time:.6f}, {old_comp_dlys:.6f})))")
                    gp.trace_print(f"           = min({task.e_proc_dly:.6f}, max(0, {gap:.4f} - {max(task.trans_time, old_comp_dlys):.6f})) = {comp_used:.6f}")
                    gp.trace_print(f"  comp_dlys(new) = max(old_comp_dlys, trans_time) + e_proc_dly")
                    gp.trace_print(f"                = max({old_comp_dlys:.6f}, {task.trans_time:.6f}) + {task.e_proc_dly:.6f}")
                    gp.trace_print(f"                = {max(old_comp_dlys, task.trans_time):.6f} + {task.e_proc_dly:.6f} = {comp_dlys:.6f}")
                    gp.trace_print(f"  new_queue_time(new) = old_nt + e_proc_dly")
                    gp.trace_print(f"                     = {old_new_queue_time:.6f} + {task.e_proc_dly:.6f} = {new_queue_time:.6f}")
                    gp.trace_print(f"  total_comp(new) = {self.total_comp_time - task.e_proc_dly:.6f} + {task.e_proc_dly:.6f} = {self.total_comp_time:.6f}")
                

            self.old_comp_times.append(task.e_proc_dly)
            tail = self.old_comp_times[-self.statSlotNum:]
            self.avg_edge_time = sum(tail) / len(tail) if tail else 0

        # Single FIFO queue evolution (scalar)
        if t_id % self.gen_task_cycle == self.start_slot:
            old_edge_ql = self.edge_queue_time_ql
            self.edge_queue_time_ql = max(
                self.edge_queue_time_ql +
                self.edge_act_queue_growth_rate * (total_comp_need - total_comp_used), 0
            )
            self.new_edge_ql_change = self.edge_act_queue_growth_rate * (total_comp_need - total_comp_used)

            self.old_virtual_edge_queue_time_ql = self.virtual_edge_queue_time_ql
            EPS = 1e-8
            if self.avg_edge_time > EPS:
                vir_bl = self.edge_queue_time_ql / self.avg_edge_time * self.delta * self.gen_task_cycle / self.avg_device_num
                self.virtual_edge_queue_time_ql = max(
                    self.virtual_edge_queue_time_ql + self.edge_vir_queue_growth_rate * (
                        vir_bl - self.edge_dly_adj_val
                    ), 0
                )
                self.new_vir_edge_ql_change = self.edge_vir_queue_growth_rate * (
                    vir_bl - self.edge_dly_adj_val
                )

            if gp.settings.enable_trace:
                gp.trace_print(f"\n  --- Edge queue update (gen cycle boundary) ---")
                gp.trace_print(f"  total_comp_need = {total_comp_need:.6f}, total_comp_used = {total_comp_used:.6f}")
                gp.trace_print(f"  edge_queue_time_ql = max(0, old_ql + growth_rate * (need - used))")
                gp.trace_print(f"                     = max(0, {old_edge_ql:.6f} + {self.edge_act_queue_growth_rate} * ({total_comp_need:.6f} - {total_comp_used:.6f}))")
                gp.trace_print(f"                     = max(0, {old_edge_ql + self.edge_act_queue_growth_rate * (total_comp_need - total_comp_used):.6f}) = {self.edge_queue_time_ql:.6f}")
                gp.trace_print(f"  avg_edge_time = {self.avg_edge_time:.6f} (tail of {len(tail) if tail else 0} samples)")
                if self.avg_edge_time > EPS:
                    gp.trace_print(f"  vir_bl = edge_ql / avg_edge_time * delta * gen_cycle / avg_device_num = {self.edge_queue_time_ql:.6f} / {self.avg_edge_time:.6f} * {self.delta} * {self.gen_task_cycle} / {self.avg_device_num} = {vir_bl:.6f}")
                    gp.trace_print(f"  virtual_edge_queue_time_ql = max(0, {self.old_virtual_edge_queue_time_ql:.6f} + {self.edge_vir_queue_growth_rate} * ({vir_bl:.6f} - {self.edge_dly_adj_val:.6f})) = {self.virtual_edge_queue_time_ql:.6f}")

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
