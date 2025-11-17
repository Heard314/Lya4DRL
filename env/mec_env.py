from env import device_env
from env.device_env import DeviceEnv
from env.edge_env import EdgeEnv
import torch
import math
import config.global_params as gp
class MECEnv():
    def __init__(self, gen_params,time_slots):
        self.device_num = gen_params.device_num
        self.expense_weights = gen_params.expense_weights
        self.energy_weights = gen_params.energy_weights
        self.time_slots = time_slots
        self.device_type_num = gen_params.device_type_num
        # edge env
        self.edge_env = EdgeEnv(gen_params)
        # device envs
        self.device_envs = []
        for i in range(self.device_num):
            self.device_envs.append(DeviceEnv(i, gen_params, self.edge_env))
        # self.lyaV = gen_params.lyaV
        self.local_reward_weight = gen_params.local_reward_weight
        self.edge_reward_weight = gen_params.edge_reward_weight
    def reset(self):
        edge_obs = self.edge_env.reset()
        
        device_obss = [None for i in range(self.device_num)]
        for i in range(self.device_num):
            device_obss[i] = self.device_envs[i].reset()
        
        return edge_obs, device_obss
    
    def step(self, device_acts, e_id, t_id):

        if e_id % 50 == 1:
            gp.settings.enable_print = True
        else:
            gp.settings.enable_print = False
        enable_print = gp.settings.enable_print
        # 首先每个设备对待执行任务做出卸载决策，然后执行任务的本地计算部分，返回远程卸载部分（以下代码中的sched_tasks）
        device_sched_tasks = [None for i in range(self.device_num)]
        for i in range(self.device_num):
            sched_tasks = self.device_envs[i].compute(device_acts[i])
            device_sched_tasks[i] = sched_tasks
        # 边缘服务器执行任务的远程卸载部分
        self.edge_env.compute(device_sched_tasks)
        
        # reward
        device_rewards = [0 for i in range(self.device_num)]
        edge_queue_num = self.device_type_num
        edge_rewards = [0 for i in range(edge_queue_num)]
        device_costs = [0 for i in range(self.device_num)]
        device_comp_dlys = [0 for i in range(self.device_num)]
        device_csum_engys = [0 for i in range(self.device_num)]
        device_comp_expns = [0 for i in range(self.device_num)]
        device_overtime_nums = [0 for i in range(self.device_num)]
        # device_delay_adjust_coefs = [1.0 for i in range(self.device_num)]
        device_task_is_available = [False for i in range(self.device_num)] #在该时间间隙下是否有任务到达
        for i in range(self.device_num):
            sched_tasks = device_sched_tasks[i]
            task_num = len(sched_tasks)
            # if(enable_print): print(f"[DEBUG] In device {i}, the task number is {task_num}")
            device_type = self.device_envs[i].device_type
            device_task_is_available[i] = task_num>=1
            for j in range(task_num):
                task = sched_tasks[j]
                comp_dly = max(task.l_comp_dly, task.e_comp_dly)
                if(enable_print): print(f"[DEBUG] the comp_dly in device {i} is {comp_dly}")
                device_comp_dlys[i] += 1 / (j + 1) * (comp_dly - device_comp_dlys[i])
                
                csum_engy = task.l_csum_engy + task.e_csum_engy
                device_csum_engys[i] += 1 / (j + 1) * (csum_engy - device_csum_engys[i])
                comp_expn = task.comp_expn
                device_comp_expns[i] += 1 / (j + 1) * (comp_expn - device_comp_expns[i])
                
                device_costs[i] += self.energy_weights[device_type] * csum_engy + \
                                   self.expense_weights[device_type] * comp_expn
                
                # if t_id % 20 == 0 and e_id % 20 == 0 and j == 0:
                #     print("[DEBUG] The device index is: ", i)
                    # print("[DEBUG] The task", j ,"'s dly_cons is: ", task.dly_cons, " The comp_dly is: ", comp_dly)
                    # print("[DEBUG] The task", j ,"'s norm_csum_engy is: ", task.norm_csum_engy, " The csum_engy is: ", csum_engy)
                    # print("[DEBUG] The task", j ,"'s norm_comp_expn is: ", task.norm_comp_expn, " The comp_expn is: ", comp_expn)
                    # print("[DEBUG] The device", i, "'s virtual comp ql is: ", self.device_envs[i].virtual_comp_ql)
                    # print("[DEBUG] The device", i, "'s completed comp is: ", self.device_envs[i].completed_comp)
                # 计算超时惩罚，其中task.dly_cons是按照设备计算能力为2Gcycles/s计算的，实际的设备计算能力在2.1~2.4Gcycles/s之间
                if comp_dly > task.dly_cons:
                    #! 考虑到每个任务的超时程度会影响到任务的执行效果，在原有惩罚的基础上多乘一个log函数（表示超时程度）
                    device_rewards[i] += -5000 * torch.log(torch.exp(torch.tensor(1.0)) -1.0 + comp_dly / task.dly_cons)
                    # device_rewards[i] += -5000
                    device_overtime_nums[i] += 1
                    #! 当设备i超时严重时，其他设备的动态时间阈值调整系数应适当增大
                    # if t_id % 20 == 0 and e_id % 20 == 0 and j == 0:
                    #     print("[DEBUG] The ratio of comp_dly to task.dly_cons in device", i, "is: ", comp_dly / task.dly_cons)
                    # if comp_dly > 1.5 * task.dly_cons:
                    #     for k in range(self.device_num):
                    #         if k == i:
                    #             continue
                    #         device_delay_adjust_coefs[k] = max(device_delay_adjust_coefs[k], comp_dly / task.dly_cons)
                else:
                    norm_csum_engy = task.norm_csum_engy
                    norm_comp_expn = task.norm_comp_expn
                    # 奖励函数 能耗比重 *实际总能耗 / 标准化能耗 + 成本比重 *实际总成本 / 标准化成本
                    device_rewards[i] += -1000 * (self.energy_weights[device_type] * 
                                                  csum_engy / norm_csum_engy +
                                                  self.expense_weights[device_type] * 
                                                  comp_expn / norm_comp_expn)
                if(enable_print): print(f"[DEBUG] The device", i, "'s navie reward is: ", device_rewards[i])
                device_rewards[i] = device_rewards[i] + \
                                    self.local_reward_weight * \
                                    (self.device_envs[i].comp_ql * (self.device_envs[i].new_local_comp - self.device_envs[i].completed_comp) + \
                                    self.device_envs[i].virtual_comp_ql * (self.device_envs[i].virtual_comp_ql_growth - self.device_envs[i].completed_comp))
                if(enable_print): print(f"[DEBUG] The device", i, "'s old_comp_ql is: ", self.device_envs[i].old_comp_ql)
                if(enable_print): print(f"[DEBUG] The device", i, "'s old_virtual_comp_ql is: ", self.device_envs[i].old_virtual_comp_ql)
                if(enable_print): print(f"[DEBUG] The device", i, "'s new_local_comp is: ", self.device_envs[i].new_local_comp)
                if(enable_print): print(f"[DEBUG] The device", i, "'s completed_comp is: ", self.device_envs[i].completed_comp)
                if(enable_print): print(f"[DEBUG] The device", i, "'s virtual_comp_ql_growth is: ", self.device_envs[i].virtual_comp_ql_growth)
                if(enable_print): print(f"[DEBUG] The device", i, "'s comp_ql is: ", self.device_envs[i].comp_ql)
                if(enable_print): print(f"[DEBUG] The device", i, "'s virtual_comp_ql is: ", self.device_envs[i].virtual_comp_ql)
                if(enable_print): print(f"[DEBUG] The device", i, "'s device_reward is: ", self.local_reward_weight * \
                                    (self.device_envs[i].comp_ql * (self.device_envs[i].new_local_comp - self.device_envs[i].completed_comp) + \
                                    self.device_envs[i].virtual_comp_ql * (self.device_envs[i].virtual_comp_ql_growth - self.device_envs[i].completed_comp)))
        

        for i in range(edge_queue_num):
            edge_rewards[i] = self.edge_reward_weight * \
                            (self.edge_env.edge_queue_comp_ql[i] * (self.edge_env.new_edge_comp[i] - self.edge_env.completed_comp[i]))
                            # + self.edge_env.virtual_edge_queue_comp_ql[i] * (self.edge_env.virtual_edge_comp_ql_growth[i] - self.edge_env.completed_comp[i]))
            
            if(enable_print): print(f"[DEBUG] The edge_queue", i, "'s old_edge_queue_comp_ql is: ", self.edge_env.old_edge_queue_comp_ql[i])
            if(enable_print): print(f"[DEBUG] The edge_queue", i, "'s old_virtual_edge_queue_comp_ql is: ", self.edge_env.old_virtual_edge_queue_comp_ql[i])
            if(enable_print): print(f"[DEBUG] The edge_queue", i, "'s new_edge_comp is: ", self.edge_env.new_edge_comp[i])
            if(enable_print): print(f"[DEBUG] The edge_queue", i, "'s completed_comp is: ", self.edge_env.completed_comp[i])
            if(enable_print): print(f"[DEBUG] The edge_queue", i, "'s virtual_edge_comp_ql_growth is: ", self.edge_env.virtual_edge_comp_ql_growth[i])
            if(enable_print): print(f"[DEBUG] The edge_queue", i, "'s edge_queue_comp_ql is: ", self.edge_env.edge_queue_comp_ql[i])
            if(enable_print): print(f"[DEBUG] The edge_queue", i, "'s virtual_edge_queue_comp_ql is: ", self.edge_env.virtual_edge_queue_comp_ql[i])
            if(enable_print): print(f"[DEBUG] The edge_queue", i, "'s edge_reward is: ", self.edge_reward_weight * \
                            (self.edge_env.edge_queue_comp_ql[i] * (self.edge_env.new_edge_comp[i] - self.edge_env.completed_comp[i])))
                            # + self.edge_env.virtual_edge_queue_comp_ql[i] * (self.edge_env.virtual_edge_comp_ql_growth[i] - self.edge_env.completed_comp[i])))
        joint_reward = sum(device_rewards) + sum(edge_rewards)
        joint_cost = sum(device_costs)
        
        # next obs
        next_edge_obs = self.edge_env.get_obs()
        next_device_obss = [None for i in range(self.device_num)]
        for i in range(self.device_num):
            next_device_obss[i] = self.device_envs[i].get_obs()
        
        # #! 更新其他设备的动态时间阈值调整系数
        # for i in range(self.device_num):
        #     self.device_envs[i].adjust_delay_threshold_coef(device_delay_adjust_coefs[i])

        return joint_reward, device_rewards, \
               joint_cost, device_costs, \
               device_comp_dlys, device_csum_engys, \
               device_comp_expns, device_overtime_nums, \
               next_edge_obs, next_device_obss,device_task_is_available