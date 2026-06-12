# 扩展到多服务器场景
现有算法针对单服务器场景进行设计，现扩展到多服务器场景，需要你对方案进行设计。我考虑了几个需要考虑设计点，供你参考，但是完善的方案需要你来完成。

## 设备端的动作选择
动作选择需要再扩展一个，即选择卸载到哪一个服务器

## 神经网络输入信息的设计
服务器端的critic网络的输入状信息应该包含哪些内容？现在既需要服务于多设备，又需要考虑多设备之间的竞争。

## 更改服务器计算模型
原来方案中，每个服务器的计算队列有3个计算队列（对应三个任务类型），先设置N个计算队列（对应于N个终端设备），每个计算队列处理单个终端设备的任务。
然后在每个时隙，每台设备添加一维服务器计算资源分配比例的决策变量r_{e,i}，且$\sum_{i=1}^N r_{e,i} = 1$，即所有设备的计算资源分配比例之和为1。
此外，修改服务器的队列模型edge_queue_time_ql和virtual_edge_queue_time_ql的演化公式为：
self.edge_queue_time_ql[i] = max(self.edge_queue_time_ql[i] + edge_act_queue_growth_rate * (total_comp_need_time_this_epi - total_comp_used_time_this_epi[i]), 0)
self.virtual_edge_queue_time_ql[i] = max(self.virtual_edge_queue_time_ql[i] + edge_vir_queue_growth_rate*(self.edge_queue_time_ql[i]/self.avg_edge_time[i]*self.delta*self.gen_task_cycle - self.edge_dly_adj_val[device_type]), 0)

## 再次更改服务计算模型
原来方案中，每个服务器的计算队列有3个计算队列（对应三个任务类型），现只设置1个计算队列，按FIFO的方式处理所有到达的终端设备任务。
去除决策变量中的服务器计算资源分配比例的决策变量r_{e,i}，服务器的队列模型edge_queue_time_ql和virtual_edge_queue_time_ql的演化公式为：
self.edge_queue_time_ql = max(self.edge_queue_time_ql + edge_act_queue_growth_rate * (total_comp_need_time_this_epi - total_comp_used_time_this_epi), 0)
self.virtual_edge_queue_time_ql = max(self.virtual_edge_queue_time_ql + edge_vir_queue_growth_rate*(self.edge_queue_time_ql/self.avg_edge_time*self.delta*self.gen_task_cycle - self.edge_dly_adj_val), 0)