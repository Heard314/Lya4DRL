# DEFINE — 多服务器场景扩展需求分析（v3）

> 输入：`docs/AGENT_OPT/ENV_OPT/SPEC.md`（修订版 — 再次更改服务器计算模型）
> 前一版本：v2 per-device 队列 + r_e 资源分配（已废弃）
> 状态：已实现

---

## 1. SPEC 变更摘要

v3 核心变化（相对于 v2 per-device + r_e）：

| 项目 | v2 | v3 |
|------|----|----|
| 每服务器队列数 | N 个（per-device） | **1 个 FIFO** |
| CPU 分配 | `r_{e,i}` softmax | **无（全频处理）** |
| 动作空间 | `[srv, offl, trpw, comp, r_e]` | **`[srv, offl, trpw, comp]`** |
| 队列演化 | per-device 公式 | **单队列标量公式** |

---

## 2. 功能需求

### F1. 多服务器环境（不变）

S=3 同构边缘服务器，均匀分布于距原点 250m 圆周。

### F2. 服务器计算模型 — 单 FIFO 队列

每台服务器维护 **1 个 FIFO 计算队列**，所有选择该服务器的设备任务按到达顺序处理。

- 无 CPU 资源分配变量，服务器以全频 `edge_comp_freq` 处理任务
- 队列长度和虚拟队列均为 **标量**（非 per-device 列表）

### F3. 设备动作空间 — 去掉 r_e

设备动作 = `[server_id, offl_rto, trpw_rto, comp_rto]`（4 维 env_act）：
- **server_id**: 选择卸载到哪个服务器（S 选 1）
- **offl_rto / trpw_rto / comp_rto**: 3 个连续动作
- ~~r_e_logit~~: 去除

策略网络输出：
- MAPPO: `S + 3*ae_dim` 维（ae_dim 为可配置编码维度）
- MADDPG: 同上

### F4. 观测空间简化

- **设备观测** `5 + S` 维（不变）：S 个 trans_rate + time_ql + virtual_time_ql + 3 task fields
- **边缘观测** `2 × S` 维：每服务器 2 个标量（actual_queue + virtual_queue），所有设备共享

边缘观测不再按 device 组织，直接按 server 拼接：

```
[s0_act_ql, s0_vir_ql, s1_act_ql, s1_vir_ql, s2_act_ql, s2_vir_ql]
```

### F5. 边缘队列演化公式 — 单队列标量

```python
self.edge_queue_time_ql = max(self.edge_queue_time_ql
    + edge_act_queue_growth_rate * (total_comp_need - total_comp_used), 0)

self.virtual_edge_queue_time_ql = max(self.virtual_edge_queue_time_ql
    + edge_vir_queue_growth_rate * (ql/avg_time * delta * cycle - edge_dly_adj_val), 0)
```

其中 `edge_dly_adj_val` 为标量，取最紧时延约束 × edge_dly_adj_fac：`min(comp_dly_thre) * edge_dly_adj_fac[0]`。

### F6. 神经网络适配

- **策略网络输入** `5 + 3S` 维（不变）：device_obs + 所有 S 个服务器的边缘队列
- **策略网络输出**: `S + 3*ae_dim` 维
- **价值网络**: MAPPO 输入包含同类型所有设备的 device_obs + 所有边缘队列 + 压缩联合动作
- MADDPG Critic joint_act 使用 `S+3` 压缩形式

### F7. 奖励函数适配

边缘队列奖励按 **服务器级别** 计算（非 per-device），同一服务器的所有设备共享同一队列奖励。

---

## 3. 架构约束

- CTDE 训练范式不变
- Lyapunov 队列框架不变（公式改为单队列标量）
- MAPPO/MADDPG 代码复用
- 评估模式兼容
- `action_encode_dim` 可配置（默认 10）

---

## 4. 与 v2 的关键差异

| 差异点 | v2 | v3 | 影响 |
|--------|----|----|------|
| 队列结构 | N×S 个队列 | S 个队列 | EdgeEnv 大幅简化 |
| CPU 分配 | per-server softmax | 无 | 去掉 softmax 逻辑 |
| 动作变量数 | 4 连续 + S logits | **3 连续 + S logits** | 动作空间缩小 |
| edge_obs 维度 | 60 维 | **6 维** | Critic 输入大幅减小 |
| edge_obs 组织 | 按 device | **按 server** | rollout/replay 简化 |
| r_e 约束 | Σr=1 per server | N/A | 去掉 |

---

## 5. 交付物清单

| 文件 | 改动级别 | 说明 |
|------|----------|------|
| `config/params.py` | 中 | 维度公式更新：S+3*ae_dim，value_input 重算 |
| `env/edge_env.py` | **高** | 结构性回退：N 队列 → 1 FIFO 队列 |
| `env/mec_env.py` | 中 | 去掉 softmax，edge_obs 按 server 组织，奖励计算调整 |
| `env/device_env.py` | 低 | 去掉 r_e 相关 |
| `agent/device_agent.py` | 中 | 动作输出改为 4 维 env_act |
| `network/policy_net.py` | 低 | act_scale/loc 去掉 r_e 维度 |
| `agent/edge_agent.py` | 低 | joint_act 压缩为 S+3 |
| `rollout.py` | 中 | edge_obs 组织简化，动作维度 |
| `util/replay_buffer.py` | 低 | 维度适配 |
| `util/utils.py` | 低 | edge_rms 维度更新 |

---

## 6. 风险点

| 风险 | 影响 | 缓解 |
|------|------|------|
| 单队列 FIFO 不区分任务优先级 | 紧急任务可能排在普通任务后面 | 确认使用严格 FIFO，不做优先级调度 |
| edge_obs 从 60 维降到 6 维 | Critic 信息量大减，可能影响训练效果 | 每设备还附加了 device_obs 在策略输入中，总信息量仍足够 |
