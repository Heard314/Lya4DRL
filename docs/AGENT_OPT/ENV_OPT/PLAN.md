# PLAN — 多服务器场景单 FIFO 队列架构方案（v3）

> 输入：`docs/AGENT_OPT/ENV_OPT/DEFINE.md`（已确认）
> 前一版本：v2 per-device + r_e（已废弃）
> 状态：已实现

---

## 1. 整体架构

```mermaid
graph TD
    subgraph "Inference (CTDE 执行端)"
        A[MECEnv] -->|device_obs[i] + edge_obs| B[DeviceAgent i]
        B -->|act: S+3*ae_dim| A
    end

    subgraph "MECEnv 核心（v3 简化）"
        A -->|server_id = argmax[:S]| C[DeviceEnv[i].compute]
        C -->|tasks tagged| D{Router}
        D -->|per server| E0["EdgeEnv[0] — 1 FIFO queue"]
        D -->|per server| E1["EdgeEnv[1] — 1 FIFO queue"]
        D -->|per server| E2["EdgeEnv[2] — 1 FIFO queue"]
    end

    subgraph "EdgeEnv v3"
        E0 -->|FIFO| G["单队列演化 (标量)"]
        E1 -->|FIFO| G
        E2 -->|FIFO| G
    end

    subgraph "Training (CTDE 训练端)"
        H[EdgeAgent] -->|holds| I["×10 PolicyNets (S+3*ae_dim)"]
        H -->|holds| J["×3 ValueNets"]
        J -->|Critic input| K["state: N_k*14 + joint_act: N_k*(S+3)"]
    end

    style D fill:#f96
    style G fill:#f96
```

**v2→v3 核心回退：**
1. EdgeEnv 从 N=10 个 per-device 队列 → **1 个 FIFO 队列**
2. 去除 `r_e_logit` 动作变量（及 per-server softmax）
3. edge_obs 从 60 维 → **6 维**（2×S，所有设备共享）
4. 动作空间从 4 连续变量 → **3 连续变量**

---

## 2. 核心接口

| 功能 | 接口 | DEFINE 映射 |
|------|------|------------|
| F1 多服务器环境 | `MECEnv` 持有 `List[EdgeEnv]` × S | F1 |
| F2 单 FIFO 队列 | `EdgeEnv` 1 个队列（标量），FIFO 处理 | F2 |
| F3 简化动作空间 | `PolicyNet` 输出 `S+3*ae_dim`；env_act 4 维 | F3 |
| F4 观测简化 | `DeviceEnv.get_obs`: 5+S; `EdgeEnv.get_obs`: 2 | F4 |
| F5 队列演化 | `EdgeEnv.compute` 单队列演化，全频处理 | F5 |
| F6 神经网络 | 输入/输出维度参数化 | F6 |
| F7 奖励 | 边缘队列奖励按服务器级别计算 | F7 |

---

## 3. 模块设计

### 3.1 `config/params.py`

**维度派生：**

```python
S = edge_server_num  # 3
ae_dim = action_encode_dim  # 10 (configurable)

# MAPPO
ppo_device_obs_dim      = 5 + S             # 8
ppo_edge_queue_obs_dim  = 2 * S             # 6 (per-server stride)
ppo_policy_input_dim    = ppo_device_obs_dim + ppo_edge_queue_obs_dim  # 14
ppo_action_dim          = S + 3 * ae_dim    # 33
ppo_value_input_dims    = [n * 14 + n * (S+3) for n in [2,4,4]]
                        # [40, 80, 80]

# MADDPG
device_obs_dim          = 5 + S             # 8
edge_queue_obs_dim      = 2 * S             # 6
policy_input_dim        = 14
action_dim              = S + 3 * ae_dim    # 33
value_input_obs_dims    = [n * 14 for n in [2,4,4]]
                        # [28, 56, 56]
value_input_act_dims    = [n * (S+3) for n in [2,4,4]]
                        # [12, 24, 24]
value_input_dims        = [40, 80, 80]
```

### 3.2 `env/edge_env.py` — 结构性简化

**从 N=10 per-device 队列 → 1 个 FIFO 队列：**

```python
class EdgeEnv:
    def __init__(self, gen_params, writer=None):
        self.device_num = gen_params.device_num
        self.queue_num = 1  # 单队列

        # 标量队列
        self.edge_queue_time_ql = 0.0
        self.virtual_edge_queue_time_ql = 0.0
        self.old_edge_queue_time_ql = 0.0
        self.old_virtual_edge_queue_time_ql = 0.0
        self.new_edge_ql_change = 0.0
        self.new_vir_edge_ql_change = 0.0
        self.total_comp_time = 0.0
        self.total_comp_amount = 0.0
        self.avg_edge_time = 0.0
        self.old_comp_times = []

        # 单队列的时延调整值
        self.edge_dly_adj_val = min(gen_params.comp_dly_thre) * gen_params.edge_dly_adj_fac[0]

    def get_obs(self):
        return [
            self.edge_queue_time_ql,
            self.virtual_edge_queue_time_ql if self.enable_virtual_queue_reward else -1.0
        ]  # 2 dims

    def compute(self, tasks, e_id, t_id, visualize=False):
        # tasks: List[Task] — 该服务器上所有设备调度来的任务（已合并）
        # 按 trans_time 排序（FIFO 到达顺序）
        # 全频处理：edge_comp_freq，无 CPU 分配
        # 单队列演化公式（标量）
```

### 3.3 `env/device_env.py`

**改动：** 动作解析从 5 维 → 4 维，去掉 r_e_logit。

```python
def compute(self, act, e_id, t_id, visualize=False):
    server_id = int(act[0])
    offl_rto = np.clip(act[1], 0.0, 1.0)
    trpw_rto = np.clip(act[2], 0.6, 1.0)
    comp_rto = np.clip(act[3], 0.6, 1.0)
    # 无 r_e_logit
```

### 3.4 `env/mec_env.py` — 核心简化

**去除：** per-server softmax、r_e_values 传递。

**edge_obs 组织（按 server，所有设备共享）：**

```python
# step() 中：
# 1. 设备本地计算（不变）
# 2. 路由任务到服务器（不变，但不传 r_e_values）
# 3. 边缘服务器计算
for s in range(self.edge_server_num):
    server_tasks = [tasks from devices that selected server s]
    self.edge_envs[s].compute(server_tasks, e_id, t_id, visualize)

# 4. edge_obs: 所有设备共享
next_edge_obs = []
for edge_env in self.edge_envs:
    next_edge_obs.extend(edge_env.get_obs())  # +2 dims each
# next_edge_obs: [s0_act, s0_vir, s1_act, s1_vir, s2_act, s2_vir] (6 dims)
```

**边缘奖励（per-type，聚合所有服务器）：**

```python
for i in range(edge_queue_num):  # type index
    for s in range(self.edge_server_num):
        edge_env = self.edge_envs[s]
        if enable_virtual_queue_reward:
            edge_queue_virtual_rewards[i] += weight * \
                edge_env.virtual_edge_queue_time_ql * edge_env.new_vir_edge_ql_change
        if enable_actual_queue_reward:
            edge_queue_actual_rewards[i] += weight * \
                edge_env.edge_queue_time_ql * edge_env.new_edge_ql_change
```

### 3.5 `network/policy_net.py`

**act_scale/loc**：去掉 r_e 维度。

```python
ae_dim = alg_params.action_encode_dim
S = alg_params.action_dim - 3 * ae_dim  # 33 - 30 = 3
low  = torch.tensor([-1.]*S + [0.6]*ae_dim + [0.6]*ae_dim + [0.6]*ae_dim)
high = torch.tensor([1.]*S + [1.0]*ae_dim + [1.0]*ae_dim + [1.0]*ae_dim)
```

### 3.6 `agent/device_agent.py`

**env_act 从 5 维 → 4 维：**

```python
# MAPPO / MADDPG
server_id = int(np.argmax(act[:S]))
# Average each ae_dim block for 3 continuous actions
cont_vals = []
for j in range(3):
    block = act[S + j*ae_dim : S + (j+1)*ae_dim]
    cont_vals.append(sum(block) / ae_dim)
env_act = [float(server_id)] + cont_vals  # 4 dims
```

**基线策略：** 去掉 r_e 返回值。

### 3.7 `agent/edge_agent.py`

**MADDPG joint_act 压缩**：`S+3*ae_dim → S+3`。

```python
compressed_dim = S + 3  # per-device compressed for Critic
# Compress: server_logits(S) + 3 averaged continuous
cont_all = batch_acts[:, S:S + 3*ae_dim]
cont_blocks = cont_all.reshape(-1, 3, ae_dim)
cont_compressed = cont_blocks.mean(dim=-1)
compressed_act = torch.cat([server_logits, cont_compressed], dim=-1)  # [B, S+3]
```

### 3.8 `rollout.py`

**edge_obs 简化**：不再按 device 组织，直接按 server 拼接。

```python
# initialization:
edge_obs = []
for edge_env in self.mec_env.edge_envs:
    edge_obs.extend(edge_env.get_obs())  # 2*S dims

# policy input (per device): device_obs + ALL edge_obs (shared)
device_value_obs = concatenate(device_obss[i], edge_obs)  # (5+S) + (2*S) = 14
```

**env_act_dim**：`S + 1 = 4`（was S+2=5）。

### 3.9 `util/replay_buffer.py`

**MAPPO `package_value_inputs`**：压缩 joint_act 从 `S+3*ae_dim → S+3`。

**MADDPG `sample()`**：joint_act 压缩从 `S+3*ae_dim → S+3`（去掉 r_e 块）。

**MAPPO `get_policy_net_training_data`**：`action_dim = S+3*ae_dim`。

### 3.10 `util/utils.py`

**ObsScaling edge_rms**：`edge_queue_obs_dim * device_num`（保持不变，仍是 6×10=60 但语义变了）。

---

## 4. 维度汇总表

| 常量 | v2 (per-device+r_e) | v3 (单FIFO无r_e) |
|------|---------------------|-------------------|
| `ppo_action_dim` | S+4*ae=43 | S+3*ae=**33** |
| `action_dim` (MADDPG) | S+4*ae=43 | S+3*ae=**33** |
| `env_act_dim` | S+2=5 | S+1=**4** |
| `ppo_value_input_dims` | [42,84,84] | [**40,80,80**] |
| `value_input_dims` (MADDPG) | [42,84,84] | [**40,80,80**] |
| edge_obs 组织 | by-device (60 dims) | **by-server (6 dims)** |
| policy_input 组织 | device_obs + own edge queues | **device_obs + all edge_obs** |

---

## 5. 改动量估计

| 模块 | 改动量 | 复杂度 |
|------|--------|--------|
| `env/edge_env.py` | ~80 行 | **高** — N队列→单队列回退 |
| `env/mec_env.py` | ~50 行 | 中 — 去softmax, edge_obs简化 |
| `config/params.py` | ~15 行 | 低 — 公式更新 |
| `agent/device_agent.py` | ~20 行 | 低 — 动作维度减1 |
| `rollout.py` | ~30 行 | 中 — edge_obs组织简化 |
| `util/replay_buffer.py` | ~20 行 | 低 — 压缩维度调整 |
| `network/policy_net.py` | ~10 行 | 低 |
| `agent/edge_agent.py` | ~10 行 | 低 |
| `util/utils.py` | ~5 行 | 低 |

**总计：~240 行，中等复杂度**（主要是 EdgeEnv 回退和 MECEnv 简化）

---

## 6. 关键风险

| 风险 | 缓解 |
|------|------|
| 单队列无法区分设备 | 策略网络仍能通过 device_obs 中的队列信息做差异化卸载决策 |
| edge_obs 从 60→6 维信息量骤降 | 6 维反映的是服务器整体负载，足够做路由决策 |
| FIFO 不区分任务紧急度 | 用户确认 FIFO 即可 |
| env_act_dim 回退破坏 roll-out | 参数化引用，一处修改全局生效 |
