# CON_PLAN — 单 FIFO 队列编码级详细设计（v3）

> 输入：`docs/AGENT_OPT/ENV_OPT/PLAN.md`
> 版本：v3 — 单 FIFO 队列，无 r_e
> 状态：已实现

---

## 1. 数据实体变更

### 1.1 设备观测 `device_obs_dim = 5 + S = 8`

| idx | 字段 | 维度 |
|-----|------|------|
| 0~S-1 | max_trans_rate[s] | S=3 |
| S | time_ql | 1 |
| S+1 | virtual_time_ql | 1 |
| S+2 | data_size | 1 |
| S+3 | comp_dens | 1 |
| S+4 | dly_cons | 1 |

### 1.2 边缘观测 `edge_queue_obs_dim = 2 × S = 6`

所有设备共享。每 EdgeEnv.get_obs() 返回 2 维 `[act_ql, vir_ql]`，S=3 个服务器拼接：

```
[s0_act_ql, s0_vir_ql, s1_act_ql, s1_vir_ql, s2_act_ql, s2_vir_ql]
```

### 1.3 动作空间

| 层 | 维度 | 结构 |
|----|------|------|
| Policy 输出 | `S + 3*ae_dim = 33` | `[s_logit×3, offl×ae, trpw×ae, comp×ae]` |
| 环境用 env_act | `S + 1 = 4` | `[server_id, offl, trpw, comp]` |
| Buffer 存储 | 33 维 | 完整 policy 输出 |
| Critic joint_act | `S + 3 = 6` | 压缩：S logits + 3 连续（取 ae 平均） |

### 1.4 动作范围（act_scale/loc，33 维）

| 维度 | ae 块 | scale | loc | 范围 |
|------|-------|-------|-----|------|
| 0~S-1 | — | 1 | 0 | [-1, 1] |
| S~S+ae-1 | offl (ae_dim) | 0.2 | 0.8 | [0.6, 1.0] |
| S+ae~S+2*ae-1 | trpw (ae_dim) | 0.2 | 0.8 | [0.6, 1.0] |
| S+2*ae~S+3*ae-1 | comp (ae_dim) | 0.2 | 0.8 | [0.6, 1.0] |

### 1.5 边缘队列统计量

| 统计量 | 类型 | 说明 |
|--------|------|------|
| `edge_queue_time_ql` | `float` | 单队列实际积压 |
| `virtual_edge_queue_time_ql` | `float` | 单队列虚拟积压 |
| `avg_edge_time` | `float` | 单队列平均处理时间 |
| `total_comp_time` | `float` | 累计处理时间 |
| `edge_dly_adj_val` | `float` | `min(comp_dly_thre) * edge_dly_adj_fac[0]` |

---

## 2. 维度汇总

```python
S = 3       # edge_server_num
ae_dim = 10 # action_encode_dim (configurable)

# ===== 观测维度 =====
device_obs_dim     = 5 + S           # 8
edge_queue_obs_dim = 2 * S           # 6
policy_input_dim   = 8 + 6          # 14

# ===== 动作维度 =====
action_dim      = S + 3 * ae_dim    # 33
env_act_dim     = S + 1             # 4
compressed_dim  = S + 3             # 6 (per-device for Critic)

# ===== MAPPO =====
ppo_value_input_dims = [n*14 + n*6 for n in [2,4,4]]
                     # [40, 80, 80]

# ===== MADDPG =====
value_input_obs_dims = [n*14 for n in [2,4,4]]
                     # [28, 56, 56]
value_input_act_dims = [n*6 for n in [2,4,4]]
                     # [12, 24, 24]
value_input_dims     = [40, 80, 80]
```

---

## 3. Value Net 与 Policy Net 输入输出详解

### 3.1 Policy Net（MappoPolicyNetLSTM / MaddpgPolicyNetLSTM）

**结构相同，均为：**

```
输入: policy_input (14 维)
  ├── device_obs (8 维): [trans_rate×3, time_ql, vir_ql, data_size, comp_dens, dly_cons]
  └── edge_obs   (6 维): [s0_act, s0_vir, s1_act, s1_vir, s2_act, s2_vir]

前向传播:
  fc1(14→400) → tanh → LSTM(400→400) → tanh → mu_head(400→33)

输出: action_mean + action_std (各 33 维)
  ├── [0:3]    = 3  server logits (tanh → scale*1 + 0 = [-1,1])
  ├── [3:13]   = 10 offl dims   (tanh → scale*0.2 + 0.8 = [0.6,1.0])
  ├── [13:23]  = 10 trpw dims   (tanh → scale*0.2 + 0.8 = [0.6,1.0])
  └── [23:33]  = 10 comp dims   (tanh → scale*0.2 + 0.8 = [0.6,1.0])
```

**choose_action 压缩：**
```python
server_id = argmax(mean[:3])
offl = mean(mean[3:13])    # 10-dim → scalar
trpw = mean(mean[13:23])
comp = mean(mean[23:33])
env_act = [server_id, offl, trpw, comp]  # 4 dims
```

### 3.2 MAPPO Value Net（MappoValueNet，per type）

**输入结构（以 type 0, n_0=2 为例，40 维）：**

```
value_input (40 维) = obs_part (28 维) + joint_act_part (12 维)

obs_part (28 维) — 2 devices × 14 维:
  Device 0:
    ├── edge_obs (6 维, 全局共享)
    └── device_obs[0] (8 维)
  Device 1:
    ├── edge_obs (6 维, 全局共享, 同上)
    └── device_obs[1] (8 维)

joint_act_part (12 维) — 2 devices × 6 维 (压缩):
  Device 0:
    ├── server_logits[0] (3 维)
    ├── offl_avg[0]       (1 维, 10-dim mean)
    ├── trpw_avg[0]       (1 维)
    └── comp_avg[0]       (1 维)
  Device 1: (同上结构, 6 维)
```

**前向传播：**
```
输入 (40 维) → fc1(40→400) → tanh → fc2(400→400) → tanh → fc3(400→1)
输出: scalar V-value
```

**Type 1/2 (n=4, 80 维)：** 同上结构 × 4 devices。

### 3.3 MADDPG Value Net（MaddpgValueNet，per type）

**输入结构（以 type 0 为例，40 维）：**

```
state (28 维) + joint_act (12 维) → concat → fc1(40→200) → tanh → fc2(200→200) → tanh → fc3(200→1)

state (28 维): 与 MAPPO obs_part 相同
joint_act (12 维): 与 MAPPO joint_act_part 相同（均来自 buffer.sample() 压缩）
```

**前向传播：**
```python
forward(state, joint_act):
    x = concat([state, joint_act], dim=-1)  # 40 维
    return fc3(tanh(fc2(tanh(fc1(x)))))     # scalar Q-value
```

---

## 4. 模块改动详情

### 4.1 `config/params.py`

**动作维度：**
```python
action_encode_dim = 10
ppo_action_dim = S + 3 * action_encode_dim  # 33
action_dim     = S + 3 * action_encode_dim  # 33
```

**价值输入维度：**
```python
ppo_value_input_dims = [n * 14 + n * (S + 3) for n in [2, 4, 4]]
# [40, 80, 80]

value_input_obs_dims = [n * 14 for n in [2, 4, 4]]
# [28, 56, 56]
value_input_act_dims = [n * (S + 3) for n in [2, 4, 4]]
# [12, 24, 24]
value_input_dims = [obs + act for obs, act in zip(...)]
# [40, 80, 80]
```

### 4.2 `env/edge_env.py` — 结构性回退

**队列：** 从 N=10 的列表 → 标量。

```python
class EdgeEnv:
    def __init__(self, gen_params, writer=None):
        self.queue_num = 1
        self.device_num = gen_params.device_num

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

        self.edge_dly_adj_val = (min(gen_params.comp_dly_thre)
                                 * gen_params.edge_dly_adj_fac[0])
```

**`get_obs()` → 2 维：**
```python
def get_obs(self):
    return [
        self.edge_queue_time_ql,
        self.virtual_edge_queue_time_ql if self.enable_virtual_queue_reward else -1.0
    ]
```

**`compute(tasks, e_id, t_id, visualize)` — FIFO 处理：**
```python
def compute(self, tasks, e_id, t_id, visualize=False):
    # tasks: List[Task] — 该服务器上所有设备任务（已合并为一级列表）
    gap = self.gen_task_cycle * self.delta

    self.old_edge_queue_time_ql = self.edge_queue_time_ql

    total_comp_need = 0.0
    total_comp_used = 0.0

    for task in sorted(tasks, key=lambda t: t.trans_time):
        if task.trans_time == 0:
            task.e_comp_dly = 0
            task.e_queue_dly = 0
            task.e_proc_dly = 0
        else:
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

    # 单队列演化（标量）
    if t_id % self.gen_task_cycle == self.start_slot:
        self.edge_queue_time_ql = max(
            self.edge_queue_time_ql +
            self.edge_act_queue_growth_rate * (total_comp_need - total_comp_used), 0
        )

        self.old_virtual_edge_queue_time_ql = self.virtual_edge_queue_time_ql
        if self.avg_edge_time > 1e-8:
            self.virtual_edge_queue_time_ql = max(
                self.virtual_edge_queue_time_ql +
                self.edge_vir_queue_growth_rate * (
                    self.edge_queue_time_ql / self.avg_edge_time *
                    self.delta * self.gen_task_cycle -
                    self.edge_dly_adj_val
                ), 0
            )
```

**`reset()` — 标量重置：**
```python
def reset(self):
    self.edge_queue_time_ql = 0.0
    self.virtual_edge_queue_time_ql = 0.0
    # ... 所有标量归零 ...
```

### 4.3 `env/mec_env.py`

**a) `step()` — 去除 softmax，简化 edge_obs：**

```python
def step(self, device_acts, e_id, t_id, visualize=False):
    # device_acts[i] = [server_id, offl, trpw, comp]  (4 维)

    # 1. 设备本地计算
    device_sched_tasks = [None] * self.device_num
    for i in range(self.device_num):
        device_sched_tasks[i] = self.device_envs[i].compute(
            device_acts[i], e_id, t_id, visualize)

    # 2. 路由到服务器（无 softmax）
    for s in range(self.edge_server_num):
        server_tasks = []
        for i in range(self.device_num):
            tasks = device_sched_tasks[i]
            if tasks:
                server_tasks.extend([t for t in tasks if t.target_server == s])
        self.edge_envs[s].compute(server_tasks, e_id, t_id, visualize)

    # 3. edge_obs（按 server，所有设备共享）
    next_edge_obs = []
    for edge_env in self.edge_envs:
        next_edge_obs.extend(edge_env.get_obs())
    # next_edge_obs: [s0_act, s0_vir, s1_act, s1_vir, s2_act, s2_vir]  (6 dims)

    # 4. 设备奖励（不变，去掉 r_e 相关）
    # 5. 边缘奖励 — 按服务器聚合
    for i in range(edge_queue_num):  # type index
        for s in range(self.edge_server_num):
            edge_env = self.edge_envs[s]
            if self.enable_virtual_queue_reward:
                edge_queue_virtual_rewards[i] += \
                    edge_vir_queue_reward_weight * \
                    edge_env.virtual_edge_queue_time_ql * \
                    edge_env.new_vir_edge_ql_change
            if self.enable_actual_queue_reward:
                edge_queue_actual_rewards[i] += \
                    edge_act_queue_reward_weight * \
                    edge_env.edge_queue_time_ql * \
                    edge_env.new_edge_ql_change
```

### 4.4 `env/device_env.py`

**`compute(act)` — 4 维 env_act 解析：**
```python
def compute(self, act, e_id, t_id, visualize=False):
    server_id = int(act[0])
    offl_rto = np.clip(act[1], 0.0, 1.0)
    trpw_rto = np.clip(act[2], 0.6, 1.0)
    comp_rto = np.clip(act[3], 0.6, 1.0)
    # 无 r_e_logit
```

### 4.5 `network/policy_net.py`

**MappoPolicyNet / MappoPolicyNetLSTM：**
```python
ae_dim = alg_params.action_encode_dim
S = alg_params.action_dim - 3 * ae_dim  # 33 - 30 = 3
low  = torch.tensor([-1.]*S + [0.6]*ae_dim + [0.6]*ae_dim + [0.6]*ae_dim)
high = torch.tensor([1.]*S + [1.0]*ae_dim + [1.0]*ae_dim + [1.0]*ae_dim)
```

**MaddpgPolicyNetLSTM：** 同上。

### 4.6 `agent/device_agent.py`

**MappoDeviceAgent.choose_action — 3 连续动作，4 维 env_act：**
```python
S = self.edge_server_num
ae_dim = self.action_encode_dim

# evaluate mode or sampling mode → action (S+3*ae_dim)

server_id = int(np.argmax(act[:S]))
cont_vals = []
for j in range(3):
    block = act[S + j*ae_dim : S + (j+1)*ae_dim]
    cont_vals.append(sum(block) / ae_dim)
env_act = [float(server_id)] + cont_vals  # 4 dims

# noise_scale: [1.0]*S + [0.75]*ae_dim + [1.0]*ae_dim + [1.0]*ae_dim
# inactive: self.last_full_act = [-1.0]*action_dim; env_act = [-1]*4
```

**MaddpgDeviceAgent.choose_action — 同理：**
```python
cont_act = act[:, S:S + 3*ae_dim]
scale = self.p_net.act_scale[S:S + 3*ae_dim]
loc = self.p_net.act_bias[S:S + 3*ae_dim]
action = cont_act * scale + loc
act_list = action.view(-1).tolist()
env_cont = []
for j in range(3):
    env_cont.append(sum(act_list[j*ae_dim:(j+1)*ae_dim]) / ae_dim)
env_act = [float(server_id)] + env_cont
```

**基线策略 — 4 维：**
```python
class LocalComputingDeviceAgent:
    def choose_action(self):
        return [0, 0, 1.0, 1.0]  # offl=0

class EdgeComputingDeviceAgent:
    def choose_action(self):
        return [0, 1.0, 1.0, 1.0]

class RandomComputingDeviceAgent:
    def choose_action(self):
        return [float(server_id), offl, trpw, comp]  # 4 dims
```

### 4.7 `rollout.py`

**a) edge_obs 组织简化（共享，6 维）：**
```python
# init:
edge_obs = []
for edge_env in self.mec_env.edge_envs:
    edge_obs.extend(edge_env.get_obs())  # 6 dims

# policy input: device_obs + edge_obs (all devices share same edge_obs)
device_value_obs = concatenate(device_obss[i], edge_obs)  # 14 dims
```

**b) env_act_dim：** `S + 1 = 4`（was 5）。

**c) edge_comp_qls（日志用）：**
```python
edge_comp_qls = [next_edge_obs[i * 2] for i in range(self.edge_server_num)]
# = [s0_act, s1_act, s2_act] — 3 values = device_type_num
```

**d) 非活跃设备：**
```python
device_acts[i] = [-1.0 for _ in range(self.action_dim)]    # 33 dims
device_acts_[i] = [-1.0 for _ in range(self.env_act_dim)]  # 4 dims
```

### 4.8 `util/replay_buffer.py`

**a) MAPPO `package_value_inputs` — 全局 edge_obs + 压缩 joint_act：**
```python
def package_value_inputs(self, train_episode, train_time_slot, queue_id):
    v_input = []
    ae_dim = self.action_encode_dim
    S = self.action_dim - 3 * ae_dim

    # Obs per device in type: edge_obs(6) + device_obs(8) = 14
    for dev_id in self.device_in_types[queue_id]:
        # 全局 edge_obs（所有设备取相同 6 维）
        for i in range(self.edge_queue_obs_dim):
            _append_flat(v_input, self.edge_obs[train_episode][train_time_slot][i])
        _append_flat(v_input, self.device_obss[train_episode][train_time_slot][dev_id])

    # Joint_act per device (S+3=6 维, 压缩)
    for dev_id in self.device_in_types[queue_id]:
        full_act = self.device_acts[train_episode][train_time_slot][dev_id]
        for i in range(S):
            _append_flat(v_input, full_act[i])
        for c in range(3):
            block_sum = sum(full_act[S + c*ae_dim : S + (c+1)*ae_dim])
            _append_flat(v_input, block_sum / ae_dim)

    return torch.tensor(v_input, dtype=torch.float32).reshape(1, -1)
```

**b) MAPPO `package_policy_input` — 全局 edge_obs：**
```python
def package_policy_input(self, train_episode, train_time_slot, device_id):
    p_input = []
    for i in range(self.edge_queue_obs_dim):
        _append_flat(p_input, self.edge_obs[train_episode][train_time_slot][i])
    _append_flat(p_input, self.device_obss[train_episode][train_time_slot][device_id])
    return torch.tensor(p_input, dtype=torch.float32).reshape(1, -1)
```

**c) MADDPG `package_value_inputs` / `package_policy_input` — 同理，取全局 edge_obs。**

**d) MADDPG `sample()` — joint_act 压缩 S+3*ae→S+3：**
```python
S = self.edge_queue_obs_dim // 2  # 3
ae_dim = self.action_encode_dim

for i in self.device_in_types[k]:
    full_act = self.device_acts[id_][i]  # [S + 3*ae_dim]
    server_logits = full_act[:S]         # S dims
    cont_all = full_act[S:S + 3*ae_dim]  # 3*ae_dim
    cont_3 = [sum(cont_all[c*ae_dim:(c+1)*ae_dim]) / ae_dim for c in range(3)]
    joint_act.extend(server_logits + cont_3)  # S+3 dims
```

### 4.9 `agent/edge_agent.py`

**MADDPG `train_policy_net` — 压缩 S+3*ae→S+3：**
```python
S = self.edge_server_num  # 3
ae_dim = self.action_encode_dim
compressed_dim = S + 3    # 6

server_logits = batch_acts[:, :S]               # [B, S]
cont_all = batch_acts[:, S:S + 3*ae_dim]        # [B, 3*ae]
cont_blocks = cont_all.reshape(-1, 3, ae_dim)   # [B, 3, ae]
cont_compressed = cont_blocks.mean(dim=-1)      # [B, 3]
compressed_act = torch.cat([server_logits, cont_compressed], dim=-1)  # [B, S+3]

s = agent_id_in_type * compressed_dim
e = (agent_id_in_type + 1) * compressed_dim
batch_joint_acts_[:, s:e] = compressed_act
```

### 4.10 `util/utils.py`

**ObsScaling.edge_rms：** `edge_queue_obs_dim * device_num` 改为 `edge_queue_obs_dim * edge_server_num`（6 × 3 = 18），或者直接改为 `edge_queue_obs_dim = 6`（edge_obs 维度本身）。

---

## 5. 实现顺序

| 序号 | 模块 | 影响 |
|------|------|------|
| 1 | `config/params.py` | 维度公式 S+3*ae, value_input_dims |
| 2 | `env/edge_env.py` | N 队列→1 FIFO（最大改动） |
| 3 | `env/mec_env.py` | 去 softmax, edge_obs 按 server |
| 4 | `env/device_env.py` | 4 维 env_act |
| 5 | `network/policy_net.py` | act_scale 去 r_e |
| 6 | `agent/device_agent.py` | 3 连续变量, 4 维 env_act |
| 7 | `rollout.py` | edge_obs 共享, env_act_dim=4 |
| 8 | `util/replay_buffer.py` | package 改为全局 edge_obs, 压缩 S+3 |
| 9 | `agent/edge_agent.py` | train_policy_net 压缩适配 |
| 10 | `util/utils.py` | edge_rms 维度 |

---

## 6. 风险点

| 风险 | 缓解 |
|------|------|
| EdgeEnv N→1 回退引入 bug | 仔细对照 v0 单服务器版的 queue 逻辑 |
| 全局 edge_obs 在 ValueNet 中重复 n_k 次 | 冗余但保持公式统一，Critic 可处理 |
| env_act_dim 从 5→4 破坏下游 | 参数化引用；统一检查 assert |
| 旧 checkpoint 维度不兼容 | v3 为新训练，不加载旧权重 |
