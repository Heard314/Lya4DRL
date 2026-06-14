# Lya4DRL 项目架构文档

## 1. 项目概述

基于多智能体深度强化学习 (MADRL) 的移动边缘计算 (MEC) 任务卸载优化系统。支持两种算法：**MAPPO** (Multi-Agent PPO) 和 **MADDPG** (Multi-Agent DDPG)。

- 设备数: `device_num = 10`，分为 `device_type_num = 3` 类
- 每类设备数: `[2, 4, 4]` (type 0: 2台, type 1: 4台, type 2: 4台)
- 边缘服务器数: `S = edge_server_num = 3`，均匀分布于距原点 250m 圆周 (0°/120°/240°)
- 每服务器: **1 个 FIFO 计算队列**，全频处理 (`edge_comp_freq = 50 Gcycles/s`)
- 动作编码维度: `action_encode_dim = 1` (每个连续动作变量由 1 维编码，压缩为恒等映射)

---

## 2. 系统总体架构

```
┌─────────────────────────────────────────────────────────────┐
│                        main.py                              │
│                          │                                  │
│                    Controller                                │
│                          │                                   │
│                      Rollout                                 │
│         ┌────────────────┼────────────────┐                  │
│         │                │                │                  │
│    DeviceAgent×10   EdgeAgent×1    MECEnv                    │
│    (每个设备一个)    (集中训练)    ┌────┴────┐                │
│         │                │       DeviceEnv  EdgeEnv×S        │
│         │                │       ×10        ×3               │
│    ┌────┴────┐    ┌─────┴─────┐    每服务器1个FIFO队列        │
│  p_net (FC)     v_nets×3 + p_nets×10                          │
│  (分布式执行)    (集中训练: 每类一个v_net, 每设备一个p_net)      │
└─────────────────────────────────────────────────────────────┘
```

### 2.1 文件结构

| 模块 | 文件 | 职责 |
|------|------|------|
| 入口 | `main.py` | 解析参数，启动训练/评估 |
| 控制 | `controller.py` | 控制训练/评估循环，收集指标 |
| 推演 | `rollout.py` | 环境交互、数据收集、训练调度 |
| 环境 | `env/mec_env.py` | MEC顶层环境，协调设备与边缘，边缘奖励按服务器聚合 |
| 环境 | `env/device_env.py` | 设备端环境（任务生成、本地计算，4维 env_act） |
| 环境 | `env/edge_env.py` | 边缘服务器环境（单FIFO队列、标量演化） |
| 智能体 | `agent/device_agent.py` | 设备策略（动作选择，6维→4维env_act） |
| 智能体 | `agent/edge_agent.py` | 集中式训练器（更新 Actor/Critic，含动作压缩） |
| 网络 | `network/policy_net.py` | Actor 网络定义（14→6，3×ae_dim 连续，纯FC无LSTM） |
| 网络 | `network/value_net.py` | Critic 网络定义（MAPPO单输入, MADDPG双输入） |
| 工具 | `util/replay_buffer.py` | 经验回放缓冲区（全局edge_obs, S+3=6 压缩） |
| 工具 | `util/utils.py` | 观测缩放(RMS)、奖励缩放、正交初始化 |
| 配置 | `config/params.py` | MAPPO/MADDPG 超参数定义（维度公式参数化） |
| 配置 | `config/global_params.py` | 全局设置（设备、路径、种子） |

### 2.2 训练模式

- **MAPPO**: On-policy, 每 `train_freq=4` 个 episode 用整个 buffer 训练
- **MADDPG**: Off-policy, 每 `train_freq` 个 time slot 从 replay buffer 采样训练，含 target network 软更新

---

## 3. 观测空间与动作空间

### 3.1 观测空间 (Observation)

**设备观测** `device_obs_dim = 5 + S = 8`:
```
[max_trans_rate[0], ..., max_trans_rate[S-1],   # S=3个: 到各边缘服务器最大传输率
 time_ql,                                         # 1个: 本地计算队列长度
 virtual_time_ql,                                 # 1个: 本地虚拟队列长度
 data_size, comp_dens, dly_cons]                 # 3个: 任务信息
```

**边缘队列观测** (by-server, 所有设备共享) `edge_queue_obs_dim = 2 × S = 6`:
```
[s0_act_ql, s0_vir_ql,   # 服务器0: 实际+虚拟队列 (标量)
 s1_act_ql, s1_vir_ql,   # 服务器1
 s2_act_ql, s2_vir_ql]   # 服务器2
```

**策略网络输入** (每设备): `policy_input_dim = device_obs_dim + edge_queue_obs_dim = 8 + 6 = 14`
每个设备拼接自己的 device_obs 与全局共享的 edge_obs。

### 3.2 动作空间 (Action)

**策略网络输出** `action_dim = S + 3 × ae_dim = 3 + 3 = 6`:

| 维度 | 含义 | ae_dim | 范围 |
|------|------|--------|------|
| 0 ~ S-1 (0~2) | 服务器选择 logits | — | [-1, 1] |
| S (3) | 卸载比例 `offl_rto` | 1 | [0.6, 1.0] |
| S+1 (4) | 传输功率比 `trpw_rto` | 1 | [0.6, 1.0] |
| S+2 (5) | 本地计算频率比 `comp_rto` | 1 | [0.6, 1.0] |

**环境动作** `env_act_dim = 3 + 1 = 4`:
```
[server_id(int), offl_rto, trpw_rto, comp_rto]  # 3个连续值 + 1个离散服务器ID
```
选择方式: argmax(server_logits) + 直接取连续变量值 (ae_dim=1时无需取均值)

**Critic 压缩动作** `compressed_dim = S + 3 = 6` (per device，ae_dim=1 时压缩为恒等映射):
```
[logit_0, logit_1, logit_2,  offl, trpw, comp]
```

| 层 | 维度 | 说明 |
|----|------|------|
| Policy 输出 | 6 | S + 3×ae_dim |
| Buffer 存储 | 6 | 完整 policy 输出 |
| env_act | 4 | 环境执行用 |
| Critic joint_act | S+3=6 | 压缩 (ae_dim=1 时为恒等映射) |

---

## 4. MAPPO 模型架构

### 4.1 Actor (Policy Network) — `MappoPolicyNet`

纯全连接网络，无 LSTM 层。

```
Input: [B, 14]  (policy_input_dim=14)
         │
    ┌────▼────┐  Linear(14, 400), Orthogonal Init (gain=tanh)
    │  fc1    │
    └────┬────┘
         │ [B, 400]
    ┌────▼────┐  Tanh
    │  tanh   │
    └────┬────┘
         │ [B, 400]
    ┌────▼────┐  Linear(400, 400), Orthogonal Init (gain=tanh)
    │  fc2    │
    └────┬────┘
         │ [B, 400]
    ┌────▼────┐  Tanh
    │  tanh   │
    └────┬────┘
         │ [B, 400]
    ┌────▼────┐  Linear(400, 6), Orthogonal Init (gain=0.01)
    │ mu_head │
    └────┬────┘
         │ mean: [B, 6]
         │
    ┌────▼────┐  log_std: Parameter [6] (初始化为0)
    │log_std  │  clamp: [LOG_STD_MIN=-5, LOG_STD_MAX]
    └────┬────┘  LOG_STD_MAX 指数退火: 0.5 → -1.5
         │
    std = exp(log_std).expand_as(mean)  → [B, 6]

    Output: mean [B,6], std [B,6]
```

**动作缩放 (Squashed Gaussian)**:
```
u = mean + std * noise        # 重参数化采样
a = tanh(u)                   # 压缩到 (-1, 1)
action = a * act_scale + act_bias  # 缩放到实际范围
```

其中：
```
act_low  = [-1]×S + [0.6]×ae_dim + [0.6]×ae_dim + [0.6]×ae_dim = [-1,-1,-1, 0.6,0.6,0.6] (6维)
act_high = [ 1]×S + [1.0]×ae_dim + [1.0]×ae_dim + [1.0]×ae_dim = [ 1, 1, 1, 1.0,1.0,1.0] (6维)
act_scale = (act_high - act_low) / 2
act_bias  = (act_high + act_low) / 2
```

**对数概率计算 (用于 PPO 训练)**:
```
log_prob = Normal(mean, std).log_prob(u).sum(-1)
           - log(1 - a² + ε).sum(-1)    # tanh 修正
           - log(act_scale).sum(-1)      # 缩放修正
```

**张量尺寸汇总**:

| 层 | 输入 Shape | 输出 Shape | 参数量 |
|----|-----------|-----------|--------|
| fc1 | [B,14] | [B,400] | 14×400+400=6,000 |
| fc2 | [B,400] | [B,400] | 400×400+400=160,400 |
| mu_head | [B,400] | [B,6] | 400×6+6=2,406 |
| log_std | - | [6] | 6 |

Actor 总参数量 ≈ **168,812** (每个设备一个独立的 Actor，共 10 个)

### 4.2 Critic (Value Network) — `MappoValueNet`

每个设备类型 (device_type) 一个独立的 Value Network，共 3 个。

**Value Net 输入结构** (以 type 0, n=2 为例，40 维):

```
value_input (40 维) = obs_part (28 维) + joint_act_part (12 维)

obs_part (28 维) — 2 devices × 14:
  Device 0: edge_obs(6,全局共享) + device_obs[0](8) = 14
  Device 1: edge_obs(6,全局共享) + device_obs[1](8) = 14

joint_act_part (12 维) — 2 devices × 6 (S+3 压缩):
  Device 0: server_logits(3) + offl_avg(1) + trpw_avg(1) + comp_avg(1) = 6
  Device 1: 同上 = 6
```

| Value Net | 设备数 | 输入维度 | 公式 |
|-----------|--------|----------|------|
| Type 0 | 2 | **40** | 2×14 + 2×6 |
| Type 1 | 4 | **80** | 4×14 + 4×6 |
| Type 2 | 4 | **80** | 4×14 + 4×6 |

```
Input: [B, V_in]  V_in ∈ {40, 80, 80}
         │
    ┌────▼────┐  Linear(V_in, 400), Orthogonal Init
    │  fc1    │
    └────┬────┘
         │ [B, 400]
    ┌────▼────┐  Tanh
    │  tanh   │
    └────┬────┘
         │ [B, 400]
    ┌────▼────┐  Linear(400, 400), Orthogonal Init
    │  fc2    │
    └────┬────┘
         │ [B, 400]
    ┌────▼────┐  Tanh
    │  tanh   │
    └────┬────┘
         │ [B, 400]
    ┌────▼────┐  Linear(400, 1), Orthogonal Init
    │  fc3    │
    └────┬────┘
         │ [B, 1]

    Output: state value [B, 1]
```

| 层 | Type 0 Input | Type 0 Output | Type 1/2 Input | Type 1/2 Output | 参数量(Type 0) |
|----|-------------|---------------|----------------|-----------------|-----------------|
| fc1 | [B,40] | [B,400] | [B,80] | [B,400] | 16,400 |
| fc2 | [B,400] | [B,400] | [B,400] | [B,400] | 160,400 |
| fc3 | [B,400] | [B,1] | [B,400] | [B,1] | 401 |

每个 Critic 参数量 ≈ **177,201** (Type 0) / **193,201** (Type 1/2)

**Critic 输入构造** (`package_value_inputs`):
将同一类型的所有设备的 policy_input (各 14 维) + 压缩 joint_act (各 6 维) 拼接 → `n_devices × (14+6) = n_devices × 20 = V_in`。

---

## 5. MADDPG 模型架构

### 5.1 Actor (Policy Network) — `MaddpgPolicyNet`

纯全连接网络，无 LSTM 层。输出确定性动作。

```
Input: [B, 14]  (policy_input_dim=14)
         │
    ┌────▼────┐  Linear(14, 128), Orthogonal Init (gain=tanh)
    │  fc1    │
    └────┬────┘
         │ [B, 128]
    ┌────▼────┐  Tanh
    │  tanh   │
    └────┬────┘
         │ [B, 128]
    ┌────▼────┐  Linear(128, 128), Orthogonal Init (gain=tanh)
    │  fc2    │
    └────┬────┘
         │ [B, 128]
    ┌────▼────┐  Tanh
    │  tanh   │
    └────┬────┘
         │ [B, 128]
    ┌────▼────┐  Linear(128, 6), Orthogonal Init (gain=0.01)
    │  fc3    │
    └────┬────┘
         │ [B, 6]
    ┌────▼────┐  Tanh → a ∈ (-1, 1)
    │  tanh   │
    └────┬────┘
         │ action: [B, 6]

    Output: action [B, 6]
```

**动作缩放 (Deterministic)**:
```
# 评估模式:
action = tanh(fc3(x)) * act_scale + act_bias

# 训练模式 (加高斯噪声):
action = clamp(tanh(fc3(x)) + noise, -1, 1) * act_scale + act_bias
```

其中：
```
act_low  = [-1]×S + [0.6]×ae_dim + [0.6]×ae_dim + [0.6]×ae_dim = [-1,-1,-1, 0.6,0.6,0.6] (6维)
act_high = [ 1]×S + [1.0]×ae_dim + [1.0]×ae_dim + [1.0]×ae_dim = [ 1, 1, 1, 1.0,1.0,1.0] (6维)
act_scale = (act_high - act_low) / 2
act_bias  = (act_high + act_low) / 2
```

**环境动作转换**: ae_dim=1 时无需取均值，直接取 3 个连续值:
```
server_id = argmax(action[0:3])
offl = action[3]     # ae_dim=1 → 直接取值
trpw = action[4]
comp = action[5]
env_act = [server_id, offl, trpw, comp]  # 4 维
```

**张量尺寸汇总**:

| 层 | 输入 Shape | 输出 Shape | 参数量 |
|----|-----------|-----------|--------|
| fc1 | [B,14] | [B,128] | 14×128+128=1,920 |
| fc2 | [B,128] | [B,128] | 128×128+128=16,512 |
| fc3 | [B,128] | [B,6] | 128×6+6=774 |

Actor 总参数量 ≈ **19,206** (每个设备一个，共 10 个)
Target Actor 同上结构。

### 5.2 Critic (Value Network) — `MaddpgValueNet`

MADDPG 的 Critic 是集中式的，输入包含全局 state + 全局 joint action。

```
Input: concat([state, joint_act])    总维度 ∈ {40, 80, 80}
         │                              state_dim ∈ {28, 56, 56}
    ┌────▼────┐  Linear(V_in, 128)    joint_act_dim ∈ {12, 24, 24}
    │  fc1    │  Orthogonal Init
    └────┬────┘
         │ [B, 128]
    ┌────▼────┐  Tanh
    │  tanh   │
    └────┬────┘
         │ [B, 128]
    ┌────▼────┐  Linear(128, 128), Orthogonal Init
    │  fc2    │
    └────┬────┘
         │ [B, 128]
    ┌────▼────┐  Tanh
    │  tanh   │
    └────┬────┘
         │ [B, 128]
    ┌────▼────┐  Linear(128, 1), Orthogonal Init
    │  fc3    │
    └────┬────┘
         │ Q-value: [B, 1]

    Output: Q-value [B, 1]
```

**Critic 输入构造**:
- **state**: 同类型所有设备的 policy_input 拼接 = `n × 14` (每设备: edge_obs 6 + device_obs 8)
- **joint_act**: 同类型所有设备的压缩动作拼接 = `n × 6` (每设备: S logits + 3 连续均值)
- **总输入**: `n × (14 + 6) = n × 20`

| Critic | 设备数 | state_dim | joint_act_dim | 总输入 | 参数量 |
|--------|--------|-----------|---------------|--------|--------|
| Type 0 | 2 | 28 | 12 | **40** | 40×128+128 + 128×128+128 + 128×1+1 = 21,897 |
| Type 1 | 4 | 56 | 24 | **80** | 80×128+128 + 128×128+128 + 128×1+1 = 27,017 |
| Type 2 | 4 | 56 | 24 | **80** | 同上 |

Target Critic 同上结构。

**Q-target 计算**:
```
# 从 target policy nets 获取下一时刻动作，并压缩为 S+3:
next_joint_act = []
for device i in type:
    next_act = target_p_net_i(next_obs_i)           # [B, 33]
    compressed = [logits[:3], mean(offl_block),     # 6 维
                  mean(trpw_block), mean(comp_block)]
    next_joint_act.append(compressed)
next_joint_act = concat(next_joint_act)              # [B, 6×n]

next_q = target_v_net(next_state, next_joint_act)
target_q = reward + γ × next_q
```

---

## 6. 维度汇总

```python
S = 3       # edge_server_num
ae_dim = 1  # action_encode_dim

# 观测
device_obs_dim     = 5 + S         # 8
edge_queue_obs_dim = 2 * S         # 6
policy_input_dim   = 14            # 8 + 6

# 动作
action_dim     = S + 3 * ae_dim   # 6
env_act_dim    = 3 + 1            # 4 (3 cont: offl/trpw/comp, 1 discrete: server_id)
compressed_dim = S + 3            # 6 (ae_dim=1 时压缩为恒等映射)

# Value Net 输入 (per type)
# MAPPO:  n×(14+6) = n×20
ppo_value_input_dims = [40, 80, 80]

# MADDPG:
value_input_obs_dims = [28, 56, 56]   # n×14
value_input_act_dims = [12, 24, 24]   # n×6
value_input_dims     = [40, 80, 80]   # obs + act
```

### 6.1 网络分布

**CTDE (集中训练, 分布执行) 架构下的网络所有权:**

| 网络 | 数量 | 拥有者 | 用途 |
|------|------|--------|------|
| Actor (p_net) | **10个** (每设备1个) | 每个终端设备独立拥有 | 分布式执行: 根据本地观测输出动作 |
| Target Actor | 10个 (MADDPG only) | EdgeAgent集中持有 | 稳定Q-target计算 |
| Critic (v_net) | **3个** (每设备类型1个) | EdgeAgent集中持有 | 集中训练: 评估全局状态-动作价值 |
| Target Critic | 3个 (MADDPG only) | EdgeAgent集中持有 | 稳定Q-target计算 |

**关键说明:**
- **边缘服务器没有自己的 Actor 网络**。服务器不参与决策，仅被动执行设备卸载的任务计算。
- Actor 网络在训练时由 `EdgeAgent` 统一更新（集中训练），更新后同步给各 `DeviceAgent`（分布执行）。
- Critic 网络仅在训练阶段使用（评估全局价值），推理阶段不需要。
- Critic 按**设备类型**（而非服务器）分组：3 个服务器共享同一 Critic（同类型设备的联合价值由一个 Critic 评估）。

---

## 7. 训练流程对比

| 特性 | MAPPO | MADDPG |
|------|-------|--------|
| 训练类型 | On-policy | Off-policy |
| 触发频率 | 每 4 episodes | 每 train_freq time slots |
| Buffer | `train_freq × buffer_train_time_slots` | `buffer_size = 24000` |
| 训练数据 | 整段 episode 序列 | 随机采样 batch |
| Critic 输入 | state + joint_act (拼接) | state + joint_act (concat) |
| Actor 输出 | 高斯分布 (mean + std) 33维 | 确定性动作 33维 + 噪声 |
| Actor 损失 | PPO-Clip + Entropy | -Q(s, a) |
| Critic 损失 | MSE(v_pred, GAE_target) | MSE(Q, target_Q) |
| Target Net | 无 | 软更新 τ=0.005 |
| 探索方式 | 随机采样 (含 OU 噪声) | 高斯噪声 (衰减: 0.2→0.02) |
| Std 退火 | LOG_STD_MAX: 0.5→-1.5 | 无 |

---

## 8. 数据流图

```
Episode Start
    │
    ▼
┌──────────────────────────────────────────────────┐
│  for t in time_slots:                            │
│    1. DeviceEnv.get_obs()  →  device_obs (8维)    │
│    2. EdgeEnv[S].get_obs() →  edge_obs   (6维,全局)│
│    3. concat → policy_input (14维)                │
│    4. Actor(policy_input) → action (33维)         │
│    5. compress → env_act (4维)                    │
│    6. env.step(env_act)   →  reward, next_obs     │
│    7. replay_buffer.store(transition)             │
│    8. (MADDPG) Agent.train_nets()  if time        │
│       (MAPPO)  Agent.train_nets()  if episode end │
└──────────────────────────────────────────────────┘
    │
    ▼
Episode End
```

### MAPPO 训练细节

```
train_nets():
  for each device_type k:
    v_inputs, v_tags, advs = buffer.get_value_net_training_data(k)
    train_value_net(k, v_inputs, v_tags)     # v_epochs
    for each device i in type k:
      train_policy_net(i, p_inputs[:,i], ...)  # p_epochs
```

`train_value_net`:
```
for epoch in v_epochs:
  for batch in BatchSampler:
    v_pred = v_net(v_input)              # [B, 1], v_input = state + joint_act 拼接
    loss = MSE(v_pred, v_target)         # v_target = adv + v_old
```

`train_policy_net`:
```
for epoch in p_epochs:
  for batch in BatchSampler:
    mean, std = p_net(obs)              # [B, 6], 纯FC无隐藏状态
    new_logp = log_prob(action | mean, std)
    ratio = exp(new_logp - old_logp)
    surr1 = ratio * adv
    surr2 = clamp(ratio, 1-ε, 1+ε) * adv
    loss = -min(surr1, surr2) - entropy_coef * entropy
```

### MADDPG 训练细节

```
train_nets():
  for _ in critic_updates_round:
    for each device_type k:
      train_value_net(k, states, joint_acts, rewards, next_states, next_obss)
  if step % policy_delay_round == 0:
    for each device_type k:
      for each device i in type k:
        train_policy_net(i, k, states, obs_i, joint_acts)
```

`train_value_net`:
```
# next_joint_acts: 从 target policy net 获取并压缩 S+3*ae→S+3 (ae=1时为恒等)
next_joint_act = []
for device i in type:
    raw = target_p_net_i(next_obs_i)     # [B, 6]
    compressed = [raw[:3], raw[3], raw[4], raw[5]]  # 6维 (恒等映射)
    next_joint_act.append(compressed)
next_joint_act = concat(next_joint_act)  # [B, 6×n]
next_q = target_v_net(next_state, next_joint_act)
target_q = reward + γ × next_q
loss = MSE(v_net(state, joint_act), target_q.detach())
```

`train_policy_net`:
```
# 从 replay buffer 取 joint_acts (已压缩 S+3), 替换当前设备的 slot
new_act = p_net(obs_i)                  # [B, 6]
compressed = [new_act[:3], new_act[3], new_act[4], new_act[5]]  # 6维 (恒等)
joint_act[:, device_slot] = compressed  # 替换对应 slot
loss = -v_net(state, joint_act).mean()  # 最大化 Q 值
```

---

## 9. 关键超参数

| 参数 | MAPPO | MADDPG |
|------|-------|--------|
| Actor 隐藏层 | [400, 400] | [128, 128] |
| Critic 隐藏层 | [400, 400] | [128, 128] |
| Actor LR | 1e-4 | 5e-5 |
| Critic LR | 1e-4 | 1e-4 |
| γ (折扣因子) | 0.99 | 0.99 |
| τ (软更新) | N/A | 0.005 |
| PPO clip ε | 0.1 | N/A |
| Entropy coef | 0.05 | N/A |
| GAE λ | 0.95 | N/A |
| 梯度裁剪 | 2.0 | 2.0 |
| 训练 episodes | 30000 | 15000 |
| 每 episode time slots | 3000 | 3000 |
| Batch size | 2400 | 600 |
| action_encode_dim | 1 | 1 |
| Actor 类型 | MappoPolicyNet (FC) | MaddpgPolicyNet (FC) |

---

## 10. 变更历史

### v2 → v3 (per-device 多队列 + r_e 资源分配 → 每服务器单 FIFO)

| 项目 | v2 | v3 |
|------|----|----|
| 每服务器队列数 | N=10 (per-device) | **1 FIFO** |
| CPU 分配 | softmax r_e | **无 (全频)** |
| 连续动作变量 | 4 (offl, trpw, comp, r_e) | **3 (offl, trpw, comp)** |
| action_dim | S+4×ae=43 | **S+3×ae=33** |
| env_act_dim | S+2=5 | **S+1=4** |
| edge_obs | per-device (60维) | **by-server 全局 (6维)** |
| ppo_value_input_dims | [42,84,84] | **[40,80,80]** |
| value_input_dims | [42,84,84] | **[40,80,80]** |
| compressed_dim | S+4=7 | **S+3=6** |
| 队列演化 | per-device 列表 | **标量单队列** |
| edge_dly_adj_val | per-device | **min(comp_dly_thre) × fac[0]** |

### 当前版本 (ae_dim=1 + 纯FC Actor)

| 项目 | v3 (LSTM, ae=10) | 当前 (FC, ae=1) |
|------|-------------------|-------------------|
| ae_dim | 10 | **1** |
| action_dim | S+3×10=33 | **S+3=6** |
| Actor 网络 | MappoPolicyNetLSTM / MaddpgPolicyNetLSTM | **MappoPolicyNet / MaddpgPolicyNet (纯FC)** |
| MAPPO Actor 参数量 | ~1,302,466 | **~168,812** |
| MADDPG Actor 参数量 | ~331,233 | **~19,206** |
| MADDPG 隐藏层 | [200, 200] | **[128, 128]** |
| LSTM 隐藏状态管理 | 需维护 (h, c) | **无** |
| 连续动作编码 | 10维编码→均值压缩 | **1维直接使用 (恒等映射)** |
| compressed_dim | S+3=6 (压缩) | **S+3=6 (恒等)** |
