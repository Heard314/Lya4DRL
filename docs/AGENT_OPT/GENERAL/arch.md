# Lya4DRL 项目架构文档

## 1. 项目概述

基于多智能体深度强化学习 (MADRL) 的移动边缘计算 (MEC) 任务卸载优化系统。支持两种算法：**MAPPO** (Multi-Agent PPO) 和 **MADDPG** (Multi-Agent DDPG)。

- 设备数: `device_num = 10`，分为 `device_type_num = 3` 类
- 每类设备数: `[2, 4, 4]` (type 0: 2台, type 1: 4台, type 2: 4台)
- 边缘服务器数: `S = edge_server_num = 3`

---

## 2. 系统总体架构

```
┌─────────────────────────────────────────────────────────────┐
│                        main.py                               │
│                          │                                   │
│                    Controller                                │
│                          │                                   │
│                      Rollout                                 │
│         ┌────────────────┼────────────────┐                  │
│         │                │                │                  │
│    DeviceAgent×10   EdgeAgent×1    MECEnv                    │
│    (每个设备一个)    (全局训练)    ┌────┴────┐                │
│         │                │       DeviceEnv  EdgeEnv×S        │
│         │                │       ×10        ×3               │
│    ┌────┴────┐    ┌─────┴─────┐                               │
│  p_net(LSTM)    v_nets×3 + p_nets×10                          │
│  (推理用)       (训练用, 每类一个v_net)                        │
└─────────────────────────────────────────────────────────────┘
```

### 2.1 文件结构

| 模块 | 文件 | 职责 |
|------|------|------|
| 入口 | `main.py` | 解析参数，启动训练/评估 |
| 控制 | `controller.py` | 控制训练/评估循环，收集指标 |
| 推演 | `rollout.py` | 环境交互、数据收集、训练调度 |
| 环境 | `env/mec_env.py` | MEC顶层环境，协调设备与边缘 |
| 环境 | `env/device_env.py` | 设备端环境（任务生成、本地计算） |
| 环境 | `env/edge_env.py` | 边缘服务器环境（远程计算、队列管理） |
| 智能体 | `agent/device_agent.py` | 设备策略（动作选择） |
| 智能体 | `agent/edge_agent.py` | 集中式训练器（更新 Actor/Critic） |
| 网络 | `network/policy_net.py` | Actor 网络定义 |
| 网络 | `network/value_net.py` | Critic 网络定义 |
| 工具 | `util/replay_buffer.py` | 经验回放缓冲区 |
| 工具 | `util/utils.py` | 观测缩放、奖励缩放、正交初始化 |
| 配置 | `config/params.py` | MAPPO/MADDPG 超参数定义 |
| 配置 | `config/global_params.py` | 全局设置（设备、路径、种子） |

### 2.2 训练模式

- **MAPPO**: On-policy, 每 `train_freq=4` 个 episode 用整个 buffer 训练
- **MADDPG**: Off-policy, 每 `train_freq` 个 time slot 从 replay buffer 采样训练，含 target network 软更新

---

## 3. 观测空间与动作空间

### 3.1 观测空间 (Observation)

**设备观测** `device_obs_dim = S + 5 = 8`:
```
[max_trans_rate[0], ..., max_trans_rate[S-1],   # S个: 到各边缘服务器最大传输率
 time_ql,                                         # 1个: 本地计算队列长度
 virtual_time_ql,                                 # 1个: 本地虚拟队列长度
 data_size, comp_dens, dly_cons]                 # 3个: 任务信息
```

**边缘队列观测** (per-device) `edge_queue_obs_dim = 2 × S = 6`:
```
[edge_queue_ql_s0, vir_edge_queue_ql_s0,   # 服务器0: 实际+虚拟队列
 edge_queue_ql_s1, vir_edge_queue_ql_s1,   # 服务器1
 edge_queue_ql_s2, vir_edge_queue_ql_s2]   # 服务器2
```

**策略网络输入** (设备i): `policy_input_dim = device_obs_dim + edge_queue_obs_dim = 8 + 6 = 14`

### 3.2 动作空间 (Action)

**MAPPO 动作** `action_dim = S + 4 = 7`:

| 维度 | 含义 | 范围 |
|------|------|------|
| 0 ~ S-1 (0~2) | 服务器选择 logits | [-1, 1] |
| S (3) | 卸载比例 `offl_rto` | [0.6, 1.0] |
| S+1 (4) | 传输功率比 `trpw_rto` | [0.6, 1.0] |
| S+2 (5) | 本地计算频率比 `comp_rto` | [0.6, 1.0] |
| S+3 (6) | 资源分配 logit `r_e_logit` | [-1, 1] |

**MADDPG 动作** `action_dim = S + 31 = 34`:
- 前 S=3 维: 服务器选择 logits
- 中间 30 维: 连续动作 (3类 × 10维), 通过平均压缩为 3 个值
- 最后 1 维: `r_e_logit`

---

## 4. MAPPO 模型架构

### 4.1 Actor (Policy Network) — `MappoPolicyNetLSTM`

```
Input: [B, 14]  (policy_input_dim=14)
         │
    ┌────▼────┐  Linear(14, 400), Orthogonal Init (gain=tanh)
    │  fc1    │
    └────┬────┘
         │ [B, T, 400]
    ┌────▼────┐  Tanh
    │  tanh   │
    └────┬────┘
         │ [B, T, 400]
    ┌────▼────┐  LSTM(400, 400, batch_first=True)
    │  lstm   │  Orthogonal Init (weights), Zero Init (bias)
    └────┬────┘
         │ [B, T, 400], (h_n:[1,B,400], c_n:[1,B,400])
    ┌────▼────┐  Tanh
    │  tanh   │
    └────┬────┘
         │ [B, T, 400]
    ┌────▼────┐  Linear(400, 7), Orthogonal Init (gain=0.01)
    │ mu_head │
    └────┬────┘
         │ mean: [B, T, 7]
         │
    ┌────▼────┐  log_std: Parameter [7] (初始化为0)
    │log_std  │  clamp: [LOG_STD_MIN=-5, LOG_STD_MAX]
    └────┬────┘  LOG_STD_MAX 指数退火: 0.5 → -1.5
         │
    std = exp(log_std).expand_as(mean)  → [B, T, 7]

    Output: mean [B,T,7], std [B,T,7], (h_n, c_n)
```

**动作缩放 (Squashed Gaussian)**:
```
u = mean + std * noise        # 重参数化采样
a = tanh(u)                   # 压缩到 (-1, 1)
action = a * act_scale + act_bias  # 缩放到实际范围
```

其中:
```
act_low  = [-1, -1, -1,  0.6, 0.6, 0.6, -1]
act_high = [ 1,  1,  1,  1.0, 1.0, 1.0,  1]
act_scale = (act_high - act_low) / 2 = [1, 1, 1, 0.2, 0.2, 0.2, 1]
act_bias  = (act_high + act_low) / 2 = [0, 0, 0, 0.8, 0.8, 0.8, 0]
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
| fc1 | [B,T,14] | [B,T,400] | 14×400+400=6,000 |
| LSTM | [B,T,400] | [B,T,400] | 4×(400×400+400×400+400+400)=1,283,200 |
| mu_head | [B,T,400] | [B,T,7] | 400×7+7=2,807 |
| log_std | - | [7] | 7 |

Actor 总参数量 ≈ **1,292,014** (每个设备一个独立的 Actor，共 10 个)

### 4.2 Critic (Value Network) — `MappoValueNet`

每个设备类型 (device_type) 一个独立的 Value Network，共 3 个。

**Value Net 0** (type 0: 2 devices):
```
Input: [B, 42]  (2 × 14 + 2 × 7 = 28 + 14 = 42)
```

**Value Net 1** (type 1: 4 devices):
```
Input: [B, 84]  (4 × 14 + 4 × 7 = 56 + 28 = 84)
```

**Value Net 2** (type 2: 4 devices):
```
Input: [B, 84]  (同 type 1)
```

```
Input: [B, V_in]  V_in ∈ {42, 84, 84}
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

**张量尺寸汇总**:

| 层 | Type 0 Input | Type 0 Output | Type 1/2 Input | Type 1/2 Output | 参数量(Type 0/1/2) |
|----|-------------|---------------|----------------|-----------------|-------------------|
| fc1 | [B,42] | [B,400] | [B,84] | [B,400] | 17,200 / 34,000 |
| fc2 | [B,400] | [B,400] | [B,400] | [B,400] | 160,400 |
| fc3 | [B,400] | [B,1] | [B,400] | [B,1] | 401 |

每个 Critic 参数量 ≈ **178,001** (Type 0) / **194,801** (Type 1/2)

**Critic 输入构造** (`package_value_inputs`):
将同一类型的所有设备的 policy_input (各 14 维) 拼接 → `n_devices × 14 = state_dim`。训练时 Critic 只接收 state（不含 action），因为 MAPPO 是 on-policy 且 action 信息已隐含在 advantage 中。

---

## 5. MADDPG 模型架构

### 5.1 Actor (Policy Network) — `MaddpgPolicyNetLSTM`

```
Input: [B, 14]  (policy_input_dim=14)
         │
    ┌────▼────┐  Linear(14, 200), Orthogonal Init (gain=tanh)
    │  fc1    │
    └────┬────┘
         │ [B, T, 200]
    ┌────▼────┐  Tanh
    │  tanh   │
    └────┬────┘
         │ [B, T, 200]
    ┌────▼────┐  LSTM(200, 200, batch_first=True)
    │  lstm   │  Orthogonal Init
    └────┬────┘
         │ [B, T, 200], (h_n:[1,B,200], c_n:[1,B,200])
    ┌────▼────┐  Tanh
    │  tanh   │
    └────┬────┘
         │ [B, T, 200]
    ┌────▼────┐  Linear(200, 34), Orthogonal Init (gain=0.01)
    │  fc3    │
    └────┬────┘
         │ [B, T, 34]
    ┌────▼────┐  Tanh → a ∈ (-1, 1)
    │  tanh   │
    └────┬────┘
         │ action: [B, T, 34]

    Output: action [B,T,34], (h_n, c_n)
```

**动作缩放 (Deterministic)**:
```
# 评估模式:
action = tanh(fc3(x)) * act_scale + act_bias

# 训练模式 (加高斯噪声):
action = clamp(tanh(fc3(x)) + noise, -1, 1) * act_scale + act_bias
```

其中:
```
act_low  = [-1]×3 + [0.6]×30 + [-1]  = [3个-1, 30个0.6, 1个-1]
act_high = [ 1]×3 + [1.0]×30 + [ 1]  = [3个1,  30个1.0,  1个1]
act_scale = (high-low)/2 = [1]×3 + [0.2]×30 + [1]
act_bias  = (high+low)/2 = [0]×3 + [0.8]×30 + [0]
```

**环境动作转换**: 30维连续动作按每组10维平均压缩为3个值:
```
env_cont[k] = mean(action[S+k*10 : S+(k+1)*10])  for k=0,1,2
```

**张量尺寸汇总**:

| 层 | 输入 Shape | 输出 Shape | 参数量 |
|----|-----------|-----------|--------|
| fc1 | [B,T,14] | [B,T,200] | 14×200+200=3,000 |
| LSTM | [B,T,200] | [B,T,200] | 4×(200×200+200×200+200+200)=321,600 |
| fc3 | [B,T,200] | [B,T,34] | 200×34+34=6,834 |

Actor 总参数量 ≈ **331,434** (每个设备一个，共 10 个)  
Target Actor 同上结构。

### 5.2 Critic (Value Network) — `MaddpgValueNet`

MADDPG 的 Critic 是集中式的，输入包含全局 state + 全局 joint action。

```
Input: concat([state, joint_act])    总维度 ∈ {42, 84, 84}
         │                              state_dim ∈ {28, 56, 56}
    ┌────▼────┐  Linear(V_in, 200)    joint_act_dim ∈ {14, 28, 28}
    │  fc1    │  Orthogonal Init
    └────┬────┘
         │ [B, 200]
    ┌────▼────┐  Tanh
    │  tanh   │
    └────┬────┘
         │ [B, 200]
    ┌────▼────┐  Linear(200, 200), Orthogonal Init
    │  fc2    │
    └────┬────┘
         │ [B, 200]
    ┌────▼────┐  Tanh
    │  tanh   │
    └────┬────┘
         │ [B, 200]
    ┌────▼────┐  Linear(200, 1), Orthogonal Init
    │  fc3    │
    └────┬────┘
         │ Q-value: [B, 1]

    Output: Q-value [B, 1]
```

**Critic 输入构造**:
- **state**: 同类型所有设备的 policy_input 拼接 = `n × 14`
- **joint_act**: 同类型所有设备的压缩动作拼接 (每设备 `S+4=7` 维) = `n × 7`
- **总输入**: `n × (14 + 7) = n × 21`

| Critic | 设备数 | state_dim | joint_act_dim | 总输入 | 参数量 |
|--------|--------|-----------|---------------|--------|--------|
| Type 0 | 2 | 28 | 14 | 42 | 42×200+200 + 200×200+200 + 200×1+1 = 49,001 |
| Type 1 | 4 | 56 | 28 | 84 | 84×200+200 + 200×200+200 + 200×1+1 = 57,401 |
| Type 2 | 4 | 56 | 28 | 84 | 同上 |

Target Critic 同上结构。

**Q-target 计算**:
```
next_joint_act = concat([target_p_net_i(next_obs_i) for i in type])
next_q = target_v_net(next_state, next_joint_act)
target_q = reward + γ × next_q
```

---

## 6. 训练流程对比

| 特性 | MAPPO | MADDPG |
|------|-------|--------|
| 训练类型 | On-policy | Off-policy |
| 触发频率 | 每 4 episodes | 每 12000 time slots |
| Buffer | `train_freq × buffer_train_time_slots` | `buffer_size = 24000` |
| 训练数据 | 整段 episode 序列 | 随机采样 batch |
| Critic 输入 | state only (GAE) | state + joint_action (Q-learning) |
| Actor 输出 | 高斯分布 (mean + std) | 确定性动作 + 噪声 |
| Actor 损失 | PPO-Clip + Entropy | -Q(s, a) |
| Critic 损失 | MSE(v_pred, GAE_target) | MSE(Q, target_Q) |
| Target Net | 无 | 软更新 τ=0.005 |
| 探索方式 | 随机采样 (或 OU 噪声) | 高斯噪声 (衰减: 0.2→0.02) |
| Std 退火 | LOG_STD_MAX: 0.5→-1.5 | 无 |

---

## 7. 数据流图

```
Episode Start
    │
    ▼
┌──────────────────────────────────────────────────┐
│  for t in time_slots:                            │
│    1. DeviceEnv.get_obs()  →  device_obs (8维)    │
│    2. EdgeEnv.get_obs()    →  edge_obs   (6维/设备)│
│    3. concat → policy_input (14维)                │
│    4. Actor(policy_input) → action (7或34维)      │
│    5. env.step(action)    →  reward, next_obs     │
│    6. replay_buffer.store(transition)             │
│    7. (MADDPG) Agent.train_nets()  if time        │
│       (MAPPO)  Agent.train_nets()  if episode end │
└──────────────────────────────────────────────────┘
    │
    ▼
Episode End
```

### MAPPO 训练细节 (edge_agent.py)

```
train_nets():
  for each device_type k:
    v_inputs, v_tags, advs = buffer.get_value_net_training_data(k)
    train_value_net(k, v_inputs, v_tags)     # v_epochs × 4
    for each device i in type k:
      train_policy_net(i, p_inputs[:,i], ...)  # p_epochs × 4
```

`train_value_net`:
```
for epoch in v_epochs:
  for batch in BatchSampler:
    v_pred = v_net(state)          # [B, 1]
    loss = MSE(v_pred, v_target)   # v_target = adv + v_old
```

`train_policy_net`:
```
for epoch in p_epochs:
  for batch in BatchSampler:
    mean, std = p_net(obs, lstm_hidden)  # [B, 7]
    new_logp = log_prob(action | mean, std)
    ratio = exp(new_logp - old_logp)
    surr1 = ratio * adv
    surr2 = clamp(ratio, 1-ε, 1+ε) * adv
    loss = -min(surr1, surr2) - entropy_coef * entropy
```

### MADDPG 训练细节 (edge_agent.py)

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
next_joint_act = concat([target_p_net_j(next_obs_j) for j in type])
next_q = target_v_net(next_state, next_joint_act)
target_q = reward + γ * next_q
loss = MSE(v_net(state, joint_act), target_q.detach())
```

`train_policy_net`:
```
new_act = p_net(obs_i)
joint_act[:, agent_slot] = new_act
loss = -v_net(state, joint_act).mean()   # 最大化 Q 值
```

---

## 8. 关键超参数

| 参数 | MAPPO | MADDPG |
|------|-------|--------|
| Actor 隐藏层 | [400, 400] | [200, 200] |
| Critic 隐藏层 | [400, 400] | [200, 200] |
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
