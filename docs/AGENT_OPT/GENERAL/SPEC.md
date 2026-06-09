# 项目总体设计规格书 — 多智能体深度强化学习多任务部分计算卸载

## 1. 项目概述

### 1.1 项目背景

本项目实现论文 **"Multi-Agent Deep Reinforcement Learning Based Multi-Task Partial Computation Offloading in Mobile Edge Computing"** 的完整代码。核心问题是移动边缘计算（MEC）场景下，多个终端设备需要通过**部分计算卸载**（Partial Computation Offloading）将任务的一部分在本地执行、另一部分卸载到边缘服务器执行，以最小化时延和能耗的联合成本。

### 1.2 技术路线

采用两类多智能体深度强化学习算法：
- **MAPPO**（Multi-Agent Proximal Policy Optimization）：基于 PPO 的 CTDE（Centralized Training Decentralized Execution）多智能体算法
- **MADDPG**（Multi-Agent Deep Deterministic Policy Gradient）：基于 DDPG 的 CTDE 多智能体算法

关键创新点：引入**李雅普诺夫（Lyapunov）虚拟队列**技术，将长期时延约束转化为队列稳定性问题，通过队列漂移加惩罚（Drift-Plus-Penalty）方法优化。

---

## 2. 系统架构总览

### 2.1 文件组织结构

```
Lya4DRL/
├── main.py                  # 主入口：训练/评估模式调度
├── controller.py            # 控制器：训练/评估循环、指标收集、持久化
├── rollout.py               # Rollout 引擎：环境交互、数据采集、训练触发
├── test.py                  # 简单的 PolicyNet 梯度测试（独立于主项目）
├── plot_exp.py              # 实验绘图工具
├── requirements.txt         # Python 依赖
├── environment.yml          # Conda 环境配置
│
├── config/
│   ├── params.py            # 所有超参数定义（通用/环境/MAPPO/MADDPG）
│   └── global_params.py     # 全局运行时设置（设备、路径、种子）
│
├── env/
│   ├── mec_env.py           # MEC 顶层环境：协调设备端与边缘端
│   ├── device_env.py        # 设备环境：任务生成、本地计算、移动性
│   └── edge_env.py          # 边缘环境：队列管理、远程计算
│
├── agent/
│   ├── device_agent.py      # 设备端智能体（MAPPO/MADDPG/基线策略）
│   └── edge_agent.py        # 边缘端智能体（策略/价值网络训练）
│
├── network/
│   ├── policy_net.py        # 策略网络（MLP + LSTM 版本）
│   └── value_net.py         # 价值网络（集中式 Critic）
│
├── util/
│   ├── replay_buffer.py     # 经验回放（MAPPO GAE 缓冲 / MADDPG 缓冲）
│   └── utils.py             # 观测缩放、奖励缩放、噪声、可视化
│
├── scripts/
│   ├── train/               # 训练启动脚本
│   ├── eval/                # 评估启动脚本
│   └── ablation/            # 消融实验脚本
│
└── docs/
    └── AGENT_OPT/
        └── GENERAL.md/
            └── SPEC.md      # 本文档
```

### 2.2 核心模块依赖图

```
main.py
  └── controller.py
        ├── config/params.py (get_general_params, get_mappo_params, get_maddpg_params)
        ├── config/global_params.py (Settings dataclass)
        └── rollout.py
              ├── env/mec_env.py
              │     ├── env/device_env.py  (×N devices)
              │     └── env/edge_env.py    (×1 edge server)
              ├── agent/device_agent.py    (×N agents)
              ├── agent/edge_agent.py      (×1 edge agent, holds all nets)
              │     ├── network/policy_net.py
              │     └── network/value_net.py
              ├── util/replay_buffer.py
              └── util/utils.py
```

---

## 3. 推理链路（Inference Pipeline）

推理链路覆盖从环境观测到动作执行、环境转移、奖励计算的完整闭环。

### 3.1 推理主循环 — `Rollout.run()`

位置：[rollout.py:206-453](rollout.py#L206)

```
每个 episode:
  ├── reset() — 重置累积统计量和 MEC 环境
  ├── 获取初始观测: edge_obs, device_obss[0..N-1]
  ├── 初始化 LSTM 隐状态: lstm_hidden_hs, lstm_hidden_cs
  │
  └── for t_id in [0, time_slots]:
        ├── 判断是否为任务生成时间槽 (t_id % gen_task_cycle == start_slot)
        │
        ├── [Step 1] 动作选择 (choose_action)
        │     ├── MAPPO: 对每个设备，拼接 device_obs + 对应类型 edge_obs → PolicyNet(LSTM)
        │     │         输出 mean/std → reparameterize → tanh → rescale → action[3]
        │     │         (训练时加 OU 噪声探索，评估时确定性 mean)
        │     ├── MADDPG: 类似流程，PolicyNet(LSTM) → tanh → +GaussianNoise → action[30]
        │     │          (action 每10维表示一个决策维度)
        │     └── 基线策略 (Local/Edge/Random): 直接返回固定或随机动作
        │
        ├── [Step 2] 环境步进 (MECEnv.step)
        │     ├── 设备端计算 (DeviceEnv.compute) — 本地+传输部分
        │     │     ├── 根据 action 拆分本地/卸载部分
        │     │     ├── 计算传输速率 (香农公式)
        │     │     ├── 计算本地计算时延 & 能耗
        │     │     ├── 更新实际计算队列 (time_ql) 和虚拟队列 (virtual_time_ql)
        │     │     └── 设备移动 (位置更新)
        │     ├── 边缘端计算 (EdgeEnv.compute) — 远程执行部分
        │     │     ├── 按任务类型分组调度 (FIFO by trans_time)
        │     │     ├── 计算边缘计算时延 & 能耗
        │     │     └── 更新边缘实际队列和虚拟队列
        │     └── 奖励计算
        │           ├── 基础奖励 (naive reward): 时延/能耗惩罚
        │           ├── 实际队列奖励 (optional): 队列长度 × 队列变化
        │           └── 虚拟队列奖励 (optional): 虚拟队列长度 × 虚拟队列变化
        │
        ├── [Step 3] 数据存储 (仅训练模式)
        │     ├── MAPPO: replay_buffer.store(edge_obs, device_obss, lstm_states,
        │     │                               acts, act_logprobs, rewards, active_masks)
        │     └── MADDPG: replay_buffer.store(edge_obs, device_obss, acts, rewards,
        │                                      next_edge_obs, next_device_obss)
        │
        └── [Step 4] 观测更新
              ├── edge_obs ← next_edge_obs
              ├── device_obss ← next_device_obss
              └── lstm_states ← next_lstm_states
```

### 3.2 观测空间（Observation Space）

**设备观测 (device_obs_dim = 6)**：
| 维度 | 含义 | 范围 |
|------|------|------|
| `obs[0]` | 最大传输速率 (max_trans_rate) | ~0-150 Mbps |
| `obs[1]` | 本地实际计算队列长度 (time_ql) | ≥0 (秒) |
| `obs[2]` | 本地虚拟队列长度 (virtual_time_ql) | ≥0 或 -1（关闭时） |
| `obs[3]` | 任务数据大小 (data_size) | 0.6-2.8 Mb |
| `obs[4]` | 任务计算密度 (comp_dens) | 1.0-4.2 GFLOPs/Mb |
| `obs[5]` | 任务时延约束 (dly_cons) | 0.5-1.0 s |

**边缘观测 (edge_queue_obs_dim = 2)**：
| 维度 | 含义 |
|------|------|
| `实际队列长度 × device_type_num` | 每种类型任务的边缘计算队列长度 |
| `虚拟队列长度 × device_type_num` | 每种类型任务的边缘虚拟队列长度 |

**策略网络输入**: 拼接本地设备观测 + 对应类型的边缘队列观测 = 6 + 2 = 8 维  
**价值网络输入**: 拼接同类型所有设备观测 + 该类型边缘队列观测（集中式 Critic）

### 3.3 动作空间（Action Space）

每个设备的动作包含 **3 个连续维度**：

| 维度 | 含义 | MAPPO 范围 | MADDPG 范围 |
|------|------|-----------|-------------|
| `act[0]` | 任务卸载比例 (offloading ratio) | [0, 1] | [0, 1] |
| `act[1]` | 传输功率利用率 | [0.6, 1] | [0.6, 1] |
| `act[2]` | 本地计算频率利用率 | [0.6, 1] | [0.6, 1] |

- MAPPO: 每个维度输出 1 个标量值（**3 维**），通过 `tanh → scale + loc` 映射到动作范围
- MADDPG: 每个维度用 **10 个标量**表示（**30 维**），每10维取平均后映射

### 3.4 环境建模

#### 设备环境 (DeviceEnv)

- **通信模型**: 香农公式计算传输速率 `R = B * log2(1 + P * g / N0)`
  - 信道增益: `g = 0.1 * d^(-3.5)`（路径损耗模型）
  - 距离 d 由设备移动性模型更新
- **移动模型**: 设备在二维平面运动，支持加速度和方向变化，距边缘服务器 100-500m
- **计算模型**: 本地计算频率 × 利用率，能耗 = `κ * f² * C`
- **队列模型**:
  - 实际队列 `time_ql`: 跟踪待处理计算时间，`Q(t+1) = max(0, Q(t) + C/f - Δt)`
  - 虚拟队列 `virtual_time_ql`: 用于 Lyapunov 优化，跟踪时延约束违反情况

#### 边缘环境 (EdgeEnv)

- **频率分配**: 边缘总计算频率按设备数量比例分配到各任务类型队列
- **调度策略**: 同一类型任务 FIFO，按传输完成时间排序
- **队列模型**: 与实际设备和虚拟队列类似，按类型分别维护

### 3.5 奖励函数设计

奖励由三部分组成：

```
device_reward[i] = base_penalty
                 + naive_reward          (时延/能耗部分)
                 + queue_actual_reward   (实际队列 Lyapunov 漂移)
                 + queue_virtual_reward  (虚拟队列 Lyapunov 漂移)
```

- **Naive 奖励**: 超时则 `timeout_reward_penalty`（-4000），否则能耗加权惩罚
- **实际队列奖励**: `weight * Q_actual * ΔQ`，鼓励减少实际队列长度
- **虚拟队列奖励**: `weight * Q_virtual * ΔQ`，鼓励满足时延约束

**联合奖励 (per-type)**: 同类型所有设备奖励 + 边缘队列奖励

---

## 4. 训练链路（Training Pipeline）

### 4.1 训练循环 — `Controller.train()`

位置：[controller.py:115-173](controller.py#L115)

```
for e_id in [0, train_episodes):
    ├── Rollout.run(e_id) — 收集一个 episode 的数据
    │     ├── 与环境交互 (推理链路)
    │     ├── 存储转换到 replay buffer
    │     └── 触发训练 (基于 train_freq)
    │
    ├── 周期性评估 (每 800 episodes)
    ├── 收集 episode 指标 (rewards, costs, queues, delays, ...)
    └── 周期性保存 (每 1000 episodes)
```

### 4.2 MAPPO 训练流程

**训练频率**: 每 `train_freq=4` 个 episodes 训练一次

**训练数据构建**（[util/replay_buffer.py:101-165](util/replay_buffer.py#L101)）：
1. 从 buffer 中提取最近 `train_freq × buffer_train_time_slots` 个时间槽的数据
2. 构建策略网络训练数据: `p_inputs, acts, act_logprobs, active_masks, lstm_hidden_states`
3. 构建价值网络训练数据: GAE 计算优势函数和 value targets

**价值网络训练**（[agent/edge_agent.py:115-132](agent/edge_agent.py#L115)）：
```
for epoch in [0, v_epochs):           # 4 epochs
    for batch in BatchSampler:
        v_pred = V_net(state)
        loss = MSE(v_target, v_pred)
        loss.backward() + grad_clip + optimizer.step()
```

**策略网络训练**（[agent/edge_agent.py:134-209](agent/edge_agent.py#L134)）：
```
for epoch in [0, p_epochs):           # 4 epochs
    for batch in BatchSampler:
        mean, std = P_net(obs, lstm_hidden)
        new_logprob = log_prob(action | mean, std)  # 逆 tanh 变换
        ratio = exp(new_logprob - old_logprob)
        
        surr1 = ratio * advantage
        surr2 = clamp(ratio, 1-p_clip, 1+p_clip) * advantage
        policy_loss = -min(surr1, surr2)
        entropy_loss = -entropy_coef * entropy
        
        total_loss = (policy_loss + entropy_loss) * active_mask
        total_loss.backward() + grad_clip + optimizer.step()
```

**关键特性**：
- 策略网络使用 **LSTM** 处理时序信息
- **PPO-Clip** 目标函数，clip 范围 `ε=0.1`
- **GAE**（Generalized Advantage Estimation），`λ=0.95, γ=0.99`
- **Active mask**: 只有任务到达的时间槽才计算策略损失
- **标准退火**（Std Annealing）: log_std_max 从 0.5 指数衰减到 -1.5
- **正交初始化**（Orthogonal Init）
- **梯度裁剪**: max_norm=2
- 学习率: `p_lr=1e-4, v_lr=1e-4`

### 4.3 MADDPG 训练流程

**训练频率**: 每 `train_freq` 个时间槽训练一次（=4 episodes × 3000 slots = 12000 slots 一次）

**训练数据采样**（[util/replay_buffer.py:340-406](util/replay_buffer.py#L340)）：
1. 从固定大小 `buffer_size` 的环形缓冲区随机采样 batch
2. 按任务类型分组构建 state、joint_action、reward、next_state

**价值网络训练**（[agent/edge_agent.py:377-409](agent/edge_agent.py#L377)）：
```
for _ in critic_updates_round:        # 1 round
    for epoch in [0, v_epochs):       # 4 epochs
        # 目标 Q 值: r + γ * Q_target(s', μ_target(s'))
        with torch.no_grad():
            next_joint_acts = concat([target_P_i(next_obs_i) for i in type])
            target_Q = target_V(s', next_joint_acts)
            y = r + γ * target_Q
        
        Q = V_net(s, joint_acts)
        loss = MSE(y, Q)
        loss.backward() + grad_clip + optimizer.step()
```

**策略网络训练**（[agent/edge_agent.py:415-439](agent/edge_agent.py#L415)）：
```
for epoch in [0, p_epochs):           # 1 epoch
    # 确定性策略梯度: ∇_θ J ≈ ∇_a Q(s, a) * ∇_θ μ_θ(s)
    new_acts = P_net(obs)             # 当前策略的动作
    joint_acts' = replace(joint_acts, agent_i's part with new_acts)
    p_loss = -V_net(s, joint_acts').mean()
    p_loss.backward() + grad_clip + optimizer.step()
```

**目标网络软更新**（[agent/edge_agent.py:442-455](agent/edge_agent.py#L442)）：
```
θ_target = (1 - τ) * θ_target + τ * θ_online    # τ=0.005
```

**关键特性**：
- 策略网络同样使用 **LSTM**
- 集中式 Critic 输入全局状态 + 联合动作
- **策略延迟**（Policy Delay）: Critic 和 Actor 交替更新频率可配
- **探索噪声**: 高斯噪声，方差从 0.2 线性衰减到 0.02
- **软更新**（Soft Update）目标网络
- 学习率: `p_lr=5e-5, v_lr=1e-4`

### 4.4 训练配置对比

| 维度 | MAPPO | MADDPG |
|------|-------|--------|
| 算法类型 | On-Policy (PPO) | Off-Policy (DDPG) |
| 训练时机 | 每 4 episodes | 每 12000 时间槽 |
| Buffer 类型 | 环形覆盖 | 固定大小环形缓冲 |
| 策略输出 | 随机策略 (mean, std) | 确定性策略 |
| 探索方式 | OU Noise / Gaussian | Gaussian Noise (衰减) |
| 目标网络 | 无（On-Policy） | 软更新 (τ=0.005) |
| 奖励缩放 | 动态 RunningMeanStd | 固定 ×1e-4 |
| 优势估计 | GAE (λ=0.95) | TD 误差 |
| 训练 episodes | 30000 | 15000 |
| 每 episode 时间槽 | 3000 | 3000 |

### 4.5 设备分组架构

默认配置（[config/params.py:68-92](config/params.py#L68)）：
- **10 个设备**，分为 **3 种类型**
  - 类型 0: devices [0,1] — 2 个设备，高优先级/小时延约束
  - 类型 1: devices [2,3,4,5] — 4 个设备
  - 类型 2: devices [6,7,8,9] — 4 个设备
- 每种类型有独立的**边缘计算队列**和**集中式 Critic**
- 每种类型的设备共享同一套超参数配置

---

## 5. 神经网络架构

### 5.1 MAPPO 策略网络 (`MappoPolicyNetLSTM`)

```
输入: [B, obs_dim]  (obs_dim = 8)
  │
  ├── fc1: Linear(8 → 400) + Tanh
  ├── LSTM: (400 → 400, batch_first)
  ├── Tanh
  ├── mu_head: Linear(400 → 3)     # 动作均值
  └── log_std: Parameter[3]        # 动作对数标准差
       │
       └── 输出: mean[B,3], std[B,3], (h_n, c_n)
  
  动作映射: a = tanh(u) * scale + loc
    scale = [5, 2, 2], loc = [5, 8, 8]
  → 最终范围: [[0,10], [6,10], [6,10]]
```

### 5.2 MAPPO 价值网络 (`MappoValueNet`)

```
输入: [B, state_dim]  (state_dim 按队列类型不同)
  ├── fc1: Linear(state_dim → 400) + Tanh
  ├── fc2: Linear(400 → 400) + Tanh
  └── fc3: Linear(400 → 1)
       │
       └── 输出: V[B, 1]  (状态价值)
```

### 5.3 MADDPG 策略网络 (`MaddpgPolicyNetLSTM`)

```
输入: [B, obs_dim]  (obs_dim = 8)
  │
  ├── fc1: Linear(8 → 200) + Tanh
  ├── LSTM: (200 → 200, batch_first)
  ├── Tanh
  └── fc3: Linear(200 → 30) + Tanh
       │
       └── 输出: act[B, 30], (h_n, c_n)
  
  动作映射: 每10维取平均, 然后 scale + loc
    scale = [1, 0.4, 0.4], loc = [1, 1.6, 1.6]
  → 最终范围: [[0,2], [1.2,2], [1.2,2]] → clamp 到 [0,1], [0.6,1], [0.6,1]
```

### 5.4 MADDPG 价值网络 (`MaddpgValueNet`)

```
输入: [B, state_dim + joint_act_dim]
  ├── fc1: Linear(state+act → 200) + Tanh
  ├── fc2: Linear(200 → 200) + Tanh
  └── fc3: Linear(200 → 1)
       │
       └── 输出: Q[B, 1]  (状态-动作价值)
```

### 5.5 网络设计要点

- 全部使用**正交初始化**（Orthogonal Init）以保证训练稳定性
- LSTM 用于捕获时序依赖（队列长度的动态变化）
- 输出层使用小增益初始化（gain=0.01）避免初始策略过强
- MADDPG 的 10 维表示法：每个决策维度用 10 维 one-hot-like 表示，增强表达能力

---

## 6. 数据流与存储

### 6.1 运行时数据流

```
Environment ──(obs)──▶ Device Agents ──(action)──▶ Environment
                          │                            │
                          ▼                            ▼
                    Edge Agent (holds nets)    MECEnv.step()
                          │                      ├── DeviceEnv.compute()
                          │                      ├── EdgeEnv.compute()
                          │                      └── reward calculation
                          ▼
                    Replay Buffer ──(samples)──▶ EdgeAgent.train_nets()
                          │
                          ▼
                    Updated Networks ──(state_dict)──▶ Device Agents
```

### 6.2 持久化存储

| 内容 | 路径 | 频率 |
|------|------|------|
| 训练指标 (rewards/costs/queues) | `result/<run_dir>/*.pkl` | 每 1000 episodes |
| 网络权重 | `weight/<run_dir>/*.pkl` | 训练: 每 1000 eps; MADDPG: 每 6M slots |
| 训练信息 (种子/恢复点) | `weight/<run_dir>/train_info_*.pkl` | 每次保存 |
| TensorBoard 日志 | `runs/<run_dir>/` | 实时 |
| 日志文件 | `log/<run_dir>.log` | 重定向 stdout/stderr |
| 可视化图表 | `runs/plot/<run_dir>/` | 每 800 episodes |

### 6.3 运行目录命名规范

```
{task_type}/{algorithm}_s_{seed}_t_{timestamp}_d_{description}/
例: train/mappo_s_7878_t_2025-06-06-10-30-00-123456_d_vir_queue/
```

---

## 7. CTDE 架构详解

项目采用标准的 **CTDE（集中式训练、分布式执行）** 架构：

### 7.1 分布式执行（Inference）

- 每个设备智能体仅使用**本地观测** + 对应类型的**边缘队列信息**做决策
- 设备间**无通信**，独立做出卸载决策
- 策略网络结构相同但**权重独立**（每个设备有自己的网络参数）

### 7.2 集中式训练（Training）
- **集中式 Critic**: 价值网络输入包含同类型所有设备的观测和动作
  - MAPPO: Critic 输入为全局状态（所有同类型设备观测拼接）
  - MADDPG: Critic 输入为全局状态 + 所有同类型设备动作拼接
- **边缘服务器**（EdgeAgent）持有所有策略网络和价值网络
- 训练完成后，策略网络权重分发给各设备智能体

---

## 8. 评估模式

### 8.1 评估类型

| 模式 | 描述 |
|------|------|
| `mappo` / `maddpg` | 加载训练好的 RL 模型评估 |
| `local_comp` | 基线：全部本地计算 |
| `edge_comp` | 基线：全部边缘卸载 |
| `random_comp` | 基线：随机卸载决策 |

### 8.2 评估流程

1. 加载预训练权重到 EdgeAgent 的网络
2. 分发到 DeviceAgent 进行确定性推理（无噪声）
3. 运行 `eval_episodes` 个独立 episode，收集平均指标
4. 输出 joint_reward, device_rewards, joint_cost, delays, energies, overtime counts

---

## 9. 李雅普诺夫优化框架

### 9.1 核心思想

将长期时延约束转化为**队列稳定性**问题：
- **实际计算队列** `Q(t)`: 跟踪当前待处理的计算负载
- **虚拟队列** `Z(t)`: 编码时延约束的违反情况

### 9.2 队列动态

**实际队列更新**:
```
Q(t+1) = max(0, Q(t) + growth_rate × (computation_time - time_slot_duration))
```

**虚拟队列更新**:
```
Z(t+1) = max(0, Z(t) + growth_rate × (
    Q(t) / avg_comp_time × slot_duration × gen_cycle - delay_threshold
))
```

### 9.3 奖励设计

```
reward = -(V * Q(t) * ΔQ + U * Z(t) * ΔZ)
```

其中 V 和 U 是 Lyapunov 惩罚权重（由 `device_act_queue_reward_weight` 和 `device_vir_queue_reward_weight` 控制），通过调节这两个权重可以权衡队列稳定性（时延约束）和能耗优化。

### 9.4 消融实验支持

通过开关控制：
- `--enable_actual_queue_reward`: 仅实际队列奖励（RT-MADDPG 方法）
- `--enable_virtual_queue_reward`: 实际 + 虚拟队列奖励（论文提出的方法）
- 两者均关闭: 仅 naive 奖励（基线 DRL）

---

## 10. 关键技术细节

### 10.1 观测预处理 (`ObsScaling`)

- **传输速率**: 除以 10 缩放到 ~[0, 15]
- **队列长度**: `log1p` 变换，裁剪到 `[0, 1e6]` 避免数值爆炸

### 10.2 奖励缩放 (`RewardScaling`)

- **MAPPO**: 动态 RunningMeanStd 归一化（含折扣因子 γ）
- **MADDPG**: 固定缩放 ×1e-4

### 10.3 探索策略

- **MAPPO**: OU 噪声（时间相关）或 Gaussian 噪声，带 per-dimension noise_scale
- **MADDPG**: Gaussian 噪声，方差线性衰减（σ: 0.2 → 0.02）

### 10.4 标准退火（MAPPO）

`log_std_max` 从 0.5 指数衰减到 -1.5，τ = total_episodes / 3：
```
cur_log_std_max(t) = -1.5 + 2.0 * exp(-t / τ)
```

### 10.5 梯度控制

- 梯度裁剪: `max_norm = 2`（Value 和 Policy 均适用）
- 学习率衰减（可选）:
  - MAPPO: 指数衰减 `lr *= decay_fac`
  - MADDPG: 线性衰减 `lr -= decay_fac`，每 300k 步

### 10.6 设备移动性

- 在 100-500m 环形区域内随机初始化位置
- 每个时间槽随机加速度和方向变化
- 速度限制在 [0, 5.0] m/s
- 用于计算动态的信道增益（影响传输速率）

---

## 11. 配置系统

### 11.1 参数层级

```
通用参数 (get_general_params)
  ├── 实验配置: evaluate, train_mode, eval_mode, seed, load_weights
  ├── 环境配置: device_num, device_types, bandwidth, task properties, mobility
  ├── 队列配置: growth_rates, reward_weights, reward_bounds
  ├── 噪声配置: ou_noise params
  └── 路径配置: results_dir, weights_dir, plot_dir

算法参数 (get_mappo_params / get_maddpg_params)
  ├── 网络结构: obs_dim, hidden_dims, action_dim
  ├── 训练参数: episodes, time_slots, batch_size, lr, epochs
  ├── RL 参数: gamma, lamda, p_clip, tau, enty_coef
  └── 优化参数: grad_clip, lr_decay
```

### 11.2 默认场景参数

| 参数 | 值 |
|------|-----|
| 设备数 | 10 (类型分布: 2+4+4) |
| 边缘计算频率 | 50 Gcycles/s |
| 总带宽 | 30 MHz |
| 时隙长度 (δ) | 0.1 s |
| 任务生成周期 | 5 个时隙 |
| 传输功率 | 300 mW |
| 设备计算频率 | 2.5 Gcycles/s |

---

## 12. 运行方式

### 训练
```bash
# MAPPO 训练
python main.py --train_mode mappo

# MADDPG 训练（带虚拟队列奖励）
python main.py --train_mode maddpg --enable_virtual_queue_reward

# 从检查点恢复训练
python main.py --train_mode mappo --load_weights --resume_episode 1000
```

### 评估
```bash
# RL 方法评估
python main.py --evaluate --eval_mode mappo --load_weights --weights_dir weight/<run_dir>/

# 基线方法评估
python main.py --evaluate --eval_mode local_comp
python main.py --evaluate --eval_mode edge_comp
python main.py --evaluate --eval_mode random_comp
```

### 消融实验
```bash
# 仅实际队列
python main.py --train_mode mappo --enable_actual_queue_reward

# 仅虚拟队列
python main.py --train_mode mappo --enable_virtual_queue_reward

# 无队列奖励
python main.py --train_mode mappo

# 全部队列
python main.py --train_mode mappo --enable_actual_queue_reward --enable_virtual_queue_reward
```
