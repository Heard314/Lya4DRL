# 训练算法流程

## 整体架构：CTDE

集中式训练、分散式执行。三台边缘服务器（S=3），每台一个 FIFO 计算队列，10 个终端设备（device_num=10），3 种任务类型（device_type_num=3）。

**网络结构：**
| 网络 | MADDPG | MAPPO |
|---|---|---|
| Policy Net | MLP(fc1→fc2→fc3→tanh) 输出 act | MLP(fc1→fc2→mu_head) 输出 (mean, std) |
| Value Net | MLP, 输入 (state, joint_act) 输出 Q | MLP, 输入 state 输出 V |
| Target Net | 有 (Polyak soft update) | 无 |
| Action 探索 | Gaussian noise 衰减 | 可学习 std + annealing |

---

## MADDPG 训练流程

### 1. 数据收集（per time-slot）
```
for t_id in 0..3000:
    if t_id % gen_task_cycle(5) == 0:  # 每5个slot一个gen cycle
        device_agent.choose_action(obs) → (tanh→noise→clamp) → env_act
        env.step(env_acts)
        replay_buffer.store(edge_obs, device_obss, device_acts, joint_rewards, next_obs)
```

### 2. 存储结构
- `MaddpgReplayBuffer`: 环形缓冲区 `buffer_size = 24000`
- 每个 gen_cycle 存一条：`(edge_obs, device_obss, device_acts, joint_rewards, next_edge_obs, next_device_obss)`
- `joint_rewards` 维度 `[3]`（3 个任务类型的联合奖励）

### 3. 训练触发（per time-slot）
```
total_time_slots = e_id * 3000 + t_id + 1
if total_time_slots % train_freq(12000) == 0:   # 每4个episode
    train_nets()
    update_target_nets()                        # 每8个episode
```

### 4. Value Net 训练（Critic）
```
1. 从 buffer 采样 batch_size=2400 条
2. 构建 batch_states[i] (per type) = [edge_obs(2S) + device_obs(device_type)] * type中设备数
3. 构建 batch_joint_acts[i] = 各设备动作压缩 (S+3 维 per device → joint)
4. Target Q = batch_joint_rewards + gamma * target_v_net(batch_next_states, next_acts)
5. Loss = MSE(Q(s,a), Target Q)
6. critic_updates_round=1 次更新
```

### 5. Policy Net 训练（Actor）
```
每 policy_delay_round(2) 次 Critic 更新 → 1 次 Actor 更新:
1. 用当前 policy 生成新动作
2. 在 joint_acts 中替换本设备动作
3. Loss = -v_net(state, joint_acts_with_new_policy).mean()
4. 梯度只通过被替换的动作部分回传
```

### 6. Target Net 更新
```
Polyak: target = (1-tau)*target + tau*online, tau=0.005
频率 = maddpg_time_slots * maddpg_update_freq = 3000*8 = 24000 slots
```

---

## MAPPO 训练流程

### 1. 数据收集（per gen_cycle）
```
for t_id in 0..3000:
    if t_id % gen_task_cycle(5) == 0:
        mean, std = policy_net(obs)
        action = tanh(mean + std*noise) * scale + loc  # reparameterization
        env.step()
        replay_buffer.store(edge_obs, device_obss, device_acts, act_logprobs, joint_rewards, device_active)
```

### 2. 存储结构
- `MappoReplayBuffer`: `[train_freq=4][buffer_train_time_slots+1=601]` 二维缓冲区
- `ps[0]` = 当前 episode (0~3), `ps[1]` = 当前 slot (0~600)
- 每个 episode 存 600 条（gen_task_cycle=5, 3000/5=600），第 601 条触发 episode 切换
- 索引 600 为 bootstrap 帧，仅用于 Value 计算

### 3. 训练触发（per episode）
```
if (e_id+1) % train_freq(4) == 0:  # 每4个episode，buffer刚好满
    train_nets(buffer)
```

### 4. GAE 计算（per task type）
```
1. 构建 v_inputs [4, 601, state_dim]
2. 用 value_net 算所有状态值 V(s_t) [4, 601]
3. rewards = joint_rewards[:, :600, type_idx]
4. deltas = rewards + gamma * V[:, 1:601] - V[:, 0:600]
5. GAE 反向累积: adv[:, t] = delta_t + lambda*gamma*adv[:, t+1]
6. value_targets = advs + V[:, 0:600]
7. advantage normalize: (adv - mean) / std
```

### 5. Value Net 训练
```
Loss = MSE(v_net(v_inputs), value_targets)
v_epochs=4, batch_size=2400
```

### 6. Policy Net 训练
```
for each device in type:
    1. 从存储的 env_action 反推 tanh 前的 u:
       a = (env_action - bias) / scale       # [-1, 1]
       u = atanh(a)                           # (-∞, +∞)
    2. 用当前 policy 算 new_logp = log N(u | mean, std) - squash_correction
    3. ratio = exp(new_logp - old_logp)
    4. PPO-clip: L = min(ratio*adv, clip(ratio, 1-ε, 1+ε)*adv)
    5. 只对 active_device 的样本计算 loss
    6. Loss = -(clip_loss + entropy_coef * entropy)
p_epochs=4, p_clip=0.2
```

### 7. Std Annealing
```
每 episode 开始时 set_episode(e_id)，更新 log_std_max:
log_std_max = LOG_STD_MAX_FINAL + (INIT - FINAL) * exp(-episode / tau)
tau = train_episodes / 3
```

---

## 关键参数对比

| 参数 | MADDPG | MAPPO |
|---|---|---|
| train_episodes | 20000 | 30000 |
| train_time_slots | 3000 | 3000 |
| gen_task_cycle | 5 | 5 |
| train_freq (训练频率) | 12000 slots (≈4 eps) | 4 eps |
| batch_size | 2400 | 2400 (2400=4×600) |
| v_epochs | 4 | 4 |
| p_epochs | 1 | 4 |
| gamma | 0.99 | 0.99 |
| v_lr | 1e-4 | 1e-4 |
| p_lr | 5e-5 | 1e-4 |
| 探索方式 | Gaussian noise 衰减 | 可学习 std + annealing |
| Target network | Polyak τ=0.005 | 无 |
| GAE | 无（TD(0)） | λ=0.95 |
| buffer 大小 | 24000（循环） | 4×601=2404 |

---

## 潜在问题分析

### 1. MADDPG 训练频率与 buffer 填充速率不匹配

`train_freq = 12000 slots`，但每个 episode 3000 slots = 4 episodes 训练一次。
buffer 每次 gen_cycle(5 slots) 存一条。12000 slots ÷ 5 = 2400 条。
第一次训练时 batch_slots=2400，正好等于 train_batch_size=2400，即用全部数据。
`warm_time_slots = 12000` 恰好等于第一次训练节点，warm 边界无浪费。

**结论：无问题。**

### 2. MADDPG Policy 更新时 Critic 梯度残留（已修复）

修复前 `set_requires_grad` 被注释，Policy 的 `p_loss.backward()` 在 Critic 参数上
留下梯度残留，直到下一次训练（4 episode 后）的 Critic `zero_grad()` 才清除。
虽然 Critic 已 `step` 过不会被直接污染，但不规范且浪费显存。
**已修复：** Policy 更新前 `set_requires_grad(v_net, False)`，更新后恢复 `True`。

### 3. MAPPO PPO-clip 的 advantage 按类型共享

同一类型各设备的 Policy 使用同一个 `advs_`（从该类型的 Value Net 计算）。
这符合 CTDE 理论：joint_reward 属于整个 type，所有 device 共享同一个 advantage。

**结论：无问题。**

### 4. MAPPO silent advantage re-normalization

Policy 训练中只对 active_device 的 advantage 做二次归一化：
```python
m = (mask_b > 0)
if m.any():
    adv_sel = adv_b[m]
    adv_sel = (adv_sel - adv_sel.mean()).div(adv_sel.std().clamp_min(1e-8))
```
这意味着 active 和非 active 样本的 advantage 不在同一尺度。
非 active 样本（device_active=0）不参与 loss 计算（被 mask 消去），所以不影响训练。

**结论：无问题，但 non-standard PPO 实现。**

### 5. MADDPG joint_reward 中边缘奖励被所有类型共享

每个 type 的 joint_reward 都包含全量 edge_queue_rewards + 该 type 的 device_rewards。
三个 Value Net 都尝试预测包含相同边缘分量的目标，存在冗余但理论上不影响收敛。

**结论：无问题，CTDE 的正常设计。**

### 6. Sample reuse 频率

- MADDPG: 每 4 个 episode 做一次训练，buffer=24000 条。当 buffer 满后，每次从 24000 条中抽 2400 条。数据利用率 = 2400/24000 = 10%/次。
- MAPPO: 每 4 个 episode 做一次训练，buffer=2400 条全部使用。数据利用率 = 100%，但 epochs=4 意味着每条数据被用 4 次。

**结论：MAPPO 样本效率更高，MADDPG 样本多样性更高。**

### 7. Std Annealing 边界条件

`std_anneal_tau = train_episodes / 3`。当 train_episodes < 3 时 tau=0 导致 exp(-t/0) = NaN。
正常训练（20000+ episodes）不受影响。

**结论：仅快速测试时触发，不影响正常训练。**

### 8. 随机种子控制

`train_seed` 和 `eval_seed` 默认均为 7878。
Controller.__init__ 中 `self.seed = train_seed`，然后在 Rollout 中 `torch.manual_seed + np.random.seed`。
Python 消融实验脚本通过 `sys.argv` 传递参数，不显式指定 `--train_seed` 时使用 params.py 的默认值。

**结论：修改 params.py 中 train_seed 的默认值即可改变所有实验的随机种子。**
