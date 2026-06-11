# 项目代码审查与重构方案

> 审查范围：全项目源代码，聚焦正确性、冗余性、可复现性及状态转移逻辑。

---

## 一、 缺陷列表（影响正确性的 Bug）

### 1.1 随机种子隔离失效（严重，影响可复现性）

**位置：** [env/device_env.py:79-81](env/device_env.py#L79-L81) 创建了独立的 `self._rnd` 和 `self._np_rnd`，但后续多处仍使用全局随机状态。

| 文件 | 行号 | 问题 |
|------|------|------|
| [device_env.py:215-219](env/device_env.py#L215-L219) | `reset()` | `np.random.uniform(...)` → 应使用 `self._np_rnd.uniform(...)` |
| [device_env.py:503-506](env/device_env.py#L503-L506) | `compute()` | `np.random.uniform(...)` → 应使用 `self._np_rnd.uniform(...)` |
| [device_env.py:255](env/device_env.py#L255) | `move()` | `random.uniform(...)` → 应使用 `self._rnd.uniform(...)` |
| [device_agent.py:267](agent/device_agent.py#L267) | `RandomComputingDeviceAgent` | `np.random.uniform(...)` → 应使用独立 RNG |

> **后果：** 多设备环境的随机性被全局共享，不同 run 之间即使设置了相同 `seed`，结果也可能不可复现，因为设备间调用全局 RNG 的顺序受代码执行顺序影响，且与独立 RNG 隔离的设计意图矛盾。

### 1.2 任务归一化常数不一致（中等）

**位置：**
- [env/device_env.py:225-226](env/device_env.py#L225-L226) (`reset()` 中 `norm_csum_engy = comp * self.engy_fac * 6.25`)
- [env/device_env.py:513-514](env/device_env.py#L513-L514) (`compute()` 中 `norm_csum_engy = comp * self.engy_fac * 9`)

> **后果：** 每个 episode 的第一个任务（在 `reset()` 中生成）使用系数 6.25，后续任务（在 `compute()` 中生成）使用系数 9。`norm_csum_engy` 在 [mec_env.py:205-206](env/mec_env.py#L205-L206) 中参与 reward 计算，导致同一 episode 内不同时间步的任务归一化尺度不一致，Lyapunov 优化目标出现系统性偏差。

### 1.3 非活跃状态的 log_prob 处理错误（中等）

**位置：** [agent/device_agent.py:87-88](agent/device_agent.py#L87-L88)

```python
if not active:
    return [0.0] * self.action_dim, 0.0, next_lstm_hidden_h, next_lstm_hidden_c
```

- 当 `active=False`（无任务到达），返回的 `act_logprob` 为 `0.0`（而非 `None`）。
- 在 [rollout.py:276-277](rollout.py#L276-L277) 中检查 `act_logprob == None` 失败，`0.0` 被存入 replay buffer。
- 在 [edge_agent.py:176-177](agent/edge_agent.py#L176-L177) PPO ratio 计算中，`old_act_logprobs = 0.0` 会使得 `exp(new_logp - 0) = exp(new_logp)`，产生极端 ratio，导致梯度爆炸或训练不稳定。
- 虽然最终被 `active_masks` 掩码过滤（[edge_agent.py:194](agent/edge_agent.py#L194)），但在 `active_masks` 应用之前，0 logprob 的数值问题已经影响了 loss 计算（`ratios * adv_b` 中 inactive 位置的 ratio 非 1）。

> **修复建议：** 返回 `act_logprob = None` 以保持与非活跃状态检查逻辑一致，或在计算 ratio 时用 mask 提前排除。

### 1.4 edge_env 队列更新与 reward 计算的时间步不一致（中等）

**位置：**
- [env/edge_env.py:156](env/edge_env.py#L156)：`if t_id % self.gen_task_cycle == 0` 触发 edge queue 更新
- [env/mec_env.py:271](env/mec_env.py#L271)：`if t_id % gen_task_cycle == start_slot` 触发 edge queue reward 计算

> **后果：** 当 `start_slot != 0`（虽然默认值为 0），queue 更新和 reward 计算会错位一个或多个时间步。当前默认参数下 (`start_slot=0`) 不触发，但参数系统预留了 `start_slot` 配置项，为潜在的静默错误留下隐患。

### 1.5 MADDPG replay buffer 采样越界（中等）

**位置：** [agent/edge_agent.py:335-340](agent/edge_agent.py#L335-L340)

```python
batch_slots = (total_time_slots + self.gen_task_cycle - 1)//self.gen_task_cycle
if batch_slots < self.buffer_size:
    batch_ids = np.random.choice(range(batch_slots),
                                    self.train_batch_size, replace=False)
```

- 当 `batch_slots < self.train_batch_size` 时（训练初期），`np.random.choice(..., replace=False)` 会抛出 `ValueError`。
- `train_batch_size` 的默认值 = `3000/5 * 4 = 2400`（[config/params.py:528](config/params.py#L528)），但 buffer 初始为 `None` 填充。
- Warm-up 阶段 `warm_time_slots = 12000`（4 × 3000 时间步 ≈ 2400 batch_slots），恰好等于 `train_batch_size`。但如果参数被修改使 `warm_time_slots` 更小，就会出现崩溃。

> 注意：这条与 1.9 关联，`batch_ids` 中可能包含指向未填充 `None` 条目的索引。

### 1.6 设备计算队列在空闲时隙不衰减（中等）

**位置：** [env/device_env.py:309-449](env/device_env.py#L309-L449)

- 传输队列 `trans_ql` 在每个时间步都更新（[line 379](env/device_env.py#L379)），即使 `task_num == 0`（被注释掉的代码 [lines 432-449](env/device_env.py#L432-L449) 曾支持此功能）。
- 计算队列 `time_ql` 和虚拟队列 `virtual_time_ql` 仅在 `task_num >= 1` 时更新。

> **后果：** 实际物理队列（传输队列）在有任务到达时才增长，但也会在无任务时自然衰减；而 Lyapunov 虚拟队列和计算时间队列仅在任务到达时更新。这导致虚拟队列的 drift 项不对等——无任务时虚拟队列停滞，不符合 Lyapunov 优化理论中"每时隙漂移"的假设。

### 1.7 无任务时传输队列衰减逻辑被注释（低）

**位置：** [env/device_env.py:432-449](env/device_env.py#L432-L449)

```python
# else:
#     total_trans_dz = self.trans_ql
#     delta_trans_dz = self.trans_rate * self.delta
#     self.trans_ql = max(0, total_trans_dz - delta_trans_dz)
#     ...queue decay logic...
```

- 注释中保留了空闲时隙的传输队列衰减和计算队列衰减逻辑，但实际代码中 `trans_ql` 的衰减始终在 `task_num >= 1` 分支内执行。
- `trans_ql` 衰减逻辑在 [device_env.py:379](env/device_env.py#L379) 中位于 `if self.task_num >= 1` 块外面——不对，让我重新检查。实际上 line 379 在 `for task_id, offl_dz in enumerate(offl_dzs):` 循环内部，而这个循环在 `task_num >= 1` 分支内。但在循环后的 line 379 `self.trans_ql = max(0, total_trans_dz - delta_trans_dz)` 确实在 `task_num >= 1` 块内。

> 再仔细看：line 379 `self.trans_ql = max(0, total_trans_dz - delta_trans_dz)` 在 `task_num>=1` 块的末尾，但在 `offl_dzs` 循环外面。当 `task_num==0` 时，line 379 不执行，`trans_ql` 不衰减。而这与实际情况不符——传输队列应在每个时隙都服务一部分数据。注释掉的 else 分支（lines 432-436）正是处理这种情况。

### 1.8 `edge_comp_dlys` 变量被覆盖（低）

**位置：** [rollout.py:198-200](rollout.py#L198-L200)

```python
self.comp_dlys = np.zeros(...)     # line 198
self.comp_dlys = np.zeros(...)     # line 199 — 覆盖了 line 198
self.edge_comp_dlys = np.zeros(...) # line 200
```

- `self.comp_dlys` 被赋值两次且值相同，冗余但不影响正确性。
- `self.edge_comp_dlys` 被赋值但**从未被读取**——在 `run()` 返回值中也没有使用它。这表明开发者原计划记录 edge 侧延迟但中途放弃了。

### 1.9 MaddpgReplayBuffer 存储了 task_num==0 时隙的数据（低）

**位置：** [rollout.py:349-358](rollout.py#L349-L358) 与 [rollout.py:294-296](rollout.py#L294-L296)

- 当 `task_num == 0` 时，`device_acts[i] = [-1.0 for _ in range(self.action_dim)]` 被存入 buffer。
- 边缘 case：`device_acts_[i] = [-1.0 for _ in range(self.action_dim // 10)]` 传入 env.step，但 device_env.compute() 中 `task_num == 0` 时不读取 action。
- 虽然逻辑上无影响，但这些 `-1.0` 伪动作会被存入 replay buffer 用于训练，可能导致 Critic 学习到伪动作对应的 Q 值。

---

## 二、 冗余代码

### 2.1 重复 import

| 文件 | 行号 | 内容 |
|------|------|------|
| [rollout.py:2-3](rollout.py#L2-L3) | L2, L3 | `from operator import xor` 重复两次 |
| [rollout.py:222](rollout.py#L222) | L222 | 函数体内 `import numpy as np`，模块顶部 L5 已导入 |
| [env/mec_env.py:1](env/mec_env.py#L1) | L1 | `from env import device_env` 未使用 |
| [config/params.py:3](config/params.py#L3) | L3 | `from multiprocessing.connection import deliver_challenge`（明显是误粘贴） |
| [config/params.py:5](config/params.py#L5) | L5 | `from torch import device` 未使用 |

### 2.2 重复赋值

| 文件 | 行号 | 内容 |
|------|------|------|
| [device_env.py:83-87](env/device_env.py#L83-L87) | L83, L86 | `self.device_type` 和 `self.device_type_num` 被赋值两次 |
| [rollout.py:175-177](rollout.py#L175-L177) | L175, L177 | `self.comp_dlys` 被赋值两次 |

### 2.3 死代码

| 文件 | 行号 | 说明 |
|------|------|------|
| [env/mec_env.py:75](env/mec_env.py#L75) | L75 | `ThreadPoolExecutor(max_workers=self.device_num)` 创建后从未使用 |
| [env/mec_env.py:89-91](env/mec_env.py#L89-L91) | L89-91 | `enable_print` 始终为 `False`，所有 `if enable_print:` 调试分支永不执行 |
| [agent/device_agent.py:33-36](agent/device_agent.py#L33-L36) | L33-36 | `reset_ou()` 方法定义了但从未被调用 |
| [util/utils.py:48-49](util/utils.py#L48-L49) | L48-49 | `edge_obs_ = edge_obs` / `device_obss_ = device_obss` 是空操作 |
| [util/utils.py:88-97](util/utils.py#L88-L97) | L88-97 | 旧版 `GaussianNoise` 类被注释掉 |
| [util/utils.py:126-134](util/utils.py#L126-L134) | L126-134 | `GetValueInputs` 函数被注释掉 |
| [network/policy_net.py:151-152](network/policy_net.py#L151-L152) | L151-152 | `mean_scale = 1.0` 然后 `mean = mean * mean_scale`，乘以1 无效 |
| [rollout.py:327](rollout.py#L327) | L327 | `self.average_always(...)` 被注释掉 |
| [agent/device_agent.py:5](agent/device_agent.py#L5) | L5 | 导入了 `MappoPolicyNet`, `MaddpgPolicyNet`（非 LSTM 版本），从未使用 |
| [network/policy_net.py:6-69](network/policy_net.py#L6-L69) | L6-69 | `MappoPolicyNet`（非 LSTM）类定义了但从未被实例化 |
| [network/policy_net.py:163-183](network/policy_net.py#L163-L183) | L163-183 | `MaddpgPolicyNet`（非 LSTM）类定义了但从未被实例化 |

### 2.4 重复的代码块

| 位置 | 说明 |
|------|------|
| [agent/device_agent.py:63-77](agent/device_agent.py#L63-L77) vs [agent/device_agent.py:180-195](agent/device_agent.py#L180-L195) | `to_hidden()` 函数在 `MappoDeviceAgent.choose_action` 和 `MaddpgDeviceAgent.choose_action` 中完全重复 |
| [rollout.py:134-138](rollout.py#L134-L138) vs [rollout.py:184-185](rollout.py#L184-L185) | `enable_actual_queue_reward` 和 `enable_virtual_queue_reward` 的 debug 打印重复两次 |
| [util/replay_buffer.py:57-65](util/replay_buffer.py#L57-L65) vs [util/replay_buffer.py:263-281](util/replay_buffer.py#L263-L281) | `add()` 辅助函数在 `MappoReplayBuffer` 和 `MaddpgReplayBuffer` 中重复出现四次（`add`, `add_next` 各两次） |

### 2.5 未使用的参数

| 文件 | 参数 | 说明 |
|------|------|------|
| [config/params.py:108-111](config/params.py#L108-L111) | `task_arrival_prob` | 定义了但代码中始终使用固定周期生成任务 (`task_num=1`)，概率从未参与决策 |
| [config/params.py:233-235](config/params.py#L233-L235) | `max_data_size`, `max_comp_dens` | 定义了但从未在代码中使用 |
| [config/params.py:255-261](config/params.py#L255-L261) | 已注释的 `vir_*_ql_growth_rate` | 被替代为统一的 `device_vir_queue_growth_rate` |
| [env/mec_env.py:14](env/mec_env.py#L14) | `edge_energy_weights` | 代码中被注释掉（[line 193](env/mec_env.py#L193)），实际只用 device_energy_weights |

---

## 三、 架构/设计问题

### 3.1 脆弱的算法模式派发

**位置：** [rollout.py:255](rollout.py#L255), [rollout.py:278](rollout.py#L278), [rollout.py:297](rollout.py#L297)

```python
if "Mappo" in type(self.device_agents[0]).__name__:
```

使用类名字符串匹配来决定控制流。如果类名重构或新增算法，会静默失败。MADDPG 和静态 agent 分支同理。

**建议：** 使用 `isinstance()` 检查或引入显式的 `alg_type` 枚举/字符串属性。

### 3.2 stdout/stderr 重定向丢失原始流

**位置：** [rollout.py:157-158](rollout.py#L157-L158)

```python
sys.stdout = log_txt_file
sys.stderr = log_txt_file
```

- 原始 `sys.stdout` / `sys.stderr` 未保存，无法恢复。
- 影响异常堆栈的可读性——所有错误信息静默写入日志文件。
- `atexit.register(log_txt_file.close)` 注册了文件关闭，但不会恢复 stdout/stderr。

### 3.3 `is_evaluate` 标志非异常安全

**位置：** [controller.py:135-137](controller.py#L135-L137)

```python
gp.settings.is_evaluate = True
self.rollout.run(e_id, visualize=visualize)
gp.settings.is_evaluate = False
```

如果 `self.rollout.run()` 抛出异常，`is_evaluate` 保持 `True`，后续训练运行将错误地使用评估模式（如确定性动作选择、不同的 reward scaling 行为）。

### 3.4 controller 中条件表达式逻辑隐患

**位置：** [controller.py:15-16](controller.py#L15-L16)

```python
if (not gen_params.evaluate and gen_params.train_mode == "mappo") or \
   (gen_params.evaluate and gen_params.eval_mode) == "mappo":
```

`(gen_params.evaluate and gen_params.eval_mode) == "mappo"` 依赖 Python 的 `and` 返回最后真值（`eval_mode` 字符串）然后与 `"mappo"` 比较。碰巧工作正常，但极易被误读。

**同样的问题在** [controller.py:63](controller.py#L63)：

```python
self.seed = gen_params.evaluate and gen_params.eval_seed or gen_params.train_seed
```

如果 `eval_seed = 0`（falsy 值），会错误地取 `train_seed`。虽然 seed=0 不太可能被使用，但这是一个潜伏的 bug。

### 3.5 非 LSTM 网络版本闲置

**位置：** [network/policy_net.py:6-69](network/policy_net.py#L6-L69), [network/policy_net.py:163-183](network/policy_net.py#L163-L183)

- `MappoPolicyNet` 和 `MaddpgPolicyNet`（无 LSTM）已实现但未被任何代码路径使用。
- agent 代码中总是实例化 LSTM 版本（如 [agent/device_agent.py:17](agent/device_agent.py#L17)）。
- 这些旧版本存在逻辑差异（如 `MaddpgPolicyNet` 输出 `tanh(x)+1` 映射到 [0,2]，而 LSTM 版本只输出 `tanh(x)` 映射到 [-1,1]），如果未来有人切换版本可能导致 action 范围错误。

### 3.6 动作值缺少范围校验

**位置：** [env/device_env.py:312-320](env/device_env.py#L312-L320)

```python
offl_rto = act[0]               # 无 clamp
trpw_rto = act[1]               # 无 clamp
device_comp_rto = act[2]        # 无 clamp
```

- 虽然策略网络输出经过 tanh 和缩放约束，但存在数值精度问题或 MADDPG 噪声导致超出 [0,1] 或 [0.6,1] 范围的可能性。
- 特别是 MADDPG `act + noise` 后虽然做了 `torch.clamp(act, -1.0, 1.0)`，但缩放后范围是 [0,2] 和 [1.2,2]，没有在此处再次 clamp。

---

## 四、 状态转移问题

### 4.1 OU 噪声状态跨 episode 漂移

**位置：** [agent/device_agent.py:33-36](agent/device_agent.py#L33-L36)

- `MappoDeviceAgent._ou_state` 在 episode 之间不重置（`reset_ou()` 存在但未被调用）。
- OU 过程是有记忆的随机过程——跨 episode 的状态延续会使 episode 间的探索行为不再独立。
- 在 [rollout.py:188-204](rollout.py#L188-L204) 的 `reset()` 中未调用 `device_agent.reset_ou()`。

### 4.2 LSTM 隐藏状态正确重置，但初始值为零列表

**位置：** [rollout.py:218-219](rollout.py#L218-L219)

```python
lstm_hidden_hs = [[0.0 for _ in range(self.lstm_hidden_dim)] for _ in range(self.device_num)]
```

- LSTM 隐藏状态以 Python float 列表形式初始化，在每个 episode 开始时正确归零。
- 但当 `self.lstm_hidden_dim = -1`（评估模式下静态 agent）时，创建了空列表，后续代码可能因索引异常而出错。
- 实际评估模式下 `self.lstm_hidden_dim = -1`（[rollout.py:74](rollout.py#L74)），导致 `range(-1)` 为空，`lstm_hidden_hs` 成为空列表的列表。幸好静态 agent 不使用 lstm hidden state，但这是洁癖问题。

### 4.3 观测缩放时机与存储的关系

**位置：** [rollout.py:226-227](rollout.py#L226-L227) 和 [rollout.py:334-335](rollout.py#L334-L335)

```
时间线（一个 episode 内）：
1. 获取原始 obs → 缩放 → 策略网络推理（使用缩放后的 obs）
2. env.step() → 获取 next_obs（原始）
3. 平均统计量 → 缩放 next_obs
4. 存储 obs（缩放后）到 replay buffer → 训练时使用缩放后的观测
5. obs = next_obs（缩放后）
```

流程是正确的，但注意训练时 replay buffer 中的数据已经过 `ObsScaling`，训练数据始终是 log1p 压缩后的值。这保证了训练/推理的数据分布一致。

### 4.4 average() 方法中的统计偏差

**位置：** [rollout.py:456-476](rollout.py#L456-L476)

```python
self.joint_rewards += 1 / gen_t_id_ * (joint_rewards - self.joint_rewards)
```

- 使用在线 Welford 风格的 running mean 公式。该公式给出精确的平均值。
- 但部分统计量（`device_csum_engys`, `device_esum_engys`, `device_overtime_nums`）使用的是累加 `+=` 而非 running mean，在 episode 结束时也未除以任务数。
- `device_task_avail_nums` 在 [line 474-476](rollout.py#L474-L476) 中被累加，注释掉的部分（[lines 478-482](rollout.py#L478-L482)）本应做除法归一化。

---

## 五、 重构方案

### 5.1 高优先级修复（影响正确性/可复现性）

| # | 任务 | 涉及文件 | 预计改动量 |
|---|------|----------|-----------|
| 1 | **统一随机数生成器**：所有设备环境的随机操作改用 `self._rnd` / `self._np_rnd` | `device_env.py`, `device_agent.py` | ~10 行 |
| 2 | **统一归一化常数**：将 `norm_csum_engy` 的计算提取为 `DeviceEnv` 的方法或统一常量 | `device_env.py` | ~5 行 |
| 3 | **修复 inactive log_prob**：`active=False` 时返回 `None` 或正确处理 mask | `device_agent.py`, `rollout.py`, `edge_agent.py` | ~5 行 |
| 4 | **OU 噪声重置**：在 `Rollout.reset()` 中调用 `device_agent.reset_ou()` | `rollout.py` | ~3 行 |
| 5 | **MADDPG 采样安全**：`np.random.choice` 前检查 `batch_slots >= train_batch_size`，不足时使用 `replace=True` 或减小 batch_size | `edge_agent.py` | ~3 行 |

### 5.2 中优先级改进（代码质量/可维护性）

| # | 任务 | 涉及文件 | 预计改动量 |
|---|------|----------|-----------|
| 6 | **消除冗余 import 和重复赋值**：移除 [二](#二-冗余代码) 节列出的所有冗余 | 7 个文件 | ~15 行删除 |
| 7 | **消除死代码**：移除 `ThreadPoolExecutor`、`enable_print` 基础设施（或恢复其功能） | `mec_env.py`, `device_env.py`, `edge_env.py` | ~20 行 |
| 8 | **提取公共 `to_hidden` 函数** 到 `util/utils.py`，消除 device_agent 中的代码重复 | `device_agent.py`, `util/utils.py` | ~20 行 |
| 9 | **算法模式派发改为显式枚举**：引入 `AlgType` 枚举或使用 `isinstance()` 检查 | `rollout.py`, `controller.py` | ~15 行 |
| 10 | **修复 `is_evaluate` 的异常安全**：使用 `try/finally` 或上下文管理器 | `controller.py` | ~5 行 |
| 11 | **修复 controller 条件表达式**：使用显式的三元表达式/if-else | `controller.py` | ~3 行 |
| 12 | **添加动作值 clamp**：在 `device_env.compute()` 中对 `offl_rto`, `trpw_rto`, `device_comp_rto` 添加 `np.clip` | `device_env.py` | ~3 行 |

### 5.3 低优先级改进（设计与一致性）

| # | 任务 | 涉及文件 | 预计改动量 |
|---|------|----------|-----------|
| 13 | **保存并恢复 stdout/stderr**：使用 `contextlib.redirect_stdout` 或手动保存 | `rollout.py` | ~5 行 |
| 14 | **移除未使用的非 LSTM 网络类** 或添加可切换的配置项 | `policy_net.py`, `agent/device_agent.py` | ~30 行 |
| 15 | **统一 edge_env 与 mec_env 的时间步触发条件**：将硬编码 `== 0` 替换为 `== start_slot` | `edge_env.py` | ~2 行 |
| 16 | **提取公共 `add()` 辅助函数**：ReplayBuffer 中的 inner function 重复 4 次，提取为静态方法 | `replay_buffer.py` | ~30 行 |
| 17 | **启用空闲时隙的设备队列衰减**：恢复 [device_env.py:432-449](env/device_env.py#L432-L449) 被注释的逻辑并验证 | `device_env.py` | ~15 行 |
| 18 | **移除未使用的配置参数**：`task_arrival_prob`, `max_data_size`, `max_comp_dens`, `edge_energy_weights` | `params.py`, `mec_env.py` | ~15 行删除 |
| 19 | **在训练开始时保存超参数快照**：连同 `train_info.pkl` 一起保存完整的超参配置，便于实验追踪 | `edge_agent.py` | ~10 行 |

### 5.4 建议的重构顺序

```
Phase 1（先修 bug，保证实验正确）
  └─ #1 随机种子 → #2 归一化常数 → #3 inactive logprob → #4 OU noise → #5 MADDPG 采样

Phase 2（清理代码，提高可维护性）
  └─ #6 删除冗余 → #7 删除死代码 → #8 提取公共函数 → #9 显式枚举派发

Phase 3（加固设计）
  └─ #10/#11 异常安全 → #12 action clamp → #13 stdout → #15 时间步统一 → #17 队列衰减
```

---

## 六、 额外建议

### 6.1 添加单元测试

当前项目完全没有测试。建议至少覆盖：
- `ObsScaling` 的 log1p 变换正确性
- `RewardScaling` 的 MAPPO/MADDPG 路径
- `ReplayBuffer` 的存储/采样循环正确性
- `RunningMeanStd` 的统计量精度（当前 n=1 时 std 行为异常——见 [util/utils.py:19](util/utils.py#L19) `self.std = x` 应为零）

### 6.2 日志系统改进

当前使用 `sys.stdout` 重定向实现日志记录，建议改用 Python `logging` 模块，支持：
- 多级别日志（DEBUG/INFO/WARNING）
- 同时输出到控制台和文件
- 结构化格式便于解析

### 6.3 配置管理

`config/params.py` 使用 argparse，但参数定义中存在 `type=list` 等非标准用法。建议：
- 迁移到 YAML/JSON 配置文件 + dataclass
- 或使用 `argparse` 的 `nargs` 和自定义 type 正确解析 list 参数
- 记录每次运行的完整配置快照到实验目录

---

> **审查完成日期：** 2026-06-06
> **审查范围：** 全项目 18 个 Python 源文件
> **发现缺陷总数：** 9 个 Bug + 6 类冗余 + 4 个可复现性问题 + 5 个设计问题
>
> ---
>
> ## 七、 重构完成记录
>
> **重构日期：** 2026-06-09
>
> ### Phase 1 — 关键 Bug 修复 (全部完成)
>
> | # | 修复内容 | 文件 | 状态 |
> |---|---------|------|------|
> | 1 | 随机种子隔离：`gen_position()`、`move()`、`reset()`、`compute()` 改用 `self._rnd`/`self._np_rnd`；`RandomComputingDeviceAgent` 添加独立 RNG | `device_env.py`, `device_agent.py` | ✅ |
> | 2 | 归一化常数统一：`reset()` 和 `compute()` 中 `norm_csum_engy` 均使用系数 9 | `device_env.py` | ✅ |
> | 3 | 非活跃 log_prob：`active=False` 时返回 `None`（与 rollout 中的 `None` 检查一致） | `device_agent.py` | ✅ |
> | 4 | OU 噪声重置：`Rollout.reset()` 中遍历 device_agents 调用 `reset_ou()` | `rollout.py` | ✅ |
> | 5 | MADDPG 采样安全：`batch_slots < train_batch_size` 时使用 `replace=True` | `edge_agent.py` | ✅ |
>
> ### Phase 2 — 代码清理 (全部完成)
>
> | # | 修复内容 | 文件 | 状态 |
> |---|---------|------|------|
> | 6 | 移除冗余 import (`from operator import xor` 重复、`import numpy as np` 重复、`from env import device_env`、`deliver_challenge`、`from torch import device`) 和重复赋值 (`comp_dlys`、`device_type`/`device_type_num`) | `rollout.py`, `mec_env.py`, `params.py`, `device_env.py` | ✅ |
> | 7 | 移除死代码：`ThreadPoolExecutor`、`enable_print=False` 硬编码块、`mec_env.py` 中未使用的 `torch`/`math` import、`utils.py` 中 no-op 赋值和注释掉的旧类/函数、`device_agent.py`/`edge_agent.py` 中未使用的 `MappoPolicyNet`/`MaddpgPolicyNet` import、`policy_net.py` 中 no-op `mean_scale` | 多个文件 | ✅ |
> | 8 | 提取公共 `to_lstm_hidden()` 到 `util/utils.py`，消除 `MappoDeviceAgent` 和 `MaddpgDeviceAgent` 中的重复代码 | `utils.py`, `device_agent.py` | ✅ |
> | 9 | 算法派发改为 `isinstance()` 检查（`MappoDeviceAgent`/`MaddpgDeviceAgent`/`StaticDeviceAgent`） | `rollout.py` | ✅ |
> | 10 | 提取公共 `_append_flat()` 辅助函数，消除 replay buffer 中 4 处重复的 inner function | `replay_buffer.py` | ✅ |
>
> ### Phase 3 — 设计加固 (全部完成)
>
> | # | 修复内容 | 文件 | 状态 |
> |---|---------|------|------|
> | 11 | `is_evaluate` 异常安全：使用 `try/finally` 确保异常时恢复 | `controller.py` | ✅ |
> | 12 | 修复 controller 条件表达式（显式 `==` 比较替代 `and/or` 短路求值）+ seed 使用三元表达式 | `controller.py` | ✅ |
> | 13 | 动作值 clamp：`offl_rto` → `[0,1]`、`trpw_rto`/`device_comp_rto` → `[0.6,1]` | `device_env.py` | ✅ |
> | 14 | 保存原始 `sys.stdout`/`sys.stderr` 到 `self._orig_stdout`/`self._orig_stderr` | `rollout.py` | ✅ |
> | 15 | 修复 `edge_env` 时间步触发条件：`== 0` → `== self.start_slot` | `edge_env.py` | ✅ |
> | 16 | 启用空闲时隙队列衰减逻辑（恢复被注释的 else 分支） | `device_env.py` | ✅ |
> | 17 | 移除未使用的参数：`task_arrival_prob` refs、`max_data_size`、`max_comp_dens`、`edge_energy_weights` refs + 清理注释掉的代码 | `params.py`, `device_env.py`, `mec_env.py` | ✅ |
> | 18 | 修复 `RunningMeanStd` n=1 时 std 应为 0 而非 x | `utils.py` | ✅ |
>
> ### 额外修复
>
> | # | 修复内容 | 文件 | 状态 |
> |---|---------|------|------|
> | 19 | 移除 `mec_env.py` 中注释掉的 `edge_energy_weights` 残留引用 | `mec_env.py` | ✅ |
> | 20 | 移除 `device_env.py` 中未使用的局部变量 `data_size_mean`、`comp_dens_mean` | `device_env.py` | ✅ |
>
> ### 未处理的项目（低优先级，保留供未来迭代）
>
> - 非 LSTM 网络类 (`MappoPolicyNet`, `MaddpgPolicyNet`)：保留以供可能的架构对比实验
> - `enable_print` 调试基础设施：保留变量定义和调试分支，仅移除了总是设为 False 的硬编码
> - `edge_comp_dlys` 变量：保留了定义（尽管未被使用），因为可能是未来功能扩展的占位
