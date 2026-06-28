# 系统模型

本文档总结当前代码中使用的通信模型、移动模型与计算模型。

---

## 1. 通信模型 (Communication Model)

通信模型刻画终端设备到边缘服务器之间的无线信道传输过程。

### 1.1 信道增益

采用自由空间路径损耗模型（指数为 3.5），信道增益为：

$$G = 0.1 \cdot d^{-3.5}$$

其中 $d$ 为设备到目标边缘服务器的三维欧式距离（单位：m），距离至少为 1 m 以避免奇异。

$$d = \sqrt{(x_{dev} - x_{srv})^2 + (y_{dev} - y_{srv})^2 + z_{dev}^2}$$

设备高度固定为 $z_{dev} = 1.8$ m。

### 1.2 传输速率

每个设备的传输速率由香农公式给出：

$$R_{tx} = B \cdot \log_2\left(1 + \frac{P_{tx} \cdot G}{N_0}\right) \cdot 10^{-6} \quad \text{(Mb/s)}$$

| 参数 | 含义 | 值 |
|------|------|-----|
| $B$ | 每设备子带宽 (Hz) | $B_{total} / N_{dev}$ |
| $B_{total}$ | 系统总带宽 | $30 \times 10^6$ Hz |
| $N_{dev}$ | 设备总数 | 10 |
| $P_{tx}$ | 设备发射功率 (mW) | 300 mW $\times$ 功率利用率 |
| $N_0$ | 噪声功率 (mW) | $\sigma_{spec} \times B$ |
| $\sigma_{spec}$ | 噪声功率谱密度 (mW/Hz) | $10^{-174/10}$ |

### 1.3 传输时延与能耗

任务的卸载传输时间为：

$$T_{trans} = T_{queue}^{tran} + \frac{D_{offl}}{R_{tx}}$$

传输能耗为：

$$E_{trans} = P_{tx} \cdot 10^{-3} \cdot \frac{D_{offl}}{R_{tx}}$$

其中 $D_{offl}$ 为卸载数据量 (Mb)，$T_{queue}^{tran}$ 为传输队列等待时间。

传输队列更新（FIFO）：

$$Q_{trans} \leftarrow \max(0,\ Q_{trans} + D_{offl}^{total} - R_{tx} \cdot \Delta \cdot T_{cycle})$$

---

## 2. 移动模型 (Mobility Model)

移动模型描述终端设备在圆形区域内的随机加速运动。

### 2.1 初始位置

设备初始位置在圆环 $[d_{min}, d_{max}]$ 内均匀采样（以原点为中心的极坐标采样）。

### 2.2 运动学更新

每个时隙，设备的加速度是随机采样的：

$$a \sim U(a_{min}, a_{max}) \quad \text{(大小, m/s}^2\text{)}$$

$$\theta_a \sim U(\theta_{min}, \theta_{max}) \quad \text{(方向, 度)}$$

加速度分解为笛卡尔分量：

$$a_x = a \cdot \cos(\theta_a \cdot \pi / 180)$$

$$a_y = a \cdot \sin(\theta_a \cdot \pi / 180)$$

速度更新（含限幅）：

$$v_x \leftarrow v_x + a_x \cdot \delta$$

$$v_y \leftarrow v_y + a_y \cdot \delta$$

$$v \leftarrow \text{clamp}\left(\sqrt{v_x^2 + v_y^2},\ v_{min},\ v_{max}\right)$$

位置更新（仅当新位置在边界圆环内时接受）：

$$x \leftarrow x + v_x \cdot \delta$$

$$y \leftarrow y + v_y \cdot \delta$$

$$\text{仅当 } d_{min} \leq \sqrt{x^2 + y^2} \leq d_{max} \text{ 时生效}$$

### 2.3 移动参数

| 参数 | 含义 | 值 |
|------|------|-----|
| $\delta$ | 时隙长度 | 0.1 s |
| $v_{min}$ / $v_{max}$ | 速度范围 | 0.0 / 5.0 m/s |
| $a_{min}$ / $a_{max}$ | 加速度大小范围 | 0.0 / 4.0 m/s² |
| $\theta_{min}$ / $\theta_{max}$ | 加速度方向范围 | -60° / 60° |
| $d_{min}$ / $d_{max}$ | 边界距离范围 | 100 / 500 m |

### 2.4 边缘服务器位置

3 个边缘服务器均匀分布在半径 $R_{srv} = 250$ m 的圆上，角度分别为 0°、120°、240°。

---

## 3. 计算模型 (Computation Model)

计算模型分为本地计算（设备端）和边缘计算（服务器端）两部分，以及相应的队列动态。

### 3.1 任务模型

每个设备每 $T_{cycle} = 5$ 个时隙生成一个任务。任务属性：

| 属性 | 符号 | 单位 | 来源 |
|------|------|------|------|
| 数据大小 | $D$ | Mb | 在区间内均匀采样 |
| 计算密度 | $C$ | Gcycles/Mb | 在区间内均匀采样 |
| 延迟约束 | $T_{thre}$ | s | 配置参数 |

不同设备类型的任务参数：

| 设备类型 | $D$ 区间 (Mb) | $C$ 区间 (Gcycles/Mb) | $T_{thre}$ (s) |
|----------|---------------|----------------------|-------------------|
| 0 | [1.0, 2.0] | [0.0, 8.0] | $5 \times 0.1 = 0.5$ |
| 1 | [2.0, 4.0] | [0.0, 4.0] | $10 \times 0.1 = 1.0$ |
| 2 | [4.0, 8.0] | [0.0, 2.0] | $10 \times 0.1 = 1.0$ |

每类设备的数量：[4, 4, 2]，共 10 台设备。

### 3.2 动作空间

每个设备的动作为 4 维向量 $[\text{server\_id}, \alpha_o, \alpha_p, \alpha_f]$：

| 维度 | 符号 | 含义 | 范围 |
|------|------|------|------|
| 0 | server_id | 目标边缘服务器索引 | {0, 1, 2} |
| 1 | $\alpha_o$ | 卸载比例 | 经 clip 后 [0.6, 1.0] |
| 2 | $\alpha_p$ | 发射功率利用率 | 经 clip 后 [0.6, 1.0] |
| 3 | $\alpha_f$ | 本地计算频率利用率 | 经 clip 后 [0.6, 1.0] |

### 3.3 本地计算（设备端）

卸载数据量：

$$D_{offl} = D \cdot \alpha_o$$

本地计算量：

$$L_{local} = (D - D_{offl}) \cdot C$$

本地处理时延：

$$T_{local}^{proc} = \frac{L_{local}}{f_{dev} \cdot \alpha_f}$$

其中 $f_{dev} = 3.0$ Gcycles/s 为设备 CPU 基准频率。

本地队列时延：

$$T_{local}^{queue} = \max(0,\ T_{total\_comp} - t \cdot \delta)$$

本地总时延：

$$T_{local} = T_{local}^{queue} + T_{local}^{proc}$$

本地能耗：

$$E_{local} = \kappa \cdot (f_{dev} \cdot \alpha_f)^2 \cdot L_{local}$$

其中 $\kappa = 1$ J/Gcycles 为能量系数。

### 3.4 实际队列（设备端）

本地计算队列基于 Lyapunov 漂移-加-罚框架更新：

$$Q_{dev}(t+1) = \max\left(0,\ Q_{dev}(t) + \gamma_{dev}^{act} \cdot \left(\frac{L_{local}}{f_{dev} \cdot \alpha_f} - \Delta \cdot T_{cycle}\right)\right)$$

其中 $\gamma_{dev}^{act} = 1$ 为实际队列增长率。

虚拟队列更新（用于 Lyapunov 引导）：

$$\hat{Q}_{dev}(t+1) = \max\left(0,\ \hat{Q}_{dev}(t) + \gamma_{dev}^{vir} \cdot \left(\frac{Q_{dev}(t)}{\bar{T}_{local}} \cdot \delta \cdot T_{cycle} - V_{dev}^{adj}\right)\right)$$

其中：
- $\bar{T}_{local}$ 为最近 100 个时隙的平均本地计算时间
- $V_{dev}^{adj} = 0.8 \cdot T_{thre}$ 为设备延迟调节因子
- $\gamma_{dev}^{vir} = 1$ 为虚拟队列增长率

### 3.5 边缘计算（服务器端）

每个任务在边缘服务器的处理（FIFO 调度，按传输时间排序）：

边缘处理时延：

$$T_{edge}^{proc} = \frac{D_{offl} \cdot C}{f_{edge}}$$

其中 $f_{edge} = 50$ Gcycles/s 为边缘服务器 CPU 频率。

边缘队列时延：

$$T_{edge}^{queue} = \max(0,\ T_{total\_comp}^{edge} - t \cdot \delta) + T_{new\_queue}$$

边缘总时延：

$$T_{edge} = \max(T_{edge}^{queue},\ T_{trans}) + T_{edge}^{proc}$$

边缘能耗（对于每个任务）：

$$E_{edge} = D_{offl} \cdot C \cdot f_{edge}^2$$

### 3.6 实际队列（边缘端）

每个边缘服务器维护单 FIFO 队列，仅在任务生成周期边界更新：

$$Q_{edge}(t+T_{cycle}) = \max\left(0,\ Q_{edge}(t) + \gamma_{edge}^{act} \cdot \left(T_{need} - T_{used}\right)\right)$$

其中：
- $T_{need}$ 为本周期所有卸载任务 $T_{edge}^{proc}$ 之和
- $T_{used} = \min(Q_{edge}(t), \Delta \cdot T_{cycle})$ 为当前时隙可处理量

边缘虚拟队列：

$$\hat{Q}_{edge}(t+T_{cycle}) = \max\left(0,\ \hat{Q}_{edge}(t) + \gamma_{edge}^{vir} \cdot \left(\frac{Q_{edge}}{\bar{T}_{edge}} \cdot \frac{\delta \cdot T_{cycle}}{\bar{N}_{dev}} - V_{edge}^{adj}\right)\right)$$

其中：
- $\bar{T}_{edge}$ 为最近 100 个时隙的平均边缘处理时间
- $\bar{N}_{dev} = 5$ 为平均设备数
- $V_{edge}^{adj} = 0.8 \cdot \min(T_{thre})$ 为边缘延迟调节因子

### 3.7 总任务完成时延

任务的整体完成时延取本地与边缘的较大值（并行执行）：

$$T_{task} = \max(T_{local},\ T_{edge})$$

若 $T_{task} > T_{thre}$，则判定为**超时**。

### 3.8 计算参数汇总

| 参数 | 含义 | 值 |
|------|------|-----|
| $f_{dev}$ | 设备 CPU 频率 | 3.0 Gcycles/s |
| $f_{edge}$ | 边缘 CPU 频率 | 50 Gcycles/s |
| $\kappa$ | 能量系数 | 1 J/Gcycles |
| $\delta$ | 时隙长度 | 0.1 s |
| $T_{cycle}$ | 任务生成周期 | 5 时隙 |
| $\gamma_{dev}^{act}$ | 设备实际队列增长率 | 1 |
| $\gamma_{dev}^{vir}$ | 设备虚拟队列增长率 | 1 |
| $\gamma_{edge}^{act}$ | 边缘实际队列增长率 | 1 |
| $\gamma_{edge}^{vir}$ | 边缘虚拟队列增长率 | 1 |
| 统计窗口 | 平均计算时间统计的时隙数 | 100 |
