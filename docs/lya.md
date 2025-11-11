

应用场景为：

任务的到达表现为泊松分布，计算量呈现异构性。

该论文假设网速较快，能达到50Mbps

# Lyapunov优化

## 队列模型

每个设备都需要维护两个队列，实际工作排队模型与延迟感知虚拟排队模型。此外，边缘服务器上需要为每个设备维护两个队列，也是实际工作排队模型与延迟感知虚拟排队模型。

###  设备上的队列模型

（1）实际工作排队模型

在每个时隙$t$，设备会以一定概率生成一个任务，以FIFO的方式进入实际工作队列$Q_{local}(t)=(Q_{local,1}(t),Q_{local,2}(t),\ldots,Q_{local,N}(t))$，其中$Q_i(t)$是第$i$个设备在第$t$个时隙中的任务积压，$N$为设备数量。默认情况下，所有队列在初始时刻都是空的。

实际工作排队模型的更新公式如下所示：
$$
Q_{local,i}(t+1)=\max\{Q_{local,i}(t)+C_{local,i}(t)-A_{local,i}(t),0\}
$$
定义$A_{local,i}(t)$为第$t$个时隙中本地处理的工作负载量，$A_{local,i}(t)=f_{local,i}\tau$，$f_{local,i}$为本地的计算频率，$\tau$是单时隙的持续时间。

其中$p_i(t)$是第$t$个时隙到达的任务的数据卸载率，$C_{local,i}(t)$表示该任务在本地处理的计算量，$C_{local,i}(t) = (1-p_i) \times D_i(t) \times I_i(t)$，当没有任务到达时，$D_i(t)=0$。$I_i(t)$是第$t$个时隙的单位计算强度。

（2）延迟感知虚拟排队模型

对于每一个队列$Q_i(t)$，设置一个虚拟队列$H_i(t)$，两个队列相结合以实现任务卸载过程中的队列稳定性与延时可控。

在每个时隙$t$，设备会生成一个固定的、虚拟计算任务（计算量为常量$\epsilon_{local,i}$），以FIFO的方式放入到虚拟工作队列$H_{local}(t)=(H_{local,1}(t),H_{local,2}(t),\ldots,H_{local,N}(t))$，其中$H_i(t)$是第$i$个设备在第$t$个时隙中的任务积压，$N$为设备数量。默认情况下，所有队列在初始时刻都是空的。

虚拟工作排队模型的更新公式如下所示：
$$
H_{local,i}(t+1)=\max\{H_{local,i}(t)+\epsilon_{local,i}-A_{local,i}(t),0\}
$$
$C_i(t)$同上。

不同任务的$\epsilon_{local,i}$值会有所不同，与其任务类型的属性有关，假设设备$i$处理的任务类型为$k$。
$$
\epsilon_{local,i}= \mu \frac{\mathbb{E}[D_k]\times \mathbb{E}[I_k] \times \mathbb{E}[P_k]}{d_{i}^c(t)}
$$
其中，$0<\mu<1$是一个调整因子，$D_k，I_k，P_k$分别表示任务类型k的数据大小，单位计算密度与到达概率。

### 服务器上的队列模型

（1）实际工作排队模型

边缘服务器按照任务类型分为$K$个工作队列，其中$K$为工作队列的数量（也是任务类型的数量），服务器以任务类型的单位时间计算量的期望值为权重，分配计算频率。

在每个时隙$t$，各个设备的卸载任务到达边缘服务器，服务器将按照其任务类型分配到相应的工作队列中，工作队列$Q_{edge}(t)=(Q_{edge,1}(t),Q_{edge,2}(t),\ldots,Q_{edge,K}(t))$将按照FIFO的方式进行处理。默认情况下，所有队列在初始时刻都是空的。

实际工作排队模型的更新公式如下所示：
$$
Q_{edge,i}(t+1)=\max\{Q_{edge,i}(t)+\sum_{j \in S_i} C_{edge,j}(t)-A_{edge,i}(t),0\}
$$
其中$S_i$为任务类型为$i$的设备的编号集合，$f_{edge,i}$表示边缘服务器上任务类型为$i$的队列所分配到的计算频率。定义$B_i(t)$为第$t$个时隙中卸载到远程服务器上的工作负载量。

$$
C_{edge,i}(t) = p_i(t) \times D_i(t) \times I_i(t)
$$
定义$A_{edge,i}(t)$为第$t$个时隙中远程卸载处理的工作负载量，$A_{edge,i}(t)=f_{edge,i}\tau$，$f_{local,i}$为本地的计算频率，

（2）延迟感知虚拟排队模型

同样地，对于每一个队列$Q_i(t)$，设置一个虚拟队列$H_i(t)$，两个队列相结合以实现任务卸载过程中的队列稳定性与延时可控。

在每个时隙$t$，服务器为每个队列$H_{edge,i}$会生成一个固定的、虚拟计算任务（计算量为常量$\epsilon_{edge,i}$），工作队列$H_{edge}(t)=(H_{edge,1}(t),H_{edge,2}(t),\ldots,H_{edge,K}(t))$将按照FIFO的方式进行处理。默认情况下，所有队列在初始时刻都是空的。

实际工作排队模型的更新公式如下所示：
$$
H_{edge,i}(t+1)=\max\{H_{edge,i}(t)+\epsilon_{edge,i}-A_{edge,i}(t),0\}
$$
$f_{edge,i}$表示边缘服务器上任务类型为$i$的队列所分配到的计算频率。

不同队列的$\epsilon_{edge,i}$值会有所不同，与其任务类型的属性有关。
$$
\epsilon_{edge,i}= \lambda {\sum_{j \in S_i} \frac{f_{edge,i}}{f_{edge,i}+f_{local,j}} \mathbb{E}[D_j]\times \mathbb{E}[I_j] \times \mathbb{E}[P_j]} \times \frac{1}{d_{i}^c(t)}
$$
附：

当队列$Q(t)$和$H(t)$保持有界时，最坏情况的时间延迟也保持有界，即$Q(t) \leq Q^{\max}$和$H(t) \leq H^{\max}$成立，最坏情况延迟$T^{\max}$有下式成立：
$$
T^{\max} = \lceil \frac{Q^{\max}+H^{\max}}{\epsilon}  \rceil
$$
具体证明可见论文《Lyapunov-Guided Delay-Aware Energy Efficient Offloading in IIoT-MEC Systems》中的Appendix D。

## 优化目标

通过联合优化设备在连续时间槽槽中的传输功率和任务的卸载比例，在尽可能保证任务的延迟阈值和保证计算队列稳定性的约束下，最小化系统能耗，实现分布式和协同的实时动态任务卸载。优化目标如下：
$$
\begin{aligned}
(\mathrm{P1}):\ & \min_{\mathcal{P},\,\Theta}\ \mathbb{E} \left\{ 
\lim_{T \to \infty} \frac{1}{T} 
\sum_{t=1}^{T} \sum_{i=1}^{N} c_{i}(t)
\right\} \\
\text{s.t.}\quad 
& (\mathrm{C1}):\ p_i(t) \in [0,\, p_i^{\max}], \quad 1 \le i \le N.\\
& (\mathrm{C2}):\ \theta_{i}(t) \in [0,\,1], \quad 1 \le i \le N.\\
& (\mathrm{C3}):\ d_{i}(t) \le d^c_{i}(t), \quad 1 \le i \le N.\\
& (\mathrm{C4}):\ \lim_{t\to\infty}\frac{\mathbb{E}\!\left[Q_{local,i}(t)\right]}{t}=0,  1 \le n \le N.\\
& (\mathrm{C5}):\ \lim_{t\to\infty}\frac{\mathbb{E}\!\left[H_{local,i}(t)\right]}{t}=0,  1 \le n \le N.\\
& (\mathrm{C6}):\ \lim_{t\to\infty}\frac{\mathbb{E}\!\left[Q_{edge,i}(t)\right]}{t}=0,  1 \le n \le N.\\
& (\mathrm{C7}):\ \lim_{t\to\infty}\frac{\mathbb{E}\!\left[H_{edge,i}(t)\right]}{t}=0,  1 \le n \le N.\\
\end{aligned}
$$
C1为传输速率约束，C2为任务卸载率约束，C3旨在确保任务的延迟不超过指定的阈值，C4~C7为队列稳定性保证。

通过引入Lyapunov优化，可以将C4-C7转换为Lyapunov漂移项放入优化目标中。

Lyapunov函数$L(\mathbf{Q}(t))$如下：
$$
L(\mathbf{Q}(t)) = \tfrac{1}{2}\!\left(\sum_{i=1}^{N} Q_{local,i}(t)^2 + \sum_{i=1}^{N} H_{local,i}(t)^2 + \sum_{i=1}^{K} Q_{edge,i}(t)^2 + \sum_{i=1}^{K} H_{edge,i}(t)^2\right)
$$
Lyapunov漂移如下：
$$
\begin{aligned}
\Delta L (Q_{local}(t))
&= \mathbb{E}[L (Q_{local,i}(t+1))-L (Q_{local,i}(t))\mid Q(t)]\\
&= \mathbb{E}[\frac{1}{2}(\sum_{i=1}^{N} Q_{local,i}(t+1)^2-\sum_{i=1}^{N} Q_{local,i}(t)^2)\mid Q(t)]\\
&\leq  \mathbb{E}[\frac{1}{2}\sum_{i=1}^{N}(Q_{local,i}(t)+C_{local,i}(t)-A_{local,i}(t))^2-Q_{local,i}(t)^2\mid Q(t)]\\
&=\mathbb{E} [ \sum_{i=1}^{N}\big(\frac{1}{2}(C_{local,i}(t)-A_{local,i}(t)^2) + Q_{local,i}(t)\big(C_{local,i}(t)-A_{local,i}(t)\big)\mid Q(t)]\\
&\leq Z_{local,Q} +  \mathbb{E}\!\left[\sum_{i=1}^{N} Q_{local,i}(t)\big(C_{local,i}(t)-A_{local,i}(t)\big)\,\middle|\, Q(t)\right] \\
\end{aligned}
$$

$$
\begin{aligned}
\Delta L (H_{local}(t))
&= \mathbb{E}[L (H_{local,i}(t+1))-L (H_{local,i}(t))\mid Q(t)]\\
&= \mathbb{E}[\frac{1}{2}(\sum_{i=1}^{N} H_{local,i}(t+1)^2-\sum_{i=1}^{N} H_{local,i}(t)^2)\mid Q(t)]\\
&\leq  \mathbb{E}[\frac{1}{2}\sum_{i=1}^{N}(H_{edge,i}(t)+\epsilon_{edge,i}-A_{edge,i}(t))^2-H_{local,i}(t)^2\mid Q(t)]\\
&=\mathbb{E} [ \sum_{i=1}^{N}\big(\frac{1}{2}(\epsilon_{edge,i}-A_{edge,i}(t))^2 + H_{local,i}(t)\big(\epsilon_{edge,i}-A_{edge,i}(t)\big)\mid Q(t)]\\
&\leq Z_{local,H} +  \mathbb{E}\!\left[\sum_{i=1}^{N} H_{local,i}(t)\big(\epsilon_{edge,i}-A_{edge,i}(t)\big)\,\middle|\, Q(t)\right] \\
\end{aligned}
$$

$$
\begin{aligned}
\Delta L (Q_{local,i}(t))
&= \mathbb{E}[L (Q_{local,i}(t+1))-L (Q_{local,i}(t))\mid Q(t)]\\
&= \mathbb{E}[\frac{1}{2}(\sum_{i=1}^{N} Q_{local,i}(t+1)^2-\sum_{i=1}^{N} Q_{local,i}(t)^2)\mid Q(t)]\\
&\leq  \mathbb{E}[\frac{1}{2}\sum_{i=1}^{N}(Q_{local,i}(t)+C_{local,i}(t)-A_{local,i}(t))^2-Q_{local,i}(t)^2\mid Q(t)]\\
&=\mathbb{E} [ \sum_{i=1}^{N}\big(\frac{1}{2}(C_{local,i}(t)-A_{local,i}(t)^2) + Q_{local,i}(t)\big(C_{local,i}(t)-A_{local,i}(t)\big)\mid Q(t)]\\
&\leq Z_{local,i} +  \mathbb{E}\!\left[\sum_{i=1}^{N} Q_{local,i}(t)\big(C_{local,i}(t)-A_{local,i}(t)\big)\,\middle|\, Q(t)\right] \\
\end{aligned}
$$








$$
\begin{aligned}
\Delta (\mathbf{Q}(t)) 
&= \mathbb{E}\!\left[\,L(\mathbf{Q}(t+1)) - L(\mathbf{Q}(t)) \,\middle|\, \mathbf{Q}(t)\right]\\
&\leq Z + \mathbb{E}\!\left[
\sum_{i=1}^{N} Q_{local,i}(t)\big(C_{local,i}(t)-A_{local,i}(t)\big)
\,\middle|\, Q(t)\right] \\
&\quad + \mathbb{E}\!\left[
\sum_{i=1}^{N} H_{local,i}(t)\big(\epsilon_{local,i}-A_{local,i}(t)\big)
\,\middle|\, Q(t)\right] \\
&\quad + \mathbb{E}\!\left[
\sum_{i=1}^{K} Q_{edge,i}(t)\big(C_{edge,i}-A_{edge,i}(t)\big)
\,\middle|\, Q(t)\right] \\
&\quad + \mathbb{E}\!\left[
\sum_{i=1}^{K} H_{edge,i}(t)\big(\epsilon_{edge,i}-A_{edge,i}(t)\big)
\,\middle|\, Q(t)\right].
\end{aligned}
$$




其中$Z$为一个常数。

带漂移惩罚的优化目标$\mathrm{P2}$如下式所示：
$$
\Gamma \mathrm{P1} + \Delta (\mathbf{Q}(t))
$$





















