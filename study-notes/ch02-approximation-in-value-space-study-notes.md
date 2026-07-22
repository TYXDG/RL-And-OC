# 第 2 章 Approximation in Value Space — 分节笔记

> **文献**：Dimitri P. Bertsekas, *Reinforcement Learning and Optimal Control*（Athena Scientific, 2019 draft）第 2 章。  
> **文本依据**：`source/ch02_clean.txt` 及 `source/parts/ch02_part*.txt`。  
> **定位**：在已知最优余值 $J_k^*$ 时，一步前瞻 (1.9) 给出最优控制；本章系统讨论以 $\tilde J_k$ 替代 $J_k^*$ 时的构造、实现与性能机理。有限时域为主，结论可延拓至第 4–5 章无限时域情形。

---

## 章首：值空间近似与策略空间近似

### 有限时域一步前瞻的标准形式

在随机有限时域问题中，次优策略由逐步最小化定义：

$$
\tilde\mu_k(x_k) \in \arg\min_{u_k \in U_k(x_k)} \mathbb{E}\big\{ g_k(x_k,u_k,w_k) + \tilde J_{k+1}\big(f_k(x_k,u_k,w_k)\big) \big\}. \tag{2.1–2.2}
$$

$\tilde J_{k+1}$ 近似 $J_{k+1}^*$；$\ell>1$ 时推广为多步前瞻（§2.2），终端仍用 $\tilde J_{k+\ell}$ 截断。

**两个正交的设计维度**（Fig. 2.1.1）：

1. **$\tilde J_k$ 的构造**（离线/在线、问题近似、rollout、参数化、聚合等）。  
2. **式 (2.1) 中关于 $u_k$ 与 $\mathbb{E}[\cdot]$ 的实现**（穷举、连续优化、确定性等价、Q 回归、Monte Carlo 等）。

### 策略空间近似及其与值近似的组合

策略空间方法在受限策略族 $\mu_k(x_k,r_k)$ 上直接优化。值空间方案的一个常用两阶段流程为：

- (a) 由 $\tilde J_k$ 经 (2.1) 得到 $\tilde\mu_k$；  
- (b) 在样本 $(x_k^s,u_k^s)$ 上回归 $\mu_k(x_k,r_k)$，其中 $u_k^s=\tilde\mu_k(x_k^s)$（式 (2.8)），以降低在线 min 的计算负担。

本章主体为值空间近似；策略回归作为 §2.1.5 的接口，与第 3–4 章衔接。

### Model-based 与 model-free（本书界定）

本书以**是否用采样估计期望**区分，而非单纯“有无 $f,g$ 的闭式表达”：

| 类型 | 含义 |
|------|------|
| **Model-based** | $p_k(w_k|x_k,u_k)$、$f_k,g_k$ 可用闭式（或确定性问题无期望），(2.1) 中期望由代数运算完成。 |
| **Model-free** | 期望由 Monte Carlo 模拟估计；**有模型但故意采样**亦归入 model-free。 |

确定性问题通常属 model-based；棋类等确定性博弈仍可能因 MCTS、随机策略而大量使用采样（§2.4.2）。真实系统采样本书不展开。

---

## §2.1 General Issues of Approximation in Value Space

### 总述

值空间近似方案分解为：(1) 计算前瞻函数 $\tilde J_k$；(2) 执行 (2.1) 的控制选择。二者可独立设计与组合。

---

### §2.1.1 Methods for Computing Approximations in Value Space

四类构造 $\tilde J_k$ 的主路径：

**(a) 问题近似（§2.3）**  
解一个**结构或概率结构更简单**的关联问题，以其最优（或近最优）余值作为 $\tilde J$。包括分解、忽略不确定性、状态空间缩减；**聚合**作为特例，在本书后续版本中与第 5 章及他书 [Ber12] 衔接。

**(b) 在线近似优化（§2.4–§2.5）**  
以**基策略/启发式**在线滚动仿真估计尾部代价。Rollout、MPC 为代表；基策略可来自 (a) 或其他来源。

**(c) 参数化余值（第 3 章）**  
$\tilde J_k(x_k,r_k)$，由特征 $\phi_k(x_k)$ 与训练算法确定 $r_k$。

**(d) 聚合**  
代表状态上的精确 DP + 插值，或分区后的聚合 MDP；可与 (a)–(c) 叠加作局部修正。

上述方法可与 **$u_k$ 的近似 min**、**期望的确定性等价**（§2.3.2）、**自适应采样 / MCTS**（§2.4.2）组合。

---

### §2.1.2 Off-Line and On-Line Methods

**离线**：控制过程开始前，对全部阶段 $k$ 与（原则上）全部 $x_{k+1}$ 计算并存储 $\tilde J_{k+1}$；在线仅查表或快速求值。神经网络、聚合多为离线。

**在线**：当前状态 $x_k$ 已知后再计算所需的 $\tilde J_{k+1}(x_{k+1})$ 并完成 (2.1)。仅对实际访问的 $N$ 个状态求控，适合**再规划**与数据时变。Rollout、MPC 典型为在线。

问题近似可离线或在线，取决于子问题求解方式。**混合**亦常见：离线训练 $\tilde J$，在线 rollout 精修。

---

### §2.1.3 Model-Based Simplification of the Lookahead Minimization

设 $f_k,g_k$ 与 $p_k(w_k|x_k,u_k)$ 已知，且 (2.2) 中期望**不用** Monte Carlo。

**确定性等价（assumed certainty equivalence）**  
取扰动典型值 $\tilde w_k$，解确定性问题：

$$
\tilde\mu_k(x_k) \in \arg\min_{u_k \in U_k(x_k)} \Big[ g_k(x_k,u_k,\tilde w_k) + \tilde J_{k+1}\big(f_k(x_k,u_k,\tilde w_k)\big) \Big]. \tag{2.3}
$$

$\tilde J$ 本身亦常由确定性子问题得到（与 §2.3 衔接）。

**关于 $u_k$ 的 min**：

- $U_k(x_k)$ 有限：穷举比较；可并行；整数规划等。  
- 确定性多步前瞻：可化为**最短路径**（label correcting、A* 等，[Ber98], [Ber17]）。  
- 连续控制：离散化或**非线性规划**；MPC（§2.5）为典型。  
- 随机 + 连续：随机规划；或 **Q 因子参数化**（§2.1.4）以分离期望估计与控选择。

---

### §2.1.4 Model-Free Q-Factor Approximation in Value Space

**设定**：有模拟器，对给定 $(x_k,u_k)$ 产生样本 $(x_{k+1},g_k)$；$\tilde J_{k+1}$ 已可得（不必同法）。

**目标**：估计

$$
Q_k(x_k,u_k) = \mathbb{E}\big\{ g_k(x_k,u_k,w_k) + \tilde J_{k+1}(f_k(x_k,u_k,w_k)) \big\}, \tag{2.4}
$$

并对各 $u_k$ 取 min。全控穷举 + 每对多次模拟往往不可行。

**Q 回归流程**（Fig. 2.1.2）：

1. 样本 $(x_k^s,u_k^s,x_{k+1}^s,g_k^s)$，$s=1,\ldots,q$；  
   $\beta_k^s = g_k^s + \tilde J_{k+1}(x_{k+1}^s)$ (2.5)  
2. 拟合 $\tilde Q_k(x_k,u_k,r_k)$：  
   $\bar r_k \in \arg\min_r \sum_s \big(\tilde Q_k(x_k^s,u_k^s,r) - \beta_k^s\big)^2$ (2.6)  
3. $\tilde\mu_k(x_k) \in \arg\min_u \tilde Q_k(x_k,u,\bar r_k)$ (2.7)

**要点**：

- 对 $f,g,p$ 的显式依赖仅体现在采样阶段；策略实现可完全基于模拟器 + $\tilde J$ + $\tilde Q$ 架构。  
- **双重近似**：$\tilde J_{k+1}$ 与 $\tilde Q_k$ 误差独立累积；(2.7) 的解一般**不等于**对 (2.2) 直接 min 的解。  
- 可用正则化最小二乘；架构见第 3 章。

---

### §2.1.5 Approximation in Policy Space on Top of Value Space

参数化策略 $\tilde\mu_k(x_k,r_k)$，在样本 $(x_k^s,u_k^s)$ 上最小化

$$
\sum_{s=1}^q \big\| u_k^s - \tilde\mu_k(x_k^s,r_k) \big\|^2 \tag{2.8}
$$

（可加正则）。$u_k^s$ 可来自专家，或来自值近似：

$$
u_k^s \in \arg\min_u \mathbb{E}\big\{ g_k(x_k^s,u,w_k) + \tilde J_{k+1}(f_k(x_k^s,u,w_k)) \big\}, \tag{2.9}
$$

或 $\arg\min_u \tilde Q_k(x_k^s,u,\bar r_k)$（(2.10)）。  
**优势**：训练后在线执行 $\tilde\mu_k(x,r_k)$ 无需重复 (2.9) 级优化；与纯策略空间方法同享此性质。

---

### §2.1.6 When is Approximation in Value Space Effective?

**$ \tilde J_k \approx J_k^*$ 非必要**：若 $\tilde J_k(x)-J_k^*(x)\equiv c$，(2.1) 仍得最优策略。

**相对余值**：$\tilde J_k(x)-\tilde J_k(x') \approx J_k^*(x)-J_k^*(x')$ 是更合理的启发，但仍忽略首段代价在排序中的作用。

**Q 误差斜率**（Fig. 2.1.3）：设 $u_k^*$ 最优、$\tilde u_k$ 最小化 $\tilde Q_k$。若 $Q_k(x_k,u)-\tilde Q_k(x_k,u)$ 在 $u_k^*,\tilde u_k$ 附近**变化平缓**（小“斜率”），则 $\tilde u_k$ 的 $Q_k$ 损失通常可控；若 $Q_k$ 与 $\tilde Q_k$ 仅差**与 $u$ 无关的常数**，两式 min 等价。**Advantage** 与 Q 差分在存在函数逼近误差时往往更稳健（第 3 章 §3.4）。

**局限**：缺乏通用的、与问题无关的次优性证书；评估策略质量仍多依赖问题结构与仿真。此为近似 DP/RL 的共性困难。

---

## §2.2 Multistep Lookahead

### 定义与实现

**$\ell$ 步前瞻**：在 $(k,x_k)$ 上优化 $u_k,\ldots,u_{k+\ell-1}$，终端代价 $\tilde J_{k+\ell}$；**仅执行**首控 $u_k$，余下丢弃（Fig. 2.2.1）。  
$\ell=2$ 时，内层 $\tilde J_{k+1}(x_{k+1})$ 本身可再为一步前瞻，等价于在终端用 $(\ell-1)$ 阶段 DP 的最优余值近似。

**阶段截断**：当 $k > N-\ell$ 时，前瞻长度应取 $N-k$。

§2.1 的确定性等价、自适应采样、model-free Q 实现均可延拓至多步。

---

### §2.2.1 Multistep Lookahead and Rolling Horizon

取 $\tilde J_{k+\ell}(x)\equiv 0$ 或 $g_N(x)$，以足够长的 $\ell$ 使终端项相对不重要——即**滚动时域（receding horizon）**。无限时域下常取平稳 $\tilde J_k\equiv\tilde J$；折扣问题可用长视界使尾部可忽略，或短 $\ell$ + 终端 $\tilde J$（第 4 章）。

**有效余值** = (a) 最后 $(\ell-1)$ 阶段的精确优化代价 + (b) $\tilde J_{k+\ell}$。$\ell$ 增大时 (b) 权重下降，故**增大 $\ell$ 并不保证策略改进**。

**例 2.2.1**（4 阶段确定性最短路，$\tilde J\equiv 0$）：2 步前瞻选最优 $u$，3 步前瞻选次优 $u'$——因视界“边缘”处代价从 0 突变为 10，更长前瞻反而被误导。

**注**：滚动时域在 $k+\ell$ 步状态分布近似与当前 $(x_k,u_k)$ 无关或集中于低成本态时往往更可靠（脚注讨论）。

---

### §2.2.2 Multistep Lookahead and Deterministic Problems

随机问题每步需求解随机 DP，计算常 prohibitive。**确定性**前瞻可化为有限（或离散化后）**最短路**（Fig. 2.2.3）；连续状态确定性问题可用 NLP——MPC 即此路线。

**部分确定性形式**：当前步保留 $w_k$ 的随机性，将 $w_{k+1},\ldots,w_{k+\ell-1}$ 固定为典型值；$\tilde J_{k+1}$ 由 $(\ell-1)$ 步**确定性**最短路得到，再算

$$
\tilde Q_k(x_k,u_k) = \mathbb{E}\big\{ g_k(x_k,u_k,w_k) + \tilde J_{k+1}(f_k(x_k,u_k,w_k)) \big\},
$$

$\tilde\mu_k \in \arg\min_u \tilde Q_k(x_k,u)$——与 §2.3.2 确定性等价一致。

---

## §2.3 Problem Approximation

核心：用**关联但更简单**的问题的最优余值（或近优余值）作为 $\tilde J$。除聚合外，本节强调：

1. **强制分解（enforced decomposition）**  
2. **概率结构简化（确定性等价类）**

---

### §2.3.1 Enforced Decomposition

适用于子系统通过动力学、代价或约束**弱耦合**的问题（“弱耦合”多凭结构识别，而非单一定义）。

**逐子系统优化**：$u_k=(u_k^1,\ldots,u_k^n)$ 时，固定其余子系统控制为名义值，轮流对某一子系统控制序列做优化——类似**坐标下降**；可多次循环、优化子系统顺序。

**例 2.3.1（车辆路径）**：状态为各车位置 + 已访问节点集合，维数指数增长。单车辆路径子问题可 DP 或启发式求解；一步前瞻中对每个联合移动 $x_{k+1}$，按固定顺序逐车算路径得 $\tilde J_{k+1}(x_{k+1})$，再选最优联合控制（Fig. 2.3.1）。

**约束松弛解耦**：耦合仅出现在控制约束（如资源分配）而子系统动力学可分解时，将耦合约束 $U$ 替换为**更大**的 decoupled 集合 $\bar U\supset U$，解 relaxed 问题得 $\tilde J$——可行原问题需验证。

**例 2.3.2（Restless Multi-Armed Bandit）**：$n$ 个项目，每步仅可做一个；做项目 $i$ 得 $R_i(x_i)$ 且 $x_i$ 按 $f^i$ 演化；不做则按 $f^i$ 被动演化且得 $R^i(x_i)$。完全 DP 状态为 $(x_1,\ldots,x_n)$，维数爆炸。

**可分离近似**：

$$
\tilde J_k(x_1,\ldots,x_n)=\sum_{i=1}^n \tilde J_k^i(x_i).
$$

一步前瞻选 $i$ 最大化（等价形式略去与 $i$ 无关项）：

$$
\tilde m_k^i(x_i)=\arg\max\Big\{R_i(x_i)-R^i(x_i)+\mathbb{E}[\tilde J_{k+1}^i(f^i(x_i,w_i))-\tilde J_{k+1}^i(f^i(x_i,w_i))]\Big\},
$$

各 $\tilde J^i$ 来自**单项目**全视界子问题——经典 RMAB 启发式结构。

**价格/对偶分解**（原文后续）：在资源约束 $\sum_i u_i^i\le C$ 下引入乘子/价格，交替优化子系统 + 更新价格——$\tilde J$ 来自对偶或 Lagrangian 松弛子问题。

**Lagrange 松弛**：硬耦合约束并入惩罚，解较易子问题；$\tilde J$ 为松弛最优值（可能乐观/悲观，需问题结构）。

---

### §2.3.2 Probabilistic Approximation — Certainty Equivalent Control

用**代表场景**或扰动**点预测**（如均值）将随机问题转为确定性，再精确 DP 或前瞻。快速启发式，一般**非最优**；可与 rollout 组合。

**情景法**：终端余值写为 $\tilde J_{k+1}(x_{k+1})=\sum_{s=1}^q r_s C_s(x_{k+1})$，$C_s$ 为场景 $s$ 下尾部代价，$r_s$ 为权重（可时变、可 Monte Carlo 生成场景）。与简化概率模型及 rollout 思想相通。

---

## §2.4 Rollout

**目标**：**策略改进**——给定基策略/启发式 $\bar\pi$，经有限前瞻 + 基策略尾部代价得 rollout 策略 $\tilde\pi$，在序贯一致或序贯改进条件下**不劣于**基策略（第 4 章将 PI 视为反复 rollout）。

**一步形式**：$\tilde J_{k+1}(x_{k+1})=$ 从 $x_{k+1}$ 起运行基策略的 tail 代价。  
**$\ell$ 步推广**：基策略从 $x_{k+\ell}$ 起算 $\tilde J_{k+\ell}$（Fig. 2.4.1）；长视界可**截断** + 终端 $\tilde J$。

基策略为**近似问题的最优策略**时，rollout 等价于以该问题最优余值为终端的多步前瞻。

**组合**：有限 horizon rollout + 终端 $J_\mu$ 或 $\tilde J$ 估计（第 4 章 §4.5.3）。

---

### §2.4.1 On-Line Rollout for Deterministic Finite-State Problems

在 $x_k$，对每个 $u_k\in U_k(x_k)$，令 $x_{k+1}=f_k(x_k,u_k)$，基**启发式**（确定性下称 base heuristic）生成 $\{x_{k+1},\ldots,x_N\}$ 及控制，得 tail 代价。选最小者：

$$
\tilde\mu_k(x_k) \in \arg\min_{u_k} \tilde Q_k(x_k,u_k), \quad
\tilde Q_k(x_k,u_k) = g_k(x_k,u_k) + H_{k+1}(f_k(x_k,u_k)), \tag{2.24–2.25}
$$

$H_{k+1}(x_{k+1})$ 为从 $x_{k+1}$ 起的启发式总代价（不含首项 $g_k$）。

**例 2.4.1（TSP + 最近邻）**：状态为部分回路；对每个候选下一城市，用最近邻补全回路得 $H_{k+1}$，选最优扩展（Fig. 2.4.3）。

**序贯一致（sequentially consistent）**：从 $x_k$ 生成的轨迹若前进到 $x_{k+1}$，启发式从 $x_{k+1}$ 起仍沿原轨迹延伸——等价于启发式来自某 DP 策略 $\bar\mu$。

**命题（代价改进）**：若基启发式序贯一致，则 $J_{k,\tilde\pi}(x_k) \le \hat J_k(x_k)$（(2.27)），归纳证明要点与 (2.28) 相同：rollout 一步 min + 归纳假设 + 基策略的 DP 方程（需序贯一致）。

**序贯改进（sequentially improving）**（弱于序贯一致）：

$$
\min_{u_k} \big[ g_k(x_k,u_k) + H_{k+1}(f_k(x_k,u_k)) \big] \le H_k(x_k). \tag{2.29}
$$

即“最优启发式 Q 因子 $\le$ 启发式总代价”。仍得 $J_{k,\tilde\pi}\le H_k$。

#### 归纳证明概要（(2.27)）

设 $J_{k,\tilde\pi}$ 为 rollout 策略从 $(k,x_k)$ 起的代价，$\hat J_k=H_k$ 为启发式总代价。基步 $k=N$：$J_{N,\tilde\pi}=g_N=H_N$。归纳步：对任意 $x_k$，记 $\tilde u_k=\tilde\mu_k(x_k)$，

$$
J_{k,\tilde\pi}(x_k)=g_k(x_k,\tilde u_k)+J_{k+1,\tilde\pi}(f_k(x_k,\tilde u_k))
\le g_k(x_k,\tilde u_k)+H_{k+1}(f_k(x_k,\tilde u_k))
=\min_u[g_k+H_{k+1}(f_k)] \le g_k(x_k,u_k^{\text{heur}})+H_{k+1}(f_k(x_k,u_k^{\text{heur}}))=H_k(x_k),
$$

其中 $u_k^{\text{heur}}$ 为启发式在 $x_k$ 的首控；最后一步等式需**序贯一致**（启发式尾段 = 某 DP 策略尾段）。

**加固 rollout**：比较全路径代价 $C(P^k\cup T^k)$ 与 $C(P^k\cup \tilde T^k)$，避免在 $H_k(x_k)>\min_u[g_k+H_{k+1}]$ 的态误用启发式尾段（Fig. 2.4.4）。

**截断 rollout**：基策略运行 $H$ 步 + 终端 $\tilde J$；计算量 $\times$ 分支因子，但可并行。

**选择性深度**：对候选 $x_{k+1}$ 用启发式粗评，仅对 top 分支展开 $x_{k+2}$（Fig. 2.4.6）；与 MCTS 思想一致。

**多启发式**：对每个 $u_k$ 可试多个基启发式，取最小 $\tilde Q_k$。

**计算代价**：约为基启发式单次代价 $\times$ $|U_k(x_k)|$（一步）或 $\times$ 树规模（多步/选择性深度）。

---

### §2.4.2 Stochastic Rollout and Monte Carlo Tree Search

设基策略 $\pi=\{\mu_0,\ldots,\mu_{N-1}\}$（序贯一致）。**随机情形下**仍有 $J_{k,\tilde\pi}(x_k)\le J_{k,\pi}(x_k)$，证明结构与确定性相同，期望下归纳（(2.27) 随机版）。

**仿真实现**：对每个 $u_k$，采样多条 $w$ 轨迹，基策略 tail + 可选终端 $\tilde J$，Monte Carlo 平均得 $\tilde Q_k(x_k,u_k)$，再 $\arg\min$（(2.31)）。

**例 2.4.2（Backgammon [TeG96]）**：TD-Gammon 为基策略；rollout 对每步合法走法模拟至终局平均得分。双人对局解释：一方固定启发式，另一方 rollout 改进——**不保证**双方同用 TD-Gammon 训练后对第三方更优，需经验验证。分支因子大，实时对弈仍多用 NN 一步前瞻而非全 rollout。

**方差缩减**：比较 $u$ 与 $\hat u$ 时用**共同随机数**估计 $Q$ 差，降低比较方差（(2.32–2.33) 一带）。

**Monte Carlo Tree Search（MCTS）**

朴素 rollout 对每个 $(x_k,u_k)$ 分配**相同**模拟次数 → 三问题：(1) 计算随 $|U_k|$ 与深度爆炸；(2) 弱分支浪费模拟；(3) 强分支模拟不足。

**MCTS 思路**：维护搜索树；从根扩展；**selection** 按 UCB/UCT 在“利用（低估计代价/高胜率）”与“探索（少访问子节点）”间平衡；**expansion** 加子节点；**simulation**（rollout）至终端或截断；**backpropagation** 更新路径上统计量。模拟预算**自适应**倾斜到 promising 分支。

**UCT 典型形式**（最大化 reward 文献；本书取 cost 时需符号相应调整）：子节点 $c$ 的选择得分含 $\sqrt{\log N/N_c}$ 探索项，$N$ 为父访问次数，$N_c$ 为子访问次数。

**与 AlphaGo/AlphaZero**：深度网络提供 prior/value，MCTS 作多步前瞻；棋类确定性但常用随机化策略与 MCTS 采样（§2.0）。

**双人博弈**：固定一方用基策略，另一方 rollout 改进（Backgammon）；零和理论本书未展开，解释需经验验证（例 2.4.2）。

**例 2.4.3（一步前瞻 MCTS + UCB）**：$m=|U_k(x_k)|$，对控 $i$ 维护样本均值 $\bar Q_{i,n}$ 与采样次数 $s_i$。下一采样控 $i_\ell$ 最小化 exploitation + exploration 指数，常取 $T_{i,n}=\bar Q_{i,n}$，UCB 探索项

$$
R_{i,n}=-c\sqrt{\frac{\log n}{s_i}}.
$$

（最小化 cost 时符号与 reward 文献相反；Fig. 2.4.8。）

**Advantage**：$A_k(x_k,u_k)=Q_k(x_k,u_k)-\min_u Q_k(x_k,u)$，与 $u$ 无关常数不干扰控比较；便于参数化（§3.4）。

---

## §2.5 On-Line Rollout for Deterministic Infinite-Spaces Problems

离散控穷举不可行时：设 $H_{k+1}$ 可微且 $g_k,f_k$ 可微，则 (2.34)–(2.35) 的 $\tilde Q_k$ 对 $u_k$ 可微，用梯度法解 (2.34)。

若 $H_{k+1}(x_{k+1})$ 本身是 $(\ell-1)$ 阶段确定性最优控制问题的最优代价，则 rollout 与首步 min **拼接**为 $\ell$ 阶段 NLP（Fig. 2.5.1）——**MPC** 即其工程实例。

---

### §2.5.1 Model Predictive Control

**调节问题**：$x_{k+1}=f_k(x_k,u_k)$，$g_k\ge 0$，约束 $x_k\in X_k$，$u_k\in U_k(x_k)$，且存在 $\bar u_k\in U_k(0)$ 使 $f_k(0,\bar u_k)=0$，$g_k(0,\bar u_k)=0$（(2.36)）。LQ 对非线性、**硬约束**不足；MPC 结合多步前瞻、无限控空间 NLP、确定性等价，并支持再规划。

**算法**（在 $x_k\in X_k$）：

1. 解 $\ell$ 阶段问题 (2.37)：最小化 $\sum_{i=k}^{k+\ell-1} g_i(x_i,u_i)$，满足动力学与约束，**终端** $x_{k+\ell}=0$。  
2. 执行最优序列的首控 $\tilde u_k$，丢弃 $\tilde u_{k+1},\ldots,\tilde u_{k+\ell-1}$。  
3. 观测 $x_{k+1}$ 后重复（Fig. 2.5.3）。

**约束能控（constrained controllability）**：存在 $\ell>1$，使任意 $x_k\in X_k$ 可在 $\ell$ 步内驱动至 0，且中间态/控均可行。该条件保证 (2.37) 可行；$\ell$ 过大则单步优化变难，需实验折中。

**与 rollout 的关系**：MPC 隐含的 $\tilde J$ 为“$(\ell-1)$ 步内将状态驱至 0 并此后保持”的基启发式最优余值。该启发式**序贯改进** (2.38) 但**未必序贯一致**。

**稳定性**：由 (2.38)  telescoping 得 $\sum_{k=0}^K g_k(x_k,u_k) \le H_0(x_0)$，故 $\sum_{k=0}^\infty g_k<\infty$（(2.39)–(2.41)）——在约束能控下闭环**代价有限**；未必到达原点。

**例 2.5.1**：标量 $x_{k+1}=x_k+u_k$，盒约束，$\ell=2$，MPC 得 $\tilde u_k=-\frac{3}{2}x_k$，闭环 $x_{k+1}=x_k$：稳定但 $x_0\neq 0$ 时不到 0；基启发式在 $x_k=1$ 与 $x_{k+1}=1/2$ 处轨迹不一致。

跟踪问题：终端约束改为 nominal 轨迹上的点；代价改为偏离 nominal 的惩罚。

---

### §2.5.2 Target Tubes and Constrained Controllability

约束能控假设**掩盖**不稳定系统在有限控下无法长期留在 $X_k$ 内的情形。

**例 2.5.2**：$x_{k+1}=2x_k+u_k$，$|u_k|\le 1$。若 $|x_0|<1$，可逐步压向 0；若 $|x_0|>1$，任意可行控下 $|x_k|\to\infty$。状态约束 $|x_k|\le\beta$：$\beta<1$ 时可约束能控；$\beta\ge 1$ 则失败（Fig. 2.5.4）。

**目标管道（target tube）**：$\{\bar X^k\}$，$\bar X^k\subset X_k$，且从 $\bar X^k$ 出发存在控进入 $\bar X^{k+1}$（**可达**）。原约束管道不可达时，须先求可达子管道再作 MPC 状态约束；内逼近算法见 [Ber71], [BBM17] 等。

**可达 vs 约束能控**：可达管道是 MPC 可行的必要步骤；边界效应下可达**不蕴含**“有限步到 0”（例 2.5.2 中 $|x_k|\le 1$ 管道可达但边界点无法在有限步内到 0）。

---

### §2.5.3 Variants of Model Predictive Control

- 终端 $x_{k+\ell}=0$ 改为**大惩罚**或**小邻域 + 局部稳定器**；需保持 (2.38) 类序贯改进。  
- **随机系统 + 确定性等价**：典型 $w$ 序列下解 $\ell$ 阶段约束问题（例 2.5.3）；最坏扰动下的管道 MPC 等见文献。  
- **Rollout + 终端余值**；无限时域变体见第 4 章 §4.6.1。  
- 在线代价高时，对 MPC 产生的 $(x_k,\tilde u_k)$ 做策略回归（§2.1.5）。

---

## 本章小结

| 机制 | 核心对象 | 典型保证/风险 |
|------|----------|----------------|
| 一步/多步前瞻 | $\tilde J_{k+\ell}$ | Q 误差斜率小则控排序稳健；$\ell$ 过大 + 差终端可退化 |
| 问题近似 | 简化问题的 $J^*$ | 启发式，需结构支持 |
| Rollout | $H_k$ 来自基策略/启发式 | 序贯改进 $\Rightarrow$ 不劣于基；加固可弱前提 |
| MPC | $\ell$ 步到 0 + 首控执行 | 约束能控 $\Rightarrow$ 有限总代价；未必到平衡点 |

**与后续章节**：参数化 $\tilde J$（Ch.3）、无限时域 VI/PI 与 rollout 界（Ch.4–5）、Actor-Critic 与 MCTS 工程化（Ch.4–5）。

---

*个人学习笔记；原著 Copyright Dimitri P. Bertsekas / Athena Scientific。*