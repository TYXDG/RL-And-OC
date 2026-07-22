# 第 3 章 Parametric Approximation — 分节笔记

> **文献**：Bertsekas, *RL and Optimal Control* Ch.3（2019 draft）。  
> **文本**：`source/ch03_clean.txt`、`source/parts/ch03_part*.txt`。  
> **位置**：第 2 章给出 $\tilde J$ 的构造方式；本章固定**架构** $\tilde J_k(x_k,r_k)$ / $\tilde Q_k(x_k,u_k,r_k)$ 与**训练算法**（LS、增量梯度、FVI、拟合 Q），并讨论与 Bellman 目标的偏差。

---

## 章首

参数化方法将“选前瞻函数”化为：

1. **架构设计**——特征、线性/非线性、神经网络；  
2. **监督目标**——$(x^s,\beta^s)$ 或 FVI 中的 Bellman 目标 $\beta_k^s$；  
3. **优化**——(3.2) 最小二乘及 (3.4)–(3.16) 增量/Newton 法。

除半穷举搜索 $r$ 外，本书强调**数值优化 + 采样**。

---

## §3.1 Approximation Architectures

### 总述

**架构** $\tilde J_k(x_k,r_k)$：**训练** = 选 $r_k$ 使前瞻或 FVI 有效。架构类记为 approximation architecture；$r_k$ 称 weights / 参数向量。

---

### §3.1.1 Linear and Nonlinear Feature-Based Architectures

#### 主要内容

**一般形式**：

$$
\tilde J_k(x_k,r_k)=\hat J_k(\phi_k(x_k),r_k),
$$

$\phi_k(x_k)=(\phi_{1,k},\ldots,\phi_{m,k})'$ 为**特征**；$\hat J_k$ 为读出头（常为线性）。

**线性架构 (3.1)**：

$$
\tilde J_k(x_k,r_k)=r_k'\phi_k(x_k)=\sum_{\ell=1}^m r_{\ell,k}\,\phi_{\ell,k}(x_k).
$$

几何上：$\tilde J_k$ 属于 $\{\phi_{\ell,k}\}$ 张成的子空间；$\phi_{\ell,k}$ 亦称 **basis functions**。

| 例 | 特征类型 | 调参局部性 |
|----|----------|------------|
| **3.1.1** 分段常数 | $\phi_\ell(x)=\mathbf{1}_{x\in S_\ell}$ | 改 $r_\ell$ 仅影响 $S_\ell$ |
| **3.1.2** 多项式 | $1,x_i,x_ix_j,\ldots$ | 全局 |
| **3.1.3** Tetris | 22 维手工特征 | 应用驱动 |
| **3.1.4** 国际象棋 | Shannon 型评估 + 线性权 | 经典 |
| **3.1.5** 部分信息 | $\phi(S_k(I_k))$ 非 $\phi(I_k)$ | 维数控制 |

**非平稳 / 有限时域**：$\phi_k,r_k$ 可随 $k$ 变（阶段、时变状态空间）。**平稳长/无限时域**：常共用同一 $(\phi,r)$（Ch.4–5）。

**NN 角色**：学习 $\phi(x,v)$ 与 $r$  simultaneously（§3.2），替代手工特征。

#### 要点

- 最优 $J_k^*$ 可能高度非线性；特征应编码主要非线性，使 $\hat J_k$ 保持简单（常线性）。  
- AlphaZero 等：**发现**特征而非 Tetris 式手工。

#### 注意点

- 特征不足 → 系统性偏差，非增量法 alone 可修复。

---

### §3.1.2 Training of Linear and Nonlinear Architectures

#### 主要内容

训练集 $(x^s,\beta^s)$，$s=1,\ldots,q$；$\beta^s\approx J(x^s)$（带噪）：

$$
\min_r \sum_{s=1}^q\big(\tilde J(x^s,r)-\beta^s\big)^2. \tag{3.2}
$$

**线性闭式 (3.3)**：

$$
r=\Big(\sum_s \phi(x^s)\phi(x^s)'\Big)^{-1}\sum_s \phi(x^s)\beta^s.
$$

（设计矩阵满秩时。）

**非线性**：(3.2) 非凸 → §3.1.3 或通用 NLP。

**$\beta^s$ 来源**：蒙特卡洛 rollout、专家标注、Bellman 目标（FVI）、$J_{\mu}$ 样本等。

#### 要点

- 线性：一次 LS 得全局最优（该凸问题内）。  
- 正则化：$+\lambda\|r-r_0\|^2$ 改善病态与过拟合。

#### 注意点

- $\beta^s$ 偏差直接传递为策略偏差；噪声大时需更多样本或更平滑架构。

---

### §3.1.3 Incremental Gradient and Newton Methods

#### 问题 (3.4)

$$
\min_y F(y)=\sum_{i=1}^m f_i(y),
$$

$y$ 为参数（$r$ 或 NN 权重）；$f_i$ 可微。可独立于 DP 阅读（非线性规划标准内容）。

#### 全梯度 (3.5)

$$
y^{k+1}=y^k-\gamma^k\sum_{i=1}^m\nabla f_i(y^k).
$$

#### 增量梯度 (3.6)

$$
y^{k+1}=y^k-\gamma^k\nabla f_{i_k}(y^k),
$$

$i_k\in\{1,\ldots,m\}$ 按 **cyclic / uniform random / cyclic+shuffle** 选取。

| 规则 | 说明 |
|------|------|
| Cyclic | 每 cycle 遍历 $1,\ldots,m$ 各一次 |
| Uniform random | 独立均匀抽 $i_k$ |
| Shuffle | 每 cycle 内随机排列——实践常用 |

#### 远/近最优行为

- **远离最优**：单分量梯度常指向“大致正确”方向；极端地 $f_i\equiv f$ 时，增量步长 $\gamma^k m$ 等价全梯度一步——**可快 ~$m$ 倍**。  
- **接近最优**：全梯度可**常数步长**收敛；增量法一般需 $\gamma^k\to 0$，极限收敛率可慢。

#### 例 3.1.6

$f(y)=\sum_i (c_i y-b_i)^2$。组件最小值 $y_i^*=b_i/c_i$；最小值 $y^*=\sum c_i b_i/\sum c_i^2$。  
**Far-out region**（$y$ 在 $[\min y_i^*,\max y_i^*]$ 外）：各 $\nabla f_i$ 与 $\nabla f$ 同号，小步长下单步增量与全梯度进步相当。  
**Region of confusion**（$y$ 在组件最小值区间内）：增量步可能不下降 → **振荡**，需衰减 $\gamma^k$。  
全梯度常数步长上界：$\gamma\le 1/\sum c_i^2$。

#### 聚合梯度 (3.10)

用最近 $m$ 个分量梯度之和——折中速度与稳定性。

#### 增量 Newton (3.12–(3.16)

分块二次近似；少迭代，高维 Hessian 存储/求逆贵。

#### SGD 关系

无限总体 vs 有限和 $m$；有限样本不必假 i.i.d.——**shuffle cyclic** 常优于 uniform 分量抽样 ([GOP15c])。

#### 与 NN

(3.18) 几乎必用增量法；minibatch = 小批量聚合梯度。

---

## §3.2 Neural Networks

### 双参数 (3.17)

$$
\tilde J(x,v,r)=r'\phi(x,v),
$$

$v=(A,b)$ 等定非线性映射；Fig 3.2.1。

**单层**：编码 → 线性 → $\sigma$（ReLU/sigmoid）→ $r'\phi$。

---

### §3.2.1 Training of Neural Networks

**目标 (3.18)**：非凸 LS + **正则化**（权重衰减、dropout 等防过拟合）。

**万能逼近**：单隐层足够宽时在紧集上一致逼近连续函数（标准定理）。

**实践**：增量梯度/Adam + 小 batch；早停与验证集。

---

### §3.2.2 Multilayer and Deep Neural Networks

**深度**：层次特征；**CNN**：局部连接+权共享（棋盘、图像）。

**ReLU 脉冲 (3.19–3.20)**：输入空间分区，区内心形分段线性。

**反向传播 (3.21)**：前向存中间量 + 反向链式法则。

**AlphaGo/AlphaZero**：策略网+值网+MCTS（Ch.2 §2.4.2）。

**GMDH**：多项式多层——同族，RL 应用少。

---

## §3.3 Sequential Dynamic Programming Approximation（FVI）

### 算法（Fig 3.3.1）

与精确 DP 相同：**$k=N-1$ 到 $0$** 反向。已知 $r_{k+1}$ 时：

$$
r_k\in\arg\min_r\sum_{s=1}^q\Big(\tilde J_k(x_k^s,r)-\beta_k^s\Big)^2, \tag{3.22}
$$

$$
\beta_k^s=\min_{u\in U_k(x_k^s)}\mathbb{E}\Big\{g_k(x_k^s,u,w_k)+\tilde J_{k+1}(f_k(x_k^s,u,w_k),r_{k+1})\Big\}.
$$

**$k=N-1$**：右端含 $g_N(f_{N-1}(\cdot))$ 而非 $\tilde J_N$。

**线性 $\tilde J_k=r_k'\phi_k$**：(3.22) → 闭式 $r_k$（同 3.3 正规方程）。

**非线性**：对 (3.22) 用 (3.6) 类增量法。

### 样本 $x_k^s$

应来自**近最优策略**下的访问分布——频率 $\propto$ 最优策略占用测度（Ch.4 概率化表述）。

### 长时域

反向多阶段 FVI 可能病态；**平稳**问题转无限时域 FVI（Ch.5），共用 $(\phi,r)$。

---

## §3.4 Q-Factor Parametric Approximation

### 定义 (3.23)–(3.24)

$$
Q_k^*(x_k,u_k)=\mathbb{E}\big\{g_k(x_k,u_k,w_k)+J_{k+1}^*(f_k(\cdot))\big\},
$$

$$
J_k^*(x_k)=\min_{u_k} Q_k^*(x_k,u_k).
$$

### Q-Bellman (3.25)

$$
Q_k^*(x_k,u_k)=\mathbb{E}\Big\{g_k+\min_{u'}Q_{k+1}^*(f_k(\cdot),u')\Big\}.
$$

可在 **Q 空间**直接 FVI，无需显式 $J_k$ 中间层。

### 架构 (3.26)–(3.27)

- $\tilde Q_k(x_k,u_k,r_k)=r_k(u_k)'\phi_k(x_k)$——$|U_k|$ 小；  
- $\tilde Q_k=r_k'\phi_k(x_k,u_k)$——一般。

### 拟合 Q 迭代 (3.28)–(3.29)

样本 $(x_k^s,u_k^s)$，目标 $\beta_k^s$ 来自 (3.25) 右端（MC 估计）；线性 $\tilde Q$ 闭式 LS。

### 在线控 (3.30)

$$
\tilde\mu_k(x_k)\in\arg\min_u \tilde Q_k(x_k,u,r_k),
$$

**执行控时不求期望**——期望已进训练。

### 模拟器

仅需采样 $(g,f)$ 构造 $\beta_k^s$；与 §2.1.4 一致。

### 策略再近似

在 $\tilde\mu_k$ 上回归 $\mu(x,r)$（§2.1.5）。

### Advantage updating [Bai93/94]

拟合 $A_k=Q_k^*-\min_u Q_k^*$，消除与 $u$ 无关大常数对 LS 的干扰；与 §2.1.6 Q 误差斜率一致。

---

## §3.5 Notes and Sources

- 架构 vs 训练两条研究线；[BeT96]、[BCN18]、[Ber16a] 增量法理论。  
- FVI 长时域/加权病理 → Ch.4 例 4.4.1、Ch.5 例 5.2.1。  
- Advantage → [Bai93/94]、[BeT96] §6.6.2。

---

## 与后续章节接口

| 本章 | 后续 |
|------|------|
| (3.22) FVI | Ch.5 §5.2 + 界 (5.19) |
| (3.6) 增量 | Actor–Critic critic |
| (3.30) $\tilde Q$ | Q-learning、SARSA、DQN |

---

*个人学习笔记；原著 Copyright Bertsekas / Athena Scientific。*