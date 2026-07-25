# Zero-Order Optimization Techniques for Robotics — 分节读书笔记

> **文献**：Armand Jordana, Jianghan Zhang, Joseph Amigo, Ludovic Righetti, *An Introduction to Zero-Order Optimization Techniques for Robotics*（arXiv:2506.22087v2, Oct 2025）。  
> **文本来源**：[`../source/full.txt`](../source/full.txt)（PDF 见 [`..`](../)）。  
> **定位**：以**随机搜索**统一理解机器人里常用的零阶（无梯度）优化——轨迹优化（TO）与策略优化（RL）共用同一套 $\min_x f(x)$ 骨架。

---

## 全文结构与阅读地图

| 节 | 主题 | 核心算法 / 概念 |
|----|------|-----------------|
| **§I** | 动机与统一问题 (1) | TO vs RL、在线/离线、MPPI / CMA / Reinforce 谱系 |
| **§II** | 随机搜索基础 | 纯随机 / 贪心局部搜索；有限差分、RCD、SPSA；**Gaussian smoothing (RS)**；模拟退火 |
| **§III** | 轨迹优化 | Predictive Sampling → **MPPI (LSE)** → **CMA / MPPI-CMA**；块对角协方差 |
| **§IV** | 策略优化 | **DPG**；RS/LSE Actor-Critic；与 **Reinforce** 公式对照 |
| **§V** | 种群算法 | Random Restart、$(N+\lambda)$-ES、SVGD 方向 |
| **§VI** | 并行计算 | JAX/PyTorch、MJX/Isaac、Hydrax/Evosax/CleanRL |
| **§VII** | 结论 | 约束零阶、全局解等开放问题 |

**统一问题**（贯穿全文）：

```math
\min_{x \in \mathbb{R}^n} f(x).
```

- **TO**：$x = (u_0,\ldots,u_{T-1})$，$n = T n_u$（single shooting，动力学约束隐式消去状态）。  
- **RL**：$x = \theta$（策略参数）；目标 $F(\theta) = \mathbb{E}_s[J(\theta,s)]$，即标准折扣 MDP 上的期望回报。

**阅读顺序建议**：§II 是理论枢纽（尤其 RS 与 LSE）；§III、§IV 分别是 TO、RL 的「同一套随机搜索」实例化；§V 解决单点搜索的多样性不足；§VI 解释机器人里零阶方法为何今天才「算得动」。

---

## §I Introduction（引言）

### 主要内容

#### 为何零阶方法在机器人里流行

| 动机 | 说明 |
|------|------|
| **非光滑目标** | 接触、摩擦、碰撞使 $f$ 不可微或梯度无意义；一阶方法需额外平滑/近似 |
| **实现成本** | 对复杂仿真器手写高效 $\nabla f$（含 contact implicit）代价高 |
| **局部极小** | 确定性梯度下降/牛顿法易陷局部极小；零阶方法多为**随机**的，有逃逸可能 |
| **并行硬件** | GPU 上每迭代 **$K$ 次并行 rollout** 使高维采样 TO/MPC 可行 [2,3] |

#### 历史与本文谱系

- **ES**：70 年代演化策略 [7] → **CMA-ES** [8]。  
- **TO**：Predictive Sampling [9]；**MPPI** [10]（最初从信息论/路径积分推导）。  
- **RL**：**Reinforce** [11] 等策略梯度。  
- **本文统一工具**：[12] 的 **Gaussian smoothing (RS)**；[15] 的 **log-sum-exp (LSE) 平滑**——把 MPPI、CMA、策略梯度都看成 Algorithm 3 的 $g$ 估计变体，并导出 **RS/LSE Actor-Critic**。

#### 与一阶方法的对比（概念层）

| | 零阶（本文） | 一阶梯度 |
|---|-------------|----------|
| 每步信息 | $f(x)$ 值（可批量） | $\nabla f(x)$ |
| 非光滑 | 自然适用 | 需平滑/次梯度 |
| 维数 $n$ 大 | 靠随机方向/SPSA/RS，$O(1)$ 次评估/方向 | 通常需 $O(n)$ 或自动微分 |
| 机器人瓶颈 | **仿真 rollout 次数**，不是矩阵求导 |

### TO vs RL（问题空间与部署）

| 维度 | 轨迹优化 TO | 策略优化 RL |
|------|-------------|-------------|
| 优化变量 | 有限维控制序列 $u_{0:T-1}$ | 策略 $\pi_\theta$ → 有限参数 $\theta$（如 NN 权重） |
| 目标 | 式 (17) 有限时域代价和 | 式 (31)–(32) 无限时域折扣回报 |
| 时域 | 有限 $T$，MPC 下滚动 | 无限 + 折扣 $\gamma$ |
| 部署 | **在线** MPC 循环，或离线生成轨迹数据 | 多为**离线**训 $\pi_\theta$，在线只前向 $a=\pi_\theta(s)$ |
| 共同性 | 都可写成 (1)；算法多为**局部随机搜索 + 批量 $f$ 评估** |

**在线 vs 离线（文中强调）**：TO 可在运行时每步解 (17)；RL 近期 SOTA [4,5] 多在仿真里离线搜 $\theta$，运行时计算量小。二者算法形态仍相似（采样 → 加权/选择 → 更新）。

### 与已有工作的关系

- **TO 侧**：[13] MPPI ↔ CMA-ES；[14] MPPI ≈ 近似梯度；[3] MPPI ↔ 扩散退火；本文补 **LSE 平滑**对 MPPI 权重的优化解释及与 risk-seeking 控制的关系。  
- **RL 侧**：[16,17] 在**参数空间**做 ES/随机搜索；[18,19] 把策略梯度与 Nesterov RS 联系（多用于 TO）；本文反向：用 RS **解释并改 actor 更新**（动作空间采样 + DPG 结构）。

### 要点

- §II 的 **Algorithm 3**（近似梯度下降）是 MPPI 加权平均、CMA 协方差更新、RS Actor-Critic 的母版。  
- **「局部搜索算法」**指采样分布依赖当前 $x$（Alg.2），≠「只收敛到局部最优解」；Alg.1/2 在条件下都可有全局收敛理论 [22]，但**维数灾难**仍在。

### 注意点

- 默认**确定性动力学**（足式/操作常见）；随机动力学扩展留作 future work。  
- RL 先讲**确定性策略 + DPG**，再通过 Reinforce 解释随机策略 = 动作空间平滑 + 额外探索噪声。

---

## §II Random Search（随机搜索）

§II 建立后文 TO/RL 算法的共同语言：样本在 TO 里是**控制序列/轨迹**，在 RL 里是**策略参数**或（经 Q）**动作扰动**。

---

### §II.A Simple random search（简单随机搜索）

虽很少单独用于高维机器人问题，但收敛性质澄清了后文「局部 vs 全局算法」术语。

#### §II.A.1 Pure (Global) Random Search — Algorithm 1

**流程**：初始化 $x \leftarrow x_0$；循环：采样 $\tilde x$（分布**不依赖**历史），若 $f(\tilde x)<f(x)$ 则 $x\leftarrow\tilde x$。

| 属性 | 内容 |
|------|------|
| 「Global」含义 | 采样分布忽略过去迭代，非指一定找到全局最优 |
| 理论 | 温和条件下**全局收敛** [22] |
| 实践 | **维数灾难**——所需 $f$ 评估次数随 $n$ 指数增长，几乎不用 |

#### §II.A.2 Greedy Local Search — Algorithm 2

**流程**：$d \sim \mathcal{N}(0,\Sigma)$；仅当 $f(x+d)<f(x)$ 时 $x\leftarrow x+d$（严格下降才接受）。

| 属性 | 内容 |
|------|------|
| 「Local」含义 | 扰动分布中心随当前 $x$ 移动 |
| vs 局部最优解 | 不同概念：Alg.2 在更强条件下也可证**全局收敛** [22]，但条件比 Alg.1 严 |
| 后继 | **Predictive Sampling**（一次采 $K$ 个 $d$）、**$(1+1)$-ES** 的直接前驱 |

**与 Predictive Sampling 的关系**：Alg.2 每步 1 个候选；Predictive Sampling 每步 $K$ 个候选取最优——仍是「只保留最好」，但并行评估 $K$ 路。

---

### §II.B Random search via gradient approximation（梯度近似）

#### 母算法 Algorithm 3

```math
x \leftarrow x - \alpha g, \quad g \approx \nabla f(x)\ \text{且仅用}\ f\ \text{的取值构造}.
```

$\alpha$ 为步长/学习率。后文 MPPI、CMA 可视为在特殊 $g$ 或分布参数更新下脱离线性下降形式，但 RS/LSE 推导仍从 (3) 出发。

#### 1) 完整有限差分 — 式 (2)

```math
g = \sum_{j=1}^n \frac{f(x+\mu e_j) - f(x)}{\mu} e_j.
```

- $\mu>0$，$e_j$ 为标准基；也可改用**中心差分**。  
- **代价**：每步 **$n+1$**（或 $2n+1$）次 $f$ 评估 → 长视界 TO（$n=T n_u$ 很大）不适用。

#### 2) Random Coordinate Descent (RCD) — 式 (3)

```math
g = \frac{f(x+\mu e_j) - f(x)}{\mu} e_j, \quad j \sim \mathrm{Uniform}\{1,\ldots,n\}.
```

- 每步 essentially **1 次新 $f$ 评估**（$f(x)$ 可缓存）。  
- **凸情形** [24]：期望迭代次数 ≤ 完整梯度下降的 $n$ 倍；某些函数上与 GD **迭代数同级**但评估少 $n$ 倍。  
- 推广：不限坐标轴，可沿任意随机方向差分（连向 SPSA/RS）。

#### 3) SPSA — 式 (4)

```math
g = \frac{f(x+\mu\Delta) - f(x-\mu\Delta)}{2\mu} \Delta, \quad \Delta_i \in \{+1,-1\}\ \text{i.i.d.}
```

- **2 次评估**得到 $n$ 维随机梯度估计；Spall [22] 用中心差分形式。  
- 性能在合理假设下可接近完整有限差分。  
- **全局收敛** [29]：即使**不加**额外高斯噪声，SPSA 估计噪声 $\xi_k$ 已足以充当 Langevin 扰动，使离散 Scheme (16) 渐近收敛到全局极小。

#### 4) Gaussian Smoothing（RS）— 核心

**平滑目标**（式 5）：

```math
f_\mu(x) = \mathbb{E}_{\epsilon \sim \mathcal{N}(0,\Sigma)}[f(x + \mu\epsilon)].
```

- $\mu\to 0$ 时 $f_\mu\to f$；小 $\mu$ 时极小点略移，但 $f$ 变**光滑** [12]——这是接触/非光滑 $f$ 上仍能「谈梯度」的关键。  
换元 $z=x+\mu\epsilon$ 可把 $f_\mu$ 写成对 $z$ 的 Gaussian 加权积分（式 6），从而对 $x$ 求导**不碰** $\nabla f$：

```math
\nabla f_\mu(x) = \mathbb{E}\left[\frac{f(x+\mu\epsilon)}{\mu}\Sigma^{-1}\epsilon\right] = \mathbb{E}\left[\frac{f(x+\mu\epsilon)-f(x)}{\mu}\Sigma^{-1}\epsilon\right].
```

| 要点 | 说明 |
|------|------|
| Baseline $f(x)$ | 利用 $\mathbb{E}[f(x)\Sigma^{-1}\epsilon]=0$，估计对 $f$ **平移不变**，降方差 |
| Log-likelihood trick（式 9） | $\nabla f_\mu = \mathbb{E}[f(z)\nabla_m \log p_{m=x}(z)]$，与 Reinforce 同型 |
| 实用 $g$（式 11–12） | 前向：$(f(x+\mu\epsilon)-f(x))\Sigma^{-1}\epsilon/\mu$；中心差分用 $(f(x+\mu\epsilon)-f(x-\mu\epsilon))$ |
| Remark 1 | 也可从**单位球面**采方向 [25]，保证 $\|\epsilon\|$ 有界 |
| 凸情形 [12] | RS 前向/中心估计的迭代复杂度 ≤ 标准梯度法的 $n$ 倍 |

**式 (11) vs 未中心化估计**：$\frac{1}{\mu}f(x+\mu\epsilon)\Sigma^{-1}\epsilon$ 合法但方差可任意大；减 $f(x)$ 后更稳。

#### §II.B 小结

| 方法 | 每步 $f$ 评估（量级） | 机器人相关性 |
|------|----------------------|--------------|
| 有限差分 | $O(n)$ | 维数太高 |
| RCD | $O(1)$ | 稀疏方向探索 |
| SPSA | $O(1)$ | 2 次评估，有全局收敛理论 |
| RS | $O(1)$ 方向 + 可批量 | **MPPI / CMA / RL actor 的理论枢纽** |

[12,22,24] 假设不同，但思想一致：**沿随机方向采样，用 $O(1)$ 次评估逼近梯度下降的行为**。

---

### §II.C Simulated Annealing（模拟退火）

#### 动机与 Langevin 框架

- RS/SPSA 在**凸**情形样本效率高；**非凸**时纯梯度下降陷局部极小。  
- **Langevin SDE**（式 13）：$dX_t = \nabla_x \log p(X_t)\,dt + dW_t$，平稳分布为 $p$ [26]。  
取 Boltzmann 型（式 14–15）：

```math
p(x) \propto \exp(-f(x)/\lambda), \quad \nabla_x \log p(x) = -\lambda^{-1}\nabla f(x).
```

$\lambda\to 0$ 时 $p$ 集中在 $f$ 的**全局极小**集上 [27]。

#### 离散算法（式 16）

```math
x_{k+1} = x_k - \alpha_k g_k + \gamma_k \epsilon_k, \quad g_k = \nabla f(x_k) + \xi_k.
```

- $\epsilon_k$：显式高斯噪声；$\xi_k$：梯度估计噪声（如 SPSA）。  
- 适当调度 $\alpha_k,\gamma_k$ 使有效温度 $\lambda$ **缓慢**→ 0 → **模拟退火** [28]。  
- **SPSA + (16)** 可证全局收敛 [29]；RS 的类似结果文中**猜想**成立。

#### 实践与 Remark 2

- 理论强但**仅渐近**；有限步温度调度难设计。  
- **Remark 2**：Alg.2 的「仅接受下降」可与 **Metropolis–Hastings** 建立联系 [30]（附录 A）——拒绝 worsening 样本 ≈ MH 接受率。  
- **与 §IV Remark 8**：RL 目标 $F(\theta)=\mathbb{E}_s[J(\theta,s)]$ 本身需 MC → 梯度估计自带噪声 → 可解释 RL 少陷坏局部极小。

---

## §III Trajectory Optimization（轨迹优化）

### 问题形式 — 式 (17)

```math
\min_{u_0,\ldots,u_{T-1}} \sum_{t=0}^{T-1} c_t(x_t, u_t) + c_T(x_T), \quad x_{t+1} = f_{\mathrm{dyn}}(x_t, u_t),\ x_0 = x.
```

| 要素 | 说明 |
|------|------|
| **Single shooting** | 只优化 $u$；状态由动力学递推，约束隐式消去 |
| 记号 | 文中 $x$ 兼指状态；优化变量实为 $u_{0:T-1}$，再打包成 (1) 的 $x$ |
| 控制约束 | 常**罚函数**进 $c_t$（实验里 Hydrax 如此） |
| 维数 | $n = T n_u$ |
| 黑盒 | $f(x)$ = 沿控制序列 rollout 的总代价，不要求 $\nabla_u$ |

**MPC 用法**：每控制周期解 (17)，只执行 $u_0$，再滚动——零阶 MPC 即每步用 MPPI/CMA 等更新整条 $u$ 序列 [2,3]。

---

### §III.A Predictive Sampling — Algorithm 4

**相对 Alg.2 的改动**：每迭代采 **$K$ 个**独立高斯方向 $d_k$，构造集合 $D=\{x\}\cup\{x+d_k\}$，令 $x\leftarrow\arg\min_{\tilde x\in D} f(\tilde x)$；若无改进则保持 $x$。

| 要点 | 说明 |
|------|------|
| 并行 | $K$ 次 rollout **完全独立**，适合 GPU 批量仿真 [9] |
| vs MPPI | 离散 **min**（赢家通吃），非加权平均 |
| ES 特例 | **$(1+\lambda)$-ES**（§V）与此等价 |
| 局限 | 简单但在 Hydrax 实验中仍具竞争力（Fig.2） |

---

### §III.B Log-sum-exp transform & MPPI — Algorithm 5

#### 原始 MPPI 更新（式 18–19）

采样 $x_k \sim \mathcal{N}(x,\Sigma)$，指数权重：

```math
w_k = \frac{\exp(-\frac{1}{\lambda}(f(x_k)-\rho))}{\sum_{j=1}^K \exp(-\frac{1}{\lambda}(f(x_j)-\rho))}, \quad \rho = \min_j f(x_j),
```

```math
x \leftarrow \sum_{k=1}^K w_k x_k.
```

- $\rho$ 仅防数值溢出；权重对 $f$ 的**加法常数**不变。  
- 权重称为 **Exponential Average weights**；更新是**软选择**，而非 Alg.2 的 hard accept/reject。

#### LSE 平滑目标（式 20–21）

```math
f_{\mu,\lambda}(x) = -\lambda \log \mathbb{E}\left[\exp\left(-\frac{1}{\lambda} f(x+\mu\epsilon)\right)\right].
```

- $\mu\to 0$：恢复原 $f$；$\lambda\to\infty$：$f_{\mu,\lambda}\to f_\mu$（RS）[15]。  
- LSE 可看作 RS 与 **Moreau envelope** 之间的插值：Moreau 近似更好但更贵；LSE 用样本换近似质量。

**梯度与 MPPI 一致**（式 21–23）：$\nabla f_{\mu,\lambda}$ 的 Monte Carlo 估计含**分母上的期望** → 需要较多样本；并行算力充足时这不是问题，但大 $K$ 使更新近乎确定性 → **易陷局部极小**（与 §II.C 退火形成张力）。

#### Natural gradient 视角（式 24）

- 设 $\mu=1,\ \alpha=1/\lambda$，MPPI 更新 = 在 Gaussian 族上的 **natural gradient** 步 [31]。  
- Fisher 信息 $F = \Sigma^{-1}$（Gaussian 情形）；natural gradient 使更新对参数化不变（附录 A）。  
- $\Sigma=\sigma^2 I$ 时 $\sigma/\lambda \le 1/L$（$L$ 为平滑函数 Lipschitz 常数）→ 步长像凸优化里的保守最优步长 [15,32]。

#### Fig.1 直觉（RS vs LSE）

一维双阱例子（一阱窄且全局、一阱宽且局部）：  
- **RS**：两阱被「拉平」，深度/宽度接近。  
- **LSE**：**全局窄阱更宽、更 dominant**，局部宽阱被压制 → 解释 MPPI 在非凸 TO 上常优于纯 RS。

#### Remark 3–4

- **Remark 3**：LSE 与 **risk-seeking** 控制 [33] 相关（附录 A）。  
- **Remark 4**：$\lambda\to 0$ 时 surrogate 集中在全局极小附近，与模拟退火 $p\propto e^{-f/\lambda}$ 同族；对 LSE 做 Langevin 更新 ≈ 对 $e^{-f/\lambda}$ 做 RS。

#### 与经典 MPC 的区别

| | 经典 MPC | MPPI |
|---|----------|------|
| 优化对象 | 常逐段 Bellman / 二次规划 | 整条 $u$ 序列作 $x$ |
| 更新 | 梯度/QP 或终端 $\tilde J$ | 指数加权粒子平均 |
| 并行 | 依实现 | 天然 $K$ 路 rollout |
| 共同点 | **Receding horizon**，每步只执行首控 |

---

### §III.C Covariance Matrix Adaptation (CMA) — Algorithm 6

#### 固定 $\Sigma$ 的问题

MPPI/RS 常取固定 $\Sigma$；实际中搜索尺度与病态程度随迭代变，需**自适应**搜索几何。

#### 分布空间视角（式 25–27）

```math
\min_{\theta=(x,\Sigma)} J(\theta), \quad J(\theta) = \mathbb{E}_{z \sim \mathcal{N}(x,\Sigma)}[f(z)].
```

- 优化 RS 的均值 $x$ = 在 Gaussian **信念** $N(x,\Sigma)$ 下最小化期望代价；$\Sigma$ 描述「不确定性椭圆」。  
- $J$ 最小时 $N(x,\Sigma)$ 集中在 $f$ 的极小附近 [34]。

**Log-likelihood trick**（式 28–29），natural gradient 得：

```math
\Delta x = \mathbb{E}[f(z)(z-x)], \quad \Delta \Sigma = \mathbb{E}[f(z)((z-x)(z-x)^\top - \Sigma)].
```

**离散样本更新**（式 30），步长 $\alpha$，$w_k=f(x_k)$ 或 $w_k=f(x_k)-f(x)$：

```math
\Sigma \leftarrow (1-\alpha\textstyle\sum_k w_k)\Sigma + \alpha\textstyle\sum_k w_k (x_k-x)(x_k-x)^\top,
```

```math
x \leftarrow (1-\alpha\textstyle\sum_k w_k)x + \alpha\textstyle\sum_k w_k x_k.
```

**更新顺序**：**先 $\Sigma$ 后 $x$**——协方差更新必须用采样时的旧均值 $x$。

#### Table I：权重变体 → 不同名算法

| 算法 | 权重 $w_k$ | 备注 |
|------|-----------|------|
| **CMA** | $f(x_k)$ 或 $f(x_k)-f(x)$ | 平移不变 |
| **CMA-ES** | 按 $f(x_k)$ **排序**的任意单调权 [8,34] | 无 evolution path 版 |
| **MPPI-CMA** | 指数权 (19) | 权非负且和为 1 |
| **CEM** | 精英权：最好 $K_e$ 个为 $1/K_e$，其余 0 | 更新顺序：均值先于协方差（附录 A） |

#### 正定性与 xNES / MPPI-CMA

- 一般 $w_k=f(x_k)$ **不**保证 $\sum w_k=1$ 或 $w_k\ge 0$ → $\Sigma$ 可能失正定。  
- 若 $0\le\eta<1,\ w_i\ge 0,\ \sum w_i=1$，则从正定初值保持正定 [35]。  
- **xNES** [36]：在指数坐标更新协方差，天然保持正定；小步长下与 natural gradient 一致。  
- **MPPI 指数权**自然非负、归一 → **MPPI-CMA** 更稳。

#### 块对角 $\Sigma$ 与长视界 $T$

- 全矩阵 CMA：$O(n^2)$ 存储/更新，$n=Tn_u$ 时不可行。  
- 设 $\Sigma=\mathrm{blockdiag}(\Sigma_1,\ldots,\Sigma_T)$，每块 $n_u\times n_u$：对**每块**做 natural gradient → 保持块对角，**$O(T)$** 复杂度；Predictive Sampling / MPPI 同理。  
- 直接对全 $\Sigma$ 做 (29) **不一定**保持块对角——应对每块 $\Sigma_t$ 单独更新（文中称 block-diagonal CMA，形式证明留 future work）。  
- **Remark 6**：块对角 MPPI-CMA = **PI²-CMA** [13]（无时间平均）。

---

### §III.D Numerical experiments

| 设置 | 内容 |
|------|------|
| 平台 | Hydrax [37] |
| 任务 | Cartpole, DoubleCartPole, PushT, Humanoid |
| 样本数 | 每迭代 **2048** |
| MPPI / MPPI-CMA | $\lambda=0.1$ |
| CMA | 实验用**全协方差**（未成为瓶颈）；块对角结果见附录 |
| MPPI-CMA 步长 | 均值与协方差**分开** lr，如 (1.0, 0.1) 明显优于共享步长 |

**Fig.2 趋势**（6 seeds 中位数）：  
- **MPPI-CMA** 稳定优于 MPPI。  
- **Predictive Sampling** 简单但强。  
- **RS**（Algorithm 3 + 式 11）**步长敏感**，不同任务需不同 lr。

---

## §IV Policy Optimization（策略优化）

### 为何从 TO 转到 RL

在线 MPC 每步解 TO，算力/时限压力大 [2,3]；RL 把搜索移到**离线**，运行时仅 $\pi_\theta(s)$ 前向，在复杂任务上更易达到 SOTA [4,5]——但优化结构仍可与 TO 侧零阶方法对照。

### 问题 — 式 (31)–(32)

```math
F(\theta) = \mathbb{E}_s[J(\theta,s)], \quad J(\theta,s) = \sum_{k=0}^\infty \gamma^k r(s_k, a_k),
```

约束：$s_0=s$，$a_k=\pi_\theta(s_k)$，$s_{k+1}=f_{\mathrm{dyn}}(s_k,a_k)$。**最大化回报**（RL 符号），与 TO 最小化代价相对。

| 对比 TO | 说明 |
|---------|------|
| 时域 | 无限 + $\gamma$ |
| 目标 | 对**初态分布**期望，非单次 rollout |
| 策略 | 先设**确定性** $\pi_\theta$ → DPG [38] |
| Remark 7 | 对动力学/环境参数再期望 → **domain randomization** |

### 两种采样哲学

1. **参数空间零阶**：扰动 $\theta$ [16,17]（ES-RL）；本文不主打。  
2. **动作空间 + 结构**：DPG 把 $\nabla_\theta F$ 写到 $\partial_a Q$，再用 RS 估 $\partial_a Q$——保留 Actor-Critic 框架，只改 actor 梯度来源。

---

### §IV.A Deterministic Policy Gradient

**Q 定义**（式 34）：从 $(s,a)$ 出发，$a_0=a$，之后 $a_k=\pi_\theta(s_k)$ 的折扣回报和。

**DPG 定理**（式 33, 35）：

```math
\frac{\partial J(\theta,s_0)}{\partial \theta} = \sum_{k=0}^\infty \gamma^k \frac{\partial \pi_\theta(s_k)}{\partial \theta} \frac{\partial Q(s_k,a_k)}{\partial a}\bigg|_{a=\pi_\theta(s_k)},
```

```math
\nabla_\theta F(\theta) = \mathbb{E}_s\left[\sum_{k=0}^\infty \gamma^k \frac{\partial \pi_\theta(s_k)}{\partial \theta} \frac{\partial Q(s_k,a_k)}{\partial a}\right].
```

- 全文轨迹期望形式 [39]（非显式 $\gamma$-discounted state distribution）。  
- **Q 不可解析** → **Actor-Critic**：critic 用 Bellman 自监督拟合 $Q$；actor 用 (35) 更新（Sutton & Barto [11]）。  
- **Remark 8**：$F(\theta)$ 需 MC → 梯度估计**固有噪声** → 联系 §II.C，解释 RL 逃逸局部极小。

---

### §IV.B Gaussian smoothing on Q — Algorithm 7

**动机**：DDPG [41] 对 critic 网络 autograd $\partial_a Q$；TD3 [42] 在此基础上改进。RS 思路：用 $Q$ 值差分估 $\partial_a Q$，**无需** $\nabla_a Q$ 闭式。

**RS 梯度估计**（式 36）：

```math
\nabla_\theta F(\theta) \approx \mathbb{E}\left[\sum_{k=0}^\infty \gamma^k \frac{\partial \pi_\theta(s_k)}{\partial \theta} (Q(s_k, a_k+\epsilon_k) - Q(s_k, a_k)) \Sigma^{-1} \epsilon_k\right].
```

**Algorithm 7（单步示意）**：

1. $a \leftarrow \pi_\theta(s)$  
2. 采 $\epsilon \sim \mathcal{N}(0,\Sigma)$  
3. Actor：$\theta \leftarrow \theta + \alpha (Q(s,a+\epsilon)-Q(s,a)) \frac{\partial \pi_\theta(s)}{\partial \theta} \Sigma^{-1}\epsilon$  
4. Critic：按原 DDPG/TD3 逻辑更新 $Q$（正文省略 rollout 与 actor/critic 交替细节）

| 变体 | 说明 |
|------|------|
| **LSE Actor-Critic** | actor 更新用指数权（类比 MPPI） |
| **Remark 9** | $\frac{\partial \pi_\theta(s)}{\partial \theta}$ 将**动作空间**扰动映射到**参数空间**更新；不同于 ES 直接扰动 $\theta$ |

#### 与 DDPG / TD3

| | DDPG / TD3 | RS-DDPG / LSE-DDPG |
|---|------------|---------------------|
| $\partial_a Q$ | Critic 网络 autograd | $Q(s,a+\epsilon)-Q(s,a)$ 有限差分 |
| 适用 | Q 光滑、可微 | Q 不可微、或不想反传仿真器 |
| Critic | 仍 NN + TD | 不变 |

---

### §IV.C Numerical experiments

| 设置 | 内容 |
|------|------|
| 基座 | CleanRL [43]，**只改 actor 更新** |
| 环境 | MuJoCo ×7 |
| 对比 | DDPG, TD3, RS-DDPG, LSE-DDPG, RS-TD3, LSE-TD3 |
| RS 采样 | 每更新 **10** 个 $\epsilon$；$\Sigma=0.1^2 I$ |
| 重复 | 每任务 5 runs |

**Fig.3**：RS-DDPG、LSE-DDPG **显著优于** DDPG；TD3 已强，改进边际小（未全面调 actor-critic 超参）。

---

### §IV.D Connection to Reinforce

**随机策略目标**（式 37）：$A_k \sim p_\theta(\cdot|S_k)$，优化 $\mathbb{E}[\sum_k \gamma^k r(S_k,A_k)]$。

高斯策略（式 38–39）：$A_k \sim \mathcal{N}(\pi_\theta(S_k),\Sigma)$，

```math
\frac{\partial}{\partial \theta}\log p_\theta(a|s) = \frac{\partial \pi_\theta(s)}{\partial \theta}^\top \Sigma^{-1}(a - \pi_\theta(s)).
```

**策略梯度**（式 40，带 baseline $V$）：

```math
\frac{\partial J}{\partial \theta} = \mathbb{E}\left[\sum_k \gamma^k (Q(S_k,A_k)-V(S_k)) \frac{\partial \pi_\theta(S_k)}{\partial \theta}^\top \Sigma^{-1}(A_k - \pi_\theta(S_k))\right].
```

| vs 式 (36) | Reinforce (40) |
|------------|----------------|
| Rollout | **随机** $S_k,A_k$ | 确定性 $s_k,a_k=\pi_\theta(s_k)$ + 仅 $\epsilon_k$ 扰动动作 |
| Q 的定义 | 随机策略下期望回报 | 确定性 rollout Q |
| Baseline | $V(S_k)$ 减项 | $Q(s,a)$ 减项 ↔ RS 减 $f(x)$ |
| 探索 | 动作分布本身采样 | 额外 $\epsilon_k$ 为 RS 噪声 |

**统一解读**：Reinforce = 在**动作空间平滑**后用 log-likelihood 求 $\nabla_\theta$；相对 RS-AC **多一层**策略随机性（探索噪声）。平滑过多可能使 surrogate 偏离原问题 → PPO/TRPO 联系留 future work。

#### 三式对照表

| 公式 | 策略 | Q 的定义 | 噪声来源 |
|------|------|----------|----------|
| (35) DPG | 确定性 | 确定性 rollout Q | 初态/轨迹 MC |
| (36) RS-AC | 确定性 | 确定性 Q | + 动作 $\epsilon_k$ |
| (40) Reinforce | 随机 | 随机 rollout Q | 动作分布 + 轨迹 |

---

## §V Population Based Algorithms（种群算法）

### 动机

单点 Langevin / 局部搜索在多模态分布上**混合慢、自相关高**；需要**多个起点**或**粒子协调**。

### Algorithm 8：Random Restart

1. 维护当前最优 $x$  
2. 采样全局起点 $\bar x$  
3. 从 $\bar x$ 运行 Greedy Local Search 得 $\tilde x$  
4. 若 $f(\tilde x)<f(x)$ 则更新 $x$  

→ 结合 Alg.1（全局采起点）与 Alg.2（局部改进）。

### Algorithm 9：$(N+\lambda)$-ES

1. 维护种群 $\{x_1,\ldots,x_N\}$  
2. 构造 $D=\{x_1,\ldots,x_N\}$  
3. 重复 $\lambda$ 次：从种群均匀抽 $x$，加高斯扰动 $d$，加入 $D$  
4. 保留 $D$ 中 $f$ 最小的 **$N$** 个作为新种群  

| ES 记号 | 等价算法 |
|---------|----------|
| $(1+1)$-ES | Greedy Local Search (Alg.2) |
| $(1+\lambda)$-ES | Predictive Sampling (Alg.4) |

**局限**：多个局部搜索可能收敛到**同一**极小——粒子缺乏排斥。

### SVGD 与零阶扩展

**Stein Variational Gradient Descent** [44]：粒子间加**排斥项**，避免扎堆；原版需梯度。[45] 给出基于 CMA-ES 的**无梯度 SVGD**——与 §II–III 工具衔接。

### 与 ES-RL

Salimans ES [16] 等在**参数空间**做 $(N+\lambda)$/CMA；本文强调 **动作空间 RS + DPG** 与 TO 侧 CMA 的公式统一，目标不是替代 PPO 式种群 RL。

---

## §VI Parallel Computing（并行计算）

### 为何 ML 框架 alone 不够

JAX/PyTorch [46,47] 擅张量并行，但 TO/RL 的 $f(x)$ 每次评估 = **整条动力学前向仿真**（式 17、31），必须并行**仿真器**而非仅并行 matmul。

### 工具链

| 层次 | 工具 | 作用 |
|------|------|------|
| 仿真 | **MJX** [48], **Isaac Sim** [49] | GPU 批量环境步进 |
| TO | **Hydrax** [37] + **Evosax** [50] | 零阶 MPC、ES 算法库（JAX） |
| RL | **CleanRL** [43] | 简洁 DDPG/TD3 等实现 |

**要点**：零阶 TO/RL 的 wall-clock 通常由 **rollout 吞吐**决定；算法层 $K=2048$ 样本只有硬件跟得上才有意义。

---

## §VII Conclusion（结论）

### 主要结论

- **随机搜索**统一 MPPI、CMA、策略梯度/Reinforce；同一 $\min f(x)$，不同 $x$ 含义（控制序列 vs $\theta$ vs 动作扰动）。  
- **LSE vs RS** 解释 MPPI 在非凸 landscape 上的优势；**CMA** = 在 Gaussian 族上的 natural gradient。  
- **RS/LSE Actor-Critic** 展示统一视角可导出**有竞争力**的 RL 变体，而非仅解释旧算法。

### 开放问题

- **约束零阶优化**（控制界、碰撞约束）的系统处理  
- **可证全局解**（RS 的 Langevin 式结果）  
- 与 **PPO/TRPO** 等信任域方法的平滑/噪声关系  
- 块对角 CMA 的**形式证明**与最优块结构  
- Reinforce 额外随机性是否**总是**有利于探索

---

## 算法速查（与附录）

| # | 名称 | 节 | 一句话 |
|---|------|-----|--------|
| 1 | Pure Random Search | §II.A | 独立采样，更好则替换 |
| 2 | Greedy Local Search | §II.A | 高斯扰动，仅接受下降 |
| 3 | Approximate Gradient Descent | §II.B | $x-\alpha g$，$g$=FD/RCD/SPSA/RS |
| 4 | Predictive Sampling | §III.A | $K$ 候选取 min |
| 5 | MPPI | §III.B | 指数权加权平均 |
| 6 | CMA | §III.C | 更新 $(x,\Sigma)$ |
| 7 | RS Actor-Critic | §IV.B | Q 上 RS 的 DPG |
| 8 | Random Restarts | §V | 多起点局部搜索 |
| 9 | $(N+\lambda)$-ES | §V | 种群 + 精英选择 |
| 10 | CEM | 附录 | 精英权 CMA 变体 |

**附录 A**（正文引用，未单独笔记）：Metropolis–Hastings 与 Alg.2；**natural gradient** / Fisher 信息；**risk-seeking** 与 LSE；CEM 更新顺序（均值先于协方差）。

---

## 个人阅读注意点

1. **三种平滑别混**：RS 平滑 $f$；LSE 平滑 RS；Reinforce 平滑**策略分布**——对象不同，公式同源 log-likelihood trick。  
2. **MPPI 样本数**：大 $K$ 降方差、更新像确定性优化 → 易陷局部极小；需退火、随机重启或 CMA 自适应 $\Sigma$ 补探索。  
3. **Actor 更新**：DPG 给精确 $\nabla_\theta F$；RS/LSE-AC 在 $\partial_a Q$ 不可用时零阶化；**critic 仍 Bellman/TD**。  
4. **TO 维数**：长视界必用块对角 $\Sigma$ 或 PI² 类结构，否则 $Tn_u$ 维全协方差不可承受。  
5. **并行是前提**：无 GPU 批量仿真，文中 $K=2048$ / 2048 样本的算法设计难以复现 wall-clock 优势。

---

## 参考文献（文中高频）

- [12] Nesterov & Spokoiny — Gaussian smoothing 理论  
- [15] Scaman et al. — log-sum-exp 平滑与 MPPI 优化解释  
- [8] Hansen & Ostermeier — CMA-ES  
- [10] Williams et al. — MPPI  
- [9] Howell et al. — Predictive Sampling  
- [38] Silver et al. — DPG  
- 代码与复现实验：[zoo-rob](https://github.com/ajordana/zoo-rob)