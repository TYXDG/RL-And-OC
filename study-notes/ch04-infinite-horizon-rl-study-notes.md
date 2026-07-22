# 第 4 章 Infinite Horizon Reinforcement Learning — 分节笔记

> **文献**：Ch.4（2019 draft，修订 2019-02）。**文本**：`source/ch04_clean.txt`。  
> **位置**：有限状态、**平稳** MDP 的精确理论（SSP、折扣）与 VI/PI/Q-learning/TD/LP/策略梯度；为 Ch.5 参数化与仿真实现提供定理与界。记号：$i,j\in\{1,\ldots,n\}$，终止态 $t$，$p_{ij}(u)$。

---

## 章首

无限时域代价为有限时域代价的极限（Fig 4.1.1）。两类问题：

| 类型 | $\alpha$ | 终止态 | 本质 |
|------|----------|--------|------|
| **SSP** | 1 | 吸收态 $t$，$p_{tt}=1$，$g(t,\cdot,t)=0$ | 几乎必然到 $t$，最小期望总代价 |
| **折扣** | $(0,1)$ | 可无 | 每步 $1-\alpha$ 人工吸收可化为 SSP |

**VI 索引**：$J_0(i)\equiv 0$，$J_{k+1}=TJ_k$（(4.1)），与 Ch.1 **反向** $J_N$ 递推对照。

---

## §4.1 An Overview of Infinite Horizon Problems

### 主要内容

**猜想 (4.2)–(4.3)**（在适当假设下成立）：

1. $J^*(i)=\lim_{N\to\infty} J_N(i)$；  
2. Bellman：$J^*(i)=\min_u\sum_j p_{ij}(u)[g(i,u,j)+J^*(j)]$（SSP 含直达 $t$ 项）；  
3. 逐点 min 得**平稳最优策略** $\mu^*$。

**转移 vs 系统方程**：$x_{k+1}=w_k$，$w_k$ 按 $p_{x_k w_k}(u_k)$ 取值——与 Ch.1 $f_k$ 表述等价。

### 要点

- 从任意态出发的“未来子问题”结构相同 → 最优性可在平稳策略类中讨论。  
- Q 因子、$F$ 算子为 §4.8 采样算法准备。

### 注意点

- $J_\pi(i)$ 极限存在性在非标准问题中需单独验证。

---

## §4.2 Stochastic Shortest Path Problems

### Bellman 方程 (4.4)

对 $i=1,\ldots,n$：

$$
J^*(i)=\min_{u\in U(i)}\Big[p_{it}(u)g(i,u,t)+\sum_{j=1}^n p_{ij}(u)\big(g(i,u,j)+J^*(j)\big)\Big].
$$

**三项解释**：(a) 本步进入 $t$ 的期望代价；(b) 本步进入非终止 $j$ 的阶段代价；(c) 下一状态余值（$J^*(t)=0$ 不入和）。

### VI (4.5)

$J_{k+1}(i)$ 用 (4.4) 右端，$J^*$ 换 $J_k$。收敛见 Prop 4.2.1。

### 假设 4.2.1

存在 $m$，使**任意策略**下 $m$ 步内未到 $t$ 的概率 $\le\rho<1$：

$$
\rho=\max_\pi\max_i P\{x_m\neq t\mid x_0=i,\pi\}<1.
$$

则 $P\{x_{km}\neq t\}\le\rho^k\to 0$——视界随机但期望总代价有限。

**Proper / improper 策略**：proper = 从任意态以概率 1 到 $t$。假设 4.2.1 等价于所有平稳策略 proper；更弱假设：存在 proper 策略且 improper 策略在至少一态期望代价 $+\infty$（[BeT89], [Ber12] Ch.3）。

**无假设反例**：单态 $1$：停留代价 $a$ 或到 $t$ 代价 $b$；$a=0$ 或 $a<0$ 时 Bellman 无解或多解。

### 主要命题（附录 4.13.1）

- VI 收敛到 $J^*$；  
- Bellman 解唯一；  
- 满足 Bellman min 的 $\mu$ 最优（Prop 4.2.2–4.2.4 型）。

### Cost shaping

修改 $g\to\hat g=g+V(j)-V(i)$（SSP）不改变最优策略，改变 $J^*$ 数值；**近似 DP 次优策略**对 $V$ 敏感。

### 与 Ch.1 归约

- §1.3.1：$p_{ij}(u)\in\{0,1\}$ 的 SSP；  
- §1.3.3：$(x_k,k)$ 增广 + 合并 $t$ 的 SSP。

---

## §4.3 Discounted Problems

### Bellman (4.12)

$$
J^*(i)=\min_u\sum_j p_{ij}(u)\big[g(i,u,j)+\alpha J^*(j)\big],\quad \alpha\in(0,1).
$$

### 主要结果

- $T$ 为 $\alpha$-**压缩**（最大范数）：$\|TJ_1-TJ_2\|_\infty\le\alpha\|J_1-J_2\|_\infty$；  
- VI 收敛；$J^*$ 唯一；  
- $J_\mu$ 满足 $(I-\alpha P_\mu)J_\mu=g_\mu$。

### Q 因子与 Q-VI (4.17)

$$
Q^*(i,u)=\sum_j p_{ij}(u)\big[g(i,u,j)+\alpha J^*(j)\big],\quad J^*(i)=\min_u Q^*(i,u).
$$

Q 空间 VI 为 Q-learning 基础。

### 折扣 → SSP (Fig 4.3.1)

每步以 $1-\alpha$ 进 $t$，否则原转移 → 分析统一。

---

## §4.4 Exact and Approximate Value Iteration

### 精确 VI (4.18)–(4.19)

SSP 含 $p_{it}g(i,u,t)$ 项；折扣无此项。状态多时不可行。

### 近似 / Fitted VI

$\tilde J_{k+1}\approx T\tilde J_k$ + 回归（Ch.3 §3.3 无限时域版）。在样本态 $i^s$ 上拟合 $(T\tilde J_k)(i^s)$。

### 理想界 (4.22)

若 $\|\tilde J_{k+1}-T\tilde J_k\|_\infty\le\delta$，则  
$\|\tilde J_k-J^*\|_\infty\le\delta/(1-\alpha)$，  
$\|J_{\tilde\mu_k}-J^*\|_\infty\le 2\delta/(1-\alpha)^2$。

### 例 4.4.1

两状态链 $1\to2\to2\cdots$，零代价，LS-FVI 可 $\tilde J_k\to\infty$。

### 稳态加权启发

按“长期重要性”加权回归（稳态下访问频率）可启发式稳定——**无一般理论保证**（[Ber12] §6.3）；例 4.4.1 中应令 $\xi_2\gg\xi_1$。

---

## §4.5 Policy Iteration

### §4.5.1 Exact Policy Iteration

**SSP 评估**：

$$
J_{\mu_k}(i)=\sum_j p_{ij}(\mu_k(i))\big[g(i,\mu_k(i),j)+J_{\mu_k}(j)\big].
$$

**改进**：

$$
\mu_{k+1}(i)\in\arg\min_u\sum_j p_{ij}(u)\big[g(i,u,j)+J_{\mu_k}(j)\big].
$$

若 $J_{\mu_{k+1}}=J_{\mu_k}$ 则停止，$\mu_k$ 最优。**折扣**评估含 $\alpha$ 因子 (4.25)。

**性质**：有限策略集下有限步收敛（精确算术）；**近似 PI 的界常优于近似 VI**（同误差水平）。

### §4.5.2 Optimistic / Multistep Lookahead PI

评估：对 $J_{\mu_k}$ 仅做少量 VI 而非解线性系统；改进可近似。≈ 带回归的短评估 FVI。**仍受** 4.4.1 放大影响。

### §4.5.3 Policy Iteration for Q-Factors

Q 空间评估/改进；避免显式 $J_\mu$。异步/乐观 Q-PI 有收敛复杂性（[WiB93]；[BeY12], [YuB13a] 解决折扣/SSP）。

### 与 Rollout

Rollout = 一次评估+改进；PI = 重复至最优。

---

## §4.6 Approximation in Value Space — Performance Bounds

### 框架 (4.37)

一步前瞻：$\tilde\mu(i)\in\arg\min_u\sum_j p_{ij}(u)[g+\alpha\tilde J(j)]$。$\tilde J$ 来源同 Ch.2（Fig 4.6.1）。

**近似 PI 序列** $\mu_0,\ldots,\mu_m$：评估 $\tilde J_{\mu_k}$ → 改进 → 末 $\tilde J$ 作前瞻。

### §4.6.1 Limited Lookahead

**Prop 4.6.1(a)**：$\ell$ 步前瞻 $\tilde\mu$ 满足 $\|J_{\tilde\mu}-J^*\|_\infty\le \frac{2\alpha^\ell}{1-\alpha}\|\tilde J-J^*\|_\infty$。

**Prop 4.6.1(b)**：控集 $U(i)\subset U(i)$，$\hat J\le\tilde J+c$ → $J_{\tilde\mu}(i)\le\hat J(i)+c/(1-\alpha)$。

### §4.6.2 Rollout

截断 rollout + 终端 $\tilde J$；TeG96、AlphaGo。终端近似可能破坏相对基策略改进保证。

### §4.6.3 Approximate PI

(4.44)–(4.45) → **Prop 4.6.4**：$\limsup\|J_{\mu_k}-J^*\|_\infty\le(\epsilon+2\alpha\delta)/(1-\alpha)^2$；Fig 4.6.4 振荡带。

**Prop 4.6.5**：若 $\mu_k$ 收敛（聚合），界 $\to(\epsilon+2\alpha\delta)/(1-\alpha)$。

**对比 4.4.1**：近似 PI 中 $\{J_{\mu_k}\}$ 有界，无 $\tilde J\to\infty$ 类不稳定性。

---

## §4.7 Simulation-Based PI with Parametric Approximation

### §4.7.1 Self-Learning and Actor–Critic

Critic：仿真 + LS/TD → $\tilde J_{\mu_k}$。Actor：Bellman 贪心或策略回归。Self-learning = 仿真 PI，非系统辨识。

### §4.7.2 Model-Based

知 $p_{ij}$：精确期望目标、稳态加权训练（详 Ch.5 §5.3.2）。

### §4.7.3 Model-Free

模拟器 $(i,u,g,j)$ 更新 critic/Q。

### §4.7.4 Implementation Issues

探索、$\epsilon$-greedy、振荡（[BeT96] §6.4）；与 Ch.5 §5.3.4–5.3.5 衔接。

---

## §4.8 Q-Learning

$Q^*$ 满足 Q-Bellman；算子 $F$ 为 $\alpha$-收缩。

**Watkins 更新**：采样 $(i_k,u_k)$，$j_k\sim p_{i_k\cdot}(u_k)$，

$$
Q_{k+1}(i_k,u_k)\leftarrow (1-\gamma^k)Q_k(i_k,u_k)+\gamma^k\big(g(i_k,u_k,j_k)+\alpha\min_v Q_k(j_k,v)\big).
$$

**无模型**：仅需 $(i,u,g,j)$ 样本。

**收敛（标准 SA）**：有限 MDP、各 $(i,u)$ 无限访问、$\sum_k\gamma^k=\infty$、$\sum_k(\gamma^k)^2<\infty$ 等 $\Rightarrow$ $Q_k\to Q^*$ w.p.1（[BeT96], [Ber12]）。

**Off-policy**：行为策略可与 max-Q 目标不同；函数逼近时需修正（§4.9）。

扩展见 [BeT96] 与 Ch.5 §5.4。

---

## §4.9 Additional Methods — Temporal Differences

### TD(0)

$$
J(i_k)\leftarrow J(i_k)+\gamma^k\big(g_k+\alpha J(i_{k+1})-J(i_k)\big).
$$

### TD($\lambda$)

$n$-step 回报与 eligibility trace 混合；$\lambda=1$ 接近 Monte Carlo，$\lambda=0$ 为 TD(0)。

### SARSA vs Q-learning

SARSA：on-policy，用 $u_{k+1}$；Q-learning：off-policy，用 $\min_v Q(j,v)$。

### 与 DP

Backup = 单态 $(T_\mu J)(i)$ 或 $(TJ)(i)$；sweep = 全 $i$。

### Deadly triad

函数逼近 + bootstrapping + off-policy → 可能发散；需投影/修正（[SuB18], [Ber12]）。

---

## §4.10 Exact and Approximate Linear Programming

**精确 LP**：变量 $J(i)$，约束 $J(i)\ge\sum_j p_{ij}(u)[g+\alpha J(j)]$（或等价形式），$\min\sum_i\beta_i J(i)$，$\beta_i>0$ → 最优 $J^*$。

**近似 LP**：$J(i,r)$ 低维；或随机子集约束。变量数仍可能大。

---

## §4.11 Approximation in Policy Space

### §4.11.1 Policy Gradient 等

$\mu(i,\theta)$；最大化 $J_\mu(\theta)$ 的样本梯度（REINFORCE：$\nabla_\theta J\approx$ 轨迹回报 × $\nabla_\theta\log P_\mu$）。

### §4.11.2 Expert Supervised Training

专家 $(i,u)$ 监督 $\mu_\theta$；与自博弈（AlphaZero）对比。

---

## §4.12 Notes and Sources

[Ber12], [Ber17], [BeS78], [Ber18a], [Put94], [BeT96], [SuB18] 等为入口；本章参考文献非 exhaustive。

---

## §4.13 Appendix（证明导读）

| 小节 | 内容 |
|------|------|
| **4.13.1** | SSP：Assumption 4.2.1 → VI 收敛、Bellman 唯一、最优平稳策略存在 |
| **4.13.2** | 折扣：压缩、唯一解、$J_\mu$ 线性方程 |
| **4.13.3** | 精确/乐观 PI 有限终止 |
| **4.13.4** | 一步前瞻、Rollout、近似 PI **界**的证明 |

---

## 本章小结

| 需求 | 节 |
|------|-----|
| 理论底座 | §4.1–4.3 + 4.13 |
| 小 MDP | VI / PI 表格 |
| 大状态 | §4.4 FVI + Ch.5 |
| 无模型 RL | §4.8, §4.7.3, §4.9 |
| 次优界 | §4.6 |

---

*个人学习笔记；原著 Copyright Bertsekas / Athena Scientific。*