# 第 5 章 Infinite Horizon Approximate Methods — 分节笔记

> **文献**：Bertsekas, *RL and Optimal Control* Ch.5（2019 draft，2019-04）。  
> **文本**：`source/ch05_clean.txt`、`source/parts/ch05_part*.txt`。  
> **位置**：Ch.4 给出 SSP/折扣下的精确 VI/PI/Q-learning；本章在**大状态空间**下讨论参数化 FVI、仿真 PI、性能界、探索与振荡，并与 Ch.2–3 的实现细节衔接。本 PDF 版无独立 Aggregation 章，相关内容见 [Ber12] 与书中片段。

---

## 章首

无限时域近似方法的共同模板：

1. 用 $\tilde J(i,r)$ 或 $\tilde Q(i,u,r)$ 近似 Bellman 对象；  
2. 用**模拟器**或已知 $p_{ij}(u)$ 生成训练目标；  
3. 在**理想逐点误差**假设下建立界（§5.1），并承认**最小二乘 FVI** 可能违背该假设（例 5.2.1）。

算子 $T,T_\mu$ 在 (5.1)–(5.3) 中与 Ch.4 重述，便于对照公式编号。

---

## §5.1 Approximation in Value Space — Performance Bounds

无限时域值空间近似：先得 $\tilde J\approx J^*$，再一步或多步前瞻实现 $\tilde\mu$（Fig 5.1.1 与 Ch.2 Fig 2.1.1 同构）。$\ell$ 步前瞻的有效终端余值为 $T^{\ell-1}\hat J$（$\hat J$ 为初始猜测）。

**近似 PI 流程**（Ch.4 §4.6 同型）：生成 $\mu_0,\ldots,\mu_m$；近似评估 $\tilde J_{\mu_k}$；改进得 $\mu_{k+1}$；末策略评估作前瞻 $\tilde J$。**Rollout** = $m=0$ 的单次评估+改进特例。

---

### §5.1.1 Limited Lookahead

#### 主要内容

**$\ell$ 步前瞻策略** $\tilde\mu$：在 $i$ 上最小化

$$
\min_{\mu_0,\ldots,\mu_{\ell-1}} \mathbb{E}\Big[\sum_{k=0}^{\ell-1}\alpha^k g(i_k,\mu_k(i_k),j_k) + \alpha^\ell \tilde J(i_\ell)\Big],
$$

仅执行首控；等价于 $T_{\tilde\mu}(T^{\ell-1}\tilde J)$ 所定义的贪心（算子记号见 Ch.4）。

**Prop 5.1.1(a) / 4.6.1(a)**：

$$
\|J_{\tilde\mu}-J^*\|_\infty \le \frac{2\alpha^\ell}{1-\alpha}\|\tilde J-J^*\|_\infty. \tag{5.x / 4.38}
$$

**受限控集前瞻**：在 $U(i)\subset U(i)$ 上 min，若 $\hat J(i)\le \tilde J(i)+c$，则 $J_{\tilde\mu}(i)\le \hat J(i)+c/(1-\alpha)$（Prop 4.6.1(b)）——用启发式筛控再前瞻可省计算。

#### 要点

- 界对 $\tilde J$ 加**常数向量**不变（只依赖差分），与 Ch.2 Q 常数偏移讨论一致。  
- 截断 rollout 引入终端 $\tilde J$ 后，**相对基策略的改进性**可能丧失（Ch.4 §4.6.2）。

#### 注意点

- 多步界较一步多因子 $2\alpha^\ell/(1-\alpha)$；$\ell$ 大时更依赖内层精确优化。

---

### §5.1.2 Rollout

#### 主要内容

**截断 rollout**（Fig 5.1.2 / 4.6.3）：基策略 $\bar\mu$ 仿真 $H$ 步 + 终端 $\tilde J$（可来自问题近似、离线回归 $J_{\bar\mu}$ 样本、或 NN）。

**Backgammon [TeG96]**：TD-Gammon 为 $\bar\mu$ 与终端；**AlphaGo [SHM16]**：深度网络 + MCTS 作多步前瞻。

#### 性能界

在序贯改进/SSP 或折扣假设下，$J_{\tilde\mu}\le J_{\bar\mu}$ 及 $\|J_{\tilde\mu}-J^*\|$ 上界见 **附录 5.9.2**（与 4.13.4 平行）。

#### 注意点

终端 $\tilde J$ 为启发式或回归时，不再保证 rollout 优于 $\bar\mu$ 的严格序贯改进证明前提。

---

### §5.1.3 Approximate Policy Iteration

#### 主要内容

**近似评估** (4.44) / (5.x)：

$$
\|\tilde J_{\mu_k}-J_{\mu_k}\|_\infty \le \delta.
$$

**近似改进** (4.45)：

$$
\max_i\Big[ T_{\mu_{k+1}}\tilde J_{\mu_k}(i) - T\tilde J_{\mu_k}(i)\Big] \le \epsilon.
$$

**Prop 5.1.4 / 4.6.4**（折扣）：

$$
\limsup_{k\to\infty}\|J_{\mu_k}-J^*\|_\infty \le \frac{\epsilon+2\alpha\delta}{(1-\alpha)^2}.
$$

**典型行为**（Fig 5.1.4 / 4.6.4）：早期单调下降，进入宽度 $\approx(\epsilon+2\alpha\delta)/(1-\alpha)^2$ 的**误差带**后 $J_{\mu_k}$ 随机振荡；界常 pessimistic，实际带常更窄。

**策略收敛情形**（聚合等，(4.46)）：Prop 4.6.5 界收紧为 $(\epsilon+2\alpha\delta)/(1-\alpha)$（Fig 4.6.5）。

**乐观 PI**：评估仅少量 VI + 回归；界结构类似但推导更繁；仍受例 5.2.1 **误差放大**影响——近似 VI = 单次评估的乐观 PI 特例。

#### 要点

- 有限策略集 $\Rightarrow$ $\{J_{\mu_k}\}$ 有界；**近似 PI 不受** 4.4.1 类 $\tilde J_k\to\infty$ 不稳定性（与近似 VI 对比）。  
- SSP 有平行界（[BeT96] §6.2.2）。

---

## §5.2 Fitted Value Iteration

### 算法流程

1. 初始化 $\tilde J_0$（参数 $r_0$）。  
2. 迭代 $k=0,1,\ldots$：  
   - 对样本态 $i^s$ 计算 $(T\tilde J_k)(i^s)$（需 $p_{ij}$ 或仿真）；  
   - 回归 $\tilde J_{k+1}$ 使 $\|\tilde J_{k+1}-T\tilde J_k\|$ 在样本意义下小。  
3. 用最终 $\tilde J$ 作前瞻 (5.1) 式 min。

与 Ch.3 有限时域 FVI 同型；平稳问题各阶段共用 $(\phi,r)$。

### 理想误差界 (5.17)–(5.19)

若 $\|\tilde J_{k+1}-T\tilde J_k\|_\infty\le\delta$ 对所有 $k,i$，则渐近  
$\|\tilde J_k-J^*\|_\infty\le\delta/(1-\alpha)$，  
$\|J_{\tilde\mu_k}-J^*\|_\infty\le 2\delta/(1-\alpha)^2$（$\tilde\mu_k$ 为对 $\tilde J_k$ 贪心）。

### 例 5.2.1（与 4.4.1 同型）

两状态、单策略、零阶段代价、$1\to2\to2\to\cdots$，折扣 $\alpha$。最小二乘 FVI 可令 $\tilde J_k$ 无界增长——**自然 LS 回归不保证** (5.19)。

**脚注（加权）**：若按“长期重要性”加权，令 $\xi_2\gg\xi_1$，可使回归标量 $\zeta$ 满足 $\alpha\zeta<1$，保证 $\tilde J_k\to J^*$——说明**样本权重/分布**对 FVI 稳定性至关重要。

### 实践要点

- 样本分布应接近**最优或近最优策略的稳态分布**（Ch.4 脚注）。  
- 与乐观 PI：一次 $T\tilde J$ + LS ≈ 一轮评估。

---

## §5.3 Simulation-Based Policy Iteration with Parametric Approximation

折扣问题为主；SSP 可类比。核心：**Critic**（评估 $\tilde J_{\mu_k}$ 或 $\tilde Q_{\mu_k}$）+ **Actor**（改进 $\mu_{k+1}$）。

---

### §5.3.1 Self-Learning and Actor–Critic

#### 主要内容

**Self-learning**：仿真 PI + 参数化评估；系统通过**观察自身在 $\mu_k$ 下产生的轨迹**更新 critic，actor 做 Bellman 贪心或样本上 min + 策略回归。

**两步骤**（每轮 PI）：

**(a) Critic**：在 $\mu_k$ 下采样代价，增量/LS 拟合 $\tilde J_{\mu_k}(i,r)$。  
**(b) Actor**：

$$
\mu_{k+1}(i)\in\arg\min_{u\in U(i)}\sum_j p_{ij}(u)\big[g(i,u,j)+\alpha\tilde J_{\mu_k}(j,r)\big],
$$

或在样本态 $i^s$ 上算 $u^s$ 再 $\mu_{k+1}(i,r)$ 回归（§2.1.5）。

**区分**：学习的是**更好策略**，非系统辨识——不学 $p_{ij}$ 的显式模型（两阶段：先系统辨识再 model-based PI 本书不展开）。

#### 终止

收敛于误差带，或 **策略振荡** 时学习停滞（Fig 5.1.4–5.1.5）。

---

### §5.3.2 A Model-Based Variant

#### 评估 (5.22)

样本 $(i^s,\beta^s)$，$\beta^s$ = 从 $i^s$ 起用 $\mu$ 仿真 $N$ 步折扣代价 + $\alpha^N \hat J(i_N)$：

$$
r\in\arg\min_r\sum_s\big(\tilde J_\mu(i^s,r)-\beta^s\big)^2.
$$

$\hat J$ 可取上一策略评估、零、或问题近似——类似**乐观 PI** 终端。

**增量梯度**：

$$
r_{k+1}=r_k-\gamma_k\nabla_{r}\big(\tilde J(i^s_k,r_k)-\beta^{s_k}\big)^2.
$$

线性架构可闭式解。

#### 改进 (5.23)

$$
\tilde\mu(i)\in\arg\min_u\sum_j p_{ij}(u)\big[g(i,u,j)+\alpha\tilde J(j,r)\big].
$$

#### 轨迹复用与 bias–variance

长轨迹 $(i_0,\ldots,i_N)$ 可从 $i_0,i_1,\ldots$ 各起算 tail 代价 → 省采样。  
**$N$ 小**：方差小、**偏差大**（尾部靠 $\alpha^N\hat J$）；**$N$ 大**：偏差小、方差与成本增。  
多短轨迹 + 多初态 → 更好**探索**。TD($\lambda$)、LSTD($\lambda$)、LSPE($\lambda$) 由此动机（§5.5）。

---

### §5.3.3 A Model-Free Variant

#### Q 架构 (5.24)–(5.25)

- $\tilde Q_\mu(i,u,r)=r(u)'\phi(i)$——控少时；  
- $\tilde Q_\mu(i,u,r)=r'\phi(i,u)$——一般。

#### 评估 (5.26)

三元组 $(i^s,u^s,\beta^s)$：首步用 $u^s$，其后 $\mu$ 共 $N$ 步，$\beta^s$ 估计 $N$ 阶段 Q：

$$
Q^N_\mu(i,u)=\sum_j p_{ij}(u)\big[g(i,u,j)+\alpha J_{\mu}^{N-1}(j)\big].
$$

#### 改进 (5.27)

$\tilde\mu(i)=\arg\min_u \tilde Q_\mu(i,u,r)$。

#### 探索缺陷

轨迹复用后样本多为 $(i,\mu(i))$，**$u\neq\mu(i)$** 的 $(i,u)$ 覆盖不足 → 需刻意从多样 $(i,u)$ 起仿真。

#### 两阶段替代

先 model-free 得 $\tilde J_\mu$，再采样回归 $\tilde Q$ 近似 $\sum_j p_{ij}(u)[g+\alpha\tilde J_\mu(j)]$ + 策略近似——更复杂，但利于轨迹复用。

---

### §5.3.4 Implementation Issues

#### 架构与 cost shaping

线性架构 → (5.22)(5.26) 闭式 + TD 族理论保证。  
**Cost shaping**（Ch.4 §4.2）：  
SSP：$\hat g(i,u,j)=g(i,u,j)+V(j)-V(i)$；  
折扣：$\hat g=g+\alpha V(j)-V(i)$。  
最优策略不变，但**次优策略**会变；$V$ 应接近 $J^*$ 或 $J_{\mu_k}$，使 $J^*-V$ 易逼近；可 NN 学 $V$ + 局部修正。

#### 探索

**轨迹复用偏置**：常访问态过代表，罕见态评估差 → 改进步大错。  
**记忆缓冲**：初态集 $I\cup I_0\cup\cdots\cup I_k$，$I_m$ 为评估 $\mu_m$ 时产生的态；评估 $\mu_k$ 时从各集按**偏近期**概率抽初态；短轨迹 + 准终端 $\hat J$。  
SSP **深探索**：晚期大代价时需长轨迹接近终止。  
Q 评估：更需在 $(i,u)$ 空间探索。

**Off-policy / behavior policy**：混合 target policy 与探索 policy；$\epsilon$-greedy 等。**偏差校正**需特殊修改（[Ber12] §6.4.2）。探索–利用权衡仍为开放问题（[RuV16], [RVK18], [OVR19] deep exploration）。

---

### §5.3.5 Oscillations

#### 机制：Greedy partition

对架构 $\tilde J(\cdot,r)$，定义

$$
\mathcal{R}_\mu=\Big\{r\;\Big|\;\mu(i)=\arg\min_u\sum_j p_{ij}(u)[g+\alpha\tilde J(j,r)],\;\forall i\Big\}.
$$

非乐观 PI：评估得唯一 $r_\mu$，若 $r_{\mu_k}\in\mathcal{R}_{\mu_{k+1}}$ 且永不出现 $r_\mu\in\mathcal{R}_\mu$，则策略在有限环上**循环**（Fig 5.3.3）。查表时 $r_\mu=J_\mu$，$r_\mu\in\mathcal{R}_\mu\Leftrightarrow J_\mu=TJ_\mu\Leftrightarrow\mu$ 最优；有函数逼近时最优性不再等价。

#### 乐观 PI

类似振荡；额外与 FVI 误差放大耦合。

#### 经验

[BeT96] §6.4.2：振荡未必严重损害**平均**性能，但工程上应监测策略周期。

---

## §5.4 Q-Learning

### 基本思想

在 Q 空间做 VI/PI；$F$ 为 $\alpha$-收缩（(4.51)）。随机更新 (4.52) 为 $FQ$ 的 SA。

### §5.4.1 Optimistic PI + Parametric Q — SARSA & DQN

**SARSA（on-policy）**：TD 目标用**实际**下一动作 $u'$：

$$
\beta = g(i,u,j)+\alpha\tilde Q(j,u',r).
$$

**Q-learning（off-policy 常见）**：目标用 $\min_v Q(j,v)$。  
**DQN**：$\tilde Q(i,u,\theta)$ 为深度网络；**经验回放**打破样本相关；**目标网络**延迟更新 stabilizing（[GBC16]）。  
**乐观 PI + 参数 Q**：Q 评估仅若干步 + 回归 + 改进。

---

## §5.5 Additional Methods — Temporal Differences

### 主要内容

**TD(0)**：单步 bootstrap 更新 $J_\mu$ 或 $V$。  
**TD($\lambda$)**：$n$-step 与 eligibility trace 加权，偏差–方差折中。  
**LSTD($\lambda$)**、**LSPE($\lambda$)**：线性架构下批量/增量最小二乘 TD 解。  
与 FVI/Q-learning 统一：**采样 Bellman backup + 函数逼近**。

### 要点

半梯度 TD 收敛需 on-policy 或特殊 off-policy 修正；见 [Ber12] §6.3、[SuB18]。

### 与 §5.3.2

短轨迹 bias–variance 是 TD($\lambda$) 族的重要动机。

---

## §5.6 Exact and Approximate Linear Programming

Bellman 不等式 $J\le TJ$（分量形式）与线性目标 $\min\sum_i\beta_i J(i)$，$\beta_i>0$ 得 $J^*$（精确 formulation 下）。  
**近似 LP**：$J(i,r)$ 低维参数化或随机约束子集；规模仍可能大。与某些对偶 RL 方法相关。

---

## §5.7 Approximation in Policy Space

### §5.7.1 Policy Gradient、CEM、Random Search

直接优化 $J_\mu(\theta)$ 或长期平均代价；**REINFORCE** 类用轨迹梯度。**交叉熵法 (CEM)**：采样参数、保留精英、重采样——无梯度黑箱。与 Actor–Critic：PG actor + TD critic。

### §5.7.2 Expert Supervised Training

行为克隆 $\mu_\theta$ 拟合专家；**covariate shift** 当测试态偏离专家分布时性能下降。

### §5.7.3 Approximate PI + Rollout + Policy Approx

Rollout 作评估或终端；在线增量 LS (5.74) 随 $(i^s,u^s)$ 更新 actor/critic。

---

## §5.8 Notes and Sources

FVI 病理、振荡、探索文献；与 Ch.4 §4.12 互补。

---

## §5.9 Appendix（证明导读）

| 节 | 内容 |
|----|------|
| 5.9.1 | 一步/多步前瞻界 |
| 5.9.2 | Rollout 界 |
| 5.9.3 | 近似 PI 界 (5.1.3) |

与 Ch.4 **4.13.4** 对照。

---

## Ch.4 与 Ch.5 分工

| | Ch.4 | Ch.5 |
|---|------|------|
| 理论 | SSP/折扣、精确 VI/PI、Q 收敛 | 同算子 + **回归误差** |
| 算法 | 表格 | FVI、仿真 Actor–Critic、DQN |
| 风险 | 例 4.4.1 | 例 5.2.1、§5.3.5 振荡 |

---

## 本章小结

无限时域近似 RL = Bellman 收缩（理想界）+ 函数逼近与采样（可能破坏假设）。Actor–Critic 将评估与改进拆为可仿真模块；Q-learning/DQN 在 Q 空间避免每轮解 $J_\mu$ 线性系统。

---

*个人学习笔记；原著 Copyright Bertsekas / Athena Scientific。*