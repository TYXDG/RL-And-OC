# 序列二次规划与有限时域最优控制iLQR、DDP、MPC

> **文献组**（四篇一体）  
> 1. Jordana 等 — *Stagewise Implementations of SQP for MPC* (2023)  
> 2. Jordana 等 — *Structure-Exploiting SQP for MPC* (*IEEE TRO*, 2025)  
> 3. Abhijeet & Chakravorty — *A Sequential Quadratic Programming Perspective on Optimal Control* (arXiv:2510.03475, 2025)  
> 4. Jordana 等 — *Supplementary Material*（KKT → Thomas → Riccati 证明）  
>
> **仓库对照**：[`bertsekas-rl-oc`](../../bertsekas-rl-oc/study-notes/) · [`zero-order-robotics`](../../zero-order-robotics/study-notes/)

---

## 阅读地图

| 章节 | 内容 |
|------|------|
| **§0** | 概述与主要结论 |
| §1 | 问题与文献分工 |
| §2 | 统一问题：OCP = NLP |
| §3 | SQP 透镜：Newton vs 修正 QP |
| §4 | 三轴分类：Hessian / 打靶 / 前向 rollout |
| §5 | Chakravorty 深读：iLQR / DDP / Newton |
| §6 | Jordana 深读：结构利用与约束 NMPC |
| §7 | 附录数学：KKT 块三对角 = Riccati |
| §8 | 实验与数字：两套 benchmark 的对照 |
| §9 | 与 Bertsekas、零阶教程的对照 |
| §10 | 选型、实现、开放问题 |

---

## §0 概述

有限时域 OCP 可表述为带动力学等式与路径/终端不等式的 NLP；MPC 为滚动求解该 NLP。iLQR、DDP、FDDP、GNMS 等可统一为：**SQP 外层迭代 + 内层 QP（时变 LQR）+ 不同的 Hessian 近似、打靶方式与前向 rollout**。

Chakravorty 从局部 QP 是否给出目标下降方向分析算法差异；Jordana 从 KKT 能否在 $O(T)$ 内求解分析实现差异。

### 0.1 主要结论

1. **修正 SQP**（QP Hessian 取 $c_{xx}$，去掉 $\bar\lambda h_{xx}$）：在 $c_{xx} \succ 0$、$\bar h=0$ 时，QP 解方向为 $c(x)$ 的下降方向；iLQR 对应该修正 QP。
2. **DDP backward pass** 仅在名义控制为 $u^*(x)$ 时等价于 Bellman 的二阶展开；一般名义点下为启发式，$Q_{uu}$ 可不定，预测 $\Delta J$ 与 rollout 代价可不一致。
3. **Riccati 递推** 等价于多重打靶 KKT 的块三对角系统经 Thomas 消元（Supplementary Prop. 1–2）。
4. **GNMS / multiple-shooting iLQR** 在结构上等价于修正 SQP；Jordana 给出 filter line search 与 KKT 终止准则下的完整表述。
5. **约束 NMPC**：瓶颈在 inner QP 的 $O(T)$ 求解；OSQP_OCP 将线性化动力学置于 $\mathrm{Dom}(g)$，ADMM 每步解 extended LQR 并对不等式分阶段投影。
6. **Benchmark**：FDDP 与 SQP(filter) 若仅差非线性/线性 rollout，Humanoid 上 SQP 收敛约快 4 倍；FDDP 的 gap 闭合缺乏 multiple-shooting SQP 型全局收敛理论。
7. **Regime**：近最优时 DDP/Newton 迭代数常更少；远离最优时 iLQR/GN-SQP 更稳健（Chakravorty §IV-D）。

### 0.2 算法分歧的三轴

```text
  Hessian：完整 Lagrangian / 仅代价 Hessian / DDP 式 λ→v 替换
  打靶：single shooting / multiple shooting（defect 是否显式）
  前向：线性 defect 更新 / 非线性 rollout
```

DDP 与 SQP 传统表述在无不等式、且 Hessian 取法相同时，可对应同一 QP 的 KKT；差异来自上列组合及其理论前提。

### 0.3 文献分工

| 文献 | 侧重 |
|------|------|
| Chakravorty 2025 | Newton / iLQR / DDP 的 SQP 对照；低维 swing-up 实验 |
| Jordana 2023 | Stagewise SQP、OSQP_OCP 雏形、与 FDDP 对照 |
| Jordana TRO 2025 | 约束 NMPC 真机、QP 计时、KUKA/Solo |
| Supplementary | KKT 块三对角 ↔ Thomas ↔ Riccati |

---

## §1 问题与文献分工

### 1.1 中心问题

离散有限时域 OCP/MPC 是否必须依赖 DDP 系实现，还是 SQP + 结构 QP 已足够？

文献结论：在利用时域稀疏结构求解 inner QP 的前提下，SQP 足够；含不等式约束时，multiple shooting + KKT/QP 处理较 penalty 方法更直接。

### 1.2 与控制理论、数值优化的关系

- **控制理论**：Bellman/HJB、反馈策略、稳定性等最优性结构。
- **数值优化**：KKT 条件、下降方向、步长、约束处理与收敛性。
- **本组文献**：已知模型、有限时域、在线 MPC 下，Riccati/DDP 命名可视为 OCP-KKT 求解的特化形式。

---

## §2 统一问题表述

### 2.1 Jordana：多重打靶约束 OCP（最一般）

```math
\min_{x,u} \sum_{k=0}^{T-1} \ell_k(x_k, u_k) + \ell_T(x_T)
```

```math
\text{s.t.}\quad x_{k+1} = f_k(x_k, u_k),\quad c_k(x_k,u_k) \ge 0,\quad c_T(x_T) \ge 0,\quad x_0 = \bar{x}_0.
```

**决策变量**：状态轨迹 $x = (x_1,\ldots,x_T)$ **与** 控制 $u = (u_0,\ldots,u_{T-1})$ **同时优化**。

**Defect（动力学间隙）**：

```math
\gamma_{k+1} = f_k(x_k, u_k) - x_{k+1}.
```

- 迭代中允许 $\gamma \ne 0$（**infeasible iterate**）→ warm-start 友好、不可行动力学初猜可用。
- SQP 逐步减小 $\|\gamma\|$ 至 KKT 容忍度。

**Lagrangian**（Jordana 式 2）：

```math
\mathcal{L} = \ell_T - \mu_T^\top c_T + \sum_{k=0}^{T-1} \Big[ \ell_k - \lambda_{k+1}^\top (x_{k+1} - f_k) - \mu_k^\top c_k \Big].
```

乘子 $\lambda_{k+1}$ 附着于 **defect 约束** $x_{k+1} - f_k = 0$，这是 SQP 与 DDP 伴随变量 **对齐的接口**。

### 2.2 Chakravorty：single shooting 简化（便于理论）

```math
\min_{\{u_t\}_{t=0}^{T-1}} \sum_{t=0}^{T-1} c(x_t,u_t) + C_T(x_T),\quad x_{t+1} = f(x_t,u_t),\quad x_0\ \text{给定}.
```

**Assumption 1**：$c(x,u) = \ell(x) + \tfrac{1}{2} u^\top R u$，$R \succ 0$，$\ell_{xx} \succeq 0$。

**Assumption 2**：$f$ 对 $u$ 仿射：$x_{t+1} = \bar f(x_t) + \bar g(x_t) u_t$。

**当前迭代假设动力学可行**：$h(\bar x) = 0$（Chakravorty 在 SQP 基础节假设 defect 为零），扰动 $(\delta x, \delta u)$ 沿 **线性化动力学 + 非线性 rollout 接受步** 推进。

### 2.3 Single vs multiple shooting

| | Single shooting | Multiple shooting |
|---|-----------------|-------------------|
| 决策变量 | 主要是 $u$；$x$ 由 rollout 隐式 | $x$ 与 $u$ 均为变量 |
| 无不等式时 | **无约束 NLP** → Newton / DDP | **等式约束 NLP** → SQP |
| 初猜 | 常需 **动力学可行** 轨线 | 可 **不可行**（gap 闭合） |
| 稀疏性 | 稠密约化 Hessian | KKT **块三对角** |
| 代表算法 | DDP, iLQR (classic) | GNMS, FDDP, Jordana SQP |
| 约束 | penalty / 增广 Lagrangian 启发式多 | **原生不等式 QP 子问题** |

**Jordana Remark 1**：single shooting + 无不等式 → Newton (Dunn 1989) 或 **非线性 rollout 的 modified Newton = DDP**。

### 2.4 MPC 闭环

每控制周期 $t$：

1. 测量 $\bar{x}_0 \leftarrow \hat{x}(t)$；
2. 解 horizon $T$ 的 OCP（SQP 迭代至 $\epsilon_{\mathrm{SQP}}$ 或迭代上限）；
3. 施加 $u_0^*$；平移 warm-start 轨迹；
4. $t \leftarrow t+1$。

**Jordana 经验**：无约束 500 Hz / 10 nodes / 5 SQP iter；**有约束** 降至 100 Hz 以换 4 SQP × 50 QP(ADMM) iter → **收敛质量 vs 反应速度** 的显式 trade-off。

---

## §3 SQP 透镜：从 NLP 到 LQR

### 3.1 一般等式约束 NLP

```math
\min_x c(x)\ \text{s.t.}\ h(x)=0.
```

Lagrangian $\mathcal{L} = c + \lambda^\top h$。在 $(\bar{x}, \bar\lambda)$ 处二阶展开，Newton 步 $(\delta x, \delta\lambda)$ 满足 KKT：

```math
\begin{bmatrix} c_{xx} + \bar\lambda h_{xx} & h_x^\top \\ h_x & 0 \end{bmatrix}
\begin{bmatrix} \delta x \\ \delta\lambda \end{bmatrix}
= - \begin{bmatrix} c_x + \bar\lambda h_x \\ \bar h \end{bmatrix}.
```

等价于解 **QP**：

```math
\min_{\delta x}\ c_x^\top \delta x + \tfrac{1}{2}\,\delta x^\top \underbrace{(c_{xx} + \bar\lambda h_{xx})}_{\text{Lagrangian Hessian 在 }x\text{ 块}} \delta x
\quad \text{s.t.}\quad h_x \delta x + \bar h = 0.
\tag{Newton-QP}
```

**修正 QP**（Chakravorty 式 4）：把 Hessian 换成 **仅 $c_{xx}$**（去掉 $\bar\lambda h_{xx}$）：

```math
\min_{\delta x}\ c_x^\top \delta x + \tfrac{1}{2}\,\delta x^\top c_{xx}\, \delta x
\quad \text{s.t.}\quad h_x \delta x + \bar h = 0.
\tag{Modified-QP}
```

### 3.2 为什么 Modified-QP 恒为下降方向（Chakravorty Prop. 1）

设 $\delta x$ 满足线性化约束 $h_x \delta x + \bar h = 0$，且当前点 **可行** $\bar h = 0$，则 $h_x \delta x = 0$。

Modified-QP 的一阶最优性：$c_x + c_{xx} \delta x + h_x^\top \lambda = 0$。

左乘 $\delta x^\top$：

```math
\delta x^\top c_x = -\delta x^\top c_{xx} \delta x - \underbrace{\delta x^\top h_x^\top \lambda}_{=0} < 0 \quad \text{若 } c_{xx} \succ 0.
```

故 $\delta x$ 是 **目标 $c(x)$ 的下降方向**（对约束的一阶可行性在 $\bar h=0$ 时成立；大步长用 $\alpha \in (0,1]$ line search 维持 **非线性** 可行性）。

**Newton-QP 无此保证**：下降性依赖 $\nabla^2_{xx} \mathcal{L} = c_{xx} + \bar\lambda h_{xx}$ 在约束切空间上正定；远离最优时 $\bar\lambda$ 不准 → **不定 → 非下降**。

**约束 NLP 的 descent 判据**（Chakravorty 引用）：$\delta X = (\nabla^2 \mathcal{L})^{-1} \nabla \mathcal{L}$ 仅当 $\nabla^2 \mathcal{L} \succ 0$ 时有 $\delta X^\top \nabla \mathcal{L} > 0$。

### 3.3 映射到 OCP：两个 LQR 子问题

Chakravorty 在标量 $(x,u)$ 上写出 **Newton LQR** (7) 与 **iLQR** (8)：

**Newton LQR** — 对应 Newton-QP，Hessian 含 $\bar\lambda_{t+1} f_{xx}$：

```math
\min_{\{\delta u_t\}} \sum_t \Big[ \ell_{x_t} \delta x_t + \bar u_t^\top R \delta u_t + \tfrac{1}{2}\delta x_t^\top \ell_{xx} \delta x_t + \tfrac{1}{2}\delta u_t^\top R \delta u_t + \bar\lambda_{t+1} f_{xx} \delta x_t^2 \Big]
```

```math
\delta x_{t+1} = f_x \delta x_t + f_u \delta u_t.
```

**iLQR** — 对应 Modified-QP，**无** $\bar\lambda f_{xx}$ 项：

```math
\min_{\{\delta u_t\}} \sum_t \Big[ \ell_{x_t} \delta x_t + \bar u_t^\top R \delta u_t + \tfrac{1}{2}\delta x_t^\top \ell_{xx} \delta x_t + \tfrac{1}{2}\delta u_t^\top R \delta u_t \Big],\quad \delta x_{t+1} = f_x \delta x_t + f_u \delta u_t.
```

**Riccati 解的形式差异**（反向 pass）：

- Newton：$v_t = \ell_{x_t} + f_x^\top v_{t+1} - (f_x^\top V_{t+1} f_u + \bar\lambda_{t+1} f_{xu})^\top H^{-1}(\cdots)$，$V_t$ 含 $\bar\lambda f_{xx}$。
- iLQR：$v_t = \ell_{x_t} + f_x^\top v_{t+1} - f_x^\top V_{t+1} f_u H^{-1}(\cdots)$，$V_t$ 无 $\bar\lambda f_{xx}$。

**Prop. 2**：iLQR 步恒为下降方向 + line search 可维持可行；Newton LQR **不必**。

### 3.4 Jordana：完整 SQP 子问题（含不等式）

第 $n$ 次 SQP 迭代解 QP (3)：

```math
\min_{\Delta x, \Delta u}\ \sum_{k=0}^{T-1} \tfrac{1}{2} \begin{bmatrix}\Delta x_k \\ \Delta u_k\end{bmatrix}^\top \begin{bmatrix} Q_k & S_k \\ S_k^\top & R_k \end{bmatrix} \begin{bmatrix}\Delta x_k \\ \Delta u_k\end{bmatrix} + \begin{bmatrix} q_k \\ r_k \end{bmatrix}^\top \begin{bmatrix}\Delta x_k \\ \Delta u_k \end{bmatrix} + \tfrac{1}{2}\Delta x_T^\top Q_T \Delta x_T + q_T^\top \Delta x_T
```

```math
\text{s.t.}\ \Delta x_{k+1} = A_k \Delta x_k + B_k \Delta u_k + \gamma_{k+1},\quad D_k \Delta x_k + E_k \Delta u_k + \hat c_k \ge 0,\quad D_T \Delta x_T + \hat c_T \ge 0.
```

**$Q_k, S_k, R_k$**（式 4–5）：Lagrangian 对 $(x_k,u_k)$ 的 **完整 Hessian 块**（含 $\lambda^{[n]}, \mu^{[n]}$ 对动力学/约束曲率的贡献）。

**Gauss–Newton 近似**（式 16，忽略约束二阶项）：

```math
Q_k = \nabla^2_{xx} \ell_k,\quad S_k = \nabla^2_{xu} \ell_k,\quad R_k = \nabla^2_{uu} \ell_k.
```

→ 子问题 **与乘子无关** → 即 **GNMS = 修正 SQP**；无不等式时用 Riccati **闭式**解 inner QP。

**外层更新**：

```math
x^{[n+1]} = x^{[n]} + \alpha \Delta x,\quad u^{[n+1]} = u^{[n]} + \alpha \Delta u.
```

$\lambda^{[n+1]}, \mu^{[n+1]}$ 取 **QP 的 KKT 乘子**（Jordana 式 10c, 21–22 给出 Riccati 形式高效计算）。

---

## §4 三轴分类：Hessian、打靶与前向 rollout

将常用算法按 Hessian 近似、打靶方式、前向 rollout 三轴分类。

### 轴 I — Hessian（局部 QP）

| 层级 | 含义 | 算法 |
|------|------|------|
| **L0 修正 GN** | QP Hessian = 仅代价 $\nabla^2 \ell$ | iLQR, GNMS, SQP(GN) |
| **L1 Newton** | QP Hessian = 完整 $\nabla^2_{xx} \mathcal{L}$（含 $\lambda f_{xx}$） | Newton LQR, exact SQP |
| **L2 启发式 Newton** | 形如 L1，但 $\bar\lambda_t \to v_t$（当前 backward 变量） | DDP |
| **L3 Quasi-Newton** | 近似 $\nabla^2 \mathcal{L}$ | 部分 legacy 控制文献 |

**Chakravorty**：L0 在 transient 有 Prop. 1 下降保证；L1 近最优具二次收敛潜力；L2 无该保证且 $Q_{uu}$ 可不定。

### 轴 II — 打靶

| | Single | Multiple |
|---|--------|----------|
| Defect | 恒 0（隐式） | 显式 $\gamma$，可非零 |
| Warm-start | 需可行轨 | 宽松 |
| 与 SQP 关系 | 无约束 GN / Newton | **标准 Bock SQP** |
| 典型 | DDP, classic iLQR | GNMS, FDDP, Jordana SQP |

**Jordana §VII**：FDDP gap 闭合后不再打开时，行为介于 single 与 multiple shooting 之间。

### 轴 III — 前向 rollout

| | 线性 rollout | 非线性 rollout |
|---|--------------|----------------|
| 如何推进 | $\Delta x_{k+1} = A_k \Delta x_k + B_k \Delta u_k + \gamma_k$ | $x \leftarrow f(x+\alpha\delta x, u+\delta u)$ 积分 |
| 与 SQP 关系 | **标准 SQP 步** (6) | DDP/FDDP 传统 |
| 理论 | Nocedal–Wright SQP 全局/局部结论可继承 | multiple shooting 下 **全局收敛未证** |
| Jordana benchmark | SQP(filter) | FDDP(filter) — 仅 rollout 不同时 SQP 约快 4× |

统一 filter 与 KKT 容差后，线性 rollout 通常 backtracking 更少；非线性 rollout 时 QP 预测与接受步可不一致。

### 4.1 算法定位

| 算法 | Hessian 轴 | 打靶轴 | 前向轴 | SQP 对应 |
|------|-----------|--------|--------|----------|
| iLQR | L0 | Single | 非线性 rollout | Modified SQP |
| DDP | L2 | Single | 非线性 | ≈ Newton 启发式 |
| GNMS | L0 | Multiple | 线性 | Modified SQP（未命名） |
| FDDP | L2 | Multiple | 非线性 | DDP 族 + gap 启发式 |
| Jordana SQP(GN) | L0 | Multiple | 线性 | Modified SQP + 完整约束 |
| Jordana SQP(full) | L1 | Multiple | 线性 | Exact SQP |
| Newton LQR | L1 | Single | 非线性/线性 | Newton SQP |

---

## §5 Chakravorty 深读

### 5.1 DDP 为何不是 Bellman 的严格二阶展开（Prop. 3）

Bellman：

```math
V_t^*(x_t) = \min_u Q(x_t,u),\quad Q(x_t,u) = c(x_t,u) + V_{t+1}^*(f(x_t,u)).
```

定义最优反馈 $u_t^*(x)$。对 **最优轨** 上点 $(\bar x_t, \bar u_t = u_t^*(\bar x_t))$，扰动展开 **才** 与 DP 一致。

对 **一般名义** $(\bar x_t, \bar u_t)$（$\bar u_t \ne u_t^*(\bar x_t)$），有：

```math
V(\bar x_t + \delta x) = \min_{\delta u} Q(\bar x_t + \delta x, \bar u_t + \delta u) \ne V^*(\bar x_t + \delta x)
```

展开 $V^*$ 时需 **$\mathrm{d}u^*/\mathrm{d}x$** 项；DDP 的 (PB) 省略了最优反馈斜率与名义控制偏差 → **(PB) 不是 (B) 的推论**。

**推论**：DDP backward 公式 = Newton LQR 方程，但把 $\bar\lambda_t$ **替换为当前** $v_t$（满足 $v_t \approx \ell_{x_t} + f_x^\top v_{t+1} + \cdots$，缺 Newton 修正项）。当 **近最优**时 $(R \bar u_t + f_u^\top v_{t+1}) \approx 0$ → 两式重合；**远离最优**时偏离大。

### 5.2 $Q_{uu}$ 与预测代价 $\Delta J$

DDP/iLQR 预测的「期望代价下降」（式 13）：

```math
\Delta J = -\alpha \sum_{t=0}^{T-1} k_t^\top \underbrace{(R + f_u^\top V_{t+1} f_u)^{-1}}_{Q_{uu}^{-1}} k_t + O(\alpha^2).
```

- **iLQR**：在 mild 假设下 $Q_{uu} \succ 0$ → $\Delta J < 0$（预测下降）。
- **DDP**：$Q_{uu}$ 可 **负定/不定** → 预测代价下降量符号错误或量级失真。

**Cart-pole swing-up 首步（Chakravorty Table II）**：

| | $J$ | $\Delta J$ | $J + \Delta J$ |
|---|-----|-----------|----------------|
| DDP | $8.9\times 10^5$ | $-8.2\times 10^5$ | $7.0\times 10^4$ |
| 但 later iter | — | — | $J+\Delta J < J_{\min}=0$（DDP） |

Cart-pole：DDP $\Delta J = +3.0\times 10^7$，iLQR $\Delta J = -5.6\times 10^5$。

### 5.3 Line search

接受条件（Armijo 型，分母为线性预测下降）：

```math
\frac{J_{k+1} - J_k}{\alpha\, \delta x^\top \nabla J_k} > \sigma > 0.
```

- **iLQR**：预测与 **一阶 rollout** 一致 → backtracking 少 → pendulum 上 $\alpha$ 可全程 1。
- **DDP**：二阶预测与 rollout 不一致 → $\alpha$ 反复减小（cart-pole 近 0）。

**Hybrid**（Chakravorty Fig. 4）：DDP 的 $\alpha$ 崩溃处切换 iLQR 可恢复收敛。

### 5.4 近最优 regime

初值靠近最优时（Fig. 5）：

| 任务 | DDP 迭代 | iLQR 迭代 |
|------|----------|-----------|
| Pendulum swing-up | 5 | 8 |
| Cart-pole swing-up | 5 | 17 |

L2 + 非线性 rollout 近 Newton；L0 为 Gauss–Newton 型，全局更稳、局部较慢。

### 5.5 与 Pantoja (1988)

Pantoja 用当前伴随 $\nu_t$ 的 stagewise Newton；Chakravorty 的 Newton LQR 用上一轮乘子 $\bar\lambda$。二者仅在最优附近与 DDP 一致。

---

## §6 Jordana 深读

### 6.1 背景

Jordana Introduction：Bock–Plitt (1984) multiple shooting 即 QP (3) 结构；1990s–2010s 优化侧 SQP/IPM 与结构 QP 求解器（Wright, Pantoja, acados, HPIPM 等）与机器人侧 iLQR/DDP 长期并行、实验少对标；2023 benchmark 起与 FDDP 等同条件比较。

### 6.2 无约束 inner QP

见 **§7** 完整推导。此处强调 **计算复杂度**：

| 方法 | 复杂度（主导项） |
|------|------------------|
| Bock 稀疏直接法 | $O(T^3)$ 量级 |
| Thomas + 结构 | $O(T \cdot m^3)$，$m = \dim u$ |
| Parallel cyclic reduction | $O(\log T \cdot m^3)$ |

### 6.3 约束 inner QP：OSQP_OCP

**标准 OSQP** 形式：$\min_v g(v)$ s.t. $Pv \in \mathcal{C}$ —— 动力学与一般等式 **同在** $P$ 中 → ADMM 迭代中 dynamics **不保证每步可行**。

**OSQP_OCP 分裂**：

- $v = (\Delta x, \Delta u)$；
- $\mathrm{Dom}(g)$：**线性化动力学** + 二次代价 + 近端项；
- $Pv \in \mathcal{C}$：**仅不等式**（分阶段）。

**ADMM 一轮**（Algorithm 2）：

1. **(13a)/(15)**：解带 $\rho$, $\sigma$ 正则的 **extended LQR** → backward Riccati + forward rollout（$\gamma$ 进入 $k_k$ 修正）；
2. **(13b–c)**：over-relaxation $z, v$；
3. **(13d)**：$z \leftarrow \Pi_{\mathcal{C}}(z + \rho^{-1} y)$ —— 逐段 $\max(c, \cdot)$；
4. **(13e)**：对偶 $y$ 更新。

**参数**：$\sigma = 10^{-6}$, $\alpha_{\mathrm{or}} = 1.6$, $\rho$ 每 25 iter 更新 → backward **重分解少**，forward 仅 matvec → **MPC 友好**。

**Warm-start**：从动力学可行的 Riccati 解启动 → ADMM 迭代次数减少。

**TRO §VI-D**：OSQP_OCP 对大规模问题 wall-clock 近线性于 $T,n_x$；HPIPM_OCP 在小规模上略优。

### 6.4 外层 SQP

**Filter line search**（式 18a–c）：接受 $(x^{[n+1]}, u^{[n+1]})$ 若相对 **历史迭代池** 在 $(\ell, \|\gamma\|, \|c\|)$ 偏序下 **支配** 某先前迭代 —— **不需** merit 权重 $\mu_\gamma, \mu_c$。

**终止**（式 20）：$\|\nabla_x \mathcal{L}\|_\infty$, $\|\nabla_u \mathcal{L}\|_\infty$, $\|\gamma\|_\infty$, $\|c\|_\infty \le \epsilon_{\mathrm{SQP}}$（benchmark 用 $10^{-4}$）。

**不用 RTI**：Jordana **每 MPC 周期让 SQP 收敛** 比单步 RTI 质量更好（与 Diehl RTI 传统对照）。

**Remark 5**（Nganga 2023）：humanoid 上精确 Hessian 可能值得计算，与 Chakravorty 关于简化 Hessian 的结论需按规模与是否近最优分别讨论。

### 6.5 约束 NMPC 实验

**Solo12 摩擦锥**（$\|F_T\|_2 \le \mu F_N$, $\mu=0.8$）：

- 250 nodes, 20 ms；CoM 圆轨迹 13 cm 直径, 0.2 rad/s；
- 无约束：34 SQP iter；有约束：31 iter；
- 无约束解 **穿锥**；约束解 **贴边界**（Fig. 4–5）。

**KUKA 约束 MPC**（100 Hz, 10 nodes, 50 ms, 4 SQP, 50 QP iter）：

| 实验 | 约束 | 现象 |
|------|------|------|
| 关节 $q_1$ | $[-0.05, 0.05]$ rad | 无约束 err 3.3 cm vs 约束 8.5 cm；约束 **严格满足** |
| EE 半空间 | $y>0$ 等 | horizon **提前避让** 不可行区 |
| EE 四半平面 | 方形可行域 | 提速 reference → 角点变 **圆滑**（预测性） |
| EE 等式 $Y=0$ + 扰动 | 直线约束 | 抗扰 **不违约**（Fig. 8） |
| 水平面约束 + 人手扰动 | 平面等式 | 视频：自动重配置 |

力矩控制机械臂上非线性硬约束闭环 NMPC；约束经 KKT/QP 处理（Jordana TRO §VI）。

---

## §7 附录数学：KKT 块三对角 = Riccati

Supplementary 考虑 **无不等式** QP，合并 $(x_k, u_{k-1}, \lambda_k)$ 为 $s_k$。

### 7.1 块三对角系统（Prop. 1）

```math
\begin{bmatrix}
\Gamma_1 & M_1^\top & & \\
M_1 & \Gamma_2 & M_2^\top & \\
& \ddots & \ddots & \ddots \\
& & M_{T-1} & \Gamma_T
\end{bmatrix}
\begin{bmatrix} s_1 \\ s_2 \\ \vdots \\ s_T \end{bmatrix}
=
\begin{bmatrix} g_1 \\ g_2 \\ \vdots \\ g_T \end{bmatrix}.
```

块定义（式 8）：

```math
\Gamma_k = \begin{bmatrix} R_{k-1} & 0 & -B_{k-1}^\top \\ 0 & Q_k & I \\ -B_{k-1} & I & 0 \end{bmatrix},\quad
M_k = \begin{bmatrix} 0 & S_k^\top & 0 \\ 0 & 0 & 0 \\ 0 & -A_k & 0 \end{bmatrix}.
```

**物理含义**：

- $\Gamma_k$ 编码 **$k$ 段** 代价 Hessian + **动力学等式** $x_k = A_{k-1} x_{k-1} + B_{k-1} u_{k-1} + \gamma_k$ 的 KKT；
- $M_k$ 耦合 **相邻时段** → 时域链式结构。

### 7.2 Thomas 算法（Algorithm 1）

**Backward**（$k = T-1 \ldots 1$）：

```math
\bar\Gamma_k = \Gamma_k - M_k^\top \bar\Gamma_{k+1}^{-1} M_k,\quad
\bar g_k = \bar\Gamma_k^{-1}(g_k - M_k^\top \bar g_{k+1}).
```

**Forward**：

```math
s_1 = \bar g_1,\quad s_{k+1} = \bar g_{k+1} - \bar\Gamma_{k+1}^{-1} M_k s_k.
```

### 7.3 归纳证明 $\bar\Gamma_k$ 的结构（Prop. 2 核心）

归纳假设：

```math
\bar\Gamma_k = \begin{bmatrix} R_{k-1} & 0 & -B_{k-1}^\top \\ 0 & V_k & I \\ -B_{k-1} & I & 0 \end{bmatrix},\quad
\bar g_k = \begin{bmatrix} -r_{k-1} \\ -v_k \\ \gamma_k \end{bmatrix}.
```

则 $V_k, v_k$ 满足 **离散 Riccati**（式 12–13 supplementary），$K_k, k_k$ 为仿射控制律 $\Delta u = K \Delta x + k$ 的增益。

**Lemma 0.1**：Schur 补公式给出 $\Gamma_k^{-1}$ 的 **$H_k^{-1}$ 修正** —— 实现上避免显式求 $3\times 3$ 块逆，只做 $m\times m$ 的 $H_k = R_k + B_k^\top V_{k+1} B_k$。

### 7.4 乘子恢复

```math
\lambda_k = V_k \Delta x_k + v_k.
```

Jordana 约束 case：$\mu_k = y_k$（ADMM 对偶）—— **同一 Riccati 骨架** 上挂不等式分裂。

**证明要点**：Riccati 为 KKT 块三对角经 Thomas 消元的形式；动态规划与 KKT 两种推导路径，对象相同。

---

## §8 实验与数字：两套 benchmark 的对照

### 8.1 Chakravorty：低维 swing-up（理论验证）

| 指标 | iLQR | DDP |
|------|------|-----|
| $Q_{uu}$ 全程 | pendulum/cart-pole **正定** | **出现负值** |
| $\alpha$ 行为 | 稳定（常 1 或 0.1） | **崩溃至 $\approx 0$** |
| 预测 $J+\Delta J$ | $\ge J_{\min}$ | 可 **$< 0$** 或 $\gg J$ |
| 近最优迭代 | 多 | 少 |
| 远初始猜 | 稳定 | 文内无正则化 |

### 8.2 Jordana benchmark

**Benchmark 问题**：Kuka ($n_x=14$), Quadrotor (13), Double pendulum (4), Humanoid Taichi ($n_x=77, n_u=32$)；100 随机初值/目标；KKT tol $10^{-4}$。

**Fig. 1 结论**：

- Multiple shooting **优于** single（Quadrotor, Pendulum）；
- Filter LS **优于** Crocoddyl 默认 LS（FDDP）；
- **SQP > FDDP(filter) > … > DDP** 在 **相同 iter 预算下 solved 比例**；
- Humanoid：**~80% @ 50 iter (SQP)** vs **~80% @ 200 iter (FDDP filter)**。

**MPC KUKA 画圆（500 Hz, 5 SQP max）**：

- 跟踪性能相近；
- SQP **3 iter** 达 KKT tol；FDDP **5 iter 触顶未收敛**（Fig. 3）。

**QP 求解器（Table III, TRO）**：OSQP_OCP / HPIPM_OCP 在 Solo/Taichi 上 **毫秒级** 收敛；通用 OSQP **慢一个数量级以上**。

### 8.3 两套实验的互补

| | Chakravorty | Jordana |
|---|-------------|---------|
| 目的 | 解释 iLQR 稳健性的机制 | 给出 SQP 相对 FDDP 的工程 benchmark 结果 |
| 维度 | 2–4 状态 | 最高 77 状态 |
| 约束 | 无 | 无 + **硬约束真机** |
| 变量 | 无正则 DDP vs iLQR | 统一 filter + KKT tol |
| 结论 | 简化 Hessian 在远离最优时更稳健 | SQP + 结构 QP 在相同迭代预算下 solve rate 更高 |

---

## §9 与 Bertsekas、零阶教程

### 9.1 Bertsekas

| Bertsekas | 本组 SQP 文献 |
|-----------|---------------|
| Ch.1 Exact DP | 离散 Bellman；DDP 从此出发但 **非严格** |
| Ch.2 值空间近似 + **MPC** | 同一滚动 OCP；Bertsekas 强调 **终端 cost 近似 $V$**；本组强调 **每步 NLP 解算** |
| Ch.3–5 参数化/RL | 模型未知或过大；本组针对模型已知 MPC |
| Rollout/Newton | 与 SQP 步概念对应（Chakravorty §III） |

### 9.2 Zero-Order 教程

| 零阶 | 本组 SQP |
|------|----------|
| $\min_x f(x)$，仅 $f$ 值 | $\min$ 用 $\nabla \ell$, $\nabla f$, $\nabla^2 \ell$, … |
| MPPI/CMA：$x$ = 控制序列 | $x,u$ 轨迹 + 乘子 |
| 非光滑/contact **自然** | 需光滑模型或平滑化 |
| 约束：**开放问题**（§VII） | **原生 KKT** |
| GPU **$K$ 并行 rollout** | GPU 可加速 Riccati/ADMM，但主线是 **结构消元** |

**方法谱系**（同一 OCP）：

```text
  零阶随机搜索 ──→ 一阶 IPM/梯度 ──→ GN-SQP/iLQR ──→ Newton-SQP/DDP
       ↑                    ↑                    ↑
   黑箱/非光滑          大规模 convex-ish      模型已知时 MPC 常用区间
```

Jordana 同时参与零阶综述与 SQP 工作：二者均属 $\min f$ 框架，差别在于是否利用模型导数。

### 9.3 三库对照

| 问题 | Bertsekas | SQP 组 | Zero-order |
|------|-----------|--------|------------|
| 模型 | 已知/未知均可 | **已知** $f,\ell,c$ | 黑箱 $f$ |
| 时域 | 有限+无限 | **有限** MPC | 有限 TO / 无限 RL |
| 约束 | 理论全面 | **硬约束 NMPC** | 弱 |
| 核心数学 | Bellman | **KKT + QP + Riccati** | RS / 有限差分 |
| 机器人默认 | RL + 近似 | **实时 NMPC** | MPPI / 仿真 batch |

---

## §10 选型、实现、开放问题

### 10.1 决策树（实践）

```text
需要硬约束（摩擦锥、关节、EE）？
  ├─ 是 → Jordana 型 multiple-shooting SQP + OSQP_OCP/HPIPM_OCP
  └─ 否 → 初猜远离最优？
        ├─ 是 → iLQR / GNMS / SQP(GN) + filter LS
        └─ 否 → 要极致 iter 数？
              ├─ 是 → DDP/Newton + 正则化 + 监控 Quu
              └─ 否 → SQP(GN) 通常足够
不可行动力学初猜？
  ├─ 是 → multiple shooting（必须）
  └─ 否 → single 也可
模型不可微 / 黑箱？
  └─ zero-order（MPPI/CMA）；或平滑化后再 SQP
```

### 10.2 实现资源

- [`mim_solvers`](https://github.com/machines-in-motion/mim_solvers) — SQP + OSQP_OCP  
- [`StagewiseSQP`](https://github.com/machines-in-motion/StagewiseSQP) — 复现 benchmark  
- **Crocoddyl** + **Pinocchio** — 动力学与导数  

### 10.3 开放问题（四篇汇总）

1. **FDDP / 非线性 rollout + multiple shooting** 的全局收敛（Jordana §VII）。
2. **Riccati 反馈增益** 在约束 MPC 闭环中的稳定性（增益已算，真机未用）。
3. **Exact vs GN Hessian** 的计算–收敛权衡（Remark 5, Nganga 2023）。
4. **RTI vs full SQP** 每周期质量–频率权衡。
5. **iLQR ↔ DDP 切换**（Chakravorty Hybrid 实验的扩展方向）。
6. **Infeasible inner QP**（ADMM 分裂）与安全 MPC 的形式保证。

### 10.4 阅读顺序

1. §0 → Chakravorty §II–IV → Supplementary Prop. 1–2 → Jordana TRO §II–VI  
2. Bertsekas Ch.2 MPC、zero-order §III–IV（背景）

---

## 附录 A — 公式速查

**Modified SQP 下降**（$c_{xx} \succ 0$, $\bar h=0$）：$\delta x^\top c_x = -\delta x^\top c_{xx} \delta x < 0$。

**iLQR 预测下降**：$\Delta J \approx -\alpha \sum_t k_t^\top Q_{uu,t}^{-1} k_t$。

**Riccati（无约束, defect $\gamma$）**：

```math
H_k = R_k + B_k^\top V_{k+1} B_k,\quad K_k = -H_k^{-1}(S_k^\top + B_k^\top V_{k+1} A_k),
```

```math
k_k = -H_k^{-1}(r_k + B_k^\top(v_{k+1} + V_{k+1}\gamma_{k+1})),\quad
V_k = Q_k + A_k^\top V_{k+1} A_k - K_k^\top H_k K_k.
```

**SQP 更新**：$x^{[n+1]} = x^{[n]} + \alpha \Delta x$，$\lambda^{[n+1]}, \mu^{[n+1]}$ 来自 QP KKT。

**GN 近似**：$Q_k,S_k,R_k$ 仅含 $\nabla^2 \ell$，不含 $\lambda,\mu$ 与约束曲率。

---

## 附录 B — 参考文献

- Abhijeet & S. Chakravorty (2025). arXiv:2510.03475.  
- A. Jordana et al. (2023). HAL stagewise SQP.  
- A. Jordana et al. (2025). *IEEE TRO*, 41.  
- A. Jordana et al. (2025). Zero-order robotics tutorial. arXiv:2506.22087.  
- D. P. Bertsekas (2019). *RL and Optimal Control*.  
- H. G. Bock & K.-J. Plitt (1984). Multiple shooting.  
- M. Giftthaler et al. (2018). GNMS. IROS.  
- J. Nocedal & S. J. Wright (1999). *Numerical Optimization*.  
- J. F. A. D. O. Pantoja (1988). DDP and Newton. *Int. J. Control*.  
- R. Wang et al. (2025). iLQR convergence. *J. Dyn. Sys. Meas. Control*.

---

*笔记修订：2026-07-25。*
