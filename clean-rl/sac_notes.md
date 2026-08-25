# SAC（Soft Actor-Critic）算法笔记

---

## 一、算法定位

SAC 是 Haarnoja 等人于 2018 年提出的 off-policy Actor-Critic，在**最大熵强化学习**框架下训练随机策略。它与 DDPG / TD3 同属「经验回放 + 目标网络」一系，但用策略熵代替往确定性动作上加噪声。

| 特性 | PPO | SAC |
|------|-----|-----|
| 类型 | On-policy 策略梯度 | Off-policy Actor-Critic |
| 目标 | 最大化期望回报 | 最大化回报 **+** 策略熵 |
| 学习方式 | 当前策略收集，用完即弃 | 经验池随机采样，可反复用 |
| 核心技巧 | Clipping + GAE | Soft Bellman + 双 Q + 重参数化 |
| 网络结构 | Actor $\pi(a\mid s)$ + Critic $V(s)$ | Actor $\pi(a\mid s)$ + 两个 soft $Q(s,a)$ |

与 DDPG / TD3 的差别：后两者输出**一个**动作，探索靠外加噪声；SAC 的策略本身是分布，探索写进目标函数。

---

## 二、前置基础：最大熵强化学习

标准 RL 最大化折扣回报：

$$
J(\pi) = \mathbb{E}_{\tau \sim \pi}\left[\sum_{t=0}^{\infty} \gamma^t r(s_t, a_t)\right]
$$

最大熵 RL 在每一步额外奖励策略的熵：

$$
J(\pi) = \mathbb{E}_{\tau \sim \pi}\left[\sum_{t=0}^{\infty} \gamma^t \Big(r(s_t, a_t) + \alpha\,\mathcal{H}\big(\pi(\cdot\mid s_t)\big)\Big)\right]
$$

其中温度 $\alpha>0$，熵定义为

$$
\mathcal{H}\big(\pi(\cdot\mid s)\big) = -\mathbb{E}_{a\sim\pi(\cdot\mid s)}\big[\log\pi(a\mid s)\big]
$$

**直观理解**：

- $\alpha$ 大：更愿意保持随机，探索多、不易锁死在单一动作
- $\alpha$ 小：更接近普通「只追回报」的 RL
- 同一状态可以保留**多峰**最优（确定性策略只能输出一个点）

### 最大熵目标下的软价值：逐步推导

**Step 1：按第一步动作拆开目标**

把轨迹期望写成「当前一步 + 折扣后的续值」：

$$
J(\pi) = \mathbb{E}_{s_0,a_0}\Big[r(s_0,a_0) + \alpha\mathcal{H}(\pi(\cdot\mid s_0)) + \gamma\, J_{s_1}(\pi)\Big]
$$

对任意状态 $s$ 定义 soft 状态价值 $V^\pi(s)$ 为从该状态起的最大熵目标。则 soft 动作价值满足备份：

$$
Q^\pi(s,a) = r(s,a) + \gamma\,\mathbb{E}_{s'\sim p(\cdot\mid s,a)}\big[V^\pi(s')\big]
$$

**Step 2：用 $Q$ 表达 $V$**

从 $s$ 出发，先按 $\pi$ 选 $a$，回报是 $Q^\pi(s,a)$，同时还要加上当前步的熵奖励 $\alpha\mathcal{H}$。熵奖励等价于在期望里减去 $\alpha\log\pi$：

$$
\begin{aligned}
V^\pi(s)
&= \mathbb{E}_{a\sim\pi(\cdot\mid s)}\big[Q^\pi(s,a)\big] + \alpha\,\mathcal{H}\big(\pi(\cdot\mid s)\big) \\
&= \mathbb{E}_{a\sim\pi(\cdot\mid s)}\big[Q^\pi(s,a) - \alpha\log\pi(a\mid s)\big]
\end{aligned}
$$

这就是 soft $V$ 的定义：普通期望 $Q$ 再加一项熵。

**Step 3：给定 $Q$，最优策略是 Boltzmann 分布**

固定 $Q$，在每个 $s$ 上选 $\pi(\cdot\mid s)$ 使 $V(s)$ 最大，且 $\pi$ 是概率分布：

$$
\max_\pi\ \mathbb{E}_{a\sim\pi}\big[Q(s,a) - \alpha\log\pi(a\mid s)\big]
\quad\text{s.t.}\quad \int \pi(a\mid s)\,da = 1
$$

拉格朗日函数（省略与 $a$ 无关的项）对 $\pi(a)$ 求变分，驻点满足

$$
Q(s,a) - \alpha\log\pi(a\mid s) - \alpha - \lambda(s) = 0
$$

解出

$$
\pi^*(a\mid s) = \frac{\exp\big(Q(s,a)/\alpha\big)}{Z(s)},\qquad
Z(s) = \int \exp\big(Q(s,a)/\alpha\big)\,da
$$

即能量函数为 $-Q/\alpha$ 的 Boltzmann 分布。$\alpha$ 越大，分布越平；$\alpha\to 0$ 时质量集中到 $\arg\max_a Q$。

等价看法：$\pi^*$ 是把 $\pi$ 投影到 $\exp(Q/\alpha)$ 上的信息投影

$$
\pi^*(\cdot\mid s) = \arg\min_\pi\ \mathrm{KL}\left(\pi(\cdot\mid s)\,\Big\|\,\frac{\exp(Q(s,\cdot)/\alpha)}{Z(s)}\right)
$$

**Step 4：把 $\pi^*$ 代回，得到 $V$ 的闭式**

由 Step 3，$\log\pi^*(a\mid s) = Q(s,a)/\alpha - \log Z(s)$，故

$$
Q - \alpha\log\pi^* = Q - \alpha\Big(\frac{Q}{\alpha} - \log Z\Big) = \alpha\log Z(s)
$$

与 $a$ 无关，对 $a$ 取期望不变：

$$
\boxed{V^*(s) = \alpha\log Z(s) = \alpha\log\int\exp\big(Q^*(s,a)/\alpha\big)\,da}
$$

连续动作下这个积分一般算不出，因此实践中的 SAC **不直接拟合 $V$**，而用从 $\pi$ 采样的 $Q-\alpha\log\pi$ 来近似 $V$。

**Step 5：消去 $V$，得到 soft Bellman 方程**

把 Step 2 的 $V(s')$ 代入 Step 1 的 $Q$：

$$
\boxed{Q^\pi(s,a) = r(s,a) + \gamma\,\mathbb{E}_{s',\,a'\sim\pi(\cdot\mid s')}\big[Q^\pi(s',a') - \alpha\log\pi(a'\mid s')\big]}
$$

这就是 Critic 要拟合的方程：相对普通 Bellman，多了一项 $-\alpha\log\pi(a'\mid s')$。

**推导链总结**：

```text
J(π) = E[Σ γ^t (r_t + α H(π(·|s_t)))]
   ↓ 拆成当前奖励 + 续值
Q(s,a) = r + γ E[V(s')]
   ↓ V = E_a[Q] + αH(π)
V(s) = E_a[Q(s,a) − α log π(a|s)]
   ↓ 对 π 做有约束变分
π*(a|s) ∝ exp(Q(s,a)/α)
   ↓ 代回 V
V*(s) = α log ∫ exp(Q/α) da
   ↓ 消去 V
Q = r + γ E[Q' − α log π']
```

---

## 三、痛点：为什么 DDPG 不够好

1. **探索与学习绑死**：确定性 Actor 必须外加噪声；噪声与 $Q$ 的梯度无关，难随状态调节探索强度
2. **$Q$ 过估计**：用 $\max_a Q(s',a)$ 或 $Q(s',\mu(s'))$ 做目标，误差向上偏，策略跟着学坏
3. **对超参数敏感**：学习率、噪声尺度稍变就发散；off-policy 又把错误 $Q$ 反复写入经验池
4. **多峰最优**：最优动作不唯一时，确定性策略只能贴其中一个峰

---

## 四、TD3 的思路：先把 $Q$ 估稳

TD3 在 DDPG 上做了三件事：两个 $Q$ 取 $\min$ 抑制过估计、延迟更新 Actor、目标动作加平滑噪声。

**问题**：探索仍然是外加噪声，与「在该状态下应有多随机」没有关系。

SAC 保留双 $Q$ 与目标网络，把探索改成最大熵，并让 $\alpha$ 可学习。

---

## 五、SAC 的核心：用 soft Bellman 学 $Q$

### 5.1 从方程到可采样的 TD 目标

第二节得到

$$
Q(s,a) = r + \gamma\,\mathbb{E}_{s',a'}\big[Q(s',a') - \alpha\log\pi(a'\mid s')\big]
$$

用目标网络 $Q_{\bar\theta}$ 提供下一状态的价值，并用**当前策略**采样 $a'\sim\pi_\phi(\cdot\mid s')$（不是行为策略，也不是 $\arg\max$）。再取两个目标 $Q$ 的较小者，减轻过估计。终止时不再备份。于是一步目标为

$$
y = r + \gamma(1-d)\Big(\min_{i=1,2} Q_{\bar\theta_i}(s',a') - \alpha\log\pi_\phi(a'\mid s')\Big)
$$

其中 $d=1$ 表示终止。

- $\min_i Q$：两个估计里取保守的那个
- $-\alpha\log\pi$：策略在 $s'$ 上越确定（$\log\pi$ 越大），目标越低，备份本身就在奖励随机性

### 5.2 Critic 损失

两个 soft $Q$ 拟合同一目标：

$$
J_Q(\theta_i) = \mathbb{E}_{(s,a,r,s')\sim\mathcal{D}}\Big[\big(Q_{\theta_i}(s,a) - y\big)^2\Big],\qquad i=1,2
$$

这是 soft Bellman 残差的均方。目标网络缓慢跟踪：

$$
\bar\theta \leftarrow \tau\theta + (1-\tau)\bar\theta
$$

$\tau$ 很小（如 $0.005$），避免 $y$ 随 $Q$ 剧烈跳动。

> 与 PPO 的对照：PPO 用 GAE 把多步残差合成 $\hat A_t$ 再更新 $V$；SAC 是 off-policy **一步** soft TD，不需要整段轨迹倒序。

---

## 六、策略更新：从 KL 投影到重参数化

### 6.1 策略应靠近 Boltzmann

给定当前 $Q_\theta$，理想策略是 $\exp(Q_\theta/\alpha)/Z$。Actor $\pi_\phi$ 做的是把当前分布拉向这个目标，即最小化正向 KL（与第二节 Step 3 同一信息投影）：

$$
J_\pi(\phi)
= \mathbb{E}_{s\sim\mathcal{D}}\left[
\mathrm{KL}\left(\pi_\phi(\cdot\mid s)\,\Big\|\,\frac{\exp(Q_\theta(s,\cdot)/\alpha)}{Z_\theta(s)}\right)
\right]
$$

### 6.2 展开 KL，丢掉与 $\phi$ 无关的配分函数

$$
\begin{aligned}
\mathrm{KL}
&= \mathbb{E}_{a\sim\pi_\phi}\Big[\log\pi_\phi(a\mid s) - \tfrac{1}{\alpha}Q_\theta(s,a) + \log Z_\theta(s)\Big]
\end{aligned}
$$

$Z_\theta(s)$ 不依赖 $\phi$（把 $Q$ 视为常数），最小化 KL 等价于

$$
\boxed{J_\pi(\phi) = \mathbb{E}_{s\sim\mathcal{D},\,a\sim\pi_\phi}\big[\alpha\log\pi_\phi(a\mid s) - Q_\theta(s,a)\big]}
$$

实践中 $Q$ 换成 $\min_{i=1,2}Q_{\theta_i}$，与 Critic 的保守估计一致。

- $Q$ 大 → 损失下降 → 提高选该动作的倾向
- $\log\pi$ 大（太确定）→ 损失上升 → 被 $\alpha$ 压回去

### 6.3 两种求梯度的方式，以及为何用重参数化

目标含 $\mathbb{E}_{a\sim\pi_\phi}[Q(s,a)]$。对数导数技巧给出

$$
\nabla_\phi\,\mathbb{E}_{a\sim\pi_\phi}[Q(s,a)]
= \mathbb{E}_{a\sim\pi_\phi}\big[Q(s,a)\,\nabla_\phi\log\pi_\phi(a\mid s)\big]
$$

这是 REINFORCE 型估计，方差大，而且 $a$ 若由不可微的 `sample` 得到，$Q(s,a)$ 对 $\phi$ 没有通过 $a$ 的路径。

重参数化把随机性从参数里抽出来：$\varepsilon\sim\mathcal{N}(0,I)$ 与 $\phi$ 无关，

$$
u = \mu_\phi(s) + \sigma_\phi(s)\odot\varepsilon,\qquad a = f(u)
$$

于是 $a=f_\phi(\varepsilon;s)$ 对 $\phi$ 可导，

$$
\nabla_\phi\,\mathbb{E}_{\varepsilon}[Q(s,f_\phi(\varepsilon;s))]
= \mathbb{E}_{\varepsilon}\big[\nabla_a Q(s,a)\cdot\nabla_\phi f_\phi(\varepsilon;s)\big]
$$

梯度顺着 $Q$ 对动作的敏感度流回 Actor，方差通常远小于 score function。

### 6.4 $\tanh$ 有界动作：密度如何变

环境动作常在有界区间。先在无界空间用高斯，再逐维压缩：

$$
a = \tanh(u),\qquad u\sim\mathcal{N}(\mu_\phi(s),\sigma_\phi^2(s))
$$

换元：$a_i=\tanh(u_i)$，雅可比对角且

$$
\frac{\partial a_i}{\partial u_i} = 1-\tanh^2(u_i) = 1-a_i^2
$$

密度与对数密度（$u=\mathrm{artanh}(a)$）：

$$
\pi(a\mid s) = \mathcal{N}(u;\mu,\sigma)\,\left\lvert\det\frac{\partial a}{\partial u}\right\rvert^{-1}
$$

$$
\boxed{\log\pi(a\mid s) = \log\mathcal{N}(u;\mu,\sigma) - \sum_i\log(1-\tanh^2(u_i))}
$$

若再仿射到环境盒 $a=\mathrm{scale}\odot\tanh(u)+\mathrm{bias}$，则多一项 $\log\lvert\mathrm{scale}_i\rvert$：

$$
\log\pi(a\mid s) = \log\mathcal{N}(u;\mu,\sigma) - \sum_i\log\big(\lvert\mathrm{scale}_i\rvert\,(1-\tanh^2(u_i))\big)
$$

漏掉雅可比项，熵 $\mathcal{H}(\pi)$ 会对错的密度计算，$\alpha$ 的尺度也会漂。实现上常把 $\log\sigma$ 限制在有限区间，避免 $\sigma\to 0$ 或过大。

### 6.5 延迟更新

$Q$ 若还很噪，Actor 会追逐虚假峰值。与 TD3 类似：Critic 更新更勤，Actor 隔若干步再更新，让 $Q$ 先跟上。

---

## 七、自动调节 $\alpha$：约束优化的对偶

固定 $\alpha$ 往往全局过探索或过贪心。更合理的提法是：**最大化回报，同时要求平均熵不低于目标 $\bar{\mathcal{H}}$**：

$$
\max_\pi\ \mathbb{E}\left[\sum_t\gamma^t r_t\right]
\quad\text{s.t.}\quad
\mathbb{E}_{s\sim\rho_\pi}\big[\mathcal{H}(\pi(\cdot\mid s))\big]\ge\bar{\mathcal{H}}
$$

拉格朗日函数为

$$
\mathbb{E}\left[\sum_t\gamma^t r_t\right]
+ \alpha\Big(\mathbb{E}[\mathcal{H}(\pi)] - \bar{\mathcal{H}}\Big)
$$

$\alpha\ge 0$ 是对偶变量。这正是最大熵目标：$\alpha$ 乘在熵奖励上。对偶问题对 $\alpha$ 最小化

$$
\min_{\alpha\ge 0}\ \alpha\left(\mathbb{E}_{a\sim\pi}\big[-\log\pi(a\mid s)\big] - \bar{\mathcal{H}}\right)
$$

即（实现里常用的等价形式）

$$
\boxed{J(\alpha) = \mathbb{E}_{a\sim\pi}\big[-\alpha\big(\log\pi(a\mid s)+\bar{\mathcal{H}}\big)\big]}
$$

为保持 $\alpha>0$，对 $\log\alpha$ 做梯度下降。

- 策略太确定：$\log\pi$ 偏大，$\log\pi+\bar{\mathcal{H}}>0$，最小化 $J$ 会**增大** $\alpha$，加强探索
- 熵已经够高：$\log\pi+\bar{\mathcal{H}}<0$，**减小** $\alpha$，更贪心

连续动作的常用启发式：$\bar{\mathcal{H}}=-\dim(\mathcal{A})$（与「各维独立、量级相当于标准差为 $1$ 的高斯」同阶）。离散动作常用均匀分布的熵再乘一个小于 $1$ 的系数，避免目标过高。

---

## 八、Actor-Critic 架构

```text
              观测 s
                 │
        ┌────────┴────────┐
        ▼                 ▼
   ┌─────────┐      拼接 (s, a)
   │  Actor  │            │
   │ μ, σ    │     ┌──────┴──────┐
   │ π(a|s)  │     ▼             ▼
   └─────────┘  ┌─────┐       ┌─────┐
        │       │ Q_θ1│       │ Q_θ2│
        │       └─────┘       └─────┘
        │            └──── min ────┘
        │                    │
        └── 最小化 α log π − min Q
```

- **Actor**：随机策略；训练用重参数化采样，评估可用 $\tanh(\mu(s))$ 的确定性动作
- **双 Critic**：输入 $(s,a)$，输出标量 $Q$；另有缓慢跟踪的目标网络
- 与 PPO 不同：不显式计算优势；$Q(s,a)-\alpha\log\pi$ 已是 soft 优势

---

## 九、完整训练流程

```text
1. 初始化 Actor、两套 Q 与目标 Q、经验池；可选地初始化 log α
2. 与环境交互：动作由 π 采样（初期可用均匀随机填池）
   将 (s, a, r, s', done) 写入经验池
3. 从经验池采样一个 batch：
     a. a' ~ π(s')（重参数化）
        y = r + γ(1−d)(min Q̄(s',a') − α log π(a'|s'))
     b. 对两个 Q 做 MSE，更新 θ1, θ2
     c. （按间隔）最小化 α log π(a|s) − min Q(s,a)，更新 Actor
     d. （若学习温度）用 J(α) 更新 α
     e. 软更新目标网络 Q̄
4. 回到步骤 2，数据保留在池中反复使用
```

---

## 十、总损失函数

三者分开优化，没有 PPO 那种「一个标量里加权 clip + 熵 + $V$」：

$$
\begin{aligned}
L_Q &= \sum_{i=1}^{2}\mathbb{E}\big[(Q_{\theta_i}(s,a)-y)^2\big] \\
L_\pi &= \mathbb{E}\big[\alpha\log\pi_\phi(a\mid s)-\min_i Q_{\theta_i}(s,a)\big] \\
L_\alpha &= \mathbb{E}\big[-\alpha(\log\pi_\phi(a\mid s)+\bar{\mathcal{H}})\big]
\end{aligned}
$$

- **$L_Q$**：拟合 soft Bellman
- **$L_\pi$**：KL 投影到 $\exp(Q/\alpha)$，含熵
- **$L_\alpha$**：把平均熵推到目标 $\bar{\mathcal{H}}$

---

## 十一、SAC 成功的关键设计

| 设计 | 解决的问题 | 实现方式 |
|------|-----------|---------|
| **最大熵目标** | 探索不足、锁死单峰 | 备份与 Actor 中的 $-\alpha\log\pi$ |
| **重参数化 + $\tanh$ 校正** | 连续动作对 $Q$ 反传、有界动作密度 | $a=f_\phi(\varepsilon)$ + 雅可比 |
| **双 $Q$ 取 $\min$** | $Q$ 过估计 | $\min(Q_{\theta_1},Q_{\theta_2})$ |
| **目标网络软更新** | TD 目标剧烈跳动 | $\bar\theta\leftarrow\tau\theta+(1-\tau)\bar\theta$ |
| **延迟策略更新** | Actor 追噪声 $Q$ | Critic 更勤、Actor 较慢 |
| **自动 $\alpha$** | 固定温度不适配训练阶段 | 熵约束的对偶 |
| **经验回放** | on-policy 样本效率低 | 随机 batch 反复更新 |

---

## 十二、直观类比

把 SAC 想象成**在陌生城市找餐馆，同时保持愿意换一家试试**：

- **Actor**：你选餐馆的随机习惯（不是每次只去同一家）
- **双 Critic**：两个朋友分别打分，你取更保守的那个，避免被吹捧带偏
- **$\alpha$**：你有多在乎「别太死板」；吃得很熟了就自动减小 $\alpha$，更贪心
- **经验池**：以前走过的路都可以再复习，不必像 PPO 那样走一段就扔掉
- **$\tanh$ 重参数化**：先在无界草稿上采样，再压到合法菜单，改概率时把「压扁」算进密度

---

## 十三、关键超参数

| 参数 | 典型取值 | 作用 |
|------|----------|------|
| $\gamma$ | $0.99$ | 折扣因子 |
| $\tau$ | $0.005$ | 目标网络软更新系数 |
| 经验池容量 | $10^6$ | 回放存储 |
| batch 大小 | $256$ | 每次更新的转移数 |
| 开始学习前的随机步 | 数千 | 先填池再更新 |
| Actor 学习率 | $3\times 10^{-4}$ | 策略 |
| Critic 学习率 | $10^{-3}$ | $Q$ 与 $\alpha$ 常共用此量级 |
| 固定 $\alpha$（不自动调时） | $0.2$ | 熵温度 |
| 目标熵 $\bar{\mathcal{H}}$ | $-\dim(\mathcal{A})$ | 自动调 $\alpha$ 的约束 |

---

## 十四、连续动作与离散动作

连续动作无法对 $a$ 求和，用单次采样 $a'\sim\pi$ 近似 soft $V$ 中的期望，并必须重参数化。

离散动作 $|\mathcal{A}|$ 有限，$Q(s)\in\mathbb{R}^{\lvert\mathcal{A}\rvert}$ 一次给出所有动作的价值，期望可写成对动作的加权和（不再需要重参数化）：

$$
y = r + \gamma(1-d)\sum_{a'}\pi(a'\mid s')\Big(\min_i Q_{\bar\theta_i}(s,a') - \alpha\log\pi(a'\mid s')\Big)
$$

$$
J_\pi(\phi) = \mathbb{E}_{s\sim\mathcal{D}}\left[
\sum_a \pi_\phi(a\mid s)\big(\alpha\log\pi_\phi(a\mid s) - \min_i Q_{\theta_i}(s,a)\big)
\right]
$$

用 $\pi$ 加权相当于对期望的无偏、低方差估计；温度损失同样可对动作加权。目标熵改为均匀分布熵乘一个小于 $1$ 的系数，而不是 $-\dim(\mathcal{A})$。
