# FlashSAC 算法笔记

> Kim et al., *FlashSAC: Fast and Stable Off-Policy Reinforcement Learning for High-Dimensional Robot Control*（arXiv:2604.04539, 2026）。项目页：<https://holiday-robot.github.io/FlashSAC>。  
> 前置：[`sac_notes.md`](sac_notes.md)（soft Bellman、双 $Q$、自动温度）。本文不重复最大熵推导，只写相对 SAC 改变了什么。

---

## 一、算法定位

FlashSAC 以 SAC 为骨架，面向**高维机器人控制**（灵巧手、人形、视觉）的 off-policy Actor-Critic。PPO 在低维、仿真极便宜时稳定且够用；状态–动作维数升高后，on-policy 数据覆盖不足，反复丢弃旧轨迹在墙钟上不可接受。标准 off-policy 虽能复用经验池，但在宽分布上拟合 bootstrapped $Q$ 需要大量梯度步，误差经备份放大，训练慢且不稳。

FlashSAC 的主张：在固定算力下用**更大网络、更大 batch、更少更新**换墙钟；同时用结构约束压住权、特征与梯度范数，使放大容量不至于把 critic 带崩；再用与动作维成比例的熵目标与噪声重复补探索。

| 特性 | PPO | SAC | FlashSAC |
|------|-----|-----|----------|
| 类型 | On-policy 策略梯度 | Off-policy Actor-Critic | 同 SAC |
| 目标 | 期望回报 | 回报 + 熵 | 同 SAC |
| 数据 | 当前策略，用完即弃 | 经验池 | 更大池 + 大规模并行仿真 |
| 网络 | 中小 MLP 常见 | 小 MLP（约 0.2–0.5M） | 约 2.5M、六层 inverted residual |
| 更新密度 | 每批 on-policy 数据多 epoch | UTD 通常较高 | GPU 设定 UTD = 2/1024 |
| 稳定手段 | Clip + GAE | 双 Q、目标网络、α | 再加 BN / RMSNorm / 权归一 / 分布 Q |
| 典型场景 | 低维、高吞吐仿真 | 连续控制通解 | 高维操纵与人形 sim-to-real |

相对 FastTD3 / FastSAC：后者墙钟快但网络约 0.2M，渐近回报受限。FlashSAC 把「放大」与「压范数」绑在一起，使大网络可训。

---

## 二、痛点：高维下 PPO 与标准 SAC 各缺一块

**On-policy（PPO）。** 策略评估依赖当前 $\pi$ 的窄支撑。高维连续动作上重要性采样方差过大，难以靠 IS 复用旧数据。仿真变贵（接触、视觉、大策略前向）后，丢数据直接转化为墙钟。

**Off-policy critic。** 经验池上的 bootstrapped 损失（标准 TD，尚未写熵）为

$$
\mathcal{L}_{Q}=\mathbb{E}_{(s,a,r,s')\sim\mathcal{D}}\Big[\big(Q_{\theta}(s,a)-(r+\gamma Q_{\theta}(s',a'))\big)^{2}\Big],
\quad a'\sim\pi(\cdot\mid s')
$$

宽覆盖要求多次梯度更新才能拟合；$y$ 依赖自身预测，逼近误差与外推误差沿备份传递（deadly triad）。容量越大，该放大越明显。

**探索。** 仅最大熵往往不足以在高维动作空间形成时间上连贯的探索轨迹。

FlashSAC 分别对应：降 UTD 并放大数据/模型；约束 critic 更新动力学；统一熵目标与噪声重复。

---

## 三、保留的 SAC 骨架

MDP $\mathcal{M}=(\mathcal{S},\mathcal{A},P,r,\gamma)$，连续动作。经验池 $\mathcal{D}$ 存 $(s,a,r,s')$。双 critic $Q_{\phi_1},Q_{\phi_2}$，目标网上软更新

$$
\bar\phi_j\leftarrow\tau\phi_j+(1-\tau)\bar\phi_j,\qquad j\in\{1,2\}
$$

策略损失与 soft 目标（与 [`sac_notes.md`](sac_notes.md) 第五节一致）：

$$
\mathcal{L}_{\pi}(\theta)=\mathbb{E}_{s\sim\mathcal{D},\,a\sim\pi_{\theta}}\big(\alpha\log\pi_{\theta}(a\mid s)-\min_{i}Q_{\phi_i}(s,a)\big)
$$

$$
y=r+\gamma\big(\min_{j}Q_{\bar\phi_j}(s',a')-\alpha\log\pi_{\theta}(a'\mid s')\big),\qquad a'\sim\pi_{\theta}(\cdot\mid s')
$$

标量 critic 时 $\mathcal{L}_{Q}(\phi_i)=\mathbb{E}[(Q_{\phi_i}(s,a)-y)^{2}]$。FlashSAC 将 $Q$ 改为分布输出并用交叉熵拟合投影后的 Bellman 目标（第五节），$\min$ 与 $-\alpha\log\pi$ 的结构不变。

---

## 四、快：数据吞吐、容量与低 UTD

动机来自监督学习的缩放经验：固定算力下，大模型 + 大 batch + 少步往往优于小模型密更新。Off-policy 中直接放大易不稳，故必须与第五节同时使用。

**并行仿真。** 默认 1024 个环境并行收集，以覆盖高维状态–动作空间。经典 SAC/TD3 常用少量环境。

**大回放。** 池容量至 10M 量级（常见 off-policy 为 1M），减轻长尾转移被覆盖及外推。消融中 10M 利于稳定；50M 会稀释近期优质样本，墙钟变慢，渐近或略高。

**大模型、大 batch、少更新。** Actor 与 critic 各约 2.5M 参数、六层。Batch 2048（接近打满 GPU）。更新–数据比

$$
\mathrm{UTD}=\frac{2}{1024}
$$

即每新增 1024 条转移只做 2 次梯度步。配合更大学习率。PyTorch JIT 与混合精度再降约 5%–10% 墙钟。

CPU 单环境、样本为瓶颈时：batch 改为 512，UTD = 1，其余设计不变。

---

## 五、稳：约束权、特征与梯度范数

Bellman 备份把 $s'$ 上的误差送入当前目标。FlashSAC 用一组结构使参数、激活与梯度范数在训练中有界，并降低 critic 损失的条件数（消融中逐项加入后条件数单调下降）。

### 5.1 Inverted residual 与 RMSNorm

主干为堆叠 inverted residual（扩张维后投影回原维并加残差，结构类 Transformer FFN / MobileNet 瓶颈）。最后一块之后对每个样本做 RMSNorm，限制进入价值头的特征范数，避免 OOD 输入产生无界激活并破坏备份。

### 5.2 预激活 Batch Normalization

回放由演变中的行为策略混合而成，输入非平稳。非线性前使用 BN，减轻死 ReLU 与梯度退化。相对 LayerNorm，大 batch 上的 BN 统计来自多样本回放，经验上损失曲面更平滑、有效条件数更低。

### 5.3 Cross-batch 价值预测

BN 按 batch 估计均值方差。当前 $Q(s,a)$ 与目标 $Q(s',a')$ 若分两次前向，归一化统计不一致。将当前转移与下一状态拼成**同一 batch** 前向（CrossQ 做法），Bellman 两侧共享 BN 统计。

### 5.4 分布 critic 与奖励缩放

将 $Q$ 表示为 $[G_{\min},G_{\max}]$ 上均匀放置的 $n_{\mathrm{atom}}$ 个原子上的范畴分布，网络输出原子概率，对投影后的分布 Bellman 目标做交叉熵（C51 型）。相对标量 MSE，对噪声目标更不敏感。

支撑固定，故对奖励做尺度归一而非只中心化回报。跟踪折现回报的运行方差 $\sigma_{t,G}^{2}$ 与最大幅值 $G_{t,\max}$：

$$
\bar r_{t}=\frac{r_{t}}{\max\bigl(\sqrt{\sigma_{t,G}^{2}+\epsilon},\; G_{t,\max}/G_{\max}\bigr)}
$$

使有效回报落在分布支撑内且训练过程尺度一致。

### 5.5 权归一

无约束的权增长会抬高 $Q$ 方差并放大备份误差。每步梯度之后将各权向量投影到单位球面，归一化层的 $(\gamma,\beta)$ 投影到范数 $\sqrt{d}$。信息主要编码在方向而非尺度。单独使用权归一增益有限，但在样本受限时提高稳健性，故保留。

---

## 六、探索：统一熵目标与噪声重复

Off-policy 允许收集策略与优化策略分离。

### 6.1 与动作维成比例的目标熵

自动温度需指定 $\bar{\mathcal{H}}$。SAC 常用 $-\lvert\mathcal{A}\rvert$，跨本体仍常按任务微调。FlashSAC 指定对角高斯的目标标准差 σ_tgt（实验中一律 0.15）：

$$
\bar{\mathcal{H}}=\frac{1}{2}\lvert\mathcal{A}\rvert\log\bigl(2\pi e\,\sigma_{\mathrm{tgt}}^{2}\bigr)
$$

随 $\lvert\mathcal{A}\rvert$ 线性增长，不同自由度上探索强度同量级。消融中 $\sigma_{\mathrm{tgt}}\in\{0.05,\ldots,0.25\}$ 渐近接近，$\sigma_{\mathrm{tgt}}=0.15$ 附近即可。

### 6.2 Noise repetition

OU / pink noise 在数千并行环境上需为每个环境维护相关过程，内存与算力开销大。改为：每隔一段重复区间采样 $\varepsilon\sim\mathcal{N}(0,I)$，在动作选择中保持 $k$ 步不变。$k$ 服从 Zeta 分布 $P(k)\propto k^{-s}$，多数为短重复、偶发长相关段。相对逐步独立噪声，相关扰动不易被高维动力学平均掉。去掉该机制会减慢收敛并降低总分。

---

## 七、架构示意

```text
观测 s（或视觉编码器输出）
        │
  inverted residual × L     预激活 BN + 非线性
        │                   残差
     RMSNorm
        │
   ┌────┴────┐
   Actor     双分布 critic（原子 logits）
   μ, σ      与目标网；Cross-batch 与 s' 同 BN
        │
   熵损失 + min Q     投影 Bellman + 交叉熵
```

视觉任务：三层卷积 + 线性瓶颈；堆叠最近三帧（84×84×9）；n-step 取 3；动作重复 2。稳定模块与状态设定正交，可再叠表示学习目标。

---

## 八、完整训练流程（GPU 默认）

```text
1. 初始化 Actor、两套分布 critic 与目标网、容量至 10M 的池、log α
2. 1024 环境并行交互；动作为 π 采样，噪声 ε 按 Zeta 间隔重复
   将转移写入池
3. 每积累 1024 条新数据，做 2 次更新（UTD=2/1024），每次 batch=2048：
     a. 将 (s,a) 与 (s',a') 拼 batch，共享 BN
        奖励按 (6) 缩放；分布 Bellman 投影得目标
     b. 交叉熵更新两套 critic；权向量投影到单位球
     c. 最小化 α log π − min Q，更新 Actor
     d. J(α) 更新温度（目标熵为 (7)）
     e. 软更新目标网
4. 回到步骤 2
```

---

## 九、损失（相对 SAC 的改动）

$$
\begin{aligned}
L_{Q} &= \sum_{i=1}^{2}\mathbb{E}\big[\mathrm{CE}\big(p_{\phi_i}(s,a),\;\mathcal{P}y_{\mathrm{dist}}\big)\big] \\
L_{\pi} &= \mathbb{E}\big[\alpha\log\pi_{\theta}(a\mid s)-\min_i Q_{\phi_i}(s,a)\big] \\
L_{\alpha} &= \mathbb{E}\big[-\alpha\big(\log\pi_{\theta}(a\mid s)+\bar{\mathcal{H}}\big)\big]
\end{aligned}
$$

$p_{\phi_i}$ 为原子概率，$Q_{\phi_i}$ 为其期望；$\mathcal{P}$ 为向固定原子网格的投影。$L_{\pi}$、$L_{\alpha}$ 与 SAC 同型，$\bar{\mathcal{H}}$ 用 (7) 而非 $-\lvert\mathcal{A}\rvert$。

---

## 十、关键设计

| 设计 | 解决的问题 | 实现 |
|------|------------|------|
| 低 UTD + 大模型 + 大 batch | 密更新墙钟差、小网上界 | 2/1024、2.5M、2048 |
| 大并行 + 10M 池 | 高维覆盖、长尾遗忘 | 1024 环境 |
| Inverted residual + RMSNorm | 深层梯度、无界特征 | 瓶颈块后 RMSNorm |
| 预激活 BN + Cross-batch | 非平稳回放、备份两侧统计不一致 | 与下一状态同 batch |
| 分布 Q + 奖励缩放 | 标量 TD 对噪声敏感、支撑溢出 | C51 型，式 (6) |
| 权归一 | 权范数增长放大 Q 方差 | 步后投影 |
| 目标熵 σ_tgt | 跨本体调目标熵 | 式 (7)，默认 0.15 |
| 噪声重复 | 高维需时间相关探索、并行开销 | Zeta 间隔固定噪声 |

---

## 十一、实验要点（论文报告）

评测超过 60 个任务、10 个仿真器；墙钟在单卡 RTX 5090 上按「交互时间 + 算法更新时间」协议估计。

- **GPU 状态（IsaacLab、ManiSkill、Genesis、Playground 等）。** Off-policy 训 50M 步；PPO 训 200M 步（约 3 倍算力）以探渐近。FlashSAC **一套超参**，仅折扣因子 γ 跟仿真器默认（如 IsaacLab 0.99，Playground 0.97）。低维夹爪 / 四足与 PPO 接近或略优；高维灵巧手与人形在渐近回报与墙钟上明显优于 PPO。相对 FastTD3 更稳、渐近更高。
- **CPU 单环境（MuJoCo、DMC、HumanoidBench、MyoSuite）。** 强调样本效率；对比 XQC、SimbaV2、TD-MPC2、MR.Q 等。FlashSAC 仍用统一配置（batch / UTD 按第四节缩小）。PPO 在该制度下弱。
- **视觉 DMC。** 对比 DrQ-v2、MR.Q；1M 步。FlashSAC 墙钟与渐近不弱于基线，且无任务级探索或辅助动力学。
- **Sim-to-real（Unitree G1 盲行走）。** 地形课程与域随机；与 PPO 共用奖励、非对称 Actor–Critic、隐式系统辨识。平地约 20 分钟对 PPO 约 3 小时；真实楼梯（训练未见同尺寸）约 4 小时对 PPO 约 20 小时。

**覆盖分析。** Shadow Hand 上 1M 池内 off-policy 样本相对终策略再滚 1M 的 on-policy 样本，在物体位置–指尖动作上支撑明显更宽，用以解释高维下 on-policy 评估困难。

---

## 十二、直观对照

把标准 SAC 看成「小网络、勤更新、池 1M、熵目标按维数启发式」。FlashSAC 是同一套 soft Actor–Critic，但：

- **少改参数、多看数据：** 每次更新吃更大、更杂的 batch，而不是把同一批误差反传很多次；
- **先捆住 critic 再加宽加深：** 范数与 BN 统计一致，备份才不会随容量发散；
- **探索按本体自动对齐：** $\sigma_{\mathrm{tgt}}$ 固定，相关噪声用重复而非每环境一条 OU。

低维、仿真近乎免费时，PPO 仍合理；维数与仿真成本上去之后，瓶颈从「策略梯度方差」转为「宽数据上的 $Q$ 是否可稳、可快地拟合」。

---

## 十三、关键超参数

| 参数 | GPU 默认 | 作用 |
|------|----------|------|
| 并行环境数 | 1024 | 覆盖与吞吐 |
| 池容量 | 至 10M | 长尾与多样性 |
| Batch | 2048（CPU：512） | 大步、BN 统计 |
| UTD | 2/1024（CPU：1） | 墙钟 vs 拟合 |
| 网络 | 约 2.5M，六层 | 容量 |
| σ_tgt | 0.15 | 目标熵，式 (7) |
| γ | 跟仿真器 | 折扣 |
| τ | 与 SAC 同类软更新 | 目标网 |
| 分布支撑 | G_min, G_max, 原子数 | 范畴 Q |

---

## 十四、与仓库内笔记的关系

- 最大熵、soft $V/Q$、Boltzmann Actor、对偶 $\alpha$：[`sac_notes.md`](sac_notes.md)。
- Clip 与 GAE、on-policy 样本效率：[`ppo_notes.md`](ppo_notes.md)。
- FlashSAC 不改这些目标的定义，改的是 **谁在何时用多大模型、如何约束 critic、如何在高维上采探索噪声**。

局限与文中展望：当前重点在状态与中等视觉；触觉、演示混合、更慢但更真的仿真器是自然延伸。稳定模块与辅助表示损失可叠加。
