# 零阶最优控制

> 分节读书笔记的延伸。文献：Jordana et al., *An Introduction to Zero-Order Optimization Techniques for Robotics*（arXiv:2506.22087）。按章节整理见 [`zero-order-optimization-study-notes.md`](zero-order-optimization-study-notes.md)。

## 摘要

从随机预测采样出发，经高斯平滑、CEM、MPPI，再到朗之万动力学、SVGD、扩散退火与 CMA-ES，可以把零阶最优控制里的常用方法放在同一套玻尔兹曼分布语言下。MPPI 有两条等价读法：用指数权重估计最优控制的均值，以及沿平滑后密度的得分走一步。温度、协方差和噪声日程决定探索与集中；是否更新协方差、是否保留朗之万噪声，则把 CEM、MPPI、CMA 和扩散式 MPC 区分开来。

## 1. 基础：随机预测采样与高斯平滑

### 1.1 问题设定与挑战

有限时域最优控制（只优化开环序列，状态由动力学递推）：

```math
\min_{\mathbf{u}_{0:H-1}} J(\mathbf{u}_{0:H-1}) = \sum_{t=0}^{H-1} c(\mathbf{x}_t, \mathbf{u}_t) + c_f(\mathbf{x}_H)
```

```math
\mathbf{x}_{t+1} = f(\mathbf{x}_t, \mathbf{u}_t)
```

决策维数是 $H\times\dim(\mathbf{u})$。接触会使 $J$ 高度非凸，MPC 还要求在很短时间内给出第一步控制。

### 1.2 随机预测采样

1. 采样：$\mathbf{u}^{(i)}\sim q(\mathbf{u})$，$i=1,\dots,N$
2. 评估：$J^{(i)}=J(\mathbf{u}^{(i)})$
3. 选择：$\mathbf{u}^*=\arg\min_i J^{(i)}$

$q$ 通常取当前均值附近的高斯。维数升高后，要靠均匀铺点覆盖空间，样本需求大致随维数指数增长，因此后面都改为局部采样并迭代更新分布。

### 1.3 高斯平滑

把 $J$ 与高斯卷积，得到光滑代理：

```math
J_\sigma(\bar{\mathbf{u}})
= \mathbb{E}_{\epsilon\sim\mathcal{N}(0,\Sigma)}[J(\bar{\mathbf{u}}+\epsilon)]
= \int J(\bar{\mathbf{u}}+\epsilon)\,\phi(\epsilon;0,\Sigma)\,d\epsilon
```

对均值求导（$J$ 在高斯测度下可积即可交换积分）：

```math
\nabla J_\sigma(\bar{\mathbf{u}})
= \mathbb{E}_{\epsilon\sim\mathcal{N}(0,\Sigma)}\bigl[J(\bar{\mathbf{u}}+\epsilon)\,\Sigma^{-1}\epsilon\bigr]
```

常数项 $J(\bar{\mathbf{u}})$ 的贡献期望为零，减去它只降方差。平滑后 $J_\sigma$ 可任意阶求导，窄阱会被抹平一些，极小点的位置也可能略有移动。

## 2. 交叉熵方法与重要性采样

### 2.1 CEM 流程

在高斯族 $\mathcal{N}(\boldsymbol{\mu},\Sigma)$ 上迭代：

1. 初始化 $\mathcal{N}(\boldsymbol{\mu}_0,\Sigma_0)$
2. 对 $k=0,1,2,\dots$：采样 $\mathbf{u}^{(i)}\sim\mathcal{N}(\boldsymbol{\mu}_k,\Sigma_k)$，计算 $J^{(i)}$
3. 保留代价最低的约 $\rho\%$ 样本（精英集）
4. 用精英集重新估计均值和协方差

这是交叉熵法的标准做法：让参数分布去拟合精英所在的区域。更新协方差时，用采样那一步的旧均值做中心，几何意义更清楚。

### 2.2 玻尔兹曼分布与指数权重

把「好的控制」写成玻尔兹曼分布

```math
p^*(\mathbf{u})\propto\exp\bigl(-J(\mathbf{u})/\lambda\bigr)
```

在高斯族里匹配它的均值和协方差，相当于

```math
\boldsymbol{\mu}
=\frac{\int\mathbf{u}\,p^*(\mathbf{u})\,d\mathbf{u}}{\int p^*(\mathbf{u})\,d\mathbf{u}},
\qquad
\Sigma
=\frac{\int(\mathbf{u}-\boldsymbol{\mu})(\mathbf{u}-\boldsymbol{\mu})^\top p^*(\mathbf{u})\,d\mathbf{u}}{\int p^*(\mathbf{u})\,d\mathbf{u}}
```

采样来自当前提议 $q=\mathcal{N}(\boldsymbol{\mu}_k,\Sigma_k)$。信息论里的最优分布还乘上这份先验，即 $p_q(\mathbf{u})\propto e^{-J/\lambda}q(\mathbf{u})$，于是重要性权重就是指数代价本身：

```math
w_i=\exp\bigl(-J(\mathbf{u}^{(i)})/\lambda\bigr),
\qquad
\boldsymbol{\mu}_{k+1}=\frac{\sum_{i=1}^N w_i\mathbf{u}^{(i)}}{\sum_{i=1}^N w_i}
```

协方差同理用 $w_i$ 做加权二阶矩。$\lambda$ 较小、权重很尖时，效果接近只留最精英的一小撮；$\rho\%$ 截断则是固定比例的均匀权。二者都在缩小搜索分布，温度和分位数是两种旋钮。

只改均值、固定 $\Sigma$，就是下一节的 MPPI。均值与 $\Sigma$ 一起按加权（或按排序权）更新，就是 CMA / MPPI-CMA；长视界下 $\Sigma$ 取块对角，每时刻一块 $n_u\times n_u$。

## 3. MPPI 的两种推导路径

本节固定 $\Sigma$，只更新均值 $\mathbf{v}$。

### 3.1 信息论与 Log-Sum-Exp

目标分布取先验与玻尔兹曼的乘积：

```math
p_q(\mathbf{u})=\frac{1}{Z}\exp\bigl(-J(\mathbf{u})/\lambda\bigr)\,q(\mathbf{u}),
\qquad
q=\mathcal{N}(\mathbf{v},\Sigma)
```

均值的自归一估计为

```math
\mathbb{E}_{p_q}[\mathbf{u}]
=\frac{\mathbb{E}_q\bigl[\mathbf{u}\exp(-J(\mathbf{u})/\lambda)\bigr]}{\mathbb{E}_q\bigl[\exp(-J(\mathbf{u})/\lambda)\bigr]}
```

令 $\mathbf{u}=\mathbf{v}+\epsilon$，$\epsilon\sim\mathcal{N}(0,\Sigma)$：

```math
\mathbf{v}^*
=\mathbf{v}
+\frac{\mathbb{E}_{\epsilon}\bigl[\epsilon\exp(-J(\mathbf{v}+\epsilon)/\lambda)\bigr]}{\mathbb{E}_{\epsilon}\bigl[\exp(-J(\mathbf{v}+\epsilon)/\lambda)\bigr]}
```

有限样本即 MPPI：

```math
\mathbf{v}^* = \mathbf{v} + \frac{\sum_{i=1}^N w_i\epsilon^{(i)}}{\sum_{i=1}^N w_i},
\qquad
w_i = \exp\bigl(-J(\mathbf{v}+\epsilon^{(i)})/\lambda\bigr)
```

同一期望也是 log-sum-exp 平滑目标

```math
J_{\Sigma,\lambda}(\mathbf{v})
=-\lambda\log\mathbb{E}_{\epsilon\sim\mathcal{N}(0,\Sigma)}\bigl[\exp(-J(\mathbf{v}+\epsilon)/\lambda)\bigr]
```

对 $\mathbf{v}$ 的梯度。$\lambda\to\infty$ 时回到 §1.3 的 $J_\sigma$；$\lambda\to 0$ 时接近硬最小。实现上常减 $\min_i J^{(i)}$ 再取指数，只防溢出。样本极多时加权平均几乎没有随机性，需要靠 $\lambda$、$\Sigma$ 或重启保持探索。

### 3.2 朗之万动力学

从密度 $p$ 中采样的过阻尼朗之万离散为

```math
\mathbf{u}_{k+1} = \mathbf{u}_k + \alpha\nabla\log p(\mathbf{u}_k) + \sqrt{2\alpha}\,\mathbf{z}_k
```

$p^*\propto\exp(-J/\lambda)$ 的得分含 $\nabla J$，零阶设定下并不直接可用。改为对高斯平滑后的密度求得分：

```math
p_\sigma(\mathbf{u})
=\int p^*(\mathbf{u}')\,\mathcal{N}(\mathbf{u}\mid\mathbf{u}',\Sigma)\,d\mathbf{u}'
```

```math
\nabla_{\mathbf{u}}\mathcal{N}(\mathbf{u}\mid\mathbf{u}',\Sigma)
=-\Sigma^{-1}(\mathbf{u}-\mathbf{u}')\,\mathcal{N}(\mathbf{u}\mid\mathbf{u}',\Sigma)
```

```math
\nabla\log p_\sigma(\mathbf{u})
=-\Sigma^{-1}
\frac{\int p^*(\mathbf{u}')(\mathbf{u}-\mathbf{u}')\mathcal{N}(\mathbf{u}\mid\mathbf{u}',\Sigma)\,d\mathbf{u}'}{p_\sigma(\mathbf{u})}
```

令 $\epsilon=\mathbf{u}'-\mathbf{u}$，核对称，用 $\epsilon\sim\mathcal{N}(0,\Sigma)$ 估计积分：

```math
\nabla\log p_\sigma(\mathbf{u})
\approx\Sigma^{-1}\frac{\sum_{i=1}^N w_i\epsilon^{(i)}}{\sum_{i=1}^N w_i},
\qquad
w_i=\exp\bigl(-J(\mathbf{u}+\epsilon^{(i)})/\lambda\bigr)
```

于是

```math
\mathbf{v}+\frac{\sum_i w_i\epsilon^{(i)}}{\sum_i w_i}
=\mathbf{v}+\Sigma\nabla\log p_\sigma(\mathbf{v})
```

正是 MPPI 更新：沿 $\log p_\sigma$ 在 $\Sigma$ 度量下走一步。加上 $\sqrt{2\alpha}\,\mathbf{z}_k$ 就是完整朗之万；MPPI 取确定性漂移，把随机性留在每步的采样 $\epsilon^{(i)}$ 里。

## 4. 朗之万动力学与扩散模型

### 4.1 物理图像

过阻尼朗之万方程：

```math
\gamma\frac{d\mathbf{x}}{dt}=-\nabla U(\mathbf{x})+\sqrt{2\gamma k_B T}\,\boldsymbol{\xi}(t)
```

$U=-\log p$ 时与上一节的得分形式一致。温度降低，样本集中到 $U$ 的低谷，也就是模拟退火。

### 4.2 得分匹配

得分 $s(\mathbf{x})=\nabla_{\mathbf{x}}\log p(\mathbf{x})$。用网络拟合平滑密度的得分，可用去噪得分匹配：

```math
\mathcal{L}(\theta)
=\mathbb{E}_{\mathbf{x}\sim p}
\mathbb{E}_{\tilde{\mathbf{x}}\sim q(\tilde{\mathbf{x}}\mid\mathbf{x})}
\bigl\|s_\theta(\tilde{\mathbf{x}})-\nabla_{\tilde{\mathbf{x}}}\log q(\tilde{\mathbf{x}}\mid\mathbf{x})\bigr\|^2
```

$q(\tilde{\mathbf{x}}\mid\mathbf{x})=\mathcal{N}(\tilde{\mathbf{x}}\mid\mathbf{x},\sigma^2 I)$ 时，

```math
\nabla_{\tilde{\mathbf{x}}}\log q(\tilde{\mathbf{x}}\mid\mathbf{x})=-(\tilde{\mathbf{x}}-\mathbf{x})/\sigma^2
```

### 4.3 退火朗之万

噪声水平 $\sigma_1>\cdots>\sigma_K$ 递减时，

```math
\mathbf{x}_{k+1}=\mathbf{x}_k+\alpha_k s_\theta(\mathbf{x}_k,\sigma_k)+\sqrt{2\alpha_k}\,\mathbf{z}_k
```

先在宽核上混合，再收到尖峰。MPC 里也可以不训练 $s_\theta$，直接按同样精神缩小 MPPI 的 $\Sigma$，见第 6 节。

## 5. Stein 变分梯度下降

用粒子逼近目标密度，每步沿 RKHS 中下降 $D_{\mathrm{KL}}(q\|p)$ 的方向走：

```math
\phi^*(\mathbf{x}')
=\mathbb{E}_{\mathbf{x}\sim q}
\bigl[k(\mathbf{x},\mathbf{x}')\nabla_{\mathbf{x}}\log p(\mathbf{x})+\nabla_{\mathbf{x}}k(\mathbf{x},\mathbf{x}')\bigr]
```

```math
\mathbf{x}_i\leftarrow\mathbf{x}_i
+\alpha\frac{1}{n}\sum_{j=1}^n
\bigl[k(\mathbf{x}_j,\mathbf{x}_i)\nabla_{\mathbf{x}_j}\log p(\mathbf{x}_j)
+\nabla_{\mathbf{x}_j}k(\mathbf{x}_j,\mathbf{x}_i)\bigr]
```

核加权的得分把粒子推向高密度区，核梯度使粒子互相推开，适合多峰。$\nabla\log p$ 在零阶里可用 §3.2 的平滑得分代替。

## 6. 基于扩散的 MPC：DIAL-MPC

生成式扩散把干净样本逐步加噪，再学逆向去噪。前向为

```math
q(\mathbf{u}_t\mid\mathbf{u}_{t-1})=\mathcal{N}(\mathbf{u}_t\mid\sqrt{1-\beta_t}\mathbf{u}_{t-1},\beta_t I)
```

逆向 $p_\theta(\mathbf{u}_{t-1}\mid\mathbf{u}_t)$ 由网络给出。这与「用得分做退火朗之万」是同一家族。

在线 MPC 更直接的做法是把退火加在采样协方差上。时间维（优化迭代 $i$）和视界维（距当前时刻的步数 $h$）各自从大噪声收到小噪声：

```math
\Sigma_{\mathrm{time}}^i
=\sigma_{\max}^2\exp\left(-\frac{N_{\mathrm{time}}-i}{\beta_{\mathrm{time}}N_{\mathrm{time}}}\log\frac{\sigma_{\max}^2}{\sigma_{\min}^2}\right)I
```

```math
\Sigma_{\mathrm{space}}^h
=\sigma_0^2\exp\left(-\frac{H-h}{\beta_{\mathrm{space}}H}\log\frac{\sigma_0^2}{\sigma_H^2}\right)I
```

合成为 $\Sigma_{t+h}^i=\Sigma_{\mathrm{time}}^i\Sigma_{\mathrm{space}}^h$（均为单位阵倍数时即方差相乘）。靠近当前控制、靠近迭代末期时更贪心，远处、早期更敢探索。

## 7. 主流零阶优化算法全景

| 类别 | 代表算法 | 核心思想 | 适用场景 |
|------|----------|----------|----------|
| 随机搜索 | 预测采样 | 采样后取最小 | 低维、易并行 |
| 交叉熵 | CEM | 精英集拟合高斯 | 中等维数 |
| 路径积分 | MPPI | 指数权重更新均值 | 实时控制 |
| 进化策略 | CMA-ES | 同时适应协方差 | 病态、尺度变化 |
| 变分推断 | SVGD | 带排斥的粒子流 | 多峰 |
| 扩散退火 | DIAL-MPC | 按日程缩小采样噪声 | 复杂地形、需先探索后利用 |

经验上：维数很低可用 CEM、CMA；要滚动实时多用 MPPI 或对其 $\Sigma$ 退火；多峰明显时加 SVGD 或多种群。理论分析则回到朗之万与玻尔兹曼。

## 8. 理论统一框架

### 8.1 信息论目标

带先验的最大熵问题

```math
\min_q\ \mathbb{E}_q[J(\mathbf{u})]+\lambda D_{\mathrm{KL}}(q\|p_0)
```

的解正是 $q\propto p_0 e^{-J/\lambda}$，也就是 MPPI 的 $p_q$。CEM 用精英样本做矩匹配，在高斯族上逼近同一类好控制区域。SVGD 在粒子上下降 $D_{\mathrm{KL}}(q\|p)$。扩散与得分匹配则提供 $\nabla\log p_\sigma$ 的估计或噪声日程。

### 8.2 一条链条

1. 高斯平滑给出不依赖 $\nabla J$ 的梯度。
2. 指数权重把 CEM 的精英更新连到 MPPI 的均值更新。
3. 平滑密度的得分与 MPPI 同一步；加上噪声与退火就进入朗之万和扩散。
4. CMA 在同一高斯上再更新 $\Sigma$。

### 8.3 可延伸的方向

把双重退火与 SVGD 的排斥合在一起；用学习模型提供更好的提议分布 $q$；以及非凸问题中有限步收敛的保证。

## 结论

MPPI 既可以看作指数变换下的均值估计（log-sum-exp），也可以看作对平滑玻尔兹曼走一步确定性朗之万漂移。

```text
随机采样 → CEM（精英或指数权）→ MPPI
              ↗ 信息论 / Log-Sum-Exp
              ↘ 平滑得分 / 朗之万 → 退火与扩散式 MPC
                                    CMA 适应 Σ
```

温度、协方差和是否保留噪声，是在同一套语言里调节探索与集中。
