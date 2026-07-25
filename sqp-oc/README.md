# 有限时域最优控制的 SQP 方法 — 文献与笔记

本目录收录以 **Sequential Quadratic Programming (SQP)** 表述离散最优控制的一组文献：有限时域 OCP 的 NLP 形式、iLQR/DDP 与多重打靶 MPC 的算法对照，以及结构利用型 QP 求解（Riccati/OSQP_OCP）。

| 文件 | 文献 | 角色 |
|------|------|------|
| [`2023-Stagewise-Implementations-...pdf`](2023-Stagewise-Implementations-of-Sequential-Quadratic-Programming-for-Model-Predictive-Control.pdf) | Jordana 等 (2023, HAL) | 提出分阶段 SQP + Riccati/OSQP_OCP，初版实验 |
| [`2025-Structure-Exploiting_...pdf`](2025-Structure-Exploiting_Sequential_Quadratic_Programming_for_Model-Predictive_Control.pdf) | Jordana 等 (*IEEE TRO*, 2025) | 期刊扩展版：约束 MPC 真机、QP 基准 |
| [`2025-A Sequential Quadratic Programming Perspective on Optimal Control.pdf`](2025-A%20Sequential%20Quadratic%20Programming%20Perspective%20on%20Optimal%20Control.pdf) | Abhijeet & Chakravorty (2025, arXiv:2510.03475) | 理论统一：Newton / iLQR / DDP 的 SQP 对照 |
| [`supplementary_material.pdf`](supplementary_material.pdf) | Jordana 等附录 | KKT 块三对角结构 → Thomas 算法 → Riccati 递推证明 |

**笔记入口**：[`study-notes/control-optimization-sqp-synthesis.md`](study-notes/control-optimization-sqp-synthesis.md)（四篇文献研读笔记）

正文抽取：[`source/`](source/)（`pdftotext` 生成，供对照，非 OCR 逐页）。

## 与本仓库其他资料的对照

| 目录 | 互补关系 |
|------|----------|
| [`bertsekas-rl-oc/`](../bertsekas-rl-oc/) | Bellman / 值函数 / MPC 滚动时域的**控制理论**骨架 |
| [`zero-order-robotics/`](../zero-order-robotics/) | 无梯度随机搜索统一 TO 与 RL；与本目录的**一阶/二阶 NLP** 形成方法谱系的两端 |

## 常用命令

从 PDF 重新抽取正文：

```bash
cd sqp-oc
for f in *.pdf; do
  base="${f%.pdf}"
  pdftotext "$f" "source/${base}.txt"
done
```

版权在原论文，仅供个人学习。
