# RL and Optimal Control — 读书笔记仓库

本仓库按**文献**分目录存放：每份资料各自包含 PDF、抽取正文、读书笔记与相关脚本，互不混杂。

| 目录 | 文献 | 笔记入口 |
|------|------|----------|
| [`bertsekas-rl-oc/`](bertsekas-rl-oc/) | Dimitri P. Bertsekas, *Reinforcement Learning and Optimal Control*（2019 draft） | [`study-notes/`](bertsekas-rl-oc/study-notes/) |
| [`zero-order-robotics/`](zero-order-robotics/) | Jordana 等, *Zero-Order Optimization Techniques for Robotics*（arXiv:2506.22087v2, 2025） | [`study-notes/`](zero-order-robotics/study-notes/) |
| [`sqp-oc/`](sqp-oc/) | Jordana 等 SQP/MPC 系列 + Chakravorty, *SQP Perspective on Optimal Control*（2023–2025） | [`study-notes/control-optimization-sqp-synthesis.md`](sqp-oc/study-notes/control-optimization-sqp-synthesis.md) |
| [`clean-rl/`](clean-rl/) | 深度 RL 算法笔记（对照单文件实现阅读） | [`ppo_notes.md`](clean-rl/ppo_notes.md)、[`sac_notes.md`](clean-rl/sac_notes.md) |

**对照阅读**：Bertsekas 提供 Bellman / MPC / 策略梯度理论骨架；Zero-Order 用随机搜索统一 TO（MPPI、CMA）与 RL；SQP 组用数值优化语言统一 iLQR、DDP、多重打靶 NMPC 与结构利用型 QP 求解器；`clean-rl/` 用 PPO、SAC 把策略梯度与 off-policy Actor-Critic 落到可对照的推导上。

版权在原书 / 原论文，仅供个人学习，请勿商业再分发。

## 常用命令

**Bertsekas — 从 PDF 重新抽章节正文：**

```bash
cd bertsekas-rl-oc
python3 scripts/extract_chapters.py
python3 scripts/split_chapter_parts.py
python3 scripts/fix_study_notes_math.py
```

**Zero-Order — 从 PDF 重新抽正文：**

```bash
cd zero-order-robotics
python3 scripts/extract_pdf.py
python3 scripts/fix_study_notes_math.py
```

**SQP / MPC — 从 PDF 重新抽正文：**

```bash
cd sqp-oc
for f in *.pdf; do pdftotext "$f" "source/${f%.pdf}.txt"; done
```

各子目录下的 `README.md` 有更细的说明。

如有漏洞或错误，欢迎留言指出。
