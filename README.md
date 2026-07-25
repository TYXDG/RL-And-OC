# RL and Optimal Control — 读书笔记仓库

本仓库按**文献**分目录存放：每份资料各自包含 PDF、抽取正文、读书笔记与相关脚本，互不混杂。

| 目录 | 文献 | 笔记入口 |
|------|------|----------|
| [`bertsekas-rl-oc/`](bertsekas-rl-oc/) | Dimitri P. Bertsekas, *Reinforcement Learning and Optimal Control*（2019 draft） | [`study-notes/`](bertsekas-rl-oc/study-notes/) |
| [`zero-order-robotics/`](zero-order-robotics/) | Jordana 等, *Zero-Order Optimization Techniques for Robotics*（arXiv:2506.22087v2, 2025） | [`study-notes/`](zero-order-robotics/study-notes/) |

两篇笔记可对照阅读：Bertsekas 讲 Bellman / MPC / 策略梯度理论；Zero-Order 教程用随机搜索统一 TO（MPPI、CMA）与 RL（DPG、Reinforce）。

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

各子目录下的 `README.md` 有更细的说明。
