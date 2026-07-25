# Bertsekas — Reinforcement Learning and Optimal Control

Dimitri P. Bertsekas, *Reinforcement Learning and Optimal Control*（Athena Scientific 2019 draft）。书版权归作者 / 出版社，本目录仅供学习。

## 目录结构

| 路径 | 用途 |
|------|------|
| [`Reinforcement learning and optimal control.pdf`](Reinforcement%20learning%20and%20optimal%20control.pdf) | 原 PDF |
| [`source/`](source/) | 抽取正文：`full.txt`、`ch01_clean.txt` … `ch05_clean.txt`，长章见 [`source/parts/`](source/parts/) |
| [`study-notes/`](study-notes/) | **分节读书笔记**（唯一笔记目录） |
| [`scripts/`](scripts/) | 抽章节、拆 part、修公式等 |

## 笔记进度

| 章节 | 状态 | 说明 |
|------|------|------|
| 第 1 章 Exact DP | 进行中 | §1.1–§1.2 已写细；§1.3–§1.4 有骨架 |
| 第 2 章 值空间近似 | 进行中 | §2.0–§2.1 已扩充；§2.2 起有初稿 |
| 第 3–5 章 | 有初稿 | 见 [`study-notes/`](study-notes/) 各文件 |

## 接着写笔记

1. 对照 [`source/chNN_clean.txt`](source/)（或 [`source/parts/`](source/parts/)）  
2. 编辑 [`study-notes/chNN-*-study-notes.md`](study-notes/)  
3. 公式预览与 GitHub 注意点见 [`study-notes/README.md`](study-notes/README.md)  
4. 改完可跑 `python3 scripts/fix_study_notes_math.py`

## 从 PDF 重新抽正文

```bash
python3 scripts/extract_chapters.py
python3 scripts/split_chapter_parts.py
```

## 相关阅读

- [`../zero-order-robotics/`](../zero-order-robotics/) — 零阶优化教程；与 Ch.2 MPC、Ch.4 策略梯度对照
