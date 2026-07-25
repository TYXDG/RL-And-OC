# 分节读书笔记

按 PDF 小节写的读书笔记：每节有主要内容、公式和个人理解上的注意点。文字对照 [`../source/`](../source/) 里的抽取正文，不是逐页 OCR 对照。

书是 Bertsekas 的 *Reinforcement Learning and Optimal Control*（2019 draft）。行内公式用 `$...$`，块级大公式用 ` ```math ` 代码块；最优星号写成 `J_k^{*}`，别用裸 `^*`。

## 各章文件

| 文件 | 对应章节 |
|------|----------|
| [`ch01-exact-dp-study-notes.md`](ch01-exact-dp-study-notes.md) | 第 1 章 Exact DP |
| [`ch02-approximation-in-value-space-study-notes.md`](ch02-approximation-in-value-space-study-notes.md) | 第 2 章 值空间近似 |
| [`ch03-parametric-approximation-study-notes.md`](ch03-parametric-approximation-study-notes.md) | 第 3 章 参数化近似 |
| [`ch04-infinite-horizon-rl-study-notes.md`](ch04-infinite-horizon-rl-study-notes.md) | 第 4 章 无限时域（理论） |
| [`ch05-infinite-horizon-approximate-study-notes.md`](ch05-infinite-horizon-approximate-study-notes.md) | 第 5 章 无限时域近似方法 |

## 相关阅读

| 路径 | 文献 |
|------|------|
| [`../../zero-order-robotics/study-notes/`](../../zero-order-robotics/study-notes/) | Jordana 等 (2025) 零阶优化教程 — 与 Ch.2 MPC、Ch.4 策略梯度对照 |

## 公式预览

在 Cursor 里 Markdown 预览（`Ctrl+Shift+V`）一般可见公式。改完笔记可在本目录上级运行：

```bash
python3 ../scripts/fix_study_notes_math.py
```

GitHub 上请用 Preview，块级公式用顶格 ` ```math `；列表项下直接接公式容易渲染失败（脚本会尽量处理）。
