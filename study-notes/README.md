# 分节读书笔记（Study Notes）

本目录存放按 PDF 小节整理的**讲解型读书笔记**（主要内容、公式、要点与局限），正文依据 [`source/`](../source/)。语体为**讲义式学术中文**；公式使用 **`$...$` / `$$...$$`**。

源文献：Bertsekas, *Reinforcement Learning and Optimal Control*（2019 draft）。正文依据 `source/` 抽取文本。

## 文件

| 文件 | 内容 |
|------|------|
| [`ch01-exact-dp-study-notes.md`](ch01-exact-dp-study-notes.md) | 第 1 章 Exact Dynamic Programming |
| [`ch02-approximation-in-value-space-study-notes.md`](ch02-approximation-in-value-space-study-notes.md) | 第 2 章 Approximation in Value Space |
| [`ch03-parametric-approximation-study-notes.md`](ch03-parametric-approximation-study-notes.md) | 第 3 章 Parametric Approximation |
| [`ch04-infinite-horizon-rl-study-notes.md`](ch04-infinite-horizon-rl-study-notes.md) | 第 4 章 Infinite Horizon RL（理论） |
| [`ch05-infinite-horizon-approximate-study-notes.md`](ch05-infinite-horizon-approximate-study-notes.md) | 第 5 章无限时域近似方法 |

## 版权声明

仅供个人学习；原著 Copyright Dimitri P. Bertsekas / Athena Scientific。

## 在 Cursor 里正确预览公式

1. 打开 Markdown **预览**：`Ctrl+Shift+V`（Mac：`Cmd+Shift+V`），或侧边预览 `Ctrl+K V`。
2. 本仓库笔记使用 **`$...$`（行内）** 与 **`$$...$$`（独立公式）**；不要用裸 `x_k`（下划线会被当成斜体）。
3. 已在用户设置中启用 `"markdown.math.enabled": true`。若公式仍不渲染，在扩展市场安装 **Markdown Math** 后重开预览。
4. 若文件曾被编辑器去掉反斜杠导致公式乱码，可在项目根运行：  
   `python3 scripts/fix_study_notes_math.py`

## 在 GitHub 上正确显示公式

GitHub 使用 MathJax，比 Cursor 更挑剔：

- 块级公式：`$$` 单独一行，**公式中间不要空行**，结束 `$$` 也单独一行且**顶格写**（不要缩进在列表里）。
- 式号请写在 `$...$` 或 `$$...$$` **内部**（或用正文 `(1.4)`），不要把 `\tag{...}` 留在 `$` 外面。
- 推送前可运行 `python3 scripts/fix_study_notes_math.py` 做上述规范化。
