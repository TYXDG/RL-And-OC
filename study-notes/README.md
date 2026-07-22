# 分节读书笔记

这里是按 PDF 小节写的读书笔记：每节有主要内容、公式和一点个人理解上的注意点。文字对照 [`source/`](../source/) 里的抽取正文，不是逐页 OCR 对照。

书是 Bertsekas 的 *Reinforcement Learning and Optimal Control*（2019 draft）。行内公式用 `$...$`，单独成行的大公式用 ` ```math ` 代码块（GitHub 上比 `$$` 稳）；最优值里的星号写成 `J_k^{*}` 这种，别用裸 `^*`，否则网页会把 `*` 当强调吃掉。

## 各章文件

| 文件 | 对应章节 |
|------|----------|
| [`ch01-exact-dp-study-notes.md`](ch01-exact-dp-study-notes.md) | 第 1 章 Exact DP |
| [`ch02-approximation-in-value-space-study-notes.md`](ch02-approximation-in-value-space-study-notes.md) | 第 2 章 值空间近似 |
| [`ch03-parametric-approximation-study-notes.md`](ch03-parametric-approximation-study-notes.md) | 第 3 章 参数化近似 |
| [`ch04-infinite-horizon-rl-study-notes.md`](ch04-infinite-horizon-rl-study-notes.md) | 第 4 章 无限时域（理论） |
| [`ch05-infinite-horizon-approximate-study-notes.md`](ch05-infinite-horizon-approximate-study-notes.md) | 第 5 章 无限时域近似方法 |

版权在原书，这里只是个人学习笔记，别当正式出版物用。

## 公式怎么预览

**在 Cursor 里**开 Markdown 预览（`Ctrl+Shift+V`，Mac 用 `Cmd+Shift+V`）一般就能看见公式。行内 `$...$`；块级用 ` ```math ` … ` ``` `（新版预览也认这个）。正文里裸写 `x_k` 而不包在 `$` 里，下划线会被当成斜体。

预览里公式还是不对的话，确认设置里开了 `markdown.math.enabled`，或者装个 Markdown Math 扩展再试。要是反斜杠被编辑器吃掉了，在项目根跑一下：

```bash
python3 scripts/fix_study_notes_math.py
```

**在 GitHub 网页上**请用文件页的 **Preview / 渲染视图**，不要看 Raw。容易踩坑：列表项下面直接接 `$$` 或缩进公式；`$J_k^*$` 里的 `*` 会被 Markdown 删掉。块级公式用顶格的 ` ```math `（单行 LaTeX），最优星号写 `^{*}`；列表里要先接公式的话，把那一项改成普通段落（脚本会自动处理）。改完跑 `python3 scripts/fix_study_notes_math.py`。
