# 分节读书笔记

这里是按 PDF 小节写的读书笔记：每节有主要内容、公式和一点个人理解上的注意点。文字对照 [`source/`](../source/) 里的抽取正文，不是逐页 OCR 对照。

书是 Bertsekas 的 *Reinforcement Learning and Optimal Control*（2019 draft）。公式统一用 `$...$` 和 `$$...$$` 写，读起来像讲义，不太像题解册。

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

**在 Cursor 里**开 Markdown 预览（`Ctrl+Shift+V`，Mac 用 `Cmd+Shift+V`）一般就能看见公式。行内用 `$...$`，单独一行的大公式用 `$$...$$`。正文里如果直接写 `x_k` 而不包在 `$` 里，下划线会被 Markdown 当成强调，看起来会怪。

预览里公式还是不对的话，确认设置里开了 `markdown.math.enabled`，或者装个 Markdown Math 扩展再试。要是反斜杠被编辑器吃掉了，在项目根跑一下：

```bash
python3 scripts/fix_study_notes_math.py
```

**在 GitHub 网页上**渲染规则和 Cursor 不完全一样：块级公式要把 `$$` 单独占一行，中间别插空行，结束的那对 `$$` 也顶格写（缩进在列表里容易整段变乱码）。式号要么写在公式里面，要么干脆用正文里的 `(1.4)`，别把 `\tag{...}` 留在 `$` 外面。改完笔记、推上去之前，同样可以用上面那个脚本扫一遍。
