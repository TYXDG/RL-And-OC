# Bertsekas — Reinforcement Learning and Optimal Control

Dimitri P. Bertsekas, *Reinforcement Learning and Optimal Control*（Athena Scientific 2019 draft）。

| 路径 | 用途 |
|------|------|
| PDF | 书稿 |
| [`source/`](source/) | 抽取正文（`ch01_clean.txt` …，长章见 `source/parts/`） |
| [`study-notes/`](study-notes/) | 分节笔记 |
| [`scripts/`](scripts/) | 抽章节、拆 part、修公式 |

行内公式 `$...$`，块级 ` ```math `；最优星号写成 `J_k^{*}`。改完笔记可运行 `python3 scripts/fix_study_notes_math.py`。

| 笔记 | 状态 |
|------|------|
| [`00-algorithm-taxonomy.md`](study-notes/00-algorithm-taxonomy.md) | 跨章索引 |
| [`ch01-exact-dp-study-notes.md`](study-notes/ch01-exact-dp-study-notes.md) | 进行中 |
| [`ch02-approximation-in-value-space-study-notes.md`](study-notes/ch02-approximation-in-value-space-study-notes.md) | 较完整 |
| [`ch03-parametric-approximation-study-notes.md`](study-notes/ch03-parametric-approximation-study-notes.md) | 较完整 |
| [`ch04-infinite-horizon-rl-study-notes.md`](study-notes/ch04-infinite-horizon-rl-study-notes.md) | 较完整 |
| [`ch05-infinite-horizon-approximate-study-notes.md`](study-notes/ch05-infinite-horizon-approximate-study-notes.md) | 较完整 |

```bash
python3 scripts/extract_chapters.py
python3 scripts/split_chapter_parts.py
```
