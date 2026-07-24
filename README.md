# RL and Optimal Control — 读书笔记

自己在读 Bertsekas 的 *Reinforcement Learning and Optimal Control*（2019 draft），笔记和抽取文本放在这个仓库里方便对照和继续改。

仓库里有 PDF 和从 PDF 抽出来的 `source/` 文本，都是学习用；书版权归 Dimitri P. Bertsekas / Athena Scientific，请勿拿去做商业再分发。

## 笔记进度

| 章节 | 状态 | 说明 |
|------|------|------|
| 第 1 章 Exact DP | 进行中 | §1.1.2 手算/DP 流程、§1.1.3 Q 因子与近似、§1.2 随机 DP（期望成本、策略 vs 开环、随机 Q 与采样动机）已写细；§1.3–§1.4 有骨架 |
| 第 2–5 章 | 有初稿 | 见 `study-notes/ch02-*` … `ch05-*` |

## 里面有什么

| 路径 | 干什么用 |
|------|----------|
| [`study-notes/`](study-notes/) | 分节读书笔记（主要看这个） |
| [`source/`](source/) | 正文抽取：`ch01_clean.txt` … `ch05_clean.txt`，长章还可以拆成 `source/parts/` |
| [`Reinforcement learning and optimal control.pdf`](Reinforcement%20learning%20and%20optimal%20control.pdf) | 原 PDF，要重新抽文本时对着看 |
| [`scripts/`](scripts/) | 抽章节、拆 part、修公式格式之类的小脚本 |

## 接着写笔记

先看 `source/chNN_clean.txt`（或按小节看 `source/parts/`），再改 `study-notes/chNN-*-study-notes.md`。公式照旧用 `$` / `$$`；不小心写成 `\(...\)` 的话，跑 `python3 scripts/fix_study_notes_math.py` 能改回来。预览和 GitHub 上公式要注意什么，写在 [`study-notes/README.md`](study-notes/README.md) 里。

## 从 PDF 重新抽正文

```bash
python3 scripts/extract_chapters.py
python3 scripts/split_chapter_parts.py
```

具体选项看脚本开头的注释就行。
