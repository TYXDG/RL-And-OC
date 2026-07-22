# RL and Optimal Control — 读书笔记仓库

Bertsekas, *Reinforcement Learning and Optimal Control*（2019 draft）个人学习笔记。

**说明**：本仓库为个人学习整理，**不包含**原著 PDF。抽取文本仅供与笔记对照；版权归 Dimitri P. Bertsekas / Athena Scientific。请自行通过正规渠道获取原书。

## 目录结构

| 路径 | 用途 |
|------|------|
| [`study-notes/`](study-notes/) | **讲解型分节读书笔记**（主要成果） |
| [`source/`](source/) | 从 PDF 抽取的正文：`full.txt`（全书合并）、`ch01_clean.txt` … `ch05_clean.txt`；长章可读 `source/parts/chXX_partNN.txt` |
| [`Reinforcement learning and optimal control.pdf`](Reinforcement%20learning%20and%20optimal%20control.pdf) | 原著 PDF（需重新抽取时对照） |
| [`scripts/`](scripts/) | 维护用：`extract_chapters.py`、`split_chapter_parts.py`、`fix_study_notes_math.py` |

## 修改 / 续写笔记时（给 Agent）

1. 读对应章 **`source/chNN_clean.txt`**（或按小节读 **`source/parts/`**）。
2. 编辑 **`study-notes/chNN-*-study-notes.md`**，公式用 **`$...$`** / **`$$...$$`**。
3. 若误用 `\(...\)`，运行：`python3 scripts/fix_study_notes_math.py`。
4. 预览说明见 [`study-notes/README.md`](study-notes/README.md)。

## 重新从 PDF 抽取正文

```bash
python3 scripts/extract_chapters.py   # 生成 source/*_clean.txt、full.txt
python3 scripts/split_chapter_parts.py  # 生成 source/parts/
```

（具体参数以各脚本内说明为准。）
