# Zero-Order Optimization Techniques for Robotics

Armand Jordana, Jianghan Zhang, Joseph Amigo, Ludovic Righetti, *An Introduction to Zero-Order Optimization Techniques for Robotics*（arXiv:2506.22087v2, 2025）。

## 目录结构

| 路径 | 用途 |
|------|------|
| [`2025-An Introduction to Zero-Order Optimization Techniques for Robotics.pdf`](2025-An%20Introduction%20to%20Zero-Order%20Optimization%20Techniques%20for%20Robotics.pdf) | 原 PDF |
| [`source/full.txt`](source/full.txt) | pdftotext 抽取正文 |
| [`study-notes/`](study-notes/) | **分节读书笔记** |
| [`scripts/`](scripts/) | `extract_pdf.py`、`fix_study_notes_math.py` |

## 笔记

- 主文件：[`study-notes/zero-order-optimization-study-notes.md`](study-notes/zero-order-optimization-study-notes.md)（§I–§VII）
- 说明：[`study-notes/README.md`](study-notes/README.md)

## 从 PDF 重新抽正文

```bash
python3 scripts/extract_pdf.py
python3 scripts/fix_study_notes_math.py
```

代码与实验：[zoo-rob](https://github.com/ajordana/zoo-rob)
