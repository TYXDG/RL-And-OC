#!/usr/bin/env python3
"""Split cleaned chapter files into parts for staged translation."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "source"
PARTS = SRC / "parts"
CHUNK = 280  # paragraphs per part


def main() -> None:
    PARTS.mkdir(parents=True, exist_ok=True)
    for path in sorted(SRC.glob("ch*_clean.txt")):
        paras = [p.strip() for p in path.read_text(encoding="utf-8").split("\n\n") if p.strip()]
        stem = path.stem.replace("_clean", "")
        for i in range(0, len(paras), CHUNK):
            part = paras[i : i + CHUNK]
            idx = i // CHUNK + 1
            out = PARTS / f"{stem}_part{idx:02d}.txt"
            out.write_text("\n\n".join(part) + "\n", encoding="utf-8")
        n_parts = (len(paras) + CHUNK - 1) // CHUNK
        print(f"{stem}: {len(paras)} paragraphs -> {n_parts} parts")


if __name__ == "__main__":
    main()
