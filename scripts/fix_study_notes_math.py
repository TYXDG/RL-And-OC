#!/usr/bin/env python3
"""Normalize LaTeX in study-notes for Cursor and GitHub Markdown math."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTES = ROOT / "study-notes"

DISPLAY_DELIM = re.compile(r"^\s*\$\$\s*$")


def convert_latex_delimiters(text: str) -> str:
    text = re.sub(r"\\\[(.+?)\\\]", r"$$\n\1\n$$", text, flags=re.DOTALL)
    text = re.sub(r"\\\((.+?)\\\)", r"$\1$", text, flags=re.DOTALL)
    return text


def fix_orphan_tags(text: str) -> str:
    """\\tag{} after closing $ is plain text on GitHub — use (eq.no) instead."""
    return re.sub(
        r"(\$[^$\n]+\$)[。.]?\s*\\tag\{([^}]+)\}",
        r"\1 (\2)",
        text,
    )


def brace_star_superscripts(text: str) -> str:
    """GitHub Markdown eats ^* inside $...$; use ^{} for optimal asterisk."""
    return re.sub(r"\^(?!\{)\*", r"^{*}", text)


def fix_bold_dollar_space(text: str) -> str:
    return re.sub(r"\*\*\$\s+", "**$", text)


def fix_text_en_dash(text: str) -> str:
    def fix_match(m: re.Match[str]) -> str:
        inner = m.group(1).replace("\u2013", "-").replace("\u2014", "-")
        return r"\text{" + inner + "}"

    return re.sub(r"\\text\{([^}]*)\}", fix_match, text)


def demote_list_items_before_math_fences(text: str) -> str:
    """GitHub: list + ```math often fails; use plain lines before fences."""
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m_num = re.match(r"^\d+\.\s+(.+)$", line)
        m_bul = re.match(r"^-\s+(.+)$", line)
        content: str | None = None
        if m_num:
            content = m_num.group(1).strip()
        elif m_bul:
            content = m_bul.group(1).strip()
        if content is not None:
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and lines[j].strip() == "```math":
                if not content.endswith(("：", ":")):
                    content += "："
                out.append(content)
                i += 1
                continue
        out.append(line)
        i += 1
    return "\n".join(out)


def replace_display_tags(text: str) -> str:
    return re.sub(
        r"\\tag\{([^}]+)\}",
        r"\\qquad \\text{(\1)}",
        text,
    )


def normalize_display_math(text: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        if not DISPLAY_DELIM.match(lines[i]):
            out.append(lines[i])
            i += 1
            continue

        i += 1
        body: list[str] = []
        while i < len(lines) and not DISPLAY_DELIM.match(lines[i]):
            stripped = lines[i].strip()
            if stripped:
                body.append(stripped)
            i += 1

        if i >= len(lines):
            out.append("$$")
            out.extend(body)
            break

        i += 1

        if out and out[-1].strip():
            out.append("")
        out.append("$$")
        out.extend(body)
        out.append("$$")
        if i < len(lines) and lines[i].strip():
            out.append("")

    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def convert_display_to_math_fences(text: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        if DISPLAY_DELIM.match(lines[i]):
            i += 1
            body: list[str] = []
            while i < len(lines) and not DISPLAY_DELIM.match(lines[i]):
                s = lines[i].strip()
                if s:
                    body.append(s)
                i += 1
            if i < len(lines):
                i += 1
            if out and out[-1].strip():
                out.append("")
            out.append("```math")
            out.extend(body)
            out.append("```")
            if i < len(lines) and lines[i].strip():
                out.append("")
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out)


def collapse_math_fences(text: str) -> str:
    """Single-line ```math bodies render more reliably on GitHub."""

    def repl(m: re.Match[str]) -> str:
        lines = [ln.strip() for ln in m.group(1).splitlines() if ln.strip()]
        return "```math\n" + " ".join(lines) + "\n```"

    return re.sub(r"```math\n(.*?)```", repl, text, flags=re.DOTALL)


def process(text: str) -> str:
    text = convert_latex_delimiters(text)
    text = fix_orphan_tags(text)
    text = brace_star_superscripts(text)
    text = fix_bold_dollar_space(text)
    text = demote_list_items_before_math_fences(text)
    text = normalize_display_math(text)
    text = convert_display_to_math_fences(text)
    text = replace_display_tags(text)
    text = fix_text_en_dash(text)
    text = collapse_math_fences(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def main() -> None:
    for md in sorted(NOTES.glob("ch*.md")):
        original = md.read_text(encoding="utf-8")
        updated = process(original)
        if updated != original:
            md.write_text(updated, encoding="utf-8")
            print(f"updated {md.name}")
        else:
            print(f"ok {md.name}")


if __name__ == "__main__":
    main()
