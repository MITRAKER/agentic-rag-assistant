"""Small text helpers shared by ingestion and answering."""

from __future__ import annotations

import re

# Lines that start a new unit in Markdown: headings, list items, quotes, table rows.
_BLOCK_LINE = re.compile(r"^(#{1,6}\s|[-*+]\s|\d+[.)]\s|>|\|)")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
# Full stops that don't end a sentence: a list marker on its own ("1.", "a.")
# or a common abbreviation at the end of the text so far ("e.g.").
_LIST_MARKER = re.compile(r"(\d+|[A-Za-z])[.)]")
_ABBREVIATION = re.compile(r"(e\.g|i\.e|etc|vs|dr|mr|mrs|ms|fig|no)\.", re.IGNORECASE)


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, treating Markdown headings and list items as their own units.

    Many Markdown and text files are "hard-wrapped": a sentence continues on
    the next line. A single newline is therefore usually *not* a sentence
    boundary, so we join wrapped lines back together before splitting on
    sentence-ending punctuation.
    """
    units: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        merged: list[str] = []
        for line in lines:
            starts_new_unit = not merged or _BLOCK_LINE.match(line) or merged[-1].startswith("#")
            if starts_new_unit:
                merged.append(line)
            else:
                merged[-1] += " " + line  # a soft line wrap inside a sentence
        for unit in merged:
            units.extend(_split_unit(unit))
    return units


def _split_unit(unit: str) -> list[str]:
    sentences: list[str] = []
    for part in (p.strip() for p in _SENTENCE_END.split(unit)):
        if not part:
            continue
        prev = sentences[-1] if sentences else ""
        if prev and (_LIST_MARKER.fullmatch(prev) or _ABBREVIATION.fullmatch(prev.split()[-1])):
            sentences[-1] += " " + part  # "1." + "Install it" -> "1. Install it"
        else:
            sentences.append(part)
    return sentences
