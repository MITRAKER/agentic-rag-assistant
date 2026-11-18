"""Print one version's section of CHANGELOG.md, to use as GitHub release notes.

The release workflow runs this when a version tag such as v0.2.0 is pushed.
It exits with status 1 when CHANGELOG.md has no section for that version, so
a release can't be published without notes.

Usage:  python tools/release_notes.py VERSION [CHANGELOG]
        python tools/release_notes.py v0.2.0          # the leading "v" is optional
"""

from __future__ import annotations

import sys
from pathlib import Path


def is_link_reference(line: str) -> bool:
    """Lines like '[0.2.0]: https://...' at the end of a Keep a Changelog file."""
    return line.startswith("[") and "]: " in line


def section(changelog: str, version: str) -> str:
    """The body of the `## [version]` section, without its heading.

    A section ends at the next `## [` heading or at the link references.
    Returns an empty string when the version has no section.
    """
    heading = f"## [{version}]"
    body: list[str] = []
    inside = False
    for line in changelog.splitlines():
        if line.startswith("## [") or is_link_reference(line):
            if inside:
                break
            inside = line.startswith(heading)
        elif inside:
            body.append(line)
    return "\n".join(body).strip()


def main(argv: list[str]) -> int:
    match argv:
        case [tag]:
            path = Path("CHANGELOG.md")
        case [tag, changelog]:
            path = Path(changelog)
        case _:
            sys.stderr.write("usage: python tools/release_notes.py VERSION [CHANGELOG]\n")
            return 2
    version = tag.removeprefix("v")
    notes = section(path.read_text(encoding="utf-8"), version)
    if not notes:
        sys.stderr.write(f"{path} has no section for version {version}.\n")
        return 1
    sys.stdout.write(f"{notes}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
