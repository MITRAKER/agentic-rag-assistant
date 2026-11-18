from pathlib import Path

from release_notes import main, section

CHANGELOG = """\
# Changelog

## [Unreleased]

## [0.2.0] - 2026-11-17

### Added

- Answers with citations.

## [0.1.0] - 2026-11-16

### Added

- Ingestion.

[Unreleased]: https://github.com/o/r/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/o/r/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/o/r/releases/tag/v0.1.0
"""


def test_section_returns_only_that_version():
    assert section(CHANGELOG, "0.2.0") == "### Added\n\n- Answers with citations."


def test_last_section_stops_at_the_link_references():
    assert section(CHANGELOG, "0.1.0") == "### Added\n\n- Ingestion."


def test_missing_or_empty_section_returns_empty_string():
    assert section(CHANGELOG, "9.9.9") == ""
    assert section(CHANGELOG, "Unreleased") == ""


def test_version_is_not_matched_as_a_prefix():
    assert section("## [0.2.10] - 2026-12-01\n\n- Later.\n", "0.2.1") == ""


def test_main_prints_notes_for_a_tag(tmp_path: Path, capsys):
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(CHANGELOG, encoding="utf-8")
    assert main(["v0.2.0", str(changelog)]) == 0
    assert capsys.readouterr().out == "### Added\n\n- Answers with citations.\n"


def test_main_reads_changelog_md_by_default(tmp_path: Path, monkeypatch, capsys):
    (tmp_path / "CHANGELOG.md").write_text(CHANGELOG, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert main(["0.1.0"]) == 0
    assert capsys.readouterr().out == "### Added\n\n- Ingestion.\n"


def test_main_fails_without_a_section(tmp_path: Path, capsys):
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(CHANGELOG, encoding="utf-8")
    assert main(["v3.0.0", str(changelog)]) == 1
    assert "no section for version 3.0.0" in capsys.readouterr().err


def test_main_needs_one_or_two_arguments(capsys):
    assert main([]) == 2
    assert main(["a", "b", "c"]) == 2
    assert "usage" in capsys.readouterr().err
