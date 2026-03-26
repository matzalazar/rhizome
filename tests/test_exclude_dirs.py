"""
Tests for the EXCLUDE_DIRS feature:
  - Config parsing (comma-separated string → list)
  - _is_excluded() helper
  - discover_notes() respects exclusions (Obsidian adapter)
  - _discover_logseq_paths() respects exclusions (Logseq adapter)
"""

from pathlib import Path

import pytest

from rhizome.config import Settings
from rhizome.vault import discover_notes
from rhizome.vault.logseq import _discover_logseq_paths
from rhizome.vault.obsidian import _is_excluded

# ---------------------------------------------------------------------------
# Config: EXCLUDE_DIRS parsing
# ---------------------------------------------------------------------------


@pytest.fixture()
def vault(tmp_path: Path) -> Path:
    (tmp_path / "note.md").write_text("# Note\nContent.")
    return tmp_path


def test_exclude_dirs_default_is_empty(vault: Path) -> None:
    s = Settings(vault_path=vault)
    assert s.exclude_dirs == []


def test_exclude_dirs_single(vault: Path) -> None:
    s = Settings(vault_path=vault, exclude_dirs="journal")
    assert s.exclude_dirs == ["journal"]


def test_exclude_dirs_comma_separated(vault: Path) -> None:
    s = Settings(vault_path=vault, exclude_dirs="journal,archive,private")
    assert s.exclude_dirs == ["journal", "archive", "private"]


def test_exclude_dirs_strips_whitespace(vault: Path) -> None:
    s = Settings(vault_path=vault, exclude_dirs=" journal , archive ")
    assert s.exclude_dirs == ["journal", "archive"]


def test_exclude_dirs_empty_string(vault: Path) -> None:
    s = Settings(vault_path=vault, exclude_dirs="")
    assert s.exclude_dirs == []


def test_exclude_dirs_accepts_list(vault: Path) -> None:
    s = Settings(vault_path=vault, exclude_dirs=["journal", "archive"])
    assert s.exclude_dirs == ["journal", "archive"]


def test_exclude_dirs_nested_path(vault: Path) -> None:
    s = Settings(vault_path=vault, exclude_dirs="projects/drafts")
    assert s.exclude_dirs == ["projects/drafts"]


# ---------------------------------------------------------------------------
# _is_excluded helper
# ---------------------------------------------------------------------------


def test_is_excluded_matches_top_level_dir() -> None:
    assert _is_excluded(Path("journal/2024/note.md"), ["journal"]) is True


def test_is_excluded_does_not_match_sibling_dir() -> None:
    assert _is_excluded(Path("journal_archive/note.md"), ["journal"]) is False


def test_is_excluded_does_not_match_nested_dir() -> None:
    """'journal' should NOT exclude 'project/journal/note.md'."""
    assert _is_excluded(Path("project/journal/note.md"), ["journal"]) is False


def test_is_excluded_matches_nested_exclude_path() -> None:
    assert _is_excluded(Path("projects/drafts/idea.md"), ["projects/drafts"]) is True


def test_is_excluded_no_exclusions() -> None:
    assert _is_excluded(Path("journal/note.md"), []) is False


def test_is_excluded_multiple_dirs() -> None:
    excls = ["journal", "archive"]
    assert _is_excluded(Path("archive/old.md"), excls) is True
    assert _is_excluded(Path("notes/keep.md"), excls) is False


# ---------------------------------------------------------------------------
# discover_notes (Obsidian) respects exclusions
# ---------------------------------------------------------------------------


@pytest.fixture()
def obsidian_vault(tmp_path: Path) -> Path:
    (tmp_path / "Alpha.md").write_text("# Alpha")
    (tmp_path / "Beta.md").write_text("# Beta")

    journal = tmp_path / "journal"
    journal.mkdir()
    (journal / "2024-01-01.md").write_text("# Day")

    private = tmp_path / "private"
    private.mkdir()
    (private / "secret.md").write_text("# Secret")

    nested = tmp_path / "projects" / "drafts"
    nested.mkdir(parents=True)
    (nested / "idea.md").write_text("# Idea")

    return tmp_path


def test_discover_notes_without_exclusions(obsidian_vault: Path) -> None:
    paths = discover_notes(obsidian_vault)
    stems = {p.stem for p in paths}
    assert {"Alpha", "Beta", "2024-01-01", "secret", "idea"} == stems


def test_discover_notes_excludes_single_dir(obsidian_vault: Path) -> None:
    paths = discover_notes(obsidian_vault, exclude_dirs=["journal"])
    stems = {p.stem for p in paths}
    assert "2024-01-01" not in stems
    assert "Alpha" in stems


def test_discover_notes_excludes_multiple_dirs(obsidian_vault: Path) -> None:
    paths = discover_notes(obsidian_vault, exclude_dirs=["journal", "private"])
    stems = {p.stem for p in paths}
    assert "2024-01-01" not in stems
    assert "secret" not in stems
    assert "Alpha" in stems
    assert "Beta" in stems


def test_discover_notes_excludes_nested_path(obsidian_vault: Path) -> None:
    paths = discover_notes(obsidian_vault, exclude_dirs=["projects/drafts"])
    stems = {p.stem for p in paths}
    assert "idea" not in stems
    assert "Alpha" in stems


def test_discover_notes_empty_exclusions_same_as_none(obsidian_vault: Path) -> None:
    assert discover_notes(obsidian_vault, []) == discover_notes(obsidian_vault, None)


# ---------------------------------------------------------------------------
# _discover_logseq_paths respects exclusions
# ---------------------------------------------------------------------------


@pytest.fixture()
def logseq_vault(tmp_path: Path) -> Path:
    pages = tmp_path / "pages"
    pages.mkdir()
    (pages / "Alpha.md").write_text("# Alpha")

    journals = tmp_path / "journals"
    journals.mkdir()
    (journals / "2024_01_01.md").write_text("# Day")

    return tmp_path


def test_logseq_discover_excludes_dir(logseq_vault: Path) -> None:
    paths = _discover_logseq_paths(logseq_vault, exclude_dirs=["journals"])
    stems = {p.stem for p in paths}
    assert "2024_01_01" not in stems
    assert "Alpha" in stems


def test_logseq_discover_no_exclusions(logseq_vault: Path) -> None:
    paths = _discover_logseq_paths(logseq_vault)
    stems = {p.stem for p in paths}
    assert {"Alpha", "2024_01_01"} == stems
