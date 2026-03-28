"""
Tests for vault/obsidian.py — file discovery, note parsing, section writing,
and VaultReader Protocol conformance.

All tests operate on a temporary directory so nothing touches a real vault.
"""

from pathlib import Path

import pytest

from rhizome.vault import (
    RELATED_NOTES_HEADER,
    RHIZOME_END,
    RHIZOME_START,
    Note,
    VaultReader,
    build_related_section,
    discover_notes,
    has_managed_section,
    parse_note,
    remove_related_section,
    write_related_notes,
)
from rhizome.vault.obsidian import (
    ObsidianVaultReader,
    _strip_frontmatter,
    _strip_wikilinks,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def vault(tmp_path: Path) -> Path:
    """Minimal vault with a few notes and one hidden directory."""
    (tmp_path / ".obsidian").mkdir()
    (tmp_path / ".obsidian" / "workspace.json").write_text("{}")

    (tmp_path / "Alpha.md").write_text("# Alpha\nSome content about alpha.")
    (tmp_path / "Beta.md").write_text("# Beta\nRelated to [[Alpha]] and more.")

    subdir = tmp_path / "subdir"
    subdir.mkdir()
    (subdir / "Gamma.md").write_text("# Gamma\nDeep note.")

    return tmp_path


# ---------------------------------------------------------------------------
# discover_notes
# ---------------------------------------------------------------------------

def test_discover_notes_finds_markdown_files(vault: Path) -> None:
    notes = discover_notes(vault)
    names = {p.name for p in notes}
    assert names == {"Alpha.md", "Beta.md", "Gamma.md"}


def test_discover_notes_excludes_hidden_dirs(vault: Path) -> None:
    paths = discover_notes(vault)
    assert not any(".obsidian" in str(p) for p in paths)


def test_discover_notes_is_sorted(vault: Path) -> None:
    paths = discover_notes(vault)
    assert paths == sorted(paths)


def test_discover_notes_empty_vault(tmp_path: Path) -> None:
    assert discover_notes(tmp_path) == []


def test_discover_notes_excludes_rhizome_backups(tmp_path: Path) -> None:
    (tmp_path / "Real.md").write_text("content")
    backup_dir = tmp_path / ".rhizome_backups" / "backup_20240101_120000"
    backup_dir.mkdir(parents=True)
    (backup_dir / "Real.md").write_text("old content")

    paths = discover_notes(tmp_path)
    assert all(".rhizome_backups" not in str(p) for p in paths)
    assert len(paths) == 1


# ---------------------------------------------------------------------------
# _strip_frontmatter
# ---------------------------------------------------------------------------

def test_strip_frontmatter_removes_yaml_block() -> None:
    text = "---\ntitle: Test\ntags: [a, b]\n---\n\nBody content."
    assert _strip_frontmatter(text) == "Body content."


def test_strip_frontmatter_no_frontmatter() -> None:
    text = "Just a plain body."
    assert _strip_frontmatter(text) == "Just a plain body."


def test_strip_frontmatter_preserves_body_with_triple_dash() -> None:
    text = "---\ntitle: X\n---\nLine one.\n---\nNot frontmatter."
    result = _strip_frontmatter(text)
    assert "title" not in result
    assert "Line one" in result


# ---------------------------------------------------------------------------
# _strip_wikilinks
# ---------------------------------------------------------------------------

def test_strip_wikilinks_preserves_link_text() -> None:
    assert _strip_wikilinks("See [[Alpha]] for more.") == "See Alpha for more."


def test_strip_wikilinks_handles_aliased_links() -> None:
    result = _strip_wikilinks("Check [[Alpha|the alpha note]] here.")
    assert result == "Check Alpha here."


def test_strip_wikilinks_multiple_links() -> None:
    result = _strip_wikilinks("[[A]] and [[B]] are related.")
    assert result == "A and B are related."


# ---------------------------------------------------------------------------
# parse_note — Note fields use .body and .raw (not .clean_text / .raw_body)
# ---------------------------------------------------------------------------

def test_parse_note_basic(tmp_path: Path) -> None:
    f = tmp_path / "Test.md"
    f.write_text("# Test\n\nHello world.")
    note = parse_note(f)
    assert note is not None
    assert note.title == "Test"
    assert "Hello world" in note.body


def test_parse_note_strips_wikilinks_from_body(tmp_path: Path) -> None:
    f = tmp_path / "Test.md"
    f.write_text("Mentions [[Other]] note.")
    note = parse_note(f)
    assert note is not None
    assert "[[" not in note.body
    assert "Other" in note.body


def test_parse_note_returns_none_on_undecodable_file(tmp_path: Path) -> None:
    f = tmp_path / "Bad.md"
    import unittest.mock as mock

    with mock.patch(
        "rhizome.vault.obsidian.Path.read_text",
        side_effect=UnicodeDecodeError("x", b"", 0, 1, "x"),
    ):
        result = parse_note(f)
    assert result is None


def test_parse_note_strips_frontmatter_from_body(tmp_path: Path) -> None:
    f = tmp_path / "Note.md"
    f.write_text("---\ntitle: Note\n---\n\nReal content here.")
    note = parse_note(f)
    assert note is not None
    assert "title:" not in note.body
    assert "Real content" in note.body


def test_parse_note_raw_is_original_content(tmp_path: Path) -> None:
    original = "---\ntitle: Note\n---\n\n[[Link]] content."
    f = tmp_path / "Note.md"
    f.write_text(original)
    note = parse_note(f)
    assert note is not None
    assert note.raw == original


# ---------------------------------------------------------------------------
# build_related_section
# ---------------------------------------------------------------------------

def test_build_related_section_format() -> None:
    section = build_related_section(["Alpha", "Beta"])
    assert RHIZOME_START in section
    assert RELATED_NOTES_HEADER in section
    assert "- [[Alpha]]" in section
    assert "- [[Beta]]" in section
    assert RHIZOME_END in section
    # sentinels must wrap the header and links
    assert section.index(RHIZOME_START) < section.index(RELATED_NOTES_HEADER)
    assert section.index(RELATED_NOTES_HEADER) < section.index(RHIZOME_END)


def test_build_related_section_accepts_custom_header() -> None:
    section = build_related_section(["Alpha"], header="## Suggested Links")
    assert "## Suggested Links" in section
    assert RELATED_NOTES_HEADER not in section


# ---------------------------------------------------------------------------
# write_related_notes / idempotency
# ---------------------------------------------------------------------------

def test_write_related_notes_appends_section(tmp_path: Path) -> None:
    f = tmp_path / "Note.md"
    f.write_text("# Note\n\nOriginal content.")
    note = Note(path=f, title="Note", raw=f.read_text(), body="Original content.")

    write_related_notes(note, ["Alpha", "Beta"])

    content = f.read_text()
    assert RELATED_NOTES_HEADER in content
    assert "[[Alpha]]" in content
    assert "Original content." in content


def test_write_related_notes_uses_custom_header(tmp_path: Path) -> None:
    f = tmp_path / "Note.md"
    f.write_text("# Note\n\nOriginal content.")
    note = Note(path=f, title="Note", raw=f.read_text(), body="Original content.")

    write_related_notes(note, ["Alpha"], header="## Suggested Links")

    content = f.read_text()
    assert "## Suggested Links" in content



def test_write_related_notes_dry_run_does_not_write(tmp_path: Path) -> None:
    original = "# Note\n\nContent."
    f = tmp_path / "Note.md"
    f.write_text(original)
    note = Note(path=f, title="Note", raw=original, body="Content.")

    write_related_notes(note, ["Alpha"], dry_run=True)

    assert f.read_text() == original


# ---------------------------------------------------------------------------
# remove_related_section
# ---------------------------------------------------------------------------

def test_remove_related_section_removes_legacy_block(tmp_path: Path) -> None:
    """Legacy format (no sentinels) is still removed — migration path."""
    content = "# Note\n\nBody.\n\n## Related Notes\n\n- [[Alpha]]\n"
    f = tmp_path / "Note.md"
    f.write_text(content)

    changed = remove_related_section(f)
    assert changed is True
    assert RELATED_NOTES_HEADER not in f.read_text()
    assert "Body." in f.read_text()


def test_remove_related_section_removes_sentinel_block(tmp_path: Path) -> None:
    """Current sentinel format is removed cleanly."""
    section = build_related_section(["Alpha"])
    content = f"# Note\n\nBody.\n\n{section}\n"
    f = tmp_path / "Note.md"
    f.write_text(content)

    changed = remove_related_section(f)
    assert changed is True
    result = f.read_text()
    assert RHIZOME_START not in result
    assert RELATED_NOTES_HEADER not in result
    assert "Body." in result


def test_write_related_notes_replaces_legacy_section(tmp_path: Path) -> None:
    """Writing over a legacy (no-sentinel) section migrates it to sentinel format."""
    initial = "# Note\n\nContent.\n\n## Related Notes\n\n- [[OldLink]]\n"
    f = tmp_path / "Note.md"
    f.write_text(initial)
    note = Note(path=f, title="Note", raw=initial, body="Content.")

    write_related_notes(note, ["NewLink"])

    content = f.read_text()
    assert "[[NewLink]]" in content
    assert "[[OldLink]]" not in content
    assert RHIZOME_START in content
    assert content.count(RELATED_NOTES_HEADER) == 1


def test_has_managed_section_detects_sentinel_format(tmp_path: Path) -> None:
    section = build_related_section(["Alpha"])
    f = tmp_path / "Note.md"
    f.write_text(f"# Note\n\nBody.\n\n{section}\n")
    assert has_managed_section(f) is True


def test_has_managed_section_detects_legacy_format(tmp_path: Path) -> None:
    f = tmp_path / "Note.md"
    f.write_text("# Note\n\nBody.\n\n## Related Notes\n\n- [[Alpha]]\n")
    assert has_managed_section(f) is True


def test_has_managed_section_false_when_absent(tmp_path: Path) -> None:
    f = tmp_path / "Note.md"
    f.write_text("# Note\n\nNo managed section here.\n")
    assert has_managed_section(f) is False


def test_remove_related_section_noop_when_absent(tmp_path: Path) -> None:
    content = "# Note\n\nNo generated section here.\n"
    f = tmp_path / "Note.md"
    f.write_text(content)

    changed = remove_related_section(f)
    assert changed is False
    assert f.read_text() == content


# ---------------------------------------------------------------------------
# VaultReader Protocol conformance
# ---------------------------------------------------------------------------

def test_obsidian_reader_satisfies_protocol(tmp_path: Path) -> None:
    """ObsidianVaultReader must satisfy the VaultReader Protocol."""
    reader = ObsidianVaultReader(tmp_path)
    assert isinstance(reader, VaultReader), (
        "ObsidianVaultReader does not implement the VaultReader Protocol"
    )


def test_mock_reader_satisfies_protocol() -> None:
    """
    Any class that implements the four required methods is accepted as a VaultReader.

    This test documents the minimum surface required to write a third-party adapter.
    """
    from collections.abc import Iterator

    class MinimalAdapter:
        def discover(self) -> Iterator[Note]:
            return iter([])

        def write_links(self, note: Note, links: list[str]) -> None:
            pass

        def clean_links(self, note: Note) -> None:
            pass

        def app_name(self) -> str:
            return "Mock"

    adapter = MinimalAdapter()
    assert isinstance(adapter, VaultReader), (
        "A minimal four-method class should satisfy the VaultReader Protocol"
    )


def test_incomplete_class_does_not_satisfy_protocol() -> None:
    """A class missing any required method is not a VaultReader."""

    class Incomplete:
        def discover(self) -> None:
            pass
        # missing write_links, clean_links, app_name

    assert not isinstance(Incomplete(), VaultReader)
