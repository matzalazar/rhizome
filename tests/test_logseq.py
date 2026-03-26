"""
Tests for vault/logseq.py — discovery, parsing, syntax stripping, and
VaultReader integration.

All tests use tmp_path so no real vault is touched.
"""

import unittest.mock as mock
from pathlib import Path

import pytest

from rhizome.vault import Note, VaultReader
from rhizome.vault.logseq import (
    LogseqVaultReader,
    _discover_logseq_paths,
    _extract_display_title,
    _page_stem_to_link_target,
    _parse_logseq_note,
    _strip_logseq_syntax,
)
from rhizome.vault.obsidian import RELATED_NOTES_HEADER

# A well-formed block reference UUID used across multiple tests.
_UUID = "550e8400-e29b-41d4-a716-446655440000"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def vault(tmp_path: Path) -> Path:
    """
    Minimal Logseq vault layout:

      pages/
        Alpha.md
        Beta.md
        namespace___subpage.md
      journals/
        2024_03_15.md
      logseq/           <- system dir, must be skipped
        config.edn
      .logseq/          <- system dir, must be skipped
        metadata.edn
      bak/              <- backup dir, must be skipped
        pages/
          Alpha.md
      assets/           <- attachment dir, must be skipped
        image.md        <- .md file inside assets, should be ignored
    """
    pages = tmp_path / "pages"
    pages.mkdir()
    (pages / "Alpha.md").write_text(
        "title:: Alpha Note\ntags:: test\n- First block.\n- Second block."
    )
    (pages / "Beta.md").write_text(
        "- Beta content referencing [[Alpha Note]].\n"
        f"- See also (({_UUID}))."
    )
    (pages / "namespace___subpage.md").write_text("- Namespaced content.")

    journals = tmp_path / "journals"
    journals.mkdir()
    (journals / "2024_03_15.md").write_text("- Daily note entry.")

    # System directories — every .md inside must be excluded.
    logseq_dir = tmp_path / "logseq"
    logseq_dir.mkdir()
    (logseq_dir / "config.edn").write_text("{}")

    dotlogseq = tmp_path / ".logseq"
    dotlogseq.mkdir()
    (dotlogseq / "metadata.edn").write_text("{}")

    bak = tmp_path / "bak" / "pages"
    bak.mkdir(parents=True)
    (bak / "Alpha.md").write_text("old content")

    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "image.md").write_text("not a note")

    return tmp_path


# ---------------------------------------------------------------------------
# _page_stem_to_link_target
# ---------------------------------------------------------------------------

def test_stem_plain_name() -> None:
    assert _page_stem_to_link_target("my-note") == "my-note"


def test_stem_namespaced_single_level() -> None:
    assert _page_stem_to_link_target("namespace___subpage") == "namespace/subpage"


def test_stem_namespaced_deep() -> None:
    assert _page_stem_to_link_target("a___b___c") == "a/b/c"


def test_stem_no_separator_unchanged() -> None:
    assert _page_stem_to_link_target("Zettelkasten") == "Zettelkasten"


def test_stem_spaces_unchanged() -> None:
    # Logseq allows spaces in filenames; they survive as-is.
    assert _page_stem_to_link_target("My Page Name") == "My Page Name"


# ---------------------------------------------------------------------------
# _strip_logseq_syntax
# ---------------------------------------------------------------------------

def test_strip_properties() -> None:
    text = "title:: My Title\ntags:: a, b\n- Actual content."
    result = _strip_logseq_syntax(text)
    assert "title::" not in result
    assert "tags::" not in result
    assert "Actual content." in result


def test_strip_block_references() -> None:
    text = f"- See (({_UUID})) for details."
    result = _strip_logseq_syntax(text)
    assert "((" not in result
    assert "for details." in result


def test_strip_embed_macro() -> None:
    result = _strip_logseq_syntax("{{embed [[Some Page]]}}")
    # The macro wrapper and its wikilink contents must both be gone.
    assert "{{" not in result
    assert "[[" not in result
    assert "Some Page" not in result


def test_strip_query_macro() -> None:
    result = _strip_logseq_syntax("{{query (and [[tag]] (page-property :type))}}")
    assert "{{" not in result


def test_strip_wikilinks_preserves_text() -> None:
    result = _strip_logseq_syntax("Related to [[Alpha Note]] and [[Beta]].")
    assert "[[" not in result
    assert "Alpha Note" in result
    assert "Beta" in result


def test_strip_aliased_wikilinks() -> None:
    result = _strip_logseq_syntax("See [[Alpha Note|the alpha page]].")
    assert "[[" not in result
    # Target text is preserved (alias is stripped, same as Obsidian).
    assert "Alpha Note" in result


def test_strip_outline_bullets() -> None:
    text = "- First item\n- Second item"
    result = _strip_logseq_syntax(text)
    assert "- " not in result
    assert "First item" in result
    assert "Second item" in result


def test_strip_nested_bullets_preserve_indentation() -> None:
    text = "- Parent\n  - Child\n    - Grandchild"
    result = _strip_logseq_syntax(text)
    # Bullet markers gone, indentation and text remain.
    assert "- " not in result
    assert "Parent" in result
    assert "Child" in result
    assert "Grandchild" in result


def test_strip_markdown_headers_preserved() -> None:
    text = "## Section\n- bullet"
    result = _strip_logseq_syntax(text)
    assert "## Section" in result


def test_strip_macro_before_wikilink_order() -> None:
    """
    {{embed [[Page]]}} must not leave a dangling [[Page]] after macro stripping.
    Macros are removed first, then wikilinks, so the inner [[...]] disappears
    with the macro rather than being processed by the wikilink pass.
    """
    result = _strip_logseq_syntax("{{embed [[Nested Page]]}}")
    assert "Nested Page" not in result


def test_strip_empty_string() -> None:
    assert _strip_logseq_syntax("") == ""


def test_strip_plain_prose_unchanged() -> None:
    text = "Just plain prose without any special syntax."
    assert _strip_logseq_syntax(text) == text


# ---------------------------------------------------------------------------
# _extract_display_title
# ---------------------------------------------------------------------------

def test_extract_title_from_property(tmp_path: Path) -> None:
    f = tmp_path / "my-note.md"
    f.write_text("title:: My Display Title\n- content")
    assert _extract_display_title(f, f.read_text()) == "My Display Title"


def test_extract_title_strips_whitespace(tmp_path: Path) -> None:
    f = tmp_path / "note.md"
    f.write_text("title::   Trimmed Title   \n- content")
    assert _extract_display_title(f, f.read_text()) == "Trimmed Title"


def test_extract_title_falls_back_to_stem(tmp_path: Path) -> None:
    f = tmp_path / "my-note.md"
    f.write_text("- No property here.")
    assert _extract_display_title(f, f.read_text()) == "my-note"


def test_extract_title_falls_back_with_namespace(tmp_path: Path) -> None:
    f = tmp_path / "ns___sub.md"
    f.write_text("- content")
    assert _extract_display_title(f, f.read_text()) == "ns/sub"


def test_extract_title_property_takes_priority_over_stem(tmp_path: Path) -> None:
    f = tmp_path / "ns___sub.md"
    f.write_text("title:: Custom Title\n- content")
    assert _extract_display_title(f, f.read_text()) == "Custom Title"


# ---------------------------------------------------------------------------
# _discover_logseq_paths
# ---------------------------------------------------------------------------

def test_discover_finds_pages_and_journals(vault: Path) -> None:
    paths = _discover_logseq_paths(vault)
    names = {p.name for p in paths}
    assert "Alpha.md" in names
    assert "Beta.md" in names
    assert "namespace___subpage.md" in names
    assert "2024_03_15.md" in names


def test_discover_skips_logseq_dir(vault: Path) -> None:
    paths = _discover_logseq_paths(vault)
    assert not any("logseq" in str(p.relative_to(vault)) for p in paths)


def test_discover_skips_dotlogseq_dir(vault: Path) -> None:
    paths = _discover_logseq_paths(vault)
    assert not any(".logseq" in str(p.relative_to(vault)) for p in paths)


def test_discover_skips_bak_dir(vault: Path) -> None:
    paths = _discover_logseq_paths(vault)
    assert not any("bak" in str(p.relative_to(vault)) for p in paths)


def test_discover_skips_assets_dir(vault: Path) -> None:
    paths = _discover_logseq_paths(vault)
    assert not any("assets" in str(p.relative_to(vault)) for p in paths)


def test_discover_skips_rhizome_backups(tmp_path: Path) -> None:
    pages = tmp_path / "pages"
    pages.mkdir()
    (pages / "Note.md").write_text("- content")
    backup = tmp_path / ".rhizome_backups" / "backup_20240101_120000" / "pages"
    backup.mkdir(parents=True)
    (backup / "Note.md").write_text("old content")

    paths = _discover_logseq_paths(tmp_path)
    assert all(".rhizome_backups" not in str(p) for p in paths)
    assert len(paths) == 1


def test_discover_returns_sorted(vault: Path) -> None:
    paths = _discover_logseq_paths(vault)
    assert paths == sorted(paths)


def test_discover_empty_vault(tmp_path: Path) -> None:
    assert _discover_logseq_paths(tmp_path) == []


# ---------------------------------------------------------------------------
# _parse_logseq_note
# ---------------------------------------------------------------------------

def test_parse_basic_note(tmp_path: Path) -> None:
    f = tmp_path / "my-note.md"
    f.write_text("- Hello world.")
    note = _parse_logseq_note(f)
    assert note is not None
    assert note.title == "my-note"
    assert "Hello world." in note.body


def test_parse_raw_is_original_content(tmp_path: Path) -> None:
    original = "title:: My Note\n- content with [[Link]]."
    f = tmp_path / "my-note.md"
    f.write_text(original)
    note = _parse_logseq_note(f)
    assert note is not None
    assert note.raw == original


def test_parse_body_strips_properties(tmp_path: Path) -> None:
    f = tmp_path / "note.md"
    f.write_text("title:: My Title\ntags:: a, b\n- Real content.")
    note = _parse_logseq_note(f)
    assert note is not None
    assert "title::" not in note.body
    assert "tags::" not in note.body
    assert "Real content." in note.body


def test_parse_body_strips_block_refs(tmp_path: Path) -> None:
    f = tmp_path / "note.md"
    f.write_text(f"- See (({_UUID})).")
    note = _parse_logseq_note(f)
    assert note is not None
    assert "((" not in note.body


def test_parse_body_strips_macros(tmp_path: Path) -> None:
    f = tmp_path / "note.md"
    f.write_text("- {{embed [[Other Page]]}}")
    note = _parse_logseq_note(f)
    assert note is not None
    assert "{{" not in note.body
    assert "Other Page" not in note.body


def test_parse_title_is_link_target_not_display_title(tmp_path: Path) -> None:
    """note.title must be the filename stem (for [[wikilink]] generation),
    not the title:: property value."""
    f = tmp_path / "my-note.md"
    f.write_text("title:: My Display Title\n- content")
    note = _parse_logseq_note(f)
    assert note is not None
    assert note.title == "my-note"


def test_parse_title_for_namespaced_page(tmp_path: Path) -> None:
    f = tmp_path / "ns___sub.md"
    f.write_text("- content")
    note = _parse_logseq_note(f)
    assert note is not None
    assert note.title == "ns/sub"


def test_parse_display_title_prepended_when_different(tmp_path: Path) -> None:
    """When title:: differs from the filename stem, it is prepended to body
    so the embedding model has the full display name in context."""
    f = tmp_path / "my-note.md"
    f.write_text("title:: My Custom Title\n- Some content.")
    note = _parse_logseq_note(f)
    assert note is not None
    assert note.body.startswith("My Custom Title")


def test_parse_display_title_not_duplicated_when_same(tmp_path: Path) -> None:
    """When the filename already equals what _extract_display_title returns
    (no title:: property), it must not be prepended twice."""
    f = tmp_path / "my-note.md"
    f.write_text("- Just content, no title property.")
    note = _parse_logseq_note(f)
    assert note is not None
    # "my-note" should not appear twice in body
    assert note.body.count("my-note") <= 1


def test_parse_returns_none_on_undecodable_file(tmp_path: Path) -> None:
    f = tmp_path / "bad.md"
    with mock.patch(
        "rhizome.vault.logseq.Path.read_text",
        side_effect=UnicodeDecodeError("utf-8", b"", 0, 1, "reason"),
    ):
        result = _parse_logseq_note(f)
    assert result is None


# ---------------------------------------------------------------------------
# LogseqVaultReader — Protocol and integration
# ---------------------------------------------------------------------------

def test_logseq_reader_satisfies_protocol(tmp_path: Path) -> None:
    reader = LogseqVaultReader(tmp_path)
    assert isinstance(reader, VaultReader)


def test_app_name(tmp_path: Path) -> None:
    assert LogseqVaultReader(tmp_path).app_name() == "Logseq"


def test_discover_yields_notes(vault: Path) -> None:
    reader = LogseqVaultReader(vault)
    notes = list(reader.discover())
    assert len(notes) == 4  # Alpha, Beta, namespace/subpage, journal


def test_discover_note_titles_are_stems(vault: Path) -> None:
    reader = LogseqVaultReader(vault)
    notes = list(reader.discover())
    stem_titles = {n.title for n in notes}
    assert "Alpha" in stem_titles
    assert "Beta" in stem_titles
    assert "namespace/subpage" in stem_titles
    assert "2024_03_15" in stem_titles


def test_write_links_appends_section(tmp_path: Path) -> None:
    f = tmp_path / "pages" / "Note.md"
    f.parent.mkdir()
    f.write_text("- Some content.")
    note = Note(path=f, title="Note", body="Some content.", raw=f.read_text())

    reader = LogseqVaultReader(tmp_path)
    reader.write_links(note, ["Alpha", "Beta"])

    content = f.read_text()
    assert RELATED_NOTES_HEADER in content
    assert "[[Alpha]]" in content
    assert "[[Beta]]" in content
    assert "Some content." in content


def test_write_links_is_idempotent(tmp_path: Path) -> None:
    f = tmp_path / "Note.md"
    f.write_text("- Content.")
    note = Note(path=f, title="Note", body="Content.", raw=f.read_text())

    reader = LogseqVaultReader(tmp_path)
    reader.write_links(note, ["Alpha"])
    # Update raw before second call (simulates pipeline re-reading the file).
    note = Note(path=f, title="Note", body="Content.", raw=f.read_text())
    reader.write_links(note, ["Alpha"])

    content = f.read_text()
    assert content.count(RELATED_NOTES_HEADER) == 1


def test_write_links_dry_run_does_not_write(tmp_path: Path) -> None:
    original = "- Content."
    f = tmp_path / "Note.md"
    f.write_text(original)
    note = Note(path=f, title="Note", body="Content.", raw=original)

    reader = LogseqVaultReader(tmp_path, dry_run=True)
    reader.write_links(note, ["Alpha"])

    assert f.read_text() == original


def test_clean_links_removes_section(tmp_path: Path) -> None:
    content = "- Content.\n\n## Related Notes\n\n- [[Alpha]]\n"
    f = tmp_path / "Note.md"
    f.write_text(content)
    note = Note(path=f, title="Note", body="Content.", raw=content)

    reader = LogseqVaultReader(tmp_path)
    reader.clean_links(note)

    result = f.read_text()
    assert RELATED_NOTES_HEADER not in result
    assert "Content." in result


def test_clean_links_noop_when_no_section(tmp_path: Path) -> None:
    original = "- Content with no generated section.\n"
    f = tmp_path / "Note.md"
    f.write_text(original)
    note = Note(path=f, title="Note", body="Content.", raw=original)

    reader = LogseqVaultReader(tmp_path)
    reader.clean_links(note)

    assert f.read_text() == original


def test_full_discover_write_clean_cycle(vault: Path) -> None:
    """End-to-end: discover notes, write links to one, then clean it."""
    reader = LogseqVaultReader(vault)
    notes = list(reader.discover())
    alpha = next(n for n in notes if n.title == "Alpha")

    reader.write_links(alpha, ["Beta", "namespace/subpage"])
    written = alpha.path.read_text()
    assert "[[Beta]]" in written
    assert "[[namespace/subpage]]" in written

    # Re-parse to get updated raw, then clean.
    alpha_updated = Note(
        path=alpha.path,
        title=alpha.title,
        body=alpha.body,
        raw=written,
    )
    reader.clean_links(alpha_updated)
    assert RELATED_NOTES_HEADER not in alpha.path.read_text()
