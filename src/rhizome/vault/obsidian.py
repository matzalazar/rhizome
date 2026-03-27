"""
Obsidian vault adapter.

Provides both:
  - Module-level functions used directly by pipeline.py and tests
  - ObsidianVaultReader class implementing the VaultReader Protocol

Obsidian uses [[wikilinks]] for cross-note references and YAML frontmatter
for metadata.  The generated links section is appended under "## Related Notes"
so Obsidian's graph view picks them up automatically.
"""

import re
from collections.abc import Iterator
from pathlib import Path

from loguru import logger

from .base import Note, VaultReader

# Directories that Obsidian (and common VCS tools) use for internal state.
# We never want to embed or link files inside these.
_HIDDEN_DIR_NAMES: frozenset[str] = frozenset(
    {".obsidian", ".git", ".trash", ".archive", ".stversions", ".rhizome_backups"}
)

# Sentinel marking the start of the auto-generated section.
# Placed here so pipeline.py and vault/__init__.py can reference it.
RELATED_NOTES_HEADER = "## Related Notes"

# HTML comment sentinels that wrap the managed block.
# Invisible in Obsidian's rendered view; make the pattern unambiguous.
RHIZOME_START = "<!-- rhizome:start -->"
RHIZOME_END = "<!-- rhizome:end -->"

# Matches the sentinel-wrapped block written by current versions of rhizome.
_RELATED_SECTION_RE = re.compile(
    r"\n?" + re.escape(RHIZOME_START) + r".*?" + re.escape(RHIZOME_END) + r"\n?",
    re.DOTALL,
)

# Matches the bare-header block written by rhizome < sentinel era.
# Kept for migration: write_related_notes and remove_related_section strip
# both formats so vaults updated in-place are handled transparently.
_LEGACY_SECTION_RE = re.compile(
    r"\n?" + re.escape(RELATED_NOTES_HEADER) + r".*$",
    re.DOTALL,
)

# YAML/TOML frontmatter fenced with --- or +++
_FRONTMATTER_RE = re.compile(r"^(?:---|\+\+\+).*?(?:---|\+\+\+)\s*", re.DOTALL)

# [[wikilinks]] with optional display alias [[target|display]]
_WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_hidden(path: Path) -> bool:
    """True if any component of *path* is in the hidden-directory blocklist."""
    return any(part in _HIDDEN_DIR_NAMES for part in path.parts)


def _is_excluded(rel_path: Path, exclude_dirs: list[str]) -> bool:
    """True if *rel_path* falls under any of the user-specified directory prefixes."""
    for excl in exclude_dirs:
        try:
            rel_path.relative_to(excl)
            return True
        except ValueError:
            continue
    return False


def _is_included(rel_path: Path, include_dirs: list[str]) -> bool:
    """True if *rel_path* falls under any of the user-specified include prefixes.

    When *include_dirs* is empty, every path is considered included (opt-in
    whitelist: no filter means no restriction).
    """
    if not include_dirs:
        return True
    for incl in include_dirs:
        try:
            rel_path.relative_to(incl)
            return True
        except ValueError:
            continue
    return False


def _strip_frontmatter(text: str) -> str:
    return _FRONTMATTER_RE.sub("", text, count=1).strip()


def _strip_wikilinks(text: str) -> str:
    # Preserve the link target text so prose remains coherent during embedding.
    return _WIKILINK_RE.sub(r"\1", text)


# ---------------------------------------------------------------------------
# Module-level functions (used by pipeline.py and tests)
# ---------------------------------------------------------------------------

def discover_notes(
    vault_path: Path,
    exclude_dirs: list[str] | None = None,
    include_dirs: list[str] | None = None,
) -> list[Path]:
    """
    Recursively find all .md files under *vault_path*, applying hidden-dir
    filtering and any user-specified inclusion/exclusion rules.

    *include_dirs* — whitelist of directory prefixes (relative to *vault_path*).
    When provided, only files under these directories are considered.
    When empty or None, all directories are in scope (default behaviour).

    *exclude_dirs* — blacklist applied after *include_dirs*.  Any file whose
    path starts with one of these prefixes is removed from the result.

    Both lists use prefix matching via Path.relative_to(), so "journal"
    matches "journal/note.md" but not "project/journal/note.md".
    Returns paths sorted for deterministic ordering across runs.
    """
    _include = include_dirs or []
    _exclude = exclude_dirs or []
    notes = [
        p for p in vault_path.rglob("*.md")
        if p.is_file()
        and not _is_hidden(p.relative_to(vault_path))
        and _is_included(p.relative_to(vault_path), _include)
        and not _is_excluded(p.relative_to(vault_path), _exclude)
    ]
    notes.sort()
    logger.info(f"Discovered {len(notes)} markdown files under {vault_path}")
    return notes


def parse_note(path: Path) -> Note | None:
    """
    Read and parse a single note.  Returns None if the file cannot be decoded.

    Tries UTF-8 first, then latin-1 to cover common mixed-encoding vaults.
    """
    raw: str | None = None
    for encoding in ("utf-8", "latin-1"):
        try:
            raw = path.read_text(encoding=encoding)
            break
        except UnicodeDecodeError:
            continue

    if raw is None:
        logger.warning(f"Could not decode {path} — skipping")
        return None

    title = path.stem
    body = _strip_wikilinks(_strip_frontmatter(raw))

    return Note(path=path, title=title, body=body, raw=raw)


def parse_notes(paths: list[Path]) -> list[Note]:
    """Parse a batch of paths, silently dropping any that fail decoding."""
    notes: list[Note] = []
    for path in paths:
        note = parse_note(path)
        if note is not None:
            notes.append(note)
    return notes


def build_related_section(linked_titles: list[str]) -> str:
    """
    Render the auto-generated section as a markdown string.

    Each entry is a [[wikilink]] on its own bullet so Obsidian's graph view
    registers the connection.  The block is wrapped in HTML comment sentinels
    so clean operations can target it unambiguously without risking false
    matches against organic user content.
    """
    lines = [RHIZOME_START, RELATED_NOTES_HEADER, ""]
    lines.extend(f"- [[{title}]]" for title in linked_titles)
    lines.append(RHIZOME_END)
    return "\n".join(lines)


def _strip_managed_section(content: str) -> str:
    """Remove any rhizome-managed block (sentinel or legacy format)."""
    content = _RELATED_SECTION_RE.sub("", content)
    content = _LEGACY_SECTION_RE.sub("", content)
    return content


def write_related_notes(note: Note, linked_titles: list[str], dry_run: bool = False) -> None:
    """
    Append (or replace) the '## Related Notes' section in *note*.

    Idempotent: running this twice with the same links produces the same file.
    Only the auto-generated block is touched; all content above it is preserved.
    """
    content_without_section = _strip_managed_section(note.raw)
    new_section = build_related_section(linked_titles)
    updated_content = content_without_section.rstrip("\n") + "\n\n" + new_section + "\n"

    if dry_run:
        logger.info(f"[DRY RUN] Would update {note.path.name}:")
        for title in linked_titles:
            logger.info(f"  -> [[{title}]]")
        return

    note.path.write_text(updated_content, encoding="utf-8")
    logger.debug(f"Updated {note.path.name} with {len(linked_titles)} related links")


def count_managed_links(note: Note) -> int:
    """
    Count [[wikilinks]] inside the managed Related Notes section of *note*.

    Returns 0 if the note has no managed section.  Only the sentinel-wrapped
    block is inspected — manual links elsewhere in the note are not counted.
    """
    match = _RELATED_SECTION_RE.search(note.raw)
    if not match:
        return 0
    return len(_WIKILINK_RE.findall(match.group(0)))


def has_managed_section(path: Path) -> bool:
    """
    Return True if *path* contains a rhizome-managed block (either format).

    Used by the clean preview to identify affected notes without modifying them.
    """
    for encoding in ("utf-8", "latin-1"):
        try:
            content = path.read_text(encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        return False

    return bool(
        _RELATED_SECTION_RE.search(content) or _LEGACY_SECTION_RE.search(content)
    )


def remove_related_section(path: Path) -> bool:
    """
    Remove the auto-generated section from the file at *path* if present.

    Returns True if the file was modified, False if there was nothing to remove.
    """
    for encoding in ("utf-8", "latin-1"):
        try:
            content = path.read_text(encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        logger.warning(f"Could not decode {path} during clean — skipping")
        return False

    cleaned = _strip_managed_section(content)
    if cleaned == content:
        return False

    path.write_text(cleaned.rstrip("\n") + "\n", encoding="utf-8")
    return True


# ---------------------------------------------------------------------------
# VaultReader implementation
# ---------------------------------------------------------------------------

class ObsidianVaultReader:
    """
    VaultReader adapter for Obsidian vaults.

    Wraps the module-level functions above so the pipeline can treat
    any vault app uniformly through the VaultReader Protocol.
    """

    def __init__(
        self,
        vault_path: Path,
        dry_run: bool = False,
        exclude_dirs: list[str] | None = None,
        include_dirs: list[str] | None = None,
    ) -> None:
        self._vault_path = vault_path
        self._dry_run = dry_run
        self._exclude_dirs = exclude_dirs or []
        self._include_dirs = include_dirs or []

    def discover(self) -> Iterator[Note]:
        paths = discover_notes(
            self._vault_path,
            exclude_dirs=self._exclude_dirs,
            include_dirs=self._include_dirs,
        )
        yield from parse_notes(paths)

    def write_links(self, note: Note, links: list[str]) -> None:
        write_related_notes(note, links, dry_run=self._dry_run)

    def clean_links(self, note: Note) -> None:
        remove_related_section(note.path)

    def app_name(self) -> str:
        return "Obsidian"


# Verify structural conformance at import time so misconfigured adapters
# are caught immediately rather than at the first pipeline run.
assert isinstance(ObsidianVaultReader(Path(".")), VaultReader), (
    "ObsidianVaultReader does not satisfy the VaultReader Protocol"
)
