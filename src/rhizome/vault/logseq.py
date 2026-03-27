"""
Logseq vault adapter.

Logseq differences from Obsidian that affect parsing:

  Directory layout
    Notes live under pages/ (regular pages) and journals/ (daily notes).
    System directories — logseq/, .logseq/, bak/, assets/ — must be skipped.

  Properties instead of YAML frontmatter
    Logseq does not use --- fences.  Page-level metadata is expressed as
    inline properties at the top of the file:
        title:: My Page Title
        tags:: zettelkasten, writing
    These are regular content blocks and must be stripped before embedding
    so metadata does not dominate the semantic representation.

  Block references
    ((550e8400-e29b-41d4-a716-446655440000)) references a specific block
    by its UUID.  They carry no semantic content for our purposes.

  Macros and embeds
    {{embed [[Page]]}}, {{embed ((uuid))}}, {{query ...}} — all stripped.

  Outline format
    All content is structured as indented bullet lists with "- " markers.
    Markdown headers (##, ###) are also supported and kept as-is.

  Namespaced pages
    A page named "namespace/subpage" is stored on disk as
    "namespace___subpage.md" (triple underscore separator).
    The wikilink target must use the slash form: [[namespace/subpage]].

  Link format
    Page links use [[Page Name]] — identical to Obsidian.
    The generated ## Related Notes section is therefore the same format.
"""

import re
from collections.abc import Iterator
from pathlib import Path

from loguru import logger

from .base import Note, VaultReader
from .obsidian import _is_excluded, _is_included, remove_related_section, write_related_notes

# Logseq system directories — never contain user notes.
_HIDDEN_DIR_NAMES: frozenset[str] = frozenset(
    {".logseq", "logseq", "bak", "assets", ".git", ".rhizome_backups"}
)

# key:: value  (page properties — equivalent to Obsidian frontmatter)
_PROPERTY_RE = re.compile(r"^[\w-]+:: .*$", re.MULTILINE)

# ((550e8400-e29b-41d4-a716-446655440000))  block references
_BLOCK_REF_RE = re.compile(
    r"\(\([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\)\)"
)

# {{embed [[Page]]}}, {{query ...}}, and all other Logseq macros
_MACRO_RE = re.compile(r"\{\{[^}]*\}\}")

# [[wikilinks]] with optional alias [[target|display]]
_WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")

# Outline bullet markers: "  - content" → "  content"
# Preserves indentation so nested structure remains readable.
_OUTLINE_BULLET_RE = re.compile(r"^(\s*)- ", re.MULTILINE)

# title:: property for display title extraction
_TITLE_PROPERTY_RE = re.compile(r"^title::\s*(.+)$", re.MULTILINE)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_hidden(path: Path) -> bool:
    return any(part in _HIDDEN_DIR_NAMES for part in path.parts)


def _page_stem_to_link_target(stem: str) -> str:
    """
    Convert a Logseq filename stem to a wikilink target.

    Logseq stores namespaced pages as "namespace___subpage.md" on disk but
    the correct wikilink form is [[namespace/subpage]].
    """
    return stem.replace("___", "/")


def _strip_logseq_syntax(text: str) -> str:
    """
    Remove Logseq-specific markup before embedding.

    Order matters: macros are stripped before wikilinks so that
    {{embed [[Page]]}} does not leave a dangling [[Page]] fragment.
    """
    text = _MACRO_RE.sub("", text)          # {{...}} — macros, embeds, queries
    text = _PROPERTY_RE.sub("", text)       # key:: value — page properties
    text = _BLOCK_REF_RE.sub("", text)      # ((uuid)) — block references
    text = _WIKILINK_RE.sub(r"\1", text)    # [[target]] → target (keep text)
    text = _OUTLINE_BULLET_RE.sub(r"\1", text)  # "- item" → "item"
    return text.strip()


def _extract_display_title(path: Path, raw: str) -> str:
    """
    Return the page's display title.

    Priority: title:: property > filename stem (with namespace separator fixed).
    The display title is prepended to the body text so the embedding model
    has explicit access to it — this is especially useful for short notes
    whose content alone would not identify the topic.
    """
    match = _TITLE_PROPERTY_RE.search(raw)
    if match:
        return match.group(1).strip()
    return _page_stem_to_link_target(path.stem)


def _discover_logseq_paths(
    vault_path: Path,
    exclude_dirs: list[str] | None = None,
    include_dirs: list[str] | None = None,
) -> list[Path]:
    """
    Recursively find all .md files, skipping Logseq system directories and
    applying any user-specified inclusion/exclusion rules.

    Scans the entire vault tree (not just pages/ or journals/) so the adapter
    works whether VAULT_PATH points to the vault root or to a subdirectory.
    See obsidian.discover_notes() for full semantics of include/exclude.
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
    logger.info(f"Discovered {len(notes)} Logseq pages under {vault_path}")
    return notes


def _parse_logseq_note(path: Path) -> Note | None:
    """
    Read and parse a single Logseq page.

    Returns None if the file cannot be decoded (same encoding fallback as
    the Obsidian adapter: UTF-8 first, then latin-1).
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

    # The link target is the on-disk stem (with namespace separator normalised).
    # This is what other Logseq pages reference when they write [[title]].
    link_target = _page_stem_to_link_target(path.stem)

    # Prepend the display title so the embedding model has it in context.
    display_title = _extract_display_title(path, raw)
    clean_content = _strip_logseq_syntax(raw)
    body = (
        f"{display_title}\n{clean_content}".strip()
        if display_title != link_target
        else clean_content
    )

    return Note(path=path, title=link_target, body=body, raw=raw)


def _parse_logseq_notes(paths: list[Path]) -> list[Note]:
    notes: list[Note] = []
    for path in paths:
        note = _parse_logseq_note(path)
        if note is not None:
            notes.append(note)
    return notes


# ---------------------------------------------------------------------------
# VaultReader implementation
# ---------------------------------------------------------------------------

class LogseqVaultReader:
    """
    VaultReader adapter for Logseq vaults.

    Uses [[wikilinks]] (not block-refs) for the generated Related Notes section
    because Logseq resolves [[Page Name]] to the corresponding page file,
    making it the correct format for cross-page semantic links.

    The generated section is identical in structure to the Obsidian adapter:
        ## Related Notes
        - [[Page One]]
        - [[Page Two]]

    Logseq renders markdown headers and bullet lists natively in both the
    outline view and the document view, so this format works in both modes.
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
        paths = _discover_logseq_paths(
            self._vault_path,
            exclude_dirs=self._exclude_dirs,
            include_dirs=self._include_dirs,
        )
        yield from _parse_logseq_notes(paths)

    def write_links(self, note: Note, links: list[str]) -> None:
        # Delegates to the shared obsidian helper — the ## Related Notes
        # section format is the same for both apps.
        write_related_notes(note, links, dry_run=self._dry_run)

    def clean_links(self, note: Note) -> None:
        remove_related_section(note.path)

    def app_name(self) -> str:
        return "Logseq"


assert isinstance(LogseqVaultReader(Path(".")), VaultReader), (
    "LogseqVaultReader does not satisfy the VaultReader Protocol"
)
