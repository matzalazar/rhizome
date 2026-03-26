"""
Vault subpackage.

Public surface:
  Note            -- app-agnostic note dataclass
  VaultReader     -- Protocol that all adapters must satisfy
  get_vault_reader() -- factory: returns the right adapter for config.vault_app

The module-level Obsidian helper functions (discover_notes, parse_note, …)
are also re-exported here so that pipeline.py and tests can import them from
the shorter path `rhizome.vault` without knowing the internal module layout.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import Note, VaultReader
from .obsidian import (
    RELATED_NOTES_HEADER,
    RHIZOME_END,
    RHIZOME_START,
    ObsidianVaultReader,
    build_related_section,
    discover_notes,
    has_managed_section,
    parse_note,
    parse_notes,
    remove_related_section,
    write_related_notes,
)

if TYPE_CHECKING:
    from rhizome.config import Settings


def get_vault_reader(config: Settings) -> VaultReader:
    """
    Factory: return the VaultReader implementation for config.vault_app.

    This is the only place in the codebase that branches on vault_app —
    the pipeline and CLI never inspect the string directly.
    """
    app = config.vault_app.lower()

    if app == "obsidian":
        return ObsidianVaultReader(config.vault_path, dry_run=config.dry_run)

    if app == "logseq":
        from .logseq import LogseqVaultReader
        return LogseqVaultReader(config.vault_path, dry_run=config.dry_run)

    raise ValueError(
        f"Unknown VAULT_APP {config.vault_app!r}. Supported values: obsidian, logseq"
    )


__all__ = [
    "Note",
    "VaultReader",
    "ObsidianVaultReader",
    "get_vault_reader",
    # Re-exported helpers
    "RELATED_NOTES_HEADER",
    "RHIZOME_START",
    "RHIZOME_END",
    "discover_notes",
    "has_managed_section",
    "parse_note",
    "parse_notes",
    "build_related_section",
    "write_related_notes",
    "remove_related_section",
]
