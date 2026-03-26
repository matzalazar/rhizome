"""
Core abstractions shared by all vault adapters.

VaultReader is the single interface any note-taking app adapter must satisfy.
Adding support for a new app (Logseq, Roam, Foam, …) means implementing this
Protocol — nothing else in the codebase needs to change.

Why a Protocol instead of an ABC?
  Structural subtyping lets third-party adapters satisfy the interface without
  depending on (or knowing about) this package.  It also makes mock
  implementations in tests trivial — no inheritance required.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass
class Note:
    """Parsed representation of a single note, app-agnostic."""

    path: Path
    title: str   # filename stem — used as the link target ([[title]])
    body: str    # clean text with frontmatter and existing links stripped — used for embedding
    raw: str     # original file content — used for write-back without data loss


@runtime_checkable
class VaultReader(Protocol):
    """
    Interface that all vault adapters must implement.

    Methods:
      discover()      -- yield every Note in the vault
      write_links()   -- add/replace the generated links section in a note
      clean_links()   -- remove the generated links section from a note
      app_name()      -- human-readable app identifier used in log messages
    """

    def discover(self) -> Iterator[Note]:
        """Yield all notes in the vault, skipping hidden/system directories."""
        ...

    def write_links(self, note: Note, links: list[str]) -> None:
        """
        Persist *links* as a generated section in *note*.

        The operation must be idempotent: calling it twice with the same
        links produces the same file as calling it once.
        """
        ...

    def clean_links(self, note: Note) -> None:
        """Remove the generated links section from *note*, if present."""
        ...

    def app_name(self) -> str:
        """Return a human-readable identifier, e.g. 'Obsidian' or 'Logseq'."""
        ...
