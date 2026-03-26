"""
Vault backup and restore.

All backup artefacts live in a single sibling directory so they are never
inside the vault itself and cannot be discovered as notes.

Layout:
  {VAULT_PATH}/../.rhizome_backups/
    backup_20240315_142301/
      <mirror of vault contents>
      backup_manifest.json
    backup_20240315_093012/
      ...

The manifest records enough metadata to display meaningful listings and to
verify integrity before a restore.
"""

import json
import shutil
from datetime import datetime, timezone
from importlib.metadata import version as pkg_version
from pathlib import Path

from loguru import logger

# Directory name relative to vault's *parent* — placed outside the vault so
# vault discovery never sees it.
BACKUP_DIR_NAME = ".rhizome_backups"
MANIFEST_FILENAME = "backup_manifest.json"


def _backup_root(vault_path: Path) -> Path:
    """Resolve the .rhizome_backups directory for a given vault path."""
    return vault_path.parent / BACKUP_DIR_NAME


def _count_md_files(path: Path) -> int:
    return sum(1 for _ in path.rglob("*.md") if _.is_file())


def _rhizome_version() -> str:
    try:
        return pkg_version("rhizome")
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------

def create_backup(vault_path: Path) -> Path:
    """
    Copy *vault_path* into a timestamped backup directory.

    Returns the path of the newly created backup.
    Raises RuntimeError (with a human-readable message) if anything goes wrong
    so the pipeline can abort cleanly without an unhandled traceback.
    """
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_dir = _backup_root(vault_path) / f"backup_{timestamp}"

    logger.info(f"Creating backup at {backup_dir} …")

    try:
        backup_dir.parent.mkdir(parents=True, exist_ok=True)

        # Copy the entire vault tree preserving file metadata (timestamps, perms).
        shutil.copytree(
            src=vault_path,
            dst=backup_dir,
            # Ignore the backups directory itself if VAULT_PATH is a parent
            # that contains .rhizome_backups.
            ignore=shutil.ignore_patterns(BACKUP_DIR_NAME),
            copy_function=shutil.copy2,
        )
    except Exception as exc:
        # A partial backup is worse than no backup — remove the incomplete
        # directory before re-raising so the next run starts clean.
        if backup_dir.exists():
            shutil.rmtree(backup_dir, ignore_errors=True)
        raise RuntimeError(
            f"Backup failed: {exc}\n"
            "The pipeline was aborted. Fix the issue (disk space? permissions?) "
            "and try again."
        ) from exc

    note_count = _count_md_files(backup_dir)

    manifest = {
        "vault_path": str(vault_path),
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "note_count": note_count,
        "rhizome_version": _rhizome_version(),
    }
    (backup_dir / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    logger.success(f"Backup complete: {backup_dir} ({note_count} notes)")
    return backup_dir


# ---------------------------------------------------------------------------
# List
# ---------------------------------------------------------------------------

def list_backups(vault_path: Path) -> list[dict]:
    """
    Return metadata for all available backups, most recent first.

    Each entry is the parsed manifest dict augmented with a 'backup_dir' key.
    Directories without a valid manifest are silently skipped.
    """
    root = _backup_root(vault_path)
    if not root.exists():
        return []

    entries: list[dict] = []
    for candidate in sorted(root.iterdir(), reverse=True):
        if not candidate.is_dir():
            continue
        manifest_path = candidate / MANIFEST_FILENAME
        if not manifest_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["backup_dir"] = str(candidate)
            entries.append(manifest)
        except Exception:
            # Corrupt manifest — skip rather than crash.
            continue

    return entries


# ---------------------------------------------------------------------------
# Restore
# ---------------------------------------------------------------------------

def restore_backup(backup_dir: Path, vault_path: Path) -> None:
    """
    Overwrite *vault_path* with the contents of *backup_dir*.

    Uses dirs_exist_ok=True so the vault directory itself is not deleted
    (Obsidian may be watching it).  Files present in the vault but absent
    from the backup are left untouched — a restore brings back what was
    there, it does not wipe additions made since.
    """
    logger.info(f"Restoring {backup_dir} → {vault_path} …")

    try:
        shutil.copytree(
            src=backup_dir,
            dst=vault_path,
            dirs_exist_ok=True,
            # Do not restore the manifest file into the vault.
            ignore=shutil.ignore_patterns(MANIFEST_FILENAME),
            copy_function=shutil.copy2,
        )
    except Exception as exc:
        raise RuntimeError(f"Restore failed: {exc}") from exc

    logger.success(f"Restore complete. Vault is now at the state of {backup_dir.name}.")
