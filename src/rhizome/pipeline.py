"""
Orchestration layer — connects vault, model, and similarity strategy.

The pipeline is intentionally thin: it delegates every concern to a
specialist module and only handles sequencing and progress reporting.
The similarity strategy is injectable so callers can override it in tests
or pass a custom implementation without touching this file.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from .config import Settings
from .inference.model import get_model
from .inference.similarity import SimilarityStrategy, select_strategy
from .vault import (
    count_managed_links,
    discover_notes,
    has_managed_section,
    parse_notes,
    write_related_notes,
)
from .vault.backup import create_backup
from .vault.obsidian import remove_related_section


def run_pipeline(
    settings: Settings,
    strategy: SimilarityStrategy | None = None,
    backup_confirmed: bool = False,
) -> None:
    """
    Full pipeline: backup -> discover -> parse -> embed -> index -> link -> write.

    *strategy* is optional; when None, select_strategy() chooses based on
    vault size.  Pass an explicit strategy to override in tests or experiments.

    *backup_confirmed* signals that the caller has already obtained user consent
    for a backup (or that no backup is needed, e.g. DRY_RUN).  The pipeline
    never prompts interactively — that responsibility belongs to cli/commands.py.
    """
    run_start = datetime.now(tz=timezone.utc)

    # --- 0. Backup (unless dry-run or caller opted out) ----------------------
    if backup_confirmed and not settings.dry_run:
        create_backup(settings.vault_path)

    # --- 1. Discover notes ---------------------------------------------------
    md_paths = discover_notes(settings.vault_path, settings.exclude_dirs)
    if not md_paths:
        logger.warning("No markdown files found — nothing to do.")
        return

    # --- 2. Parse notes (extract clean text for embedding) -------------------
    notes = parse_notes(md_paths)
    if not notes:
        logger.warning("All notes failed to parse — aborting.")
        return

    logger.info(f"Parsed {len(notes)} notes successfully")

    # --- 3. Embed all notes --------------------------------------------------
    model = get_model(settings.model_dir, settings.model_name)
    texts = [note.body or note.title for note in notes]

    logger.info(f"Encoding {len(texts)} notes …")
    embeddings = model.encode(texts)
    logger.info("Encoding complete")

    # --- 4. Build similarity index -------------------------------------------
    chosen_strategy = strategy or select_strategy(len(notes))
    chosen_strategy.build(embeddings)

    # --- 5. Query: for each note find its top-K neighbours -------------------
    neighbours = chosen_strategy.query(
        embeddings,
        top_k=settings.top_k,
        threshold=settings.similarity_threshold,
    )

    # --- 6. Write related notes sections -------------------------------------
    updated_count = 0
    skipped_count = 0
    modified_notes: list[dict] = []

    for i, note in enumerate(notes):
        related_indices_and_scores = neighbours[i]

        if not related_indices_and_scores:
            skipped_count += 1
            continue

        linked_titles = [notes[idx].title for idx, _score in related_indices_and_scores]
        write_related_notes(note, linked_titles, dry_run=settings.dry_run)
        updated_count += 1
        modified_notes.append({"title": note.title, "path": str(note.path), "links": linked_titles})
        logger.debug("{} → {}", note.title, ", ".join(f"[[{t}]]" for t in linked_titles))

    action = "Would update" if settings.dry_run else "Updated"
    logger.success(
        f"{action} {updated_count} notes, "
        f"{skipped_count} notes had no matches above threshold "
        f"({settings.similarity_threshold})"
    )

    # --- 7. Write execution log ----------------------------------------------
    _write_run_log(settings, run_start, modified_notes, updated_count, skipped_count, len(notes))


def _write_run_log(
    settings: Settings,
    run_start: datetime,
    modified_notes: list[dict],
    updated_count: int,
    skipped_count: int,
    total_count: int,
) -> None:
    """
    Persist a JSON record of this run to LOG_DIR.

    Each file is named run_YYYYMMDD_HHMMSS.json and contains the full list
    of notes that were modified (or would be modified in dry-run mode).
    Failures are logged as warnings — a log write error must never abort a
    pipeline run that has already succeeded.
    """
    try:
        settings.log_dir.mkdir(parents=True, exist_ok=True)
        timestamp_str = run_start.strftime("%Y%m%d_%H%M%S")
        log_path = settings.log_dir / f"run_{timestamp_str}.json"

        record = {
            "timestamp": run_start.isoformat(),
            "vault_path": str(settings.vault_path),
            "model_name": settings.model_name,
            "dry_run": settings.dry_run,
            "similarity_threshold": settings.similarity_threshold,
            "top_k": settings.top_k,
            "summary": {
                "total_notes": total_count,
                "updated": updated_count,
                "skipped": skipped_count,
            },
            "modified_notes": modified_notes,
        }

        log_path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
        logger.debug(f"Run log written to {log_path}")
    except Exception as exc:
        logger.warning(f"Could not write run log: {exc}")


def preview_pipeline(
    settings: Settings,
    strategy: SimilarityStrategy | None = None,
) -> dict:
    """
    Dry-run pass: compute what run_pipeline would do without writing anything.

    Returns:
        note_count       -- total notes discovered
        notes_to_modify  -- notes that would receive a Related Notes section
        link_count       -- total wikilinks that would be written
    """
    md_paths = discover_notes(settings.vault_path, settings.exclude_dirs)
    if not md_paths:
        return {"note_count": 0, "notes_to_modify": 0, "link_count": 0}

    notes = parse_notes(md_paths)
    if not notes:
        return {"note_count": 0, "notes_to_modify": 0, "link_count": 0}

    model = get_model(settings.model_dir, settings.model_name)
    texts = [note.body or note.title for note in notes]
    embeddings = model.encode(texts)

    chosen_strategy = strategy or select_strategy(len(notes))
    chosen_strategy.build(embeddings)
    neighbours = chosen_strategy.query(
        embeddings,
        top_k=settings.top_k,
        threshold=settings.similarity_threshold,
    )

    return {
        "note_count": len(notes),
        "notes_to_modify": sum(1 for n in neighbours if n),
        "link_count": sum(len(n) for n in neighbours),
    }


def get_clean_preview(
    vault_path: Path,
    exclude_dirs: list[str] | None = None,
) -> list[Path]:
    """Return paths of notes that contain a rhizome-managed section."""
    md_paths = discover_notes(vault_path, exclude_dirs)
    return [p for p in md_paths if has_managed_section(p)]


def run_clean(
    vault_path: Path,
    exclude_dirs: list[str] | None = None,
) -> None:
    """
    Remove all '## Related Notes' sections added by this tool.

    Idempotent: running clean on an already-clean vault does nothing.
    """
    md_paths = discover_notes(vault_path, exclude_dirs)
    removed = sum(1 for p in md_paths if remove_related_section(p))
    logger.success(f"Removed 'Related Notes' sections from {removed} notes")


def audit_vault(
    settings: Settings,
    strategy: SimilarityStrategy | None = None,
) -> dict:
    """
    Analyze vault connectivity without modifying any file.

    Returns:
        note_count         -- total notes discovered
        connection_buckets -- {label: count} for the four connectivity bands
        potential_links    -- total wikilinks the pipeline would write
        notes_affected     -- notes that would receive at least one link
    """
    md_paths = discover_notes(settings.vault_path, settings.exclude_dirs)
    if not md_paths:
        return {
            "note_count": 0,
            "connection_buckets": {"none": 0, "1-2": 0, "3-5": 0, "6+": 0},
            "potential_links": 0,
            "notes_affected": 0,
        }

    notes = parse_notes(md_paths)

    # --- Existing connections per note (Related Notes section only) ----------
    existing = [count_managed_links(note) for note in notes]
    connection_buckets = {
        "none": sum(1 for c in existing if c == 0),
        "1-2":  sum(1 for c in existing if 1 <= c <= 2),
        "3-5":  sum(1 for c in existing if 3 <= c <= 5),
        "6+":   sum(1 for c in existing if c >= 6),
    }

    # --- Potential new links (full pipeline, in-memory only) -----------------
    model = get_model(settings.model_dir, settings.model_name)
    texts = [note.body or note.title for note in notes]
    embeddings = model.encode(texts)

    chosen_strategy = strategy or select_strategy(len(notes))
    chosen_strategy.build(embeddings)
    neighbours = chosen_strategy.query(
        embeddings,
        top_k=settings.top_k,
        threshold=settings.similarity_threshold,
    )

    return {
        "note_count": len(notes),
        "connection_buckets": connection_buckets,
        "potential_links": sum(len(n) for n in neighbours),
        "notes_affected": sum(1 for n in neighbours if n),
    }


def get_vault_stats(settings: Settings) -> dict:
    """
    Collect summary statistics for the `status` CLI command.

    Returns a plain dict so the CLI layer can format it however it likes.
    """
    md_paths = discover_notes(settings.vault_path, settings.exclude_dirs)
    notes = parse_notes(md_paths)

    avg_length = (
        sum(len(n.body) for n in notes) / len(notes) if notes else 0
    )

    model_cached = (settings.model_dir / "model.onnx").exists()

    return {
        "note_count": len(notes),
        "avg_note_length_chars": round(avg_length),
        "vault_path": str(settings.vault_path),
        "vault_app": settings.vault_app,
        "model_name": settings.model_name,
        "model_cached": model_cached,
        "model_dir": str(settings.model_dir),
        "similarity_threshold": settings.similarity_threshold,
        "similarity_level": settings.similarity_level,
        "top_k": settings.top_k,
        "dry_run": settings.dry_run,
    }
