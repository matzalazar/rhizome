"""
Typer command definitions.

All CLI commands live here so main.py stays a thin bootstrap.  Each command
follows the same pattern:
  1. Configure logging
  2. Load and validate settings
  3. Delegate to the relevant pipeline / backup function
  4. Surface errors cleanly (no unhandled tracebacks for expected failures)
"""

import copy
import shutil
import sys
import textwrap
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import typer
from loguru import logger

from rhizome.config import Settings
from rhizome.vault import RELATED_NOTES_HEADER

app = typer.Typer(
    name="rhizome",
    help="Generate semantic [[wikilinks]] between Obsidian notes using local ONNX inference.",
    add_completion=False,
)


_LEVEL_SYMBOLS: dict[str, str] = {
    "DEBUG":    "[.]",
    "INFO":     "[i]",
    "SUCCESS":  "[-]",
    "WARNING":  "[!]",
    "ERROR":    "[x]",
    "CRITICAL": "[!!]",
}

_THRESHOLD_PRESETS: dict[str, float] = {
    "low": 0.60,
    "medium": 0.75,
    "high": 0.88,
}

_RECOMMENDED_TOP_K = 5
_RECOMMENDED_THRESHOLD = "medium (0.75)"
_RECOMMENDED_CHUNK_SIZE = 512
_RECOMMENDED_CHUNK_OVERLAP = 32


def _log_format(record: dict) -> str:
    symbol = _LEVEL_SYMBOLS.get(record["level"].name, "[?]")
    return f"<green>{{time:HH:mm:ss}}</green> <level>{symbol}</level> {{message}}\n"


def _configure_logging(verbose: bool) -> None:
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(sys.stderr, format=_log_format, level=level, colorize=True)


def _clone_settings(settings: Settings) -> Settings:
    return copy.deepcopy(settings)


def _replace_settings(settings: Settings, **updates: Any) -> Settings:
    data = settings.model_dump()
    data.update(updates)
    return Settings.model_validate(data)


def _format_threshold_value(value: float) -> str:
    for label, preset_value in _THRESHOLD_PRESETS.items():
        if abs(value - preset_value) < 1e-9:
            return label
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _echo_wrapped_detail(label: str, text: str) -> None:
    width = max(60, shutil.get_terminal_size(fallback=(100, 20)).columns)
    indent = f"  {label:<8}"
    typer.echo(
        textwrap.fill(
            text,
            width=width,
            initial_indent=indent,
            subsequent_indent=" " * len(indent),
            break_long_words=False,
            break_on_hyphens=False,
        )
    )


def _echo_setting_help(name: str, default: str, effect: str) -> None:
    typer.echo("")
    typer.echo(name)
    _echo_wrapped_detail("Default", default)
    _echo_wrapped_detail("Effect", effect)


def _echo_selected_notes(settings: Settings, selected_note_paths: Sequence[Path]) -> None:
    typer.echo("")
    typer.echo("Selected notes")
    typer.echo("--------------")
    for i, path in enumerate(selected_note_paths, start=1):
        typer.echo(f"  [{i}] {path.relative_to(settings.vault_path)}")


def _prompt_manual_targets(settings: Settings) -> list[Path]:
    from rhizome.vault import discover_notes

    note_paths = discover_notes(
        settings.vault_path, settings.exclude_dirs, settings.include_dirs
    )
    if not note_paths:
        typer.echo("No notes found in the current scope.")
        raise typer.Exit(code=1)

    selected_note_paths: list[Path] = []

    while True:
        typer.echo("")
        query = typer.prompt("Search note filename", default="").strip()
        if not query:
            typer.echo("Please enter part of a filename or path.")
            continue

        lowered = query.lower()
        matches = [
            path for path in note_paths
            if lowered in path.stem.lower()
            or lowered in str(path.relative_to(settings.vault_path)).lower()
        ]

        if not matches:
            typer.echo(f'No notes matched "{query}". Try again.')
            continue

        typer.echo("\nMatching notes")
        typer.echo("--------------")
        for i, path in enumerate(matches, start=1):
            typer.echo(f"  [{i}] {path.relative_to(settings.vault_path)}")
        typer.echo("")

        choice = typer.prompt(f"Select a note [1-{len(matches)}]")
        try:
            index = int(choice) - 1
            if not (0 <= index < len(matches)):
                raise ValueError
        except ValueError:
            typer.echo(f"Invalid selection: {choice}")
            continue

        selected_path = matches[index]
        if selected_path in selected_note_paths:
            typer.echo(
                "That note is already selected. Choose another one or finish the selection."
            )
        else:
            selected_note_paths.append(selected_path)
            typer.echo(
                f"Added: {selected_path.relative_to(settings.vault_path)}"
            )

        _echo_selected_notes(settings, selected_note_paths)
        typer.echo("")
        if typer.confirm("Add another note?", default=False):
            continue
        if selected_note_paths:
            return selected_note_paths
        typer.echo("Select at least one note before finishing.")


def _prompt_top_k(settings: Settings) -> Settings:
    _echo_setting_help(
        "TOP_K",
        str(_RECOMMENDED_TOP_K),
        "Higher values add more links, but they can also include weaker matches.",
    )
    while True:
        raw_value = typer.prompt("TOP_K", default=str(settings.top_k)).strip()
        try:
            return _replace_settings(settings, top_k=int(raw_value))
        except Exception as exc:
            typer.echo(f"Invalid TOP_K: {exc}")


def _prompt_similarity_threshold(settings: Settings) -> Settings:
    _echo_setting_help(
        "SIMILARITY_THRESHOLD",
        _RECOMMENDED_THRESHOLD,
        "Lower values surface more links; higher values keep results stricter.",
    )
    while True:
        raw_value = typer.prompt(
            "SIMILARITY_THRESHOLD",
            default=_format_threshold_value(settings.similarity_threshold),
        ).strip()
        try:
            return _replace_settings(settings, similarity_threshold=raw_value)
        except Exception as exc:
            typer.echo(f"Invalid SIMILARITY_THRESHOLD: {exc}")


def _prompt_chunk_size(settings: Settings) -> Settings:
    _echo_setting_help(
        "CHUNK_SIZE",
        str(_RECOMMENDED_CHUNK_SIZE),
        "Lower values can help very long notes, but they increase embedding time.",
    )
    while True:
        raw_chunk_size = typer.prompt("CHUNK_SIZE", default=str(settings.chunk_size)).strip()
        try:
            return _replace_settings(
                settings,
                chunk_size=int(raw_chunk_size),
            )
        except Exception as exc:
            typer.echo(f"Invalid CHUNK_SIZE: {exc}")


def _prompt_chunk_overlap(settings: Settings) -> Settings:
    _echo_setting_help(
        "CHUNK_OVERLAP",
        str(_RECOMMENDED_CHUNK_OVERLAP),
        "Higher values preserve more context but add extra work. It must stay below CHUNK_SIZE.",
    )

    while True:
        raw_chunk_overlap = typer.prompt(
            "CHUNK_OVERLAP", default=str(settings.chunk_overlap)
        ).strip()
        try:
            return _replace_settings(
                settings,
                chunk_overlap=int(raw_chunk_overlap),
            )
        except Exception as exc:
            typer.echo(f"Invalid CHUNK_OVERLAP: {exc}")


def _prompt_related_notes_header(current_header: str) -> str:
    _echo_setting_help(
        "RELATED_NOTES_HEADER",
        RELATED_NOTES_HEADER,
        "Changes the section heading written in this run only. "
        "Use a markdown header such as ## Suggested Links.",
    )
    if not typer.confirm("Change the section header for this run?", default=False):
        return current_header

    while True:
        header = typer.prompt("RELATED_NOTES_HEADER", default=current_header).strip()
        if header:
            return header
        typer.echo("RELATED_NOTES_HEADER cannot be empty.")


def _prompt_runtime_overrides(
    settings: Settings,
    related_notes_header: str,
) -> tuple[Settings, str]:
    if not settings.manual_override_fields:
        return settings, related_notes_header

    if not typer.confirm("Review runtime settings for this run only?", default=False):
        return settings, related_notes_header

    typer.echo("")
    typer.echo("Press Enter to keep the current value shown in each prompt.")
    tuned = _clone_settings(settings)
    for field_name in settings.manual_override_fields:
        if field_name == "top_k":
            tuned = _prompt_top_k(tuned)
        elif field_name == "similarity_threshold":
            tuned = _prompt_similarity_threshold(tuned)
        elif field_name == "chunk_size":
            tuned = _prompt_chunk_size(tuned)
        elif field_name == "chunk_overlap":
            tuned = _prompt_chunk_overlap(tuned)
        elif field_name == "related_notes_header":
            related_notes_header = _prompt_related_notes_header(related_notes_header)
    typer.echo("")
    return tuned, related_notes_header


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

@app.command()
def run(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation prompt and auto-confirm backup (for CI / scripted usage)",
    ),
    manual: bool = typer.Option(
        False,
        "--manual",
        help=(
            "Interactively pick one or more notes to update while matching "
            "against the full vault."
        ),
    ),
) -> None:
    """
    Execute the full semantic linking pipeline.

    By default, a dry-run preview is shown first and explicit confirmation is
    required before any file is written.  Pass --yes / -y to skip all prompts
    (useful in CI or scripted environments).

    Set DRY_RUN=true to preview proposed links without writing anything at all.
    """
    _configure_logging(verbose)

    from rhizome.config import load_settings
    from rhizome.pipeline import preview_pipeline, run_pipeline
    from rhizome.vault import discover_notes

    if manual and yes:
        raise typer.BadParameter(
            "--manual cannot be used with --yes because note selection is interactive.",
            param_hint="--manual",
        )

    try:
        settings = load_settings()
    except Exception as exc:
        logger.error(f"Configuration error: {exc}")
        raise typer.Exit(code=1) from exc

    target_note_paths: list[Path] = []
    related_notes_header = RELATED_NOTES_HEADER
    if manual:
        target_note_paths = _prompt_manual_targets(settings)
        settings, related_notes_header = _prompt_runtime_overrides(
            settings,
            related_notes_header,
        )

    logger.info(f"Vault: {settings.vault_path} ({settings.vault_app})")
    logger.info(
        f"Settings: threshold={settings.similarity_threshold}, "
        f"top_k={settings.top_k}, dry_run={settings.dry_run}"
    )
    if target_note_paths:
        logger.info(
            f"Manual targets: {len(target_note_paths)} selected"
        )

    # --- DRY_RUN=true: legacy behaviour — preview only, no writes, no prompt --
    if settings.dry_run:
        try:
            run_pipeline(
                settings,
                backup_confirmed=False,
                target_note_paths=target_note_paths or None,
                related_notes_header=related_notes_header,
            )
        except Exception as exc:
            logger.exception(f"Pipeline failed: {exc}")
            raise typer.Exit(code=1) from exc
        return

    # --- Preview pass --------------------------------------------------------
    logger.info("Running preview …")
    try:
        preview = preview_pipeline(settings, target_note_paths=target_note_paths or None)
    except Exception as exc:
        logger.exception(f"Preview failed: {exc}")
        raise typer.Exit(code=1) from exc

    if target_note_paths:
        typer.echo("\n  Manual targets")
        for path in target_note_paths:
            typer.echo(f"    - {path.relative_to(settings.vault_path)}")
        typer.echo(f"  Notes selected   : {len(target_note_paths)}")
    typer.echo(f"\n  Notes to modify  : {preview['notes_to_modify']}")
    typer.echo(f"  Links to write   : {preview['link_count']}")
    typer.echo("  (A timestamped backup will be created before writing.)")
    typer.echo("")

    # --- Confirmation --------------------------------------------------------
    if yes:
        # Non-interactive mode: proceed and auto-confirm backup.
        backup_confirmed = True
    else:
        confirmed = typer.confirm("  Proceed?", default=True)
        typer.echo("")
        if not confirmed:
            typer.echo("Aborted.")
            raise typer.Exit()

        note_paths = discover_notes(
            settings.vault_path, settings.exclude_dirs, settings.include_dirs
        )
        typer.echo(f"  Vault path  : {settings.vault_path}")
        typer.echo(f"  Notes found : {len(note_paths)}")
        if target_note_paths:
            typer.echo(f"  Notes selected : {len(target_note_paths)}")
        backup_confirmed = typer.confirm(
            "  Do you want to create a backup before proceeding?",
            default=True,
        )
        typer.echo("")

    try:
        run_pipeline(
            settings,
            backup_confirmed=backup_confirmed,
            target_note_paths=target_note_paths or None,
            related_notes_header=related_notes_header,
        )
    except RuntimeError as exc:
        # RuntimeError is raised by create_backup() on failure — show the
        # message without a full traceback so the output stays readable.
        logger.error(str(exc))
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        logger.exception(f"Pipeline failed: {exc}")
        raise typer.Exit(code=1) from exc


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

@app.command()
def status(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
) -> None:
    """Show vault statistics and model cache status."""
    _configure_logging(verbose)

    from rhizome.config import load_settings
    from rhizome.pipeline import get_vault_stats

    try:
        settings = load_settings()
    except Exception as exc:
        logger.error(f"Configuration error: {exc}")
        raise typer.Exit(code=1) from exc

    stats = get_vault_stats(settings)

    typer.echo("\nVault status")
    typer.echo("------------")
    typer.echo(f"  Path            : {stats['vault_path']}")
    typer.echo(f"  App             : {stats['vault_app']}")
    typer.echo(f"  Notes found     : {stats['note_count']}")
    typer.echo(f"  Avg note length : {stats['avg_note_length_chars']} chars")
    typer.echo("\nModel cache")
    typer.echo(f"  Model           : {stats['model_name']}")
    typer.echo(f"  Directory       : {stats['model_dir']}")
    typer.echo(
        f"  Cached          : "
        f"{'yes' if stats['model_cached'] else 'no — will download on next run'}"
    )
    typer.echo("\nActive settings")
    typer.echo(f"  Threshold       : {stats['similarity_threshold']} ({stats['similarity_level']})")
    typer.echo(f"  Top-K           : {stats['top_k']}")
    typer.echo(f"  Dry-run         : {stats['dry_run']}")
    typer.echo("")


# ---------------------------------------------------------------------------
# clean
# ---------------------------------------------------------------------------

@app.command()
def clean(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
) -> None:
    """Remove all '## Related Notes' sections added by this tool."""
    _configure_logging(verbose)

    from rhizome.config import load_settings
    from rhizome.pipeline import get_clean_preview, run_clean

    try:
        settings = load_settings()
    except Exception as exc:
        logger.error(f"Configuration error: {exc}")
        raise typer.Exit(code=1) from exc

    preview = get_clean_preview(settings.vault_path, settings.exclude_dirs, settings.include_dirs)

    if not preview:
        typer.echo("No notes with rhizome sections found. Nothing to do.")
        return

    typer.echo(f"\n  {len(preview)} notes have a managed section:\n")
    for p in preview[:20]:
        typer.echo(f"    {p.stem}")
    if len(preview) > 20:
        typer.echo(f"    … and {len(preview) - 20} more")
    typer.echo("")

    confirmed = typer.confirm(
        f"  Remove Related Notes sections from {len(preview)} notes?",
        default=False,
    )
    if not confirmed:
        typer.echo("Aborted.")
        raise typer.Exit()

    run_clean(settings.vault_path, settings.exclude_dirs, settings.include_dirs)


# ---------------------------------------------------------------------------
# backups
# ---------------------------------------------------------------------------

@app.command()
def backups(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
) -> None:
    """List all available backups with their metadata."""
    _configure_logging(verbose)

    from rhizome.config import load_settings
    from rhizome.vault.backup import list_backups

    try:
        settings = load_settings()
    except Exception as exc:
        logger.error(f"Configuration error: {exc}")
        raise typer.Exit(code=1) from exc

    available = list_backups(settings.vault_path)

    if not available:
        typer.echo("No backups found.")
        return

    typer.echo(f"\n{len(available)} backup(s) available:\n")
    for i, entry in enumerate(available, start=1):
        typer.echo(f"  [{i}] {entry.get('timestamp', 'unknown time')}")
        typer.echo(f"       Notes : {entry.get('note_count', '?')}")
        typer.echo(f"       Path  : {entry['backup_dir']}")
        typer.echo(f"       rhizome v{entry.get('rhizome_version', '?')}")
        typer.echo("")


# ---------------------------------------------------------------------------
# restore
# ---------------------------------------------------------------------------

@app.command()
def restore(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
) -> None:
    """Interactively restore a previous backup."""
    _configure_logging(verbose)

    from rhizome.config import load_settings
    from rhizome.vault.backup import list_backups, restore_backup

    try:
        settings = load_settings()
    except Exception as exc:
        logger.error(f"Configuration error: {exc}")
        raise typer.Exit(code=1) from exc

    available = list_backups(settings.vault_path)

    if not available:
        typer.echo("No backups found.")
        raise typer.Exit()

    typer.echo(f"\n{len(available)} backup(s) available:\n")
    for i, entry in enumerate(available, start=1):
        typer.echo(
            f"  [{i}] {entry.get('timestamp', 'unknown')} "
            f"— {entry.get('note_count', '?')} notes"
        )
    typer.echo("")

    choice = typer.prompt(f"Select a backup to restore [1–{len(available)}]")
    try:
        index = int(choice) - 1
        if not (0 <= index < len(available)):
            raise ValueError
    except ValueError:
        typer.echo(f"Invalid selection: {choice}")
        raise typer.Exit(code=1) from None

    selected = available[index]
    typer.echo(f"\nSelected: {selected['backup_dir']}")

    # Default is N — accidental Enter does not overwrite the vault.
    confirmed = typer.confirm(
        "This will overwrite current vault contents. Confirm?",
        default=False,
    )
    if not confirmed:
        typer.echo("Aborted.")
        raise typer.Exit()

    try:
        restore_backup(
            backup_dir=Path(selected["backup_dir"]),
            vault_path=settings.vault_path,
        )
    except RuntimeError as exc:
        logger.error(str(exc))
        raise typer.Exit(code=1) from exc


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------

@app.command()
def audit(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
) -> None:
    """
    Analyze vault connectivity without modifying any file.

    Reports the distribution of existing connections and estimates how many
    new links the pipeline would produce.
    """
    _configure_logging(verbose)

    from rhizome.config import load_settings
    from rhizome.pipeline import audit_vault

    try:
        settings = load_settings()
    except Exception as exc:
        logger.error(f"Configuration error: {exc}")
        raise typer.Exit(code=1) from exc

    try:
        result = audit_vault(settings)
    except Exception as exc:
        logger.exception(f"Audit failed: {exc}")
        raise typer.Exit(code=1) from exc

    n = result["note_count"]
    if n == 0:
        logger.warning("No notes found — nothing to audit.")
        return

    def _pct(count: int) -> str:
        return f"{count / n * 100:3.0f}%"

    buckets = result["connection_buckets"]

    logger.info(f"Vault audit — {settings.vault_path} ({n} notes)")
    typer.echo("Connectivity distribution")
    typer.echo("─────────────────────────")
    typer.echo(f"No connections       : {buckets['none']:3d} notes  ({_pct(buckets['none'])})")
    typer.echo(f"1–2 connections      : {buckets['1-2']:3d} notes  ({_pct(buckets['1-2'])})")
    typer.echo(f"3–5 connections      : {buckets['3-5']:3d} notes  ({_pct(buckets['3-5'])})")
    typer.echo(f"6+  connections      : {buckets['6+']:3d} notes  ({_pct(buckets['6+'])})")
    typer.echo(f"Potential new links  : {result['potential_links']:3d}  (dry-run to preview them)")
    typer.echo(f"Est. notes affected  : {result['notes_affected']:3d}")
    logger.info("Run `rhizome run` to generate links.")


# ---------------------------------------------------------------------------
# download-model
# ---------------------------------------------------------------------------

@app.command(name="download-model")
def download_model(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
) -> None:
    """
    Pre-download and cache the ONNX model without running the pipeline.

    Useful for Docker builds or CI environments where you want to bake the
    model into an image layer separately from the pipeline execution.
    """
    _configure_logging(verbose)

    from rhizome.config import load_settings
    from rhizome.inference.model import PureONNXEmbeddingModel

    try:
        settings = load_settings()
    except Exception as exc:
        logger.error(f"Configuration error: {exc}")
        raise typer.Exit(code=1) from exc

    try:
        PureONNXEmbeddingModel(settings.model_dir, settings.model_name)._load_model()
    except RuntimeError as exc:
        logger.error(str(exc))
        raise typer.Exit(code=1) from exc

    typer.echo(f"Model ready at {settings.model_dir}")
