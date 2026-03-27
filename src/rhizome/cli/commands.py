"""
Typer command definitions.

All CLI commands live here so main.py stays a thin bootstrap.  Each command
follows the same pattern:
  1. Configure logging
  2. Load and validate settings
  3. Delegate to the relevant pipeline / backup function
  4. Surface errors cleanly (no unhandled tracebacks for expected failures)
"""

import sys
from pathlib import Path

import typer
from loguru import logger

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


def _log_format(record: dict) -> str:
    symbol = _LEVEL_SYMBOLS.get(record["level"].name, "[?]")
    return f"<green>{{time:HH:mm:ss}}</green> <level>{symbol}</level> {{message}}\n"


def _configure_logging(verbose: bool) -> None:
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(sys.stderr, format=_log_format, level=level, colorize=True)


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

    try:
        settings = load_settings()
    except Exception as exc:
        logger.error(f"Configuration error: {exc}")
        raise typer.Exit(code=1) from exc

    logger.info(f"Vault: {settings.vault_path} ({settings.vault_app})")
    logger.info(
        f"Settings: threshold={settings.similarity_threshold}, "
        f"top_k={settings.top_k}, dry_run={settings.dry_run}"
    )

    # --- DRY_RUN=true: legacy behaviour — preview only, no writes, no prompt --
    if settings.dry_run:
        try:
            run_pipeline(settings, backup_confirmed=False)
        except Exception as exc:
            logger.exception(f"Pipeline failed: {exc}")
            raise typer.Exit(code=1) from exc
        return

    # --- Preview pass --------------------------------------------------------
    logger.info("Running preview …")
    try:
        preview = preview_pipeline(settings)
    except Exception as exc:
        logger.exception(f"Preview failed: {exc}")
        raise typer.Exit(code=1) from exc

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
        backup_confirmed = typer.confirm(
            "  Do you want to create a backup before proceeding?",
            default=True,
        )
        typer.echo("")

    try:
        run_pipeline(settings, backup_confirmed=backup_confirmed)
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
