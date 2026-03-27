"""
Tests for cli/commands.py — focused on the run command's new safe-by-default
dry-run flow and the --yes / -y flag.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from rhizome.cli.commands import app

runner = CliRunner()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_settings(*, dry_run: bool = False, vault_path: Path | None = None) -> MagicMock:
    s = MagicMock()
    s.dry_run = dry_run
    s.similarity_threshold = 0.75
    s.top_k = 5
    s.vault_path = vault_path or Path("/fake/vault")
    s.vault_app = "obsidian"
    s.exclude_dirs = []
    s.include_dirs = []
    s.chunk_size = 512
    s.chunk_overlap = 32
    return s


_PREVIEW = {"notes_to_modify": 3, "link_count": 7, "note_count": 10}


# ---------------------------------------------------------------------------
# DRY_RUN=true (legacy env-var behaviour)
# ---------------------------------------------------------------------------


def test_dry_run_env_calls_run_pipeline_not_preview():
    """DRY_RUN=true skips the preview pass and goes straight to run_pipeline."""
    settings = _mock_settings(dry_run=True)
    with (
        patch("rhizome.config.load_settings", return_value=settings),
        patch("rhizome.pipeline.run_pipeline") as mock_run,
        patch("rhizome.pipeline.preview_pipeline") as mock_preview,
    ):
        result = runner.invoke(app, ["run"])

    assert result.exit_code == 0, result.output
    mock_preview.assert_not_called()
    mock_run.assert_called_once()


def test_dry_run_env_calls_run_pipeline_backup_not_confirmed():
    """In DRY_RUN mode backup_confirmed must be False."""
    settings = _mock_settings(dry_run=True)
    with (
        patch("rhizome.config.load_settings", return_value=settings),
        patch("rhizome.pipeline.run_pipeline") as mock_run,
        patch("rhizome.pipeline.preview_pipeline"),
    ):
        runner.invoke(app, ["run"])

    mock_run.assert_called_once_with(settings, backup_confirmed=False)


# ---------------------------------------------------------------------------
# --yes / -y flag (non-interactive mode)
# ---------------------------------------------------------------------------


def test_yes_flag_skips_proceed_prompt():
    """--yes must not print a 'Proceed?' prompt."""
    settings = _mock_settings()
    with (
        patch("rhizome.config.load_settings", return_value=settings),
        patch("rhizome.pipeline.preview_pipeline", return_value=_PREVIEW),
        patch("rhizome.pipeline.run_pipeline"),
    ):
        result = runner.invoke(app, ["run", "--yes"])

    assert "Proceed?" not in result.output
    assert result.exit_code == 0, result.output


def test_yes_flag_skips_backup_prompt():
    """--yes must not show an interactive backup-creation question."""
    settings = _mock_settings()
    with (
        patch("rhizome.config.load_settings", return_value=settings),
        patch("rhizome.pipeline.preview_pipeline", return_value=_PREVIEW),
        patch("rhizome.pipeline.run_pipeline"),
    ):
        result = runner.invoke(app, ["run", "--yes"])

    assert "Do you want to create a backup" not in result.output
    assert result.exit_code == 0, result.output


def test_yes_flag_calls_run_pipeline_with_backup_confirmed():
    """--yes must pass backup_confirmed=True to run_pipeline."""
    settings = _mock_settings()
    with (
        patch("rhizome.config.load_settings", return_value=settings),
        patch("rhizome.pipeline.preview_pipeline", return_value=_PREVIEW),
        patch("rhizome.pipeline.run_pipeline") as mock_run,
    ):
        runner.invoke(app, ["run", "--yes"])

    mock_run.assert_called_once_with(settings, backup_confirmed=True)


def test_short_yes_flag():
    """-y is an alias for --yes."""
    settings = _mock_settings()
    with (
        patch("rhizome.config.load_settings", return_value=settings),
        patch("rhizome.pipeline.preview_pipeline", return_value=_PREVIEW),
        patch("rhizome.pipeline.run_pipeline") as mock_run,
    ):
        result = runner.invoke(app, ["run", "-y"])

    assert result.exit_code == 0, result.output
    mock_run.assert_called_once_with(settings, backup_confirmed=True)


# ---------------------------------------------------------------------------
# Interactive mode (no --yes, no DRY_RUN)
# ---------------------------------------------------------------------------


def test_interactive_shows_preview_summary():
    """Without --yes the summary (notes to modify, links to write) must appear."""
    settings = _mock_settings()
    with (
        patch("rhizome.config.load_settings", return_value=settings),
        patch("rhizome.pipeline.preview_pipeline", return_value=_PREVIEW),
        patch("rhizome.pipeline.run_pipeline"),
        patch("rhizome.vault.discover_notes", return_value=[]),
    ):
        # Input: confirm proceed=y, confirm backup=y
        result = runner.invoke(app, ["run"], input="y\ny\n")

    assert "Notes to modify" in result.output
    assert "Links to write" in result.output
    assert "backup" in result.output.lower()


def test_interactive_abort_does_not_call_pipeline():
    """Answering 'n' to 'Proceed?' must abort without calling run_pipeline."""
    settings = _mock_settings()
    with (
        patch("rhizome.config.load_settings", return_value=settings),
        patch("rhizome.pipeline.preview_pipeline", return_value=_PREVIEW),
        patch("rhizome.pipeline.run_pipeline") as mock_run,
    ):
        result = runner.invoke(app, ["run"], input="n\n")

    mock_run.assert_not_called()
    assert "Aborted" in result.output
