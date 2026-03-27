"""
Tests for the rhizome audit feature:
  - count_managed_links()  (vault/obsidian.py)
  - audit_vault()          (pipeline.py)
  - `rhizome audit` CLI command
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from typer.testing import CliRunner

from rhizome.cli.commands import app
from rhizome.pipeline import audit_vault
from rhizome.vault import count_managed_links, parse_note
from rhizome.vault.obsidian import RHIZOME_END, RHIZOME_START

runner = CliRunner()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def vault(tmp_path: Path) -> Path:
    """Vault with three notes; Gamma already has one managed link."""
    (tmp_path / "Alpha.md").write_text("# Alpha\nAbout alpha.")
    (tmp_path / "Beta.md").write_text("# Beta\nAbout beta.")
    (tmp_path / "Gamma.md").write_text(
        "# Gamma\nAbout gamma.\n\n"
        f"{RHIZOME_START}\n## Related Notes\n\n"
        "- [[Alpha]]\n"
        f"{RHIZOME_END}\n"
    )
    return tmp_path


def _mock_settings(vault: Path) -> MagicMock:
    s = MagicMock()
    s.vault_path = vault
    s.similarity_threshold = 0.0  # accept everything so results are deterministic
    s.top_k = 2
    s.model_dir = vault / "models"
    s.model_name = "Xenova/test-model"
    s.exclude_dirs = []
    s.include_dirs = []
    s.chunk_size = 512
    s.chunk_overlap = 32
    return s


def _make_model(n: int) -> MagicMock:
    """Return a mock model that yields n orthonormal unit embeddings."""
    rng = np.random.default_rng(0)
    embs = rng.random((n, 4)).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    m = MagicMock()
    m.encode.return_value = embs
    return m


# ---------------------------------------------------------------------------
# count_managed_links
# ---------------------------------------------------------------------------


def test_count_managed_links_with_section(tmp_path: Path) -> None:
    path = tmp_path / "note.md"
    path.write_text(
        f"{RHIZOME_START}\n## Related Notes\n\n"
        "- [[Alpha]]\n- [[Beta]]\n"
        f"{RHIZOME_END}\n"
    )
    note = parse_note(path)
    assert note is not None
    assert count_managed_links(note) == 2


def test_count_managed_links_no_section(tmp_path: Path) -> None:
    path = tmp_path / "note.md"
    path.write_text("# Plain note\nNo links here.")
    note = parse_note(path)
    assert note is not None
    assert count_managed_links(note) == 0


def test_count_managed_links_ignores_body_links(tmp_path: Path) -> None:
    """Links outside the managed section must not be counted."""
    path = tmp_path / "note.md"
    path.write_text(
        "# Note\nSee [[Manual]] for details.\n\n"
        f"{RHIZOME_START}\n## Related Notes\n\n"
        "- [[Alpha]]\n"
        f"{RHIZOME_END}\n"
    )
    note = parse_note(path)
    assert note is not None
    assert count_managed_links(note) == 1


# ---------------------------------------------------------------------------
# audit_vault
# ---------------------------------------------------------------------------


def test_audit_vault_returns_required_keys(vault: Path) -> None:
    settings = _mock_settings(vault)
    with patch("rhizome.pipeline.get_model", return_value=_make_model(3)):
        result = audit_vault(settings)

    assert "note_count" in result
    assert "connection_buckets" in result
    assert "potential_links" in result
    assert "notes_affected" in result


def test_audit_vault_note_count(vault: Path) -> None:
    settings = _mock_settings(vault)
    with patch("rhizome.pipeline.get_model", return_value=_make_model(3)):
        result = audit_vault(settings)

    assert result["note_count"] == 3


def test_audit_vault_connection_buckets_sum_to_note_count(vault: Path) -> None:
    settings = _mock_settings(vault)
    with patch("rhizome.pipeline.get_model", return_value=_make_model(3)):
        result = audit_vault(settings)

    assert sum(result["connection_buckets"].values()) == result["note_count"]


def test_audit_vault_existing_connections(vault: Path) -> None:
    """Gamma has 1 managed link; Alpha and Beta have none."""
    settings = _mock_settings(vault)
    with patch("rhizome.pipeline.get_model", return_value=_make_model(3)):
        result = audit_vault(settings)

    buckets = result["connection_buckets"]
    assert buckets["none"] == 2
    assert buckets["1-2"] == 1
    assert buckets["3-5"] == 0
    assert buckets["6+"] == 0


def test_audit_vault_does_not_write(vault: Path) -> None:
    """audit_vault must leave all files unchanged."""
    before = {p: p.read_text() for p in vault.glob("*.md")}
    settings = _mock_settings(vault)
    with patch("rhizome.pipeline.get_model", return_value=_make_model(3)):
        audit_vault(settings)
    after = {p: p.read_text() for p in vault.glob("*.md")}
    assert before == after


def test_audit_vault_empty(tmp_path: Path) -> None:
    settings = _mock_settings(tmp_path)
    result = audit_vault(settings)

    assert result["note_count"] == 0
    assert result["potential_links"] == 0
    assert result["notes_affected"] == 0
    assert sum(result["connection_buckets"].values()) == 0


# ---------------------------------------------------------------------------
# CLI: rhizome audit
# ---------------------------------------------------------------------------


def test_audit_command_output_contains_connectivity_header(vault: Path) -> None:
    settings = _mock_settings(vault)
    with (
        patch("rhizome.config.load_settings", return_value=settings),
        patch(
            "rhizome.pipeline.audit_vault",
            return_value={
                "note_count": 3,
                "connection_buckets": {"none": 2, "1-2": 1, "3-5": 0, "6+": 0},
                "potential_links": 4,
                "notes_affected": 2,
            },
        ),
    ):
        result = runner.invoke(app, ["audit"])

    assert result.exit_code == 0, result.output
    assert "Connectivity distribution" in result.output
    assert "Potential new links" in result.output
    assert "Est. notes affected" in result.output


def test_audit_command_exit_zero_on_success(vault: Path) -> None:
    settings = _mock_settings(vault)
    with (
        patch("rhizome.config.load_settings", return_value=settings),
        patch(
            "rhizome.pipeline.audit_vault",
            return_value={
                "note_count": 3,
                "connection_buckets": {"none": 3, "1-2": 0, "3-5": 0, "6+": 0},
                "potential_links": 0,
                "notes_affected": 0,
            },
        ),
    ):
        result = runner.invoke(app, ["audit"])

    assert result.exit_code == 0
