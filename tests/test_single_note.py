import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from typer.testing import CliRunner

from rhizome.cli.commands import app
from rhizome.config import Settings
from rhizome.pipeline import preview_pipeline, run_pipeline
from rhizome.vault import RHIZOME_START

runner = CliRunner()


def _build_settings(vault: Path, **updates) -> Settings:
    data = {
        "vault_path": vault,
        "vault_app": "obsidian",
        "similarity_threshold": 0.75,
        "top_k": 5,
        "model_dir": vault / "models",
        "model_name": "Xenova/test-model",
        "dry_run": False,
        "log_dir": vault / "logs",
        "exclude_dirs": [],
        "include_dirs": [],
        "chunk_size": 512,
        "chunk_overlap": 32,
    }
    data.update(updates)
    return Settings.model_validate(data)


def _make_model(n: int) -> MagicMock:
    rng = np.random.default_rng(0)
    embs = rng.random((n, 4)).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    model = MagicMock()
    model.encode.return_value = embs
    return model


def _make_strategy(neighbours: list[list[tuple[int, float]]]) -> MagicMock:
    strategy = MagicMock()
    strategy.query.return_value = neighbours
    return strategy


def test_single_note_flag_rejects_yes(tmp_path: Path) -> None:
    settings = _build_settings(tmp_path)
    with patch("rhizome.config.load_settings", return_value=settings):
        result = runner.invoke(app, ["run", "--single-note", "--yes"])

    assert result.exit_code != 0
    assert "--single-note cannot be used with --yes" in result.output


def test_single_note_run_prompts_for_search_and_selection(tmp_path: Path) -> None:
    (tmp_path / "Alpha.md").write_text("# Alpha\nBody")
    (tmp_path / "Beta.md").write_text("# Beta\nBody")
    settings = _build_settings(tmp_path)

    with (
        patch("rhizome.config.load_settings", return_value=settings),
        patch(
            "rhizome.pipeline.preview_pipeline",
            return_value={"notes_to_modify": 1, "link_count": 2},
        ) as mock_preview,
        patch("rhizome.pipeline.run_pipeline") as mock_run,
    ):
        result = runner.invoke(app, ["run", "--single-note"], input="alp\n1\n\ny\ny\n")

    assert result.exit_code == 0, result.output
    preview_args = mock_preview.call_args.kwargs
    assert preview_args["target_note_path"] == tmp_path / "Alpha.md"
    run_args = mock_run.call_args.kwargs
    assert run_args["target_note_path"] == tmp_path / "Alpha.md"
    assert run_args["related_notes_header"] == "## Related Notes"
    assert "Target note" in result.output


def test_single_note_run_handles_no_match_and_invalid_selection(tmp_path: Path) -> None:
    (tmp_path / "Alpha.md").write_text("# Alpha\nBody")
    settings = _build_settings(tmp_path)

    with (
        patch("rhizome.config.load_settings", return_value=settings),
        patch(
            "rhizome.pipeline.preview_pipeline",
            return_value={"notes_to_modify": 1, "link_count": 1},
        ),
        patch("rhizome.pipeline.run_pipeline") as mock_run,
    ):
        result = runner.invoke(
            app,
            ["run", "--single-note"],
            input="zzz\nalp\n9\n1\n\nn\n",
        )

    assert "No notes matched" in result.output
    assert "Invalid selection: 9" in result.output
    assert "Aborted." in result.output
    mock_run.assert_not_called()


def test_single_note_lists_relative_paths_for_duplicate_names(tmp_path: Path) -> None:
    (tmp_path / "projects").mkdir()
    (tmp_path / "areas").mkdir()
    (tmp_path / "projects" / "Plan.md").write_text("# Plan\nProjects")
    (tmp_path / "areas" / "Plan.md").write_text("# Plan\nAreas")
    settings = _build_settings(tmp_path)

    with (
        patch("rhizome.config.load_settings", return_value=settings),
        patch(
            "rhizome.pipeline.preview_pipeline",
            return_value={"notes_to_modify": 1, "link_count": 1},
        ),
        patch("rhizome.pipeline.run_pipeline"),
    ):
        result = runner.invoke(app, ["run", "--single-note"], input="plan\n1\n\nn\n")

    assert "projects\\Plan.md" in result.output or "projects/Plan.md" in result.output
    assert "areas\\Plan.md" in result.output or "areas/Plan.md" in result.output


def test_single_note_can_override_runtime_settings_for_one_run(tmp_path: Path) -> None:
    (tmp_path / "Alpha.md").write_text("# Alpha\nBody")
    settings = _build_settings(tmp_path)

    with (
        patch("rhizome.config.load_settings", return_value=settings),
        patch(
            "rhizome.pipeline.preview_pipeline",
            return_value={"notes_to_modify": 1, "link_count": 1},
        ) as mock_preview,
        patch("rhizome.pipeline.run_pipeline"),
    ):
        result = runner.invoke(
            app,
            ["run", "--single-note"],
            input="alp\n1\ny\n7\nlow\n256\n16\nn\nn\n",
        )

    assert result.exit_code == 0, result.output
    tuned_settings = mock_preview.call_args.args[0]
    assert tuned_settings.top_k == 7
    assert tuned_settings.similarity_threshold == 0.60
    assert tuned_settings.chunk_size == 256
    assert tuned_settings.chunk_overlap == 16
    assert "TOP_K" in result.output
    assert "Default" in result.output
    assert "5" in result.output
    assert "medium (0.75)" in result.output
    assert "512" in result.output
    assert "32" in result.output


def test_single_note_can_override_header_for_one_run(tmp_path: Path) -> None:
    (tmp_path / "Alpha.md").write_text("# Alpha\nBody")
    settings = _build_settings(tmp_path)

    with (
        patch("rhizome.config.load_settings", return_value=settings),
        patch(
            "rhizome.pipeline.preview_pipeline",
            return_value={"notes_to_modify": 1, "link_count": 1},
        ),
        patch("rhizome.pipeline.run_pipeline") as mock_run,
    ):
        result = runner.invoke(
            app,
            ["run", "--single-note"],
            input="alp\n1\ny\n\n\n\n\ny\n## Suggested Links\ny\ny\n",
        )

    assert result.exit_code == 0, result.output
    run_args = mock_run.call_args.kwargs
    assert run_args["related_notes_header"] == "## Suggested Links"
    assert "Change the section header for this run?" in result.output
    assert "RELATED_NOTES_HEADER" in result.output


def test_single_note_dry_run_still_selects_target(tmp_path: Path) -> None:
    (tmp_path / "Alpha.md").write_text("# Alpha\nBody")
    settings = _build_settings(tmp_path, dry_run=True)

    with (
        patch("rhizome.config.load_settings", return_value=settings),
        patch("rhizome.pipeline.preview_pipeline") as mock_preview,
        patch("rhizome.pipeline.run_pipeline") as mock_run,
    ):
        result = runner.invoke(app, ["run", "--single-note"], input="alp\n1\n\n")

    assert result.exit_code == 0, result.output
    mock_preview.assert_not_called()
    run_args = mock_run.call_args.kwargs
    assert run_args["backup_confirmed"] is False
    assert run_args["target_note_path"] == tmp_path / "Alpha.md"
    assert run_args["related_notes_header"] == "## Related Notes"


def test_preview_pipeline_counts_only_target_note(tmp_path: Path) -> None:
    alpha = tmp_path / "Alpha.md"
    beta = tmp_path / "Beta.md"
    gamma = tmp_path / "Gamma.md"
    alpha.write_text("# Alpha\nBody")
    beta.write_text("# Beta\nBody")
    gamma.write_text("# Gamma\nBody")
    settings = _build_settings(tmp_path)
    model = _make_model(3)
    strategy = _make_strategy([
        [(1, 0.91), (2, 0.88)],
        [(0, 0.91)],
        [],
    ])

    with patch("rhizome.pipeline.get_model", return_value=model):
        result = preview_pipeline(settings, strategy=strategy, target_note_path=alpha)

    assert model.encode.call_args.args[0] == ["# Alpha\nBody", "# Beta\nBody", "# Gamma\nBody"]
    assert result == {"note_count": 1, "notes_to_modify": 1, "link_count": 2}


def test_run_pipeline_writes_and_logs_only_selected_note(tmp_path: Path) -> None:
    alpha = tmp_path / "Alpha.md"
    beta = tmp_path / "Beta.md"
    gamma = tmp_path / "Gamma.md"
    alpha.write_text("# Alpha\nAlpha body.")
    beta.write_text("# Beta\nBeta body.")
    gamma.write_text("# Gamma\nGamma body.")

    settings = _build_settings(tmp_path)
    model = _make_model(3)
    strategy = _make_strategy([
        [(1, 0.91), (2, 0.88)],
        [(0, 0.91)],
        [(0, 0.88)],
    ])

    with patch("rhizome.pipeline.get_model", return_value=model):
        run_pipeline(settings, strategy=strategy, target_note_path=alpha)

    assert RHIZOME_START in alpha.read_text()
    assert RHIZOME_START not in beta.read_text()
    assert RHIZOME_START not in gamma.read_text()
    assert model.encode.call_args.args[0] == [
        "# Alpha\nAlpha body.",
        "# Beta\nBeta body.",
        "# Gamma\nGamma body.",
    ]

    log_files = sorted((tmp_path / "logs").glob("run_*.json"))
    assert len(log_files) == 1
    record = json.loads(log_files[0].read_text())
    assert record["summary"]["total_notes"] == 1
    assert [entry["title"] for entry in record["modified_notes"]] == ["Alpha"]


def test_run_pipeline_uses_custom_header_for_selected_note(tmp_path: Path) -> None:
    alpha = tmp_path / "Alpha.md"
    beta = tmp_path / "Beta.md"
    alpha.write_text("# Alpha\nAlpha body.")
    beta.write_text("# Beta\nBeta body.")

    settings = _build_settings(tmp_path)
    model = _make_model(2)
    strategy = _make_strategy([
        [(1, 0.91)],
        [(0, 0.91)],
    ])

    with patch("rhizome.pipeline.get_model", return_value=model):
        run_pipeline(
            settings,
            strategy=strategy,
            target_note_path=alpha,
            related_notes_header="## Suggested Links",
        )

    assert "## Suggested Links" in alpha.read_text()
