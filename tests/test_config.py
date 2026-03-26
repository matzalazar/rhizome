"""
Tests for config.py — focused on the human-readable SIMILARITY_THRESHOLD
levels and the similarity_level property.
"""

import pytest
from pydantic import ValidationError

from rhizome.config import Settings


@pytest.fixture()
def vault(tmp_path):
    """Minimal valid vault directory."""
    (tmp_path / "note.md").write_text("# Note\nContent.")
    return tmp_path


# ---------------------------------------------------------------------------
# Named level → float resolution
# ---------------------------------------------------------------------------


def test_threshold_level_low(vault):
    s = Settings(vault_path=vault, similarity_threshold="low")
    assert s.similarity_threshold == 0.60


def test_threshold_level_medium(vault):
    s = Settings(vault_path=vault, similarity_threshold="medium")
    assert s.similarity_threshold == 0.75


def test_threshold_level_high(vault):
    s = Settings(vault_path=vault, similarity_threshold="high")
    assert s.similarity_threshold == 0.88


def test_threshold_level_case_insensitive(vault):
    s = Settings(vault_path=vault, similarity_threshold="HIGH")
    assert s.similarity_threshold == 0.88


# ---------------------------------------------------------------------------
# Numeric pass-through (backward compatibility)
# ---------------------------------------------------------------------------


def test_threshold_float_passthrough(vault):
    s = Settings(vault_path=vault, similarity_threshold=0.85)
    assert s.similarity_threshold == pytest.approx(0.85)


def test_threshold_float_string(vault):
    """A numeric string like '0.85' should be accepted as 0.85."""
    s = Settings(vault_path=vault, similarity_threshold="0.85")
    assert s.similarity_threshold == pytest.approx(0.85)


def test_threshold_zero_accepted(vault):
    s = Settings(vault_path=vault, similarity_threshold=0.0)
    assert s.similarity_threshold == 0.0


def test_threshold_one_accepted(vault):
    s = Settings(vault_path=vault, similarity_threshold=1.0)
    assert s.similarity_threshold == 1.0


# ---------------------------------------------------------------------------
# Invalid input raises ValidationError
# ---------------------------------------------------------------------------


def test_threshold_invalid_string_raises(vault):
    with pytest.raises(ValidationError) as exc_info:
        Settings(vault_path=vault, similarity_threshold="very_high")
    assert "SIMILARITY_THRESHOLD" in str(exc_info.value)


def test_threshold_invalid_string_lists_valid_options(vault):
    with pytest.raises(ValidationError) as exc_info:
        Settings(vault_path=vault, similarity_threshold="extreme")
    error_text = str(exc_info.value)
    assert '"low"' in error_text
    assert '"medium"' in error_text
    assert '"high"' in error_text


def test_threshold_out_of_range_raises(vault):
    with pytest.raises(ValidationError):
        Settings(vault_path=vault, similarity_threshold=1.5)


def test_threshold_negative_raises(vault):
    with pytest.raises(ValidationError):
        Settings(vault_path=vault, similarity_threshold=-0.1)


# ---------------------------------------------------------------------------
# similarity_level property
# ---------------------------------------------------------------------------


def test_similarity_level_low(vault):
    s = Settings(vault_path=vault, similarity_threshold="low")
    assert s.similarity_level == "low"


def test_similarity_level_medium(vault):
    s = Settings(vault_path=vault, similarity_threshold="medium")
    assert s.similarity_level == "medium"


def test_similarity_level_high(vault):
    s = Settings(vault_path=vault, similarity_threshold="high")
    assert s.similarity_level == "high"


def test_similarity_level_custom_for_raw_float(vault):
    s = Settings(vault_path=vault, similarity_threshold=0.80)
    assert s.similarity_level == "custom"


def test_similarity_level_custom_for_numeric_string(vault):
    s = Settings(vault_path=vault, similarity_threshold="0.80")
    assert s.similarity_level == "custom"
