"""
Configuration loaded from environment variables (or a .env file).

Using pydantic-settings so every field is validated at startup —
a missing or malformed VAULT_PATH fails fast rather than crashing mid-run.
"""

from pathlib import Path
from typing import Annotated

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# Named presets for SIMILARITY_THRESHOLD.
# Users may write SIMILARITY_THRESHOLD=medium instead of SIMILARITY_THRESHOLD=0.75.
_THRESHOLD_LEVELS: dict[str, float] = {
    "low": 0.60,
    "medium": 0.75,
    "high": 0.88,
}

_MANUAL_OVERRIDE_FIELD_ALIASES: dict[str, str] = {
    "top_k": "top_k",
    "top-k": "top_k",
    "similarity_threshold": "similarity_threshold",
    "similarity-threshold": "similarity_threshold",
    "threshold": "similarity_threshold",
    "chunk_size": "chunk_size",
    "chunk-size": "chunk_size",
    "chunk_overlap": "chunk_overlap",
    "chunk-overlap": "chunk_overlap",
    "related_notes_header": "related_notes_header",
    "header": "related_notes_header",
}
_DEFAULT_MANUAL_OVERRIDE_FIELDS = [
    "top_k",
    "similarity_threshold",
    "chunk_size",
    "chunk_overlap",
    "related_notes_header",
]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    vault_path: Path
    vault_app: str = "obsidian"
    similarity_threshold: float = 0.75
    top_k: int = 5
    model_dir: Path = Path("./models")
    model_name: str = "Xenova/paraphrase-multilingual-MiniLM-L12-v2"
    dry_run: bool = False
    log_dir: Path = Path("./logs")
    exclude_dirs: Annotated[list[str], NoDecode] = []
    include_dirs: Annotated[list[str], NoDecode] = []
    chunk_size: int = 512
    chunk_overlap: int = 32
    manual_override_fields: Annotated[list[str], NoDecode] = (
        _DEFAULT_MANUAL_OVERRIDE_FIELDS.copy()
    )

    @field_validator("exclude_dirs", mode="before")
    @classmethod
    def parse_exclude_dirs(cls, v: object) -> list[str]:
        """Accept a comma-separated string (env var) or a list (Python)."""
        if isinstance(v, str):
            return [d.strip() for d in v.split(",") if d.strip()]
        if isinstance(v, list):
            return [str(item).strip() for item in v if str(item).strip()]
        return []

    @field_validator("include_dirs", mode="before")
    @classmethod
    def parse_include_dirs(cls, v: object) -> list[str]:
        """Accept a comma-separated string (env var) or a list (Python)."""
        if isinstance(v, str):
            return [d.strip() for d in v.split(",") if d.strip()]
        if isinstance(v, list):
            return [str(item).strip() for item in v if str(item).strip()]
        return []

    @field_validator("manual_override_fields", mode="before")
    @classmethod
    def parse_manual_override_fields(cls, v: object) -> list[str]:
        """Accept a comma-separated string (env var) or a list (Python)."""
        if isinstance(v, str):
            raw_items = [item.strip() for item in v.split(",") if item.strip()]
        elif isinstance(v, list):
            raw_items = [str(item).strip() for item in v if str(item).strip()]
        elif v is None:
            return []
        else:
            raw_items = [str(v).strip()]

        resolved: list[str] = []
        unknown: list[str] = []
        for item in raw_items:
            canonical = _MANUAL_OVERRIDE_FIELD_ALIASES.get(item.lower())
            if canonical is None:
                unknown.append(item)
                continue
            if canonical not in resolved:
                resolved.append(canonical)

        if unknown:
            valid_values = ", ".join(sorted(_MANUAL_OVERRIDE_FIELD_ALIASES))
            raise ValueError(
                "MANUAL_OVERRIDE_FIELDS contains unsupported values: "
                f"{', '.join(unknown)}. Supported values: {valid_values}"
            )

        return resolved

    @field_validator("chunk_size")
    @classmethod
    def chunk_size_valid(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"CHUNK_SIZE must be 0 (disabled) or >= 1, got: {v}")
        return v

    @field_validator("chunk_overlap")
    @classmethod
    def chunk_overlap_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"CHUNK_OVERLAP must be >= 0, got: {v}")
        return v

    @model_validator(mode="after")
    def chunk_overlap_less_than_chunk_size(self) -> "Settings":
        if self.chunk_size > 0 and self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"CHUNK_OVERLAP ({self.chunk_overlap}) must be less than "
                f"CHUNK_SIZE ({self.chunk_size})"
            )
        return self

    @field_validator("vault_path")
    @classmethod
    def vault_must_exist(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"VAULT_PATH does not exist: {v}")
        if not v.is_dir():
            raise ValueError(f"VAULT_PATH must be a directory, got: {v}")
        return v.resolve()

    @field_validator("vault_app")
    @classmethod
    def vault_app_supported(cls, v: str) -> str:
        supported = {"obsidian", "logseq"}
        normalised = v.lower()
        if normalised not in supported:
            raise ValueError(f"VAULT_APP must be one of {sorted(supported)}, got: {v!r}")
        return normalised

    @field_validator("similarity_threshold", mode="before")
    @classmethod
    def resolve_threshold(cls, v: object) -> object:
        """Accept named levels ('low', 'medium', 'high') or a numeric string / float."""
        if isinstance(v, str):
            lowered = v.strip().lower()
            if lowered in _THRESHOLD_LEVELS:
                return _THRESHOLD_LEVELS[lowered]
            try:
                return float(lowered)
            except ValueError as err:
                valid_levels = ", ".join(f'"{k}"' for k in _THRESHOLD_LEVELS)
                raise ValueError(
                    f"SIMILARITY_THRESHOLD must be one of {valid_levels}, "
                    f"or a float in [0, 1]. Got: {v!r}"
                ) from err
        return v  # let pydantic coerce numeric types

    @field_validator("similarity_threshold")
    @classmethod
    def threshold_in_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"SIMILARITY_THRESHOLD must be in [0, 1], got: {v}")
        return v

    @field_validator("top_k")
    @classmethod
    def top_k_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"TOP_K must be >= 1, got: {v}")
        return v

    @field_validator("model_name")
    @classmethod
    def model_name_valid(cls, v: str) -> str:
        if "/" not in v:
            raise ValueError(
                f"MODEL_NAME must be in 'org/model' format "
                f"(e.g. Xenova/paraphrase-multilingual-MiniLM-L12-v2), got: {v!r}"
            )
        return v

    @property
    def similarity_level(self) -> str:
        """Human-readable label for the current threshold ('low', 'medium', 'high', or 'custom')."""
        reverse = {v: k for k, v in _THRESHOLD_LEVELS.items()}
        return reverse.get(self.similarity_threshold, "custom")


def load_settings() -> Settings:
    """Entry point for obtaining validated configuration."""
    return Settings()  # type: ignore[call-arg]
