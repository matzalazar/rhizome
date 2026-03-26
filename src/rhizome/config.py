"""
Configuration loaded from environment variables (or a .env file).

Using pydantic-settings so every field is validated at startup —
a missing or malformed VAULT_PATH fails fast rather than crashing mid-run.
"""

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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


def load_settings() -> Settings:
    """Entry point for obtaining validated configuration."""
    return Settings()  # type: ignore[call-arg]
