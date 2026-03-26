# Contributing

## Setup

```bash
pip install -e ".[dev]"
```

## Running tests

```bash
pytest                        # run the full test suite
pytest -v tests/test_vault.py # run a single file
```

Tests cover vault parsing, VaultReader Protocol conformance, similarity
strategy correctness, and `mean_pool` with known numerical inputs — no
model download or GPU required.

## Linting and formatting

```bash
ruff check src tests
ruff format src tests
```

## Adding a new vault adapter

The only contract a new adapter must satisfy is the `VaultReader` Protocol:

```python
from typing import Iterator
from rhizome.vault.base import Note, VaultReader

class MyAppVaultReader:
    def discover(self) -> Iterator[Note]: ...
    def write_links(self, note: Note, links: list[str]) -> None: ...
    def clean_links(self, note: Note) -> None: ...
    def app_name(self) -> str: ...
```

Then:
1. Register it in `vault/__init__.py` → `get_vault_reader()`
2. Add the new value to the `VAULT_APP` validator in `config.py`

No other files change.
