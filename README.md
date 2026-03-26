# rhizome

<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/banner_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="assets/banner_light.png">
    <img alt="rhizome banner" src="assets/banner_light.png">
  </picture>
</div>

[![CI](https://github.com/matzalazar/rhizome/actions/workflows/ci.yml/badge.svg)](https://github.com/matzalazar/rhizome/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![runs locally · no cloud](https://img.shields.io/badge/runs%20locally-no%20cloud-brightgreen)](#)

> Semantic backlinks for your notes — generated locally, stored as `[[wikilinks]]`.

Rhizome reads your vault, embeds every note with a multilingual sentence transformer, and writes a `## Related Notes` section at the bottom of each file.
No cloud API, no database, no daemon — the knowledge graph lives entirely in the filesystem and syncs with the rest of your vault.

```
## Related Notes

- [[Zettelkasten]]
- [[Evergreen notes]]
- [[How I take notes]]
```

---

## Features

- **Fully local** — ONNX Runtime on CPU, no GPU or network call after first run
- **Multilingual** — `paraphrase-multilingual-MiniLM-L12-v2` handles 50+ languages
- **Scales automatically** — exact numpy search for small vaults, approximate HNSW for large ones
- **Idempotent** — re-running the pipeline replaces the section, never duplicates it
- **Safe by default** — prompts for a timestamped vault backup before writing anything
- **Dry-run mode** — preview every proposed link without touching a single file
- **Extensible** — a four-method `VaultReader` Protocol is the only contract a new adapter needs

---

## Requirements

- Python 3.10+
- An Obsidian or Logseq vault

---

## Installation

```bash
git clone https://github.com/matzalazar/rhizome
cd rhizome
pip install -e .
```

For development tools (pytest, ruff):

```bash
pip install -e ".[dev]"
```

---

## Quick start

```bash
# 1. Copy the example config and fill in your vault path
cp .env.example .env

# 2. Check vault stats and model cache status
rhizome status

# 3. Run — the model downloads automatically on first use (~250 MB, once)
rhizome run
```

On first run you will be asked:

```
  Vault path  : /home/you/notes
  Notes found : 312
  Do you want to create a backup before proceeding? [Y/n]:
```

The backup is written to `{vault}/../.rhizome_backups/backup_YYYYMMDD_HHMMSS/`
and can be restored interactively with `rhizome restore`.

---

## Configuration

All settings are read from environment variables or a `.env` file.

| Variable               | Default                                          | Description                                          |
|------------------------|--------------------------------------------------|------------------------------------------------------|
| `VAULT_PATH`           | *(required)*                                     | Absolute path to your vault directory                |
| `VAULT_APP`            | `obsidian`                                       | Adapter to use: `obsidian` or `logseq`               |
| `SIMILARITY_THRESHOLD` | `0.75`                                           | Minimum cosine similarity to create a link (0–1)     |
| `TOP_K`                | `5`                                              | Maximum related notes to surface per note            |
| `MODEL_DIR`            | `./models`                                       | Directory for the cached ONNX model files            |
| `MODEL_NAME`           | `Xenova/paraphrase-multilingual-MiniLM-L12-v2`  | HuggingFace model identifier (Xenova ONNX exports). If you change this, clear `MODEL_DIR` first to avoid stale cache. |
| `LOG_DIR`              | `./logs`                                         | Directory where per-run JSON logs are written        |
| `DRY_RUN`              | `false`                                          | Preview proposed links without modifying any files   |

---

## CLI reference

```
rhizome run              Execute the full pipeline
rhizome status           Show vault stats and model cache status
rhizome clean            Remove all generated ## Related Notes sections
rhizome download-model   Pre-cache the ONNX model (useful for CI / Docker)
rhizome backups          List available backups with metadata
rhizome restore          Interactively restore a previous backup
```

Every command accepts `--verbose` / `-v` to enable debug logging.

### Log output format

All log output uses a symbol-based format:

| Symbol | Level    |
|--------|----------|
| `[.]`  | debug    |
| `[i]`  | info     |
| `[-]`  | success  |
| `[!]`  | warning  |
| `[x]`  | error    |
| `[!!]` | critical |

### Verbose mode

```bash
rhizome run --verbose
```

Prints one debug line per modified note showing the note title and the links that were written:

```
05:33:45 [.] Zettelkasten → [[Evergreen notes]], [[How I take notes]], [[PKM]]
05:33:45 [.] Atomic habits → [[Deep work]], [[Focus]], [[GTD]]
```

### Dry run

```bash
DRY_RUN=true rhizome run
```

Logs every proposed link without writing to disk. Safe to run at any time.

### Pre-caching the model

```bash
rhizome download-model
```

Downloads and exports the model to `MODEL_DIR` without touching the vault.
Useful as a dedicated step in a Docker build or CI pipeline so the model layer
is cached separately from the application code.

---

## How it works

See [docs/architecture.md](docs/architecture.md) for a walkthrough of the pipeline,
including the embedding strategy, similarity backends, and section writing logic.

---

## Backup and restore

Before modifying any files, `rhizome run` prompts to create a timestamped backup.
Backups are stored at `{vault}/../.rhizome_backups/backup_YYYYMMDD_HHMMSS/` and
include a manifest with vault path, timestamp, note count, and rhizome version.

```bash
rhizome backups    # list all backups
rhizome restore    # select and restore interactively
```

Restore overwrites files present in the backup but leaves files created after the
backup untouched.

---

## Supported apps

| App      | Status | Link format      |
|----------|--------|------------------|
| Obsidian | Stable | `[[wikilinks]]`  |
| Logseq   | Stable | `[[wikilinks]]`  |

Both adapters generate the same `## Related Notes` section with `[[wikilinks]]`.
Logseq renders markdown headers and bullet lists natively in both outline and
document view, so the format works without any special configuration.

---

## Project structure

```
src/rhizome/
├── config.py            pydantic-settings — validated at startup
├── pipeline.py          orchestration (run / clean / status)
├── main.py              CLI bootstrap (load_dotenv → delegate to cli/)
│
├── inference/
│   ├── model.py         ONNX export, mean pooling, L2 normalisation
│   └── similarity.py    SimilarityStrategy Protocol · Numpy · HNSW
│
├── vault/
│   ├── base.py          Note dataclass · VaultReader Protocol
│   ├── obsidian.py      Obsidian adapter + module-level helpers
│   ├── logseq.py        Logseq adapter
│   ├── backup.py        create / list / restore backups
│   └── __init__.py      get_vault_reader() factory · public re-exports
│
└── cli/
    └── commands.py      Typer app with all command definitions
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, testing, and how to add a new vault adapter.

---

## License

MIT
