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

![Rhizome interface demo](assets/demo1.png)

---

## Features

- **Fully local** — ONNX Runtime on CPU, no GPU or network call after first run
- **Multilingual** — default model handles 50+ languages; swap to any Xenova ONNX export via `MODEL_NAME`
- **Scales automatically** — exact numpy search for small vaults, approximate HNSW for large ones
- **Long-document aware** — notes exceeding 512 tokens are split into overlapping chunks; chunk embeddings are averaged so every section of the note influences its semantic representation
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

On first run, rhizome performs a dry-run preview and asks for confirmation:

```
  Notes to modify  : 47
  Links to write   : 214
  (A timestamped backup will be created before writing.)

  Proceed? [Y/n]:
```

If you confirm, you will also be asked whether to create a backup before writing.

The backup is written to `{vault}/../.rhizome_backups/backup_YYYYMMDD_HHMMSS/`
and can be restored interactively with `rhizome restore`.

---

## Configuration

All settings are read from environment variables or a `.env` file.

| Variable               | Default                                          | Description                                          |
|------------------------|--------------------------------------------------|------------------------------------------------------|
| `VAULT_PATH`           | *(required)*                                     | Absolute path to your vault directory                |
| `VAULT_APP`            | `obsidian`                                       | Adapter to use: `obsidian` or `logseq`               |
| `SIMILARITY_THRESHOLD` | `medium` (0.75)                                  | Minimum cosine similarity. Accepts a float in [0, 1] or a named level: `low` (0.60), `medium` (0.75), `high` (0.88) |
| `TOP_K`                | `5`                                              | Maximum related notes to surface per note            |
| `MODEL_DIR`            | `./models`                                       | Directory for the cached ONNX model files            |
| `MODEL_NAME`           | `Xenova/paraphrase-multilingual-MiniLM-L12-v2`  | HuggingFace model identifier (Xenova ONNX exports). If you change this, clear `MODEL_DIR` first to avoid stale cache. |
| `LOG_DIR`              | `./logs`                                         | Directory where per-run JSON logs are written        |
| `DRY_RUN`              | `false`                                          | Preview proposed links without modifying any files   |
| `MANUAL_OVERRIDE_FIELDS` | `top_k,similarity_threshold,chunk_size,chunk_overlap,related_notes_header` | Comma-separated list of runtime prompts to show in `rhizome run --manual`. Accepts `top_k`, `similarity_threshold`, `chunk_size`, `chunk_overlap`, `related_notes_header`, plus aliases `threshold` and `header`. |
| `EXCLUDE_DIRS`         | *(empty)*                                        | Comma-separated list of directories (relative to `VAULT_PATH`) to skip. Uses prefix matching: `journal` excludes `vault/journal/` but not `vault/project/journal/`. |
| `INCLUDE_DIRS`         | *(empty)*                                        | Comma-separated whitelist of directories to scan exclusively. When set, only files under these paths are processed. `EXCLUDE_DIRS` is applied afterwards, so you can narrow within the whitelist (e.g. `INCLUDE_DIRS=projects` + `EXCLUDE_DIRS=projects/drafts`). Leave empty to process the entire vault. |
| `CHUNK_SIZE`           | `512`                                            | Maximum tokens per chunk when embedding long notes. Notes exceeding this limit are split into overlapping windows and their embeddings averaged into one vector. Set to `0` to disable chunking (notes truncated at 512 tokens). |
| `CHUNK_OVERLAP`        | `32`                                             | Tokens shared between adjacent chunks. Preserves sentence context across chunk boundaries. Must be less than `CHUNK_SIZE`. Ignored when `CHUNK_SIZE=0`. |

> [!IMPORTANT]
> **`CHUNK_SIZE` directly affects embedding time.**
> Each chunk requires a full forward pass through the model.
> With the default (`512`), chunking only triggers for notes longer than ~400 words — typical PKM notes are unaffected.
> Lowering `CHUNK_SIZE` increases the number of chunks per long note and embedding time grows proportionally:
> a 2 000-token note with `CHUNK_SIZE=256` produces ~8 chunks instead of 4.
> Set `CHUNK_SIZE=0` to disable chunking entirely and restore the original truncation behaviour,
> at the cost of losing content beyond the first ~400 words of each note.

---

## Choosing a model

The default model works well for most vaults. If your vault is entirely in one
language, or you want a different speed/quality trade-off, you can switch to any
compatible model by setting `MODEL_NAME`.

| Model | Size | Languages | Best for |
|-------|------|-----------|----------|
| `Xenova/paraphrase-multilingual-MiniLM-L12-v2` | ~250 MB | 50+ | **Default** — mixed-language or unknown vault |
| `Xenova/all-MiniLM-L6-v2` | ~90 MB | English | Fast and lean: English-only vault, low-RAM hardware |
| `Xenova/all-MiniLM-L12-v2` | ~130 MB | English | Balanced: English vault, better recall than L6 |
| `Xenova/all-mpnet-base-v2` | ~430 MB | English | High precision: English vault where quality matters most |
| `Xenova/paraphrase-multilingual-mpnet-base-v2` | ~1.1 GB | 50+ | Best multilingual quality, if disk space allows |

> **When you change `MODEL_NAME`, delete `MODEL_DIR` first.**
> Embeddings from different models are not comparable — mixing them produces
> meaningless similarity scores.

```bash
# Example: switch to a leaner English-only model
MODEL_NAME=Xenova/all-MiniLM-L6-v2
MODEL_DIR=./models   # delete contents of this directory first
```

See [docs/models.md](docs/models.md) for a detailed comparison, a decision guide,
and notes on compatibility requirements.

---

## CLI reference

```
rhizome run              Execute the full pipeline (dry-run preview + confirmation)
rhizome run --yes        Skip confirmation and auto-confirm backup (CI / scripted)
rhizome run --manual     Interactively choose one or more notes to update while matching against the full vault
rhizome audit            Analyze vault connectivity without modifying any file
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

`rhizome run` always shows a dry-run preview before writing.  Confirm at the
prompt to proceed, or press `n` (or Ctrl-C) to abort without touching any file.

For fully non-interactive execution (CI / scripts):

```bash
rhizome run --yes        # skip all prompts, auto-confirm backup
```

To update one or more notes interactively while still comparing them against the
full vault:

```bash
rhizome run --manual
```

Rhizome will ask you to search by filename or path, add notes to a manual target
list, and optionally override the runtime settings configured in
`MANUAL_OVERRIDE_FIELDS` for that run only. The `.env` file remains unchanged.

For example, if you usually only tweak the threshold:

```bash
MANUAL_OVERRIDE_FIELDS=similarity_threshold
```

To preview proposed links without writing anything at all (no prompt):

```bash
DRY_RUN=true rhizome run
```

### Auditing the vault

```bash
rhizome audit
```

Analyzes your vault and reports its connectivity state without modifying anything:

```
[i] Vault audit — /home/you/notes (312 notes)
Connectivity distribution
─────────────────────────
No connections       :  47 notes  ( 15%)
1–2 connections      :  83 notes  ( 27%)
3–5 connections      :  92 notes  ( 30%)
6+  connections      :  90 notes  ( 29%)
Potential new links  : 214  (dry-run to preview them)
Est. notes affected  :  98
[i] Run `rhizome run` to generate links.
```

"Connections" counts existing `[[wikilinks]]` in the `## Related Notes` section only.
"Potential new links" runs the full embedding + similarity pipeline in memory — no files
are written.

### Pre-caching the model

```bash
rhizome download-model
```

Downloads and exports the model to `MODEL_DIR` without touching the vault.
Useful as a dedicated step in a Docker build or CI pipeline so the model layer
is cached separately from the application code.

---

## How it works

- [docs/architecture.md](docs/architecture.md) — pipeline walkthrough: embedding strategy, similarity backends, section writing logic
- [docs/models.md](docs/models.md) — model selection guide: compatibility, comparison table, decision criteria, switching instructions

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
