# How it works

## 1 · Note discovery

`vault.discover()` walks `VAULT_PATH` recursively, collecting `.md` files and
skipping internal directories (`.obsidian`, `.git`, `.rhizome_backups`, etc.).

## 2 · Text preparation

Before embedding, each note is stripped of:

- **YAML/TOML frontmatter** — metadata that describes the note rather than its content
- **`[[wikilinks]]`** — existing links would bias embeddings toward already-connected
  notes, obscuring genuinely similar but currently unlinked ones

## 3 · Sentence embeddings

Notes are encoded with
[`paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
exported to ONNX via `optimum`. The model is downloaded once and cached locally;
all subsequent runs are fully offline.

**Why mean pooling over the CLS token?**

A transformer produces one embedding per input token. To obtain a single
sentence-level vector we need to aggregate across the sequence dimension.

The `[CLS]` token was designed for *classification* pre-training. Models in the
`sentence-transformers` family are instead fine-tuned with a *pooling objective*
over the full sequence, so CLS is not a reliable summary of meaning.

Mean pooling with attention mask weighting averages the hidden states of all
*real* (non-padding) tokens:

$$\mathbf{e} = \frac{\displaystyle\sum_{i=1}^{L} m_i \cdot \mathbf{h}_i}{\displaystyle\sum_{i=1}^{L} m_i}$$

where $\mathbf{h}_i$ is the hidden state at position $i$, $m_i \in \{0,1\}$
is the attention mask value, and $L$ is the padded sequence length. Padding
tokens contribute zero to both numerator and denominator and therefore cannot
dilute the result.

## 4 · Similarity search

Embeddings are L2-normalised at inference time, projecting them onto the unit
hypersphere. This means **cosine similarity reduces to a plain dot product**,
and length bias is eliminated:

$$\cos(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|\,\|\mathbf{v}\|}$$

Two retrieval backends are available, selected automatically:

| Vault size   | Backend                 | Complexity  |
|--------------|-------------------------|-------------|
| ≤ 500 notes  | `NumpyStrategy` (exact) | O(N²)       |
| > 500 notes  | `HNSWStrategy` (approx) | O(N log N)  |

Both implement the `SimilarityStrategy` Protocol and are interchangeable — the
pipeline never inspects the concrete type.

## 5 · Writing links

For each note, the top-K neighbours above `SIMILARITY_THRESHOLD` are written
as a `## Related Notes` section at the end of the file. If the section already
exists from a previous run it is **replaced**, never duplicated.
