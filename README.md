# Telegram Meme-Phrase Clustering

This repository is a research-first portfolio project on unsupervised discovery of recurring phrase-like and topic-like patterns in a private Telegram group chat.

## What is the task?

The project studies whether short chat messages form dense semantic neighborhoods in embedding space. The goal is not full trend forecasting, but discovery of recurring conversational patterns such as:

- repeated catchphrases
- phrase templates with small lexical variation
- narrow topic bursts
- generic conversational fillers that clustering methods should ideally separate from more interesting recurring content

## Why is this interesting?

Short informal chat messages are noisy, multilingual, often misspelled, and full of in-group references. That makes them a useful stress test for sentence embeddings and density-based clustering. The project is also deliberately honest about where unsupervised methods help and where they still blur together meme-like phrases, local topics, and generic conversation.

## What is included?

- a canonical CLI pipeline in main.py with four entrypoints:
  - `preprocess`
  - `baseline`
  - `density`
  - `report`
- reusable data-loading and summarization helpers in [utils.py](./utils.py)
- a compact paper-style note in [report/bmvc_final.tex](./report/bmvc_final.tex)
- sanitized experiment summary in [experiment_summary.json](./experiment_summary.json).

## Main findings

- `K-Means` provides a readable baseline, but it mainly discovers broad semantic themes.
- `HDBSCAN` on raw embeddings is too conservative and marks most messages as noise.
- `UMAP + HDBSCAN` improves coverage substantially, but still fragments many related patterns.
- `BERTopic` makes inspection easier, but does not fundamentally change the clustering behavior.
- `DBSCAN` after UMAP gives the strongest control over granularity and is the most useful setup for isolating tight recurring expressions.

The public summary used in the note is available in [experiment_summary.json](./experiment_summary.json).

## Limitations

- The dataset is private and is not published in this repository.
- The evaluation is still partly qualitative.
- Temporal trend detection is intentionally framed as future work, not as a solved contribution here.
- Representative examples in public artifacts are paraphrased or synthetic summaries rather than raw chat messages.

## Private data policy

This public repository is `code only`. The original Telegram export, media files, user identifiers, and raw message excerpts should stay private.

Expected private input layout is documented in [data/README.md](./data/README.md). By default, private inputs and generated caches live under ignored paths such as `data/` and `artifacts/private/`.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Canonical pipeline

1. Summarize the private export and filtering:

```bash
python main.py --json-path data/result.json preprocess --output artifacts/private/dataset_summary.json
```

2. Run the K-Means baselines:

```bash
python main.py --json-path data/result.json --embeddings-path artifacts/private/embeddings.npy baseline
```

3. Run the density-based experiments:

```bash
python main.py --json-path data/result.json --embeddings-path artifacts/private/embeddings.npy density
```

4. Produce the full private report bundle:

```bash
python main.py --json-path data/result.json --embeddings-path artifacts/private/embeddings.npy report --output-dir artifacts/private
```

The `report` command writes:

- `dataset_summary.json`
- `experiment_summary.json`
