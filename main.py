from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils import (
    DEFAULT_MODEL_NAME,
    dataset_summary,
    ensure_parent_dir,
    load_telegram_export,
    preprocess_df,
    public_method_name,
    summarize_cluster_labels,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Canonical experiment pipeline for the Telegram recurring-phrase clustering study."
    )
    parser.add_argument("--json-path", default="data/result.json", help="Path to the private Telegram export JSON.")
    parser.add_argument(
        "--embeddings-path",
        default="artifacts/private/embeddings.npy",
        help="Path to a cached embedding matrix. It will be created if missing.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="SentenceTransformer model used to compute embeddings when the cache is missing.",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=6,
        help="Minimum cleaned message length kept for clustering.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess_parser = subparsers.add_parser("preprocess", help="Summarize dataset filtering without running models.")
    preprocess_parser.add_argument(
        "--output",
        default="artifacts/private/dataset_summary.json",
        help="Where to store the dataset summary JSON.",
    )

    baseline_parser = subparsers.add_parser("baseline", help="Run K-Means baselines.")
    baseline_parser.add_argument(
        "--clusters",
        nargs="+",
        type=int,
        default=[100, 1000],
        help="List of K values for K-Means.",
    )
    baseline_parser.add_argument(
        "--output",
        default="artifacts/private/baseline_summary.json",
        help="Where to store baseline metrics.",
    )

    density_parser = subparsers.add_parser("density", help="Run density-based clustering experiments.")
    density_parser.add_argument(
        "--dbscan-eps",
        nargs="+",
        type=float,
        default=[0.01, 0.005, 0.001],
        help="DBSCAN epsilon values evaluated after UMAP.",
    )
    density_parser.add_argument(
        "--output",
        default="artifacts/private/density_summary.json",
        help="Where to store density-based metrics.",
    )

    report_parser = subparsers.add_parser("report", help="Run the full pipeline and export report artifacts.")
    report_parser.add_argument(
        "--output-dir",
        default="artifacts/private",
        help="Directory where all private report artifacts will be written.",
    )
    report_parser.add_argument(
        "--dbscan-eps",
        nargs="+",
        type=float,
        default=[0.01, 0.005, 0.001],
        help="DBSCAN epsilon values evaluated after UMAP.",
    )
    report_parser.add_argument(
        "--clusters",
        nargs="+",
        type=int,
        default=[100, 1000],
        help="List of K values for K-Means.",
    )
    return parser.parse_args()


def make_hdbscan(**kwargs: Any):
    try:
        from sklearn.cluster import HDBSCAN as HDBSCANClusterer
    except ImportError:
        from hdbscan import HDBSCAN as HDBSCANClusterer

    return HDBSCANClusterer(**kwargs)


def compute_or_load_embeddings(texts: list[str], embeddings_path: str | Path, model_name: str) -> np.ndarray:
    embeddings_file = Path(embeddings_path)
    if embeddings_file.exists():
        return np.load(embeddings_file)

    from sentence_transformers import SentenceTransformer

    ensure_parent_dir(embeddings_file)
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True)
    np.save(embeddings_file, embeddings)
    return embeddings


def make_umap():
    import umap

    return umap.UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = ensure_parent_dir(path)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_private_dataset(json_path: str | Path, min_chars: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw_df = load_telegram_export(json_path)
    processed_df = preprocess_df(raw_df, min_chars=min_chars)
    summary = dataset_summary(processed_df, total_messages=len(raw_df), min_chars=min_chars)
    return processed_df, summary


def run_kmeans_baseline(embeddings: np.ndarray, cluster_counts: list[int]) -> dict[str, dict[str, Any]]:
    from sklearn.cluster import KMeans

    results: dict[str, dict[str, Any]] = {}
    for cluster_count in cluster_counts:
        model = KMeans(n_clusters=cluster_count, random_state=42)
        labels = model.fit_predict(embeddings)
        key = f"kmeans_k{cluster_count}"
        results[key] = {
            "method": key,
            "display_name": public_method_name(key),
            "parameters": {"n_clusters": cluster_count, "random_state": 42},
            **summarize_cluster_labels(labels, noise_label=-999999),
        }
        results[key]["noise_messages"] = 0
    return results


def run_hdbscan_raw(embeddings: np.ndarray) -> dict[str, Any]:
    clusterer = make_hdbscan(min_cluster_size=5)
    labels = clusterer.fit_predict(embeddings)
    return {
        "method": "hdbscan_raw",
        "display_name": public_method_name("hdbscan_raw"),
        "parameters": {"min_cluster_size": 5},
        **summarize_cluster_labels(labels),
    }


def run_umap_hdbscan(embeddings: np.ndarray) -> dict[str, Any]:
    reduced_embeddings = make_umap().fit_transform(embeddings)
    clusterer = make_hdbscan(min_cluster_size=5)
    labels = clusterer.fit_predict(reduced_embeddings)
    summary = {
        "method": "umap_hdbscan",
        "display_name": public_method_name("umap_hdbscan"),
        "parameters": {
            "umap": {"n_neighbors": 15, "n_components": 5, "min_dist": 0.0, "metric": "cosine"},
            "hdbscan": {"min_cluster_size": 5},
        },
        **summarize_cluster_labels(labels),
    }
    return summary


def summarize_topic_model(topic_info: pd.DataFrame, method_key: str, parameters: dict[str, Any]) -> dict[str, Any]:
    counts = topic_info.set_index("Topic")["Count"]
    cluster_sizes = counts.drop(labels=-1, errors="ignore")
    noise_count = int(counts.get(-1, 0))
    return {
        "method": method_key,
        "display_name": public_method_name(method_key),
        "parameters": parameters,
        "num_clusters": int(cluster_sizes.shape[0]),
        "assigned_messages": int(cluster_sizes.sum()),
        "noise_messages": noise_count,
        "cluster_size_median": float(cluster_sizes.median()) if not cluster_sizes.empty else 0.0,
        "cluster_size_mean": float(cluster_sizes.mean()) if not cluster_sizes.empty else 0.0,
        "cluster_size_min": int(cluster_sizes.min()) if not cluster_sizes.empty else 0,
        "cluster_size_max": int(cluster_sizes.max()) if not cluster_sizes.empty else 0,
        "top_cluster_sizes": [int(value) for value in cluster_sizes.sort_values(ascending=False).head(5).tolist()],
    }


def run_bertopic_hdbscan(texts: list[str], embeddings: np.ndarray) -> dict[str, Any]:
    from bertopic import BERTopic

    topic_model = BERTopic(
        embedding_model=None,
        hdbscan_model=make_hdbscan(min_cluster_size=5),
        verbose=True,
    )
    topic_model.fit_transform(texts, embeddings)
    topic_info = topic_model.get_topic_info()
    return summarize_topic_model(topic_info, "bertopic_hdbscan", {"hdbscan": {"min_cluster_size": 5}})


def run_dbscan_sweep(texts: list[str], embeddings: np.ndarray, eps_values: list[float]) -> dict[str, dict[str, Any]]:
    from bertopic import BERTopic
    from sklearn.cluster import DBSCAN

    results: dict[str, dict[str, Any]] = {}
    for eps in eps_values:
        clusterer = DBSCAN(eps=eps, min_samples=3)
        topic_model = BERTopic(
            embedding_model=None,
            umap_model=make_umap(),
            hdbscan_model=clusterer,
            verbose=True,
        )
        topic_model.fit_transform(texts, embeddings)
        topic_info = topic_model.get_topic_info()
        key = f"dbscan_eps_{eps}"
        results[key] = summarize_topic_model(
            topic_info,
            key,
            {"dbscan": {"eps": eps, "min_samples": 3}, "input_space": "UMAP(15,5,0.0,cosine)"},
        )
    return results


def preprocess_command(args: argparse.Namespace) -> None:
    _, summary = load_private_dataset(args.json_path, args.min_chars)
    write_json(args.output, {"dataset": summary})


def baseline_command(args: argparse.Namespace) -> None:
    df, summary = load_private_dataset(args.json_path, args.min_chars)
    embeddings = compute_or_load_embeddings(df["clean_text"].tolist(), args.embeddings_path, args.model_name)
    payload = {
        "dataset": summary,
        "experiments": list(run_kmeans_baseline(embeddings, args.clusters).values()),
    }
    write_json(args.output, payload)


def density_command(args: argparse.Namespace) -> None:
    df, summary = load_private_dataset(args.json_path, args.min_chars)
    texts = df["clean_text"].tolist()
    embeddings = compute_or_load_embeddings(texts, args.embeddings_path, args.model_name)

    hdbscan_raw = run_hdbscan_raw(embeddings)
    umap_hdbscan = run_umap_hdbscan(embeddings)
    bertopic_hdbscan = run_bertopic_hdbscan(texts, embeddings)
    dbscan_results = run_dbscan_sweep(texts, embeddings, args.dbscan_eps)

    payload = {
        "dataset": summary,
        "experiments": [hdbscan_raw, umap_hdbscan, bertopic_hdbscan, *dbscan_results.values()],
    }
    write_json(args.output, payload)


def report_command(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    df, summary = load_private_dataset(args.json_path, args.min_chars)
    texts = df["clean_text"].tolist()
    embeddings = compute_or_load_embeddings(texts, args.embeddings_path, args.model_name)

    baseline_results = run_kmeans_baseline(embeddings, args.clusters)
    hdbscan_raw = run_hdbscan_raw(embeddings)
    umap_hdbscan = run_umap_hdbscan(embeddings)
    bertopic_hdbscan = run_bertopic_hdbscan(texts, embeddings)
    dbscan_results = run_dbscan_sweep(texts, embeddings, args.dbscan_eps)

    experiments = [
        *baseline_results.values(),
        hdbscan_raw,
        umap_hdbscan,
        bertopic_hdbscan,
        *dbscan_results.values(),
    ]

    payload = {"dataset": summary, "experiments": experiments}
    write_json(output_dir / "experiment_summary.json", payload)

    dataset_path = ensure_parent_dir(output_dir / "dataset_summary.json")
    dataset_path.write_text(json.dumps({"dataset": summary}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    command_handlers = {
        "preprocess": preprocess_command,
        "baseline": baseline_command,
        "density": density_command,
        "report": report_command,
    }
    command_handlers[args.command](args)


if __name__ == "__main__":
    main()
