import json
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def extract_text(text_field: Any) -> str:
    if isinstance(text_field, str):
        return text_field

    if isinstance(text_field, list):
        parts: list[str] = []
        for part in text_field:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(str(part.get("text", "")))
        return "".join(parts)

    return ""


def load_telegram_export(json_path: str | Path) -> pd.DataFrame:
    with Path(json_path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    raw_messages = payload["messages"] if isinstance(payload, dict) and "messages" in payload else payload
    return pd.DataFrame(raw_messages)


def preprocess_df(df: pd.DataFrame, min_chars: int = 6) -> pd.DataFrame:
    columns = [column for column in ("date", "text", "from_id") if column in df.columns]
    df = df.loc[df["type"].eq("message"), columns].copy()

    df["clean_text"] = df["text"].map(extract_text)

    s = df["clean_text"].astype("string")
    mask = s.str.strip().ne("") & s.str.len().ge(min_chars)
    df = df.loc[mask].copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def dataset_summary(df: pd.DataFrame, *, total_messages: int | None = None, min_chars: int = 6) -> dict[str, Any]:
    start_date = df["date"].min()
    end_date = df["date"].max()

    return {
        "messages_before_filtering": total_messages,
        "messages_after_filtering": int(len(df)),
        "unique_authors": int(df["from_id"].nunique()) if "from_id" in df.columns else None,
        "min_chars": min_chars,
        "date_range": {
            "start": start_date.date().isoformat() if pd.notna(start_date) else None,
            "end": end_date.date().isoformat() if pd.notna(end_date) else None,
        },
    }


def summarize_cluster_labels(labels: Any, *, noise_label: int = -1) -> dict[str, Any]:
    counts = pd.Series(labels).value_counts().sort_index()
    cluster_sizes = counts.drop(labels=noise_label, errors="ignore")
    noise_count = int(counts.get(noise_label, 0))

    return {
        "num_clusters": int(cluster_sizes.shape[0]),
        "assigned_messages": int(cluster_sizes.sum()),
        "noise_messages": noise_count,
        "cluster_size_median": float(cluster_sizes.median()) if not cluster_sizes.empty else 0.0,
        "cluster_size_mean": float(cluster_sizes.mean()) if not cluster_sizes.empty else 0.0,
        "cluster_size_min": int(cluster_sizes.min()) if not cluster_sizes.empty else 0,
        "cluster_size_max": int(cluster_sizes.max()) if not cluster_sizes.empty else 0,
        "top_cluster_sizes": [int(value) for value in cluster_sizes.sort_values(ascending=False).head(5).tolist()],
    }


def ensure_parent_dir(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def public_method_name(method_key: str) -> str:
    mapping = {
        "kmeans_k100": "K-Means (k=100)",
        "kmeans_k1000": "K-Means (k=1000)",
        "hdbscan_raw": "HDBSCAN on raw embeddings",
        "umap_hdbscan": "UMAP + HDBSCAN",
        "bertopic_hdbscan": "BERTopic (UMAP + HDBSCAN)",
        "dbscan_eps_0.01": "DBSCAN after UMAP (eps=0.01)",
        "dbscan_eps_0.005": "DBSCAN after UMAP (eps=0.005)",
        "dbscan_eps_0.001": "DBSCAN after UMAP (eps=0.001)",
    }
    return mapping.get(method_key, method_key)
