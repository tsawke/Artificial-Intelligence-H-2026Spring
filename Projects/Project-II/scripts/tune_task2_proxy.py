import json
import pickle
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK1_DIR = PROJECT_ROOT / "project2_code" / "task1"
TASK2_DIR = PROJECT_ROOT / "project2_code" / "task2"
RESULTS_PATH = PROJECT_ROOT / "experiment_results.json"


def load_pickle(path):
    with path.open("rb") as file:
        return pickle.load(file)


def squared_distances(X):
    norms = np.sum(X * X, axis=1, keepdims=True)
    distances = np.maximum(norms + norms.T - 2.0 * (X @ X.T), 0.0)
    np.fill_diagonal(distances, np.inf)
    return distances


def top5(distances):
    return np.argpartition(distances, 5, axis=1)[:, :5]


def same_class_score(top_indices, labels):
    return float(np.mean(labels[top_indices] == labels[:, None]))


def rank_scores(distances, candidates=50):
    top = np.argpartition(distances, candidates - 1, axis=1)[:, :candidates]
    top_distances = np.take_along_axis(distances, top, axis=1)
    order = np.argsort(top_distances, axis=1)
    ranked = np.take_along_axis(top, order, axis=1)
    scores = np.zeros_like(distances)
    scores[np.arange(distances.shape[0])[:, None], ranked] = 1.0 / (
        np.arange(candidates) + 1.0
    )
    return scores


def main():
    repository = load_pickle(TASK2_DIR / "image_retrieval_repository_data.pkl")
    label_data = load_pickle(TASK1_DIR / "classification_train_label.pkl")
    label_map = {int(idx): int(label) for idx, label in label_data}
    labels = np.array([label_map[int(idx)] for idx in repository[:, 0]], dtype=int)

    X = repository[:, 1:].astype(np.float64)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    Z = (X - mean) / std
    X_l2 = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    Z_l2 = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)

    raw = squared_distances(X)
    standardized = squared_distances(Z)
    cosine = 1.0 - X_l2 @ X_l2.T
    standardized_cosine = 1.0 - Z_l2 @ Z_l2.T
    np.fill_diagonal(cosine, np.inf)
    np.fill_diagonal(standardized_cosine, np.inf)

    matrices = [raw, standardized, cosine, standardized_cosine]
    names = ["raw", "standardized", "cosine", "standardized_cosine"]
    simple = {
        name: same_class_score(top5(distances), labels)
        for name, distances in zip(names, matrices)
    }
    print(json.dumps(simple, indent=2), flush=True)

    result = {
        "simple": simple,
        "selected_metric": {
            "name": "standardized_euclidean",
            "same_class_top5": simple["standardized"],
            "raw_baseline_same_class_top5": simple["raw"],
        },
    }
    print(json.dumps(result, indent=2), flush=True)

    data = {}
    if RESULTS_PATH.exists():
        data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    data["task2_proxy"] = result
    RESULTS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
