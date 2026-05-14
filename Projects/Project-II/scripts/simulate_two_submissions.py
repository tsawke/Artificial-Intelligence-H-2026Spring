import contextlib
import importlib.util
import json
import os
import pickle
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK1_DIR = PROJECT_ROOT / "project2_code" / "task1"
TASK2_DIR = PROJECT_ROOT / "project2_code" / "task2"
TASK3_DIR = PROJECT_ROOT / "project2_code" / "task3"
TEMP_ROOT = PROJECT_ROOT / "Comparison" / "_tmp_eval"
TEMP_ROOT.mkdir(parents=True, exist_ok=True)

SUBMISSIONS = {
    "12411454": PROJECT_ROOT / "Comparison" / "12411454",
    "oj_submission": PROJECT_ROOT / "oj_submission",
}


def load_pickle(path):
    with path.open("rb") as file:
        return pickle.load(file)


def save_pickle(path, obj):
    with path.open("wb") as file:
        pickle.dump(obj, file)


@contextlib.contextmanager
def chdir(path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def task1_baseline_and_data():
    data = load_pickle(TASK1_DIR / "classification_train_data.pkl")
    labels = load_pickle(TASK1_DIR / "classification_train_label.pkl")
    y = labels[:, 1:].reshape(-1).astype(int)
    train_idx, val_idx = train_test_split(
        np.arange(data.shape[0]), test_size=0.2, random_state=123, stratify=y
    )

    X_train = data[train_idx, 1:].astype(np.float64)
    X_val = data[val_idx, 1:].astype(np.float64)
    y_train = y[train_idx]
    y_val = y[val_idx]

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_val_std = scaler.transform(X_val)
    model = LogisticRegression(C=0.1, max_iter=1000, solver="lbfgs", n_jobs=-1)
    start = time.time()
    model.fit(X_train_std, y_train)
    baseline_acc = accuracy_score(y_val, model.predict(X_val_std))

    return {
        "data": data,
        "labels": labels,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "X_val": X_val,
        "y_val": y_val,
        "baseline_accuracy": float(baseline_acc),
        "baseline_seconds": time.time() - start,
    }


def evaluate_task1(submission_name, submission_path, split):
    with tempfile.TemporaryDirectory(dir=str(TEMP_ROOT)) as tmp:
        tmp_path = Path(tmp)
        save_pickle(tmp_path / "classification_train_data.pkl", split["data"][split["train_idx"]])
        save_pickle(tmp_path / "classification_train_label.pkl", split["labels"][split["train_idx"]])

        module = load_module(f"{submission_name}_classifier", submission_path / "classifier.py")
        with chdir(tmp_path):
            start = time.time()
            classifier = module.Classifier()
            init_seconds = time.time() - start
            start = time.time()
            pred = classifier.inference(split["X_val"])
            inference_seconds = time.time() - start

    pred = np.asarray(pred).reshape(-1)
    return {
        "accuracy": float(accuracy_score(split["y_val"], pred)),
        "beats_baseline": bool(accuracy_score(split["y_val"], pred) > split["baseline_accuracy"]),
        "init_seconds": init_seconds,
        "inference_seconds": inference_seconds,
        "prediction_shape": list(pred.shape),
    }


def pairwise_sqdist(X, Y):
    X_norm = np.sum(X * X, axis=1, keepdims=True)
    Y_norm = np.sum(Y * Y, axis=1, keepdims=True).T
    return np.maximum(X_norm + Y_norm - 2.0 * (X @ Y.T), 0.0)


def raw_baseline_retrieve(repository, queries):
    distances = pairwise_sqdist(queries, repository)
    top = np.argpartition(distances, 5, axis=1)[:, :5]
    order = np.argsort(np.take_along_axis(distances, top, axis=1), axis=1)
    return np.take_along_axis(top, order, axis=1)


def labels_for_repository(repository_data):
    train_labels = load_pickle(TASK1_DIR / "classification_train_label.pkl")
    label_map = {int(idx): int(label) for idx, label in train_labels}
    return np.array([label_map[int(idx)] for idx in repository_data[:, 0]], dtype=int)


def same_class_score(top_indices, repository_labels, query_labels):
    return float(np.mean(repository_labels[top_indices] == query_labels[:, None]))


def shift_features(X, dy, dx):
    images = X.reshape(-1, 16, 16)
    shifted = np.zeros_like(images)
    src_y0 = max(0, -dy)
    src_y1 = min(16, 16 - dy)
    dst_y0 = max(0, dy)
    dst_y1 = min(16, 16 + dy)
    src_x0 = max(0, -dx)
    src_x1 = min(16, 16 - dx)
    dst_x0 = max(0, dx)
    dst_x1 = min(16, 16 + dx)
    shifted[:, dst_y0:dst_y1, dst_x0:dst_x1] = images[:, src_y0:src_y1, src_x0:src_x1]
    return shifted.reshape(X.shape)


def blur_features(X):
    images = X.reshape(-1, 16, 16)
    padded = np.pad(images, ((0, 0), (1, 1), (1, 1)), mode="constant")
    blurred = np.zeros_like(images)
    for dy in range(3):
        for dx in range(3):
            blurred += padded[:, dy : dy + 16, dx : dx + 16]
    return (blurred / 9.0).reshape(X.shape)


def task2_baselines(repository_data, sample_count=256):
    repository = repository_data[:, 1:].astype(np.float64)
    labels = labels_for_repository(repository_data)

    sample = repository[:sample_count]
    sample_labels = labels[:sample_count]
    distances = pairwise_sqdist(sample, repository)
    for idx in range(sample_count):
        distances[idx, idx] = np.inf
    top = np.argpartition(distances, 5, axis=1)[:, :5]
    raw_proxy = same_class_score(top, labels, sample_labels)

    original = np.arange(sample_count)
    transformed_sets = {
        "shift_right_1": shift_features(sample, 0, 1),
        "shift_down_1": shift_features(sample, 1, 0),
        "rotate_90": np.rot90(sample.reshape(-1, 16, 16), 1, axes=(1, 2)).reshape(sample.shape),
        "blur": blur_features(sample),
    }
    transform_baselines = {}
    for name, queries in transformed_sets.items():
        retrieved = raw_baseline_retrieve(repository, queries)
        transform_baselines[name] = float(np.mean([idx in row for idx, row in zip(original, retrieved)]))

    return {
        "repository": repository,
        "labels": labels,
        "sample_count": sample_count,
        "raw_same_class_proxy": float(raw_proxy),
        "transformed_sets": transformed_sets,
        "transform_baselines": transform_baselines,
    }


def evaluate_task2(submission_name, submission_path, repository_data, baselines):
    module = load_module(f"{submission_name}_retrieval", submission_path / "retrieval.py")
    start = time.time()
    retrieval = module.Retrieval(repository_data)
    init_seconds = time.time() - start

    repository = baselines["repository"]
    labels = baselines["labels"]
    sample_count = baselines["sample_count"]
    sample = repository[:sample_count]
    sample_labels = labels[:sample_count]

    start = time.time()
    output = np.asarray(retrieval.inference(sample))
    self_seconds = time.time() - start
    # Exact self matches are excluded for the proxy if the method returns them.
    for row_idx in range(output.shape[0]):
        output[row_idx] = np.array(
            [idx for idx in output[row_idx].tolist() if idx != row_idx]
            + [idx for idx in range(repository.shape[0]) if idx not in output[row_idx] and idx != row_idx][:5],
            dtype=np.int64,
        )[:5]
    self_proxy = same_class_score(output, labels, sample_labels)

    transform_scores = {}
    transform_seconds = {}
    for name, queries in baselines["transformed_sets"].items():
        start = time.time()
        retrieved = np.asarray(retrieval.inference(queries))
        transform_seconds[name] = time.time() - start
        transform_scores[name] = float(
            np.mean([idx in row for idx, row in zip(range(queries.shape[0]), retrieved)])
        )

    return {
        "same_class_proxy": float(self_proxy),
        "beats_raw_same_class_proxy": bool(self_proxy > baselines["raw_same_class_proxy"]),
        "transform_top5_recall": transform_scores,
        "beats_raw_transform_baseline": {
            name: bool(score > baselines["transform_baselines"][name])
            for name, score in transform_scores.items()
        },
        "init_seconds": init_seconds,
        "self_query_seconds": self_seconds,
        "estimated_1000_query_seconds": self_seconds * (1000.0 / sample_count),
        "transform_seconds": transform_seconds,
        "output_shape": list(output.shape),
        "valid_row_indices": bool(
            output.shape == (sample_count, 5)
            and np.all(output >= 0)
            and np.all(output < repository.shape[0])
            and all(len(set(row.tolist())) == 5 for row in output)
        ),
    }


def task3_baseline():
    X = load_pickle(TASK3_DIR / "classification_validation_data.pkl")[:, 1:].astype(np.float64)
    y = load_pickle(TASK3_DIR / "classification_validation_label.pkl")[:, 1:].reshape(-1).astype(int)
    weights = load_pickle(TASK3_DIR / "image_recognition_model_weights.pkl").astype(np.float64)

    rng = np.random.RandomState(42)
    mask = np.zeros((1, X.shape[1]), dtype=np.float64)
    mask[0, rng.choice(np.arange(X.shape[1]), size=30, replace=False)] = 1.0
    pred = predict_fixed_model(X * mask, weights)
    return {
        "X": X,
        "y": y,
        "weights": weights,
        "random_seed_42_accuracy": float(np.mean(pred == y)),
    }


def predict_fixed_model(X, weights):
    X_bias = np.hstack((np.ones((X.shape[0], 1)), X))
    return np.argmax(X_bias @ weights, axis=1)


def evaluate_task3(submission_name, submission_path, data):
    module = load_module(f"{submission_name}_selector", submission_path / "selector.py")
    with tempfile.TemporaryDirectory(dir=str(TEMP_ROOT)) as tmp:
        tmp_path = Path(tmp)
        save_pickle(tmp_path / "classification_validation_data.pkl", np.column_stack([np.arange(data["X"].shape[0]), data["X"]]))
        save_pickle(tmp_path / "classification_validation_label.pkl", np.column_stack([np.arange(data["y"].shape[0]), data["y"]]))
        save_pickle(tmp_path / "image_recognition_model_weights.pkl", data["weights"])
        with chdir(tmp_path):
            start = time.time()
            selector = module.Selector()
            init_seconds = time.time() - start
            mask = np.asarray(selector.get_mask_code(), dtype=np.float64)

    pred = predict_fixed_model(data["X"] * mask, data["weights"])
    accuracy = float(np.mean(pred == data["y"]))
    return {
        "accuracy": accuracy,
        "beats_random_baseline": bool(accuracy > data["random_seed_42_accuracy"]),
        "mask_shape": list(mask.shape),
        "mask_sum": float(mask.sum()),
        "is_binary": bool(np.all((mask == 0) | (mask == 1))),
        "selected_features": np.flatnonzero(mask[0]).astype(int).tolist(),
        "init_seconds": init_seconds,
    }


def main():
    results = {"notes": []}
    print("Preparing Task 1 baseline and split...", flush=True)
    results["notes"].append(
        "Task 1 uses a local stratified holdout because hidden test labels are unavailable. "
        "The baseline is sklearn multinomial logistic regression, a strong softmax proxy."
    )
    results["notes"].append(
        "Task 2 uses local proxies because hidden retrieval relevance is unavailable: "
        "same-class repository neighbors and transform top-5 recall."
    )

    split = task1_baseline_and_data()
    print("Preparing Task 2 baselines...", flush=True)
    results["task1_baseline"] = {
        "strong_softmax_proxy_accuracy": split["baseline_accuracy"],
        "seconds": split["baseline_seconds"],
    }

    repository_data = load_pickle(TASK2_DIR / "image_retrieval_repository_data.pkl")
    task2_base = task2_baselines(repository_data)
    print("Preparing Task 3 baseline...", flush=True)
    results["task2_baseline"] = {
        "raw_same_class_proxy": task2_base["raw_same_class_proxy"],
        "raw_transform_top5_recall": task2_base["transform_baselines"],
    }

    task3_base = task3_baseline()
    results["task3_baseline"] = {
        "random_seed_42_accuracy": task3_base["random_seed_42_accuracy"]
    }

    results["submissions"] = {}
    for name, path in SUBMISSIONS.items():
        print(f"Evaluating {name}...", flush=True)
        entry = {"path": str(path)}
        py_sizes = {p.name: p.stat().st_size for p in path.glob("*.py")}
        entry["python_file_sizes"] = py_sizes
        entry["total_python_size"] = int(sum(py_sizes.values()))
        entry["compliance_warning"] = None
        if py_sizes.get("classifier.py", 0) > 1_000_000:
            entry["compliance_warning"] = (
                "classifier.py is over 1 MB and appears to embed trained weights; "
                "this may conflict with the updated Q&A no-pretrained-weight intent."
            )

        entry["task1"] = evaluate_task1(name, path, split)
        entry["task2"] = evaluate_task2(name, path, repository_data, task2_base)
        entry["task3"] = evaluate_task3(name, path, task3_base)
        results["submissions"][name] = entry
        print(json.dumps({name: entry}, indent=2), flush=True)

    out_path = PROJECT_ROOT / "Comparison" / "simulated_evaluation_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
