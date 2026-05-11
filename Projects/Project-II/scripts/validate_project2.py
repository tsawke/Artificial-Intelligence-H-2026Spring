import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK1_DIR = PROJECT_ROOT / "project2_code" / "task1"
TASK2_DIR = PROJECT_ROOT / "project2_code" / "task2"
TASK3_DIR = PROJECT_ROOT / "project2_code" / "task3"
RESULTS_PATH = PROJECT_ROOT / "experiment_results.json"


def load_pickle(path: Path):
    with path.open("rb") as file:
        return pickle.load(file)


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def validate_task1():
    classifier_module = load_module("project2_classifier", TASK1_DIR / "classifier.py")
    classifier = classifier_module.Classifier()
    data = load_pickle(TASK1_DIR / "classification_train_data.pkl")[:128, 1:]
    pred = classifier.inference(data)
    if pred.shape != (128,):
        raise AssertionError(f"Task 1 prediction shape mismatch: {pred.shape}")
    if not np.issubdtype(pred.dtype, np.integer):
        raise AssertionError(f"Task 1 predictions must be integer labels, got {pred.dtype}")
    if np.any((pred < 0) | (pred > 9)):
        raise AssertionError("Task 1 predictions must be in [0, 9]")
    return {"smoke_samples": 128, "prediction_shape": list(pred.shape)}


def validate_task2():
    retrieval_module = load_module("project2_retrieval", TASK2_DIR / "retrieval.py")
    repository = load_pickle(TASK2_DIR / "image_retrieval_repository_data.pkl")
    retrieval = retrieval_module.Retrieval(repository)
    output = retrieval.inference(repository[:64, 1:])
    if output.shape != (64, 5):
        raise AssertionError(f"Task 2 output shape mismatch: {output.shape}")
    if np.any(output < 0) or np.any(output >= repository.shape[0]):
        raise AssertionError("Task 2 output contains invalid repository indices")
    if any(len(set(row.tolist())) != 5 for row in output):
        raise AssertionError("Task 2 output contains duplicate indices in at least one row")
    return {"smoke_queries": 64, "output_shape": list(output.shape)}


def softmax_predict(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    X_bias = np.hstack((np.ones((X.shape[0], 1)), X))
    logits = X_bias @ weights
    return np.argmax(logits, axis=1)


def validate_task3():
    selector_module = load_module("project2_selector", TASK3_DIR / "selector.py")
    selector = selector_module.Selector()
    mask = selector.get_mask_code()
    if mask.shape != (1, 256):
        raise AssertionError(f"Task 3 mask shape mismatch: {mask.shape}")
    if not np.all((mask == 0) | (mask == 1)):
        raise AssertionError("Task 3 mask must be binary")
    if int(mask.sum()) != 30:
        raise AssertionError(f"Task 3 mask must select 30 features, got {mask.sum()}")

    with (TASK3_DIR / "mask_code.pkl").open("wb") as file:
        pickle.dump(mask, file)

    validation_data = load_pickle(TASK3_DIR / "classification_validation_data.pkl")[:, 1:]
    validation_label = load_pickle(TASK3_DIR / "classification_validation_label.pkl")[:, 1:].reshape(-1)
    weights = load_pickle(TASK3_DIR / "image_recognition_model_weights.pkl")

    selected_pred = softmax_predict(validation_data * mask, weights)
    selected_accuracy = float(np.mean(selected_pred == validation_label))

    rng = np.random.RandomState(42)
    random_mask = np.zeros((1, 256), dtype=np.float64)
    random_mask[0, rng.choice(np.arange(256), size=30, replace=False)] = 1.0
    random_pred = softmax_predict(validation_data * random_mask, weights)
    random_accuracy = float(np.mean(random_pred == validation_label))

    return {
        "selected_features": np.flatnonzero(mask[0]).astype(int).tolist(),
        "selected_count": int(mask.sum()),
        "validation_accuracy": selected_accuracy,
        "random_seed_42_accuracy": random_accuracy,
    }


def main():
    results = {}
    if RESULTS_PATH.exists():
        results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))

    results["interface_validation"] = {
        "task1": validate_task1(),
        "task2": validate_task2(),
        "task3": validate_task3(),
    }
    results["task3"] = results["interface_validation"]["task3"]

    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results["interface_validation"], indent=2), flush=True)


if __name__ == "__main__":
    main()
