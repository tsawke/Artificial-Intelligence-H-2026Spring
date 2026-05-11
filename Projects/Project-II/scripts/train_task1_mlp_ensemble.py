import json
import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK1_DIR = PROJECT_ROOT / "project2_code" / "task1"
RESULTS_PATH = PROJECT_ROOT / "experiment_results.json"
ARTIFACT_PATH = TASK1_DIR / "classification_mlp_ensemble.npz"


CONFIGS = [
    {"name": "mlp512_s123", "hidden": (512,), "seed": 123, "alpha": 1e-4, "lr": 1e-3},
    {"name": "mlp512_s456", "hidden": (512,), "seed": 456, "alpha": 1e-4, "lr": 1e-3},
    {"name": "mlp512_s789", "hidden": (512,), "seed": 789, "alpha": 1e-4, "lr": 1e-3},
    {"name": "mlp512_s2026", "hidden": (512,), "seed": 2026, "alpha": 1e-4, "lr": 1e-3},
    {"name": "mlp256_s123", "hidden": (256,), "seed": 123, "alpha": 1e-4, "lr": 1e-3},
    {"name": "mlp256_s456", "hidden": (256,), "seed": 456, "alpha": 1e-4, "lr": 1e-3},
    {"name": "mlp512_256_s123", "hidden": (512, 256), "seed": 123, "alpha": 2e-4, "lr": 8e-4},
    {"name": "mlp512_256_s456", "hidden": (512, 256), "seed": 456, "alpha": 2e-4, "lr": 8e-4},
]


def load_pickle(path: Path):
    with path.open("rb") as file:
        return pickle.load(file)


def load_classification_data():
    data = load_pickle(TASK1_DIR / "classification_train_data.pkl")[:, 1:]
    labels = load_pickle(TASK1_DIR / "classification_train_label.pkl")[:, 1:].reshape(-1)
    return data.astype(np.float64), labels.astype(int)


def make_model(config):
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=config["hidden"],
            activation="relu",
            solver="adam",
            alpha=config["alpha"],
            learning_rate_init=config["lr"],
            batch_size=256,
            max_iter=120,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=12,
            random_state=config["seed"],
            verbose=False,
        ),
    )


def evaluate_configs(X, y):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=123, stratify=y
    )

    probabilities = []
    results = []
    for config in CONFIGS:
        start = time.time()
        model = make_model(config)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_val)
        acc = accuracy_score(y_val, np.argmax(proba, axis=1))
        probabilities.append(proba)
        elapsed = time.time() - start
        result = {
            "name": config["name"],
            "hidden_layers": list(config["hidden"]),
            "seed": config["seed"],
            "validation_accuracy": float(acc),
            "seconds": float(elapsed),
        }
        results.append(result)
        print(f"{config['name']}: acc={acc:.6f}, seconds={elapsed:.1f}", flush=True)

    ensemble_curve = []
    running = np.zeros_like(probabilities[0])
    for idx, proba in enumerate(probabilities):
        running += proba
        acc = accuracy_score(y_val, np.argmax(running / (idx + 1), axis=1))
        ensemble_curve.append(
            {
                "models": idx + 1,
                "last_model": CONFIGS[idx]["name"],
                "validation_accuracy": float(acc),
            }
        )
        print(f"ensemble_first_{idx + 1}: acc={acc:.6f}", flush=True)

    return {
        "task1": {
            "split": {"test_size": 0.2, "random_state": 123, "stratified": True},
            "single_models": results,
            "ensemble_curve": ensemble_curve,
            "final_validation_accuracy": ensemble_curve[-1]["validation_accuracy"],
        }
    }


def train_final_artifact(X, y):
    artifact = {}
    n_layers = []
    common_mean = None
    common_scale = None
    classes = None

    for model_idx, config in enumerate(CONFIGS):
        start = time.time()
        model = make_model(config)
        model.fit(X, y)
        scaler = model.named_steps["standardscaler"]
        mlp = model.named_steps["mlpclassifier"]

        if common_mean is None:
            common_mean = scaler.mean_.astype(np.float64)
            common_scale = scaler.scale_.astype(np.float64)
            classes = mlp.classes_.astype(int)

        n_layers.append(len(mlp.coefs_))
        for layer_idx, (coef, intercept) in enumerate(zip(mlp.coefs_, mlp.intercepts_)):
            artifact[f"coef_{model_idx}_{layer_idx}"] = coef.astype(np.float64)
            artifact[f"intercept_{model_idx}_{layer_idx}"] = intercept.astype(np.float64)

        elapsed = time.time() - start
        print(f"final {config['name']}: seconds={elapsed:.1f}", flush=True)

    artifact["mean"] = common_mean
    artifact["scale"] = common_scale
    artifact["classes"] = classes
    artifact["n_models"] = np.array([len(CONFIGS)], dtype=np.int64)
    artifact["n_layers"] = np.array(n_layers, dtype=np.int64)
    np.savez_compressed(ARTIFACT_PATH, **artifact)
    print(f"saved {ARTIFACT_PATH}", flush=True)


def main():
    X, y = load_classification_data()
    results = evaluate_configs(X, y)
    train_final_artifact(X, y)

    if RESULTS_PATH.exists():
        existing = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        existing.update(results)
        results = existing
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"saved {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
