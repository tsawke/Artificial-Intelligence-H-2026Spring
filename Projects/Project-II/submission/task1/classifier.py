import os
from pathlib import Path

import numpy as np


class Classifier:
    def __init__(self):
        """
        Load a portable MLP ensemble saved as NumPy arrays.

        The training script uses scikit-learn only to fit the models.  At
        inference time we intentionally avoid unpickling sklearn estimators so
        the submitted classifier is stable across Python/sklearn versions.
        """
        root_path = Path(os.path.dirname(os.path.abspath(__file__)))
        artifact_path = root_path / "classification_mlp_ensemble.npz"
        artifact = np.load(artifact_path, allow_pickle=False)

        self.mean = artifact["mean"]
        self.scale = artifact["scale"]
        self.classes = artifact["classes"].astype(int)
        self.n_models = int(artifact["n_models"][0])
        self.n_layers = artifact["n_layers"].astype(int)
        self.coefs = []
        self.intercepts = []

        for model_idx in range(self.n_models):
            model_coefs = []
            model_intercepts = []
            for layer_idx in range(self.n_layers[model_idx]):
                model_coefs.append(artifact[f"coef_{model_idx}_{layer_idx}"])
                model_intercepts.append(artifact[f"intercept_{model_idx}_{layer_idx}"])
            self.coefs.append(model_coefs)
            self.intercepts.append(model_intercepts)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def _predict_proba_one(self, X: np.ndarray, model_idx: int) -> np.ndarray:
        hidden = X
        last_layer = self.n_layers[model_idx] - 1

        for layer_idx, (coef, intercept) in enumerate(
            zip(self.coefs[model_idx], self.intercepts[model_idx])
        ):
            hidden = hidden @ coef + intercept
            if layer_idx != last_layer:
                hidden = np.maximum(hidden, 0.0)

        return self._softmax(hidden)

    def inference(self, X: np.array) -> np.array:
        """
        Inference all data points.

        Args:
            X: Feature matrix with shape [a, 256], where a is the number of
                test data points. If an ID-prefixed [a, 257] matrix is passed
                accidentally, the first column is ignored.

        Returns:
            Integer class labels with shape [a].
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be a 2-D feature matrix")
        if X.shape[1] == self.mean.shape[0] + 1:
            X = X[:, 1:]
        if X.shape[1] != self.mean.shape[0]:
            raise ValueError(f"Expected {self.mean.shape[0]} features, got {X.shape[1]}")

        X = (X - self.mean) / self.scale
        probabilities = np.zeros((X.shape[0], self.classes.shape[0]), dtype=np.float64)
        for model_idx in range(self.n_models):
            probabilities += self._predict_proba_one(X, model_idx)
        probabilities /= self.n_models
        return self.classes[np.argmax(probabilities, axis=1)]
