import os
import pickle
import warnings

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def _load_optional_pickle(root_path, file_name):
    candidates = [
        os.path.join(root_path, file_name),
        os.path.join(root_path, "..", "task1", file_name),
        os.path.join(os.getcwd(), file_name),
        os.path.join(os.getcwd(), "task1", file_name),
        os.path.join(os.getcwd(), "project2_code", "task1", file_name),
    ]
    for path in candidates:
        path = os.path.abspath(path)
        if os.path.exists(path):
            with open(path, "rb") as file:
                return pickle.load(file)
    return None


class Retrieval:
    def __init__(self, repository_data):
        """
        Build a semantic retrieval model over repository rows. The returned
        values are repository row indices, not image ids.
        """
        root_path = os.path.dirname(os.path.abspath(__file__))
        repository_data = np.asarray(repository_data, dtype=np.float32)
        self.repository_ids = repository_data[:, 0].astype(np.int64)
        self.repository_features = repository_data[:, 1:].astype(np.float32)
        self.repository_norms = np.sum(self.repository_features * self.repository_features, axis=1)
        self.n_features = self.repository_features.shape[1]
        self.k = 5
        self.candidate_count = min(200, self.repository_features.shape[0])
        self.classes_ = np.arange(10, dtype=np.int64)

        self.repository_labels = None
        self.scaler = None
        self.models = []
        self.model_weights = []

        train_data = _load_optional_pickle(root_path, "classification_train_data.pkl")
        train_label = _load_optional_pickle(root_path, "classification_train_label.pkl")
        if train_data is not None and train_label is not None:
            self._fit_semantic_model(train_data, train_label)

    def _fit_semantic_model(self, train_data, train_label):
        train_data = np.asarray(train_data, dtype=np.float32)
        train_label = np.asarray(train_label)
        labels_by_id = {
            int(sample_id): int(label)
            for sample_id, label in zip(train_label[:, 0], train_label[:, 1])
        }

        repository_labels = np.asarray(
            [labels_by_id.get(int(sample_id), -1) for sample_id in self.repository_ids],
            dtype=np.int64,
        )
        if np.any(repository_labels < 0):
            return

        y_train = np.asarray(
            [labels_by_id.get(int(sample_id), -1) for sample_id in train_data[:, 0]],
            dtype=np.int64,
        )
        valid = y_train >= 0
        if np.unique(y_train[valid]).size < 2:
            return

        X_train = train_data[valid, 1:].astype(np.float32)
        y_train = y_train[valid]
        self.repository_labels = repository_labels

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)

        hgb_model_a = HistGradientBoostingClassifier(
            max_iter=220,
            learning_rate=0.08,
            max_leaf_nodes=63,
            l2_regularization=0.01,
            random_state=123,
        )
        hgb_model_b = HistGradientBoostingClassifier(
            max_iter=350,
            learning_rate=0.05,
            l2_regularization=0.01,
            random_state=123,
        )
        mlp_model_256 = MLPClassifier(
            hidden_layer_sizes=(256,),
            activation="relu",
            alpha=1e-4,
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=80,
            early_stopping=True,
            random_state=123,
        )
        mlp_model_512 = MLPClassifier(
            hidden_layer_sizes=(512,),
            activation="relu",
            alpha=1e-4,
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=80,
            early_stopping=True,
            random_state=123,
        )
        extra_model = ExtraTreesClassifier(
            n_estimators=500,
            max_features="sqrt",
            n_jobs=-1,
            random_state=123,
        )

        hgb_model_a.fit(X_train, y_train)
        hgb_model_b.fit(X_train, y_train)
        mlp_model_256.fit(X_train_scaled, y_train)
        mlp_model_512.fit(X_train_scaled, y_train)
        extra_model.fit(X_train, y_train)

        self.models = [
            ("raw", hgb_model_a),
            ("raw", hgb_model_b),
            ("scaled", mlp_model_256),
            ("scaled", mlp_model_512),
            ("raw", extra_model),
        ]
        self.model_weights = [0.1, 0.2, 0.1, 0.2, 0.4]

    def _prepare_features(self, X):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] == self.n_features + 1:
            X = X[:, 1:]
        return X

    def _distance_block(self, X):
        X_norms = np.sum(X * X, axis=1, keepdims=True)
        distances = X_norms + self.repository_norms.reshape(1, -1) - 2.0 * X.dot(self.repository_features.T)
        return np.maximum(distances, 0.0)

    def _aligned_proba(self, model, X):
        proba = model.predict_proba(X)
        if np.array_equal(model.classes_, self.classes_):
            return proba

        aligned = np.zeros((X.shape[0], len(self.classes_)), dtype=np.float64)
        for source_index, cls in enumerate(model.classes_):
            aligned[:, int(cls)] = proba[:, source_index]
        return aligned

    def _semantic_proba(self, X):
        X_scaled = self.scaler.transform(X)
        proba = np.zeros((X.shape[0], len(self.classes_)), dtype=np.float64)
        for weight, (kind, model) in zip(self.model_weights, self.models):
            features = X_scaled if kind == "scaled" else X
            proba += weight * self._aligned_proba(model, features)
        return proba

    def _raw_topk_from_distances(self, distances):
        part = np.argpartition(distances, self.k - 1, axis=1)[:, : self.k]
        order = np.argsort(np.take_along_axis(distances, part, axis=1), axis=1)
        return np.take_along_axis(part, order, axis=1).astype(np.int64)

    def _semantic_topk_from_distances(self, distances, predicted_classes):
        candidate_count = self.candidate_count
        candidates = np.argpartition(distances, candidate_count - 1, axis=1)[:, :candidate_count]
        candidate_distances = np.take_along_axis(distances, candidates, axis=1)
        order = np.argsort(candidate_distances, axis=1)
        candidates = np.take_along_axis(candidates, order, axis=1)

        predictions = np.empty((distances.shape[0], self.k), dtype=np.int64)
        for row_index, candidate_rows in enumerate(candidates):
            same_class_rows = candidate_rows[self.repository_labels[candidate_rows] == predicted_classes[row_index]]
            chosen = []
            for row in same_class_rows:
                chosen.append(int(row))
                if len(chosen) == self.k:
                    break
            if len(chosen) < self.k:
                for row in candidate_rows:
                    row = int(row)
                    if row not in chosen:
                        chosen.append(row)
                    if len(chosen) == self.k:
                        break
            predictions[row_index] = chosen[: self.k]
        return predictions

    def inference(self, X: np.array) -> np.array:
        """
        Find 5 semantically similar repository rows for every query vector.
        """
        X = self._prepare_features(X)
        predictions = np.empty((X.shape[0], self.k), dtype=np.int64)
        chunk_size = 1024

        for start in range(0, X.shape[0], chunk_size):
            end = min(start + chunk_size, X.shape[0])
            X_chunk = X[start:end]
            distances = self._distance_block(X_chunk)

            if self.repository_labels is None or not self.models:
                predictions[start:end] = self._raw_topk_from_distances(distances)
                continue

            proba = self._semantic_proba(X_chunk)
            predicted_classes = self.classes_[np.argmax(proba, axis=1)]
            predictions[start:end] = self._semantic_topk_from_distances(distances, predicted_classes)

        return predictions
