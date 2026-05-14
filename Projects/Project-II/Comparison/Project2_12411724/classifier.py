import os
import pickle
import warnings

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def _load_pickle(root_path, file_name):
    candidates = [
        os.path.join(root_path, file_name),
        os.path.join(os.getcwd(), file_name),
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path, "rb") as file:
                return pickle.load(file)
    raise FileNotFoundError(file_name)


class Classifier:
    def __init__(self):
        """
        Train the classification model from the data available in the judge
        environment. The submitted file is self-contained and does not require
        pre-generated pickle model files.
        """
        root_path = os.path.dirname(os.path.abspath(__file__))
        train_data = _load_pickle(root_path, "classification_train_data.pkl")
        train_label = _load_pickle(root_path, "classification_train_label.pkl")

        X_train = np.asarray(train_data[:, 1:], dtype=np.float32)
        y_train = np.asarray(train_label[:, 1], dtype=np.int64)
        self.n_features = X_train.shape[1]

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)

        self.hgb_model_a = HistGradientBoostingClassifier(
            max_iter=220,
            learning_rate=0.08,
            max_leaf_nodes=63,
            l2_regularization=0.01,
            random_state=123,
        )
        self.hgb_model_b = HistGradientBoostingClassifier(
            max_iter=350,
            learning_rate=0.05,
            l2_regularization=0.01,
            random_state=123,
        )
        self.mlp_model_256 = MLPClassifier(
            hidden_layer_sizes=(256,),
            activation="relu",
            alpha=1e-4,
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=80,
            early_stopping=True,
            random_state=123,
        )
        self.mlp_model_512 = MLPClassifier(
            hidden_layer_sizes=(512,),
            activation="relu",
            alpha=1e-4,
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=80,
            early_stopping=True,
            random_state=123,
        )
        self.extra_model = ExtraTreesClassifier(
            n_estimators=500,
            max_features="sqrt",
            n_jobs=-1,
            random_state=123,
        )

        self.hgb_model_a.fit(X_train, y_train)
        self.hgb_model_b.fit(X_train, y_train)
        self.mlp_model_256.fit(X_train_scaled, y_train)
        self.mlp_model_512.fit(X_train_scaled, y_train)
        self.extra_model.fit(X_train, y_train)
        self.classes_ = np.arange(10, dtype=np.int64)

    def _prepare_features(self, X):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] == self.n_features + 1:
            X = X[:, 1:]
        return X

    def _aligned_proba(self, model, X):
        proba = model.predict_proba(X)
        if np.array_equal(model.classes_, self.classes_):
            return proba

        aligned = np.zeros((X.shape[0], len(self.classes_)), dtype=np.float64)
        for source_index, cls in enumerate(model.classes_):
            aligned[:, int(cls)] = proba[:, source_index]
        return aligned

    def inference(self, X: np.array) -> np.array:
        """
        Inference all test vectors.

        Args:
            X: Feature vectors with shape [a, 256]. If the index column is
               accidentally included, it is ignored.

        Returns:
            Integer class predictions with shape [a].
        """
        X = self._prepare_features(X)
        X_scaled = self.scaler.transform(X)

        proba = (
            0.1 * self._aligned_proba(self.hgb_model_a, X)
            + 0.2 * self._aligned_proba(self.hgb_model_b, X)
            + 0.1 * self._aligned_proba(self.mlp_model_256, X_scaled)
            + 0.2 * self._aligned_proba(self.mlp_model_512, X_scaled)
            + 0.4 * self._aligned_proba(self.extra_model, X)
        )
        return np.argmax(proba, axis=1).astype(np.int64)
