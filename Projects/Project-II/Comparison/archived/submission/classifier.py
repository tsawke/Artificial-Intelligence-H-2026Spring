from pathlib import Path
import pickle
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier


def _load_pickle(path: Path):
    with path.open("rb") as file:
        return pickle.load(file)


def _find_data_file(filename: str) -> Path:
    """
    The OJ links data files into the current task directory.  The fallback
    locations only make the same file runnable in this local project layout.
    """
    here = Path(__file__).resolve().parent
    candidates = (
        here / filename,
        Path.cwd() / filename,
        here.parent / "task1" / filename,
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Cannot find {filename}")


class Classifier:
    def __init__(self):
        """
        Train the classifier once after import/instantiation, as used by OJ.
        """
        data = _load_pickle(_find_data_file("classification_train_data.pkl"))
        labels = _load_pickle(_find_data_file("classification_train_label.pkl"))

        x = np.asarray(data[:, 1:], dtype=np.float64)
        y = np.asarray(labels[:, 1:].reshape(-1), dtype=np.int64)

        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)
        self.std = np.where(self.std == 0, 1.0, self.std)
        x = (x - self.mean) / self.std

        self.model = MLPClassifier(
            hidden_layer_sizes=(1024,),
            activation="relu",
            solver="adam",
            alpha=0.0001,
            batch_size=256,
            learning_rate_init=0.001,
            max_iter=120,
            tol=0.0001,
            random_state=321,
            early_stopping=False,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            self.model.fit(x, y)

    def inference(self, X: np.array) -> np.array:
        """
        Inference all input feature vectors.

        Args:
            X: Feature vectors with shape [a, 256].

        Returns:
            Predicted category indices with shape [a].
        """
        x = np.asarray(X, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.shape[1] == self.mean.shape[0] + 1:
            x = x[:, 1:]
        if x.shape[1] != self.mean.shape[0]:
            raise ValueError(f"Expected {self.mean.shape[0]} features, got {x.shape[1]}")

        x = (x - self.mean) / self.std
        return self.model.predict(x).astype(np.int64)
