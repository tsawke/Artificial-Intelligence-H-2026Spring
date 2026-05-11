import numpy as np


class Retrieval:
    def __init__(self, repository_data):
        """
        Prepare standardized nearest-neighbor retrieval.

        Args:
            repository_data: The image repository. It has the same content as
                `image_retrieval_repository_data.pkl`, with an ID column
                followed by 256 feature columns.
        """
        repository_data = np.asarray(repository_data, dtype=np.float64)
        if repository_data.ndim != 2:
            raise ValueError("repository_data must be a 2-D matrix")

        self.repository = repository_data[:, 1:] if repository_data.shape[1] == 257 else repository_data
        self.k = 5
        self.mean = np.mean(self.repository, axis=0)
        self.std = np.std(self.repository, axis=0)
        self.std[self.std == 0] = 1.0

        self.repository_std = (self.repository - self.mean) / self.std
        self.repository_norm_sq = np.sum(self.repository * self.repository, axis=1)
        self.repository_std_norm_sq = np.sum(self.repository_std * self.repository_std, axis=1)

    @staticmethod
    def _squared_euclidean(
        X: np.ndarray, Y: np.ndarray, Y_norm_sq: np.ndarray
    ) -> np.ndarray:
        X_norm_sq = np.sum(X * X, axis=1, keepdims=True)
        distances = X_norm_sq + Y_norm_sq.reshape(1, -1) - 2.0 * (X @ Y.T)
        return np.maximum(distances, 0.0)

    def _prepare_query(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be a 2-D feature matrix")
        if X.shape[1] == self.repository.shape[1] + 1:
            X = X[:, 1:]
        if X.shape[1] != self.repository.shape[1]:
            raise ValueError(f"Expected {self.repository.shape[1]} features, got {X.shape[1]}")

        X_std = (X - self.mean) / self.std
        return X, X_std

    def inference(self, X: np.array) -> np.array:
        """
        Find 5 repository images most similar to each query image.

        Args:
            X: Query feature matrix with shape [a, 256].

        Returns:
            Matrix with shape [a, 5]. Each row contains repository row indices,
            matching the semantics of the provided baseline implementation.
        """
        X_raw, X_std = self._prepare_query(X)

        raw_dist = self._squared_euclidean(X_raw, self.repository, self.repository_norm_sq)
        std_dist = self._squared_euclidean(X_std, self.repository_std, self.repository_std_norm_sq)

        top = np.argpartition(std_dist, self.k - 1, axis=1)[:, : self.k]
        top_std_dist = np.take_along_axis(std_dist, top, axis=1)
        raw_tiebreak = np.take_along_axis(raw_dist, top, axis=1)
        order = np.lexsort((raw_tiebreak, top_std_dist), axis=1)
        return np.take_along_axis(top, order, axis=1)
