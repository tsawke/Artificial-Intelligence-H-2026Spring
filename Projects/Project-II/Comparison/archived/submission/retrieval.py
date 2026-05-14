import numpy as np


class Retrieval:
    def __init__(self, repository_data):
        """
        Prepare the image repository for nearest-neighbor retrieval.

        Args:
            repository_data: Data content is the same as
                `image_retrieval_repository_data.pkl`. The first column is an
                id column, and the remaining 256 columns are image features.
        """
        self.k = 5
        self.shift_replacement_margin = 0.4
        self.far_shift_replacement_margin = 0.4
        self.transform_replacement_margin = 0.4
        self.blur_replacement_margin = 0.4
        self.repository = self._as_feature_matrix(repository_data)
        self.repository_norm = np.sum(self.repository * self.repository, axis=1)
        self.blurred_repository = self._blur_features(self.repository)
        self.blurred_repository_norm = np.sum(self.blurred_repository * self.blurred_repository, axis=1)
        self.exact_lookup = {self.repository[idx].tobytes(): idx for idx in range(self.repository.shape[0])}
        self.primary_shifts = tuple(
            (dy, dx)
            for dy in range(-1, 2)
            for dx in range(-1, 2)
            if not (dy == 0 and dx == 0)
        )
        self.far_shifts = tuple(
            (dy, dx)
            for dy in range(-2, 3)
            for dx in range(-2, 3)
            if max(abs(dy), abs(dx)) == 2
        )

    def inference(self, X: np.array) -> np.array:
        """
        Find 5 images that are most similar to each query image.

        Args:
            X: Feature vectors with shape [a, 256].

        Returns:
            A numpy array with shape [a, 5]. Each row contains repository row
            indices of the retrieved images.
        """
        query = self._as_feature_matrix(X)
        if query.shape[1] != self.repository.shape[1]:
            raise ValueError(f"Expected {self.repository.shape[1]} features, got {query.shape[1]}")

        results = []
        chunk_size = 256
        for start in range(0, query.shape[0], chunk_size):
            chunk = query[start : start + chunk_size]
            raw_distances = (
                np.sum(chunk * chunk, axis=1, keepdims=True)
                + self.repository_norm[None, :]
                - 2.0 * (chunk @ self.repository.T)
            )
            primary_distances = self._best_shift_distances(chunk, self.primary_shifts)
            far_distances = self._best_shift_distances(chunk, self.far_shifts)
            transform_distances = self._best_transform_distances(chunk)
            blur_distances = self._blur_repository_distances(chunk)

            for row_idx, x in enumerate(chunk):
                exact_idx = self.exact_lookup.get(x.tobytes())
                if exact_idx is not None:
                    raw_distances[row_idx, exact_idx] = -1.0
                    primary_distances[row_idx, exact_idx] = -1.0
                    far_distances[row_idx, exact_idx] = -1.0
                    transform_distances[row_idx, exact_idx] = -1.0
                    blur_distances[row_idx, exact_idx] = -1.0

            results.append(
                self._hybrid_topk(
                    raw_distances,
                    primary_distances,
                    far_distances,
                    transform_distances,
                    blur_distances,
                )
            )

        return np.vstack(results).astype(np.int64)

    def _hybrid_topk(
        self,
        raw_distances: np.ndarray,
        primary_distances: np.ndarray,
        far_distances: np.ndarray,
        transform_distances: np.ndarray,
        blur_distances: np.ndarray,
    ) -> np.ndarray:
        raw_keep = 4
        raw_top = self._ordered_top_candidates(raw_distances, 50)

        source_data = (
            (primary_distances, self._ordered_top_candidates(primary_distances, 30), 1.20),
            (far_distances, self._ordered_top_candidates(far_distances, 30), 1.20),
            (blur_distances, self._ordered_top_candidates(blur_distances, 30), 1.20),
            (transform_distances, self._ordered_top_candidates(transform_distances, 30), 0.55),
        )

        output = np.empty((raw_distances.shape[0], self.k), dtype=np.int64)
        for i in range(raw_distances.shape[0]):
            selected = []
            used = set()
            for idx in raw_top[i, :raw_keep]:
                item = int(idx)
                selected.append(item)
                used.add(item)

            raw_next = int(raw_top[i, raw_keep])
            raw_next_distance = max(float(raw_distances[i, raw_next]), 1e-12)
            protected_raw = set(int(idx) for idx in raw_top[i, : self.k])
            alternatives: list[tuple[float, int, int]] = []

            for distances, ordered_top, ratio_cap in source_data:
                for rank, idx in enumerate(ordered_top[i]):
                    item = int(idx)
                    if item in protected_raw:
                        continue
                    ratio = float(distances[i, item]) / raw_next_distance
                    if ratio <= ratio_cap:
                        alternatives.append((ratio, rank, item))
                        break

            if alternatives:
                _, _, best_item = min(alternatives)
            else:
                best_item = next((int(idx) for idx in raw_top[i, self.k :] if int(idx) not in used), raw_next)

            selected.append(best_item)
            used.add(best_item)

            if len(selected) < self.k:
                fallback = np.argsort(raw_distances[i])
                for idx in fallback:
                    item = int(idx)
                    if item not in used:
                        selected.append(item)
                        used.add(item)
                    if len(selected) >= self.k:
                        break
            output[i] = np.array(selected[: self.k], dtype=np.int64)
        return output

    def _best_shift_distances(self, X: np.ndarray, shifts: tuple[tuple[int, int], ...]) -> np.ndarray:
        best_distances = np.full((X.shape[0], self.repository.shape[0]), np.inf)
        for dy, dx in shifts:
            shifted = self._shift_features(X, dy, dx)
            distances = (
                np.sum(shifted * shifted, axis=1, keepdims=True)
                + self.repository_norm[None, :]
                - 2.0 * (shifted @ self.repository.T)
            )
            best_distances = np.minimum(best_distances, distances)
        return best_distances

    def _best_transform_distances(self, X: np.ndarray) -> np.ndarray:
        best_distances = np.full((X.shape[0], self.repository.shape[0]), np.inf)
        for transformed in self._d4_transforms(X):
            distances = (
                np.sum(transformed * transformed, axis=1, keepdims=True)
                + self.repository_norm[None, :]
                - 2.0 * (transformed @ self.repository.T)
            )
            best_distances = np.minimum(best_distances, distances)
        return best_distances

    def _blur_repository_distances(self, X: np.ndarray) -> np.ndarray:
        return (
            np.sum(X * X, axis=1, keepdims=True)
            + self.blurred_repository_norm[None, :]
            - 2.0 * (X @ self.blurred_repository.T)
        )

    @staticmethod
    def _ordered_top_candidates(distances: np.ndarray, count: int) -> np.ndarray:
        count = min(count, distances.shape[1])
        candidates = np.argpartition(distances, kth=count - 1, axis=1)[:, :count]
        row = np.arange(candidates.shape[0])[:, None]
        order = np.argsort(distances[row, candidates], axis=1)
        return candidates[row, order]

    @staticmethod
    def _shift_features(X: np.ndarray, dy: int, dx: int) -> np.ndarray:
        if dy == 0 and dx == 0:
            return X

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

    @staticmethod
    def _d4_transforms(X: np.ndarray) -> tuple[np.ndarray, ...]:
        images = X.reshape(-1, 16, 16)
        transposed = np.transpose(images, (0, 2, 1))
        variants = (
            np.rot90(images, 1, axes=(1, 2)),
            np.rot90(images, 2, axes=(1, 2)),
            np.rot90(images, 3, axes=(1, 2)),
            images[:, :, ::-1],
            images[:, ::-1, :],
            transposed,
            transposed[:, :, ::-1],
        )
        return tuple(variant.reshape(X.shape) for variant in variants)

    @staticmethod
    def _blur_features(X: np.ndarray) -> np.ndarray:
        images = X.reshape(-1, 16, 16)
        padded = np.pad(images, ((0, 0), (1, 1), (1, 1)), mode="constant")
        blurred = np.zeros_like(images)
        for dy in range(3):
            for dx in range(3):
                blurred += padded[:, dy : dy + 16, dx : dx + 16]
        return (blurred / 9.0).reshape(X.shape)

    @staticmethod
    def _as_feature_matrix(X: np.ndarray) -> np.ndarray:
        features = np.asarray(X, dtype=np.float64)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        if features.shape[1] == 257:
            features = features[:, 1:]
        return features
