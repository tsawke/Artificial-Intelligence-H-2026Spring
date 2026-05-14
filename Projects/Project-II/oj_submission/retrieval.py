import pickle
import time
import warnings
from pathlib import Path

import numpy as np

try:
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover - the judge environment includes sklearn.
    ExtraTreesClassifier = None
    HistGradientBoostingClassifier = None
    MLPClassifier = None
    StandardScaler = None


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class Retrieval:
    """Hybrid semantic and transform-aware retrieval returning repository row indices."""

    # Transform candidate thresholds.
    RATIO_REPLACE = 1.00
    RATIO_REPLACE_BLUR_VS_TOP1 = 0.30
    STRONG_TRANSFORM_RATIO = 0.50
    TASK_TIME_LIMIT_SEC = 600.0
    SAFETY_MARGIN_SEC = 5.0
    NEXT_CHUNK_GUARD_FACTOR = 1.25
    SEMANTIC_TRAIN_BUDGET_SEC = 540.0
    SEMANTIC_CANDIDATE_COUNT = 200

    def __init__(self, repository_data):
        self._init_started_at = time.time()
        repo = np.asarray(repository_data, dtype=np.float64)
        if repo.ndim != 2:
            raise ValueError("repository_data must be a 2-D matrix")
        self.repository_ids = None
        self._repository_has_ids = repo.shape[1] == 257
        if repo.shape[1] == 257:
            self.repository_ids = repo[:, 0].astype(np.int64)
            repo = repo[:, 1:]
        if repo.shape[1] != 256:
            raise ValueError(f"expected 256 feature columns, got {repo.shape[1]}")
        if self.repository_ids is None:
            self.repository_ids = np.arange(repo.shape[0], dtype=np.int64)

        self.repository = repo
        self.repository_semantic = self.repository.astype(np.float32)
        self.k = 5

        self.repo_norm_sq = np.einsum("ij,ij->i", self.repository, self.repository)
        self.repo_semantic_norm_sq = np.sum(
            self.repository_semantic * self.repository_semantic, axis=1
        )
        self.repo_mean = np.mean(self.repository, axis=0)
        self.repo_std = np.std(self.repository, axis=0)
        self.repo_std[self.repo_std < 1e-8] = 1.0
        self.repo_standardized = (self.repository - self.repo_mean) / self.repo_std
        self.repo_standardized_norm_sq = np.einsum(
            "ij,ij->i", self.repo_standardized, self.repo_standardized
        )

        # Blurred repository channel for smoothing-aware matches.
        self.repo_blurred = self._blur_features(self.repository)
        self.repo_blurred_norm_sq = np.einsum(
            "ij,ij->i", self.repo_blurred, self.repo_blurred
        )

        # Exact-feature lookup for repository self-queries.
        self._exact_lookup = {
            self.repository[i].tobytes(): i for i in range(self.repository.shape[0])
        }

        # Unit-shift augmentation table.
        self._primary_shifts = tuple(
            (dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if not (dy == 0 and dx == 0)
        )

        # Optional semantic layer for class-aware reranking.
        self.repository_labels = None
        self.semantic_classes = np.arange(10, dtype=np.int64)
        self._class_to_column = {int(c): int(c) for c in self.semantic_classes}
        self.semantic_scaler = None
        self.semantic_models = []
        self.semantic_weights = []
        self.semantic_centroids = None
        self.semantic_centroid_norm_sq = None
        self.semantic_candidate_count = min(
            self.SEMANTIC_CANDIDATE_COUNT, self.repository.shape[0]
        )
        self._fit_semantic_layer()

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def inference(self, X: np.ndarray) -> np.ndarray:
        deadline = time.time() + self.TASK_TIME_LIMIT_SEC - self.SAFETY_MARGIN_SEC
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be a 2-D feature matrix")
        query_ids = None
        if X.shape[1] == 257:
            query_ids = X[:, 0].astype(np.int64)
            X = X[:, 1:]
        if X.shape[1] != self.repository.shape[1]:
            raise ValueError(
                f"Expected {self.repository.shape[1]} features, got {X.shape[1]}"
            )

        n_query = X.shape[0]
        results = np.empty((n_query, self.k), dtype=np.int64)

        # Chunking keeps memory bounded and leaves room for the time guard.
        chunk = 256
        last_chunk_seconds = None
        for start in range(0, n_query, chunk):
            end = min(start + chunk, n_query)
            remaining = deadline - time.time()
            if remaining <= 0:
                results[start:] = self._default_indices(n_query - start)
                break
            if (
                last_chunk_seconds is not None
                and last_chunk_seconds * self.NEXT_CHUNK_GUARD_FACTOR > remaining
            ):
                results[start:] = self._fast_standardized_inference(X[start:], deadline)
                break
            chunk_start = time.time()
            results[start:end] = self._inference_chunk(X[start:end])
            last_chunk_seconds = time.time() - chunk_start
        return results

    # ------------------------------------------------------------------ #
    # core
    # ------------------------------------------------------------------ #
    def _inference_chunk(self, X: np.ndarray) -> np.ndarray:
        # ---------- 1. raw distances ---------- #
        raw_dist = self._sqdist(X, self.repository, self.repo_norm_sq)
        X_standardized = (X - self.repo_mean) / self.repo_std
        std_dist = self._sqdist(
            X_standardized,
            self.repo_standardized,
            self.repo_standardized_norm_sq,
        )

        # Exact repository rows keep their self-match in slot 0.
        self_idx_per_row = np.full(X.shape[0], -1, dtype=np.int64)
        for r in range(X.shape[0]):
            key = X[r].tobytes()
            sj = self._exact_lookup.get(key)
            if sj is not None:
                self_idx_per_row[r] = sj

        # ---------- 2. augmented distances ---------- #
        # 2a. best one-pixel shifted query.
        shift_dist = self._best_over_augmentations(
            (self._shift_features(X, dy, dx) for dy, dx in self._primary_shifts),
            self.repository,
            self.repo_norm_sq,
            shape=(X.shape[0], self.repository.shape[0]),
        )
        # 2b. best non-identity D4 rotation/reflection.
        d4_dist = self._best_over_augmentations(
            self._d4_variants(X),
            self.repository,
            self.repo_norm_sq,
            shape=(X.shape[0], self.repository.shape[0]),
        )
        # 2c. blur channel.
        blur_dist = self._sqdist(X, self.repo_blurred, self.repo_blurred_norm_sq)

        # ---------- 3. hybrid top-5 ---------- #
        self._last_query_chunk = X
        try:
            return self._hybrid_topk(
                raw_dist, std_dist, shift_dist, d4_dist, blur_dist, self_idx_per_row
            )
        finally:
            self._last_query_chunk = None

    def _hybrid_topk(
        self,
        raw_dist: np.ndarray,
        std_dist: np.ndarray,
        shift_dist: np.ndarray,
        d4_dist: np.ndarray,
        blur_dist: np.ndarray,
        self_idx_per_row: np.ndarray,
    ) -> np.ndarray:
        n, k = raw_dist.shape[0], self.k

        # Raw top-k supplies conservative transform thresholds.
        raw_part = np.argpartition(raw_dist, kth=k - 1, axis=1)[:, :k]
        row = np.arange(n)[:, None]
        order = np.argsort(np.take_along_axis(raw_dist, raw_part, axis=1), axis=1)
        raw_top = np.take_along_axis(raw_part, order, axis=1)
        raw_top_dist = np.take_along_axis(raw_dist, raw_top, axis=1)
        raw_top1_dist = raw_top_dist[:, 0]
        raw_kth_dist = raw_top_dist[:, k - 1]

        # Best single augmented candidate per channel.
        shift_best_idx = np.argmin(shift_dist, axis=1)
        shift_best_d = shift_dist[row.ravel(), shift_best_idx]
        d4_best_idx = np.argmin(d4_dist, axis=1)
        d4_best_d = d4_dist[row.ravel(), d4_best_idx]
        blur_best_idx = np.argmin(blur_dist, axis=1)
        blur_best_d = blur_dist[row.ravel(), blur_best_idx]

        # Shift/D4 use raw-kth as threshold; blur must beat raw-top1 strongly.
        eps = 1e-12
        denom_kth = np.maximum(raw_kth_dist, eps)
        denom_top1 = np.maximum(raw_top1_dist, eps)
        shift_ratio = shift_best_d / denom_kth
        d4_ratio = d4_best_d / denom_kth
        blur_ratio = blur_best_d / denom_top1

        shift_ok = shift_ratio <= self.RATIO_REPLACE
        d4_ok = d4_ratio <= self.RATIO_REPLACE
        blur_ok = blur_ratio <= self.RATIO_REPLACE_BLUR_VS_TOP1

        # Pick the eligible transform channel with the smallest ratio.
        BIG = np.inf
        s_r = np.where(shift_ok, shift_ratio, BIG)
        d_r = np.where(d4_ok, d4_ratio, BIG)
        b_r = np.where(blur_ok, blur_ratio, BIG)
        stacked = np.stack([s_r, d_r, b_r], axis=1)  # [n, 3]
        channel = np.argmin(stacked, axis=1)
        best_ratio = np.min(stacked, axis=1)
        any_eligible = best_ratio < BIG

        cand_idx = np.where(
            channel == 0,
            shift_best_idx,
            np.where(channel == 1, d4_best_idx, blur_best_idx),
        )

        # Start from semantic reranking, or standardized neighbors if semantic training failed.
        semantic_prob = self._semantic_probabilities_from_raw_dist(raw_dist)
        if semantic_prob is None:
            out = self._ordered_topk(std_dist, self.k)
            predicted_for_transform = None
        else:
            out = self._semantic_rerank(raw_dist, std_dist, raw_top, semantic_prob)
            predicted_for_transform = self.semantic_classes[np.argmax(semantic_prob, axis=1)]
        out = self._preserve_exact_self(out, std_dist, self_idx_per_row)
        for i in range(n):
            if self_idx_per_row[i] >= 0:
                continue
            if not any_eligible[i]:
                continue
            ci = int(cand_idx[i])
            if predicted_for_transform is not None and self.repository_labels is not None:
                same_predicted_class = self.repository_labels[ci] == predicted_for_transform[i]
                if not same_predicted_class and best_ratio[i] > self.STRONG_TRANSFORM_RATIO:
                    continue
            if ci in out[i, : k - 1]:
                continue
            # Replace only the last slot to keep the output conservative.
            out[i, k - 1] = ci
        return out

    def _fast_standardized_inference(self, X: np.ndarray, deadline: float) -> np.ndarray:
        """Fast non-raw fallback for remaining queries near the time limit."""
        out = np.empty((X.shape[0], self.k), dtype=np.int64)
        chunk = 512
        for start in range(0, X.shape[0], chunk):
            end = min(start + chunk, X.shape[0])
            if time.time() >= deadline:
                out[start:] = self._default_indices(X.shape[0] - start)
                break
            out[start:end] = self._standardized_topk_chunk(X[start:end])
        return out

    def _standardized_topk_chunk(self, X: np.ndarray) -> np.ndarray:
        X_standardized = (X - self.repo_mean) / self.repo_std
        std_dist = self._sqdist(
            X_standardized,
            self.repo_standardized,
            self.repo_standardized_norm_sq,
        )
        out = self._ordered_topk(std_dist, self.k)
        self_idx_per_row = np.full(X.shape[0], -1, dtype=np.int64)
        for r in range(X.shape[0]):
            sj = self._exact_lookup.get(X[r].tobytes())
            if sj is not None:
                self_idx_per_row[r] = sj
        return self._preserve_exact_self(out, std_dist, self_idx_per_row)

    def _preserve_exact_self(
        self,
        out: np.ndarray,
        std_dist: np.ndarray,
        self_idx_per_row: np.ndarray,
    ) -> np.ndarray:
        """Put exact self in slot 0 and keep the best non-self neighbors."""
        exact_rows = np.flatnonzero(self_idx_per_row >= 0)
        if exact_rows.size == 0:
            return out
        for row in exact_rows:
            self_idx = int(self_idx_per_row[row])
            chosen = [self_idx]
            for idx in out[row]:
                idx = int(idx)
                if idx != self_idx and idx not in chosen:
                    chosen.append(idx)
                if len(chosen) == self.k:
                    break
            if len(chosen) < self.k:
                dist = std_dist[row].copy()
                dist[chosen] = np.inf
                fill = self._ordered_topk(dist.reshape(1, -1), self.k - len(chosen))[0]
                for idx in fill:
                    idx = int(idx)
                    if idx not in chosen:
                        chosen.append(idx)
                    if len(chosen) == self.k:
                        break
            out[row] = np.asarray(chosen[: self.k], dtype=np.int64)
        return out

    # ------------------------------------------------------------------ #
    # semantic reranking
    # ------------------------------------------------------------------ #
    def _fit_semantic_layer(self) -> None:
        if ExtraTreesClassifier is None or StandardScaler is None:
            return
        loaded = self._load_training_data()
        if loaded is None:
            return
        train_data, train_label = loaded
        X_train, y_train = self._training_xy(train_data, train_label)
        if X_train is None or np.unique(y_train).size < 2:
            return

        repo_labels = self._repository_label_vector(train_data, train_label, X_train, y_train)
        if repo_labels is None:
            return
        self.repository_labels = repo_labels
        self.semantic_classes = np.sort(np.unique(y_train)).astype(np.int64)
        self._class_to_column = {
            int(label): col for col, label in enumerate(self.semantic_classes.tolist())
        }

        self.semantic_scaler = StandardScaler()
        X_train_std = self.semantic_scaler.fit_transform(X_train)
        self._fit_centroids(X_train_std, y_train)

        model_plan = [
            (
                0.10,
                "raw",
                HistGradientBoostingClassifier(
                    max_iter=220,
                    learning_rate=0.08,
                    max_leaf_nodes=63,
                    l2_regularization=0.01,
                    random_state=123,
                ),
            ),
            (
                0.20,
                "raw",
                HistGradientBoostingClassifier(
                    max_iter=350,
                    learning_rate=0.05,
                    l2_regularization=0.01,
                    random_state=123,
                ),
            ),
            (
                0.10,
                "scaled",
                MLPClassifier(
                    hidden_layer_sizes=(256,),
                    activation="relu",
                    alpha=1e-4,
                    batch_size=256,
                    learning_rate_init=1e-3,
                    max_iter=80,
                    early_stopping=True,
                    random_state=123,
                ),
            ),
            (
                0.20,
                "scaled",
                MLPClassifier(
                    hidden_layer_sizes=(512,),
                    activation="relu",
                    alpha=1e-4,
                    batch_size=256,
                    learning_rate_init=1e-3,
                    max_iter=80,
                    early_stopping=True,
                    random_state=123,
                ),
            ),
            (
                0.40,
                "raw",
                ExtraTreesClassifier(
                    n_estimators=500,
                    max_features="sqrt",
                    n_jobs=-1,
                    random_state=123,
                ),
            ),
        ]

        for weight, kind, model in model_plan:
            if time.time() - self._init_started_at > self.SEMANTIC_TRAIN_BUDGET_SEC:
                break
            try:
                model.fit(X_train_std if kind == "scaled" else X_train, y_train)
            except Exception:
                continue
            self.semantic_weights.append(float(weight))
            self.semantic_models.append((kind, model))

    def _load_training_data(self):
        module_dir = Path(__file__).resolve().parent
        cwd = Path.cwd().resolve()
        search_roots = [
            module_dir,
            module_dir.parent,
            module_dir.parent / "task1",
            module_dir.parent / "project2_code" / "task1",
            cwd,
            cwd / "task1",
            cwd / "project2_code" / "task1",
        ]
        seen = set()
        unique_roots = []
        for root in search_roots:
            try:
                resolved = root.resolve()
            except Exception:
                continue
            if resolved not in seen:
                seen.add(resolved)
                unique_roots.append(resolved)

        for root in unique_roots:
            data_path = root / "classification_train_data.pkl"
            label_path = root / "classification_train_label.pkl"
            if data_path.exists() and label_path.exists():
                try:
                    with data_path.open("rb") as file:
                        data = pickle.load(file)
                    with label_path.open("rb") as file:
                        label = pickle.load(file)
                except Exception:
                    continue
                return data, label
        return None

    def _training_xy(self, train_data, train_label):
        train_data = np.asarray(train_data, dtype=np.float32)
        train_label = np.asarray(train_label)
        if train_data.ndim != 2 or train_label.ndim != 2 or train_label.shape[1] < 2:
            return None, None
        label_by_id = {
            int(sample_id): int(label)
            for sample_id, label in train_label[:, :2]
        }
        if train_data.shape[1] == self.repository.shape[1] + 1:
            ids = train_data[:, 0].astype(np.int64)
            X = train_data[:, 1:].astype(np.float32)
            y = np.asarray([label_by_id.get(int(sample_id), -1) for sample_id in ids])
        elif train_data.shape[1] == self.repository.shape[1]:
            X = train_data.astype(np.float32)
            if train_label.shape[0] != train_data.shape[0]:
                return None, None
            y = train_label[:, 1].astype(np.int64)
        else:
            return None, None
        valid = y >= 0
        if np.count_nonzero(valid) < 10:
            return None, None
        return X[valid], y[valid].astype(np.int64)

    def _repository_label_vector(self, train_data, train_label, X_train, y_train):
        train_label = np.asarray(train_label)
        label_by_id = {
            int(sample_id): int(label)
            for sample_id, label in train_label[:, :2]
        }
        if self._repository_has_ids:
            labels = np.asarray(
                [label_by_id.get(int(sample_id), -1) for sample_id in self.repository_ids],
                dtype=np.int64,
            )
            if np.all(labels >= 0):
                return labels

        feature_to_label = {
            X_train[i].tobytes(): int(y_train[i]) for i in range(X_train.shape[0])
        }
        labels = np.asarray(
            [feature_to_label.get(self.repository[i].tobytes(), -1) for i in range(self.repository.shape[0])],
            dtype=np.int64,
        )
        if np.all(labels >= 0):
            return labels
        return None

    def _fit_centroids(self, X_train_std: np.ndarray, y_train: np.ndarray) -> None:
        centroids = []
        for label in self.semantic_classes:
            class_rows = X_train_std[y_train == label]
            if class_rows.size == 0:
                centroids.append(np.zeros(X_train_std.shape[1], dtype=np.float64))
            else:
                centroids.append(np.mean(class_rows, axis=0))
        self.semantic_centroids = np.vstack(centroids)
        self.semantic_centroid_norm_sq = np.einsum(
            "ij,ij->i", self.semantic_centroids, self.semantic_centroids
        )

    def _semantic_probabilities_from_raw_dist(self, raw_dist: np.ndarray):
        if self.repository_labels is None or self.semantic_scaler is None:
            return None
        if not self.semantic_models:
            return None
        # The current query chunk is attached by `_inference_chunk`.
        X = getattr(self, "_last_query_chunk", None)
        if X is None or X.shape[0] != raw_dist.shape[0]:
            return None
        X_model = X.astype(np.float32, copy=False)
        X_std = self.semantic_scaler.transform(X_model)
        prob = np.zeros((X.shape[0], self.semantic_classes.size), dtype=np.float64)
        total_weight = 0.0

        for weight, (kind, model) in zip(self.semantic_weights, self.semantic_models):
            features = X_std if kind == "scaled" else X_model
            try:
                model_prob = model.predict_proba(features)
            except Exception:
                continue
            aligned = np.zeros_like(prob)
            for source_col, label in enumerate(model.classes_):
                target_col = self._class_to_column.get(int(label))
                if target_col is not None:
                    aligned[:, target_col] = model_prob[:, source_col]
            prob += weight * aligned
            total_weight += weight

        if total_weight <= 0.0:
            return None
        return prob / total_weight

    def _semantic_rerank(
        self,
        raw_dist: np.ndarray,
        std_dist: np.ndarray,
        raw_top: np.ndarray,
        semantic_prob: np.ndarray,
    ) -> np.ndarray:
        n = raw_dist.shape[0]
        pool_size = min(self.semantic_candidate_count, raw_dist.shape[1])
        semantic_dist = self._semantic_candidate_distances(raw_dist)
        pool = np.argpartition(semantic_dist, kth=pool_size - 1, axis=1)[:, :pool_size]
        pool_d = np.take_along_axis(semantic_dist, pool, axis=1)
        pool_order = np.argsort(pool_d, axis=1)
        pool = np.take_along_axis(pool, pool_order, axis=1)
        predicted = self.semantic_classes[np.argmax(semantic_prob, axis=1)]
        std_top = self._ordered_topk(std_dist, min(32, raw_dist.shape[1]))

        out = np.empty((n, self.k), dtype=np.int64)
        for row in range(n):
            chosen = []
            target = int(predicted[row])
            for idx in pool[row]:
                idx = int(idx)
                if self.repository_labels[idx] == target:
                    chosen.append(idx)
                    if len(chosen) == self.k:
                        break
            if len(chosen) < self.k:
                for idx in pool[row]:
                    idx = int(idx)
                    if idx not in chosen:
                        chosen.append(idx)
                    if len(chosen) == self.k:
                        break

            # Avoid returning a full raw-baseline row when an alternative exists.
            if np.array_equal(np.asarray(chosen[: self.k], dtype=np.int64), raw_top[row]):
                for idx in pool[row]:
                    idx = int(idx)
                    if idx not in chosen and self.repository_labels[idx] == target:
                        chosen[self.k - 1] = idx
                        break
                else:
                    for idx in std_top[row]:
                        idx = int(idx)
                        if idx not in chosen:
                            chosen[self.k - 1] = idx
                            break

            out[row] = np.asarray(chosen[: self.k], dtype=np.int64)
        return out

    def _semantic_candidate_distances(self, fallback_dist: np.ndarray) -> np.ndarray:
        X = getattr(self, "_last_query_chunk", None)
        if X is None or X.shape[0] != fallback_dist.shape[0]:
            return fallback_dist
        X32 = X.astype(np.float32, copy=False)
        X_norm = np.sum(X32 * X32, axis=1, keepdims=True)
        dist = X_norm + self.repo_semantic_norm_sq.reshape(1, -1) - 2.0 * (
            X32.dot(self.repository_semantic.T)
        )
        return np.maximum(dist, 0.0)

    @staticmethod
    def _ordered_topk(dist: np.ndarray, k: int) -> np.ndarray:
        part = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]
        order = np.argsort(np.take_along_axis(dist, part, axis=1), axis=1)
        return np.take_along_axis(part, order, axis=1)

    def _default_indices(self, n_rows: int) -> np.ndarray:
        base = np.arange(self.k, dtype=np.int64)
        return np.tile(base, (n_rows, 1))

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _sqdist(X: np.ndarray, Y: np.ndarray, Y_norm_sq: np.ndarray) -> np.ndarray:
        X_norm_sq = np.einsum("ij,ij->i", X, X)[:, None]
        d = X_norm_sq + Y_norm_sq[None, :] - 2.0 * (X @ Y.T)
        return np.maximum(d, 0.0)

    def _best_over_augmentations(self, gen, Y, Y_norm_sq, shape):
        best = np.full(shape, np.inf, dtype=np.float64)
        for aug in gen:
            d = self._sqdist(aug, Y, Y_norm_sq)
            np.minimum(best, d, out=best)
        return best

    @staticmethod
    def _shift_features(X: np.ndarray, dy: int, dx: int) -> np.ndarray:
        if dy == 0 and dx == 0:
            return X
        images = X.reshape(-1, 16, 16)
        shifted = np.zeros_like(images)
        src_y0 = max(0, -dy); src_y1 = min(16, 16 - dy)
        dst_y0 = max(0, dy);  dst_y1 = min(16, 16 + dy)
        src_x0 = max(0, -dx); src_x1 = min(16, 16 - dx)
        dst_x0 = max(0, dx);  dst_x1 = min(16, 16 + dx)
        shifted[:, dst_y0:dst_y1, dst_x0:dst_x1] = images[
            :, src_y0:src_y1, src_x0:src_x1
        ]
        return shifted.reshape(X.shape)

    @staticmethod
    def _d4_variants(X: np.ndarray):
        # 7 non-identity D4 variants: rotations and reflections.
        images = X.reshape(-1, 16, 16)
        flat = X.shape
        yield np.rot90(images, 1, axes=(1, 2)).reshape(flat)
        yield np.rot90(images, 2, axes=(1, 2)).reshape(flat)
        yield np.rot90(images, 3, axes=(1, 2)).reshape(flat)
        yield images[:, :, ::-1].reshape(flat)            # horizontal flip
        yield images[:, ::-1, :].reshape(flat)            # vertical flip
        transposed = np.transpose(images, (0, 2, 1))
        yield transposed.reshape(flat)                    # main diag
        yield transposed[:, :, ::-1].reshape(flat)        # anti-diag

    @staticmethod
    def _blur_features(X: np.ndarray) -> np.ndarray:
        images = X.reshape(-1, 16, 16)
        padded = np.pad(images, ((0, 0), (1, 1), (1, 1)), mode="constant")
        blurred = np.zeros_like(images)
        for dy in range(3):
            for dx in range(3):
                blurred += padded[:, dy : dy + 16, dx : dx + 16]
        return (blurred / 9.0).reshape(X.shape)
