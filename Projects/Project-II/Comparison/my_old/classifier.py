import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


# 16x16 grayscale image features -- the 256-d vector is a flattened image.
IMG_H = 16
IMG_W = 16
FEATURE_DIM = IMG_H * IMG_W

# Shifts used for both training-time augmentation and inference-time TTA.
# Original + 4 cardinal one-pixel shifts gives 5x data without exploding the
# training time. Cardinal shifts dominate the gain on small digit-like images;
# diagonal shifts add little and would double training time.
_TTA_SHIFTS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))

# The official Q&A gives each task a 600 s limit.  This soft deadline leaves
# room for inference and judge overhead.  If the machine is slower than local
# validation, the ensemble simply stops growing and uses the members already
# trained.
TASK_TIME_LIMIT_SEC = 600.0
SAFETY_MARGIN_SEC = 60.0
NEXT_MEMBER_GUARD_FACTOR = 1.15


def _shift_batch(X: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Shift a stack of flattened 16x16 images by (dy, dx) pixels.

    Pixels shifted in from outside are zeros (matches the natural padding of
    centered digit-like images).
    """
    if dy == 0 and dx == 0:
        return X
    images = X.reshape(-1, IMG_H, IMG_W)
    shifted = np.zeros_like(images)
    src_y0 = max(0, -dy)
    src_y1 = min(IMG_H, IMG_H - dy)
    dst_y0 = max(0, dy)
    dst_y1 = min(IMG_H, IMG_H + dy)
    src_x0 = max(0, -dx)
    src_x1 = min(IMG_W, IMG_W - dx)
    dst_x0 = max(0, dx)
    dst_x1 = min(IMG_W, IMG_W + dx)
    shifted[:, dst_y0:dst_y1, dst_x0:dst_x1] = images[:, src_y0:src_y1, src_x0:src_x1]
    return shifted.reshape(X.shape)


def _augment(X: np.ndarray, y: np.ndarray):
    """Return an augmented copy of (X, y) using the TTA shift set."""
    parts_X = []
    parts_y = []
    for dy, dx in _TTA_SHIFTS:
        parts_X.append(_shift_batch(X, dy, dx))
        parts_y.append(y)
    return np.concatenate(parts_X, axis=0), np.concatenate(parts_y, axis=0)


class Classifier:
    def __init__(self):
        """Train the classifier when the judge instantiates the class.

        The official Q&A says the judge imports this file, soft-links the data
        files into the working directory, constructs this class once, and calls
        inference once. Therefore the submitted file trains from the provided
        pkl data instead of loading pre-trained sidecar weights.

        Design summary
        --------------
        - 16x16 grayscale images: apply pixel-shift data augmentation
          (original + 4 cardinal shifts -> 5x training data).
        - Heterogeneous soft-vote ensemble over the augmented data:
            * 3 MLPClassifiers with diverse seeds and topologies,
            * 1 low-weight HistGradientBoostingClassifier.
          The members make different mistakes, so the averaged probabilities
          are noticeably more accurate and more stable across seeds than any
          single member.
        - Test-time augmentation (TTA): at inference, predict on the original
          plus the same 4 cardinal-shift copies of X and average the soft
          probabilities. This costs only inference-time and gives free gain.
        """
        start_time = time.time()
        self._soft_deadline = start_time + TASK_TIME_LIMIT_SEC - SAFETY_MARGIN_SEC

        train_data = self._load_data("classification_train_data.pkl")[:, 1:].astype(np.float64)
        train_label = self._load_data("classification_train_label.pkl")[:, 1:].reshape(-1).astype(int)

        # Standardize on the original (non-augmented) data, then transform
        # everything through the same scaler. Shifts are zero-padding in raw
        # space; the scaler still applies the same affine, which is fine.
        self.scaler = StandardScaler()
        self.scaler.fit(train_data)

        # Augment in scaled space: shifting is linear, but to keep the zeros
        # consistent we shift in raw space and then standardize. That keeps
        # the introduced border pixels at the same scaled value as a true
        # background pixel.
        X_aug_raw, y_aug = _augment(train_data, train_label)
        X_aug_std = self.scaler.transform(X_aug_raw)

        self.models = []
        self.weights = []  # ensemble weights per model
        train_seconds = []

        # --- MLP members (trained on augmented data) ---
        mlp_configs = [
            # (hidden_layer_sizes, seed, alpha, learning_rate_init, weight)
            ((384, 192), 123, 1e-4, 1e-3, 1.0),
            ((256, 128), 456, 1e-4, 1e-3, 1.0),
            ((512,),     789, 1e-4, 1e-3, 1.0),
        ]
        for hidden, seed, alpha, lr, w in mlp_configs:
            if self.models and not self._has_time_for_next_member(train_seconds):
                break
            model = MLPClassifier(
                hidden_layer_sizes=hidden,
                activation="relu",
                solver="adam",
                alpha=alpha,
                learning_rate_init=lr,
                learning_rate="adaptive",
                batch_size=256,
                max_iter=80,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=8,
                random_state=seed,
                verbose=False,
            )
            member_start = time.time()
            model.fit(X_aug_std, y_aug)
            train_seconds.append(time.time() - member_start)
            self.models.append(("mlp", model))
            self.weights.append(w)

        # --- Gradient boosting member (trained on augmented data) ---
        if self._has_time_for_next_member(train_seconds):
            hgb = HistGradientBoostingClassifier(
                max_iter=200,
                learning_rate=0.1,
                max_depth=None,
                max_leaf_nodes=31,
                l2_regularization=0.0,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=15,
                random_state=2026,
            )
            member_start = time.time()
            hgb.fit(X_aug_std, y_aug)
            train_seconds.append(time.time() - member_start)
            self.models.append(("hgb", hgb))
            # Local stratified validation showed HGB helps only as a weak
            # diversity member. A higher weight, and especially KNN, reduced
            # the soft-vote accuracy under TTA.
            self.weights.append(0.25)

        self.classes = self.models[0][1].classes_.astype(int)
        self._total_weight = float(sum(self.weights))

    def _has_time_for_next_member(self, train_seconds):
        """Return False when starting another ensemble member is too risky."""
        remaining = self._soft_deadline - time.time()
        if remaining <= 0:
            return False
        if not train_seconds:
            return True
        predicted_next = max(train_seconds) * NEXT_MEMBER_GUARD_FACTOR
        return remaining > predicted_next

    # ------------------------------------------------------------------ #
    # Data loading helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _candidate_paths(file_name):
        file_dir = Path(__file__).resolve().parent
        cwd = Path.cwd()
        return [cwd / file_name, file_dir / file_name]

    @classmethod
    def _load_data(cls, file_name):
        for path in cls._candidate_paths(file_name):
            if path.exists():
                with path.open("rb") as file:
                    return pickle.load(file)
        raise FileNotFoundError(f"Cannot find {file_name}")

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def _probabilities_for_view(self, X_std: np.ndarray) -> np.ndarray:
        """Sum weighted predict_proba across all ensemble members."""
        probs = np.zeros((X_std.shape[0], self.classes.shape[0]), dtype=np.float64)
        for model_idx, ((_, model), w) in enumerate(zip(self.models, self.weights)):
            probs += w * model.predict_proba(X_std)
            if model_idx > 0 and time.time() >= self._soft_deadline:
                break
        return probs

    def inference(self, X: np.array) -> np.array:
        """Predict integer class labels for X.

        Accepts both [N, 256] and [N, 257] (ID-prefixed) feature matrices.
        Uses TTA: averages soft probabilities over the original input and 4
        cardinal one-pixel shifts of it (the same shift set used for training
        augmentation).
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be a 2-D feature matrix")
        if X.shape[1] == FEATURE_DIM + 1:
            X = X[:, 1:]
        if X.shape[1] != FEATURE_DIM:
            raise ValueError(f"Expected {FEATURE_DIM} features, got {X.shape[1]}")

        total = np.zeros((X.shape[0], self.classes.shape[0]), dtype=np.float64)
        for shift_idx, (dy, dx) in enumerate(_TTA_SHIFTS):
            X_shift = _shift_batch(X, dy, dx)
            X_std = self.scaler.transform(X_shift)
            total += self._probabilities_for_view(X_std)
            if shift_idx > 0 and time.time() >= self._soft_deadline:
                break
        # No need to normalize -- argmax is invariant to positive scaling.
        return self.classes[np.argmax(total, axis=1)]
