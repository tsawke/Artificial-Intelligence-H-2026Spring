from pathlib import Path
import pickle

import numpy as np


BASE_FEATURES = (
    0,
    2,
    3,
    4,
    5,
    6,
    7,
    9,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    19,
    20,
    24,
    26,
    28,
    29,
    33,
    46,
    47,
    54,
    59,
    62,
    67,
    71,
    205,
)


def _load_pickle(path: Path):
    with path.open("rb") as file:
        return pickle.load(file)


def _find_data_file(filename: str) -> Path:
    """
    OJ links the data files into the current task directory.  The fallback
    locations keep this file runnable in the local project layout.
    """
    here = Path(__file__).resolve().parent
    candidates = (
        here / filename,
        Path.cwd() / filename,
        here.parent / "task3" / filename,
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Cannot find {filename}")


class Selector:
    def __init__(self):
        validation_data = _load_pickle(_find_data_file("classification_validation_data.pkl"))
        validation_label = _load_pickle(_find_data_file("classification_validation_label.pkl"))
        weights = _load_pickle(_find_data_file("image_recognition_model_weights.pkl"))

        x = np.asarray(validation_data[:, 1:], dtype=np.float64)
        y = np.asarray(validation_label[:, 1:].reshape(-1), dtype=np.int64)
        weights = np.asarray(weights, dtype=np.float64)

        selected = [feature for feature in BASE_FEATURES if feature < x.shape[1]]
        if len(selected) < min(30, x.shape[1]):
            selected = self._select_features(x, y, weights, max_features=30)
        selected = self._swap_search(x, y, weights, selected, max_passes=2)

        self.mask_code = np.zeros((1, x.shape[1]), dtype=np.float64)
        self.mask_code[0, selected] = 1.0

    def get_mask_code(self) -> np.array:
        """
        Returns: The binary mask vector for the selected features.
        """
        return self.mask_code

    @staticmethod
    def _accuracy_count(y: np.ndarray, logits: np.ndarray) -> int:
        return int(np.sum(np.argmax(logits, axis=1) == y))

    def _select_features(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        max_features: int,
    ) -> list[int]:
        candidates = list(range(x.shape[1]))
        selected: list[int] = []
        current_logits = np.broadcast_to(weights[0], (x.shape[0], weights.shape[1])).copy()

        for _ in range(max_features):
            best_feature = -1
            best_score = -1
            best_logits = None
            for feature in candidates:
                if feature in selected:
                    continue
                logits = current_logits + x[:, feature, None] * weights[feature + 1][None, :]
                score = self._accuracy_count(y, logits)
                if score > best_score:
                    best_feature = feature
                    best_score = score
                    best_logits = logits

            if best_feature < 0 or best_logits is None:
                break
            selected.append(best_feature)
            current_logits = best_logits

        selected = self._swap_search(x, y, weights, selected, max_passes=3)
        return sorted(selected[:max_features])

    def _swap_search(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        selected: list[int],
        max_passes: int,
    ) -> list[int]:
        selected = sorted(selected)
        all_features = set(range(x.shape[1]))
        current_logits = self._logits_for_selected(x, weights, selected)
        current_score = self._accuracy_count(y, current_logits)

        for _ in range(max_passes):
            best_swap = None
            best_score = current_score
            best_logits = None
            selected_set = set(selected)
            replacements = sorted(all_features - selected_set)

            for out_feature in selected:
                base_logits = current_logits - x[:, out_feature, None] * weights[out_feature + 1][None, :]
                for in_feature in replacements:
                    logits = base_logits + x[:, in_feature, None] * weights[in_feature + 1][None, :]
                    score = self._accuracy_count(y, logits)
                    if score > best_score:
                        best_swap = (out_feature, in_feature)
                        best_score = score
                        best_logits = logits

            if best_swap is None or best_logits is None:
                break
            out_feature, in_feature = best_swap
            selected.remove(out_feature)
            selected.append(in_feature)
            selected = sorted(selected)
            current_logits = best_logits
            current_score = best_score

        return selected

    @staticmethod
    def _logits_for_selected(
        x: np.ndarray,
        weights: np.ndarray,
        selected: list[int],
    ) -> np.ndarray:
        logits = np.broadcast_to(weights[0], (x.shape[0], weights.shape[1])).copy()
        if selected:
            selected_array = np.asarray(selected, dtype=np.int64)
            logits += x[:, selected_array] @ weights[selected_array + 1]
        return logits
