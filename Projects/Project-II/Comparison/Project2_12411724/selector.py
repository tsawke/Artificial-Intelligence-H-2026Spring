import os
import pickle

import numpy as np


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


class Selector:
    def __init__(self):
        root_path = os.path.dirname(os.path.abspath(__file__))
        validation_data = _load_pickle(root_path, "classification_validation_data.pkl")
        validation_label = _load_pickle(root_path, "classification_validation_label.pkl")
        weights = _load_pickle(root_path, "image_recognition_model_weights.pkl")

        X = np.asarray(validation_data[:, 1:], dtype=np.float64)
        y = np.asarray(validation_label[:, 1], dtype=np.int64)
        weights = np.asarray(weights, dtype=np.float64)

        selected = self._select_features(X, y, weights, num_features=30)
        mask_code = np.zeros((1, X.shape[1]), dtype=np.float64)
        mask_code[0, selected] = 1.0
        self.mask_code = mask_code

    def _accuracy(self, logits, y):
        return float(np.mean(np.argmax(logits, axis=1) == y))

    def _select_features(self, X, y, weights, num_features):
        bias = weights[:1]
        feature_weights = weights[1:]
        logits = np.ones((X.shape[0], 1), dtype=np.float64).dot(bias)

        selected = []
        remaining = set(range(X.shape[1]))
        for _ in range(num_features):
            best_feature = None
            best_accuracy = -1.0
            for feature in remaining:
                candidate_logits = logits + X[:, feature : feature + 1] * feature_weights[feature : feature + 1]
                candidate_accuracy = self._accuracy(candidate_logits, y)
                if candidate_accuracy > best_accuracy:
                    best_accuracy = candidate_accuracy
                    best_feature = feature

            selected.append(best_feature)
            remaining.remove(best_feature)
            logits += X[:, best_feature : best_feature + 1] * feature_weights[best_feature : best_feature + 1]

        selected_set = set(selected)
        current_accuracy = self._accuracy(logits, y)
        for _ in range(3):
            best_swap = None
            for removed in list(selected_set):
                logits_without_removed = logits - X[:, removed : removed + 1] * feature_weights[removed : removed + 1]
                for added in range(X.shape[1]):
                    if added in selected_set:
                        continue
                    candidate_logits = logits_without_removed + X[:, added : added + 1] * feature_weights[added : added + 1]
                    candidate_accuracy = self._accuracy(candidate_logits, y)
                    if candidate_accuracy > current_accuracy:
                        current_accuracy = candidate_accuracy
                        best_swap = (removed, added, candidate_logits)

            if best_swap is None:
                break

            removed, added, logits = best_swap
            selected_set.remove(removed)
            selected_set.add(added)

        return sorted(selected_set)

    def get_mask_code(self) -> np.array:
        """
        Returns: The binary mask matrix for the selected features.
        """
        return self.mask_code
