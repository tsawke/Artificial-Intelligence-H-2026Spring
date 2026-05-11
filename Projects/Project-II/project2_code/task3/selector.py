import numpy as np


class Selector:
    def __init__(self):
        """
        Deterministic 30-feature mask selected by validation-set search.

        The indices were obtained with beam forward selection on the fixed
        recognition model supplied with the project, followed by a swap check.
        """
        selected_features = np.array(
            [
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
                42,
                46,
                47,
                59,
                62,
                71,
                120,
                130,
                221,
            ],
            dtype=int,
        )
        self.mask_code = np.zeros((1, 256), dtype=np.float64)
        self.mask_code[0, selected_features] = 1.0

    def get_mask_code(self) -> np.array:
        """
        Returns:
            A binary feature mask with shape [1, 256].
        """
        return self.mask_code
