import numpy as np


SELECTED_FEATURES = (
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


class Selector:
    def __init__(self):
        self.mask_code = np.zeros((1, 256), dtype=np.float64)
        self.mask_code[0, SELECTED_FEATURES] = 1.0

    def get_mask_code(self) -> np.array:
        """
        Returns: The binary mask vector for the selected features.
        """
        return self.mask_code
