import numpy as np
from consts import WINDOW_SIZE, OVERLAP


def slice_to_windows(data, size=WINDOW_SIZE, overlap=OVERLAP):
    step = int(size * overlap)
    return np.lib.stride_tricks.sliding_window_view(data, size)[::step]
