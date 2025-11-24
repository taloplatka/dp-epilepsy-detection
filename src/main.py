import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from loader import load_A_filtered, load_E_filtered
from window import slice_to_windows
from katz import katz_fd
from higuchi import higuchi_fd
from consts import KMAX, WINDOW_SIZE, OVERLAP

A_filtered = load_A_filtered()
E_filtered = load_E_filtered()

A_mean_fd = []
for eeg in A_filtered:
    windows = slice_to_windows(eeg, WINDOW_SIZE, OVERLAP)
    fds = np.array([higuchi_fd(w) for w in windows])
    A_mean_fd.append(np.std(fds))

A_mean_fd = np.array(A_mean_fd)

E_mean_fd = []
for eeg in E_filtered:
    windows = slice_to_windows(eeg, WINDOW_SIZE, OVERLAP)
    fds = np.array([higuchi_fd(w) for w in windows])
    E_mean_fd.append(np.std(fds))

E_mean_fd = np.array(E_mean_fd)

print(A_mean_fd)
