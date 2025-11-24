import numpy as np
import glob

from filter import bandpass_filter_bonn_eeg

A_paths = sorted(glob.glob("Z/Z*.txt"))
E_paths = sorted(glob.glob("S/S*.txt"))

A_raw = np.vstack([np.loadtxt(p) for p in A_paths])
E_raw = np.vstack([np.loadtxt(p) for p in E_paths])

A_filtered = np.vstack([bandpass_filter_bonn_eeg(eeg, 0.5, 40) for eeg in A_raw])

E_filtered = np.vstack([bandpass_filter_bonn_eeg(eeg, 0.5, 40) for eeg in E_raw])

np.savez("A_filtered.npz", A=A_filtered)
np.savez("E_filtered.npz", E=E_filtered)
