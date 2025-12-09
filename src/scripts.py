# # import numpy as np
# # import glob

# # from filter import bandpass_filter_bonn_eeg

# # # A_paths = sorted(glob.glob("Z/Z*.txt"))
# # # E_paths = sorted(glob.glob("S/S*.txt"))

# # # A_raw = np.vstack([np.loadtxt(p) for p in A_paths])
# # # E_raw = np.vstack([np.loadtxt(p) for p in E_paths])

# # # A_filtered = np.vstack([bandpass_filter_bonn_eeg(eeg, 0.5, 40) for eeg in A_raw])

# # # E_filtered = np.vstack([bandpass_filter_bonn_eeg(eeg, 0.5, 40) for eeg in E_raw])

# # # np.savez("A_filtered.npz", A=A_filtered)
# # # np.savez("E_filtered.npz", E=E_filtered)
# import numpy as np
# import glob
# from filter import bandpass_filter_bonn_eeg

# D_paths = sorted(glob.glob("F/F*.txt"))
# D_raw = np.vstack([np.loadtxt(p) for p in D_paths])
# D_filtered = np.vstack([bandpass_filter_bonn_eeg(eeg, 0.5, 40) for eeg in D_raw])
# np.savez("D_filtered.npz", D=D_filtered)


import numpy as np
from loader import load_E_filtered
from helpers import extract_fd_features

E_filtered = load_E_filtered()

import numpy as np
import matplotlib.pyplot as plt

# נניח שזה האות שלך
eeg_signal = E_filtered[0]  # או סתם: eeg_signal = your_array

info = extract_fd_features(eeg_signal, want_vectors=True)
print(info["higuchi_vector"])


# # הגדר קצב דגימה (בHz) – תשנה לפי מה שיש לך
# sampling_rate = 173.61  # דוגמה – תעדכן אם צריך
# time = np.arange(len(eeg_signal)) / sampling_rate  # יוצר ציר זמן בשניות

# # ציור
# plt.figure(figsize=(12, 4))
# plt.plot(time, eeg_signal)
# plt.title("EEG Signal")
# plt.xlabel("Time (seconds)")
# plt.ylabel("Amplitude (μV)")
# plt.grid(True)
# plt.tight_layout()
# plt.show()
