# This experiment examines Higuchi vs Katz stats where the feature vector is the mean FD

import numpy as np
from loader import load_D_filtered, load_E_filtered
from helpers import run_knn_with_results, extract_fd_features

D_filtered = load_D_filtered()
E_filtered = load_E_filtered()

D_hfd = []
D_kfd = []
E_hfd = []
E_kfd = []

for eeg in D_filtered:
    results = extract_fd_features(
        eeg, want_higuchi=True, want_katz=True, want_means=True
    )
    D_hfd.append(results["higuchi_mean"])
    D_kfd.append(results["katz_mean"])

for eeg in E_filtered:
    results = extract_fd_features(
        eeg, want_higuchi=True, want_katz=True, want_means=True
    )
    E_hfd.append(results["higuchi_mean"])
    E_kfd.append(results["katz_mean"])

D_hfd = np.array(D_hfd)
D_kfd = np.array(D_kfd)
E_hfd = np.array(E_hfd)
E_kfd = np.array(E_kfd)


# Higuchi with mean FD as feature vector experiment

X = np.concatenate([D_hfd, E_hfd])
X = X.reshape(-1, 1)
y = np.array([0] * 100 + [1] * 100)

higuchi_accs = []
higuchi_sens = []
higuchi_spes = []

for seed in range(25):
    acc, sen, spe = run_knn_with_results(X, y, seed)
    higuchi_accs.append(acc)
    higuchi_sens.append(sen)
    higuchi_spes.append(spe)


print("Higuchi mean accuracy:", np.mean(higuchi_accs))
print("Higuchi mean sensitivity: ", np.mean(higuchi_sens))
print("Higuchi mean specificity: ", np.mean(higuchi_spes))

print("Higuchi std accuracy:", np.std(higuchi_accs))
print("Higuchi std sensitivity: ", np.std(higuchi_sens))
print("Higuchi std specificity: ", np.std(higuchi_spes))


# # Katz Experiment

katz_accs = []
katz_sens = []
katz_spes = []

X = np.concatenate([D_kfd, E_kfd])
X = X.reshape(-1, 1)
y = np.array([0] * 100 + [1] * 100)

for seed in range(25):
    acc, sen, spe = run_knn_with_results(X, y, seed)
    katz_accs.append(acc)
    katz_sens.append(sen)
    katz_spes.append(spe)

print("\nKatz mean accuracy: ", np.mean(katz_accs))
print("Katz mean sensitivity: ", np.mean(katz_sens))
print("Katz mean specificity: ", np.mean(katz_spes))

print("Katz std accuracy: ", np.std(katz_accs))
print("Katz std sensitivity: ", np.std(katz_sens))
print("Katz std specificity: ", np.std(katz_spes))
