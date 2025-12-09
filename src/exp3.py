# This experiment examines Higuchi's stats with KNN vs Logistic Regression classifiers

import numpy as np
from loader import load_D_filtered, load_E_filtered
from helpers import run_knn_with_results, run_LR_with_results, extract_fd_features

D_filtered = load_D_filtered()
E_filtered = load_E_filtered()

D_hfd = []
D_kfd = []
E_hfd = []
E_kfd = []

for eeg in D_filtered:
    results = extract_fd_features(
        eeg, want_higuchi=True, want_katz=False, want_means=False, want_vectors=True
    )
    D_hfd.append(results["higuchi_vector"])

for eeg in E_filtered:
    results = extract_fd_features(
        eeg, want_higuchi=True, want_katz=False, want_means=False, want_vectors=True
    )
    E_hfd.append(results["higuchi_vector"])

D_hfd = np.array(D_hfd)
D_kfd = np.array(D_kfd)
E_hfd = np.array(E_hfd)
E_kfd = np.array(E_kfd)


# Higuchi with mean FD as feature vector using a KNN classifier

X = np.vstack([D_hfd, E_hfd])
y = np.array([0] * 100 + [1] * 100)

accs = []
sens = []
spes = []

for seed in range(25):
    acc, sen, spe = run_knn_with_results(X, y, seed)
    accs.append(acc)
    sens.append(sen)
    spes.append(spe)


print("Higuchi with KNN mean accuracy:", np.mean(accs))
print("Higuchi with KNN mean sensitivity: ", np.mean(sens))
print("Higuchi with KNN mean specificity: ", np.mean(spes))

print("Higuchi with KNN std accuracy:", np.std(accs))
print("Higuchi with KNN std sensitivity: ", np.std(sens))
print("Higuchi with KNN std specificity: ", np.std(spes))


# Higuchi with mean FD using LR classifier

accs = []
sens = []
spes = []

for seed in range(25):
    acc, sen, spe = run_LR_with_results(X, y, seed)
    accs.append(acc)
    sens.append(sen)
    spes.append(spe)

print("\nHiguchi with LR mean accuracy: ", np.mean(accs))
print("Higuchi with LR mean sensitivity: ", np.mean(sens))
print("Katz mean specificity: ", np.mean(spes))

print("Higuchi with LR std accuracy: ", np.std(accs))
print("Higuchi with LR  std sensitivity: ", np.std(sens))
print("Higuchi with LR  std specificity: ", np.std(spes))
