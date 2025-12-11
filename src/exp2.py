# This experiment examiens examines Higuchi vs Katz stats where the feature vector is the 10D vector
# of all FDs of windows

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
        eeg, want_higuchi=True, want_katz=True, want_means=True, want_vectors=True
    )
    D_hfd.append(results["higuchi_vector"])
    D_kfd.append(results["katz_vector"])

for eeg in E_filtered:
    results = extract_fd_features(
        eeg, want_higuchi=True, want_katz=True, want_means=True, want_vectors=True
    )
    E_hfd.append(results["higuchi_vector"])
    E_kfd.append(results["katz_vector"])

D_hfd = np.array(D_hfd)
D_kfd = np.array(D_kfd)
E_hfd = np.array(E_hfd)
E_kfd = np.array(E_kfd)


# Higuchi

X = np.vstack([D_hfd, E_hfd])
y = np.array([0] * 100 + [1] * 100)

accs = []
sens = []
spes = []
train_accs = []

for seed in range(25):
    train_acc, acc, sen, spe = run_knn_with_results(X, y, seed, return_train=True)
    train_accs.append(train_acc)
    accs.append(acc)
    sens.append(sen)
    spes.append(spe)

print("Higuchi train mean accuracy: ", np.mean(train_accs))
print("Higuchi mean accuracy:", np.mean(accs))
print("Higuchi mean sensitivity: ", np.mean(sens))
print("Higuchi mean specificity: ", np.mean(spes))

print("Higuchi std accuracy:", np.std(accs))
print("Higuchi std sensitivity: ", np.std(sens))
print("Higuchi std specificity: ", np.std(spes))


# Katz Experiment

accs = []
sens = []
spes = []

X = np.vstack([D_kfd, E_kfd])
y = np.array([0] * 100 + [1] * 100)

for seed in range(25):
    acc, sen, spe = run_knn_with_results(X, y, seed)
    accs.append(acc)
    sens.append(sen)
    spes.append(spe)

print("\nKatz mean accuracy: ", np.mean(accs))
print("Katz mean sensitivity: ", np.mean(sens))
print("Katz mean specificity: ", np.mean(spes))

print("Katz std accuracy: ", np.std(accs))
print("Katz std sensitivity: ", np.std(sens))
print("Katz std specificity: ", np.std(spes))
