import numpy as np

from loader import load_D_filtered, load_E_filtered
from helpers import run_knn_with_results, extract_fd_features

D_filtered = load_D_filtered()
E_filtered = load_E_filtered()

D_kfd = []
D_mkfd = []
E_kfd = []
E_mkfd = []

for eeg in D_filtered:
    results = extract_fd_features(
        eeg, want_higuchi=False, want_katz=True, want_mod_katz=True, want_means=True
    )
    D_kfd.append(results["katz_mean"])
    D_mkfd.append(results["modified_katz_mean"])

for eeg in E_filtered:
    results = extract_fd_features(
        eeg, want_higuchi=False, want_katz=True, want_mod_katz=True, want_means=True
    )
    E_kfd.append(results["katz_mean"])
    E_mkfd.append(results["modified_katz_mean"])

D_kfd = np.array(D_kfd)
E_kfd = np.array(E_kfd)
D_mkfd = np.array(D_mkfd)
E_mkfd = np.array(E_mkfd)


# Run with katz

accs = []
sens = []
spes = []

X = np.concatenate([D_kfd, E_kfd])
X = X.reshape(-1, 1)
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

# Run with modified katz

accs = []
sens = []
spes = []

X = np.concatenate([D_mkfd, E_mkfd])
X = X.reshape(-1, 1)
y = np.array([0] * 100 + [1] * 100)

for seed in range(25):
    acc, sen, spe = run_knn_with_results(X, y, seed)
    accs.append(acc)
    sens.append(sen)
    spes.append(spe)

print("\nMKatz mean accuracy: ", np.mean(accs))
print("MKatz mean sensitivity: ", np.mean(sens))
print("MKatz mean specificity: ", np.mean(spes))

print("MKatz std accuracy: ", np.std(accs))
print("MKatz std sensitivity: ", np.std(sens))
print("MKatz std specificity: ", np.std(spes))
