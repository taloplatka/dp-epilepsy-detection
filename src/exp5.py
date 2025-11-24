# This experiment compares Higuchi with mean FD as the feature vector vs. Higuchi with list of window FDs as the feature vector
# All is done on a KNN model

import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from loader import load_A_filtered, load_E_filtered
from consts import TEST_SIZE, WINDOW_SIZE, OVERLAP, K
from normalize import z_score
from window import slice_to_windows
from higuchi import higuchi_fd
from katz import katz_fd
from stats import acc, sen, spe

A_filtered = load_A_filtered()
E_filtered = load_E_filtered()

A_hfd_mean = []
E_hfd_mean = []

A_hfd_vecs = []
E_hfd_vecs = []

for eeg in A_filtered:
    windows = slice_to_windows(eeg, WINDOW_SIZE, OVERLAP)
    h_fds = np.array([higuchi_fd(w) for w in windows])
    A_hfd_mean.append(np.mean(h_fds))
    A_hfd_vecs.append(h_fds)

for eeg in E_filtered:
    windows = slice_to_windows(eeg, WINDOW_SIZE, OVERLAP)
    h_fds = np.array([higuchi_fd(w) for w in windows])
    E_hfd_mean.append(np.mean(h_fds))
    E_hfd_vecs.append(h_fds)

A_hfd_mean = np.array(A_hfd_mean)
E_hfd_mean = np.array(E_hfd_mean)

A_hfd_vecs = np.array(A_hfd_vecs)
E_hfd_vecs = np.array(E_hfd_vecs)

# Main experiment

# Higuchi with 1 feature experiment

X = np.concatenate([A_hfd_mean, E_hfd_mean])
X = X.reshape(-1, 1)
y = np.array([0] * 100 + [1] * 100)

accs = []
sens = []
spes = []

for seed in range(25):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=seed
    )

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_norm = scaler.transform(X_train)
    X_test_norm = scaler.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=K)
    knn.fit(X_train_norm, y_train)

    y_pred = knn.predict(X_test_norm)

    accs.append(acc(y_test, y_pred))
    sens.append(sen(y_test, y_pred))
    spes.append(spe(y_test, y_pred))

print("Higuchi 1 mean accuracy:", np.mean(accs))
print("Higuchi 1 mean sensitivity: ", np.mean(sens))
print("Higuchi 1 mean specificity: ", np.mean(spes))

print("Higuchi 1 std accuracy:", np.std(accs))
print("Higuchi 1 std sensitivity: ", np.std(sens))
print("Higuchi 1 std specificity: ", np.std(spes))

# Higuchi 2 feature experiment

X = np.vstack([A_hfd_vecs, E_hfd_vecs])
y = np.array([0] * 100 + [1] * 100)

accs = []
sens = []
spes = []

for seed in range(25):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=seed
    )

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_norm = scaler.transform(X_train)
    X_test_norm = scaler.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=K)
    knn.fit(X_train_norm, y_train)

    y_pred = knn.predict(X_test_norm)

    accs.append(acc(y_test, y_pred))
    sens.append(sen(y_test, y_pred))
    spes.append(spe(y_test, y_pred))

print("\nHiguchi 2 mean accuracy:", np.mean(accs))
print("Higuchi 2 mean sensitivity: ", np.mean(sens))
print("Higuchi 2 mean specificity: ", np.mean(spes))

print("Higuchi 2 std accuracy:", np.std(accs))
print("Higuchi 2 std sensitivity: ", np.std(sens))
print("Higuchi 2 std specificity: ", np.std(spes))
