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

A_hfd = []
E_hfd = []
A_kfd = []
E_kfd = []

for eeg in A_filtered:
    windows = slice_to_windows(eeg, WINDOW_SIZE, OVERLAP)
    h_fds = np.array([higuchi_fd(w) for w in windows])
    k_fds = np.array([katz_fd(w) for w in windows])
    A_hfd.append(np.mean(h_fds))
    A_kfd.append(np.mean(k_fds))

for eeg in E_filtered:
    windows = slice_to_windows(eeg, WINDOW_SIZE, OVERLAP)
    h_fds = np.array([higuchi_fd(w) for w in windows])
    k_fds = np.array([katz_fd(w) for w in windows])
    E_hfd.append(np.mean(h_fds))
    E_kfd.append(np.mean(k_fds))

A_hfd = np.array(A_hfd)
A_kfd = np.array(A_kfd)
E_hfd = np.array(E_hfd)
E_kfd = np.array(E_kfd)

# Main experiment

# Higuchi experiment

X = np.concatenate([A_hfd, E_hfd])
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

print("Higuchi mean accuracy:", np.mean(accs))
print("Higuchi mean sensitivity: ", np.mean(sens))
print("Higuchi mean specificity: ", np.mean(spes))

print("Higuchi std accuracy:", np.std(accs))
print("Higuchi std sensitivity: ", np.std(sens))
print("Higuchi std specificity: ", np.std(spes))


# Katz Experiment

X = np.concatenate([A_kfd, E_kfd])
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

print("\nKatz mean accuracy: ", np.mean(accs))
print("Katz mean sensitivity: ", np.mean(sens))
print("Katz mean specificity: ", np.mean(spes))

print("Katz std accuracy: ", np.std(accs))
print("Katz std sensitivity: ", np.std(sens))
print("Katz std specificity: ", np.std(spes))
