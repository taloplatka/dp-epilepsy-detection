import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
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

# Higuchi with DT experiment

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

    dt = DecisionTreeClassifier(random_state=seed)
    dt.fit(X_train_norm, y_train)
    y_pred = dt.predict(X_test_norm)

    accs.append(acc(y_test, y_pred))
    sens.append(sen(y_test, y_pred))
    spes.append(spe(y_test, y_pred))

print("Higuchi with DT mean accuracy:", np.mean(accs))
print("Higuchi with DT mean sensitivity: ", np.mean(sens))
print("Higuchi with DT mean specificity: ", np.mean(spes))

print("Higuchi with DT std accuracy:", np.std(accs))
print("Higuchi with DT std sensitivity: ", np.std(sens))
print("Higuchi with DT std specificity: ", np.std(spes))

# Higuchi with KNN experiment


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

print("Higuchi with KNN mean accuracy:", np.mean(accs))
print("Higuchi with KNN mean sensitivity: ", np.mean(sens))
print("Higuchi with KNN mean specificity: ", np.mean(spes))

print("Higuchi with KNN std accuracy:", np.std(accs))
print("Higuchi with KNN std sensitivity: ", np.std(sens))
print("Higuchi with KNN std specificity: ", np.std(spes))

# LogisticRegression

X = np.concatenate([A_hfd, E_hfd])
X = X.reshape(-1, 1)
y = np.array([0] * 100 + [1] * 100)

accs = []
sens = []
spes = []

for seed in range(100):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=seed
    )

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_norm = scaler.transform(X_train)
    X_test_norm = scaler.transform(X_test)

    clf = LogisticRegression()
    clf.fit(X_train_norm, y_train)
    y_pred = clf.predict(X_test_norm)

    accs.append(acc(y_test, y_pred))
    sens.append(sen(y_test, y_pred))
    spes.append(spe(y_test, y_pred))

print("Higuchi with LR mean accuracy:", np.mean(accs))
print("Higuchi with LR mean sensitivity: ", np.mean(sens))
print("Higuchi with LR mean specificity: ", np.mean(spes))

print("Higuchi with LR std accuracy:", np.std(accs))
print("Higuchi with LR std sensitivity: ", np.std(sens))
print("Higuchi with LR std specificity: ", np.std(spes))
