import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from consts import K, TEST_SIZE, WINDOW_SIZE, OVERLAP, KMAX
from stats import acc, sen, spe
from window import slice_to_windows
from katz import katz_fd, modified_katz_fd
from higuchi import higuchi_fd


def extract_fd_features(
    eeg,
    window_size=WINDOW_SIZE,
    overlap=OVERLAP,
    want_higuchi=True,
    want_katz=True,
    want_vectors=False,
    want_means=True,
    want_std=False,
    kmax=KMAX,
    normalize=False,
    want_mod_katz=False,
):
    if normalize:
        eeg = (eeg - np.mean(eeg)) / np.std(eeg)

    windows = slice_to_windows(eeg, window_size, overlap)
    results = {}

    h_fds = (
        np.array([higuchi_fd(w, kmax=kmax) for w in windows]) if want_higuchi else None
    )
    k_fds = np.array([katz_fd(w) for w in windows]) if want_katz else None

    mk_fds = np.array([modified_katz_fd(w) for w in windows]) if want_mod_katz else None
    if want_vectors:
        results["higuchi_vector"] = h_fds
        results["katz_vector"] = k_fds
        results["modified_katz_vector"] = mk_fds

    if want_means:
        results["higuchi_mean"] = np.mean(h_fds) if want_higuchi else None
        results["katz_mean"] = np.mean(k_fds) if want_katz else None
        results["modified_katz_mean"] = np.mean(mk_fds) if want_mod_katz else None

    if want_std:
        results["higuchi_std"] = np.std(h_fds) if want_higuchi else None
        results["katz_std"] = np.std(k_fds) if want_katz else None
        results["modified_katz_std"] = np.std(mk_fds) if want_mod_katz else None

    return results


def run_knn_with_results(X, y, seed, k=K, test_size=TEST_SIZE, return_train=False):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_norm = scaler.transform(X_train)
    X_test_norm = scaler.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_norm, y_train)

    y_pred_train = knn.predict(X_train_norm)
    train_acc = acc(y_train, y_pred_train)

    y_pred = knn.predict(X_test_norm)

    accuracy = acc(y_test, y_pred)
    sensitivity = sen(y_test, y_pred)
    specificity = spe(y_test, y_pred)

    if return_train:
        return train_acc, accuracy, sensitivity, specificity
    return accuracy, sensitivity, specificity


def run_LR_with_results(X, y, seed, test_size=TEST_SIZE):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_norm = scaler.transform(X_train)
    X_test_norm = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=500)
    lr.fit(X_train_norm, y_train)

    y_pred = lr.predict(X_test_norm)

    accuracy = acc(y_test, y_pred)
    sensitivity = sen(y_test, y_pred)
    specificity = spe(y_test, y_pred)
    return accuracy, sensitivity, specificity


def run_DT_with_results(X, y, seed, test_size=TEST_SIZE):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_norm = scaler.transform(X_train)
    X_test_norm = scaler.transform(X_test)

    lr = DecisionTreeClassifier()
    lr.fit(X_train_norm, y_train)

    y_pred = lr.predict(X_test_norm)

    accuracy = acc(y_test, y_pred)
    sensitivity = sen(y_test, y_pred)
    specificity = spe(y_test, y_pred)
    return accuracy, sensitivity, specificity
